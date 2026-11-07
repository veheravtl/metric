"""Сквозная Qt-проверка кнопки автоматического совмещения."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from aerial_mapper.gui.main_window import MainWindow
from aerial_mapper.synthetic import SyntheticFrameSpec, generate_synthetic_frame


@pytest.fixture(scope="module")
def qt_application():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def automatic_alignment_sources(tmp_path: Path) -> tuple[Path, Path]:
    """Создать карту и видео, связанные скрытой известной гомографией."""

    random_generator = np.random.default_rng(20261105)
    reference = random_generator.integers(
        0,
        256,
        size=(800, 800, 3),
        dtype=np.uint8,
    )
    reference = cv2.GaussianBlur(reference, (0, 0), sigmaX=1.0)
    cv2.rectangle(reference, (80, 90), (360, 280), (240, 210, 20), 10)
    cv2.circle(reference, (620, 190), 85, (20, 220, 240), 12)
    cv2.line(reference, (70, 720), (730, 470), (245, 245, 245), 16)
    cv2.putText(
        reference,
        "GUI",
        (250, 620),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.5,
        (20, 20, 20),
        10,
        cv2.LINE_AA,
    )
    synthetic = generate_synthetic_frame(
        reference,
        reference_resolution_m_per_pixel=0.1,
        spec=SyntheticFrameSpec(
            footprint_width_m=55.0,
            footprint_height_m=42.0,
            output_width_pixels=640,
            output_height_pixels=480,
            rotation_degrees=16.0,
            perspective_strength=0.25,
        ),
    )

    map_path = tmp_path / "reference.png"
    assert cv2.imwrite(
        str(map_path),
        cv2.cvtColor(reference, cv2.COLOR_RGB2BGR),
    )
    video_path = tmp_path / "frame.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        1.0,
        (640, 480),
    )
    if not writer.isOpened():
        pytest.skip("Локальная сборка OpenCV не умеет записывать тестовый MJPG")
    writer.write(cv2.cvtColor(synthetic.image_rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    return video_path, map_path


def test_auto_align_action_builds_visible_overlay(
    qt_application: QApplication,
    automatic_alignment_sources: tuple[Path, Path],
) -> None:
    """Нажатие меню должно пройти от декодера до слоя карты и метрик."""

    video_path, map_path = automatic_alignment_sources
    window = MainWindow()
    assert not window.ui.actionAuto_align.isEnabled()
    assert not window.ui.actionExport_alignment_report.isEnabled()
    assert window.video.load_video(video_path)
    assert window.map_image.load_image(map_path)
    assert window.ui.actionAuto_align.isEnabled()
    assert not window.ui.actionExport_alignment_report.isEnabled()

    window.ui.actionAuto_align.trigger()

    result = window._last_automatic_alignment
    assert result is not None
    assert result.inlier_count >= 8
    assert window.alignment.overlay_visible
    assert "АВТО-ЭКСПЕРИМЕНТ" in window.ui.statusbar.currentMessage()
    assert "без известной коррекции объектива" in window.ui.statusbar.currentMessage()
    assert not window.ui.actionExport_alignment_report.isEnabled()
    window.close()


def test_report_action_exports_current_manual_and_automatic_results(
    qt_application: QApplication,
    automatic_alignment_sources: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Команда меню должна собрать ZIP без ручного копирования артефактов."""

    video_path, map_path = automatic_alignment_sources
    window = MainWindow()
    assert window.video.load_video(video_path)
    assert window.map_image.load_image(map_path)

    frame_points = [
        (20.0, 20.0),
        (610.0, 25.0),
        (615.0, 450.0),
        (25.0, 455.0),
        (320.0, 240.0),
    ]
    map_points = [
        (110.0, 120.0),
        (700.0, 130.0),
        (690.0, 660.0),
        (120.0, 670.0),
        # Первые четыре пары задают проективное преобразование. Пятая точка
        # обязана ему соответствовать: тест проверяет экспорт отчёта, а не
        # защиту от ошибочного ручного клика.
        (409.4258, 402.8197),
    ]
    for frame_point, map_point in zip(frame_points, map_points, strict=True):
        assert window.alignment.add_correspondence(frame_point, map_point)
    assert window.alignment.compute_alignment() is not None
    assert window.ui.actionExport_alignment_report.isEnabled()

    output_path = tmp_path / "gui-report.zip"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(output_path), "ZIP archive (*.zip)"),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Ok,
    )

    window.ui.actionExport_alignment_report.trigger()
    assert output_path.is_file()
    assert window._last_automatic_alignment is not None
    assert "Диагностический пакет" in window.ui.statusbar.currentMessage()
    window.close()
