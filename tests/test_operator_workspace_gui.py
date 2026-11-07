"""Сквозные проверки плавающей панели и защиты рабочего прогресса."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox

from aerial_mapper.gui.main_window import (
    DOCK_GEOMETRY_KEY,
    WINDOW_STATE_KEY,
    MainWindow,
)


@pytest.fixture(scope="module")
def qt_application():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def workspace_sources(tmp_path: Path) -> tuple[Path, Path]:
    video_path = tmp_path / "workspace.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (100, 80),
    )
    if not writer.isOpened():
        pytest.skip("Локальная сборка OpenCV не записывает тестовый MJPG")
    writer.write(np.full((80, 100, 3), 80, dtype=np.uint8))
    writer.write(np.full((80, 100, 3), 120, dtype=np.uint8))
    writer.release()
    map_path = tmp_path / "map.png"
    assert cv2.imwrite(str(map_path), np.full((140, 160, 3), 100, dtype=np.uint8))
    return video_path, map_path


def _settings(tmp_path: Path, name: str) -> QSettings:
    return QSettings(str(tmp_path / name), QSettings.Format.IniFormat)


def test_panel_navigates_with_one_based_frame_number_and_restores_project(
    qt_application: QApplication,
    workspace_sources: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Оператор вводит кадр 2, а проект восстанавливает индекс 1 и маяки."""

    video_path, map_path = workspace_sources
    window = MainWindow(settings=_settings(tmp_path, "first.ini"))
    assert window.video.load_video(video_path)
    assert window.map_image.load_image(map_path)

    frame_points = [(5.0, 5.0), (90.0, 5.0), (90.0, 70.0), (5.0, 70.0)]
    map_points = [(15.0, 25.0), (100.0, 25.0), (100.0, 90.0), (15.0, 90.0)]
    for frame_point, map_point in zip(frame_points, map_points, strict=True):
        assert window.alignment.add_correspondence(frame_point, map_point)
    assert window.alignment.compute_alignment() is not None

    assert window.operator_dock.frame_number_spin.value() == 1
    window.operator_dock.frame_number_spin.setValue(2)
    window.operator_dock.go_to_frame_button.click()
    assert window.video.current_frame_index == 1
    assert window.operator_dock.frame_status_label.text().startswith("Кадр 2")

    project_path = tmp_path / "flight.aerial-project.json"
    assert window._write_operator_project(project_path, automatic=False)
    assert not window._project_dirty

    restored = MainWindow(settings=_settings(tmp_path, "second.ini"))
    assert restored._load_operator_project(project_path)
    assert restored.video.current_frame_index == 1
    assert restored.operator_dock.frame_number_spin.value() == 2
    assert restored.alignment.remembered_frame_indices == (0,)
    assert restored._resume_frame_index == 1

    restored.video.seek(0)
    assert restored.alignment.current_landmark_ids == (0, 1, 2, 3)
    restored.operator_dock.landmark_number_spin.setValue(4)
    restored.operator_dock.focus_landmark_button.click()
    snapshot = restored.alignment.debug_snapshot()
    selected = [
        item["landmark_id"]
        for item in snapshot["observations"]
        if item["frame_marker_selected"]
    ]
    assert selected == [3]

    window.close()
    restored.close()


def test_protected_close_can_be_cancelled_then_discarded(
    qt_application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cancel оставляет окно открытым, Discard разрешает закрытие."""

    window = MainWindow(
        settings=_settings(tmp_path, "protected.ini"),
        protect_unsaved_progress=True,
    )
    window._project_dirty = True
    window.show()
    qt_application.processEvents()

    answers = iter(
        [
            QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Discard,
        ]
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: next(answers),
    )
    assert not window.close()
    assert window.isVisible()
    assert window.close()


def test_hidden_saved_panel_is_shown_and_has_view_menu_recovery(
    qt_application: QApplication,
    tmp_path: Path,
) -> None:
    """Скрытое сохранённое состояние не должно лишать оператора панели."""

    settings = _settings(tmp_path, "hidden-panel.ini")
    source = MainWindow(settings=settings)
    source.operator_dock.hide()
    settings.setValue(WINDOW_STATE_KEY, source.saveState())
    settings.setValue(DOCK_GEOMETRY_KEY, source.operator_dock.saveGeometry())
    source.close()

    restored = MainWindow(
        remember_recent_sources=True,
        settings=settings,
    )
    restored.show()
    qt_application.processEvents()

    assert restored.operator_dock.isVisible()
    assert restored.action_toggle_operator_dock.isChecked()
    assert restored.action_toggle_operator_dock.text() == "Операторская панель"

    # Даже после ручного закрытия переключатель меню остаётся путём возврата.
    restored.action_toggle_operator_dock.trigger()
    assert not restored.operator_dock.isVisible()
    restored.action_toggle_operator_dock.trigger()
    assert restored.operator_dock.isVisible()
    restored.close()
