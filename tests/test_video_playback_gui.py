"""Проверки загрузки и покадровой навигации локального видео."""

import os
from pathlib import Path

# Платформа offscreen позволяет создать QPixmap без настоящего экрана. Значение
# нужно задать до первого импорта PySide6 в этом процессе pytest.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QSettings, QTimer
from PySide6.QtWidgets import QApplication, QGraphicsView

from aerial_mapper.gui.main_window import MainWindow
from aerial_mapper.gui.video_playback import VideoPlaybackController


@pytest.fixture(scope="module")
def qt_application():
    """Предоставить единственный QApplication для всех GUI-проверок модуля."""

    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def three_frame_video(tmp_path: Path) -> Path:
    """Создать три различимых кадра без добавления тестового видео в Git.

    Цвета намеренно сильно различаются. Даже с потерями кодека проверка заметит,
    если кнопка оставила прежний кадр вместо перехода к следующему.
    """

    video_path = tmp_path / "three_frames.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("Локальная сборка OpenCV не умеет записывать тестовый MJPG")

    for bgr_color in ((240, 20, 20), (20, 240, 20), (20, 20, 240)):
        frame = np.full((48, 64, 3), bgr_color, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    return video_path


@pytest.fixture
def three_frame_flight(tmp_path: Path) -> Path:
    """Создать маленький каталог пролёта для интеграции с Qt-контроллером."""

    images_directory = tmp_path / "flight/images"
    images_directory.mkdir(parents=True)
    for image_id, value in enumerate((30, 130, 230)):
        assert cv2.imwrite(
            str(images_directory / f"{image_id:06d}.jpg"),
            np.full((24, 32, 3), value, dtype=np.uint8),
        )

    origin = 1_000_000_000
    (tmp_path / "flight/image_ids.csv").write_text(
        "Timestamp,Image Id\n"
        f"{origin},0\n{origin + 100_000_000},1\n{origin + 200_000_000},2\n",
        encoding="utf-8",
    )
    (tmp_path / "flight/gps.csv").write_text(
        "Timestamp,latitude,longitude,altitude\n"
        f"{origin},22.0,114.0,100.0\n"
        f"{origin + 200_000_000},22.2,114.2,102.0\n",
        encoding="utf-8",
    )
    return tmp_path / "flight"


def test_controller_loads_and_moves_by_exact_frame(
    qt_application: QApplication, three_frame_video: Path
):
    """Обе стрелки должны менять индекс ровно на единицу и не выходить за края."""

    view = QGraphicsView()
    controller = VideoPlaybackController(view)

    assert controller.load_video(three_frame_video)
    assert controller.current_frame_index == 0
    assert controller.frame_count == 3
    assert controller.fps == pytest.approx(10.0)

    first_frame = controller.current_frame_bgr
    assert first_frame is not None
    assert first_frame.shape == (48, 64, 3)

    assert controller.step_forward()
    assert controller.current_frame_index == 1
    second_frame = controller.current_frame_bgr
    assert second_frame is not None
    assert not np.array_equal(first_frame, second_frame)

    assert controller.step_backward()
    assert controller.current_frame_index == 0
    assert not controller.step_backward()

    assert controller.step_forward()
    assert controller.step_forward()
    assert controller.current_frame_index == 2
    assert not controller.step_forward()

    controller.stop()
    assert controller.current_frame_index == 0
    controller.close()
    assert not controller.is_loaded


def test_controller_loads_flight_and_exposes_current_telemetry(
    qt_application: QApplication, three_frame_flight: Path
):
    """Каталог пролёта должен использовать те же кнопки и сохранять GPS-связь."""

    controller = VideoPlaybackController(QGraphicsView())

    assert controller.load_flight(three_frame_flight)
    assert controller.is_flight
    assert controller.current_frame_index == 0
    assert controller.step_forward()

    telemetry = controller.current_frame_telemetry
    assert telemetry is not None
    assert telemetry.image_id == 1
    assert telemetry.elapsed_seconds == pytest.approx(0.1)
    assert telemetry.latitude_degrees == pytest.approx(22.1)
    controller.close()


def test_main_window_shows_first_flight_frame_telemetry(
    qt_application: QApplication, three_frame_flight: Path
):
    """После открытия координаты первого кадра должны быть видны сразу."""

    window = MainWindow()
    assert window.video.load_flight(three_frame_flight)

    status = window.ui.statusbar.currentMessage()
    assert "Image Id 0" in status
    assert "GPS 22.0000000, 114.0000000" in status
    window.close()


def test_play_advances_until_last_frame(
    qt_application: QApplication, three_frame_video: Path
):
    """Play должен последовательно дойти до конца и сам остановить таймер.

    Здесь обрабатывается настоящий цикл событий Qt, поэтому проверяется не
    только арифметика индексов, но и связь ``QTimer`` с декодированием кадров.
    """

    controller = VideoPlaybackController(QGraphicsView())
    assert controller.load_video(three_frame_video)

    event_loop = QEventLoop()
    QTimer.singleShot(400, event_loop.quit)
    controller.play()
    event_loop.exec()

    assert controller.current_frame_index == 2
    assert not controller.is_playing
    controller.close()


def test_main_window_buttons_follow_frame_boundaries(
    qt_application: QApplication, three_frame_video: Path
):
    """Интерфейс не должен предлагать переход за первый или последний кадр."""

    window = MainWindow()
    assert not window.ui.pbPlay.isEnabled()
    assert not window.ui.pbBack.isEnabled()
    assert not window.ui.pbForward.isEnabled()
    assert not window.ui.frameSlider.isEnabled()

    assert window.video.load_video(three_frame_video)
    assert window.ui.pbPlay.isEnabled()
    assert not window.ui.pbBack.isEnabled()
    assert window.ui.pbForward.isEnabled()
    assert window.ui.frameSlider.isEnabled()
    assert window.ui.frameSlider.minimum() == 0
    assert window.ui.frameSlider.maximum() == 2

    window.ui.pbForward.click()
    assert window.video.current_frame_index == 1
    assert window.ui.pbBack.isEnabled()
    assert window.ui.pbForward.isEnabled()

    window.ui.pbForward.click()
    assert window.video.current_frame_index == 2
    assert window.ui.pbBack.isEnabled()
    assert not window.ui.pbForward.isEnabled()

    window.close()


def test_controller_seeks_to_exact_frame(
    qt_application: QApplication, three_frame_video: Path
) -> None:
    """Произвольный переход должен выбирать точный индекс и проверять границы."""

    controller = VideoPlaybackController(QGraphicsView())
    assert controller.load_video(three_frame_video)

    assert controller.seek(2)
    assert controller.current_frame_index == 2
    assert not controller.seek(3)
    assert not controller.seek(-1)
    assert controller.current_frame_index == 2
    controller.close()


def test_frame_slider_seeks_and_follows_other_navigation(
    qt_application: QApplication, three_frame_video: Path
) -> None:
    """Слайдер задаёт кадр, а кнопки и Stop синхронно возвращают его ползунок."""

    window = MainWindow()
    assert window.video.load_video(three_frame_video)

    window.ui.frameSlider.setValue(2)
    assert window.video.current_frame_index == 2

    window.ui.pbBack.click()
    assert window.video.current_frame_index == 1
    assert window.ui.frameSlider.value() == 1

    window.ui.pbStop.click()
    assert window.video.current_frame_index == 0
    assert window.ui.frameSlider.value() == 0
    window.close()


def test_main_window_restores_last_flight_and_map(
    qt_application: QApplication,
    three_frame_flight: Path,
    tmp_path: Path,
) -> None:
    """Новый запуск читает только последние успешно загруженные локальные пути."""

    map_path = tmp_path / "reference.png"
    assert cv2.imwrite(
        str(map_path),
        np.full((30, 40, 3), (20, 80, 160), dtype=np.uint8),
    )
    settings_path = tmp_path / "application.ini"

    first_settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
    first_window = MainWindow(
        remember_recent_sources=True,
        settings=first_settings,
    )
    assert first_window.video.load_flight(three_frame_flight)
    assert first_window.map_image.load_image(map_path)
    first_window.close()

    restored_window = MainWindow(
        remember_recent_sources=True,
        settings=QSettings(str(settings_path), QSettings.Format.IniFormat),
    )
    assert restored_window.video.is_flight
    assert restored_window.video.source_path == three_frame_flight.resolve()
    assert restored_window.video.current_frame_index == 0
    assert restored_window.map_image.source_path == map_path.resolve()
    assert restored_window.ui.frameSlider.maximum() == 2
    restored_window.close()
