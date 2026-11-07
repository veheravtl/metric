"""Сквозная проверка моста диагностики между Qt и файловой системой."""

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from aerial_mapper.gui.main_window import MainWindow
from aerial_mapper.live_debug import LiveDebugSession


@pytest.fixture(scope="module")
def qt_application():
    """Создать Qt-приложение без зависимости от физического дисплея."""

    application = QApplication.instance() or QApplication([])
    yield application


def test_f12_incident_is_readable_before_window_closes(
    qt_application: QApplication,
    tmp_path: Path,
) -> None:
    """F12 должен сразу сохранить состояние, исходники и видимые Qt-сцены."""

    video_path = tmp_path / "debug.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("Локальная сборка OpenCV не умеет записывать тестовый MJPG")
    writer.write(np.full((48, 64, 3), (40, 100, 220), dtype=np.uint8))
    writer.write(np.full((48, 64, 3), (220, 100, 40), dtype=np.uint8))
    writer.release()

    map_path = tmp_path / "debug-map.png"
    assert cv2.imwrite(
        str(map_path),
        np.full((72, 96, 3), (80, 80, 80), dtype=np.uint8),
    )

    session = LiveDebugSession(tmp_path / "live")
    window = MainWindow(live_debug=session)
    window.show()
    qt_application.processEvents()
    assert window.video.load_video(video_path)
    assert window.map_image.load_image(map_path)

    window._mark_live_debug_incident()
    qt_application.processEvents()

    # Окно и сессия всё ещё открыты: это ключевое отличие от ручного ZIP.
    assert not session.closed
    pointer = json.loads(
        (tmp_path / "live/latest_session.json").read_text(encoding="utf-8")
    )
    assert Path(pointer["session_directory"]) == session.session_directory

    state = json.loads(session.state_path.read_text(encoding="utf-8"))["state"]
    assert state["video"]["frame_index"] == 0
    assert state["video"]["frame_size_px"] == [64, 48]
    assert state["map"]["image_size_px"] == [96, 72]
    assert state["alignment"]["editor_landmark_ids"] == []
    assert (session.session_directory / "latest-window.jpg").is_file()
    # Полноразмерные latest-PNG раньше кодировались на каждом движении мыши
    # и блокировали GUI; теперь они создаются только внутри явного инцидента.
    assert not (session.session_directory / "latest-frame.png").exists()

    incidents = list(session.incidents_directory.iterdir())
    assert len(incidents) == 1
    incident_directory = incidents[0]
    assert (incident_directory / "incident.json").is_file()
    for filename in (
        "frame.png",
        "map.png",
        "video-view.png",
        "map-view.png",
        "window.png",
    ):
        assert (incident_directory / filename).is_file()

    event_types = [
        json.loads(line)["event_type"]
        for line in session.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert "video_loaded" in event_types
    assert "map_loaded" in event_types
    assert "incident_marked" in event_types
    assert "incident_captured" in event_types
