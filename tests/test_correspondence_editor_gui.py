"""Сквозные проверки ручного совмещения в Qt-сценах."""

import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGraphicsView, QInputDialog

from aerial_mapper.beacon_tracking import track_beacons_with_machine_support_to_map
from aerial_mapper.gui.correspondence_editor import CorrespondenceEditor
from aerial_mapper.gui.main_window import MainWindow
from aerial_mapper.gui.map_image import MapImageController
from aerial_mapper.gui.video_playback import VideoPlaybackController
from aerial_mapper.live_debug import LiveDebugSession


@pytest.fixture(scope="module")
def qt_application():
    """Создать QApplication без зависимости от физического экрана."""

    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def synthetic_sources(tmp_path: Path) -> tuple[Path, Path]:
    """Создать двухкадровое видео и карту с известными размерами.

    Реальное содержимое изображения для геометрии не используется: точки
    задаются явно. Разные цвета двух кадров нужны, чтобы проверить обязательный
    сброс разметки при переходе на следующий кадр.
    """

    video_path = tmp_path / "alignment.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (100, 80),
    )
    if not writer.isOpened():
        pytest.skip("Локальная сборка OpenCV не умеет записывать тестовый MJPG")
    writer.write(np.full((80, 100, 3), (40, 100, 220), dtype=np.uint8))
    writer.write(np.full((80, 100, 3), (220, 100, 40), dtype=np.uint8))
    writer.release()

    map_path = tmp_path / "map.png"
    assert cv2.imwrite(
        str(map_path),
        np.full((140, 160, 3), (80, 80, 80), dtype=np.uint8),
    )
    return video_path, map_path


def _create_editor(
    video_path: Path, map_path: Path
) -> tuple[
    VideoPlaybackController,
    MapImageController,
    CorrespondenceEditor,
    QGraphicsView,
    QGraphicsView,
]:
    video_view = QGraphicsView()
    map_view = QGraphicsView()
    video = VideoPlaybackController(video_view)
    map_image = MapImageController(map_view)
    editor = CorrespondenceEditor(video_view, map_view, video, map_image)
    video.frame_changed.connect(editor.on_frame_changed)
    map_image.map_loaded.connect(editor.on_map_loaded)

    assert video.load_video(video_path)
    assert map_image.load_image(map_path)
    return video, map_image, editor, video_view, map_view


def test_known_manual_alignment_builds_overlay(
    qt_application: QApplication,
    synthetic_sources: tuple[Path, Path],
) -> None:
    """Известное преобразование должно восстановиться и дать наложение.

    Используем шесть распределённых точек, а не математический минимум из
    четырёх. Поэтому независимая проверка «оставить одну точку вне расчёта»
    тоже обязана показать почти нулевую ошибку.
    """

    video_path, map_path = synthetic_sources
    video, map_image, editor, _video_view, _map_view = _create_editor(
        video_path, map_path
    )
    homography = np.array(
        [[1.2, 0.03, 15.0], [-0.02, 1.1, 12.0], [0.0002, -0.0001, 1.0]],
        dtype=np.float64,
    )
    frame_points = np.array(
        [
            [5.0, 5.0],
            [90.0, 8.0],
            [92.0, 70.0],
            [7.0, 72.0],
            [48.0, 20.0],
            [55.0, 62.0],
        ],
        dtype=np.float64,
    )
    map_points = cv2.perspectiveTransform(
        frame_points.reshape(1, -1, 2), homography
    ).reshape(-1, 2)

    for frame_point, map_point in zip(frame_points, map_points, strict=True):
        assert editor.add_correspondence(tuple(frame_point), tuple(map_point))

    result = editor.compute_alignment()
    assert result is not None
    assert editor.pair_count == 6
    assert editor.overlay_available
    assert editor.overlay_visible
    assert result.fit_rmse_map_px < 1e-4
    assert result.leave_one_out_max_map_px is not None
    assert result.leave_one_out_max_map_px < 1e-3

    frame = video.current_frame_bgr
    assert frame is not None
    assert editor.show_automatic_alignment_overlay(frame, homography)
    # Показ авто-слоя не должен уничтожать ручной эталон того же кадра: оба
    # результата нужны одновременно при формировании диагностического ZIP.
    assert editor.result is result

    editor.set_overlay_visible(False)
    assert not editor.overlay_visible
    editor.set_overlay_visible(True)
    assert editor.overlay_visible

    editor.clear()
    video.close()
    map_image.clear()


def test_bad_manual_pair_is_rejected_without_showing_distorted_overlay(
    qt_application: QApplication,
    synthetic_sources: tuple[Path, Path],
) -> None:
    """Явная перепутанная пара остаётся черновиком, а не ложной опорой."""

    video_path, map_path = synthetic_sources
    video, map_image, editor, _video_view, _map_view = _create_editor(
        video_path, map_path
    )
    frame_points = np.asarray(
        [
            [5.0, 5.0],
            [90.0, 8.0],
            [92.0, 70.0],
            [7.0, 72.0],
            [48.0, 20.0],
            [55.0, 62.0],
        ]
    )
    map_points = frame_points.copy()
    mismatched_index = 4
    map_points[mismatched_index] += [40.0, 40.0]
    errors: list[str] = []
    debug_events: list[tuple[str, object, bool]] = []
    editor.error_occurred.connect(errors.append)
    editor.debug_event.connect(
        lambda event_type, payload, capture: debug_events.append(
            (event_type, payload, capture)
        )
    )

    for frame_point, map_point in zip(frame_points, map_points, strict=True):
        assert editor.add_correspondence(tuple(frame_point), tuple(map_point))

    assert editor.compute_alignment() is None
    assert editor.result is None
    assert not editor.overlay_available
    assert editor.pair_count == len(frame_points)
    assert editor.correction_pending
    assert editor.debug_snapshot()["suspect_landmark_ids"] == [mismatched_index]
    assert errors and "пара 5" in errors[-1]
    rejection = [event for event in debug_events if event[0] == "alignment_rejected"]
    assert len(rejection) == 1
    assert rejection[0][1]["suspected_landmark_id"] == mismatched_index
    assert rejection[0][2]

    editor.dispose()
    video.close()
    map_image.clear()


def test_frame_change_hides_unknown_frame_and_restores_remembered_frame(
    qt_application: QApplication,
    synthetic_sources: tuple[Path, Path],
) -> None:
    """Неизвестный кадр пуст, но возврат восстанавливает сохранённый черновик."""

    video_path, map_path = synthetic_sources
    video, map_image, editor, _video_view, _map_view = _create_editor(
        video_path, map_path
    )
    assert editor.add_correspondence((10.0, 10.0), (20.0, 20.0))
    assert editor.pair_count == 1

    assert video.step_forward()
    assert editor.pair_count == 0
    assert not editor.has_pending_point
    assert editor.result is None
    assert editor.remembered_frame_indices == (0,)

    assert video.step_backward()
    assert editor.pair_count == 1
    assert editor.current_landmark_ids == (0,)

    video.close()
    map_image.clear()


def test_frame_change_invalidates_automatic_overlay(
    qt_application: QApplication,
    synthetic_sources: tuple[Path, Path],
) -> None:
    """Автоматический слой также должен принадлежать ровно одному кадру."""

    video_path, map_path = synthetic_sources
    video, map_image, editor, _video_view, _map_view = _create_editor(
        video_path, map_path
    )
    frame = video.current_frame_bgr
    assert frame is not None
    assert editor.show_automatic_alignment_overlay(frame, np.eye(3))
    assert editor.overlay_visible

    assert video.step_forward()
    assert not editor.overlay_available
    assert not editor.overlay_visible

    editor.dispose()
    video.close()
    map_image.clear()


def test_mouse_clicks_are_converted_to_image_coordinates(
    qt_application: QApplication,
    synthetic_sources: tuple[Path, Path],
) -> None:
    """Настоящие клики Qt должны образовать пару в правильном порядке."""

    video_path, map_path = synthetic_sources
    video, map_image, editor, video_view, map_view = _create_editor(
        video_path, map_path
    )
    video_view.resize(400, 300)
    map_view.resize(400, 300)
    video_view.show()
    map_view.show()
    video.fit_to_view()
    map_image.fit_to_view()
    qt_application.processEvents()

    editor.set_active(True)
    video_click = video_view.mapFromScene(QPointF(20.0, 20.0))
    QTest.mouseClick(video_view.viewport(), Qt.MouseButton.LeftButton, pos=video_click)
    assert editor.has_pending_point

    map_click = map_view.mapFromScene(QPointF(40.0, 35.0))
    QTest.mouseClick(map_view.viewport(), Qt.MouseButton.LeftButton, pos=map_click)
    assert not editor.has_pending_point
    assert editor.pair_count == 1

    editor.dispose()
    video.close()
    map_image.clear()


def test_tracking_failure_keeps_history_and_accepts_corrected_keyframe(
    qt_application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Срыв не стирает маяки, а ручное уточнение создаёт новую опору."""

    random_generator = np.random.default_rng(20261105)
    gray = random_generator.integers(0, 256, size=(240, 320), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
    anchor_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    motion = np.asarray(
        [[1.0, 0.0, 3.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    next_bgr = cv2.warpPerspective(anchor_bgr, motion, (320, 240))
    detected = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=30,
        qualityLevel=0.05,
        minDistance=15,
        blockSize=7,
    )
    assert detected is not None
    frame_points = detected.reshape(-1, 2).astype(np.float64)

    video_path = tmp_path / "tracking.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (320, 240),
    )
    if not writer.isOpened():
        pytest.skip("Локальная сборка OpenCV не умеет записывать тестовый MJPG")
    writer.write(anchor_bgr)
    writer.write(next_bgr)
    writer.write(np.zeros_like(anchor_bgr))
    writer.release()

    map_path = tmp_path / "tracking-map.png"
    assert cv2.imwrite(str(map_path), anchor_bgr)

    debug_session = LiveDebugSession(tmp_path / "live-debug")
    window = MainWindow(live_debug=debug_session)
    assert window.video.load_video(video_path)
    assert window.map_image.load_image(map_path)
    for point in frame_points:
        coordinates = (float(point[0]), float(point[1]))
        assert window.alignment.add_correspondence(coordinates, coordinates)
    assert window.alignment.compute_alignment() is not None
    assert window.ui.actionTrack_points.isEnabled()

    # Регресс памяти кадров: заранее посещённый черновик следующего кадра не
    # должен перехватывать переход и останавливать явно включённый трекер.
    window.ui.pbForward.click()
    assert window.video.current_frame_index == 1
    assert window.alignment.add_correspondence((40.0, 40.0), (40.0, 40.0))
    assert window.alignment.correction_pending
    window.ui.pbBack.click()
    assert window.video.current_frame_index == 0
    assert window.alignment.current_frame_is_keyframe

    initial_landmark_ids = window.alignment.current_landmark_ids
    excluded_landmark_ids: set[int] = set()

    # Воспроизводим полевой случай: поток нашёл все точки, но геометрический
    # фильтр исключил маяки 3, 4 и 6 из оценки гомографии.
    def tracker_with_forced_outliers(*args, **kwargs):
        result = track_beacons_with_machine_support_to_map(*args, **kwargs)
        inlier_mask = result.homography_inlier_mask.copy()
        for index in (2, 3, 5):
            inlier_mask[index] = False
        excluded_landmark_ids.update(
            initial_landmark_ids[index] for index in np.flatnonzero(~inlier_mask)
        )
        return replace(result, homography_inlier_mask=inlier_mask)

    monkeypatch.setattr(
        "aerial_mapper.gui.correspondence_editor."
        "track_beacons_with_machine_support_to_map",
        tracker_with_forced_outliers,
    )

    window.ui.actionTrack_points.setChecked(True)
    assert window.alignment.tracking_active
    window.ui.pbForward.click()

    assert window.video.current_frame_index == 1
    assert window.alignment.tracking_active
    assert window.alignment.last_tracking_result is not None
    assert window.alignment.pair_count == len(initial_landmark_ids)
    assert window.alignment.overlay_visible
    snapshot = window.alignment.debug_snapshot()
    assert set(snapshot["suspect_landmark_ids"]) == excluded_landmark_ids
    assert tuple(snapshot["frame_scene_landmark_ids"]) == initial_landmark_ids
    assert tuple(snapshot["map_scene_landmark_ids"]) == initial_landmark_ids
    assert snapshot["last_tracking"]["automatic_detected_count"] > 0
    assert snapshot["last_tracking"]["automatic_inlier_count"] > 0
    assert len(snapshot["machine_observations"]) > 0
    remembered_count = len(initial_landmark_ids)
    assert "сопровождение активно" in window.ui.statusbar.currentMessage()

    window.ui.pbForward.click()
    assert window.video.current_frame_index == 2
    assert not window.alignment.tracking_active
    assert window.alignment.correction_pending
    assert window.alignment.pair_count == remembered_count
    assert window.alignment.remembered_frame_indices == (0, 1, 2)
    assert "последних позиций сохранено" in window.ui.statusbar.currentMessage()
    rejected_snapshot = window.alignment.debug_snapshot()
    assert rejected_snapshot["annotation_status"] == "rejected"
    assert "почти полностью чёрный" in rejected_snapshot["rejection_reasons"][0]

    failure_events = [
        json.loads(line)
        for line in debug_session.events_path.read_text(encoding="utf-8").splitlines()
        if '"event_type":"tracking_failed"' in line
    ]
    assert len(failure_events) == 1
    failure = failure_events[0]["payload"]
    assert failure["previous_frame_index"] == 1
    assert failure["frame_index"] == 2
    assert failure["failure_stage"] == "frame_quality"
    assert failure["image_quality_warnings"] == []
    assert len(failure["landmarks"]) == remembered_count
    assert all("landmark_id" in item for item in failure["landmarks"])
    assert all("forward_backward_error_px" in item for item in failure["landmarks"])

    success_events = [
        json.loads(line)
        for line in debug_session.events_path.read_text(encoding="utf-8").splitlines()
        if '"event_type":"tracking_succeeded"' in line
    ]
    assert len(success_events) == 1
    success = success_events[0]["payload"]
    assert success["dropped_landmark_ids"] == []
    assert set(success["excluded_from_alignment_landmark_ids"]) == excluded_landmark_ids

    incident_reasons = [
        json.loads(path.read_text(encoding="utf-8"))["reason"]
        for path in debug_session.incidents_directory.glob("*/incident.json")
    ]
    assert "tracking_failed" in incident_reasons
    assert "tracking_succeeded" not in incident_reasons

    # Возврат к хорошему кадру восстанавливает и точки, и его гомографию.
    window.ui.pbBack.click()
    assert window.video.current_frame_index == 1
    assert window.alignment.pair_count == remembered_count
    restored_snapshot = window.alignment.debug_snapshot()
    assert set(restored_snapshot["suspect_landmark_ids"]) == excluded_landmark_ids
    assert window.alignment.result is not None
    assert window.alignment.overlay_visible

    # Черновик сорванного кадра тоже восстанавливается. Пустой тестовый кадр не
    # содержит реальных ориентиров, поэтому воспроизводимо возвращаем все
    # видеоточки к известным согласованным координатам. Одна произвольная правка
    # больше не должна проходить защитный порог как надёжная опора.
    window.ui.pbForward.click()
    assert window.video.current_frame_index == 2
    assert window.alignment.correction_pending
    for landmark_id, point in zip(
        window.alignment.current_landmark_ids, frame_points, strict=True
    ):
        assert window.alignment.refine_frame_point(landmark_id, tuple(point))
    assert window.alignment.compute_alignment() is not None
    assert window.alignment.current_frame_is_keyframe
    assert not window.alignment.correction_pending
    assert window.alignment.tracking_active
    assert window.ui.actionTrack_points.isChecked()
    assert window.alignment.overlay_visible
    window.close()
    assert debug_session.closed


def test_operator_can_exclude_current_frame_and_advance(
    qt_application: QApplication,
    synthetic_sources: tuple[Path, Path],
) -> None:
    """Явный пропуск не должен оставлять кадр неразмеченным или repairable."""

    video_path, map_path = synthetic_sources
    window = MainWindow()
    assert window.video.load_video(video_path)
    assert window.map_image.load_image(map_path)
    assert window.ui.pbExcludeFrame.isEnabled()
    assert window.ui.actionExclude_frame.isEnabled()

    window.ui.pbExcludeFrame.click()

    assert window.video.current_frame_index == 1
    assert window.alignment.remembered_frame_indices == (0,)
    assert "Кадр 0 исключён" in window.ui.statusbar.currentMessage()

    window.ui.pbBack.click()
    snapshot = window.alignment.debug_snapshot()
    assert snapshot["annotation_status"] == "excluded"
    assert snapshot["observations"] == []
    assert not snapshot["correction_pending"]
    assert "оператор исключил кадр" in snapshot["rejection_reasons"][0]

    # Исключение обратимо обычной новой разметкой: первый маяк переводит кадр
    # обратно в черновик и убирает причину исключения.
    assert window.alignment.add_correspondence((10.0, 10.0), (20.0, 20.0))
    restored = window.alignment.debug_snapshot()
    assert restored["annotation_status"] == "draft"
    assert restored["rejection_reasons"] == ()
    window.close()


def test_operator_can_exclude_range_without_visiting_every_frame(
    qt_application: QApplication,
    synthetic_sources: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Диапазонная команда сохраняет статус каждого индекса за одно действие."""

    video_path, map_path = synthetic_sources
    window = MainWindow()
    assert window.video.load_video(video_path)
    assert window.map_image.load_image(map_path)
    monkeypatch.setattr(
        QInputDialog,
        "getInt",
        lambda *args, **kwargs: (1, True),
    )

    window.ui.actionExclude_frame_range.trigger()

    assert window.video.current_frame_index == 1
    assert window.alignment.remembered_frame_indices == (0, 1)
    assert "Исключено кадров: 2" in window.ui.statusbar.currentMessage()
    current = window.alignment.debug_snapshot()
    assert current["annotation_status"] == "excluded"
    assert not current["correction_pending"]

    window.ui.pbBack.click()
    previous = window.alignment.debug_snapshot()
    assert previous["annotation_status"] == "excluded"
    assert previous["observations"] == []
    window.close()


def test_tracking_never_uses_preexcluded_frame_as_temporal_support(
    qt_application: QApplication,
    synthetic_sources: tuple[Path, Path],
) -> None:
    """Статус excluded должен разрывать трек, а не запускать скрытую попытку."""

    video_path, map_path = synthetic_sources
    window = MainWindow()
    assert window.video.load_video(video_path)
    assert window.map_image.load_image(map_path)
    points = (
        (5.0, 5.0),
        (90.0, 5.0),
        (90.0, 70.0),
        (5.0, 70.0),
        (48.0, 20.0),
        (55.0, 62.0),
    )
    for point in points:
        assert window.alignment.add_correspondence(point, point)
    assert window.alignment.compute_alignment() is not None

    # Будущий кадр исключается без перехода на него. Если затем включить
    # сопровождение, редактор обязан восстановить статус и остановиться до
    # вызова оптического потока.
    assert (
        window.alignment.exclude_frame_range(
            1,
            1,
            reason="камера заранее известна как находящаяся вне ROI",
        )
        == 1
    )
    window.ui.actionTrack_points.setChecked(True)
    assert window.alignment.tracking_active

    window.ui.pbForward.click()

    snapshot = window.alignment.debug_snapshot()
    assert window.video.current_frame_index == 1
    assert snapshot["annotation_status"] == "excluded"
    assert not snapshot["tracking_active"]
    assert snapshot["last_tracking"] is None
    assert snapshot["observations"] == []
    window.close()
