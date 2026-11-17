"""Сквозные проверки плавающей панели и защиты рабочего прогресса."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from pyproj import Geod
from PySide6.QtCore import QEvent, QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QColor, QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from aerial_mapper.alignment_session import AlignmentSession
from aerial_mapper.gui import main_window as main_window_module
from aerial_mapper.gui.main_window import (
    DOCK_GEOMETRY_KEY,
    WINDOW_STATE_KEY,
    MainWindow,
)
from aerial_mapper.gui.repeat_measurement import RepeatMeasurementWidget
from aerial_mapper.reference_event_measurement import (
    GeographicPoint,
    NorthUpMapCalibration,
)
from aerial_mapper.repeat_localization import (
    RepeatLocalizationResult,
    RepeatLocalizationService,
    TeachIndexPreparationResult,
)
from aerial_mapper.visual_event_detection import (
    ConfirmedVisualEvent,
    VisualEventDetectionStep,
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


def _make_repeat_flight(tmp_path: Path) -> Path:
    """Создать серию кадров с теми же метаданными, что у рабочего пролёта."""

    flight_directory = tmp_path / "repeat-flight"
    images_directory = flight_directory / "images"
    images_directory.mkdir(parents=True)
    for image_id in range(6):
        frame = np.full((60, 80, 3), 25 + image_id * 30, dtype=np.uint8)
        assert cv2.imwrite(str(images_directory / f"{image_id:06d}.jpg"), frame)

    origin = 1_700_000_000_000_000_000
    image_rows = ["Timestamp,Image Id"]
    gps_rows = ["Timestamp,latitude,longitude,altitude"]
    for image_id in range(6):
        timestamp = origin + image_id * 100_000_000
        image_rows.append(f"{timestamp},{image_id}")
        gps_rows.append(
            f"{timestamp},{22.0 + image_id * 0.0001},"
            f"{114.0 + image_id * 0.0001},{100.0 + image_id}"
        )
    (flight_directory / "image_ids.csv").write_text(
        "\n".join(image_rows) + "\n", encoding="utf-8"
    )
    (flight_directory / "gps.csv").write_text(
        "\n".join(gps_rows) + "\n", encoding="utf-8"
    )
    return flight_directory


def _make_roi_tracking_flight(tmp_path: Path) -> Path:
    """Создать два смещённых текстурных кадра и кадр потери ROI."""

    flight_directory = tmp_path / "roi-tracking-flight"
    images_directory = flight_directory / "images"
    images_directory.mkdir(parents=True)
    generator = np.random.default_rng(20261113)
    noise = generator.integers(0, 256, (80, 100), dtype=np.uint8)
    smooth_texture = cv2.GaussianBlur(noise, (0, 0), 1.6)
    first = cv2.cvtColor(smooth_texture, cv2.COLOR_GRAY2BGR)
    second = cv2.warpAffine(
        first,
        np.float32([[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]]),
        (100, 80),
    )
    frames = (first, second, np.zeros_like(first))
    for image_id, frame in enumerate(frames):
        assert cv2.imwrite(
            str(images_directory / f"{image_id:06d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 98],
        )

    origin = 1_700_000_000_000_000_000
    image_rows = ["Timestamp,Image Id"]
    gps_rows = ["Timestamp,latitude,longitude,altitude"]
    for image_id in range(len(frames)):
        timestamp = origin + image_id * 100_000_000
        image_rows.append(f"{timestamp},{image_id}")
        gps_rows.append(f"{timestamp},22.0,114.0,100.0")
    (flight_directory / "image_ids.csv").write_text(
        "\n".join(image_rows) + "\n", encoding="utf-8"
    )
    (flight_directory / "gps.csv").write_text(
        "\n".join(gps_rows) + "\n", encoding="utf-8"
    )
    return flight_directory


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


def test_large_project_defers_blocking_autosave(
    qt_application: QApplication,
    workspace_sources: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Большой проект не должен самопроизвольно блокировать GUI полной записью."""

    video_path, map_path = workspace_sources
    window = MainWindow(settings=_settings(tmp_path, "large-autosave.ini"))
    assert window.video.load_video(video_path)
    assert window.map_image.load_image(map_path)

    project_path = tmp_path / "large.aerial-project.json"
    project_path.write_text("already large", encoding="utf-8")
    window._project_path = project_path
    window._project_dirty = True
    monkeypatch.setattr(main_window_module, "MAX_AUTOSAVE_PROJECT_BYTES", 1)

    write_attempted = False

    def unexpected_write(*args, **kwargs):
        nonlocal write_attempted
        write_attempted = True
        return True

    monkeypatch.setattr(window, "_write_operator_project", unexpected_write)
    window._autosave_project()

    assert not write_attempted
    assert window._project_dirty
    assert "Ctrl+S" in window.ui.statusbar.currentMessage()
    window._project_dirty = False
    window.close()


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


def test_measurement_is_second_tab_of_main_window(
    qt_application: QApplication,
    workspace_sources: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Измерение встраивается в MainWindow и не создаёт отдельное окно."""

    _video_path, map_path = workspace_sources
    teach_flight = _make_repeat_flight(tmp_path)
    window = MainWindow(settings=_settings(tmp_path, "tabs.ini"))
    assert window.video.load_flight(teach_flight)
    assert window.map_image.load_image(map_path)
    # В рабочем приложении путь появляется после сохранения проекта. В тесте
    # достаточно зафиксировать уже чистое состояние: запись JSON здесь не
    # относится к проверяемой геометрии виджетов.
    window._project_path = tmp_path / "teach.aerial-project.json"
    window._project_dirty = False
    window.show()
    qt_application.processEvents()

    assert window.centralWidget() is window.workspace_tabs
    assert window.workspace_tabs.count() == 2
    assert window.workspace_tabs.tabText(window.annotation_tab_index) == (
        "Привязка и разметка"
    )
    assert window.workspace_tabs.tabText(window.measurement_tab_index) == "Измерение"

    window.action_open_repeat_measurement.trigger()
    qt_application.processEvents()

    workspace = window._measurement_workspace
    assert window.workspace_tabs.currentIndex() == window.measurement_tab_index
    assert isinstance(workspace, RepeatMeasurementWidget)
    assert workspace.parentWidget() is window.measurement_tab
    assert workspace.window() is window
    assert not workspace.isWindow()
    assert not window.operator_dock.isVisible()

    # Модальное окно раньше не давало менять Teach во время расчёта. После
    # переноса во вкладку ту же гарантию обеспечивает временная блокировка.
    workspace._set_busy(True)
    assert not window.workspace_tabs.isTabEnabled(window.annotation_tab_index)
    workspace._set_busy(False)
    assert window.workspace_tabs.isTabEnabled(window.annotation_tab_index)

    window.workspace_tabs.setCurrentIndex(window.annotation_tab_index)
    qt_application.processEvents()
    assert window.operator_dock.isVisible()
    window.close()


def test_repeat_measurement_prepares_visual_index_in_background(
    qt_application: QApplication,
    workspace_sources: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Открытие вкладки должно прогреть индекс, не блокируя главный поток Qt."""

    _video_path, map_path = workspace_sources
    expected = TeachIndexPreparationResult(
        candidate_count=17,
        loaded_from_disk_cache=True,
        processing_time_seconds=0.25,
    )
    calls: list[bool] = []

    def prepare_index(_service: RepeatLocalizationService):
        calls.append(True)
        return expected

    monkeypatch.setattr(
        RepeatLocalizationService,
        "prepare_index",
        prepare_index,
    )
    dialog = RepeatMeasurementWidget(
        teach_flight_directory=tmp_path,
        map_path=map_path,
        session=AlignmentSession(),
        repeat_start_directory=tmp_path,
    )

    try:
        for _ in range(50):
            qt_application.processEvents()
            if dialog._index_preparation_result is not None:
                break
            QTest.qWait(10)

        assert calls == [True]
        assert dialog._index_preparation_result == expected
        assert "17 кадров" in dialog.status_label.text()
    finally:
        dialog.shutdown()


def test_repeat_measurement_plays_frame_stream_and_freezes_exact_frame(
    qt_application: QApplication,
    workspace_sources: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Поток движется по серии кадров, а фиксация сохраняет выбранный кадр.

    Проверка специально использует каталог ``images`` и CSV-метаданные, а не
    видеофайл: именно в таком виде хранятся Repeat-данные пользователя.
    """

    _video_path, map_path = workspace_sources
    repeat_flight = _make_repeat_flight(tmp_path)
    dialog = RepeatMeasurementWidget(
        teach_flight_directory=tmp_path,
        map_path=map_path,
        session=AlignmentSession(),
        repeat_start_directory=tmp_path,
    )

    try:
        assert dialog.video.load_flight(repeat_flight)
        dialog.show()
        qt_application.processEvents()

        assert dialog.video.current_frame_index == 0
        assert dialog.play_button.isEnabled()
        assert not dialog.freeze_button.isEnabled()
        assert dialog.locate_button.isEnabled()
        assert "Стоп-кадр" in dialog.stream_state_label.text()

        dialog.play_button.click()
        assert dialog.video.is_playing
        assert not dialog.play_button.isEnabled()
        assert dialog.freeze_button.isEnabled()
        assert not dialog.frame_spin.isEnabled()
        assert not dialog.previous_button.isEnabled()
        assert not dialog.next_button.isEnabled()
        assert not dialog.locate_button.isEnabled()
        assert "воспроизводится" in dialog.stream_state_label.text()

        # Частота синтетического пролёта равна 10 кадрам/с. Ожидание чуть
        # длиннее одного интервала доказывает, что работает реальный QTimer.
        QTest.qWait(160)
        assert dialog.video.current_frame_index is not None
        assert dialog.video.current_frame_index >= 1

        dialog.freeze_button.click()
        frozen_frame_index = dialog.video.current_frame_index
        assert not dialog.video.is_playing
        assert dialog.play_button.isEnabled()
        assert not dialog.freeze_button.isEnabled()
        assert dialog.frame_spin.isEnabled()
        assert dialog.locate_button.isEnabled()
        assert "Стоп-кадр" in dialog.stream_state_label.text()

        QTest.qWait(160)
        assert dialog.video.current_frame_index == frozen_frame_index
    finally:
        dialog.shutdown()


def test_repeat_measurement_stream_crosshair_captures_and_tracks_roi(
    qt_application: QApplication,
    workspace_sources: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Прицел захватывает ROI, двигает метку и явно сообщает о потере."""

    _video_path, map_path = workspace_sources
    repeat_flight = _make_roi_tracking_flight(tmp_path)
    dialog = RepeatMeasurementWidget(
        teach_flight_directory=tmp_path,
        map_path=map_path,
        session=AlignmentSession(),
        repeat_start_directory=tmp_path,
    )

    try:
        assert dialog.video.load_flight(repeat_flight)
        dialog.resize(1500, 900)
        dialog.show()
        qt_application.processEvents()

        assert not dialog._stream_crosshair.is_visible()
        scene_target = QPointF(50.0, 40.0)
        viewport_target = dialog.video_view.mapFromScene(scene_target)

        dialog.play_button.click()
        assert dialog.video.is_playing
        dialog.video._timer.setInterval(10_000)
        viewport = dialog.video_view.viewport()
        global_target = QPointF(viewport.mapToGlobal(viewport_target))
        mouse_move = QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(viewport_target),
            global_target,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(viewport, mouse_move)
        qt_application.processEvents()

        crosshair = dialog._stream_crosshair
        assert crosshair.is_visible()
        assert viewport.cursor().shape() == Qt.CursorShape.BlankCursor
        assert crosshair.color() == QColor("#ff0000")
        assert crosshair.center_circle.rect().width() == pytest.approx(14.4)
        center = crosshair.center_circle.pos()
        scale_x = abs(dialog.video_view.transform().m11())
        scale_y = abs(dialog.video_view.transform().m22())
        expected_gap_x = crosshair.CIRCLE_RADIUS_SCREEN_PX / scale_x
        expected_gap_y = crosshair.CIRCLE_RADIUS_SCREEN_PX / scale_y
        assert crosshair.left_line.line().x1() == pytest.approx(0.0)
        assert crosshair.left_line.line().x2() == pytest.approx(
            center.x() - expected_gap_x
        )
        assert crosshair.right_line.line().x1() == pytest.approx(
            center.x() + expected_gap_x
        )
        assert crosshair.right_line.line().x2() == pytest.approx(100.0)
        assert crosshair.top_line.line().y1() == pytest.approx(0.0)
        assert crosshair.top_line.line().y2() == pytest.approx(
            center.y() - expected_gap_y
        )
        assert crosshair.bottom_line.line().y1() == pytest.approx(
            center.y() + expected_gap_y
        )
        assert crosshair.bottom_line.line().y2() == pytest.approx(80.0)
        pen = crosshair.left_line.pen()
        assert pen.isCosmetic()
        assert pen.widthF() == pytest.approx(1.0)

        mouse_press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(viewport_target),
            global_target,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(viewport, mouse_press)
        qt_application.processEvents()

        assert dialog._roi_tracker.is_active
        assert dialog._roi_marker.is_visible()
        assert crosshair.color() == QColor("#ffcc00")
        assert dialog._roi_marker.item.brush().color().alpha() == 80
        assert dialog._roi_marker.item.rect().width() == pytest.approx(40.0)
        assert dialog._roi_marker.item.pos().x() == pytest.approx(50.0, abs=0.2)
        assert dialog._roi_marker.item.pos().y() == pytest.approx(40.0, abs=0.2)

        dialog.freeze_button.click()
        assert not dialog.video.is_playing
        assert dialog._roi_tracker.is_active
        assert dialog._roi_marker.is_visible()
        dialog.play_button.click()
        dialog.video._timer.setInterval(10_000)
        assert dialog.video.is_playing
        assert crosshair.color() == QColor("#ffcc00")

        assert dialog.video._show_frame(1)
        tracked = dialog._roi_marker.item.pos()
        assert dialog._roi_tracker.is_active
        assert tracked.x() == pytest.approx(53.0, abs=0.5)
        assert tracked.y() == pytest.approx(42.0, abs=0.5)
        assert crosshair.color() == QColor("#ffcc00")

        assert dialog.video._show_frame(2)
        assert dialog.video.is_playing
        assert not dialog._roi_tracker.is_active
        assert not dialog._roi_marker.is_visible()
        assert crosshair.color() == QColor("#ff0000")
        assert "потеряна" in dialog.status_label.text()
        assert "Поток продолжается" in dialog.status_label.text()

        dialog.freeze_button.click()
        assert not crosshair.is_visible()
        assert viewport.cursor().shape() == Qt.CursorShape.ArrowCursor
    finally:
        dialog.shutdown()


def test_right_click_holds_frame_then_roi_catches_up_to_live_stream(
    qt_application: QApplication,
    workspace_sources: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """ПКМ удерживает картинку, а источник и последующий трекинг не стоят."""

    _video_path, map_path = workspace_sources
    repeat_flight = _make_roi_tracking_flight(tmp_path)
    dialog = RepeatMeasurementWidget(
        teach_flight_directory=tmp_path,
        map_path=map_path,
        session=AlignmentSession(),
        repeat_start_directory=tmp_path,
    )

    try:
        assert dialog.video.load_flight(repeat_flight)
        dialog.resize(1500, 900)
        dialog.show()
        qt_application.processEvents()
        dialog.play_button.click()
        dialog.video._timer.setInterval(10_000)

        scene_target = QPointF(50.0, 40.0)
        viewport_target = dialog.video_view.mapFromScene(scene_target)
        viewport = dialog.video_view.viewport()
        global_target = QPointF(viewport.mapToGlobal(viewport_target))
        right_press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(viewport_target),
            global_target,
            Qt.MouseButton.RightButton,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(viewport, right_press)
        qt_application.processEvents()

        assert dialog.video.is_playing
        assert dialog.video.is_display_frozen
        assert dialog.video.current_frame_index == 0
        assert dialog.video.displayed_frame_index == 0
        assert dialog._stream_selection_frame_index == 0
        assert "поток продолжает идти" in dialog.stream_state_label.text().lower()

        # Источник переходит на следующий кадр, но оператор всё ещё видит
        # кадр 0 и ставит точку в его геометрии.
        assert dialog.video._show_frame(1)
        assert dialog.video.current_frame_index == 1
        assert dialog.video.displayed_frame_index == 0

        left_press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(viewport_target),
            global_target,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(viewport, left_press)
        qt_application.processEvents()

        assert dialog.video.is_playing
        assert not dialog.video.is_display_frozen
        assert dialog.video.current_frame_index == 1
        assert dialog.video.displayed_frame_index == 1
        assert dialog._stream_selection_frame_index is None
        assert dialog._roi_tracker.is_active
        assert dialog._roi_tracker.frame_index == 1
        tracked = dialog._roi_marker.item.pos()
        assert tracked.x() == pytest.approx(53.0, abs=0.5)
        assert tracked.y() == pytest.approx(42.0, abs=0.5)
        assert "догнало живой кадр 2" in dialog.status_label.text()
    finally:
        dialog.shutdown()


def test_confirmed_event_stops_stream_marks_frame_and_starts_measurement(
    qt_application: QApplication,
    workspace_sources: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Подтверждение ставит диагональный крест и запускает замер POI→событие."""

    _video_path, map_path = workspace_sources
    repeat_flight = _make_roi_tracking_flight(tmp_path)
    dialog = RepeatMeasurementWidget(
        teach_flight_directory=tmp_path,
        map_path=map_path,
        session=AlignmentSession(),
        repeat_start_directory=tmp_path,
    )

    try:
        assert dialog.video.load_flight(repeat_flight)
        dialog.show()
        qt_application.processEvents()
        dialog.play_button.click()
        dialog.video._timer.setInterval(10_000)
        frame = dialog.video.current_frame_bgr
        assert frame is not None
        dialog._roi_tracker.capture(frame, frame_index=0, point_px=(50.0, 40.0))
        dialog._roi_marker.show_at((50.0, 40.0))

        event = ConfirmedVisualEvent(
            onset_frame_index=0,
            confirmation_frame_index=1,
            point_px=(70.0, 45.0),
            bounding_box_px=(65.0, 40.0, 10.0, 10.0),
        )

        def confirmed_step(_frame, *, frame_index: int) -> VisualEventDetectionStep:
            assert frame_index == 1
            return VisualEventDetectionStep(
                frame_index=frame_index,
                status="confirmed",
                candidate=None,
                confirmed_event=event,
                tracked_feature_count=100,
                inlier_count=90,
                inlier_ratio=0.9,
                residual_threshold=20.0,
                changed_fraction=0.01,
                processing_time_ms=5.0,
            )

        localization_started: list[bool] = []
        monkeypatch.setattr(dialog._event_detector, "advance", confirmed_step)
        monkeypatch.setattr(
            dialog,
            "_start_localization",
            lambda: localization_started.append(True),
        )

        assert dialog.video._show_frame(1)

        assert not dialog.video.is_playing
        assert localization_started == [True]
        assert dialog._event_marker.is_visible()
        event_position = dialog._event_marker.first_line.pos()
        assert (event_position.x(), event_position.y()) == pytest.approx((70.0, 45.0))
        first = dialog._event_marker.first_line.line()
        second = dialog._event_marker.second_line.line()
        radius = dialog._event_marker.RADIUS_SCREEN_PX
        assert (first.x1(), first.y1(), first.x2(), first.y2()) == pytest.approx(
            (-radius, -radius, radius, radius)
        )
        assert (second.x1(), second.y1(), second.x2(), second.y2()) == pytest.approx(
            (-radius, radius, radius, -radius)
        )
        assert dialog._event_marker.first_line.pen().isCosmetic()
        assert dialog._event_marker.first_line.pen().widthF() == pytest.approx(2.0)
        assert 2.0 * radius == pytest.approx(
            dialog._stream_crosshair.center_circle.rect().width()
        )
        assert dialog._segment_points[0] == pytest.approx((53.0, 42.0), abs=0.5)
        assert dialog._segment_points[1] == pytest.approx((70.0, 45.0))
        assert dialog._pending_event_measurement

        calculations: list[bool] = []
        monkeypatch.setattr(
            dialog,
            "_calculate_measurement",
            lambda *, event_measurement=False: calculations.append(event_measurement),
        )
        dialog._on_localization_completed(
            RepeatLocalizationResult(
                repeat_frame_index=1,
                repeat_image_id=1,
                teach_frame_index=1,
                teach_image_id=1,
                homography_repeat_to_map=np.eye(3, dtype=np.float64),
                retrieval_rank=1,
                eligible_candidate_count=1,
                inlier_count=90,
                inlier_ratio=0.9,
                frame_coverage_fraction=0.8,
                reprojection_p95_px=1.0,
                stability_p95_corner_shift_px=1.0,
                processing_time_seconds=0.1,
            )
        )
        assert calculations == [True]
        assert not dialog._pending_event_measurement
    finally:
        dialog.shutdown()
def test_automatic_event_outputs_reference_angles_and_distances(
    qt_application: QApplication,
    workspace_sources: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Автособытие использует репер, а ручная пара остаётся обычным отрезком."""

    _video_path, map_path = workspace_sources
    anchor = GeographicPoint(22.416088, 114.042978)
    calibration = NorthUpMapCalibration(
        anchor=anchor,
        anchor_map_point_px=(50.0, 50.0),
        map_width_pixels=160,
        map_height_pixels=140,
    )
    reference_lon, reference_lat, _ = Geod(ellps="WGS84").fwd(
        anchor.longitude_degrees,
        anchor.latitude_degrees,
        180.0,
        1000.0,
    )
    dialog = RepeatMeasurementWidget(
        teach_flight_directory=tmp_path,
        map_path=map_path,
        session=AlignmentSession(),
        repeat_start_directory=tmp_path,
        map_scale_meters_per_pixel=1.0,
        map_calibration=calibration,
        reference_point=GeographicPoint(reference_lat, reference_lon),
    )

    try:
        dialog._localization = RepeatLocalizationResult(
            repeat_frame_index=0,
            repeat_image_id=0,
            teach_frame_index=0,
            teach_image_id=0,
            homography_repeat_to_map=np.eye(3, dtype=np.float64),
            retrieval_rank=1,
            eligible_candidate_count=1,
            inlier_count=90,
            inlier_ratio=0.9,
            frame_coverage_fraction=0.8,
            reprojection_p95_px=1.0,
            stability_p95_corner_shift_px=1.0,
            processing_time_seconds=0.1,
        )
        dialog._segment_points[:] = [(50.0, 50.0), (60.0, 50.0)]

        dialog._calculate_measurement(event_measurement=True)

        result = dialog.reference_event_measurement
        assert result is not None
        assert result.target.distance_meters == pytest.approx(1000.0, abs=1e-6)
        assert result.event_azimuth_deviation_degrees > 0.0
        assert "Цель (ROI)" in dialog.result_label.text()
        assert "Азимут относительно цели: 0.000°" in dialog.result_label.text()
        assert "Дистанция от репера: 1000.0 м" in dialog.result_label.text()
        assert "Событие" in dialog.result_label.text()
        assert "Разность дистанций" in dialog.result_label.text()

        dialog._calculate_measurement()

        assert dialog.reference_event_measurement is None
        assert "Длина: 10.000 м" in dialog.result_label.text()
        assert "Цель (ROI)" not in dialog.result_label.text()
    finally:
        dialog.shutdown()


def test_repeat_measurement_views_zoom_independently_and_resize_with_mouse(
    qt_application: QApplication,
    workspace_sources: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Колесо меняет нужную сцену, а мышь двигает границу двух областей."""

    video_path, map_path = workspace_sources
    dialog = RepeatMeasurementWidget(
        teach_flight_directory=tmp_path,
        map_path=map_path,
        session=AlignmentSession(),
        repeat_start_directory=tmp_path,
    )

    try:
        assert dialog.video.load_video(video_path)
        dialog.resize(1500, 900)
        dialog.show()
        qt_application.processEvents()

        def wheel_up(view) -> None:
            viewport = view.viewport()
            local_position = QPointF(viewport.rect().center())
            global_position = QPointF(viewport.mapToGlobal(local_position.toPoint()))
            event = QWheelEvent(
                local_position,
                global_position,
                QPoint(),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
            QApplication.sendEvent(viewport, event)

        video_scale_before = dialog.video_view.transform().m11()
        map_scale_before = dialog.map_view.transform().m11()
        wheel_up(dialog.video_view)
        assert dialog.video_view.transform().m11() == pytest.approx(
            video_scale_before * 1.2
        )
        assert dialog.map_view.transform().m11() == pytest.approx(map_scale_before)

        wheel_up(dialog.map_view)
        assert dialog.map_view.transform().m11() == pytest.approx(
            map_scale_before * 1.2
        )

        splitter = dialog.view_splitter
        assert not splitter.childrenCollapsible()
        assert splitter.handleWidth() == 8
        sizes_before = splitter.sizes()
        handle = splitter.handle(1)
        start = handle.rect().center()
        target = QPoint(start.x() + 100, start.y())
        QTest.mousePress(
            handle,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start,
        )
        QTest.mouseMove(handle, target, delay=10)
        QTest.mouseRelease(
            handle,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            target,
        )
        qt_application.processEvents()
        sizes_after = splitter.sizes()
        assert sizes_after != sizes_before
        assert all(size > 0 for size in sizes_after)
    finally:
        dialog.shutdown()
