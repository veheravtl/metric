# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Отдельное окно замеров на стоп-кадре независимого Repeat-пролёта."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QSignalBlocker,
    Qt,
    QThread,
    Signal,
    Slot,
)
from PySide6.QtGui import QCloseEvent, QColor, QCursor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from aerial_mapper.alignment_session import AlignmentSession
from aerial_mapper.local_roi_tracking import (
    LocalRoiTracker,
    LocalRoiTrackingFailure,
)
from aerial_mapper.reference_event_measurement import (
    GeographicPoint,
    NorthUpMapCalibration,
    ReferenceEventMeasurementFailure,
    TargetEventReferenceMeasurement,
    measure_target_and_event_from_reference,
)
from aerial_mapper.repeat_localization import (
    RepeatLocalizationFailure,
    RepeatLocalizationResult,
    RepeatLocalizationService,
    ScaledMapSegmentMeasurement,
    TeachIndexPreparationResult,
    measure_scaled_map_segment,
)
from aerial_mapper.visual_event_detection import (
    ConfirmedVisualEvent,
    VisualEventDetector,
)

from .map_image import MapImageController
from .video_playback import VideoPlaybackController

# Ручная калибровка пользовательского снимка:
# 226,50 м / 891,842054 px = 0,2539687368 м/px.
HK_AIRFIELD_SCALE_METERS_PER_PIXEL = 0.2539687368

# Координатная опора получена из сохранённого операторского проекта priv.json.
# Она относится только к исходному север-ориентированному PNG 1242×986 px.
HK_AIRFIELD_MAP_CALIBRATION = NorthUpMapCalibration(
    anchor=GeographicPoint(22.416088, 114.042978),
    anchor_map_point_px=(506.91896724872566, 471.26229897800636),
    map_width_pixels=1242,
    map_height_pixels=986,
)
HK_REFERENCE_POINT = GeographicPoint(22.367007, 114.055838)


def _format_reference_event_measurement(
    measurement: TargetEventReferenceMeasurement,
) -> str:
    """Собрать компактный, но однозначный операторский результат."""

    target = measurement.target
    event = measurement.event
    return (
        "Цель (ROI)\n"
        f"Координаты: {target.point.latitude_degrees:.6f}, "
        f"{target.point.longitude_degrees:.6f}\n"
        "Азимут относительно цели: 0.000°\n"
        f"Дистанция от репера: {target.distance_meters:.1f} м\n\n"
        "Событие\n"
        f"Координаты: {event.point.latitude_degrees:.6f}, "
        f"{event.point.longitude_degrees:.6f}\n"
        "Отклонение от цели: "
        f"{measurement.event_azimuth_deviation_degrees:+.3f}°\n"
        f"Дистанция от репера: {event.distance_meters:.1f} м\n"
        "Разность дистанций (событие − цель): "
        f"{measurement.distance_difference_meters:+.1f} м"
    )


class _StreamPointCrosshair:
    """Прицел выбора точки, не изменяющий пиксели видеокадра."""

    READY_COLOR = QColor("#ff0000")
    CAPTURED_COLOR = QColor("#ffcc00")
    CIRCLE_RADIUS_SCREEN_PX = 7.2

    def __init__(self, scene: QGraphicsScene, view: QGraphicsView) -> None:
        self._view = view
        self.left_line = QGraphicsLineItem()
        self.right_line = QGraphicsLineItem()
        self.top_line = QGraphicsLineItem()
        self.bottom_line = QGraphicsLineItem()
        self._lines = (
            self.left_line,
            self.right_line,
            self.top_line,
            self.bottom_line,
        )
        pen = QPen(self.READY_COLOR, 1.0)
        pen.setCosmetic(True)
        for line in self._lines:
            line.setPen(pen)
            line.setZValue(100.0)
            line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            scene.addItem(line)

        radius = self.CIRCLE_RADIUS_SCREEN_PX
        self.center_circle = QGraphicsEllipseItem(
            -radius, -radius, 2.0 * radius, 2.0 * radius
        )
        self.center_circle.setPen(pen)
        self.center_circle.setZValue(101.0)
        self.center_circle.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.center_circle.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
        )
        scene.addItem(self.center_circle)
        self._color = QColor(self.READY_COLOR)
        self.hide()

    def is_visible(self) -> bool:
        """Показывается ли сейчас прицел над допустимым пикселем кадра."""

        return self.center_circle.isVisible()

    def color(self) -> QColor:
        """Вернуть текущий цвет состояния прицела."""

        return QColor(self._color)

    def set_color(self, color: QColor) -> None:
        """Переключить красное и жёлтое состояние всего прицела."""

        self._color = QColor(color)
        for item in (*self._lines, self.center_circle):
            pen = item.pen()
            pen.setColor(color)
            item.setPen(pen)

    def show_at(
        self,
        x: float,
        y: float,
        *,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Поставить центр в пиксель и оставить разрыв внутри окружности."""

        transform = self._view.transform()
        scale_x = max(abs(transform.m11()), 1e-9)
        scale_y = max(abs(transform.m22()), 1e-9)
        gap_x = self.CIRCLE_RADIUS_SCREEN_PX / scale_x
        gap_y = self.CIRCLE_RADIUS_SCREEN_PX / scale_y
        self.left_line.setLine(0.0, y, max(0.0, x - gap_x), y)
        self.right_line.setLine(
            min(float(frame_width), x + gap_x), y, float(frame_width), y
        )
        self.top_line.setLine(x, 0.0, x, max(0.0, y - gap_y))
        self.bottom_line.setLine(
            x, min(float(frame_height), y + gap_y), x, float(frame_height)
        )
        self.center_circle.setPos(x, y)
        for item in (*self._lines, self.center_circle):
            item.show()

    def hide(self) -> None:
        """Скрыть все части прицела как единый визуальный инструмент."""

        for item in (*self._lines, self.center_circle):
            item.hide()


class _TrackedRoiMarker:
    """Полупрозрачная область 41×41 px, закреплённая в геометрии кадра."""

    def __init__(self, scene: QGraphicsScene, *, radius_px: float) -> None:
        self.item = QGraphicsEllipseItem(
            -radius_px, -radius_px, 2.0 * radius_px, 2.0 * radius_px
        )
        pen = QPen(QColor("#ff0000"), 1.0)
        pen.setCosmetic(True)
        self.item.setPen(pen)
        self.item.setBrush(QColor(255, 0, 0, 80))
        self.item.setZValue(90.0)
        self.item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        scene.addItem(self.item)
        self.item.hide()

    def show_at(self, point_px: tuple[float, float]) -> None:
        """Перенести маркер в подтверждённый центр области."""

        self.item.setPos(*point_px)
        self.item.show()

    def hide(self) -> None:
        """Скрыть область после сброса или потери сопровождения."""

        self.item.hide()

    def is_visible(self) -> bool:
        """Видна ли подтверждённая область на текущем кадре."""

        return self.item.isVisible()


class _DetectedEventMarker:
    """Экранный крест, отмечающий подтверждённую точку события."""

    COLOR = QColor("#ff0000")
    RADIUS_SCREEN_PX = _StreamPointCrosshair.CIRCLE_RADIUS_SCREEN_PX
    PEN_WIDTH_SCREEN_PX = 2.0

    def __init__(self, scene: QGraphicsScene) -> None:
        radius = self.RADIUS_SCREEN_PX
        self.first_line = QGraphicsLineItem(-radius, -radius, radius, radius)
        self.second_line = QGraphicsLineItem(-radius, radius, radius, -radius)
        self._lines = (self.first_line, self.second_line)
        pen = QPen(self.COLOR, self.PEN_WIDTH_SCREEN_PX)
        pen.setCosmetic(True)
        for line in self._lines:
            line.setPen(pen)
            line.setZValue(110.0)
            line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            line.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
            )
            scene.addItem(line)
        self.hide()

    def show_at(self, point_px: tuple[float, float]) -> None:
        """Поставить центр диагонального креста в пиксель события."""

        for line in self._lines:
            line.setPos(*point_px)
            line.show()

    def hide(self) -> None:
        """Скрыть метку до следующего подтверждённого события."""

        for line in self._lines:
            line.hide()

    def is_visible(self) -> bool:
        """Показывается ли метка подтверждённого события."""

        return self.first_line.isVisible()
class _LocalizationWorker(QObject):
    """Выполняет DINOv2 и SIFT вне главного потока Qt."""

    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    preparation_completed = Signal(object)
    preparation_failed = Signal(str)

    def __init__(self, service: RepeatLocalizationService) -> None:
        super().__init__()
        self._service = service

    @Slot()
    def prepare_index(self) -> None:
        """Заранее построить визуальную Teach-базу в том же рабочем потоке."""

        try:
            result = self._service.prepare_index()
        except (OSError, RuntimeError, ValueError) as error:
            self.preparation_failed.emit(str(error))
            return
        self.preparation_completed.emit(result)

    @Slot(str, int)
    def localize(self, repeat_directory: str, frame_index: int) -> None:
        self.progress.emit(
            "Ищу Teach-кандидатов и проверяю геометрию…"
        )
        try:
            result = self._service.localize(repeat_directory, frame_index)
        except (OSError, RuntimeError, ValueError) as error:
            self.failed.emit(str(error))
            return
        self.completed.emit(result)


class RepeatMeasurementWidget(QWidget):
    """Вкладка измерения Repeat-кадра через сохранённую Teach-привязку."""

    localization_requested = Signal(str, int)
    index_preparation_requested = Signal()
    busy_changed = Signal(bool)

    def __init__(
        self,
        *,
        teach_flight_directory: str | Path,
        map_path: str | Path,
        session: AlignmentSession,
        repeat_start_directory: str | Path,
        map_scale_meters_per_pixel: float = HK_AIRFIELD_SCALE_METERS_PER_PIXEL,
        map_calibration: NorthUpMapCalibration = HK_AIRFIELD_MAP_CALIBRATION,
        reference_point: GeographicPoint = HK_REFERENCE_POINT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._repeat_start_directory = Path(repeat_start_directory)
        self._localization: RepeatLocalizationResult | None = None
        self._measurement: ScaledMapSegmentMeasurement | None = None
        self._reference_event_measurement: (
            TargetEventReferenceMeasurement | None
        ) = None
        self._last_measurement_is_event = False
        self._map_calibration = map_calibration
        self._reference_point = reference_point
        self._segment_points: list[tuple[float, float]] = []
        self._measurement_items: list[QGraphicsItem] = []
        self._overlay_item: QGraphicsPixmapItem | None = None
        self._event_detector = VisualEventDetector()
        self._pending_event_measurement = False
        self._handled_event_confirmation_frame: int | None = None
        self._busy = False
        self._thread_stopped = False
        self._index_preparation_result: TeachIndexPreparationResult | None = None
        self._index_preparation_error: str | None = None

        self.video_view = QGraphicsView(self)
        self.map_view = QGraphicsView(self)
        self._video_viewport = self.video_view.viewport()
        self._map_viewport = self.map_view.viewport()
        self.video = VideoPlaybackController(self.video_view, self)
        self.map_image = MapImageController(self.map_view, self)
        video_scene = self.video_view.scene()
        assert video_scene is not None
        self._roi_tracker = LocalRoiTracker()
        self._stream_crosshair = _StreamPointCrosshair(video_scene, self.video_view)
        self._event_marker = _DetectedEventMarker(video_scene)
        self._roi_marker = _TrackedRoiMarker(
            video_scene, radius_px=self._roi_tracker.config.roi_radius_px
        )
        self._stream_point_tool_active = False
        self._stream_selection_frame_bgr: np.ndarray | None = None
        self._stream_selection_frame_index: int | None = None
        self._video_viewport.setMouseTracking(True)
        self.video.frame_changed.connect(self._on_frame_changed)
        self.video.video_loaded.connect(self._on_video_loaded)
        self.video.playback_changed.connect(self._on_playback_changed)
        self.video.error_occurred.connect(self._show_error)
        self.map_image.error_occurred.connect(self._show_error)
        for view in (self.video_view, self.map_view):
            # Масштаб меняется относительно положения курсора: оператор не
            # теряет рассматриваемый ориентир при увеличении кадра или карты.
            view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
            view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        self._build_interface(map_scale_meters_per_pixel)
        for viewport in (self._video_viewport, self._map_viewport):
            viewport.installEventFilter(self)
        if not self.map_image.load_image(map_path):
            self.locate_button.setEnabled(False)

        service = RepeatLocalizationService(
            teach_flight_directory=teach_flight_directory,
            session=session,
        )
        self._worker_thread = QThread(self)
        self._worker = _LocalizationWorker(service)
        self._worker.moveToThread(self._worker_thread)
        self.index_preparation_requested.connect(self._worker.prepare_index)
        self.localization_requested.connect(self._worker.localize)
        self._worker.progress.connect(self.status_label.setText)
        self._worker.completed.connect(self._on_localization_completed)
        self._worker.failed.connect(self._on_localization_failed)
        self._worker.preparation_completed.connect(
            self._on_index_preparation_completed
        )
        self._worker.preparation_failed.connect(self._on_index_preparation_failed)
        self._worker_thread.start()
        self.index_preparation_requested.emit()

    @property
    def localization(self) -> RepeatLocalizationResult | None:
        """Возвращает последнюю подтверждённую привязку."""

        return self._localization

    @property
    def measurement(self) -> ScaledMapSegmentMeasurement | None:
        """Возвращает последний законченный замер."""

        return self._measurement

    @property
    def reference_event_measurement(
        self,
    ) -> TargetEventReferenceMeasurement | None:
        """Вернуть последний автоматический замер ROI и события от репера."""

        return self._reference_event_measurement

    def _build_interface(self, scale: float) -> None:
        root = QHBoxLayout(self)
        self.view_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.view_splitter.setChildrenCollapsible(False)
        self.view_splitter.setHandleWidth(8)
        self.view_splitter.setOpaqueResize(True)
        self.view_splitter.addWidget(self.video_view)
        self.view_splitter.addWidget(self.map_view)
        self.view_splitter.setSizes([750, 750])
        self.view_splitter.setToolTip(
            "Тяните разделитель, чтобы менять ширину пролёта и карты"
        )
        root.addWidget(self.view_splitter, 1)

        panel = QWidget()
        panel.setMinimumWidth(350)
        panel.setMaximumWidth(440)
        layout = QVBoxLayout(panel)
        layout.addWidget(self._repeat_group())
        layout.addWidget(self._localization_group())
        layout.addWidget(self._measurement_group(scale))
        layout.addStretch(1)
        root.addWidget(panel)

    def _repeat_group(self) -> QGroupBox:
        group = QGroupBox("Repeat-пролёт")
        layout = QVBoxLayout(group)
        self.open_repeat_button = QPushButton("Открыть независимый пролёт…")
        self.open_repeat_button.clicked.connect(self._choose_repeat_flight)
        layout.addWidget(self.open_repeat_button)

        playback_row = QHBoxLayout()
        self.play_button = QPushButton("▶ Запустить поток")
        self.play_button.setToolTip(
            "Последовательно показывать кадры Repeat-пролёта с записанной частотой"
        )
        self.freeze_button = QPushButton("⏸ Зафиксировать кадр")
        self.freeze_button.setToolTip(
            "Остановить показ на текущем кадре, чтобы выполнить привязку и замер"
        )
        self.play_button.clicked.connect(self.video.play)
        self.freeze_button.clicked.connect(self.video.pause)
        for widget in (self.play_button, self.freeze_button):
            widget.setEnabled(False)
            playback_row.addWidget(widget)
        layout.addLayout(playback_row)

        self.stream_state_label = QLabel("Поток не открыт")
        self.stream_state_label.setWordWrap(True)
        layout.addWidget(self.stream_state_label)

        row = QHBoxLayout()
        self.previous_button = QPushButton("◀")
        self.next_button = QPushButton("▶")
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, 1)
        self.frame_spin.setPrefix("Кадр ")
        self.go_button = QPushButton("Перейти")
        self.previous_button.clicked.connect(self.video.step_backward)
        self.next_button.clicked.connect(self.video.step_forward)
        self.go_button.clicked.connect(self._seek_selected_frame)
        self.frame_spin.editingFinished.connect(self._seek_selected_frame)
        for widget in (
            self.previous_button,
            self.next_button,
            self.frame_spin,
            self.go_button,
        ):
            widget.setEnabled(False)
            row.addWidget(widget)
        row.setStretch(2, 1)
        layout.addLayout(row)
        self.frame_label = QLabel("Пролёт не открыт")
        self.frame_label.setWordWrap(True)
        layout.addWidget(self.frame_label)
        return group

    def _localization_group(self) -> QGroupBox:
        group = QGroupBox("Автоматическая привязка")
        layout = QVBoxLayout(group)
        self.locate_button = QPushButton("Привязать этот стоп-кадр")
        self.locate_button.setEnabled(False)
        self.locate_button.clicked.connect(self._start_localization)
        layout.addWidget(self.locate_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Откройте Repeat-пролёт и выберите стоп-кадр.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.quality_label = QLabel("Геометрия ещё не проверена")
        self.quality_label.setWordWrap(True)
        layout.addWidget(self.quality_label)
        return group

    def _measurement_group(self, scale: float) -> QGroupBox:
        group = QGroupBox("Отрезок на поверхности земли")
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(8)
        self.scale_spin.setRange(0.00000001, 1000.0)
        self.scale_spin.setValue(scale)
        self.scale_spin.setSuffix(" м/px")
        self.scale_spin.setToolTip("Ручная калибровка: 226,50 м / 891,842054 px")
        self.scale_spin.valueChanged.connect(self._recompute_measurement)
        form.addRow("Масштаб карты:", self.scale_spin)
        layout.addLayout(form)

        row = QHBoxLayout()
        self.measure_button = QPushButton("Поставить 2 точки")
        self.measure_button.setCheckable(True)
        self.measure_button.setEnabled(False)
        self.measure_button.toggled.connect(self._set_measurement_active)
        self.clear_button = QPushButton("Очистить")
        self.clear_button.clicked.connect(self._clear_measurement)
        row.addWidget(self.measure_button)
        row.addWidget(self.clear_button)
        layout.addLayout(row)
        self.result_label = QLabel(
            "После успешной привязки поставьте две точки на левом кадре."
        )
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.result_label)
        return group

    def _choose_repeat_flight(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Выберите независимый Repeat-пролёт",
            str(self._repeat_start_directory),
        )
        if directory and self.video.load_flight(directory):
            self._repeat_start_directory = Path(directory).parent

    def _seek_selected_frame(self) -> None:
        self.video.seek(self.frame_spin.value() - 1)

    def _on_video_loaded(self, path: str, frame_count: int, _fps: float) -> None:
        self._release_stream_selection_freeze()
        self._clear_stream_roi_tracking()
        self._reset_visual_event_detection()
        self.frame_spin.setRange(1, max(1, frame_count))
        self._update_playback_controls()
        self._update_stream_state()
        self.status_label.setText(
            f"Открыт {Path(path).name}. Запустите поток или привяжите "
            "текущий стоп-кадр."
        )

    def _on_frame_changed(self, frame_index: int, frame_count: int) -> None:
        blocker = QSignalBlocker(self.frame_spin)
        self.frame_spin.setRange(1, max(1, frame_count))
        self.frame_spin.setValue(frame_index + 1)
        del blocker
        self._update_playback_controls()
        telemetry = self.video.current_frame_telemetry
        text = f"Кадр {frame_index + 1} из {frame_count}"
        if telemetry is not None:
            text += f" | Image Id {telemetry.image_id}"
            if telemetry.has_position:
                text += (
                    f" | {telemetry.latitude_degrees:.7f}, "
                    f"{telemetry.longitude_degrees:.7f}"
                )
        self.frame_label.setText(text)
        self._invalidate_localization()
        tracking_error = self._advance_stream_roi_tracking(frame_index)
        event_handled = (
            self._advance_visual_event_detection(frame_index)
            if self.video.is_playing
            else False
        )
        if tracking_error is not None and not event_handled:
            self.status_label.setText(tracking_error)
        elif not self.video.is_playing and not event_handled:
            self.status_label.setText(
                "Стоп-кадр зафиксирован. Можно запускать автоматическую привязку."
            )

    def _on_playback_changed(self, is_playing: bool) -> None:
        """Переключить окно между наблюдением потока и стоп-кадром.

        Во время движения нельзя начинать геометрическую привязку: к моменту
        завершения тяжёлого расчёта на экране уже находился бы другой кадр.
        Поэтому запуск потока сразу удаляет старую привязку и результат замера,
        а рабочие кнопки возвращаются только после явной фиксации кадра.
        """

        if is_playing:
            self._start_visual_event_detection()
            self._invalidate_localization()
            self.status_label.setText(
                "Поток идёт. ПКМ временно удерживает кадр для точного "
                "выбора области интереса."
            )
        else:
            self._release_stream_selection_freeze()
            self.status_label.setText(
                "Стоп-кадр зафиксирован. Можно запускать автоматическую привязку."
            )
        self._set_stream_point_tool_active(is_playing)
        self._update_playback_controls()
        self._update_stream_state()

    def _reset_visual_event_detection(self) -> None:
        """Сбросить историю события перед новым источником или запуском."""

        self._event_detector.reset()
        self._event_marker.hide()
        self._pending_event_measurement = False
        self._handled_event_confirmation_frame = None

    def _start_visual_event_detection(self) -> None:
        """Начать анализ с уже показанного кадра без ожидания лишнего шага."""

        self._reset_visual_event_detection()
        frame = self.video.current_frame_bgr
        frame_index = self.video.current_frame_index
        if frame is not None and frame_index is not None:
            self._event_detector.start(frame, frame_index=frame_index)

    def _advance_visual_event_detection(self, frame_index: int) -> bool:
        """Обработать кадр и остановить поток при первом подтверждении."""

        frame = self.video.current_frame_bgr
        if frame is None:
            return False
        try:
            step = self._event_detector.advance(frame, frame_index=frame_index)
        except (ValueError, cv2.error) as error:
            self._event_detector.reset()
            self.status_label.setText(
                f"Детектор события сброшен после ошибки: {error}"
            )
            return False
        event = step.confirmed_event
        if event is None:
            return False
        if self._handled_event_confirmation_frame == event.confirmation_frame_index:
            return False
        self._handled_event_confirmation_frame = event.confirmation_frame_index
        self._handle_confirmed_visual_event(event)
        return True

    def _handle_confirmed_visual_event(
        self, event: ConfirmedVisualEvent
    ) -> None:
        """Зафиксировать событие и начать замер от сопровождаемой области."""

        self._release_stream_selection_freeze()
        self._event_marker.show_at(event.point_px)
        self.video.pause()
        roi_point = self._roi_tracker.point_px
        if roi_point is None:
            self._pending_event_measurement = False
            self.status_label.setText(
                f"Событие зафиксировано на кадре {event.onset_frame_index + 1}, "
                "но область интереса не захвачена или потеряна."
            )
            self.result_label.setText(
                "Расстояние не вычислено: нет сопровождаемой исходной точки."
            )
            return

        self._clear_measurement()
        self._segment_points.extend((roi_point, event.point_px))
        scene = self.video_view.scene()
        if scene is not None:
            line = scene.addLine(
                roi_point[0],
                roi_point[1],
                event.point_px[0],
                event.point_px[1],
                QPen(QColor("#ffcc00"), 3.0),
            )
            line.setZValue(30.0)
            self._measurement_items.append(line)

        self._pending_event_measurement = True
        self.status_label.setText(
            f"Событие началось на кадре {event.onset_frame_index + 1} и "
            f"подтверждено на кадре {event.confirmation_frame_index + 1}. "
            "Поток остановлен; выполняется привязка для замера."
        )
        self.result_label.setText(
            "Точки POI и события зафиксированы. Ожидается геометрическая привязка."
        )
        if not self.video.is_flight or not self.map_image.is_loaded:
            self._pending_event_measurement = False
            self.status_label.setText(
                "Событие зафиксировано, но автоматическому замеру нужны "
                "Repeat-каталог и загруженная карта."
            )
            return
        self._start_localization()
    def _update_stream_state(self) -> None:
        """Явно показать оператору, движется изображение или зафиксировано."""

        if not self.video.is_loaded:
            text = "Поток не открыт"
        elif self._stream_selection_frame_index is not None:
            text = "◎ Кадр удерживается для выбора — поток продолжает идти"
        elif self.video.is_playing:
            text = "● Поток воспроизводится — измерение заблокировано"
        else:
            text = "❚❚ Стоп-кадр зафиксирован — измерение разрешено после привязки"
        self.stream_state_label.setText(text)

    def _update_playback_controls(self) -> None:
        """Согласовать все кнопки с единым состоянием потока.

        Ручной переход, привязка и измерение допустимы только на неподвижном
        кадре. Это не даёт незаметно смешать пиксели одного кадра с результатом
        локализации другого.
        """

        loaded = self.video.is_loaded
        playing = self.video.is_playing
        stopped_and_ready = loaded and not playing and not self._busy
        current = self.video.current_frame_index

        self.play_button.setEnabled(stopped_and_ready)
        self.freeze_button.setEnabled(loaded and playing and not self._busy)
        self.frame_spin.setEnabled(stopped_and_ready)
        self.go_button.setEnabled(stopped_and_ready)
        self.previous_button.setEnabled(
            stopped_and_ready and current is not None and current > 0
        )
        self.next_button.setEnabled(
            stopped_and_ready
            and current is not None
            and current < self.video.frame_count - 1
        )
        self.locate_button.setEnabled(
            stopped_and_ready and self.video.is_flight and self.map_image.is_loaded
        )

    @Slot(object)
    def _on_index_preparation_completed(self, value: object) -> None:
        """Запомнить, что тяжёлая неизменяемая часть поиска уже готова."""

        if not isinstance(value, TeachIndexPreparationResult):
            self._on_index_preparation_failed(
                "Рабочий поток вернул неизвестный результат подготовки индекса"
            )
            return
        self._index_preparation_result = value
        self._index_preparation_error = None
        if not self.video.is_loaded and not self._busy:
            source = (
                "загружен с диска"
                if value.loaded_from_disk_cache
                else "построен и сохранён"
            )
            self.status_label.setText(
                f"Визуальный Teach-индекс {source}: "
                f"{value.candidate_count} кадров за "
                f"{value.processing_time_seconds:.1f} с."
            )

    @Slot(str)
    def _on_index_preparation_failed(self, message: str) -> None:
        """Не скрывать сбой прогрева; локализация позднее повторит попытку."""

        self._index_preparation_result = None
        self._index_preparation_error = message
        if not self.video.is_loaded and not self._busy:
            self.status_label.setText(
                "Фоновая подготовка визуального Teach-индекса не выполнена: "
                + message
            )

    def _start_localization(self) -> None:
        source = self.video.source_path
        frame_index = self.video.current_frame_index
        if self.video.is_playing:
            self._show_error("Сначала зафиксируйте интересующий кадр")
            return
        if source is None or frame_index is None or not self.video.is_flight:
            self._show_error("Для автоматической привязки нужен каталог пролёта")
            return
        self._set_busy(True)
        self.localization_requested.emit(str(source), frame_index)

    @Slot(object)
    def _on_localization_completed(self, value: object) -> None:
        self._set_busy(False)
        if not isinstance(value, RepeatLocalizationResult):
            self._on_localization_failed("Рабочий поток вернул неизвестный результат")
            return
        if value.repeat_frame_index != self.video.current_frame_index:
            self._on_localization_failed(
                "Во время расчёта изменился кадр; повторите привязку"
            )
            return
        self._localization = value
        self._show_overlay(value.homography_repeat_to_map)
        self.measure_button.setEnabled(True)
        self.status_label.setText(
            f"Repeat Image Id {value.repeat_image_id} → Teach Image Id "
            f"{value.teach_image_id} (кадр {value.teach_frame_index + 1}), "
            f"ранг {value.retrieval_rank}; {value.processing_time_seconds:.1f} с."
        )
        self.quality_label.setText(
            f"Принято: inlier {value.inlier_count}, доля "
            f"{value.inlier_ratio:.3f}, покрытие {value.frame_coverage_fraction:.3f}, "
            f"p95 остатка {value.reprojection_p95_px:.2f} px, "
            f"p95 нестабильности {value.stability_p95_corner_shift_px:.2f} px."
        )
        self.result_label.setText("Привязка готова. Нажмите «Поставить 2 точки».")
        if self._pending_event_measurement:
            self._pending_event_measurement = False
            self._calculate_measurement(event_measurement=True)

    @Slot(str)
    def _on_localization_failed(self, message: str) -> None:
        self._set_busy(False)
        self._pending_event_measurement = False
        self._invalidate_localization()
        self.status_label.setText(message)
        self.quality_label.setText("Измерение запрещено: привязка не подтверждена")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.progress_bar.setRange(0, 0 if busy else 1)
        if not busy:
            self.progress_bar.setValue(0)
        self.open_repeat_button.setEnabled(not busy)
        self._update_playback_controls()
        self.busy_changed.emit(busy)

    @property
    def is_busy(self) -> bool:
        """Выполняется ли сейчас фоновая геометрическая привязка."""

        return self._busy

    def _invalidate_localization(self) -> None:
        self._localization = None
        self.measure_button.setChecked(False)
        self.measure_button.setEnabled(False)
        self.quality_label.setText("Геометрия ещё не проверена")
        self._clear_overlay()
        self._clear_measurement()

    def _set_measurement_active(self, active: bool) -> None:
        if active and self._localization is None:
            self.measure_button.setChecked(False)
            return
        if active:
            self._clear_measurement()
        self.video_view.setDragMode(
            QGraphicsView.DragMode.NoDrag
            if active
            else QGraphicsView.DragMode.ScrollHandDrag
        )
        self._restore_video_cursor()
        if active:
            self.result_label.setText("Поставьте первую точку на левом кадре.")

    def _set_stream_point_tool_active(self, active: bool) -> None:
        """Включить визуальный выбор точки только на движущемся потоке."""

        self._stream_point_tool_active = active
        color = (
            _StreamPointCrosshair.CAPTURED_COLOR
            if self._roi_tracker.is_active
            else _StreamPointCrosshair.READY_COLOR
        )
        self._stream_crosshair.set_color(color)
        if active:
            position = self._video_viewport.mapFromGlobal(QCursor.pos())
            self._update_stream_crosshair(position)
        else:
            self._stream_crosshair.hide()
            self._restore_video_cursor()

    def _update_stream_crosshair(self, viewport_position: QPoint) -> None:
        """Перенести прицел из координат viewport в пиксели кадра."""

        frame = self.video.current_frame_bgr
        if not self._stream_point_tool_active or frame is None:
            self._stream_crosshair.hide()
            self._restore_video_cursor()
            return
        point = self.video_view.mapToScene(viewport_position)
        x, y = float(point.x()), float(point.y())
        height, width = frame.shape[:2]
        if not 0.0 <= x <= width - 1.0 or not 0.0 <= y <= height - 1.0:
            self._stream_crosshair.hide()
            self._video_viewport.setCursor(Qt.CursorShape.ArrowCursor)
            return
        self._stream_crosshair.show_at(x, y, frame_width=width, frame_height=height)
        self._video_viewport.setCursor(Qt.CursorShape.BlankCursor)

    def _begin_stream_selection_freeze(self) -> None:
        """Удержать видимый кадр, оставив источник и детектор работающими."""

        if not self.video.is_playing:
            return
        frame = self.video.current_frame_bgr
        frame_index = self.video.current_frame_index
        if frame is None or frame_index is None or not self.video.freeze_display():
            return
        self._stream_selection_frame_bgr = frame
        self._stream_selection_frame_index = frame_index
        self._roi_marker.hide()
        self._update_stream_state()
        self.status_label.setText(
            f"Кадр {frame_index + 1} удерживается только на экране. "
            "Поток и детектор продолжают идти; ЛКМ — выбрать точку, "
            "ПКМ или Esc — отменить."
        )

    def _release_stream_selection_freeze(self) -> None:
        """Вернуть на экран живой край и удалить временный снимок."""

        self._stream_selection_frame_bgr = None
        self._stream_selection_frame_index = None
        self.video.resume_display()
        tracked_point = self._roi_tracker.point_px
        if tracked_point is not None:
            self._roi_marker.show_at(tracked_point)
        self._update_stream_state()
        if self._stream_point_tool_active:
            position = self._video_viewport.mapFromGlobal(QCursor.pos())
            self._update_stream_crosshair(position)

    def _catch_up_roi_to_live_frame(self) -> tuple[float, float]:
        """Быстро провести новый захват через пропущенные соседние кадры."""

        frozen_index = self._stream_selection_frame_index
        live_index = self.video.current_frame_index
        if frozen_index is None or live_index is None or live_index < frozen_index:
            raise LocalRoiTrackingFailure(
                "Не определён диапазон возврата к живому кадру"
            )
        point = self._roi_tracker.point_px
        if point is None:
            raise LocalRoiTrackingFailure("Область интереса не захвачена")
        try:
            for frame_index, frame in self.video.held_frames_after(frozen_index):
                step = self._roi_tracker.advance(
                    frame,
                    frame_index=frame_index,
                )
                point = step.point_px
        except RuntimeError as error:
            self._roi_tracker.clear()
            raise LocalRoiTrackingFailure(str(error)) from error
        return point


    def _capture_stream_roi(self, viewport_position: QPoint) -> None:
        """Захватить область и при необходимости догнать живой кадр."""

        frozen_selection = self._stream_selection_frame_index is not None
        frame = (
            self._stream_selection_frame_bgr
            if frozen_selection
            else self.video.current_frame_bgr
        )
        frame_index = (
            self._stream_selection_frame_index
            if frozen_selection
            else self.video.current_frame_index
        )
        if frame is None or frame_index is None:
            return
        point = self.video_view.mapToScene(viewport_position)
        point_px = (float(point.x()), float(point.y()))
        try:
            self._roi_tracker.capture(frame, frame_index=frame_index, point_px=point_px)
        except (LocalRoiTrackingFailure, ValueError) as error:
            self.status_label.setText(f"Захват не выполнен: {error}")
            return

        marker_point = point_px
        if frozen_selection:
            try:
                marker_point = self._catch_up_roi_to_live_frame()
            except LocalRoiTrackingFailure as error:
                self._clear_stream_roi_tracking()
                self._release_stream_selection_freeze()
                self.status_label.setText(
                    f"{error}. Поток продолжается без захваченной области."
                )
                return
            frozen_frame_number = frame_index + 1
            live_frame_number = (self.video.current_frame_index or 0) + 1
            self._release_stream_selection_freeze()
            status = (
                f"Точка выбрана на кадре {frozen_frame_number}; "
                f"сопровождение догнало живой кадр {live_frame_number}."
            )
        else:
            size = self._roi_tracker.config.roi_size_px
            status = (
                f"Область {size}×{size} px захвачена на кадре "
                f"{frame_index + 1}. Повторный клик уточнит точку; "
                "поток продолжается."
            )

        self._roi_marker.show_at(marker_point)
        self._stream_crosshair.set_color(_StreamPointCrosshair.CAPTURED_COLOR)
        self.status_label.setText(status)

    def _advance_stream_roi_tracking(self, frame_index: int) -> str | None:
        """Перенести маркер или вернуть явное сообщение о потере."""

        if not self._roi_tracker.is_active:
            return None
        previous_index = self._roi_tracker.frame_index
        if previous_index == frame_index:
            return None
        if (
            not self.video.is_playing
            or previous_index is None
            or frame_index != previous_index + 1
        ):
            self._clear_stream_roi_tracking()
            return None
        frame = self.video.current_frame_bgr
        if frame is None:
            self._clear_stream_roi_tracking()
            return (
                "Область интереса потеряна: текущий кадр не читается. "
                "Поток продолжается."
            )
        try:
            step = self._roi_tracker.advance(frame, frame_index=frame_index)
        except LocalRoiTrackingFailure as error:
            self._clear_stream_roi_tracking()
            return f"{error}. Поток продолжается."
        if self._stream_selection_frame_index is None:
            self._roi_marker.show_at(step.point_px)
        self._stream_crosshair.set_color(_StreamPointCrosshair.CAPTURED_COLOR)
        return None

    def _clear_stream_roi_tracking(self) -> None:
        """Сбросить захват, скрыть область и вернуть красный прицел."""

        self._roi_tracker.clear()
        self._roi_marker.hide()
        self._stream_crosshair.set_color(_StreamPointCrosshair.READY_COLOR)

    def _restore_video_cursor(self) -> None:
        """Вернуть курсор обычного просмотра или ручного замера отрезка."""

        cursor = (
            Qt.CursorShape.CrossCursor
            if self.measure_button.isChecked()
            else Qt.CursorShape.ArrowCursor
        )
        self._video_viewport.setCursor(cursor)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Масштабировать обе сцены колесом и принимать точки на кадре.

        Преобразование существует только на уровне отображения
        ``QGraphicsView``. Исходные пиксели кадра и карты не пересчитываются,
        поэтому масштабирование интерфейса не меняет геометрию измерения.
        """

        view = self._view_for_viewport(watched)
        if view is None:
            return super().eventFilter(watched, event)

        if (
            view is self.video_view
            and self._stream_selection_frame_index is not None
            and event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Escape
        ):
            self._release_stream_selection_freeze()
            self.status_label.setText(
                "Выбор точки отменён; показан текущий кадр потока."
            )
            return True

        if (
            view is self.video_view
            and self._stream_point_tool_active
            and event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.RightButton
        ):
            if self._stream_selection_frame_index is None:
                self._begin_stream_selection_freeze()
            else:
                self._release_stream_selection_freeze()
                self.status_label.setText(
                    "Выбор точки отменён; показан текущий кадр потока."
                )
            return True

        if view is self.video_view and self._stream_point_tool_active:
            if event.type() == QEvent.Type.MouseMove:
                self._update_stream_crosshair(
                    event.position().toPoint()  # type: ignore[attr-defined]
                )
            elif event.type() == QEvent.Type.Enter:
                self._update_stream_crosshair(
                    event.position().toPoint()  # type: ignore[attr-defined]
                )
            elif event.type() == QEvent.Type.Leave:
                self._stream_crosshair.hide()
                self._video_viewport.setCursor(Qt.CursorShape.ArrowCursor)

        if event.type() == QEvent.Type.Wheel:
            delta = event.angleDelta().y()  # type: ignore[attr-defined]
            if delta == 0:
                return True
            factor = 1.2 if delta > 0 else 1.0 / 1.2
            current_scale = abs(view.transform().m11())
            next_scale = current_scale * factor
            if 0.02 <= next_scale <= 50.0:
                view.scale(factor, factor)
                if view is self.video_view and self._stream_point_tool_active:
                    self._update_stream_crosshair(
                        event.position().toPoint()  # type: ignore[attr-defined]
                    )
            return True

        if (
            view is self.video_view
            and self._stream_point_tool_active
            and event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._capture_stream_roi(
                event.position().toPoint()  # type: ignore[attr-defined]
            )
            return True

        if (
            view is self.video_view
            and self.measure_button.isChecked()
            and event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
        ):
            point = self.video_view.mapToScene(
                event.position().toPoint()  # type: ignore[attr-defined]
            )
            self._accept_point((float(point.x()), float(point.y())))
            return True
        return super().eventFilter(watched, event)

    def _view_for_viewport(self, watched: QObject) -> QGraphicsView | None:
        """Вернуть область просмотра, которой принадлежит событие мыши."""

        if watched is self._video_viewport:
            return self.video_view
        if watched is self._map_viewport:
            return self.map_view
        return None

    def _accept_point(self, point: tuple[float, float]) -> None:
        frame = self.video.current_frame_bgr
        if frame is None:
            return
        height, width = frame.shape[:2]
        x, y = point
        if not 0.0 <= x <= width - 1.0 or not 0.0 <= y <= height - 1.0:
            self.result_label.setText("Клик находится за пределами кадра.")
            return
        self._segment_points.append(point)
        self._add_marker(self.video_view, point, QColor("#ffcc00"))
        if len(self._segment_points) == 1:
            self.result_label.setText("Поставьте вторую точку.")
            return
        scene = self.video_view.scene()
        if scene is not None:
            line = scene.addLine(
                self._segment_points[0][0],
                self._segment_points[0][1],
                self._segment_points[1][0],
                self._segment_points[1][1],
                QPen(QColor("#ffcc00"), 3.0),
            )
            line.setZValue(30.0)
            self._measurement_items.append(line)
        self._calculate_measurement()
        self.measure_button.setChecked(False)

    def _calculate_measurement(self, *, event_measurement: bool = False) -> None:
        """Перенести две точки на карту и выбрать смысл результата.

        Для ручного инструмента две точки остаются концами обычного отрезка.
        Автоматическая пара имеет иной смысл: первая точка является ROI,
        вторая — событием, а обе измеряются от удалённого репера.
        """

        if self._localization is None or len(self._segment_points) != 2:
            return
        map_bgr = self.map_image.current_image_bgr
        if map_bgr is None:
            return
        try:
            measurement = measure_scaled_map_segment(
                self._segment_points,
                self._localization.homography_repeat_to_map,
                map_width_pixels=map_bgr.shape[1],
                map_height_pixels=map_bgr.shape[0],
                map_scale_meters_per_pixel=self.scale_spin.value(),
            )
            reference_event_measurement = (
                measure_target_and_event_from_reference(
                    measurement.map_points_px,
                    calibration=self._map_calibration,
                    reference=self._reference_point,
                    map_width_pixels=map_bgr.shape[1],
                    map_height_pixels=map_bgr.shape[0],
                    map_scale_meters_per_pixel=self.scale_spin.value(),
                )
                if event_measurement
                else None
            )
        except (RepeatLocalizationFailure, ReferenceEventMeasurementFailure) as error:
            self._measurement = None
            self._reference_event_measurement = None
            self._last_measurement_is_event = False
            self.result_label.setText(str(error))
            return
        self._measurement = measurement
        self._reference_event_measurement = reference_event_measurement
        self._last_measurement_is_event = event_measurement
        self._remove_map_measurement_items()
        label_text = (
            f"Δ {reference_event_measurement.event_azimuth_deviation_degrees:+.3f}°"
            if reference_event_measurement is not None
            else None
        )
        self._draw_map_measurement(measurement, label_text=label_text)
        if reference_event_measurement is not None:
            self.result_label.setText(
                _format_reference_event_measurement(reference_event_measurement)
            )
            return
        points = measurement.map_points_px
        self.result_label.setText(
            f"Длина: {measurement.length_meters:.3f} м\n"
            f"Карта: ({points[0, 0]:.2f}, {points[0, 1]:.2f}) → "
            f"({points[1, 0]:.2f}, {points[1, 1]:.2f}) px"
        )

    def _recompute_measurement(self) -> None:
        if len(self._segment_points) == 2:
            self._calculate_measurement(
                event_measurement=self._last_measurement_is_event
            )

    def _draw_map_measurement(
        self,
        measurement: ScaledMapSegmentMeasurement,
        *,
        label_text: str | None = None,
    ) -> None:
        scene = self.map_view.scene()
        if scene is None:
            return
        first, second = measurement.map_points_px
        for point in (first, second):
            self._add_marker(
                self.map_view,
                (float(point[0]), float(point[1])),
                QColor("#00e5ff"),
            )
        line = scene.addLine(
            float(first[0]),
            float(first[1]),
            float(second[0]),
            float(second[1]),
            QPen(QColor("#00e5ff"), 3.0),
        )
        line.setZValue(30.0)
        self._measurement_items.append(line)
        middle = (first + second) / 2.0
        label = QGraphicsSimpleTextItem(
            label_text or f"{measurement.length_meters:.3f} м"
        )
        label.setBrush(QColor("#00e5ff"))
        label.setPos(float(middle[0]) + 6.0, float(middle[1]) + 6.0)
        label.setZValue(31.0)
        label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        scene.addItem(label)
        self._measurement_items.append(label)

    def _add_marker(
        self,
        view: QGraphicsView,
        point: tuple[float, float],
        color: QColor,
    ) -> None:
        scene = view.scene()
        if scene is None:
            return
        marker = QGraphicsEllipseItem(-5.0, -5.0, 10.0, 10.0)
        marker.setPos(*point)
        marker.setPen(QPen(QColor("#101010"), 1.5))
        marker.setBrush(color)
        marker.setZValue(31.0)
        marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        scene.addItem(marker)
        self._measurement_items.append(marker)

    def _clear_measurement(self) -> None:
        for item in self._measurement_items:
            scene = item.scene()
            if scene is not None:
                scene.removeItem(item)
        self._measurement_items.clear()
        self._segment_points.clear()
        self._pending_event_measurement = False
        self._measurement = None
        self._reference_event_measurement = None
        self._last_measurement_is_event = False
        if self._localization is not None and not self.measure_button.isChecked():
            self.result_label.setText("Нажмите «Поставить 2 точки».")

    def _remove_map_measurement_items(self) -> None:
        map_scene = self.map_view.scene()
        retained: list[QGraphicsItem] = []
        for item in self._measurement_items:
            if map_scene is not None and item.scene() is map_scene:
                map_scene.removeItem(item)
            else:
                retained.append(item)
        self._measurement_items = retained

    def _show_overlay(self, homography: np.ndarray) -> None:
        self._clear_overlay()
        frame = self.video.current_frame_bgr
        map_bgr = self.map_image.current_image_bgr
        if frame is None or map_bgr is None:
            return
        height, width = map_bgr.shape[:2]
        warped = cv2.warpPerspective(frame, homography, (width, height))
        mask = cv2.warpPerspective(
            np.full(frame.shape[:2], 255, dtype=np.uint8),
            homography,
            (width, height),
            flags=cv2.INTER_NEAREST,
        )
        rgba = cv2.cvtColor(warped, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = np.where(mask > 0, 128, 0).astype(np.uint8)
        image = QImage(
            rgba.data,
            width,
            height,
            width * 4,
            QImage.Format.Format_RGBA8888,
        ).copy()
        self._overlay_item = QGraphicsPixmapItem(QPixmap.fromImage(image))
        self._overlay_item.setZValue(10.0)
        scene = self.map_view.scene()
        if scene is not None:
            scene.addItem(self._overlay_item)

    def _clear_overlay(self) -> None:
        if self._overlay_item is not None and self._overlay_item.scene() is not None:
            self._overlay_item.scene().removeItem(self._overlay_item)
        self._overlay_item = None

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Режим замеров", message)

    def shutdown(self) -> None:
        """Освободить источники и штатно остановить рабочий поток вкладки."""

        if self._thread_stopped:
            return
        self._thread_stopped = True
        for viewport in (self._video_viewport, self._map_viewport):
            viewport.removeEventFilter(self)
        self.video.close()
        self.map_image.clear()
        self._worker_thread.quit()
        self._worker_thread.wait()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._busy:
            self._show_busy_message()
            event.ignore()
            return
        self.shutdown()
        super().closeEvent(event)

    def _show_busy_message(self) -> None:
        QMessageBox.information(
            self,
            "Привязка выполняется",
            "Дождитесь завершения текущей геометрической проверки.",
        )
