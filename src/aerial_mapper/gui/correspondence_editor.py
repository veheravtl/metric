"""Интерактивная разметка соответствий между кадром и картой.

Этот модуль отвечает только за ручной эксперимент: оператор указывает одну и
ту же наземную точку сначала на кадре, затем на карте. Математическая оценка
вынесена в :mod:`aerial_mapper.manual_alignment`, поэтому её можно проверять
без графического интерфейса.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from aerial_mapper.alignment_session import (
    AlignmentSession,
    AnnotationStatus,
    FrameAnnotation,
    FrameObservation,
    MachineObservation,
)
from aerial_mapper.beacon_tracking import (
    BeaconTrackingFailure,
    HybridTrackingResult,
    track_beacons_with_machine_support_to_map,
)
from aerial_mapper.manual_alignment import (
    ManualAlignmentFailure,
    ManualAlignmentQuality,
    ManualAlignmentResult,
    assess_manual_alignment_quality,
    estimate_manual_alignment,
)

from .map_image import MapImageController
from .video_playback import VideoPlaybackController

Point = tuple[float, float]

_POINT_COLORS = (
    QColor("#ff5252"),
    QColor("#40c4ff"),
    QColor("#69f0ae"),
    QColor("#ffd740"),
    QColor("#e040fb"),
    QColor("#ff6e40"),
    QColor("#7c4dff"),
    QColor("#64ffda"),
)


class _FrameMarker(QGraphicsEllipseItem):
    """Перетаскиваемый маркер, сообщающий редактору новую координату.

    Qt посылает ``ItemPositionHasChanged`` только при включённом флаге
    ``ItemSendsGeometryChanges``. Обратный вызов назначается после начального
    ``setPos``, поэтому создание и восстановление маркера не выглядит как
    операторская правка.
    """

    def __init__(self) -> None:
        super().__init__(-5.0, -5.0, 10.0, 10.0)
        self._move_callback: Callable[[Point], None] | None = None
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
            True,
        )

    def set_move_callback(self, callback: Callable[[Point], None]) -> None:
        """Назначить обработчик последующих ручных и программных перемещений."""

        self._move_callback = callback

    def itemChange(  # noqa: N802
        self,
        change: QGraphicsItem.GraphicsItemChange,
        value: object,
    ) -> object:
        result = super().itemChange(change, value)
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
            and self._move_callback is not None
            and isinstance(value, QPointF)
        ):
            self._move_callback((float(value.x()), float(value.y())))
        return result


@dataclass
class _Correspondence:
    """Одна завершённая пара и её маркеры в двух сценах."""

    landmark_id: int
    frame_point: Point
    map_point: Point
    frame_marker: _FrameMarker
    map_marker: QGraphicsEllipseItem


class CorrespondenceEditor(QObject):
    """Собирать ручные пары точек и показывать результат поверх карты.

    Координаты всех точек хранятся в исходных пикселях изображений, а не в
    координатах окна. Поэтому увеличение, уменьшение и размер ``QGraphicsView``
    не меняют математический результат.
    """

    message = Signal(str)
    error_occurred = Signal(str)
    state_changed = Signal(int, bool, bool)
    alignment_computed = Signal(object)
    tracking_changed = Signal(bool)
    debug_event = Signal(str, object, bool)

    def __init__(
        self,
        video_view: QGraphicsView,
        map_view: QGraphicsView,
        video: VideoPlaybackController,
        map_image: MapImageController,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._video_view = video_view
        self._map_view = map_view
        self._video_viewport = video_view.viewport()
        self._map_viewport = map_view.viewport()
        self._disposed = False
        self._video = video
        self._map_image = map_image

        self._active = False
        self._session = AlignmentSession()
        self._correspondences: list[_Correspondence] = []
        self._suspect_landmark_ids: set[int] = set()
        self._pending_frame_point: Point | None = None
        self._pending_frame_marker: QGraphicsEllipseItem | None = None
        self._bound_frame_index: int | None = None
        self._overlay_item: QGraphicsPixmapItem | None = None
        self._result: ManualAlignmentResult | None = None
        self._current_annotation_status: AnnotationStatus = "draft"
        self._correction_pending = False
        self._resume_tracking_after_confirmation = False
        self._suppress_marker_callbacks = False
        self._video_source_path: str | None = None
        self._tracking_active = False
        self._tracking_frame_bgr: np.ndarray | None = None
        self._last_tracking_result: HybridTrackingResult | None = None
        self._current_machine_observations: tuple[MachineObservation, ...] = ()
        self._current_image_quality = None
        self._current_rejection_reasons: tuple[str, ...] = ()
        self._last_debug_previous_frame_bgr: np.ndarray | None = None
        self._highlighted_landmark_id: int | None = None

        for view, viewport in (
            (self._video_view, self._video_viewport),
            (self._map_view, self._map_viewport),
        ):
            viewport.installEventFilter(self)
            view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
            view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    @property
    def pair_count(self) -> int:
        """Число полностью заданных пар кадр–карта."""

        return len(self._correspondences)

    @property
    def has_pending_point(self) -> bool:
        """Есть ли точка кадра, для которой ещё не указана точка карты."""

        return self._pending_frame_point is not None

    @property
    def result(self) -> ManualAlignmentResult | None:
        """Последнее рассчитанное преобразование или ``None``."""

        return self._result

    @property
    def overlay_available(self) -> bool:
        """Построено ли наложение текущего кадра на карту."""

        return self._overlay_item is not None

    @property
    def overlay_visible(self) -> bool:
        """Видно ли рассчитанное наложение в данный момент."""

        return self._overlay_item is not None and self._overlay_item.isVisible()

    @property
    def source_ready(self) -> bool:
        """Загружены ли одновременно кадр и карта."""

        return self._video.is_loaded and self._map_image.is_loaded

    @property
    def tracking_active(self) -> bool:
        """Включено ли сопровождение маяков при смене кадров."""

        return self._tracking_active

    @property
    def tracking_available(self) -> bool:
        """Можно ли начать сопровождение с текущего опорного кадра."""

        return (
            self._result is not None
            and self.pair_count >= 4
            and not self.has_pending_point
        )

    @property
    def last_tracking_result(self) -> HybridTrackingResult | None:
        """Диагностика последнего успешно обработанного перехода."""

        return self._last_tracking_result

    @property
    def correction_pending(self) -> bool:
        """Требует ли текущий кадр ручного уточнения и подтверждения."""

        return self._correction_pending

    @property
    def remembered_frame_indices(self) -> tuple[int, ...]:
        """Индексы кадров с сохранённой разметкой или черновиком."""

        return self._session.frame_indices

    @property
    def current_landmark_ids(self) -> tuple[int, ...]:
        """Стабильные номера маяков, наблюдаемых на текущем кадре."""

        return tuple(pair.landmark_id for pair in self._correspondences)

    @property
    def point_editing_active(self) -> bool:
        """Включён ли режим добавления новых пар кликами."""

        return self._active

    @property
    def current_annotation_status(self) -> AnnotationStatus:
        """Состояние разметки, показанной оператору прямо сейчас."""

        return self._current_annotation_status

    @property
    def session(self) -> AlignmentSession:
        """Вернуть модель для немедленной сериализации главным окном."""

        return self._session

    @property
    def has_operator_progress(self) -> bool:
        """Есть ли работа, потерю которой нужно предотвращать при закрытии."""

        return bool(self._session.frame_indices) or self.has_pending_point

    @property
    def current_frame_is_keyframe(self) -> bool:
        """Подтверждён ли текущий кадр оператором как опорный."""

        if self._bound_frame_index is None:
            return False
        annotation = self._session.frame(self._bound_frame_index)
        return annotation is not None and annotation.is_keyframe

    def commit_visible_state(self) -> FrameAnnotation | None:
        """Перенести текущее законченное состояние кадра в модель проекта."""

        return self._remember_current_frame(status=self._current_annotation_status)

    def restore_session(
        self,
        session: AlignmentSession,
        *,
        current_frame_index: int,
    ) -> None:
        """Заменить рабочую память проверенной загруженной сессией."""

        if not self.source_ready:
            raise RuntimeError("Для восстановления проекта нужны видео и карта")
        if current_frame_index != self._video.current_frame_index:
            raise ValueError("Сначала нужно открыть сохранённый кадр проекта")
        self._stop_tracking()
        self.set_active(False)
        self._clear_visible_state(reset_bound_frame=True)
        self._session = session
        self._highlighted_landmark_id = None
        annotation = self._session.frame(current_frame_index)
        if annotation is None:
            self._bound_frame_index = current_frame_index
            self._current_annotation_status = "draft"
            self._correction_pending = False
        else:
            self._restore_annotation(annotation)
        self.message.emit(
            f"Проект восстановлен: {len(session.frame_indices)} кадров, "
            f"{session.landmark_count} маяков"
        )
        self._emit_state()

    def first_frame_requiring_attention(
        self,
        *,
        start_frame_index: int,
        frame_count: int,
    ) -> int | None:
        """Найти первый отсутствующий, черновой или отклонённый кадр."""

        first = max(0, int(start_frame_index))
        last = max(first, int(frame_count))
        for frame_index in range(first, last):
            annotation = self._session.frame(frame_index)
            if annotation is None or annotation.status in (
                "draft", "rejected", "stale"
            ):
                return frame_index
        return None

    def begin_current_frame_correction(self) -> bool:
        """Явно приостановить трекер и разблокировать текущие маяки."""

        if not self._correspondences:
            self.message.emit("На текущем кадре нет маяков для коррекции")
            return False
        self.set_active(False)
        if self._tracking_active:
            self.set_tracking_active(False)
        self._set_markers_editable(True)
        self.message.emit(
            "Коррекция: маяки разблокированы. Перетащите точку и подтвердите "
            "совмещение"
        )
        self._emit_state()
        return True

    def cancel_pending_point(self) -> bool:
        """Отменить только одиночный клик, не удаляя готовую пару."""

        if not self.has_pending_point:
            return False
        self._remove_pending_marker()
        self._invalidate_result()
        self.message.emit("Незавершённая точка кадра отменена")
        self._emit_state()
        return True

    def focus_landmark(self, landmark_id: int) -> bool:
        """Выбрать маяк по стабильному ID и центрировать обе области."""

        target = next(
            (item for item in self._correspondences if item.landmark_id == landmark_id),
            None,
        )
        if target is None:
            self.message.emit(f"На текущем кадре нет маяка {landmark_id + 1}")
            return False
        self._highlighted_landmark_id = landmark_id
        for item in self._correspondences:
            highlighted = item.landmark_id == landmark_id
            item.frame_marker.setSelected(highlighted)
            item.map_marker.setSelected(highlighted)
            item.frame_marker.setScale(1.8 if highlighted else 1.0)
            item.map_marker.setScale(1.8 if highlighted else 1.0)
            item.frame_marker.setZValue(50.0 if highlighted else 20.0)
            item.map_marker.setZValue(50.0 if highlighted else 20.0)
        self._video_view.centerOn(target.frame_marker)
        self._map_view.centerOn(target.map_marker)
        self.message.emit(f"Маяк {landmark_id + 1} выбран и подсвечен")
        return True

    @property
    def last_debug_previous_frame_bgr(self) -> np.ndarray | None:
        """Вернуть прошлый кадр последнего перехода только для диагностики."""

        if self._last_debug_previous_frame_bgr is None:
            return None
        return self._last_debug_previous_frame_bgr.copy()

    def debug_snapshot(self) -> dict[str, object]:
        """Снять состояние модели, редактора и реально показанных маркеров.

        Три перечня ID позволяют отличить математическую потерю маяка от
        ошибки представления: постоянный маяк может существовать в сессии,
        иметь наблюдение редактора, но по ошибке отсутствовать в Qt-сцене.
        """

        alignment: dict[str, object] | None = None
        if self._result is not None:
            alignment = {
                "homography_frame_to_map": self._result.homography_frame_to_map,
                "fit_rmse_map_px": self._result.fit_rmse_map_px,
                "fit_max_map_px": self._result.fit_max_map_px,
                "leave_one_out_median_map_px": (
                    self._result.leave_one_out_median_map_px
                ),
                "leave_one_out_max_map_px": self._result.leave_one_out_max_map_px,
                "frame_coverage_fraction": self._result.frame_coverage_fraction,
                "map_coverage_fraction": self._result.map_coverage_fraction,
                "projected_frame_corners_map_px": (
                    self._result.projected_frame_corners_map_px
                ),
            }

        tracking: dict[str, object] | None = None
        if self._last_tracking_result is not None:
            tracking = {
                "tracked_count": self._last_tracking_result.tracked_count,
                "inlier_count": self._last_tracking_result.inlier_count,
                "automatic_detected_count": (
                    self._last_tracking_result.automatic_detected_count
                ),
                "automatic_tracked_count": (
                    self._last_tracking_result.automatic_tracked_count
                ),
                "automatic_inlier_count": (
                    self._last_tracking_result.automatic_inlier_count
                ),
                "flow_valid_mask": self._last_tracking_result.flow_valid_mask,
                "homography_inlier_mask": (
                    self._last_tracking_result.homography_inlier_mask
                ),
                "forward_backward_errors_px": (
                    self._last_tracking_result.forward_backward_errors_px
                ),
            }

        image_quality: dict[str, object] | None = None
        if self._current_image_quality is not None:
            quality = self._current_image_quality
            image_quality = {
                "previous_laplacian_variance": quality.previous_laplacian_variance,
                "current_laplacian_variance": quality.current_laplacian_variance,
                "sharpness_ratio_to_previous": quality.sharpness_ratio_to_previous,
                "mean_luminance": quality.mean_luminance,
                "dark_pixel_fraction": quality.dark_pixel_fraction,
                "bright_pixel_fraction": quality.bright_pixel_fraction,
                "warning_reasons": quality.warning_reasons,
                "rejection_reasons": quality.rejection_reasons,
            }

        video_scene = self._video_view.scene()
        map_scene = self._map_view.scene()
        return {
            "bound_frame_index": self._bound_frame_index,
            "annotation_status": self._current_annotation_status,
            "point_editing_active": self._active,
            "tracking_active": self._tracking_active,
            "tracking_available": self.tracking_available,
            "correction_pending": self._correction_pending,
            "resume_tracking_after_confirmation": (
                self._resume_tracking_after_confirmation
            ),
            "pending_frame_point_px": self._pending_frame_point,
            "overlay_available": self.overlay_available,
            "overlay_visible": self.overlay_visible,
            "session_landmark_ids": self._session.landmark_ids,
            "remembered_frame_indices": self._session.frame_indices,
            "editor_landmark_ids": self.current_landmark_ids,
            "suspect_landmark_ids": sorted(self._suspect_landmark_ids),
            "frame_scene_landmark_ids": [
                pair.landmark_id
                for pair in self._correspondences
                if pair.frame_marker.scene() is video_scene
            ],
            "map_scene_landmark_ids": [
                pair.landmark_id
                for pair in self._correspondences
                if pair.map_marker.scene() is map_scene
            ],
            "observations": [
                {
                    "landmark_id": pair.landmark_id,
                    "frame_point_px": pair.frame_point,
                    "map_point_px": pair.map_point,
                    "frame_marker_selected": pair.frame_marker.isSelected(),
                    "map_marker_selected": pair.map_marker.isSelected(),
                }
                for pair in self._correspondences
            ],
            "machine_observations": [
                {
                    "source_frame_index": item.source_frame_index,
                    "source_frame_point_px": item.source_frame_point_px,
                    "frame_point_px": item.frame_point_px,
                    "map_point_px": item.map_point_px,
                    "forward_backward_error_px": item.forward_backward_error_px,
                    "used_for_alignment": item.used_for_alignment,
                }
                for item in self._current_machine_observations
            ],
            "image_quality": image_quality,
            "rejection_reasons": self._current_rejection_reasons,
            "alignment": alignment,
            "last_tracking": tracking,
        }

    def exclude_current_frame(
        self,
        reason: str = "оператор исключил кадр из измерений",
    ) -> bool:
        """Исключить показанный кадр, не выдавая его за неудачную привязку.

        Это операторское решение отличается от автоматического ``rejected``:
        исключённый кадр не требует исправления и никогда не используется как
        источник следующего перехода. Новая ручная разметка позднее может
        обычным способом заменить исключение на черновик и ключевой кадр.
        """

        frame_index = self._video.current_frame_index
        if frame_index is None:
            self.error_occurred.emit("Сначала загрузите источник кадров")
            return False
        self.exclude_frame_range(frame_index, frame_index, reason=reason)
        return True

    def exclude_frame_range(
        self,
        first_frame_index: int,
        last_frame_index: int,
        *,
        reason: str = "оператор исключил диапазон кадров из измерений",
    ) -> int:
        """Исключить замкнутый диапазон индексов без декодирования всех кадров.

        Для каждого индекса сохраняется самостоятельная пустая аннотация. Так
        будущий экспорт сможет отличить сознательно отброшенный материал от
        ещё не просмотренного. Уже удачные кадры вне диапазона не меняются.
        """

        first = int(first_frame_index)
        last = int(last_frame_index)
        if first < 0 or last < first:
            raise ValueError("Диапазон исключения кадров задан неверно")
        frame_count = self._video.frame_count
        if frame_count > 0 and last >= frame_count:
            raise ValueError("Диапазон исключения выходит за пределы видео")
        normalized_reason = str(reason).strip()
        if not normalized_reason:
            raise ValueError("Причина исключения кадра не может быть пустой")

        self._video.pause()
        self.set_active(False)
        self._stop_tracking()
        self._resume_tracking_after_confirmation = False
        for frame_index in range(first, last + 1):
            self._session.remember_frame(
                frame_index,
                (),
                alignment=None,
                status="excluded",
                rejection_reasons=(normalized_reason,),
            )

        current_index = self._video.current_frame_index
        if current_index is not None and first <= current_index <= last:
            self._clear_visible_state(reset_bound_frame=True)
            self._bound_frame_index = current_index
            self._current_annotation_status = "excluded"
            self._correction_pending = False
            self._current_rejection_reasons = (normalized_reason,)

        excluded_count = last - first + 1
        self.debug_event.emit(
            "frames_excluded",
            {
                "first_frame_index": first,
                "last_frame_index": last,
                "frame_count": excluded_count,
                "reason": normalized_reason,
            },
            False,
        )
        self.message.emit(
            f"Исключены кадры {first}…{last}: {normalized_reason}. "
            "Они не участвуют в совмещении и измерениях"
        )
        self._emit_state()
        return excluded_count

    def set_active(self, active: bool) -> None:
        """Включить режим последовательных кликов «кадр, затем карта»."""

        self._active = bool(active) and self.source_ready
        if self._active:
            self._video.pause()
        drag_mode = (
            QGraphicsView.DragMode.NoDrag
            if self._active
            else QGraphicsView.DragMode.ScrollHandDrag
        )
        cursor = (
            Qt.CursorShape.CrossCursor if self._active else Qt.CursorShape.ArrowCursor
        )
        for view in (self._video_view, self._map_view):
            view.setDragMode(drag_mode)
            view.viewport().setCursor(cursor)
        self._set_markers_editable(not self._active and not self._tracking_active)

        if self._active:
            self.message.emit(
                "Режим точек: щёлкните наземный ориентир на кадре, затем на карте"
            )
        self._emit_state()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Масштабировать колесом и принимать клики в исходных координатах."""

        # При завершении QApplication C++ может послать последнее событие уже
        # после очистки Python-атрибутов объекта. Такой поздний вызов безопасно
        # игнорируем вместо аварии интерфейса.
        if getattr(self, "_disposed", True):
            return False

        view = self._view_for_viewport(watched)
        if view is None:
            return super().eventFilter(watched, event)

        if event.type() == QEvent.Type.Wheel:
            delta = event.angleDelta().y()  # type: ignore[attr-defined]
            if delta == 0:
                return True
            factor = 1.2 if delta > 0 else 1.0 / 1.2
            current_scale = abs(view.transform().m11())
            next_scale = current_scale * factor
            if 0.02 <= next_scale <= 50.0:
                view.scale(factor, factor)
            return True

        if not self._active or event.type() != QEvent.Type.MouseButtonPress:
            return super().eventFilter(watched, event)
        if event.button() != Qt.MouseButton.LeftButton:  # type: ignore[attr-defined]
            return super().eventFilter(watched, event)

        viewport_position = event.position().toPoint()  # type: ignore[attr-defined]
        scene_position = view.mapToScene(viewport_position)
        self._accept_click(view, scene_position)
        return True

    def add_correspondence(self, frame_point: Point, map_point: Point) -> bool:
        """Добавить готовую пару; публичный метод нужен и для GUI-тестов.

        Обычный операторский сценарий использует клики. Здесь выполняется та же
        проверка границ, чтобы тест или будущий импорт разметки не мог добавить
        координаты за пределами изображений.
        """

        if self._tracking_active:
            self.set_tracking_active(False)
        if not self.source_ready:
            self.error_occurred.emit("Сначала загрузите кадр и карту")
            return False
        if not self._point_inside(frame_point, self._video.current_frame_bgr):
            self.error_occurred.emit("Точка кадра находится вне изображения")
            return False
        if not self._point_inside(map_point, self._map_image.current_image_bgr):
            self.error_occurred.emit("Точка карты находится вне изображения")
            return False

        if self.has_pending_point:
            self._remove_pending_marker()
        self._bind_current_frame()
        self._append_pair(frame_point, map_point)
        return True

    def undo(self) -> None:
        """Удалить незавершённую точку или последнюю завершённую пару."""

        if self._tracking_active:
            self.set_tracking_active(False)
        if self.has_pending_point:
            self._remove_pending_marker()
            self._invalidate_result()
            self.message.emit("Незавершённая точка кадра удалена")
            self._emit_state()
            return
        if not self._correspondences:
            return

        pair = self._correspondences.pop()
        self._suspect_landmark_ids.discard(pair.landmark_id)
        self._remove_item(self._video_view, pair.frame_marker)
        self._remove_item(self._map_view, pair.map_marker)
        self._invalidate_result()
        self._correction_pending = True
        self._current_annotation_status = "draft"
        self._invalidate_future_after_operator_change()
        self._remember_current_frame(status="draft")
        self.debug_event.emit(
            "landmark_observation_removed",
            {
                "frame_index": self._bound_frame_index,
                "landmark_ids": [pair.landmark_id],
                "reason": "undo",
                "remaining_count": self.pair_count,
            },
            False,
        )
        self.message.emit(f"Удалена последняя пара; осталось {self.pair_count}")
        self._emit_state()

    def remove_selected(self) -> None:
        """Удалить выбранные наблюдения только с текущего кадра.

        Сам картографический маяк и его наблюдения на предыдущих кадрах остаются
        в памяти. Поэтому ошибочный или невидимый сейчас маяк не уничтожает
        уже подтверждённую работу оператора.
        """

        if self._tracking_active:
            self.set_tracking_active(False)
        selected = {
            pair.landmark_id
            for pair in self._correspondences
            if pair.frame_marker.isSelected() or pair.map_marker.isSelected()
        }
        if not selected:
            self.message.emit("Выберите маяк на кадре или карте")
            return
        self._remove_current_observations(selected)
        self.debug_event.emit(
            "landmark_observation_removed",
            {
                "frame_index": self._bound_frame_index,
                "landmark_ids": sorted(selected),
                "reason": "remove_selected",
                "remaining_count": self.pair_count,
            },
            False,
        )
        self.message.emit(
            f"Удалено наблюдений: {len(selected)}; осталось {self.pair_count}"
        )

    def refine_frame_point(self, landmark_id: int, point: Point) -> bool:
        """Уточнить видеокоординату известного маяка без повторного клика по карте.

        В обычной работе оператор просто перетаскивает маркер мышью. Публичный
        метод даёт тому же действию проверяемый интерфейс для тестов и будущего
        импорта внешней разметки; вся фиксация черновика остаётся в общем
        обработчике перемещения маркера.
        """

        if self._tracking_active:
            self.set_tracking_active(False)
        pair = next(
            (
                candidate
                for candidate in self._correspondences
                if candidate.landmark_id == landmark_id
            ),
            None,
        )
        if pair is None:
            self.error_occurred.emit(f"На текущем кадре нет маяка {landmark_id + 1}")
            return False
        if not self._point_inside(point, self._video.current_frame_bgr):
            self.error_occurred.emit("Новая координата маяка находится вне кадра")
            return False

        pair.frame_marker.setPos(*point)
        return True

    def clear(self) -> None:
        """Удалить всю операторскую сессию, включая историю кадров."""

        cleared_landmark_ids = self._session.landmark_ids
        cleared_frame_indices = self._session.frame_indices
        self._stop_tracking()
        self._clear_visible_state(reset_bound_frame=True)
        self._session.clear()
        self._correction_pending = False
        self._resume_tracking_after_confirmation = False
        self._current_annotation_status = "draft"
        self.debug_event.emit(
            "alignment_session_cleared",
            {
                "landmark_ids": cleared_landmark_ids,
                "frame_indices": cleared_frame_indices,
            },
            False,
        )
        self.message.emit("Маяки и история кадров очищены")
        self._emit_state()

    def set_tracking_active(self, active: bool) -> bool:
        """Начать или остановить перенос маяков на соседние кадры.

        Начальная гомография должна быть заранее подтверждена действием
        ``Compute alignment``. Это не даёт случайно запустить сопровождение по
        четырём ещё не проверенным кликам.
        """

        active = bool(active)
        if not active:
            self._stop_tracking()
            self._resume_tracking_after_confirmation = False
            self.debug_event.emit(
                "tracking_stopped",
                {
                    "frame_index": self._bound_frame_index,
                    "landmark_ids": self.current_landmark_ids,
                },
                False,
            )
            self.message.emit("Сопровождение маяков остановлено")
            self._emit_state()
            return True
        if self._tracking_active:
            return True
        if not self.tracking_available:
            self.error_occurred.emit(
                "Сначала задайте минимум четыре пары и вычислите совмещение"
            )
            self.tracking_changed.emit(False)
            return False

        frame_bgr = self._video.current_frame_bgr
        frame_index = self._video.current_frame_index
        if frame_bgr is None or frame_index is None:
            self.error_occurred.emit("Текущий кадр недоступен")
            self.tracking_changed.emit(False)
            return False

        # Во время сопровождения клики не должны незаметно менять набор точек.
        self.set_active(False)
        self._tracking_active = True
        self._resume_tracking_after_confirmation = False
        self._set_markers_editable(False)
        self._tracking_frame_bgr = frame_bgr
        self._last_tracking_result = None
        self._bound_frame_index = frame_index
        self.tracking_changed.emit(True)
        self.debug_event.emit(
            "tracking_started",
            {
                "frame_index": frame_index,
                "landmark_ids": self.current_landmark_ids,
                "landmark_count": self.pair_count,
            },
            False,
        )
        self.message.emit(
            f"Сопровождение запущено с кадра {frame_index}; "
            f"активных маяков: {self.pair_count}"
        )
        self._emit_state()
        return True

    def compute_alignment(self) -> ManualAlignmentResult | None:
        """Оценить гомографию и создать полупрозрачный кадр на карте."""

        frame_bgr = self._video.current_frame_bgr
        map_bgr = self._map_image.current_image_bgr
        if frame_bgr is None or map_bgr is None:
            self.error_occurred.emit("Сначала загрузите кадр и карту")
            return None
        if self.pair_count < 4:
            self.error_occurred.emit("Для расчёта нужны минимум четыре пары точек")
            return None

        frame_points = [pair.frame_point for pair in self._correspondences]
        map_points = [pair.map_point for pair in self._correspondences]
        try:
            result = estimate_manual_alignment(
                frame_points,
                map_points,
                frame_size_pixels=(frame_bgr.shape[1], frame_bgr.shape[0]),
                map_size_pixels=(map_bgr.shape[1], map_bgr.shape[0]),
            )
        except ManualAlignmentFailure as error:
            self._invalidate_result()
            self.error_occurred.emit(str(error))
            self._emit_state()
            return None

        quality = assess_manual_alignment_quality(
            result,
            frame_size_pixels=(frame_bgr.shape[1], frame_bgr.shape[0]),
            map_size_pixels=(map_bgr.shape[1], map_bgr.shape[0]),
        )
        if not quality.is_acceptable:
            self._reject_low_quality_alignment(result, quality)
            return None

        should_resume_tracking = self._resume_tracking_after_confirmation
        self._invalidate_result()
        self._result = result
        self._last_tracking_result = None
        self._overlay_item = self._make_overlay(
            frame_bgr, map_bgr, result.homography_frame_to_map
        )
        self._overlay_item.setVisible(True)
        self._current_annotation_status = "keyframe"
        self._correction_pending = False
        self._resume_tracking_after_confirmation = False
        self._suspect_landmark_ids.clear()
        for pair in self._correspondences:
            self._style_frame_marker(pair.frame_marker, suspect=False)
        self._remember_current_frame(status="keyframe")
        self._set_markers_editable(not self._tracking_active)
        self.alignment_computed.emit(result)
        self.debug_event.emit(
            "alignment_computed",
            {
                "frame_index": self._bound_frame_index,
                "landmark_ids": self.current_landmark_ids,
                "fit_rmse_map_px": result.fit_rmse_map_px,
                "fit_max_map_px": result.fit_max_map_px,
                "frame_coverage_fraction": result.frame_coverage_fraction,
                "map_coverage_fraction": result.map_coverage_fraction,
                "resume_tracking_after_confirmation": should_resume_tracking,
            },
            False,
        )
        alignment_summary = self._result_summary(result)
        if should_resume_tracking and self.set_tracking_active(True):
            self.message.emit(
                f"{alignment_summary} | сопровождение продолжено автоматически"
            )
        else:
            self.message.emit(alignment_summary)
            self._emit_state()
        return result

    def _reject_low_quality_alignment(
        self,
        result: ManualAlignmentResult,
        quality: ManualAlignmentQuality,
    ) -> None:
        """Не показывать заведомо искажённое наложение и указать кандидата.

        Отклонённая матрица не становится опорой. Все клики сохраняются как
        черновик, поэтому оператор может исправить одну пару без повторной
        разметки остальных ориентиров.
        """

        suspected_landmark_id: int | None = None
        if quality.suspected_outlier_index is not None:
            suspected_landmark_id = self._correspondences[
                quality.suspected_outlier_index
            ].landmark_id
            suspect_ids = {suspected_landmark_id}
        else:
            suspect_ids = set(self.current_landmark_ids)

        self._invalidate_result()
        self._suspect_landmark_ids = suspect_ids
        for pair in self._correspondences:
            self._style_frame_marker(
                pair.frame_marker,
                suspect=pair.landmark_id in suspect_ids,
            )
        self._correction_pending = True
        self._current_annotation_status = "draft"
        self._remember_current_frame(status="draft")

        if suspected_landmark_id is None:
            correction_hint = "Проверьте соответствия кадр–карта."
        else:
            assert quality.fit_rmse_without_suspected_map_px is not None
            correction_hint = (
                f"Вероятно неверна пара {suspected_landmark_id + 1}: без неё "
                "средняя ошибка снижается до "
                f"{quality.fit_rmse_without_suspected_map_px:.2f} px. "
                "Если неверна точка карты, удалите эту пару и создайте заново."
            )
        message = (
            "Совмещение отклонено: средняя ошибка "
            f"{result.fit_rmse_map_px:.2f} px при допустимой "
            f"≤{quality.fit_rmse_limit_map_px:.2f}; максимальная "
            f"{result.fit_max_map_px:.2f} px при допустимой "
            f"≤{quality.fit_max_limit_map_px:.2f}. {correction_hint}"
        )
        self.debug_event.emit(
            "alignment_rejected",
            {
                "frame_index": self._bound_frame_index,
                "landmark_ids": self.current_landmark_ids,
                "suspected_landmark_id": suspected_landmark_id,
                "fit_rmse_map_px": result.fit_rmse_map_px,
                "fit_max_map_px": result.fit_max_map_px,
                "fit_rmse_limit_map_px": quality.fit_rmse_limit_map_px,
                "fit_max_limit_map_px": quality.fit_max_limit_map_px,
                "fit_rmse_without_suspected_map_px": (
                    quality.fit_rmse_without_suspected_map_px
                ),
                "fit_max_without_suspected_map_px": (
                    quality.fit_max_without_suspected_map_px
                ),
            },
            True,
        )
        self.error_occurred.emit(message)
        self._emit_state()

    def set_overlay_visible(self, visible: bool) -> None:
        """Показать или скрыть уже рассчитанное наложение."""

        if self._overlay_item is not None:
            self._overlay_item.setVisible(visible)

    def show_automatic_alignment_overlay(
        self,
        frame_bgr: np.ndarray,
        homography_frame_to_map: np.ndarray,
    ) -> bool:
        """Показать принятый автоматический результат тем же слоем карты.

        Кадр может быть исправлен по калибровке и потому не обязан совпадать с
        сырым массивом видеоконтроллера. Матрица должна относиться именно к
        переданному здесь изображению.
        """

        map_bgr = self._map_image.current_image_bgr
        if map_bgr is None:
            self.error_occurred.emit("Карта была выгружена до показа результата")
            return False
        # Автоматический слой заменяет только картинку наложения. Ручная
        # гомография остаётся эталоном текущего кадра, чтобы её можно было
        # экспортировать и количественно сравнить с автоматической попыткой.
        # Любое изменение самих ручных точек по-прежнему удаляет этот эталон.
        self._remove_overlay()
        self._overlay_item = self._make_overlay(
            frame_bgr, map_bgr, homography_frame_to_map
        )
        self._overlay_item.setVisible(True)
        self._emit_state()
        return True

    def dispose(self) -> None:
        """Снять фильтры событий до того, как Qt удалит области просмотра.

        PySide удаляет дочерние C++-виджеты окна в собственном порядке. Явное
        снятие фильтров не даёт закрытому редактору получить позднее событие
        уничтоженного ``viewport`` во время завершения приложения.
        """

        if self._disposed:
            return
        self._disposed = True
        for viewport in (self._video_viewport, self._map_viewport):
            try:
                viewport.removeEventFilter(self)
            except RuntimeError:
                # Виджет уже мог быть удалён при аварийном закрытии окна.
                pass

    def on_frame_changed(self, frame_index: int, _frame_count: int) -> None:
        """Восстановить известный кадр либо обработать новый соседний кадр."""

        previous_index = self._bound_frame_index
        if previous_index == frame_index:
            return
        if previous_index is not None:
            self._remember_current_frame(status=self._current_annotation_status)

        cached = self._session.frame(frame_index)
        if cached is not None:
            # Явно включённое сопровождение означает просьбу повторить перенос
            # на соседний кадр. Старый черновик мог появиться после прежнего
            # срыва и не должен блокировать новую попытку. Подтверждённые
            # ``tracked``/``keyframe`` кадры, напротив, безопасно восстанавливаем
            # из памяти, чтобы не портить уже принятую оператором работу.
            should_retry_draft = (
                self._tracking_active
                and cached.alignment is None
                and cached.status in ("draft", "rejected", "stale")
                and previous_index is not None
                and abs(frame_index - previous_index) == 1
            )
            if should_retry_draft:
                self._track_to_frame(frame_index)
                return

            self._restore_annotation(cached)
            if cached.alignment is None and self._tracking_active:
                self._stop_tracking()
            self.message.emit(
                f"Кадр {frame_index} восстановлен из памяти: "
                f"{len(cached.observations)} маяков, состояние {cached.status}"
            )
            self._emit_state()
            return

        if self._tracking_active and previous_index is not None:
            self._track_to_frame(frame_index)
            return

        self._clear_visible_state(reset_bound_frame=True)
        self._correction_pending = False
        self._current_annotation_status = "draft"
        self.message.emit(
            f"Кадр {frame_index} ещё не размечен; предыдущая история сохранена"
        )
        self._emit_state()

    def on_video_loaded(self, path: str, _frame_count: int, _fps: float) -> None:
        """Очистить память только при фактической замене источника кадров."""

        if self._video_source_path is not None and path != self._video_source_path:
            self.clear()
            self.message.emit("Источник кадров изменён — старая история очищена")
        self._video_source_path = path

    def on_map_loaded(self, _path: str, _width: int, _height: int) -> None:
        """Сбросить всю сессию после замены опорного изображения карты."""

        if (
            self.remembered_frame_indices
            or self.pair_count
            or self.has_pending_point
            or self.overlay_available
        ):
            self.clear()
            self.message.emit("Карта изменена — старая история очищена")

    def _accept_click(self, view: QGraphicsView, position: QPointF) -> None:
        if self._tracking_active:
            self.message.emit("Сначала остановите сопровождение маяков")
            return
        point = (float(position.x()), float(position.y()))
        if view is self._video_view:
            if self.has_pending_point:
                self.message.emit("Теперь укажите эту же точку на карте")
                return
            if not self._point_inside(point, self._video.current_frame_bgr):
                self.message.emit("Клик вне кадра")
                return
            self._bind_current_frame()
            self._invalidate_result()
            self._correction_pending = True
            self._pending_frame_point = point
            self._pending_frame_marker = self._add_static_marker(
                self._video_view,
                point,
                self._session.landmark_count,
            )
            self.message.emit("Точка кадра принята; укажите её положение на карте")
            self._emit_state()
            return

        if not self.has_pending_point:
            self.message.emit("Сначала укажите точку на кадре")
            return
        if not self._point_inside(point, self._map_image.current_image_bgr):
            self.message.emit("Клик вне карты")
            return

        assert self._pending_frame_point is not None
        assert self._pending_frame_marker is not None
        frame_point = self._pending_frame_point
        self._remove_item(self._video_view, self._pending_frame_marker)
        self._pending_frame_point = None
        self._pending_frame_marker = None
        self._append_pair(frame_point, point)
        self.message.emit(
            f"Пара {self.pair_count} принята; следующая точка начинается с кадра"
        )

    def _append_pair(self, frame_point: Point, map_point: Point) -> None:
        """Создать новый картографический маяк и наблюдение текущего кадра."""

        self._bind_current_frame()
        self._invalidate_result()
        landmark_id = self._session.add_landmark(map_point)
        self._append_existing_observation(
            landmark_id,
            frame_point,
            map_point,
            suspect=False,
        )
        self._correction_pending = True
        self._current_annotation_status = "draft"
        self._invalidate_future_after_operator_change()
        self._remember_current_frame(status="draft")
        self.debug_event.emit(
            "landmark_added",
            {
                "frame_index": self._bound_frame_index,
                "landmark_id": landmark_id,
                "frame_point_px": frame_point,
                "map_point_px": map_point,
                "landmark_count": self.pair_count,
            },
            False,
        )
        self.message.emit(
            f"Маяк {landmark_id + 1} добавлен; всего на кадре {self.pair_count}"
        )
        self._emit_state()

    def _append_existing_observation(
        self,
        landmark_id: int,
        frame_point: Point,
        map_point: Point,
        *,
        suspect: bool,
    ) -> None:
        """Показать существующий маяк на текущем кадре без создания нового ID."""

        frame_marker = self._add_frame_marker(frame_point, landmark_id)
        map_marker = self._add_static_marker(
            self._map_view,
            map_point,
            landmark_id,
        )
        self._style_frame_marker(frame_marker, suspect=suspect)
        self._correspondences.append(
            _Correspondence(
                landmark_id=landmark_id,
                frame_point=frame_point,
                map_point=map_point,
                frame_marker=frame_marker,
                map_marker=map_marker,
            )
        )
        if landmark_id == self._highlighted_landmark_id:
            frame_marker.setScale(1.8)
            map_marker.setScale(1.8)
            frame_marker.setSelected(True)
            map_marker.setSelected(True)

    @staticmethod
    def _style_frame_marker(marker: _FrameMarker, *, suspect: bool) -> None:
        """Отличить наблюдение, не принятое геометрическим фильтром.

        Оранжевый цвет означает только «не использован в текущей гомографии».
        Сам маяк и координата, найденная оптическим потоком, сохраняются: один
        статистический фильтр не должен уничтожать операторскую разметку.
        """

        marker.setPen(QPen(QColor("#ffb300" if suspect else "#2196f3"), 3.0))
        marker.setToolTip(
            "Не использован в текущем совмещении: проверьте точку" if suspect else ""
        )

    def _add_frame_marker(self, point: Point, landmark_id: int) -> _FrameMarker:
        """Создать выбираемый и при остановленном трекере подвижный маркер."""

        marker = _FrameMarker()
        self._decorate_marker(marker, point, landmark_id, self._video_view)
        marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        marker.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
            not self._active and not self._tracking_active,
        )
        marker.set_move_callback(
            lambda moved_point, marker_id=landmark_id: self._on_frame_marker_moved(
                marker_id, moved_point
            )
        )
        return marker

    def _add_static_marker(
        self,
        view: QGraphicsView,
        point: Point,
        landmark_id: int,
    ) -> QGraphicsEllipseItem:
        """Создать неподвижный маркер карты или незавершённой точки кадра."""

        marker = QGraphicsEllipseItem(-5.0, -5.0, 10.0, 10.0)
        self._decorate_marker(marker, point, landmark_id, view)
        marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        return marker

    @staticmethod
    def _decorate_marker(
        marker: QGraphicsEllipseItem,
        point: Point,
        landmark_id: int,
        view: QGraphicsView,
    ) -> None:
        """Назначить общий вид, подпись и сцену маркера."""

        scene = view.scene()
        if scene is None:
            raise RuntimeError("Для области просмотра не создана графическая сцена")

        color = _POINT_COLORS[landmark_id % len(_POINT_COLORS)]
        marker.setPen(QPen(QColor("#101010"), 1.5))
        marker.setBrush(QBrush(color))
        marker.setPos(*point)
        marker.setZValue(20.0)
        marker.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True,
        )
        label = QGraphicsSimpleTextItem(str(landmark_id + 1), marker)
        label.setBrush(QBrush(Qt.GlobalColor.white))
        label.setPos(7.0, -11.0)
        scene.addItem(marker)

    def _make_overlay(
        self,
        frame_bgr: np.ndarray,
        map_bgr: np.ndarray,
        homography_frame_to_map: np.ndarray,
    ) -> QGraphicsPixmapItem:
        map_height, map_width = map_bgr.shape[:2]
        warped_bgr = cv2.warpPerspective(
            frame_bgr,
            homography_frame_to_map,
            (map_width, map_height),
            flags=cv2.INTER_LINEAR,
        )
        source_mask = np.full(frame_bgr.shape[:2], 255, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(
            source_mask,
            homography_frame_to_map,
            (map_width, map_height),
            flags=cv2.INTER_NEAREST,
        )

        rgba = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = np.where(warped_mask > 0, 128, 0).astype(np.uint8)
        image = QImage(
            rgba.data,
            map_width,
            map_height,
            map_width * 4,
            QImage.Format.Format_RGBA8888,
        ).copy()
        overlay = QGraphicsPixmapItem(QPixmap.fromImage(image))
        overlay.setZValue(10.0)
        scene = self._map_view.scene()
        if scene is None:
            raise RuntimeError("Для карты не создана графическая сцена")
        scene.addItem(overlay)
        return overlay

    def _remove_overlay(self) -> None:
        if self._overlay_item is not None:
            self._remove_item(self._map_view, self._overlay_item)
        self._overlay_item = None

    def _invalidate_result(self) -> None:
        self._remove_overlay()
        self._result = None
        self._last_tracking_result = None
        self._current_machine_observations = ()
        self._current_image_quality = None
        self._current_rejection_reasons = ()

    def _invalidate_future_after_operator_change(self) -> tuple[int, ...]:
        """Отозвать только машинные результаты, зависящие от текущего кадра."""

        if self._bound_frame_index is None:
            return ()
        invalidated = self._session.invalidate_derived_frames_after(
            self._bound_frame_index
        )
        if invalidated:
            self.debug_event.emit(
                "future_frames_invalidated",
                {
                    "corrected_frame_index": self._bound_frame_index,
                    "invalidated_frame_indices": invalidated,
                },
                False,
            )
        return invalidated

    def _remember_current_frame(
        self,
        *,
        status: AnnotationStatus | None = None,
    ) -> FrameAnnotation | None:
        """Сохранить видимые наблюдения, не затрагивая остальные кадры."""

        if self._bound_frame_index is None:
            return None
        requested_status = status or self._current_annotation_status
        if self._result is None and requested_status not in (
            "draft",
            "rejected",
            "excluded",
        ):
            requested_status = "draft"
        observations = [
            FrameObservation(pair.landmark_id, pair.frame_point)
            for pair in self._correspondences
        ]
        annotation = self._session.remember_frame(
            self._bound_frame_index,
            observations,
            alignment=self._result,
            status=requested_status,
            suspect_landmark_ids=self._suspect_landmark_ids,
            machine_observations=self._current_machine_observations,
            image_quality=self._current_image_quality,
            rejection_reasons=self._current_rejection_reasons,
        )
        self._current_annotation_status = requested_status
        return annotation

    def _restore_annotation(self, annotation: FrameAnnotation) -> None:
        """Показать сохранённые наблюдения и гомографию выбранного кадра."""

        self._clear_visible_state(reset_bound_frame=True)
        self._bound_frame_index = annotation.frame_index
        self._current_annotation_status = annotation.status
        self._correction_pending = annotation.status in (
            "draft", "rejected", "stale"
        )
        self._suspect_landmark_ids = set(annotation.suspect_landmark_ids)
        self._current_machine_observations = annotation.machine_observations
        self._current_image_quality = annotation.image_quality
        self._current_rejection_reasons = annotation.rejection_reasons
        for observation in annotation.observations:
            landmark = self._session.landmark(observation.landmark_id)
            self._append_existing_observation(
                observation.landmark_id,
                observation.frame_point_px,
                landmark.map_point_px,
                suspect=(observation.landmark_id in annotation.suspect_landmark_ids),
            )

        self._result = annotation.alignment
        frame_bgr = self._video.current_frame_bgr
        map_bgr = self._map_image.current_image_bgr
        if self._result is not None and frame_bgr is not None and map_bgr is not None:
            self._overlay_item = self._make_overlay(
                frame_bgr,
                map_bgr,
                self._result.homography_frame_to_map,
            )
            self._overlay_item.setVisible(True)
        self._tracking_frame_bgr = (
            frame_bgr if self._tracking_active and self._result is not None else None
        )
        self._set_markers_editable(not self._tracking_active)

    def _clear_visible_state(self, *, reset_bound_frame: bool) -> None:
        """Убрать только показ текущего кадра, не очищая память сессии."""

        self._remove_pending_marker()
        for pair in self._correspondences:
            self._remove_item(self._video_view, pair.frame_marker)
            self._remove_item(self._map_view, pair.map_marker)
        self._correspondences.clear()
        self._suspect_landmark_ids.clear()
        self._invalidate_result()
        if reset_bound_frame:
            self._bound_frame_index = None

    def _on_frame_marker_moved(self, landmark_id: int, point: Point) -> None:
        """Зафиксировать ручное уточнение координаты перетаскиванием."""

        if self._suppress_marker_callbacks:
            return
        pair = next(
            (
                candidate
                for candidate in self._correspondences
                if candidate.landmark_id == landmark_id
            ),
            None,
        )
        if pair is None:
            return
        if not self._point_inside(point, self._video.current_frame_bgr):
            self._suppress_marker_callbacks = True
            pair.frame_marker.setPos(*pair.frame_point)
            self._suppress_marker_callbacks = False
            self.message.emit(f"Маяк {landmark_id + 1} нельзя вынести за кадр")
            return

        pair.frame_point = point
        self._suspect_landmark_ids.discard(landmark_id)
        self._style_frame_marker(pair.frame_marker, suspect=False)
        self._invalidate_result()
        self._correction_pending = True
        self._current_annotation_status = "draft"
        self._invalidate_future_after_operator_change()
        self._remember_current_frame(status="draft")
        self.debug_event.emit(
            "landmark_moved",
            {
                "frame_index": self._bound_frame_index,
                "landmark_id": landmark_id,
                "frame_point_px": point,
                "map_point_px": pair.map_point,
            },
            False,
        )
        self.message.emit(
            f"Маяк {landmark_id + 1} уточнён; подтвердите Compute alignment"
        )
        self._emit_state()

    def _set_markers_editable(self, editable: bool) -> None:
        """Разрешить перетаскивание видеомаяков, когда трекер остановлен."""

        for pair in self._correspondences:
            pair.frame_marker.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                editable,
            )

    def _remove_current_observations(self, landmark_ids: set[int]) -> None:
        """Удалить наблюдения с текущего кадра и сохранить новый черновик."""

        retained: list[_Correspondence] = []
        for pair in self._correspondences:
            if pair.landmark_id not in landmark_ids:
                retained.append(pair)
                continue
            self._remove_item(self._video_view, pair.frame_marker)
            self._remove_item(self._map_view, pair.map_marker)
        self._correspondences = retained
        self._suspect_landmark_ids.difference_update(landmark_ids)
        self._invalidate_result()
        self._correction_pending = True
        self._current_annotation_status = "draft"
        self._invalidate_future_after_operator_change()
        self._remember_current_frame(status="draft")
        self._emit_state()

    def _track_to_frame(self, frame_index: int) -> None:
        """Обработать ровно один переход вперёд или назад по видео."""

        previous_index = self._bound_frame_index
        previous_frame = self._tracking_frame_bgr
        current_frame = self._video.current_frame_bgr
        map_bgr = self._map_image.current_image_bgr
        if (
            previous_index is None
            or previous_frame is None
            or current_frame is None
            or map_bgr is None
            or self._result is None
        ):
            self._fail_tracking(
                "Исходные изображения сопровождения недоступны",
                frame_index,
            )
            return
        if abs(frame_index - previous_index) != 1:
            self._fail_tracking(
                "Сопровождение работает только между соседними кадрами; "
                "создайте новый опорный кадр",
                frame_index,
            )
            return
        self._last_debug_previous_frame_bgr = previous_frame.copy()
        input_landmark_ids = self.current_landmark_ids

        frame_points = [pair.frame_point for pair in self._correspondences]
        map_points = [pair.map_point for pair in self._correspondences]
        try:
            tracking = track_beacons_with_machine_support_to_map(
                previous_frame,
                current_frame,
                frame_points,
                map_points,
                previous_alignment=self._result,
                map_size_pixels=(map_bgr.shape[1], map_bgr.shape[0]),
            )
        except BeaconTrackingFailure as error:
            self._fail_tracking(str(error), frame_index, partial=error)
            return
        transition_landmarks = []
        for index, landmark_id in enumerate(input_landmark_ids):
            transition_landmarks.append(
                {
                    "landmark_id": landmark_id,
                    "previous_frame_point_px": frame_points[index],
                    "current_frame_point_px": (tracking.current_frame_points_px[index]),
                    "map_point_px": map_points[index],
                    "forward_backward_error_px": (
                        tracking.forward_backward_errors_px[index]
                    ),
                    "flow_valid": tracking.flow_valid_mask[index],
                    "homography_inlier": tracking.homography_inlier_mask[index],
                }
            )

        suspect_landmark_ids = {
            landmark_id
            for index, landmark_id in enumerate(input_landmark_ids)
            if not tracking.homography_inlier_mask[index]
        }
        self._suppress_marker_callbacks = True
        for index, pair in enumerate(self._correspondences):
            if tracking.flow_valid_mask[index]:
                point_array = tracking.current_frame_points_px[index]
                pair.frame_point = (float(point_array[0]), float(point_array[1]))
                pair.frame_marker.setPos(*pair.frame_point)
            self._style_frame_marker(
                pair.frame_marker,
                suspect=pair.landmark_id in suspect_landmark_ids,
            )
        self._suppress_marker_callbacks = False
        self._suspect_landmark_ids = suspect_landmark_ids

        self._remove_overlay()
        self._result = tracking.alignment
        self._last_tracking_result = tracking
        self._current_machine_observations = tuple(
            MachineObservation(
                source_frame_index=previous_index,
                source_frame_point_px=(
                    float(tracking.automatic_previous_frame_points_px[index, 0]),
                    float(tracking.automatic_previous_frame_points_px[index, 1]),
                ),
                frame_point_px=(
                    float(tracking.automatic_current_frame_points_px[index, 0]),
                    float(tracking.automatic_current_frame_points_px[index, 1]),
                ),
                map_point_px=(
                    float(tracking.automatic_map_points_px[index, 0]),
                    float(tracking.automatic_map_points_px[index, 1]),
                ),
                forward_backward_error_px=float(
                    tracking.automatic_forward_backward_errors_px[index]
                ),
                used_for_alignment=bool(
                    tracking.automatic_homography_inlier_mask[index]
                ),
            )
            for index in np.flatnonzero(tracking.automatic_flow_valid_mask)
        )
        self._current_image_quality = tracking.image_quality
        self._current_rejection_reasons = ()
        self._overlay_item = self._make_overlay(
            current_frame,
            map_bgr,
            tracking.alignment.homography_frame_to_map,
        )
        self._overlay_item.setVisible(True)
        self._tracking_frame_bgr = current_frame
        self._bound_frame_index = frame_index
        self._current_annotation_status = "tracked"
        self._correction_pending = False
        self._remember_current_frame(status="tracked")
        self._set_markers_editable(False)
        self.alignment_computed.emit(tracking.alignment)
        self.debug_event.emit(
            "tracking_succeeded",
            {
                "previous_frame_index": previous_index,
                "frame_index": frame_index,
                "tracked_count": tracking.tracked_count,
                "inlier_count": tracking.inlier_count,
                "automatic_detected_count": tracking.automatic_detected_count,
                "automatic_tracked_count": tracking.automatic_tracked_count,
                "automatic_inlier_count": tracking.automatic_inlier_count,
                "image_quality_warnings": tracking.image_quality.warning_reasons,
                "dropped_landmark_ids": [],
                "excluded_from_alignment_landmark_ids": sorted(suspect_landmark_ids),
                "landmarks": transition_landmarks,
            },
            False,
        )
        quality_warning_reasons = "; ".join(tracking.image_quality.warning_reasons)
        quality_warning = (
            f", предупреждение качества: {quality_warning_reasons}"
            if tracking.image_quality.warning_reasons
            else ""
        )
        self.message.emit(
            f"Кадр {frame_index}: ручные {tracking.tracked_count}→"
            f"{tracking.inlier_count}, машинные "
            f"{tracking.automatic_tracked_count}→"
            f"{tracking.automatic_inlier_count}, RMSE "
            f"{tracking.alignment.fit_rmse_map_px:.2f} px, "
            f"сомнительных ручных: {len(suspect_landmark_ids)} (сохранены)"
            f"{quality_warning}, сопровождение активно"
        )
        self._emit_state()

    def _fail_tracking(
        self,
        reason: str,
        frame_index: int,
        *,
        partial: BeaconTrackingFailure | None = None,
    ) -> None:
        """Остановить автоматический перенос, сохранив историю и подсказки."""

        previous_pairs = list(self._correspondences)
        tracked_candidates: dict[int, Point] = {}
        previous_index = self._bound_frame_index
        if (
            partial is not None
            and partial.current_frame_points_px.shape[0] == len(previous_pairs)
            and partial.retained_mask.size == len(previous_pairs)
        ):
            for index in np.flatnonzero(partial.retained_mask):
                point_array = partial.current_frame_points_px[index]
                if np.all(np.isfinite(point_array)):
                    tracked_candidates[int(index)] = (
                        float(point_array[0]),
                        float(point_array[1]),
                    )

        self._video.pause()
        failure_landmarks: list[dict[str, object]] = []
        for index, previous_pair in enumerate(previous_pairs):
            current_point: object = None
            retained = False
            forward_backward_error: object = None
            if partial is not None:
                if partial.current_frame_points_px.shape[0] == len(previous_pairs):
                    current_point = partial.current_frame_points_px[index]
                if partial.retained_mask.size == len(previous_pairs):
                    retained = bool(partial.retained_mask[index])
                if partial.forward_backward_errors_px.size == len(previous_pairs):
                    forward_backward_error = partial.forward_backward_errors_px[index]
            failure_landmarks.append(
                {
                    "landmark_id": previous_pair.landmark_id,
                    "previous_frame_point_px": previous_pair.frame_point,
                    "current_frame_point_px": current_point,
                    "map_point_px": previous_pair.map_point,
                    "retained": retained,
                    "forward_backward_error_px": forward_backward_error,
                }
            )

        self._stop_tracking()
        self._resume_tracking_after_confirmation = True
        self._clear_visible_state(reset_bound_frame=True)
        self._bound_frame_index = frame_index
        self._suspect_landmark_ids = {pair.landmark_id for pair in previous_pairs}
        self._current_image_quality = (
            partial.image_quality if partial is not None else None
        )
        self._current_rejection_reasons = (reason,)
        for index, previous_pair in enumerate(previous_pairs):
            # Если поток не подтвердил точку, её последняя позиция всё равно
            # полезна как редактируемая подсказка. Она не участвует в расчёте
            # автоматически: весь кадр остаётся черновиком до подтверждения.
            point = tracked_candidates.get(index, previous_pair.frame_point)
            self._append_existing_observation(
                previous_pair.landmark_id,
                point,
                previous_pair.map_point,
                suspect=True,
            )

        self._correction_pending = True
        self._current_annotation_status = "rejected"
        self._remember_current_frame(status="rejected")
        self.debug_event.emit(
            "tracking_failed",
            {
                "previous_frame_index": previous_index,
                "frame_index": frame_index,
                "failure_stage": (
                    partial.failure_stage if partial is not None else "gui_precondition"
                ),
                "reason": reason,
                "input_landmark_count": len(previous_pairs),
                "retained_count": len(tracked_candidates),
                "image_quality_warnings": (
                    partial.image_quality.warning_reasons
                    if partial is not None and partial.image_quality is not None
                    else ()
                ),
                "landmarks": failure_landmarks,
            },
            True,
        )
        self.message.emit(
            f"Сопровождение остановлено: {reason}. "
            f"Автоматически перенесено: {len(tracked_candidates)}; "
            f"последних позиций сохранено: {len(previous_pairs)}. "
            "Перетащите сомнительные маяки, удалите невидимые и нажмите "
            "Compute alignment"
        )
        self._emit_state()

    def _stop_tracking(self) -> None:
        """Сбросить внутренний опорный кадр и сообщить об изменении режима."""

        was_active = self._tracking_active
        self._tracking_active = False
        self._tracking_frame_bgr = None
        self._set_markers_editable(True)
        if was_active:
            self.tracking_changed.emit(False)

    def _remove_pending_marker(self) -> None:
        if self._pending_frame_marker is not None:
            self._remove_item(self._video_view, self._pending_frame_marker)
        self._pending_frame_point = None
        self._pending_frame_marker = None
        if not self._correspondences:
            self._bound_frame_index = None

    @staticmethod
    def _remove_item(view: QGraphicsView, item: QGraphicsItem) -> None:
        scene = view.scene()
        if scene is not None and item.scene() is scene:
            scene.removeItem(item)

    def _bind_current_frame(self) -> None:
        if self._bound_frame_index is None:
            self._bound_frame_index = self._video.current_frame_index

    @staticmethod
    def _point_inside(point: Point, image: np.ndarray | None) -> bool:
        if image is None:
            return False
        x, y = point
        height, width = image.shape[:2]
        return 0.0 <= x <= width - 1.0 and 0.0 <= y <= height - 1.0

    def _view_for_viewport(self, watched: QObject) -> QGraphicsView | None:
        if watched is self._video_viewport:
            return self._video_view
        if watched is self._map_viewport:
            return self._map_view
        return None

    def _emit_state(self) -> None:
        self.state_changed.emit(
            self.pair_count,
            self.has_pending_point,
            self.overlay_available,
        )

    @staticmethod
    def _result_summary(result: ManualAlignmentResult) -> str:
        coverage = (
            f"покрытие кадра {result.frame_coverage_fraction:.1%}, "
            f"карты {result.map_coverage_fraction:.1%}"
        )
        loo_median = result.leave_one_out_median_map_px
        loo_max = result.leave_one_out_max_map_px
        if loo_median is None or loo_max is None:
            independent = "независимая проверка недоступна: нужны ≥5 пар"
        else:
            independent = (
                f"проверка без одной точки: медиана {loo_median:.2f} px, "
                f"максимум {loo_max:.2f} px"
            )
        return (
            f"Совмещение по {result.frame_points_px.shape[0]} парам | "
            f"подгонка RMSE {result.fit_rmse_map_px:.2f} px | "
            f"{independent} | {coverage}"
        )
