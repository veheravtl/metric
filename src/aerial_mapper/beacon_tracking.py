"""Сопровождение вручную размеченных наземных маяков между кадрами.

Оператор один раз задаёт пары ``точка кадра ↔ точка карты``. Между соседними
кадрами меняется только положение точки в видеопотоке, а её координата на
неподвижной карте остаётся прежней. Модуль отслеживает видеоточки пирамидальным
оптическим потоком Лукаса—Канаде и заново оценивает гомографию текущего кадра к
карте. Qt здесь намеренно отсутствует: математическое поведение проверяется
отдельно от интерфейса.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from aerial_mapper.frame_quality import FrameImageQuality, assess_frame_image_quality
from aerial_mapper.manual_alignment import (
    Homography,
    ManualAlignmentFailure,
    ManualAlignmentResult,
    estimate_manual_alignment,
)

FloatPoints = NDArray[np.float64]
BooleanMask = NDArray[np.bool_]


class BeaconTrackingFailure(RuntimeError):
    """Надёжных маяков недостаточно для нового совмещения.

    Ошибка может нести частичный результат оптического потока. Это не готовое
    совмещение, а подсказка оператору: сохранившиеся наблюдения можно поправить
    и дополнить вместо повторной разметки всех пар.
    """

    def __init__(
        self,
        message: str,
        *,
        failure_stage: str = "input_validation",
        current_frame_points_px: NDArray[np.floating] | None = None,
        retained_mask: NDArray[np.bool_] | None = None,
        forward_backward_errors_px: NDArray[np.floating] | None = None,
        image_quality: FrameImageQuality | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_stage = failure_stage
        self.image_quality = image_quality
        self.forward_backward_errors_px = (
            np.asarray(forward_backward_errors_px, dtype=np.float64).reshape(-1)
            if forward_backward_errors_px is not None
            else np.empty((0,), dtype=np.float64)
        )
        if current_frame_points_px is None or retained_mask is None:
            self.current_frame_points_px = np.empty((0, 2), dtype=np.float64)
            self.retained_mask = np.empty((0,), dtype=bool)
            return

        points = np.asarray(current_frame_points_px, dtype=np.float64)
        mask = np.asarray(retained_mask, dtype=bool).reshape(-1)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] != mask.size:
            raise ValueError("Частичный результат сопровождения имеет неверную форму")
        if self.forward_backward_errors_px.size not in (0, points.shape[0]):
            raise ValueError(
                "Ошибки обратной проверки должны соответствовать числу маяков"
            )
        self.current_frame_points_px = points.copy()
        self.retained_mask = mask.copy()

    @property
    def retained_count(self) -> int:
        """Число наблюдений, которые можно предложить оператору."""

        return int(np.count_nonzero(self.retained_mask))


@dataclass(frozen=True)
class BeaconTrackingResult:
    """Результат переноса маяков с предыдущего кадра на текущий.

    Маски имеют длину, равную числу входных маяков. ``flow_valid_mask``
    показывает точки, прошедшие прямую и обратную проверку оптического потока.
    ``homography_inlier_mask`` дополнительно оставляет только точки,
    согласующиеся с одной плоской проективной моделью.

    В ``current_frame_points_px`` для потерянных точек хранится ``NaN``. Это
    не позволяет случайно принять старую координату за новое наблюдение.
    """

    current_frame_points_px: FloatPoints
    map_points_px: FloatPoints
    flow_valid_mask: BooleanMask
    homography_inlier_mask: BooleanMask
    forward_backward_errors_px: NDArray[np.float64]
    alignment: ManualAlignmentResult

    @property
    def tracked_count(self) -> int:
        """Число точек, прошедших проверку оптического потока."""

        return int(np.count_nonzero(self.flow_valid_mask))

    @property
    def inlier_count(self) -> int:
        """Число маяков, использованных в итоговой гомографии."""

        return int(np.count_nonzero(self.homography_inlier_mask))

    @property
    def accepted_indices(self) -> NDArray[np.int64]:
        """Индексы маяков, использованных для итоговой гомографии."""

        return np.flatnonzero(self.homography_inlier_mask).astype(np.int64)

    @property
    def accepted_forward_backward_max_px(self) -> float:
        """Худшая ошибка обратного прохода среди принятых маяков."""

        return float(
            np.max(self.forward_backward_errors_px[self.homography_inlier_mask])
        )


@dataclass(frozen=True)
class HybridTrackingResult:
    """Совмещение, поддержанное ручными и скрытыми машинными точками.

    Поля без префикса ``automatic`` относятся только к ручным маякам и потому
    совместимы с существующим интерфейсом. Машинные массивы описывают переход
    от прошлого кадра к текущему и сохраняются для диагностики, но не рисуются
    как сотни новых операторских маркеров.
    """

    current_frame_points_px: FloatPoints
    map_points_px: FloatPoints
    flow_valid_mask: BooleanMask
    homography_inlier_mask: BooleanMask
    forward_backward_errors_px: NDArray[np.float64]
    automatic_previous_frame_points_px: FloatPoints
    automatic_current_frame_points_px: FloatPoints
    automatic_map_points_px: FloatPoints
    automatic_flow_valid_mask: BooleanMask
    automatic_homography_inlier_mask: BooleanMask
    automatic_forward_backward_errors_px: NDArray[np.float64]
    alignment: ManualAlignmentResult
    image_quality: FrameImageQuality

    @property
    def tracked_count(self) -> int:
        """Число ручных маяков, прошедших проверку потока."""

        return int(np.count_nonzero(self.flow_valid_mask))

    @property
    def inlier_count(self) -> int:
        """Число ручных маяков, согласных с итоговой геометрией."""

        return int(np.count_nonzero(self.homography_inlier_mask))

    @property
    def automatic_detected_count(self) -> int:
        """Число машинных опор, найденных на исходном кадре."""

        return int(self.automatic_previous_frame_points_px.shape[0])

    @property
    def automatic_tracked_count(self) -> int:
        """Число машинных опор, прошедших проверку потока."""

        return int(np.count_nonzero(self.automatic_flow_valid_mask))

    @property
    def automatic_inlier_count(self) -> int:
        """Число машинных опор, использованных в гомографии."""

        return int(np.count_nonzero(self.automatic_homography_inlier_mask))

    @property
    def total_inlier_count(self) -> int:
        """Общее число ручных и машинных опор итоговой гомографии."""

        return self.inlier_count + self.automatic_inlier_count


def track_beacons_to_map(
    previous_frame_bgr: NDArray[np.uint8],
    current_frame_bgr: NDArray[np.uint8],
    previous_frame_points_px: NDArray[np.floating] | list[tuple[float, float]],
    map_points_px: NDArray[np.floating] | list[tuple[float, float]],
    *,
    map_size_pixels: tuple[int, int],
    max_forward_backward_error_px: float = 1.5,
    ransac_reprojection_threshold_map_px: float = 4.0,
) -> BeaconTrackingResult:
    """Перенести маяки на соседний кадр и оценить новое совмещение с картой.

    Проверка «вперёд–назад» работает так: точка переносится с предыдущего кадра
    на текущий, затем найденная позиция переносится обратно. Если возвращённая
    позиция отличается от исходной больше чем на
    ``max_forward_backward_error_px``, наблюдение считается ненадёжным.

    После проверки потока применяется устойчивый поиск гомографии. Он нужен,
    потому что отдельная точка может уверенно проследиться по похожей текстуре,
    но двигаться не так, как остальные точки плоской поверхности земли.
    """

    previous_gray = _as_gray_u8(previous_frame_bgr, name="Предыдущий кадр")
    current_gray = _as_gray_u8(current_frame_bgr, name="Текущий кадр")
    if previous_gray.shape != current_gray.shape:
        raise BeaconTrackingFailure(
            "Размер кадра изменился; сопровождение требует одинаковой геометрии"
        )
    if max_forward_backward_error_px <= 0.0:
        raise ValueError("Порог ошибки вперёд–назад должен быть положительным")
    if ransac_reprojection_threshold_map_px <= 0.0:
        raise ValueError("Порог RANSAC должен быть положительным")

    previous_points = _validated_points(
        previous_frame_points_px, name="Точки предыдущего кадра"
    )
    map_points = _validated_points(map_points_px, name="Точки карты")
    if previous_points.shape != map_points.shape:
        raise BeaconTrackingFailure(
            "Число видеомаяков и соответствующих точек карты должно совпадать"
        )
    _validate_points_inside(previous_points, previous_gray.shape)

    lk_parameters = {
        "winSize": (31, 31),
        "maxLevel": 3,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
        "minEigThreshold": 1e-4,
    }
    previous_cv = previous_points.astype(np.float32).reshape(-1, 1, 2)
    current_cv, forward_status, _forward_error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_cv,
        None,
        **lk_parameters,
    )
    if current_cv is None or forward_status is None:
        raise BeaconTrackingFailure(
            "Оптический поток не нашёл ни одного маяка",
            failure_stage="forward_flow",
        )

    backward_cv, backward_status, _backward_error = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        current_cv,
        None,
        **lk_parameters,
    )
    if backward_cv is None or backward_status is None:
        raise BeaconTrackingFailure(
            "Не выполнена обратная проверка маяков",
            failure_stage="backward_flow",
        )

    current_points = current_cv.reshape(-1, 2).astype(np.float64)
    backward_points = backward_cv.reshape(-1, 2).astype(np.float64)
    forward_backward_errors = np.linalg.norm(backward_points - previous_points, axis=1)

    height, width = current_gray.shape
    inside_current = (
        (current_points[:, 0] >= 0.0)
        & (current_points[:, 0] <= width - 1.0)
        & (current_points[:, 1] >= 0.0)
        & (current_points[:, 1] <= height - 1.0)
    )
    finite = np.all(np.isfinite(current_points), axis=1) & np.isfinite(
        forward_backward_errors
    )
    flow_valid = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & inside_current
        & finite
        & (forward_backward_errors <= max_forward_backward_error_px)
    )
    tracked_count = int(np.count_nonzero(flow_valid))
    if tracked_count < 4:
        raise BeaconTrackingFailure(
            "Сопровождение потеряно: проверку оптического потока прошли "
            f"{tracked_count} из {previous_points.shape[0]} маяков, нужны минимум 4",
            failure_stage="flow_validation",
            current_frame_points_px=current_points,
            retained_mask=flow_valid,
            forward_backward_errors_px=forward_backward_errors,
        )

    valid_indices = np.flatnonzero(flow_valid)
    robust_method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    homography, robust_mask = cv2.findHomography(
        current_points[valid_indices],
        map_points[valid_indices],
        method=robust_method,
        ransacReprojThreshold=ransac_reprojection_threshold_map_px,
        maxIters=10_000,
        confidence=0.999,
    )
    if homography is None or robust_mask is None:
        raise BeaconTrackingFailure(
            "Прослеженные маяки не задают согласованную гомографию",
            failure_stage="homography_estimation",
            current_frame_points_px=current_points,
            retained_mask=flow_valid,
            forward_backward_errors_px=forward_backward_errors,
        )

    homography_inliers = np.zeros(previous_points.shape[0], dtype=bool)
    homography_inliers[valid_indices] = robust_mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(homography_inliers))
    if inlier_count < 4:
        raise BeaconTrackingFailure(
            "Сопровождение потеряно: общей геометрии соответствуют "
            f"{inlier_count} маяков, нужны минимум 4",
            failure_stage="homography_consensus",
            current_frame_points_px=current_points,
            retained_mask=homography_inliers,
            forward_backward_errors_px=forward_backward_errors,
        )

    try:
        alignment = estimate_manual_alignment(
            current_points[homography_inliers],
            map_points[homography_inliers],
            frame_size_pixels=(width, height),
            map_size_pixels=map_size_pixels,
        )
    except ManualAlignmentFailure as error:
        raise BeaconTrackingFailure(
            str(error),
            failure_stage="alignment_estimation",
            current_frame_points_px=current_points,
            retained_mask=homography_inliers,
            forward_backward_errors_px=forward_backward_errors,
        ) from error

    reported_points = current_points.copy()
    reported_points[~flow_valid] = np.nan
    return BeaconTrackingResult(
        current_frame_points_px=reported_points,
        map_points_px=map_points.copy(),
        flow_valid_mask=flow_valid,
        homography_inlier_mask=homography_inliers,
        forward_backward_errors_px=forward_backward_errors,
        alignment=alignment,
    )


def track_beacons_with_machine_support_to_map(
    previous_frame_bgr: NDArray[np.uint8],
    current_frame_bgr: NDArray[np.uint8],
    previous_frame_points_px: NDArray[np.floating] | list[tuple[float, float]],
    map_points_px: NDArray[np.floating] | list[tuple[float, float]],
    *,
    previous_alignment: ManualAlignmentResult,
    map_size_pixels: tuple[int, int],
    max_automatic_points: int = 160,
    max_forward_backward_error_px: float = 1.5,
    ransac_reprojection_threshold_map_px: float = 4.0,
) -> HybridTrackingResult:
    """Поддержать ручные маяки автоматически найденными углами изображения.

    Машинные точки ищутся внутри выпуклой оболочки ручных маяков, а координаты
    карты наследуют через прошлую гомографию. Они улучшают непрерывность, но не
    являются независимым наземным контролем: для принятия всё равно нужны
    минимум три согласованных ручных маяка.
    """

    if max_automatic_points < 0:
        raise ValueError("Число машинных точек не может быть отрицательным")
    manual_previous = _validated_points(
        previous_frame_points_px, name="Ручные точки предыдущего кадра"
    )
    manual_map = _validated_points(map_points_px, name="Ручные точки карты")
    if manual_previous.shape != manual_map.shape:
        raise BeaconTrackingFailure(
            "Число ручных видеомаяков и точек карты должно совпадать"
        )

    try:
        image_quality = assess_frame_image_quality(
            previous_frame_bgr, current_frame_bgr
        )
    except ValueError as error:
        raise BeaconTrackingFailure(
            str(error), failure_stage="frame_quality"
        ) from error
    if image_quality.is_rejected:
        point_count = manual_previous.shape[0]
        raise BeaconTrackingFailure(
            "Сопровождение потеряно: " + "; ".join(image_quality.rejection_reasons),
            failure_stage="frame_quality",
            current_frame_points_px=np.full((point_count, 2), np.nan),
            retained_mask=np.zeros(point_count, dtype=bool),
            forward_backward_errors_px=np.full(point_count, np.inf),
            image_quality=image_quality,
        )

    automatic_previous, automatic_map = _automatic_support_for_anchor(
        previous_frame_bgr,
        manual_previous,
        previous_alignment,
        map_size_pixels=map_size_pixels,
        maximum_count=max_automatic_points,
    )
    combined_previous = np.vstack((manual_previous, automatic_previous))
    combined_map = np.vstack((manual_map, automatic_map))
    try:
        combined = track_beacons_to_map(
            previous_frame_bgr,
            current_frame_bgr,
            combined_previous,
            combined_map,
            map_size_pixels=map_size_pixels,
            max_forward_backward_error_px=max_forward_backward_error_px,
            ransac_reprojection_threshold_map_px=ransac_reprojection_threshold_map_px,
        )
    except BeaconTrackingFailure as error:
        raise _manual_part_of_failure(
            error,
            manual_count=manual_previous.shape[0],
            combined_count=combined_previous.shape[0],
            image_quality=image_quality,
        ) from error

    return _split_hybrid_result(
        combined,
        manual_map=manual_map,
        automatic_previous=automatic_previous,
        automatic_map=automatic_map,
        image_quality=image_quality,
    )


def _manual_part_of_failure(
    error: BeaconTrackingFailure,
    *,
    manual_count: int,
    combined_count: int,
    image_quality: FrameImageQuality,
) -> BeaconTrackingFailure:
    """Не отдавать GUI внутренние машинные точки при неудачном переходе."""

    current = error.current_frame_points_px
    retained = error.retained_mask
    errors = error.forward_backward_errors_px
    current = (
        current[:manual_count]
        if current.shape[0] == combined_count
        else np.full((manual_count, 2), np.nan)
    )
    retained = (
        retained[:manual_count]
        if retained.size == combined_count
        else np.zeros(manual_count, dtype=bool)
    )
    errors = (
        errors[:manual_count]
        if errors.size == combined_count
        else np.full(manual_count, np.inf)
    )
    return BeaconTrackingFailure(
        str(error),
        failure_stage=error.failure_stage,
        current_frame_points_px=current,
        retained_mask=retained,
        forward_backward_errors_px=errors,
        image_quality=image_quality,
    )


def _split_hybrid_result(
    combined: BeaconTrackingResult,
    *,
    manual_map: FloatPoints,
    automatic_previous: FloatPoints,
    automatic_map: FloatPoints,
    image_quality: FrameImageQuality,
) -> HybridTrackingResult:
    """Проверить ручной контроль и разделить общие массивы по происхождению."""

    manual_count = manual_map.shape[0]
    manual_inliers = int(
        np.count_nonzero(combined.homography_inlier_mask[:manual_count])
    )
    total_inliers = int(np.count_nonzero(combined.homography_inlier_mask))
    # Четыре ручных согласованных маяка достаточны сами по себе. Если один
    # потерян, машина может поддержать переход, но только при восьми общих
    # опорах. Два ручных маяка не дают достаточно независимого контроля.
    if manual_inliers < 4 and (manual_inliers < 3 or total_inliers < 8):
        raise BeaconTrackingFailure(
            "Сопровождение потеряно: машинные точки не получили достаточного "
            f"ручного подтверждения ({manual_inliers} ручных и "
            f"{total_inliers} общих опор)",
            failure_stage="manual_consensus",
            current_frame_points_px=combined.current_frame_points_px[:manual_count],
            retained_mask=combined.flow_valid_mask[:manual_count],
            forward_backward_errors_px=(
                combined.forward_backward_errors_px[:manual_count]
            ),
            image_quality=image_quality,
        )

    return HybridTrackingResult(
        current_frame_points_px=combined.current_frame_points_px[:manual_count],
        map_points_px=manual_map.copy(),
        flow_valid_mask=combined.flow_valid_mask[:manual_count],
        homography_inlier_mask=combined.homography_inlier_mask[:manual_count],
        forward_backward_errors_px=combined.forward_backward_errors_px[:manual_count],
        automatic_previous_frame_points_px=automatic_previous,
        automatic_current_frame_points_px=combined.current_frame_points_px[
            manual_count:
        ],
        automatic_map_points_px=automatic_map,
        automatic_flow_valid_mask=combined.flow_valid_mask[manual_count:],
        automatic_homography_inlier_mask=combined.homography_inlier_mask[manual_count:],
        automatic_forward_backward_errors_px=combined.forward_backward_errors_px[
            manual_count:
        ],
        alignment=combined.alignment,
        image_quality=image_quality,
    )


def _automatic_support_for_anchor(
    previous_frame_bgr: NDArray[np.uint8],
    manual_points_px: FloatPoints,
    previous_alignment: ManualAlignmentResult,
    *,
    map_size_pixels: tuple[int, int],
    maximum_count: int,
) -> tuple[FloatPoints, FloatPoints]:
    """Найти машинные точки и унаследовать их координаты карты от опоры."""

    previous_gray = _as_gray_u8(previous_frame_bgr, name="Предыдущий кадр")
    automatic_previous = _detect_automatic_support_points(
        previous_gray, manual_points_px, maximum_count=maximum_count
    )
    automatic_map = _project_points(
        automatic_previous, previous_alignment.homography_frame_to_map
    )
    map_width, map_height = map_size_pixels
    inside = (
        np.all(np.isfinite(automatic_map), axis=1)
        & (automatic_map[:, 0] >= 0.0)
        & (automatic_map[:, 0] <= map_width - 1.0)
        & (automatic_map[:, 1] >= 0.0)
        & (automatic_map[:, 1] <= map_height - 1.0)
    )
    return automatic_previous[inside], automatic_map[inside]


def _detect_automatic_support_points(
    previous_gray: NDArray[np.uint8],
    manual_points_px: FloatPoints,
    *,
    maximum_count: int,
) -> FloatPoints:
    """Найти воспроизводимые углы внутри области ручных маяков."""

    if maximum_count == 0:
        return np.empty((0, 2), dtype=np.float64)
    mask = np.zeros(previous_gray.shape, dtype=np.uint8)
    hull = cv2.convexHull(manual_points_px.astype(np.float32)).reshape(-1, 2)
    cv2.fillConvexPoly(mask, np.rint(hull).astype(np.int32), 255)
    # Расширение не обрезает локальные окна признаков у границы ручной области,
    # но всё ещё не открывает для машинного поиска весь кадр.
    mask = cv2.dilate(mask, np.ones((21, 21), dtype=np.uint8))
    for x, y in manual_points_px:
        cv2.circle(mask, (round(float(x)), round(float(y))), 12, 0, -1)

    detected = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=maximum_count,
        qualityLevel=0.01,
        minDistance=12.0,
        mask=mask,
        blockSize=7,
        useHarrisDetector=False,
    )
    if detected is None:
        return np.empty((0, 2), dtype=np.float64)
    return detected.reshape(-1, 2).astype(np.float64)


def _project_points(points: FloatPoints, homography: Homography) -> FloatPoints:
    """Перенести массив точек (N, 2) проективной матрицей 3 × 3."""

    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise BeaconTrackingFailure("Предыдущая гомография машинных точек неверна")
    projected = cv2.perspectiveTransform(points.reshape(1, -1, 2), matrix)
    return projected.reshape(-1, 2).astype(np.float64)


def _as_gray_u8(image: NDArray[np.uint8], *, name: str) -> NDArray[np.uint8]:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise BeaconTrackingFailure(f"{name} должен иметь тип uint8")
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim == 3 and array.shape[2] == 4:
        gray = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    else:
        raise BeaconTrackingFailure(
            f"{name} должен быть серым, BGR или BGRA-изображением"
        )
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        raise BeaconTrackingFailure(f"{name} слишком мал для сопровождения")
    return np.ascontiguousarray(gray)


def _validated_points(
    values: NDArray[np.floating] | list[tuple[float, float]], *, name: str
) -> FloatPoints:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise BeaconTrackingFailure(f"{name} должны иметь форму (N, 2)")
    if points.shape[0] < 4:
        raise BeaconTrackingFailure(f"{name}: нужны минимум четыре маяка")
    if not np.all(np.isfinite(points)):
        raise BeaconTrackingFailure(f"{name} содержат NaN или бесконечность")
    return points


def _validate_points_inside(points: FloatPoints, image_shape: tuple[int, int]) -> None:
    height, width = image_shape
    outside = (
        (points[:, 0] < 0.0)
        | (points[:, 0] > width - 1.0)
        | (points[:, 1] < 0.0)
        | (points[:, 1] > height - 1.0)
    )
    if np.any(outside):
        raise BeaconTrackingFailure("Хотя бы один маяк находится вне предыдущего кадра")
