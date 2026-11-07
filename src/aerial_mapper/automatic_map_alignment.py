"""Экспериментальное автоматическое совмещение кадра с растровой картой.

Задача сложнее обычного поиска одного объекта: спутниковая карта и камера
дрона отличаются временем съёмки, цветом, разрешением и перспективой. Поэтому
модуль не возвращает матрицу только потому, что RANSAC нашёл минимальные четыре
пары. После поиска признаков отдельно проверяются распределение подтверждённых
точек и форма проекции всего кадра на карту.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np
from numpy.typing import NDArray

from aerial_mapper.synthetic import FloatPoints, Homography, RgbImage


class AutomaticMapAlignmentFailure(RuntimeError):
    """Ни одна найденная гипотеза не прошла диагностические проверки."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: AutomaticMapAlignmentDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class AutomaticMapAlignmentConfig:
    """Параметры воспроизводимого экспериментального алгоритма.

    Несколько ``ratio_thresholds`` не означают перебор до первого успеха.
    Для каждого порога строится отдельная геометрическая гипотеза, после чего
    все они проходят одинаковые проверки. Это уменьшает зависимость результата
    от единственного произвольно выбранного уровня строгости дескрипторов.

    Поворот измеряется в градусах, а масштаб — как двоичный логарифм отношения
    размеров SIFT-точек карты и кадра. Поэтому допуск 0,5 по логарифму означает
    примерно множитель √2 в любую сторону. Радиус удаления дублей выражен в
    пикселях уменьшенной копии для анализа. Вес голоса равен
    ``(1 - mean_ratio) ** match_quality_vote_exponent``, где
    ``mean_ratio`` — средняя неоднозначность пары в двух направлениях.

    Пороги геометрии получены как стартовые защитные ограничения по локальному
    HKairport-опыту. Они не являются доказанными эксплуатационными допусками и
    должны позднее проверяться на независимых картах и пролётах.
    """

    analysis_maximum_side_pixels: int = 1_200
    maximum_features_per_image: int = 15_000
    sift_contrast_threshold: float = 0.01
    sift_edge_threshold: float = 15.0
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (8, 8)
    ratio_thresholds: tuple[float, ...] = (0.76, 0.80, 0.84, 0.88)
    match_orientation_tolerance_degrees: float = 20.0
    match_log2_scale_tolerance: float = 0.5
    match_spatial_deduplication_radius_px: float = 3.0
    match_quality_vote_exponent: float = 2.0
    magsac_reprojection_threshold_px: float = 4.0
    magsac_max_iterations: int = 20_000
    magsac_confidence: float = 0.999
    minimum_mutual_matches: int = 8
    minimum_inliers: int = 8
    minimum_inlier_ratio: float = 0.04
    minimum_frame_coverage_fraction: float = 0.04
    minimum_projected_area_fraction: float = 0.02
    maximum_projected_area_fraction: float = 0.80
    minimum_map_overlap_fraction: float = 0.50
    maximum_projected_edge_ratio: float = 5.0
    random_seed: int = 0

    def validate(self) -> None:
        """Проверить конфигурацию до дорогого извлечения признаков."""

        if self.analysis_maximum_side_pixels < 64:
            raise ValueError("Сторона анализа должна быть не меньше 64 пикселей")
        if self.maximum_features_per_image < 4:
            raise ValueError("Для гомографии нужны минимум четыре признака")
        if self.sift_contrast_threshold <= 0 or self.sift_edge_threshold <= 0:
            raise ValueError("Пороги SIFT должны быть положительными")
        if self.clahe_clip_limit <= 0 or min(self.clahe_grid_size) <= 0:
            raise ValueError("Параметры CLAHE должны быть положительными")
        if not self.ratio_thresholds or any(
            not 0.0 < value < 1.0 for value in self.ratio_thresholds
        ):
            raise ValueError("Пороги ratio test должны находиться между 0 и 1")
        if tuple(sorted(set(self.ratio_thresholds))) != self.ratio_thresholds:
            raise ValueError("Пороги ratio test должны быть уникальны и отсортированы")
        if not 0.0 < self.match_orientation_tolerance_degrees <= 180.0:
            raise ValueError(
                "Допуск согласованности поворота должен быть между 0 и 180 градусами"
            )
        if self.match_log2_scale_tolerance <= 0.0:
            raise ValueError(
                "Допуск согласованности масштаба должен быть положительным"
            )
        if self.match_spatial_deduplication_radius_px < 0.0:
            raise ValueError("Радиус удаления дублей не может быть отрицательным")
        if self.match_quality_vote_exponent <= 0.0:
            raise ValueError("Степень веса голосования должна быть положительной")

        if self.magsac_reprojection_threshold_px <= 0:
            raise ValueError("Порог MAGSAC должен быть положительным")
        if self.magsac_max_iterations <= 0:
            raise ValueError("Число итераций MAGSAC должно быть положительным")
        if not 0.0 < self.magsac_confidence < 1.0:
            raise ValueError("Уверенность MAGSAC должна находиться между 0 и 1")
        if self.minimum_mutual_matches < 4 or self.minimum_inliers < 4:
            raise ValueError("Для гомографии нужны минимум четыре пары")
        for name, value in (
            ("minimum_inlier_ratio", self.minimum_inlier_ratio),
            ("minimum_frame_coverage_fraction", self.minimum_frame_coverage_fraction),
            ("minimum_projected_area_fraction", self.minimum_projected_area_fraction),
            ("maximum_projected_area_fraction", self.maximum_projected_area_fraction),
            ("minimum_map_overlap_fraction", self.minimum_map_overlap_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} должен находиться между 0 и 1")
        if self.minimum_projected_area_fraction >= self.maximum_projected_area_fraction:
            raise ValueError("Минимальная площадь должна быть меньше максимальной")
        if self.maximum_projected_edge_ratio < 1.0:
            raise ValueError("Отношение рёбер не может быть меньше единицы")


@dataclass(frozen=True)
class AutomaticMapAlignmentResult:
    """Принятая гомография и наблюдаемая диагностика без метрической истины.

    Все точки и матрица относятся к исходным размерам переданных изображений,
    хотя поиск ради скорости выполняется на уменьшенных копиях. Ошибки
    репроекции выражены в пикселях исходной карты.
    """

    homography_frame_to_reference: Homography
    frame_points_px: FloatPoints
    reference_points_px: FloatPoints
    inlier_mask: NDArray[np.bool_]
    selected_ratio_threshold: float
    tested_hypothesis_count: int
    reference_keypoint_count: int
    frame_keypoint_count: int
    frame_coverage_fraction: float
    projected_frame_corners_reference_px: FloatPoints
    projected_area_fraction: float
    map_overlap_fraction: float
    projected_edge_ratio: float
    inlier_reprojection_median_reference_px: float
    inlier_reprojection_p95_reference_px: float
    inlier_reprojection_max_reference_px: float
    processing_time_seconds: float
    unfiltered_mutual_match_count: int = 0
    scale_orientation_center_degrees: float | None = None
    scale_orientation_center_log2_scale: float | None = None
    scale_orientation_vote_score: float | None = None

    @property
    def mutual_match_count(self) -> int:
        """Число взаимно однозначных пар выбранной гипотезы."""

        return int(self.frame_points_px.shape[0])

    @property
    def inlier_count(self) -> int:
        """Число пар, согласованных с итоговой гомографией."""

        return int(np.count_nonzero(self.inlier_mask))

    @property
    def inlier_ratio(self) -> float:
        """Доля геометрически согласованных пар среди взаимных совпадений."""

        return self.inlier_count / self.mutual_match_count


@dataclass(frozen=True)
class AutomaticMapAlignmentDiagnostics:
    """Данные неудачной попытки, достаточные для последующего разбора.

    ``best_candidate`` намеренно не называется результатом: эта гипотеза не
    прошла защитные проверки и не должна использоваться для измерений. Она
    сохраняется только для сравнения с ручной разметкой и поиска этапа, на
    котором автоматический алгоритм ошибся.
    """

    reference_keypoint_count: int
    frame_keypoint_count: int
    ratio_match_counts: tuple[tuple[float, int], ...]
    tested_hypothesis_count: int
    best_candidate: AutomaticMapAlignmentResult | None
    rejection_reasons: tuple[str, ...]
    processing_time_seconds: float
    filtered_match_counts: tuple[tuple[float, int], ...] = ()


@dataclass(frozen=True)
class _ResizeResult:
    image_rgb: RgbImage
    scale_x: float
    scale_y: float

    @property
    def matrix_original_to_analysis(self) -> Homography:
        return np.asarray(
            [[self.scale_x, 0.0, 0.0], [0.0, self.scale_y, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class _MutualMatch:
    """Взаимная дескрипторная пара и качество проверки Лоу."""

    dmatch: cv2.DMatch
    forward_ratio: float
    reverse_ratio: float

    @property
    def mean_ratio(self) -> float:
        """Средняя неоднозначность пары в двух направлениях; меньше лучше."""

        return (self.forward_ratio + self.reverse_ratio) / 2.0


@dataclass(frozen=True)
class _MatchFilterResult:
    """Пары доминирующего кластера масштаба и поворота."""

    matches: tuple[_MutualMatch, ...]
    center_degrees: float
    center_log2_scale: float
    vote_score: float
    unfiltered_match_count: int


@dataclass(frozen=True)
class _Candidate:
    homography_analysis: Homography
    frame_points_analysis: FloatPoints
    reference_points_analysis: FloatPoints
    inlier_mask: NDArray[np.bool_]
    ratio_threshold: float
    frame_coverage_fraction: float
    projected_area_fraction: float
    map_overlap_fraction: float
    projected_edge_ratio: float
    is_convex: bool
    reprojection_p95_px: float
    rejection_reasons: tuple[str, ...]
    unfiltered_mutual_match_count: int
    scale_orientation_center_degrees: float
    scale_orientation_center_log2_scale: float
    scale_orientation_vote_score: float

    @property
    def mutual_match_count(self) -> int:
        return int(self.frame_points_analysis.shape[0])

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def inlier_ratio(self) -> float:
        return self.inlier_count / self.mutual_match_count

    @property
    def selection_score(self) -> float:
        """Совместно поощрить число подтверждений и чистоту набора пар."""

        return self.inlier_count * self.inlier_ratio


def align_frame_to_map_automatically(
    reference_rgb: RgbImage,
    frame_rgb: RgbImage,
    *,
    config: AutomaticMapAlignmentConfig | None = None,
) -> AutomaticMapAlignmentResult:
    """Найти кадр на карте либо явно отказаться от сомнительного ответа.

    Этапы алгоритма:

    1. уменьшить изображения только для анализа, сохранив матрицы масштаба;
    2. усилить локальный контраст CLAHE и извлечь SIFT-признаки;
    3. оставить пары, прошедшие ratio test Лоу в обоих направлениях;
    4. выбрать согласованный кластер локальных масштабов и поворотов и удалить
       почти совпадающие пары;
    5. для нескольких уровней строгости оценить гомографию USAC_MAGSAC;
    6. проверить число и распределение inlier-точек, выпуклость, площадь,
       пересечение с картой и отсутствие экстремально растянутых рёбер;
    7. вернуть лучшую прошедшую гипотезу в координатах исходных изображений.

    Эти проверки обнаруживают наблюдавшиеся схлопывания в линию и огромные
    проекции. Они не способны доказать, что правдоподобный участок карты выбран
    верно среди нескольких внешне одинаковых участков.
    """

    effective_config = config or AutomaticMapAlignmentConfig()
    effective_config.validate()
    _validate_rgb_image(reference_rgb, name="Карта")
    _validate_rgb_image(frame_rgb, name="Кадр")
    if not hasattr(cv2, "USAC_MAGSAC"):
        raise AutomaticMapAlignmentFailure(
            "Установленная сборка OpenCV не содержит USAC_MAGSAC"
        )

    started_at = perf_counter()
    reference = _resize_for_analysis(
        reference_rgb, effective_config.analysis_maximum_side_pixels
    )
    frame = _resize_for_analysis(
        frame_rgb, effective_config.analysis_maximum_side_pixels
    )

    clahe = cv2.createCLAHE(
        clipLimit=effective_config.clahe_clip_limit,
        tileGridSize=effective_config.clahe_grid_size,
    )
    reference_gray = clahe.apply(cv2.cvtColor(reference.image_rgb, cv2.COLOR_RGB2GRAY))
    frame_gray = clahe.apply(cv2.cvtColor(frame.image_rgb, cv2.COLOR_RGB2GRAY))
    sift = cv2.SIFT_create(
        nfeatures=effective_config.maximum_features_per_image,
        contrastThreshold=effective_config.sift_contrast_threshold,
        edgeThreshold=effective_config.sift_edge_threshold,
    )
    reference_keypoints, reference_descriptors = sift.detectAndCompute(
        reference_gray, None
    )
    frame_keypoints, frame_descriptors = sift.detectAndCompute(frame_gray, None)
    if reference_descriptors is None or len(reference_keypoints) < 4:
        raise AutomaticMapAlignmentFailure(
            "На карте найдено недостаточно SIFT-признаков"
        )
    if frame_descriptors is None or len(frame_keypoints) < 4:
        raise AutomaticMapAlignmentFailure(
            "На кадре найдено недостаточно SIFT-признаков"
        )

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    frame_to_reference = matcher.knnMatch(frame_descriptors, reference_descriptors, k=2)
    reference_to_frame = matcher.knnMatch(reference_descriptors, frame_descriptors, k=2)

    candidates: list[_Candidate] = []
    ratio_match_counts: list[tuple[float, int]] = []
    filtered_match_counts: list[tuple[float, int]] = []
    for ratio_threshold in effective_config.ratio_thresholds:
        matches = _mutual_ratio_matches(
            frame_to_reference,
            reference_to_frame,
            ratio_threshold=ratio_threshold,
        )
        ratio_match_counts.append((ratio_threshold, len(matches)))
        if len(matches) < effective_config.minimum_mutual_matches:
            filtered_match_counts.append((ratio_threshold, 0))
            continue
        filter_result = _filter_matches_by_scale_and_orientation(
            matches,
            frame_keypoints=frame_keypoints,
            reference_keypoints=reference_keypoints,
            config=effective_config,
        )
        filtered_match_counts.append((ratio_threshold, len(filter_result.matches)))
        if len(filter_result.matches) < effective_config.minimum_mutual_matches:
            continue
        candidate = _estimate_candidate(
            filter_result.matches,
            frame_keypoints=frame_keypoints,
            reference_keypoints=reference_keypoints,
            frame_size=(frame.image_rgb.shape[1], frame.image_rgb.shape[0]),
            reference_size=(
                reference.image_rgb.shape[1],
                reference.image_rgb.shape[0],
            ),
            config=effective_config,
            ratio_threshold=ratio_threshold,
            filter_result=filter_result,
        )
        if candidate is not None:
            candidates.append(candidate)

    accepted = [
        candidate for candidate in candidates if not candidate.rejection_reasons
    ]
    if not accepted:
        best = max(
            candidates,
            key=lambda candidate: (candidate.inlier_count, candidate.selection_score),
            default=None,
        )
        elapsed = perf_counter() - started_at
        best_result = (
            _candidate_to_result(
                best,
                reference=reference,
                frame=frame,
                reference_rgb=reference_rgb,
                frame_rgb=frame_rgb,
                reference_keypoint_count=len(reference_keypoints),
                frame_keypoint_count=len(frame_keypoints),
                tested_hypothesis_count=len(candidates),
                processing_time_seconds=elapsed,
            )
            if best is not None
            else None
        )
        raise AutomaticMapAlignmentFailure(
            _failure_summary(
                candidates,
                ratio_match_counts=ratio_match_counts,
                filtered_match_counts=filtered_match_counts,
                minimum_mutual_matches=effective_config.minimum_mutual_matches,
            ),
            diagnostics=AutomaticMapAlignmentDiagnostics(
                reference_keypoint_count=len(reference_keypoints),
                frame_keypoint_count=len(frame_keypoints),
                ratio_match_counts=tuple(ratio_match_counts),
                tested_hypothesis_count=len(candidates),
                best_candidate=best_result,
                rejection_reasons=(best.rejection_reasons if best is not None else ()),
                processing_time_seconds=elapsed,
                filtered_match_counts=tuple(filtered_match_counts),
            ),
        )
    selected = max(
        accepted,
        key=lambda candidate: (candidate.selection_score, candidate.inlier_count),
    )

    return _candidate_to_result(
        selected,
        reference=reference,
        frame=frame,
        reference_rgb=reference_rgb,
        frame_rgb=frame_rgb,
        reference_keypoint_count=len(reference_keypoints),
        frame_keypoint_count=len(frame_keypoints),
        tested_hypothesis_count=len(candidates),
        processing_time_seconds=perf_counter() - started_at,
    )


def _candidate_to_result(
    selected: _Candidate,
    *,
    reference: _ResizeResult,
    frame: _ResizeResult,
    reference_rgb: RgbImage,
    frame_rgb: RgbImage,
    reference_keypoint_count: int,
    frame_keypoint_count: int,
    tested_hypothesis_count: int,
    processing_time_seconds: float,
) -> AutomaticMapAlignmentResult:
    """Поднять гипотезу из масштаба анализа в исходные пиксели.

    Одна и та же процедура используется для принятого ответа и для лучшей
    отвергнутой гипотезы в диагностике. Это исключает расхождение систем
    координат между рабочим результатом и экспортируемым разбором ошибки.
    """

    frame_scale = frame.matrix_original_to_analysis
    reference_scale = reference.matrix_original_to_analysis
    homography_original = (
        np.linalg.inv(reference_scale) @ selected.homography_analysis @ frame_scale
    )
    homography_original /= homography_original[2, 2]
    frame_points_original = selected.frame_points_analysis / np.asarray(
        [frame.scale_x, frame.scale_y], dtype=np.float64
    )
    reference_points_original = selected.reference_points_analysis / np.asarray(
        [reference.scale_x, reference.scale_y], dtype=np.float64
    )
    projected_corners, area_fraction, overlap_fraction, edge_ratio, _ = (
        _projected_geometry(
            homography_original,
            frame_size=(frame_rgb.shape[1], frame_rgb.shape[0]),
            reference_size=(reference_rgb.shape[1], reference_rgb.shape[0]),
        )
    )
    inlier_frame = frame_points_original[selected.inlier_mask]
    inlier_reference = reference_points_original[selected.inlier_mask]
    predicted_reference = cv2.perspectiveTransform(
        inlier_frame.reshape(1, -1, 2).astype(np.float64),
        homography_original,
    ).reshape(-1, 2)
    errors = np.linalg.norm(predicted_reference - inlier_reference, axis=1)

    return AutomaticMapAlignmentResult(
        homography_frame_to_reference=homography_original.astype(np.float64),
        frame_points_px=frame_points_original.astype(np.float64),
        reference_points_px=reference_points_original.astype(np.float64),
        inlier_mask=selected.inlier_mask.copy(),
        selected_ratio_threshold=selected.ratio_threshold,
        tested_hypothesis_count=tested_hypothesis_count,
        reference_keypoint_count=reference_keypoint_count,
        frame_keypoint_count=frame_keypoint_count,
        frame_coverage_fraction=selected.frame_coverage_fraction,
        projected_frame_corners_reference_px=projected_corners,
        projected_area_fraction=area_fraction,
        map_overlap_fraction=overlap_fraction,
        projected_edge_ratio=edge_ratio,
        inlier_reprojection_median_reference_px=float(np.median(errors)),
        inlier_reprojection_p95_reference_px=float(np.percentile(errors, 95)),
        inlier_reprojection_max_reference_px=float(np.max(errors)),
        processing_time_seconds=processing_time_seconds,
        unfiltered_mutual_match_count=selected.unfiltered_mutual_match_count,
        scale_orientation_center_degrees=(
            selected.scale_orientation_center_degrees
        ),
        scale_orientation_center_log2_scale=(
            selected.scale_orientation_center_log2_scale
        ),
        scale_orientation_vote_score=selected.scale_orientation_vote_score,
    )


def _validate_rgb_image(image_rgb: RgbImage, *, name: str) -> None:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"{name} должна быть RGB-изображением с тремя каналами")
    if image_rgb.dtype != np.uint8:
        raise ValueError(f"{name} должна иметь тип uint8")
    if min(image_rgb.shape[:2]) < 2:
        raise ValueError(f"{name} должна быть больше одного пикселя")


def _resize_for_analysis(image_rgb: RgbImage, maximum_side: int) -> _ResizeResult:
    height, width = image_rgb.shape[:2]
    longest = max(width, height)
    if longest <= maximum_side:
        return _ResizeResult(image_rgb.copy(), 1.0, 1.0)
    nominal_scale = maximum_side / longest
    resized_width = max(2, round(width * nominal_scale))
    resized_height = max(2, round(height * nominal_scale))
    resized = cv2.resize(
        image_rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    return _ResizeResult(
        resized,
        resized_width / width,
        resized_height / height,
    )


def _mutual_ratio_matches(
    frame_to_reference: tuple[tuple[cv2.DMatch, ...], ...] | list,
    reference_to_frame: tuple[tuple[cv2.DMatch, ...], ...] | list,
    *,
    ratio_threshold: float,
) -> list[_MutualMatch]:
    """Оставить взаимные пары и сохранить качество ratio test с двух сторон."""

    reverse_best: dict[int, tuple[int, float]] = {}
    for neighbours in reference_to_frame:
        if len(neighbours) != 2:
            continue
        best, second = neighbours
        if second.distance <= 0.0:
            continue
        reverse_ratio = best.distance / second.distance
        if reverse_ratio < ratio_threshold:
            reverse_best[best.queryIdx] = (best.trainIdx, reverse_ratio)

    mutual: list[_MutualMatch] = []
    for neighbours in frame_to_reference:
        if len(neighbours) != 2:
            continue
        best, second = neighbours
        if second.distance <= 0.0:
            continue
        forward_ratio = best.distance / second.distance
        reverse = reverse_best.get(best.trainIdx)
        if (
            forward_ratio < ratio_threshold
            and reverse is not None
            and reverse[0] == best.queryIdx
        ):
            mutual.append(
                _MutualMatch(
                    dmatch=best,
                    forward_ratio=forward_ratio,
                    reverse_ratio=reverse[1],
                )
            )
    return mutual


def _filter_matches_by_scale_and_orientation(
    matches: list[_MutualMatch],
    *,
    frame_keypoints: tuple[cv2.KeyPoint, ...],
    reference_keypoints: tuple[cv2.KeyPoint, ...],
    config: AutomaticMapAlignmentConfig,
) -> _MatchFilterResult:
    """Выбрать согласованный кластер локальных поворотов и масштабов.

    Каждая SIFT-пара задаёт грубую гипотезу относительного поворота и масштаба
    двух изображений. Правильные пары при умеренной перспективе образуют
    кластер, тогда как неоднозначные линии голосуют почти случайно. Голос пары
    взвешивается средней однозначностью проверки Лоу в обоих направлениях.

    Это не замена гомографии: локальный поворот и масштаб меняются по кадру из-за
    перспективы. Поэтому допуски намеренно широкие, а строгая геометрическая
    проверка по-прежнему выполняется затем методом MAGSAC.
    """

    if not matches:
        raise ValueError("Фильтр масштаба и поворота получил пустой набор")

    frame_sizes = np.asarray(
        [frame_keypoints[item.dmatch.queryIdx].size for item in matches],
        dtype=np.float64,
    )
    reference_sizes = np.asarray(
        [reference_keypoints[item.dmatch.trainIdx].size for item in matches],
        dtype=np.float64,
    )
    if np.any(frame_sizes <= 0.0) or np.any(reference_sizes <= 0.0):
        raise ValueError("SIFT вернул неположительный масштаб ключевой точки")
    log2_scales = np.log2(reference_sizes / frame_sizes)

    frame_angles = np.asarray(
        [frame_keypoints[item.dmatch.queryIdx].angle for item in matches],
        dtype=np.float64,
    )
    reference_angles = np.asarray(
        [reference_keypoints[item.dmatch.trainIdx].angle for item in matches],
        dtype=np.float64,
    )
    angle_differences = (
        reference_angles - frame_angles + 180.0
    ) % 360.0 - 180.0
    mean_ratios = np.asarray([item.mean_ratio for item in matches], dtype=np.float64)
    vote_weights = np.power(
        np.maximum(1.0 - mean_ratios, 0.0),
        config.match_quality_vote_exponent,
    )

    # В обычном кадре взаимных пар сотни. Ограничение числа возможных центров
    # защищает от квадратичного времени на почти одинаковых изображениях, где
    # пар могут быть тысячи. Голосовать продолжают все пары; центры предлагают
    # 2000 наиболее однозначных.
    center_indices = np.argsort(mean_ratios, kind="stable")[:2_000]
    best_index = int(center_indices[0])
    best_score = -1.0
    for center_index in center_indices:
        circular_distance = np.abs(
            (angle_differences - angle_differences[center_index] + 180.0) % 360.0
            - 180.0
        )
        neighbours = (
            (circular_distance <= config.match_orientation_tolerance_degrees)
            & (
                np.abs(log2_scales - log2_scales[center_index])
                <= config.match_log2_scale_tolerance
            )
        )
        score = float(np.sum(vote_weights[neighbours]))
        if score > best_score:
            best_score = score
            best_index = int(center_index)

    circular_distance = np.abs(
        (angle_differences - angle_differences[best_index] + 180.0) % 360.0
        - 180.0
    )
    selected_mask = (
        (circular_distance <= config.match_orientation_tolerance_degrees)
        & (
            np.abs(log2_scales - log2_scales[best_index])
            <= config.match_log2_scale_tolerance
        )
    )
    selected = [item for item, keep in zip(matches, selected_mask, strict=True) if keep]
    deduplicated = _deduplicate_spatial_matches(
        selected,
        frame_keypoints=frame_keypoints,
        reference_keypoints=reference_keypoints,
        radius_px=config.match_spatial_deduplication_radius_px,
    )
    return _MatchFilterResult(
        matches=tuple(deduplicated),
        center_degrees=float(angle_differences[best_index]),
        center_log2_scale=float(log2_scales[best_index]),
        vote_score=best_score,
        unfiltered_match_count=len(matches),
    )


def _deduplicate_spatial_matches(
    matches: list[_MutualMatch],
    *,
    frame_keypoints: tuple[cv2.KeyPoint, ...],
    reference_keypoints: tuple[cv2.KeyPoint, ...],
    radius_px: float,
) -> list[_MutualMatch]:
    """Оставить сильнейшую пару среди почти совпадающих SIFT-точек.

    Пара считается дублем, если её конец близок к уже выбранному концу хотя бы
    на одном изображении. Это консервативное правило не позволяет нескольким
    масштабам одного угла выдать себя за независимые геометрические
    свидетельства.
    Радиус задан в пикселях копий, уменьшенных для анализа.
    """

    ordered = sorted(
        matches,
        key=lambda item: (
            item.mean_ratio,
            item.dmatch.distance,
            item.dmatch.queryIdx,
            item.dmatch.trainIdx,
        ),
    )
    if radius_px == 0.0:
        return ordered

    radius_squared = radius_px * radius_px
    selected: list[_MutualMatch] = []
    selected_frame_points: list[np.ndarray] = []
    selected_reference_points: list[np.ndarray] = []
    for item in ordered:
        frame_point = np.asarray(
            frame_keypoints[item.dmatch.queryIdx].pt, dtype=np.float64
        )
        reference_point = np.asarray(
            reference_keypoints[item.dmatch.trainIdx].pt, dtype=np.float64
        )
        duplicates_existing = any(
            np.sum((frame_point - existing_frame) ** 2) < radius_squared
            or np.sum((reference_point - existing_reference) ** 2) < radius_squared
            for existing_frame, existing_reference in zip(
                selected_frame_points,
                selected_reference_points,
                strict=True,
            )
        )
        if duplicates_existing:
            continue
        selected.append(item)
        selected_frame_points.append(frame_point)
        selected_reference_points.append(reference_point)
    return selected


def _estimate_candidate(
    matches: tuple[_MutualMatch, ...],
    *,
    frame_keypoints: tuple[cv2.KeyPoint, ...],
    reference_keypoints: tuple[cv2.KeyPoint, ...],
    frame_size: tuple[int, int],
    reference_size: tuple[int, int],
    config: AutomaticMapAlignmentConfig,
    ratio_threshold: float,
    filter_result: _MatchFilterResult,
) -> _Candidate | None:
    frame_points = np.asarray(
        [frame_keypoints[item.dmatch.queryIdx].pt for item in matches],
        dtype=np.float64,
    )
    reference_points = np.asarray(
        [reference_keypoints[item.dmatch.trainIdx].pt for item in matches],
        dtype=np.float64,
    )
    cv2.setRNGSeed(config.random_seed)
    homography, mask = cv2.findHomography(
        frame_points.astype(np.float32),
        reference_points.astype(np.float32),
        cv2.USAC_MAGSAC,
        config.magsac_reprojection_threshold_px,
        None,
        config.magsac_max_iterations,
        config.magsac_confidence,
    )
    if homography is None or mask is None or not np.all(np.isfinite(homography)):
        return None
    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inlier_mask))
    frame_coverage = _point_coverage(
        frame_points[inlier_mask], width=frame_size[0], height=frame_size[1]
    )
    projected, area, overlap, edge_ratio, is_convex = _projected_geometry(
        homography,
        frame_size=frame_size,
        reference_size=reference_size,
    )

    if inlier_count:
        predicted = cv2.perspectiveTransform(
            frame_points[inlier_mask].reshape(1, -1, 2), homography
        ).reshape(-1, 2)
        errors = np.linalg.norm(predicted - reference_points[inlier_mask], axis=1)
        reprojection_p95 = float(np.percentile(errors, 95))
    else:
        reprojection_p95 = float("inf")

    reasons: list[str] = []
    if inlier_count < config.minimum_inliers:
        reasons.append(f"inlier {inlier_count} < {config.minimum_inliers}")
    inlier_ratio = inlier_count / len(matches)
    if inlier_ratio < config.minimum_inlier_ratio:
        reasons.append(f"доля inlier {inlier_ratio:.3f} слишком мала")
    if frame_coverage < config.minimum_frame_coverage_fraction:
        reasons.append(f"покрытие кадра {frame_coverage:.3f} слишком мало")
    if not is_convex:
        reasons.append("контур кадра невыпуклый или самопересекается")
    if not (
        config.minimum_projected_area_fraction
        <= area
        <= config.maximum_projected_area_fraction
    ):
        reasons.append(f"доля площади проекции {area:.3f} вне диапазона")
    if overlap < config.minimum_map_overlap_fraction:
        reasons.append(f"пересечение с картой {overlap:.3f} слишком мало")
    if edge_ratio > config.maximum_projected_edge_ratio:
        reasons.append(f"отношение рёбер {edge_ratio:.2f} слишком велико")
    if reprojection_p95 > config.magsac_reprojection_threshold_px:
        reasons.append(f"p95 репроекции {reprojection_p95:.2f} px слишком велик")

    return _Candidate(
        homography_analysis=homography.astype(np.float64),
        frame_points_analysis=frame_points,
        reference_points_analysis=reference_points,
        inlier_mask=inlier_mask,
        ratio_threshold=ratio_threshold,
        frame_coverage_fraction=frame_coverage,
        projected_area_fraction=area,
        map_overlap_fraction=overlap,
        projected_edge_ratio=edge_ratio,
        is_convex=is_convex,
        reprojection_p95_px=reprojection_p95,
        rejection_reasons=tuple(reasons),
        unfiltered_mutual_match_count=filter_result.unfiltered_match_count,
        scale_orientation_center_degrees=filter_result.center_degrees,
        scale_orientation_center_log2_scale=filter_result.center_log2_scale,
        scale_orientation_vote_score=filter_result.vote_score,
    )


def _point_coverage(points: FloatPoints, *, width: int, height: int) -> float:
    if points.shape[0] < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))
    return float(cv2.contourArea(hull)) / float((width - 1) * (height - 1))


def _projected_geometry(
    homography: Homography,
    *,
    frame_size: tuple[int, int],
    reference_size: tuple[int, int],
) -> tuple[FloatPoints, float, float, float, bool]:
    frame_width, frame_height = frame_size
    reference_width, reference_height = reference_size
    corners = np.asarray(
        [
            [0.0, 0.0],
            [frame_width - 1.0, 0.0],
            [frame_width - 1.0, frame_height - 1.0],
            [0.0, frame_height - 1.0],
        ],
        dtype=np.float64,
    )
    projected = cv2.perspectiveTransform(
        corners.reshape(1, -1, 2), homography.astype(np.float64)
    ).reshape(-1, 2)
    if not np.all(np.isfinite(projected)):
        return projected, float("inf"), 0.0, float("inf"), False

    projected_float = projected.astype(np.float32)
    polygon_area = abs(float(cv2.contourArea(projected_float)))
    reference_area = float((reference_width - 1) * (reference_height - 1))
    area_fraction = polygon_area / reference_area
    rounded = np.rint(projected).astype(np.int32)
    is_convex = bool(cv2.isContourConvex(rounded)) and polygon_area > 0.0

    overlap_fraction = 0.0
    if is_convex:
        reference_rectangle = np.asarray(
            [
                [0.0, 0.0],
                [reference_width - 1.0, 0.0],
                [reference_width - 1.0, reference_height - 1.0],
                [0.0, reference_height - 1.0],
            ],
            dtype=np.float32,
        )
        intersection_area, _ = cv2.intersectConvexConvex(
            projected_float, reference_rectangle
        )
        overlap_fraction = float(intersection_area) / polygon_area

    edge_lengths = np.linalg.norm(
        np.roll(projected, -1, axis=0) - projected,
        axis=1,
    )
    shortest_edge = float(np.min(edge_lengths))
    edge_ratio = (
        float(np.max(edge_lengths)) / shortest_edge
        if shortest_edge > 1e-9
        else float("inf")
    )
    return projected, area_fraction, overlap_fraction, edge_ratio, is_convex


def _failure_summary(
    candidates: list[_Candidate],
    *,
    ratio_match_counts: list[tuple[float, int]],
    filtered_match_counts: list[tuple[float, int]],
    minimum_mutual_matches: int,
) -> str:
    if not candidates:
        maximum_unfiltered = max((count for _, count in ratio_match_counts), default=0)
        maximum_filtered = max((count for _, count in filtered_match_counts), default=0)
        return (
            "Автосовмещение отклонено: после фильтра согласованности масштаба "
            "и поворота недостаточно пар; лучший порог дал "
            f"{maximum_unfiltered}→{maximum_filtered}, требуется "
            f"{minimum_mutual_matches}"
        )
    best = max(
        candidates,
        key=lambda candidate: (candidate.inlier_count, candidate.selection_score),
    )
    reasons = "; ".join(best.rejection_reasons) or "неизвестная причина"
    return (
        "Автосовмещение отклонено геометрической проверкой: "
        f"лучшая гипотеза {best.inlier_count}/{best.mutual_match_count} inlier; "
        f"{reasons}"
    )
