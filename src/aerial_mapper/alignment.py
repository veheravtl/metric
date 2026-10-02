"""Независимая привязка кадра к эталону локальными признаками.

Этот модуль намеренно ничего не знает о том, как был создан синтетический кадр.
Функция привязки получает только два RGB-изображения. Истинная гомография,
координаты вырезанного участка и параметры генератора в её интерфейс не входят.
Такое разделение не даёт экспериментальному коду случайно воспользоваться
правильным ответом.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np
from numpy.typing import NDArray

from aerial_mapper.synthetic import FloatPoints, Homography, RgbImage


class AlignmentFailure(RuntimeError):
    """Ожидаемый отказ привязки, когда данных недостаточно для честного ответа."""


@dataclass(frozen=True)
class SiftRansacConfig:
    """Явные параметры воспроизводимого baseline SIFT + RANSAC.

    ``ratio_threshold`` относится к ratio test Лоу. Чем меньше значение, тем
    строже отбрасываются неоднозначные соответствия. Порог RANSAC измеряется в
    пикселях эталона, потому что оцениваемая матрица переводит точки кадра в
    систему координат эталона.
    """

    max_features_per_image: int = 8_000
    ratio_threshold: float = 0.72
    ransac_reprojection_threshold_px: float = 3.0
    ransac_max_iterations: int = 10_000
    ransac_confidence: float = 0.999
    minimum_ratio_matches: int = 8
    random_seed: int = 0

    def validate(self) -> None:
        """Проверяет диапазоны параметров до дорогостоящего поиска признаков."""

        if self.max_features_per_image < 4:
            raise ValueError("Для гомографии нужно разрешить не менее четырёх точек")
        if not 0.0 < self.ratio_threshold < 1.0:
            raise ValueError("Порог ratio test должен находиться между 0 и 1")
        if self.ransac_reprojection_threshold_px <= 0:
            raise ValueError("Порог ошибки RANSAC должен быть положительным")
        if self.ransac_max_iterations <= 0:
            raise ValueError("Число итераций RANSAC должно быть положительным")
        if not 0.0 < self.ransac_confidence < 1.0:
            raise ValueError("Уверенность RANSAC должна находиться между 0 и 1")
        if self.minimum_ratio_matches < 4:
            raise ValueError("Для гомографии необходимы минимум четыре пары")


@dataclass(frozen=True)
class AlignmentResult:
    """Результат, полученный исключительно из содержимого двух изображений."""

    homography_frame_to_reference: Homography
    frame_points_px: FloatPoints
    reference_points_px: FloatPoints
    descriptor_distances: NDArray[np.float32]
    inlier_mask: NDArray[np.bool_]
    reference_keypoint_count: int
    frame_keypoint_count: int
    candidate_match_count: int
    inlier_spatial_coverage_fraction: float
    processing_time_seconds: float

    @property
    def ratio_match_count(self) -> int:
        """Количество пар, прошедших проверку однозначности дескрипторов."""

        return int(self.frame_points_px.shape[0])

    @property
    def inlier_count(self) -> int:
        """Количество пар, согласованных с найденной гомографией."""

        return int(np.count_nonzero(self.inlier_mask))

    @property
    def inlier_ratio(self) -> float:
        """Доля геометрически согласованных пар среди принятых SIFT-пар."""

        return self.inlier_count / self.ratio_match_count


def _validate_rgb_image(image_rgb: RgbImage, *, name: str) -> None:
    """Проверяет соглашение о формате изображения на границе модуля."""

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"{name} должен быть RGB-изображением с тремя каналами")
    if image_rgb.dtype != np.uint8:
        raise ValueError(f"{name} должен иметь тип uint8")


def _calculate_frame_coverage(
    frame_points_px: FloatPoints,
    inlier_mask: NDArray[np.bool_],
    *,
    frame_width_pixels: int,
    frame_height_pixels: int,
) -> float:
    """Оценивает долю кадра внутри выпуклой оболочки inlier-точек.

    Большое число точек в одном маленьком углу может породить нестабильную
    экстраполяцию на остальной кадр. Эта простая метрика пока только выводится
    для диагностики; порог отказа будет выбран по серии экспериментов.
    """

    inlier_points = frame_points_px[inlier_mask]
    if inlier_points.shape[0] < 3:
        return 0.0

    hull = cv2.convexHull(inlier_points.reshape(-1, 1, 2))
    hull_area = float(cv2.contourArea(hull))
    frame_area = float((frame_width_pixels - 1) * (frame_height_pixels - 1))
    return hull_area / frame_area


def align_frame_to_reference(
    reference_rgb: RgbImage,
    frame_rgb: RgbImage,
    *,
    config: SiftRansacConfig | None = None,
) -> AlignmentResult:
    """Независимо оценивает гомографию из кадра в полный эталон.

    SIFT находит точки и 128-мерные дескрипторы отдельно на каждом изображении.
    Точный brute-force matcher ищет два ближайших дескриптора эталона для каждой
    точки кадра. Ratio test оставляет пары, у которых лучший кандидат заметно
    лучше второго. RANSAC затем отбрасывает пары, не согласующиеся с одной
    плоской проективной моделью.

    Функция выбрасывает ``AlignmentFailure`` вместо правдоподобной матрицы,
    когда признаков или соответствий недостаточно.
    """

    effective_config = config or SiftRansacConfig()
    effective_config.validate()
    _validate_rgb_image(reference_rgb, name="Эталон")
    _validate_rgb_image(frame_rgb, name="Кадр")

    started_at = perf_counter()
    reference_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create(nfeatures=effective_config.max_features_per_image)
    reference_keypoints, reference_descriptors = sift.detectAndCompute(
        reference_gray,
        None,
    )
    frame_keypoints, frame_descriptors = sift.detectAndCompute(frame_gray, None)

    if reference_descriptors is None or len(reference_keypoints) < 4:
        raise AlignmentFailure("На эталоне найдено недостаточно SIFT-признаков")
    if frame_descriptors is None or len(frame_keypoints) < 4:
        raise AlignmentFailure("На кадре найдено недостаточно SIFT-признаков")

    # Для первого PoC используем точный перебор, а не приближённый FLANN.
    # При текущем размере эталона он достаточно быстр и убирает ещё один источник
    # вариативности. Для большой карты позже понадобится поиск по тайлам или
    # отдельный этап грубой локализации.
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    nearest_pairs = matcher.knnMatch(frame_descriptors, reference_descriptors, k=2)

    ratio_matches: list[cv2.DMatch] = []
    for neighbours in nearest_pairs:
        # Теоретически список может содержать меньше двух элементов, если на
        # эталоне слишком мало дескрипторов. Такая пара не позволяет проверить
        # однозначность и потому безопасно отбрасывается.
        if len(neighbours) != 2:
            continue
        best_match, second_match = neighbours
        if (
            best_match.distance
            < effective_config.ratio_threshold * second_match.distance
        ):
            ratio_matches.append(best_match)

    if len(ratio_matches) < effective_config.minimum_ratio_matches:
        raise AlignmentFailure(
            f"После ratio test осталось недостаточно соответствий: {len(ratio_matches)}"
        )

    frame_points = np.asarray(
        [frame_keypoints[match.queryIdx].pt for match in ratio_matches],
        dtype=np.float32,
    )
    reference_points = np.asarray(
        [reference_keypoints[match.trainIdx].pt for match in ratio_matches],
        dtype=np.float32,
    )
    descriptor_distances = np.asarray(
        [match.distance for match in ratio_matches],
        dtype=np.float32,
    )

    # OpenCV использует собственный генератор случайных чисел при выборе
    # минимальных подмножеств RANSAC. Фиксированный seed делает учебный опыт
    # воспроизводимым при неизменной версии библиотеки и одинаковом окружении.
    cv2.setRNGSeed(effective_config.random_seed)
    estimated_homography, ransac_mask = cv2.findHomography(
        frame_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=effective_config.ransac_reprojection_threshold_px,
        maxIters=effective_config.ransac_max_iterations,
        confidence=effective_config.ransac_confidence,
    )
    if estimated_homography is None or ransac_mask is None:
        raise AlignmentFailure("RANSAC не смог оценить гомографию")

    inlier_mask = ransac_mask.ravel().astype(bool)
    if np.count_nonzero(inlier_mask) < 4:
        raise AlignmentFailure("RANSAC нашёл меньше четырёх согласованных пар")

    estimated_homography = estimated_homography.astype(np.float64)
    if np.isclose(estimated_homography[2, 2], 0.0):
        raise AlignmentFailure("Получена вырожденная матрица гомографии")
    estimated_homography /= estimated_homography[2, 2]

    frame_height, frame_width = frame_rgb.shape[:2]
    coverage = _calculate_frame_coverage(
        frame_points,
        inlier_mask,
        frame_width_pixels=frame_width,
        frame_height_pixels=frame_height,
    )

    return AlignmentResult(
        homography_frame_to_reference=estimated_homography,
        frame_points_px=frame_points,
        reference_points_px=reference_points,
        descriptor_distances=descriptor_distances,
        inlier_mask=inlier_mask,
        reference_keypoint_count=len(reference_keypoints),
        frame_keypoint_count=len(frame_keypoints),
        candidate_match_count=len(nearest_pairs),
        inlier_spatial_coverage_fraction=coverage,
        processing_time_seconds=perf_counter() - started_at,
    )
