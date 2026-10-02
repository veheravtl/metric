"""Диагностика надёжности найденной привязки без знания правильного ответа.

Метрики этого модуля можно вычислить на реальном кадре: им нужны только
соответствия SIFT, маска RANSAC и уже оценённая гомография. Истинное положение
камеры, высота и параметры синтетического генератора сюда не передаются.

Важно: диагностика намеренно не содержит порогов «годен/не годен». Сначала мы
собираем распределения метрик на разных сценах, включая обязательную 3D-
синтетику и впоследствии реальные данные. Только после этого пороги можно
зафиксировать на отдельной валидационной выборке.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aerial_mapper.alignment import AlignmentResult


@dataclass(frozen=True)
class AlignmentQualityDiagnostics:
    """Наблюдаемые признаки устойчивости одной оценки гомографии.

    ``occupied_grid_cells`` дополняет площадь выпуклой оболочки: показывает,
    сколько участков кадра действительно поддержано соответствиями.

    ``horizontal_span_fraction`` и ``vertical_span_fraction`` отдельно
    обнаруживают почти линейную конфигурацию точек. Большая площадь или хороший
    размах только по одной оси не гарантируют устойчивую гомографию.

    Последние поля измеряют чувствительность решения к данным. Из inlier-пар
    несколько раз исключается случайная доля, гомография оценивается заново, а
    затем её четыре угла сравниваются с углами исходной оценки. Большой сдвиг
    означает, что ответ сильно зависит от нескольких конкретных совпадений.
    Единица измерения — пиксель эталона, а не кадра.
    """

    grid_rows: int
    grid_columns: int
    occupied_grid_cells: int
    grid_occupancy_fraction: float
    horizontal_span_fraction: float
    vertical_span_fraction: float
    stability_trials_requested: int
    stability_trials_succeeded: int
    stability_median_max_corner_shift_reference_px: float | None
    stability_p95_max_corner_shift_reference_px: float | None
    stability_max_corner_shift_reference_px: float | None


def _project_frame_corners(
    homography_frame_to_reference: np.ndarray,
    *,
    frame_width_pixels: int,
    frame_height_pixels: int,
) -> np.ndarray | None:
    """Переносит углы кадра в эталон либо сообщает о численной вырожденности."""

    corners = np.asarray(
        [
            [0.0, 0.0],
            [frame_width_pixels - 1.0, 0.0],
            [frame_width_pixels - 1.0, frame_height_pixels - 1.0],
            [0.0, frame_height_pixels - 1.0],
        ],
        dtype=np.float64,
    )
    projected = cv2.perspectiveTransform(
        corners.reshape(1, -1, 2),
        homography_frame_to_reference.astype(np.float64),
    ).reshape(-1, 2)
    if not np.all(np.isfinite(projected)):
        return None
    return projected


def analyze_alignment_quality(
    alignment: AlignmentResult,
    *,
    frame_width_pixels: int,
    frame_height_pixels: int,
    grid_rows: int = 4,
    grid_columns: int = 4,
    stability_trials: int = 30,
    stability_subsample_fraction: float = 0.8,
    random_seed: int = 0,
) -> AlignmentQualityDiagnostics:
    """Вычисляет диагностику, доступную без ground truth.

    Параметры сетки и повторной выборки — не критерии прохождения, а только
    разрешение измерительного инструмента. Они оставлены аргументами функции,
    чтобы последующие эксперименты могли явно менять их, не переписывая ядро.

    Если inlier-пар слишком мало для осмысленного исключения хотя бы одной
    точки, поля сдвига остаются ``None``. Это честнее, чем выдавать нулевую
    «устойчивость» там, где она фактически не была измерена.
    """

    if frame_width_pixels <= 1 or frame_height_pixels <= 1:
        raise ValueError("Размеры кадра должны быть больше одного пикселя")
    if grid_rows <= 0 or grid_columns <= 0:
        raise ValueError("Число строк и столбцов сетки должно быть положительным")
    if stability_trials < 0:
        raise ValueError("Число испытаний устойчивости не может быть отрицательным")
    if not 0.0 < stability_subsample_fraction < 1.0:
        raise ValueError("Доля подвыборки должна находиться строго между 0 и 1")

    inlier_frame_points = alignment.frame_points_px[alignment.inlier_mask]
    inlier_reference_points = alignment.reference_points_px[alignment.inlier_mask]

    # Номера ячеек вычисляются независимо по X и Y. Ограничение диапазона
    # защищает от редкой субпиксельной координаты ровно на правой/нижней границе.
    cell_columns = np.floor(
        inlier_frame_points[:, 0] / frame_width_pixels * grid_columns
    ).astype(np.int64)
    cell_rows = np.floor(
        inlier_frame_points[:, 1] / frame_height_pixels * grid_rows
    ).astype(np.int64)
    cell_columns = np.clip(cell_columns, 0, grid_columns - 1)
    cell_rows = np.clip(cell_rows, 0, grid_rows - 1)
    occupied_cells = len(
        set(zip(cell_rows.tolist(), cell_columns.tolist(), strict=True))
    )

    horizontal_span = float(
        np.ptp(inlier_frame_points[:, 0]) / (frame_width_pixels - 1)
    )
    vertical_span = float(np.ptp(inlier_frame_points[:, 1]) / (frame_height_pixels - 1))

    full_solution_corners = _project_frame_corners(
        alignment.homography_frame_to_reference,
        frame_width_pixels=frame_width_pixels,
        frame_height_pixels=frame_height_pixels,
    )
    maximum_corner_shifts: list[float] = []
    inlier_count = inlier_frame_points.shape[0]

    # Для проверки чувствительности нужно оставить хотя бы четыре точки и при
    # этом действительно исключить хотя бы одну. При четырёх inlier-парах
    # повторная оценка невозможна — единственного минимального набора не хватит
    # для оценки зависимости результата от состава данных.
    if inlier_count >= 5 and full_solution_corners is not None:
        sample_size = max(4, int(round(inlier_count * stability_subsample_fraction)))
        sample_size = min(sample_size, inlier_count - 1)
        random_generator = np.random.default_rng(random_seed)

        for _ in range(stability_trials):
            sample_indices = random_generator.choice(
                inlier_count,
                size=sample_size,
                replace=False,
            )
            sampled_homography, _ = cv2.findHomography(
                inlier_frame_points[sample_indices],
                inlier_reference_points[sample_indices],
                method=0,
            )
            if sampled_homography is None:
                continue
            sampled_corners = _project_frame_corners(
                sampled_homography,
                frame_width_pixels=frame_width_pixels,
                frame_height_pixels=frame_height_pixels,
            )
            if sampled_corners is None:
                continue

            corner_shifts = np.linalg.norm(
                sampled_corners - full_solution_corners,
                axis=1,
            )
            maximum_corner_shifts.append(float(np.max(corner_shifts)))

    if maximum_corner_shifts:
        shift_array = np.asarray(maximum_corner_shifts, dtype=np.float64)
        median_shift: float | None = float(np.median(shift_array))
        p95_shift: float | None = float(np.percentile(shift_array, 95))
        maximum_shift: float | None = float(np.max(shift_array))
    else:
        median_shift = None
        p95_shift = None
        maximum_shift = None

    grid_cell_count = grid_rows * grid_columns
    return AlignmentQualityDiagnostics(
        grid_rows=grid_rows,
        grid_columns=grid_columns,
        occupied_grid_cells=occupied_cells,
        grid_occupancy_fraction=occupied_cells / grid_cell_count,
        horizontal_span_fraction=horizontal_span,
        vertical_span_fraction=vertical_span,
        stability_trials_requested=stability_trials,
        stability_trials_succeeded=len(maximum_corner_shifts),
        stability_median_max_corner_shift_reference_px=median_shift,
        stability_p95_max_corner_shift_reference_px=p95_shift,
        stability_max_corner_shift_reference_px=maximum_shift,
    )
