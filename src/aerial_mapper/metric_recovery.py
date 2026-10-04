"""Рабочая цепочка: RGB дрона -> привязка к карте -> расстояния в метрах.

Модуль не импортирует Blender и не принимает позу, высоту или внутренние
параметры Repeat-камеры. Эталон считается заранее размеченной картой: вместе с
RGB известны его проецированная система координат и affine-преобразование
пикселей в метры. Пользовательские отрезки задаются двумя точками в кадре.

Для одного эталона гомография оценивается по SIFT-соответствиям и RANSAC.
Наблюдаемые показатели качества проверяются до измерения. Это не решает
неоднозначный поиск среди похожих карт; на текущей идеальной ступени правильный
эталон уже выбран, а проверяется именно восстановление метрического масштаба.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from rasterio import Affine

from aerial_mapper.alignment import (
    AlignmentResult,
    SiftRansacConfig,
    align_frame_to_reference,
)
from aerial_mapper.measurement import SegmentMeasurement, measure_segment
from aerial_mapper.quality import (
    AlignmentQualityDiagnostics,
    analyze_alignment_quality,
)
from aerial_mapper.synthetic import RgbImage


class MetricRecoveryFailure(RuntimeError):
    """Безопасный отказ: наблюдаемых доказательств недостаточно для измерения."""


@dataclass(frozen=True)
class MetricRecoveryThresholds:
    """Заранее заданные пороги для одной плоской пары изображений.

    Все поля до maximum_stability вычисляются без знания правильной позы.
    Поэтому ими можно пользоваться и на реальном кадре. Метрические ошибки
    ground truth сюда намеренно не входят: они относятся только к оценке
    эксперимента после получения рабочего ответа.
    """

    minimum_inlier_count: int
    minimum_inlier_ratio: float
    minimum_coverage_fraction: float
    maximum_reprojection_p95_px: float
    maximum_stability_p95_corner_shift_px: float

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, float | int],
    ) -> MetricRecoveryThresholds:
        """Читает наблюдаемые пороги из общей конфигурации эксперимента."""

        return cls(
            minimum_inlier_count=int(values["minimum_inlier_count"]),
            minimum_inlier_ratio=float(values["minimum_inlier_ratio"]),
            minimum_coverage_fraction=float(values["minimum_coverage_fraction"]),
            maximum_reprojection_p95_px=float(values["maximum_reprojection_p95_px"]),
            maximum_stability_p95_corner_shift_px=float(
                values["maximum_stability_p95_corner_shift_px"]
            ),
        )

    def validate(self) -> None:
        """Запрещает бессмысленные значения до запуска дорогого сопоставления."""

        if self.minimum_inlier_count < 4:
            raise ValueError("Для гомографии нужны минимум четыре inlier-пары")
        if not 0.0 <= self.minimum_inlier_ratio <= 1.0:
            raise ValueError("Доля inlier должна находиться в диапазоне [0, 1]")
        if not 0.0 <= self.minimum_coverage_fraction <= 1.0:
            raise ValueError("Покрытие кадра должно находиться в диапазоне [0, 1]")
        if self.maximum_reprojection_p95_px <= 0.0:
            raise ValueError("Порог репроекционной ошибки должен быть положительным")
        if self.maximum_stability_p95_corner_shift_px <= 0.0:
            raise ValueError("Порог нестабильности должен быть положительным")


@dataclass(frozen=True)
class ObservedSegment:
    """Один отрезок, выбранный непосредственно в RGB-кадре дрона.

    Базовая гомография доказана только для земли. Значение unknown
    заставляет систему отказаться, пока источник разметки или человек
    явно не подтвердит, что обе точки лежат на земле.
    """

    name: str
    frame_points_px: NDArray[np.float64]
    surface: Literal["ground", "roof", "unknown"] = "unknown"


@dataclass(frozen=True)
class RecoveredSegment:
    """Измеренный отрезок и его координаты во всех рабочих системах."""

    name: str
    measurement: SegmentMeasurement


@dataclass(frozen=True)
class MetricRecoveryResult:
    """Принятый результат сопоставления и измерений без скрытой истины."""

    alignment: AlignmentResult
    quality: AlignmentQualityDiagnostics
    segments: tuple[RecoveredSegment, ...]


def alignment_gate_failures(
    alignment: AlignmentResult,
    quality: AlignmentQualityDiagnostics,
    *,
    thresholds: MetricRecoveryThresholds,
) -> tuple[str, ...]:
    """Возвращает все причины отказа, доступные только по рабочим данным."""

    thresholds.validate()
    failures: list[str] = []
    if alignment.inlier_count < thresholds.minimum_inlier_count:
        failures.append(
            f"inlier-пар {alignment.inlier_count} < {thresholds.minimum_inlier_count}"
        )
    if alignment.inlier_ratio < thresholds.minimum_inlier_ratio:
        failures.append(
            f"доля inlier {alignment.inlier_ratio:.3f} < "
            f"{thresholds.minimum_inlier_ratio:.3f}"
        )
    if (
        alignment.inlier_spatial_coverage_fraction
        < thresholds.minimum_coverage_fraction
    ):
        failures.append(
            "покрытие кадра "
            f"{alignment.inlier_spatial_coverage_fraction:.3f} < "
            f"{thresholds.minimum_coverage_fraction:.3f}"
        )
    if (
        quality.inlier_reprojection_p95_reference_px
        > thresholds.maximum_reprojection_p95_px
    ):
        failures.append(
            "p95 репроекционной ошибки "
            f"{quality.inlier_reprojection_p95_reference_px:.3f} px > "
            f"{thresholds.maximum_reprojection_p95_px:.3f} px"
        )

    stability = quality.stability_p95_max_corner_shift_reference_px
    if stability is None:
        failures.append("устойчивость гомографии не удалось оценить")
    elif stability > thresholds.maximum_stability_p95_corner_shift_px:
        failures.append(
            f"p95 нестабильности {stability:.3f} px > "
            f"{thresholds.maximum_stability_p95_corner_shift_px:.3f} px"
        )
    return tuple(failures)


def recover_metric_segments(
    reference_rgb: RgbImage,
    drone_frame_rgb: RgbImage,
    segments: tuple[ObservedSegment, ...],
    *,
    reference_transform: Affine,
    reference_crs: object,
    thresholds: MetricRecoveryThresholds,
    alignment_config: SiftRansacConfig | None = None,
    random_seed: int = 0,
) -> MetricRecoveryResult:
    """Сопоставляет две позы и измеряет выбранные отрезки в метрах.

    Входы рабочей ветки:
    1. размеченный RGB-эталон и его преобразование пикселей в метры;
    2. только RGB Repeat-кадра;
    3. пары пикселей, расстояние между которыми требуется измерить.

    Поза дрона, высота, фокусное расстояние и контрольные мировые координаты
    отсутствуют. При слабой геометрии функция выбрасывает
    MetricRecoveryFailure до вычисления какого-либо расстояния.
    """

    thresholds.validate()
    if not segments:
        raise ValueError("Нужен хотя бы один измеряемый отрезок")
    if len({segment.name for segment in segments}) != len(segments):
        raise ValueError("Имена измеряемых отрезков должны быть уникальны")
    unsupported = [segment.name for segment in segments if segment.surface != "ground"]
    if unsupported:
        raise MetricRecoveryFailure(
            "Плоский baseline измеряет только подтверждённые точки земли; "
            "неподтверждённые или неплоские отрезки: " + ", ".join(unsupported)
        )

    alignment = align_frame_to_reference(
        reference_rgb,
        drone_frame_rgb,
        config=alignment_config,
    )
    frame_height, frame_width = drone_frame_rgb.shape[:2]
    quality = analyze_alignment_quality(
        alignment,
        frame_width_pixels=frame_width,
        frame_height_pixels=frame_height,
        random_seed=random_seed,
    )
    failures = alignment_gate_failures(
        alignment,
        quality,
        thresholds=thresholds,
    )
    if failures:
        raise MetricRecoveryFailure(
            "Привязка отклонена до измерения: " + "; ".join(failures)
        )

    reference_height, reference_width = reference_rgb.shape[:2]
    recovered: list[RecoveredSegment] = []
    for segment in segments:
        points = np.asarray(segment.frame_points_px, dtype=np.float64)
        if points.shape != (2, 2):
            raise ValueError(
                f"Отрезок {segment.name!r} должен содержать массив формы (2, 2)"
            )
        measurement = measure_segment(
            points,
            alignment.homography_frame_to_reference,
            reference_transform=reference_transform,
            reference_crs=reference_crs,
            reference_width_pixels=reference_width,
            reference_height_pixels=reference_height,
        )
        recovered.append(RecoveredSegment(segment.name, measurement))

    return MetricRecoveryResult(
        alignment=alignment,
        quality=quality,
        segments=tuple(recovered),
    )
