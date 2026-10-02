"""Генерация контролируемого синтетического кадра из эталонной карты.

На первом этапе эталон остаётся неизменным, а кадр виртуальной камеры получается
из меньшего четырёхугольного участка. Для такого примера точная гомография
известна заранее, поэтому результат алгоритма впоследствии можно сравнивать не
с визуальным впечатлением, а с проверяемой геометрической истиной.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

import cv2
import numpy as np
from numpy.typing import NDArray

RgbImage = NDArray[np.uint8]
FloatPoints = NDArray[np.float32]
Homography = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticFrameSpec:
    """Параметры одного контролируемого синтетического кадра.

    Физические размеры задаются в метрах, а размер результата — в пикселях.
    Связь между ними не обязана быть постоянным масштабом: после перспективного
    преобразования количество сантиметров на пиксель меняется по кадру.
    """

    footprint_width_m: float = 120.0
    footprint_height_m: float = 90.0
    output_width_pixels: int = 1280
    output_height_pixels: int = 960
    rotation_degrees: float = 12.0
    perspective_strength: float = 0.35

    def validate(self) -> None:
        """Проверяет параметры до начала геометрических вычислений."""

        if self.footprint_width_m <= 0 or self.footprint_height_m <= 0:
            raise ValueError("Физические размеры кадра должны быть положительными")
        if self.output_width_pixels <= 0 or self.output_height_pixels <= 0:
            raise ValueError("Размер изображения в пикселях должен быть положительным")
        if not 0.0 <= self.perspective_strength <= 1.0:
            raise ValueError("Сила перспективы должна находиться в диапазоне [0, 1]")


@dataclass(frozen=True)
class SyntheticFrameResult:
    """Синтетический кадр вместе с полной геометрической истиной."""

    image_rgb: RgbImage
    source_corners_reference_px: FloatPoints
    destination_corners_frame_px: FloatPoints
    homography_reference_to_frame: Homography
    homography_frame_to_reference: Homography
    spec: SyntheticFrameSpec


def _build_source_corners(
    *,
    reference_width_pixels: int,
    reference_height_pixels: int,
    reference_resolution_m_per_pixel: float,
    spec: SyntheticFrameSpec,
) -> FloatPoints:
    """Строит четырёхугольный след виртуальной камеры на эталоне.

    Точки перечисляются по часовой стрелке: левая верхняя, правая верхняя,
    правая нижняя, левая нижняя. Сначала строится прямоугольник требуемого
    физического размера, затем его углы умеренно сдвигаются и вся фигура
    поворачивается вокруг центра эталона.
    """

    footprint_width_px = spec.footprint_width_m / reference_resolution_m_per_pixel
    footprint_height_px = spec.footprint_height_m / reference_resolution_m_per_pixel

    half_width = footprint_width_px / 2.0
    half_height = footprint_height_px / 2.0
    corners = np.array(
        [
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height],
        ],
        dtype=np.float64,
    )

    # Нормированный шаблон сдвигов создаёт выпуклый, но не симметричный
    # четырёхугольник. Это имитирует умеренный наклон камеры без привязки к
    # конкретной модели дрона, которой пока нет.
    perspective_offsets = np.array(
        [
            [0.22 * footprint_width_px, 0.08 * footprint_height_px],
            [-0.06 * footprint_width_px, -0.10 * footprint_height_px],
            [0.10 * footprint_width_px, -0.03 * footprint_height_px],
            [-0.14 * footprint_width_px, 0.08 * footprint_height_px],
        ],
        dtype=np.float64,
    )
    corners += spec.perspective_strength * perspective_offsets

    # Сохраняем центр следа в центре эталона, чтобы деформация не вносила
    # неявного смещения всего выбранного участка.
    corners -= corners.mean(axis=0)

    angle = radians(spec.rotation_degrees)
    rotation = np.array(
        [
            [cos(angle), -sin(angle)],
            [sin(angle), cos(angle)],
        ],
        dtype=np.float64,
    )
    corners = corners @ rotation.T
    corners += np.array(
        [reference_width_pixels / 2.0, reference_height_pixels / 2.0],
        dtype=np.float64,
    )

    if not cv2.isContourConvex(corners.astype(np.float32)):
        raise ValueError("Получившийся след камеры не является выпуклым")

    x_coordinates = corners[:, 0]
    y_coordinates = corners[:, 1]
    if (
        x_coordinates.min() < 0
        or y_coordinates.min() < 0
        or x_coordinates.max() >= reference_width_pixels
        or y_coordinates.max() >= reference_height_pixels
    ):
        raise ValueError(
            "След виртуальной камеры выходит за границы эталонного изображения"
        )

    return corners.astype(np.float32)


def generate_synthetic_frame(
    reference_rgb: RgbImage,
    *,
    reference_resolution_m_per_pixel: float,
    spec: SyntheticFrameSpec | None = None,
) -> SyntheticFrameResult:
    """Создаёт перспективный кадр и возвращает обе точные гомографии.

    Матрица reference_to_frame переводит пиксели полного эталона в пиксели
    синтетического кадра. Обратная матрица потребуется для наложения кадра на
    эталон и позже станет образцом для проверки оцениваемой гомографии.
    """

    if reference_rgb.ndim != 3 or reference_rgb.shape[2] != 3:
        raise ValueError("Эталон должен быть RGB-изображением с тремя каналами")
    if reference_rgb.dtype != np.uint8:
        raise ValueError("Эталон должен иметь тип uint8")
    if reference_resolution_m_per_pixel <= 0:
        raise ValueError("Разрешение эталона должно быть положительным")

    effective_spec = spec or SyntheticFrameSpec()
    effective_spec.validate()

    reference_height, reference_width = reference_rgb.shape[:2]
    source_corners = _build_source_corners(
        reference_width_pixels=reference_width,
        reference_height_pixels=reference_height,
        reference_resolution_m_per_pixel=reference_resolution_m_per_pixel,
        spec=effective_spec,
    )
    destination_corners = np.array(
        [
            [0.0, 0.0],
            [effective_spec.output_width_pixels - 1.0, 0.0],
            [
                effective_spec.output_width_pixels - 1.0,
                effective_spec.output_height_pixels - 1.0,
            ],
            [0.0, effective_spec.output_height_pixels - 1.0],
        ],
        dtype=np.float32,
    )

    homography_reference_to_frame = cv2.getPerspectiveTransform(
        source_corners,
        destination_corners,
    )
    homography_frame_to_reference = np.linalg.inv(homography_reference_to_frame)

    frame_rgb = cv2.warpPerspective(
        reference_rgb,
        homography_reference_to_frame,
        (
            effective_spec.output_width_pixels,
            effective_spec.output_height_pixels,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return SyntheticFrameResult(
        image_rgb=frame_rgb,
        source_corners_reference_px=source_corners,
        destination_corners_frame_px=destination_corners,
        homography_reference_to_frame=homography_reference_to_frame,
        homography_frame_to_reference=homography_frame_to_reference,
        spec=effective_spec,
    )
