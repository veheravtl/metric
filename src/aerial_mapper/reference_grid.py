"""Описание регулярной метрической сетки эталонного изображения.

Модуль намеренно не знает ничего о Черкассах или конкретном картографическом
сервисе. Его задача — описать прямоугольный растр, центр которого задан
географическими координатами, а размеры — метрами. Та же структура пригодится
для будущего собственного полигона, если он будет представлен ортофотопланом.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from affine import Affine
from pyproj import Transformer
from rasterio.transform import from_bounds


@dataclass(frozen=True)
class ProjectedRasterGrid:
    """Геометрия прямоугольного растра в проекционной системе координат.

    Границы west, south, east и north относятся к внешним границам растра,
    а не к центрам крайних пикселей. Такое соглашение совпадает с моделью
    GeoTIFF и позволяет однозначно построить аффинное преобразование между
    номером пикселя и метрическими координатами.
    """

    crs: str
    west: float
    south: float
    east: float
    north: float
    width_pixels: int
    height_pixels: int

    @property
    def width_m(self) -> float:
        """Возвращает ширину растра на местности в метрах."""

        return self.east - self.west

    @property
    def height_m(self) -> float:
        """Возвращает высоту растра на местности в метрах."""

        return self.north - self.south

    @property
    def resolution_x_m_per_pixel(self) -> float:
        """Возвращает размер пикселя по горизонтальной оси в метрах."""

        return self.width_m / self.width_pixels

    @property
    def resolution_y_m_per_pixel(self) -> float:
        """Возвращает размер пикселя по вертикальной оси в метрах."""

        return self.height_m / self.height_pixels

    @property
    def affine_transform(self) -> Affine:
        """Строит преобразование пиксель -> координаты CRS для GeoTIFF.

        Номер строки в изображении растёт сверху вниз, тогда как северная
        координата растёт снизу вверх. Поэтому вертикальный коэффициент
        полученного преобразования отрицательный — это нормальное свойство
        геопривязанного растра, ориентированного севером вверх.
        """

        return from_bounds(
            self.west,
            self.south,
            self.east,
            self.north,
            self.width_pixels,
            self.height_pixels,
        )

    @classmethod
    def centered_on_wgs84(
        cls,
        *,
        longitude: float,
        latitude: float,
        output_crs: str,
        width_m: float,
        height_m: float,
        resolution_m_per_pixel: float,
    ) -> ProjectedRasterGrid:
        """Создаёт метрическую сетку вокруг точки, заданной долготой и широтой.

        Параметр always_xy=True принципиален: EPSG:4326 формально допускает
        порядок осей широта, долгота, тогда как пользователь и манифест передают
        привычную пару долгота, широта. Явная настройка предотвращает опасную
        ошибку, при которой корректные числа меняются местами.
        """

        if width_m <= 0 or height_m <= 0:
            raise ValueError("Размеры участка должны быть положительными")
        if resolution_m_per_pixel <= 0:
            raise ValueError("Размер пикселя должен быть положительным")

        width_pixels_float = width_m / resolution_m_per_pixel
        height_pixels_float = height_m / resolution_m_per_pixel
        width_pixels = round(width_pixels_float)
        height_pixels = round(height_pixels_float)

        # Разрешаем только сетку, в которой физический размер делится на размер
        # пикселя без остатка. Иначе фактическое разрешение незаметно отличалось
        # бы от записанного в манифесте.
        if not isclose(width_pixels_float, width_pixels, abs_tol=1e-9):
            raise ValueError("Ширина участка не кратна размеру пикселя")
        if not isclose(height_pixels_float, height_pixels, abs_tol=1e-9):
            raise ValueError("Высота участка не кратна размеру пикселя")

        transformer = Transformer.from_crs(
            "EPSG:4326",
            output_crs,
            always_xy=True,
        )
        center_x, center_y = transformer.transform(longitude, latitude)

        return cls(
            crs=output_crs,
            west=center_x - width_m / 2,
            south=center_y - height_m / 2,
            east=center_x + width_m / 2,
            north=center_y + height_m / 2,
            width_pixels=width_pixels,
            height_pixels=height_pixels,
        )
