"""Загрузка растровой карты в правую область главного окна."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


class MapImageController(QObject):
    """Показывать растровую карту и сохранять её исходные пиксели.

    Координаты графической сцены совпадают с координатами исходного изображения:
    левый верхний пиксель имеет координаты ``(0, 0)``. Это соглашение пригодится
    для ручной разметки соответствующих точек между кадром дрона и картой.
    """

    map_loaded = Signal(str, int, int)
    error_occurred = Signal(str)

    def __init__(self, view: QGraphicsView, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._view = view
        self._source_path: Path | None = None
        self._image_bgr: np.ndarray | None = None

        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation
        )
        self._scene.addItem(self._pixmap_item)
        self._view.setScene(self._scene)
        self._view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._view.setBackgroundBrush(Qt.GlobalColor.black)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    @property
    def is_loaded(self) -> bool:
        """Загружена ли читаемая карта."""

        return self._image_bgr is not None

    @property
    def source_path(self) -> Path | None:
        """Путь к текущему изображению карты."""

        return self._source_path

    @property
    def current_image_bgr(self) -> np.ndarray | None:
        """Вернуть копию исходных пикселей карты в порядке каналов BGR."""

        if self._image_bgr is None:
            return None
        return self._image_bgr.copy()

    def load_image(self, path: str | Path) -> bool:
        """Прочитать PNG/JPEG/TIFF, показать его и вернуть признак успеха."""

        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            self.error_occurred.emit(f"Файл карты не найден: {source_path}")
            return False

        image_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image_bgr is None or image_bgr.size == 0:
            self.error_occurred.emit(
                f"Не удалось прочитать изображение карты: {source_path}"
            )
            return False

        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape
        bytes_per_line = width * channels
        qt_image = QImage(
            rgb_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()

        self._source_path = source_path
        self._image_bgr = image_bgr
        self._pixmap_item.setPixmap(QPixmap.fromImage(qt_image))
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_to_view()
        self.map_loaded.emit(str(source_path), width, height)
        return True

    def fit_to_view(self) -> None:
        """Вписать карту в правую область без искажения пропорций."""

        if self._pixmap_item.pixmap().isNull():
            return
        self._view.fitInView(
            self._pixmap_item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

    def clear(self) -> None:
        """Освободить пиксели и очистить графическую сцену."""

        self._source_path = None
        self._image_bgr = None
        self._pixmap_item.setPixmap(QPixmap())
        self._scene.setSceneRect(0.0, 0.0, 0.0, 0.0)
