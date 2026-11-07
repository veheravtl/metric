"""Покадровое воспроизведение видео и каталогов пролёта в ``QGraphicsView``.

Qt Multimedia хорошо подходит для обычного просмотра, но его публичная
навигация задаётся временем в миллисекундах. Для измерительного приложения
важнее однозначно знать индекс и пиксели текущего кадра. Поэтому файл декодирует
OpenCV, а таймер Qt только задаёт моменты перехода к следующему кадру.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from .flight_sequence import FrameTelemetry, MarsLvigFlight

CaptureFactory = Callable[[str], cv2.VideoCapture]


class VideoPlaybackController(QObject):
    """Управлять локальным видео и отображать точный текущий кадр.

    Индексы кадров начинаются с нуля. После чтения кадра его BGR-массив
    сохраняется в ``current_frame_bgr`` для будущего алгоритма совмещения, а
    отдельная RGB-копия передаётся Qt для показа. Разделение важно: OpenCV и Qt
    используют разный порядок цветовых каналов.
    """

    frame_changed = Signal(int, int)
    video_loaded = Signal(str, int, float)
    playback_changed = Signal(bool)
    error_occurred = Signal(str)

    def __init__(
        self,
        view: QGraphicsView,
        parent: QObject | None = None,
        *,
        capture_factory: CaptureFactory = cv2.VideoCapture,
    ) -> None:
        super().__init__(parent)
        self._view = view
        self._capture_factory = capture_factory
        self._capture: cv2.VideoCapture | None = None
        self._flight: MarsLvigFlight | None = None
        self._source_path: Path | None = None
        self._frame_count = 0
        self._fps = 0.0
        self._current_frame_index: int | None = None
        self._current_frame_bgr: np.ndarray | None = None

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

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._advance_during_playback)

    @property
    def is_loaded(self) -> bool:
        """Загружен ли читаемый источник кадров."""

        has_source = self._capture is not None or self._flight is not None
        return has_source and self._current_frame_index is not None

    @property
    def is_flight(self) -> bool:
        """Является ли источник каталогом извлечённого пролёта."""

        return self._flight is not None

    @property
    def is_playing(self) -> bool:
        """Работает ли сейчас таймер воспроизведения."""

        return self._timer.isActive()

    @property
    def source_path(self) -> Path | None:
        """Путь к видеофайлу либо каталогу текущего пролёта."""

        return self._source_path

    @property
    def current_frame_index(self) -> int | None:
        """Индекс отображаемого кадра, начиная с нуля."""

        return self._current_frame_index

    @property
    def frame_count(self) -> int:
        """Число кадров по метаданным текущего источника."""

        return self._frame_count

    @property
    def fps(self) -> float:
        """Частота кадров, используемая таймером, в кадрах в секунду."""

        return self._fps

    @property
    def current_frame_bgr(self) -> np.ndarray | None:
        """Вернуть копию текущего кадра в порядке каналов OpenCV BGR."""

        if self._current_frame_bgr is None:
            return None
        return self._current_frame_bgr.copy()

    @property
    def current_frame_telemetry(self) -> FrameTelemetry | None:
        """Телеметрия текущего кадра либо ``None`` для обычного видео."""

        if self._flight is None or self._current_frame_index is None:
            return None
        return self._flight.telemetry(self._current_frame_index)

    def load_video(self, path: str | Path) -> bool:
        """Открыть локальный файл и показать его первый кадр.

        Возвращается ``False``, если контейнер или первый кадр прочитать не
        удалось. В таком случае дополнительно испускается ``error_occurred``.
        """

        source_path = Path(path).expanduser().resolve()
        self.close()

        capture = self._capture_factory(str(source_path))
        if not capture.isOpened():
            capture.release()
            self.error_occurred.emit(f"Не удалось открыть видео: {source_path}")
            return False

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0.0:
            # Некоторые контейнеры не сообщают FPS. Значение 25 обеспечивает
            # просмотр, но не должно использоваться как измеренная частота.
            fps = 25.0

        raw_frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = (
            max(0, int(round(raw_frame_count))) if math.isfinite(raw_frame_count) else 0
        )

        self._capture = capture
        self._source_path = source_path
        self._fps = fps
        self._frame_count = frame_count
        self._timer.setInterval(max(1, round(1000.0 / fps)))

        if not self._show_frame(0):
            self.close()
            self.error_occurred.emit(
                f"Видео открылось, но первый кадр не читается: {source_path}"
            )
            return False

        self.video_loaded.emit(str(source_path), self._frame_count, self._fps)
        return True

    def load_flight(self, directory: str | Path) -> bool:
        """Открыть извлечённый пролёт MARS-LVIG и показать первый кадр.

        Ожидается каталог с ``images``, ``image_ids.csv`` и ``gps.csv``. MCAP
        намеренно не читается интерактивно: подготовленные JPEG обеспечивают
        быстрый произвольный переход и уже сохраняют связь с телеметрией.
        """

        source_path = Path(directory).expanduser().resolve()
        self.close()
        try:
            flight = MarsLvigFlight.load(source_path)
        except (OSError, ValueError) as error:
            self.error_occurred.emit(str(error))
            return False

        self._flight = flight
        self._source_path = source_path
        self._frame_count = flight.frame_count
        self._fps = flight.nominal_fps
        self._timer.setInterval(max(1, round(1000.0 / self._fps)))

        if not self._show_frame(0):
            self.close()
            self.error_occurred.emit(
                f"Пролёт открыт, но первый кадр не читается: {source_path}"
            )
            return False

        self.video_loaded.emit(str(source_path), self._frame_count, self._fps)
        return True

    def play(self) -> None:
        """Продолжить воспроизведение с текущего кадра."""

        if not self.is_loaded or self.is_playing:
            return

        if self._at_last_frame():
            self._show_frame(0)

        self._timer.start()
        self.playback_changed.emit(True)

    def pause(self) -> None:
        """Остановить таймер, сохранив текущий кадр."""

        if not self.is_playing:
            return
        self._timer.stop()
        self.playback_changed.emit(False)

    def stop(self) -> None:
        """Остановить воспроизведение и вернуться к первому кадру."""

        self.pause()
        if self.is_loaded and self._current_frame_index != 0:
            self._show_frame(0)

    def step_forward(self) -> bool:
        """Остановиться и показать следующий кадр, если он существует."""

        self.pause()
        if self._current_frame_index is None or self._at_last_frame():
            return False
        return self._show_frame(self._current_frame_index + 1)

    def step_backward(self) -> bool:
        """Остановиться и показать предыдущий кадр, если он существует."""

        self.pause()
        if self._current_frame_index is None or self._current_frame_index <= 0:
            return False
        return self._show_frame(self._current_frame_index - 1)

    def seek(self, frame_index: int) -> bool:
        """Мгновенно перейти к кадру с заданным индексом.

        Воспроизведение сначала останавливается, чтобы таймер не успел сразу
        заменить выбранный оператором кадр следующим. Для обычного видео
        OpenCV перемещает декодер по ``CAP_PROP_POS_FRAMES``; для извлечённого
        пролёта соответствующий JPEG читается напрямую по индексу.

        Возвращается ``False``, если источник ещё не загружен или индекс лежит
        вне известного диапазона. Индексы, как и во всём контроллере,
        начинаются с нуля.
        """

        self.pause()
        if self._current_frame_index is None:
            return False
        if frame_index == self._current_frame_index:
            return True
        return self._show_frame(frame_index)

    def fit_to_view(self) -> None:
        """Вписать кадр в область просмотра, сохранив пропорции."""

        if self._pixmap_item.pixmap().isNull():
            return
        self._view.fitInView(
            self._pixmap_item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

    def close(self) -> None:
        """Освободить декодер и очистить состояние видео."""

        self.pause()
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._flight = None
        self._source_path = None
        self._frame_count = 0
        self._fps = 0.0
        self._current_frame_index = None
        self._current_frame_bgr = None
        self._pixmap_item.setPixmap(QPixmap())
        self._scene.setSceneRect(0.0, 0.0, 0.0, 0.0)

    def _at_last_frame(self) -> bool:
        return (
            self._current_frame_index is not None
            and self._frame_count > 0
            and self._current_frame_index >= self._frame_count - 1
        )

    def _advance_during_playback(self) -> None:
        if self._current_frame_index is None or self._at_last_frame():
            self.pause()
            return

        target_index = self._current_frame_index + 1
        if not self._show_frame(target_index):
            self.pause()
            self.error_occurred.emit(
                f"Не удалось прочитать кадр с индексом {target_index}"
            )

    def _show_frame(self, frame_index: int) -> bool:
        """Декодировать кадр по индексу и обновить графическую сцену."""

        if frame_index < 0 or (self._capture is None and self._flight is None):
            return False
        if self._frame_count > 0 and frame_index >= self._frame_count:
            return False

        if self._flight is not None:
            try:
                frame_bgr = self._flight.read_frame_bgr(frame_index)
            except (OSError, IndexError):
                return False
        else:
            assert self._capture is not None
            next_decoder_index = int(round(self._capture.get(cv2.CAP_PROP_POS_FRAMES)))
            if next_decoder_index != frame_index:
                if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index)):
                    return False

            ok, frame_bgr = self._capture.read()
            if not ok or frame_bgr is None or frame_bgr.size == 0:
                return False

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = width * channels
        image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(image)
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self._current_frame_bgr = frame_bgr
        self._current_frame_index = frame_index
        self.fit_to_view()
        self.frame_changed.emit(frame_index, self._frame_count)
        return True
