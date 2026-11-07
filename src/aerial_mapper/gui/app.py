"""Создание Qt-приложения и показ главного окна."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

from aerial_mapper.live_debug import LiveDebugSession

from .build_ui import build_ui


def main(argv: Sequence[str] | None = None) -> int:
    """Обновить класс из ``dron.ui`` и запустить цикл событий Qt.

    Генерация выполняется до импорта ``MainWindow``. Благодаря такому порядку
    приложение можно запустить и после удаления старого сгенерированного файла:
    сначала будет восстановлен ``ui_dron.py``, затем он будет импортирован.
    """

    build_ui()

    # Импорт намеренно локальный: модуль зависит от файла, который создаётся
    # вызовом build_ui() выше.
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    arguments = list(argv) if argv is not None else sys.argv
    application = QApplication(arguments)
    application.setApplicationName("Aerial Map Measurement")

    # Только настоящий запуск приложения создаёт диагностическую сессию.
    # Изолированные GUI-тесты передают MainWindow значение None и ничего не
    # записывают в рабочий каталог.
    live_debug = LiveDebugSession(
        Path(__file__).resolve().parents[3] / "outputs/live_debug"
    )
    original_excepthook = sys.excepthook

    def record_unhandled_exception(
        exception_type: type[BaseException],
        exception: BaseException,
        traceback_object: TracebackType | None,
    ) -> None:
        if not live_debug.closed:
            live_debug.record_exception(exception_type, exception, traceback_object)
        original_excepthook(exception_type, exception, traceback_object)

    sys.excepthook = record_unhandled_exception
    window = MainWindow(
        remember_recent_sources=True,
        live_debug=live_debug,
        protect_unsaved_progress=True,
    )
    window.show()
    try:
        return application.exec()
    finally:
        if not live_debug.closed:
            live_debug.close()
        sys.excepthook = original_excepthook
