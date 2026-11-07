"""Преобразование файла Qt Designer ``dron.ui`` в импортируемый Python-класс."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UI_SOURCE_PATH = PROJECT_ROOT / "dron.ui"
GENERATED_UI_PATH = Path(__file__).parent / "generated" / "ui_dron.py"


def find_uic_command() -> list[str]:
    """Найти официальный компилятор интерфейсов ``pyside6-uic``.

    Сначала проверяется каталог текущего Python-интерпретатора. Это важно при
    одновременной установке нескольких версий PySide6: файл должен собираться
    именно той версией Qt, которая запускает приложение.
    """

    environment_tool = Path(sys.executable).with_name("pyside6-uic")
    if environment_tool.is_file():
        return [str(environment_tool)]

    path_tool = shutil.which("pyside6-uic")
    if path_tool is not None:
        return [path_tool]

    # Резервный официальный способ запуска инструмента из пакета PySide6.
    return [sys.executable, "-m", "PySide6.scripts.pyside_tool", "uic"]


def generated_ui_is_current(
    source_path: Path = UI_SOURCE_PATH,
    output_path: Path = GENERATED_UI_PATH,
) -> bool:
    """Проверить, что Python-файл существует и не старее исходного ``.ui``."""

    return (
        output_path.is_file()
        and source_path.is_file()
        and output_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns
    )


def build_ui(
    source_path: Path = UI_SOURCE_PATH,
    output_path: Path = GENERATED_UI_PATH,
    *,
    force: bool = False,
) -> bool:
    """Собрать ``.ui`` в Python и вернуть признак фактической генерации.

    Если исходник не менялся, внешний процесс не запускается. Ошибка ``uic``
    не скрывается: приложение не должно открываться со старым интерфейсом,
    когда свежий ``dron.ui`` содержит ошибку.
    """

    source_path = Path(source_path)
    output_path = Path(output_path)

    if not source_path.is_file():
        raise FileNotFoundError(f"Не найден исходный интерфейс: {source_path}")

    if not force and generated_ui_is_current(source_path, output_path):
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [*find_uic_command(), str(source_path), "-o", str(output_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        details = result.stdout.strip() or result.stderr.strip()
        raise RuntimeError(
            "Не удалось преобразовать Qt Designer UI в Python. "
            f"Команда: {' '.join(command)}\n{details}"
        )

    # ``pyside6-uic`` оставляет лишнюю пустую строку в конце файла. Убираем её,
    # чтобы воспроизводимо сгенерированный исходник проходил ``git diff --check``.
    generated_source = output_path.read_text(encoding="utf-8")
    output_path.write_text(
        generated_source.rstrip("\r\n") + "\n",
        encoding="utf-8",
    )

    return True


def main() -> int:
    """Командная точка входа для ручного обновления сгенерированного файла."""

    changed = build_ui(force=True)
    state = "обновлён" if changed else "уже актуален"
    print(f"[build-ui] {GENERATED_UI_PATH.relative_to(PROJECT_ROOT)}: {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
