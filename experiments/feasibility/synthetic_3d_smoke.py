"""Сквозная техническая проверка минимальной 3D-сцены G0.

Эксперимент связывает четыре части: зафиксированный JSON, пакетный Blender,
артефакты проекта и независимые метрики PNG. Он намеренно не оценивает
геометрическую точность — это следующий, более строгий опыт.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerial_mapper.synthetic_3d import run_synthetic_3d_smoke

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESCRIPTION = PROJECT_ROOT / "experiments/configs/synthetic_3d_g0_smoke.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/synthetic_3d/g0_smoke"
DEFAULT_BLENDER = PROJECT_ROOT / ".tools/blender-4.5.14-linux-x64/blender"
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts/blender_generate_scene.py"


def parse_arguments() -> argparse.Namespace:
    """Разбирает пути, позволяя перенести опыт на другую машину."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description", type=Path, default=DEFAULT_DESCRIPTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    return parser.parse_args()


def main() -> None:
    """Запускает smoke-тест и завершает процесс ошибкой при безопасном отказе."""

    arguments = parse_arguments()
    if not arguments.blender.is_file():
        raise FileNotFoundError(
            f"Blender не найден: {arguments.blender}. "
            "Выполните ./scripts/install_blender.sh"
        )

    report = run_synthetic_3d_smoke(
        description_path=arguments.description,
        output_directory=arguments.output,
        blender_executable=arguments.blender,
        generator_script=GENERATOR_SCRIPT,
    )
    compact_result = {
        "scenario_id": report["scenario_id"],
        "passed": report["passed"],
        "elapsed_seconds": report["generator"]["elapsed_seconds"],
        "image_checks": report["image_checks"],
        "report": str(arguments.output / "smoke_report.json"),
        "scene": str(arguments.output / "scene.blend"),
    }
    print(json.dumps(compact_result, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
