"""Оркестрация минимального воспроизводимого 3D-smoke-теста.

Модуль намеренно не импортирует bpy — программный интерфейс Blender.
Обычный Python проекта отвечает за проверку описания сцены, запуск отдельного
процесса Blender и анализ готовых PNG. Само построение геометрии выполняет
scripts/blender_generate_scene.py внутри Python, встроенного в Blender.

Такое разделение не смешивает два окружения и позволяет тестировать метрики
изображения обычным pytest даже на машине без Blender.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class RenderMetrics:
    """Простые признаки того, что рендер не пуст и содержит структуру.

    Эти величины не измеряют геометрическую правильность сцены. Они являются
    дешёвыми предохранителями: обнаруживают чёрный кадр, однотонную заливку,
    неверное разрешение или отсутствие заметных границ объектов.
    """

    width_pixels: int
    height_pixels: int
    mean_luminance: float
    luminance_standard_deviation: float
    robust_dynamic_range: float
    edge_density: float


@dataclass(frozen=True)
class RenderCheck:
    """Результат проверки одного изображения с явными причинами отказа."""

    image_path: str
    metrics: RenderMetrics
    passed: bool
    failures: tuple[str, ...]


def load_scene_description(path: Path) -> dict[str, Any]:
    """Читает и проверяет минимально необходимые поля описания сцены.

    Полная физическая валидация остаётся в Blender-генераторе, потому что он
    знает точный тип создаваемой геометрии. Здесь проверяются ошибки, которые
    полезно показать до запуска тяжёлого внешнего процесса.
    """

    description = json.loads(path.read_text(encoding="utf-8"))
    required_top_level = {
        "schema_version",
        "scenario_id",
        "seed",
        "world",
        "render",
        "cameras",
        "objects",
    }
    missing = sorted(required_top_level - description.keys())
    if missing:
        raise ValueError(f"В описании сцены отсутствуют поля: {', '.join(missing)}")

    if description["schema_version"] != 1:
        raise ValueError("Поддерживается только schema_version = 1")
    if description["world"].get("units") != "metres":
        raise ValueError("Единицы мировой системы должны быть 'metres'")

    width = description["render"].get("width_pixels", 0)
    height = description["render"].get("height_pixels", 0)
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("Размер рендера должен быть задан целыми пикселями")
    if width <= 0 or height <= 0:
        raise ValueError("Размер рендера должен быть положительным")

    cameras = description["cameras"]
    roles = {camera.get("role") for camera in cameras}
    if not {"reference", "repeat"}.issubset(roles):
        raise ValueError("Нужны камеры ролей reference и repeat")

    identifiers = [camera.get("id") for camera in cameras]
    identifiers.extend(obj.get("id") for obj in description["objects"])
    if any(not identifier for identifier in identifiers):
        raise ValueError("Каждая камера и каждый объект должны иметь id")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Идентификаторы камер и объектов должны быть уникальны")

    return description


def analyze_render(image_path: Path) -> RenderMetrics:
    """Считает базовые метрики структуры RGB-рендера.

    Яркость вычисляется OpenCV в диапазоне 0..255. Робастный динамический
    диапазон — разность 99-го и 1-го процентилей яркости; крайние выбросы не
    могут искусственно сделать почти однотонный кадр контрастным. Плотность
    границ — доля пикселей, отмеченных детектором Canny как резкий переход.
    """

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    low_percentile, high_percentile = np.percentile(gray, [1.0, 99.0])

    return RenderMetrics(
        width_pixels=int(gray.shape[1]),
        height_pixels=int(gray.shape[0]),
        mean_luminance=float(np.mean(gray)),
        luminance_standard_deviation=float(np.std(gray)),
        robust_dynamic_range=float(high_percentile - low_percentile),
        edge_density=float(np.count_nonzero(edges) / edges.size),
    )


def check_render(
    image_path: Path,
    *,
    expected_width: int,
    expected_height: int,
) -> RenderCheck:
    """Проверяет один рендер по заранее заданным мягким smoke-порогам.

    Пороги специально ловят только грубые сбои. Их нельзя использовать как
    критерий качества привязки или фотореализма.
    """

    metrics = analyze_render(image_path)
    failures: list[str] = []
    if (
        metrics.width_pixels != expected_width
        or metrics.height_pixels != expected_height
    ):
        failures.append(
            "неверное разрешение: "
            f"{metrics.width_pixels}x{metrics.height_pixels} вместо "
            f"{expected_width}x{expected_height}"
        )
    if not 15.0 <= metrics.mean_luminance <= 240.0:
        failures.append("средняя яркость указывает на почти чёрный или белый кадр")
    if metrics.luminance_standard_deviation < 12.0:
        failures.append("слишком малый контраст яркости")
    if metrics.robust_dynamic_range < 45.0:
        failures.append("слишком малый робастный динамический диапазон")
    if metrics.edge_density < 0.005:
        failures.append("слишком мало выраженных границ")

    return RenderCheck(
        image_path=str(image_path),
        metrics=metrics,
        passed=not failures,
        failures=tuple(failures),
    )


def run_synthetic_3d_smoke(
    *,
    description_path: Path,
    output_directory: Path,
    blender_executable: Path,
    generator_script: Path,
) -> dict[str, Any]:
    """Запускает Blender, анализирует рендеры и сохраняет итоговый отчёт.

    Возвращаемый словарь одновременно записывается в smoke_report.json.
    Ненулевой код Blender, отсутствие файлов и провал порогов не маскируются:
    вызывающий эксперимент получает исключение либо отчёт с passed=false.
    """

    description = load_scene_description(description_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    command = [
        str(blender_executable),
        "--background",
        "--factory-startup",
        "--python",
        str(generator_script),
        "--",
        str(description_path.resolve()),
        str(output_directory.resolve()),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)

    generator_metadata_path = output_directory / "generation_metadata.json"
    if not generator_metadata_path.is_file():
        raise RuntimeError("Blender не создал generation_metadata.json")
    generator_metadata = json.loads(generator_metadata_path.read_text(encoding="utf-8"))

    expected_width = description["render"]["width_pixels"]
    expected_height = description["render"]["height_pixels"]
    checks = [
        check_render(
            output_directory / relative_path,
            expected_width=expected_width,
            expected_height=expected_height,
        )
        for relative_path in generator_metadata["rendered_images"]
    ]

    required_artifacts = [
        output_directory / "scene.blend",
        output_directory / "scene_description.json",
        *[Path(check.image_path) for check in checks],
    ]
    missing_artifacts = [str(path) for path in required_artifacts if not path.is_file()]
    report = {
        "schema_version": 1,
        "scenario_id": description["scenario_id"],
        "passed": all(check.passed for check in checks) and not missing_artifacts,
        "command": command,
        "blender_stdout_tail": completed.stdout.splitlines()[-30:],
        "generator": generator_metadata,
        "image_checks": [
            {
                **asdict(check),
                "failures": list(check.failures),
            }
            for check in checks
        ],
        "missing_artifacts": missing_artifacts,
        "interpretation": (
            "Smoke-тест проверяет запуск, сохранение сцены и непустые RGB-рендеры. "
            "Он не проверяет метрическую точность, глубину и преобразование координат."
        ),
    }
    report_path = output_directory / "smoke_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
