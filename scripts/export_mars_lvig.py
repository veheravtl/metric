#!/usr/bin/env python3
"""Извлекает RGB и RTK выбранных MARS-LVIG-полётов из локальных ROS bag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aerial_mapper.mars_lvig_dataset import export_mars_lvig_bag

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/manifests/mars_lvig_hkairport.json"


def _load_manifest(path: Path) -> dict[str, Any]:
    """Читает манифест происхождения данных и параметров экспорта."""

    with path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest.get("sequences"), list):
        raise ValueError("Манифест MARS-LVIG не содержит список sequences")
    return manifest


def _find_sequence(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Требует точное и единственное совпадение имени последовательности."""

    matches = [
        sequence
        for sequence in manifest["sequences"]
        if str(sequence.get("name")) == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Ожидалась ровно одна последовательность {name!r}, найдено {len(matches)}"
        )
    return matches[0]


def parse_args() -> argparse.Namespace:
    """Не допускает случайного извлечения всех многогигабайтных полётов."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--sequence",
        action="append",
        required=True,
        help="HKairport-последовательность; параметр можно повторять",
    )
    parser.add_argument(
        "--source",
        choices=("official_rosbag", "huggingface_mcap"),
        default="official_rosbag",
        help="Формат локального файла, который был явно загружен",
    )
    return parser.parse_args()


def main() -> None:
    """Экспортирует явно выбранные bag согласно зафиксированному манифесту."""

    args = parse_args()
    manifest = _load_manifest(args.manifest)
    storage_directory = PROJECT_ROOT / str(manifest["storage_directory"])
    camera_topic = str(manifest["topics"]["camera"])
    rtk_topic = str(manifest["topics"]["rtk_position"])

    for sequence_name in args.sequence:
        sequence = _find_sequence(manifest, sequence_name)
        if args.source == "official_rosbag":
            artifact = sequence
            input_directory = storage_directory / "bags"
        else:
            artifact = sequence["mirrors"]["huggingface_mcap"]
            input_directory = storage_directory / "mcap"
        summary = export_mars_lvig_bag(
            input_directory / str(artifact["filename"]),
            storage_directory / "extracted" / sequence_name,
            camera_topic=camera_topic,
            rtk_topic=rtk_topic,
            route_start_offset_seconds=float(
                sequence["route_start_offset_seconds"]
            ),
        )
        print(
            f"{sequence_name}: {summary.image_count} кадров, "
            f"{summary.rtk_count} валидных RTK-точек"
        )
        if summary.rejected_rtk_count:
            print(
                f"{sequence_name}: отклонено RTK-точек "
                f"{summary.rejected_rtk_count}"
            )


if __name__ == "__main__":
    main()
