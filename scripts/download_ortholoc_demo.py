#!/usr/bin/env python3
"""Загружает два официальных demo-образца OrthoLoC и проверяет SHA-256.

Сами NPZ не коммитятся: лицензия CC BY-NC-SA 4.0 разрешает учебное
некоммерческое использование с атрибуцией, но крупные внешние данные должны
оставаться воспроизводимыми локальными входами, а не частью нашего репозитория.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/manifests/ortholoc_demo.json"
HTTP_TIMEOUT_SECONDS = 120
CHUNK_SIZE_BYTES = 1024 * 1024


def load_manifest(path: Path) -> dict[str, Any]:
    """Читает описание источника, лицензии, файлов и контрольных сумм."""

    with path.open(encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def calculate_sha256(path: Path) -> str:
    """Вычисляет SHA-256 потоково, не загружая весь NPZ в память."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checked_file(
    *,
    url: str,
    destination: Path,
    expected_sha256: str,
    force: bool,
) -> str:
    """Атомарно загружает файл и не принимает повреждённый результат."""

    if destination.exists() and not force:
        actual_hash = calculate_sha256(destination)
        if actual_hash == expected_sha256:
            return "уже существует, SHA-256 совпадает"
        raise RuntimeError(
            f"{destination} уже существует, но SHA-256 не совпадает. "
            "Удалите подозрительный файл или осознанно передайте --force."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "aerial-map-measurement/0.1"})
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".download",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        digest = hashlib.sha256()
        try:
            with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                while chunk := response.read(CHUNK_SIZE_BYTES):
                    temporary_file.write(chunk)
                    digest.update(chunk)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    actual_hash = digest.hexdigest()
    if actual_hash != expected_sha256:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Контрольная сумма {destination.name} не совпала: "
            f"получено {actual_hash}, ожидалось {expected_sha256}"
        )
    temporary_path.replace(destination)
    return "загружен и проверен"


def parse_args() -> argparse.Namespace:
    """Разбирает путь к манифесту и осознанное разрешение перезаписи."""

    parser = argparse.ArgumentParser(
        description="Загрузить официальные demo-образцы OrthoLoC",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Повторно загрузить и заменить существующие файлы",
    )
    return parser.parse_args()


def main() -> None:
    """Загружает все перечисленные образцы и печатает атрибуцию."""

    args = parse_args()
    manifest = load_manifest(args.manifest)
    storage_directory = PROJECT_ROOT / manifest["storage_directory"]
    for sample in manifest["samples"]:
        destination = storage_directory / sample["filename"]
        status = download_checked_file(
            url=sample["url"],
            destination=destination,
            expected_sha256=sample["sha256"],
            force=args.force,
        )
        print(f"{destination.relative_to(PROJECT_ROOT)}: {status}")

    source = manifest["source"]
    print(f"Источник: {source['title']}")
    print(f"Лицензия: {source['license']} — {source['license_url']}")


if __name__ == "__main__":
    main()
