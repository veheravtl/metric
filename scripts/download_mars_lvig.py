#!/usr/bin/env python3
"""Загружает выбранные MARS-LVIG ROS bag с безопасным продолжением.

Файлы имеют размер 9–20 ГБ, поэтому незавершённая загрузка сохраняется с
суффиксом .part и продолжается HTTP Range-запросом. Готовый файл появляется
только после проверки точного размера. SHA-256 вычисляется потоково и
сохраняется рядом для последующего закрепления в проектном манифесте.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/manifests/mars_lvig_hkairport.json"
CHUNK_SIZE_BYTES = 4 * 1024 * 1024
NETWORK_TIMEOUT_SECONDS = 120
MINIMUM_FREE_SPACE_RESERVE_BYTES = 5 * 1024**3
CONTENT_RANGE_PATTERN = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def calculate_sha256(path: Path) -> str:
    """Считает SHA-256 потоково, не загружая многогигабайтный bag в память."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    """Читает проверяемое описание источника и ожидаемых файлов."""

    with path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest.get("sequences"), list):
        raise ValueError("Манифест MARS-LVIG не содержит список sequences")
    return manifest


def find_sequence(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Находит последовательность по точному имени и запрещает неоднозначность."""

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


def parse_content_range(
    value: str | None,
    *,
    expected_start: int,
    expected_total: int,
) -> tuple[int, int, int]:
    """Проверяет, что сервер продолжил именно нужный диапазон того же файла.

    Без этой проверки сервер может проигнорировать Range и прислать файл с
    начала. Простое добавление такого ответа к .part тихо испортило бы bag.
    """

    match = CONTENT_RANGE_PATTERN.fullmatch(value or "")
    if match is None:
        raise RuntimeError(f"Некорректный Content-Range: {value!r}")
    start, end, total = (int(part) for part in match.groups())
    if start != expected_start or end < start or total != expected_total:
        raise RuntimeError(
            "Content-Range не совпал с запрошенным продолжением: "
            f"получено {start}-{end}/{total}, ожидалось начало "
            f"{expected_start} и полный размер {expected_total}"
        )
    return start, end, total


def _response_status(response: BinaryIO) -> int:
    """Возвращает HTTP-статус и даёт понятную ошибку для тестовых ответов."""

    status = getattr(response, "status", None)
    if status is None:
        getcode = getattr(response, "getcode", None)
        status = getcode() if getcode is not None else None
    if not isinstance(status, int):
        raise RuntimeError("HTTP-ответ не содержит числовой status")
    return status


def download_with_resume(
    *,
    url: str,
    destination: Path,
    expected_size_bytes: int,
    expected_sha256: str | None,
    open_request: Callable[..., BinaryIO] = urlopen,
) -> tuple[str, str]:
    """Загружает один bag, сохраняя частичный файл при сетевом сбое.

    Возвращает текстовый статус и фактическую SHA-256. Существующий готовый файл
    никогда не перезаписывается: несовпадение размера или суммы означает
    отдельную диагностическую ошибку.
    """

    if expected_size_bytes <= 0:
        raise ValueError("Ожидаемый размер должен быть положительным")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(f"{destination.name}.part")

    if destination.exists():
        actual_size = destination.stat().st_size
        if actual_size != expected_size_bytes:
            raise RuntimeError(
                f"{destination} уже существует, но имеет размер {actual_size}, "
                f"ожидалось {expected_size_bytes}"
            )
        actual_sha256 = calculate_sha256(destination)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise RuntimeError(f"SHA-256 существующего {destination} не совпала")
        return "уже существует; размер и SHA-256 проверены", actual_sha256

    current_size = partial_path.stat().st_size if partial_path.exists() else 0
    if current_size > expected_size_bytes:
        raise RuntimeError(
            f"Частичный файл больше ожидаемого: {current_size} > "
            f"{expected_size_bytes}"
        )

    if current_size == expected_size_bytes:
        actual_sha256 = calculate_sha256(partial_path)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise RuntimeError(f"SHA-256 завершённого {partial_path} не совпала")
        partial_path.replace(destination)
        return "готовый .part принят после проверки размера", actual_sha256

    remaining_bytes = expected_size_bytes - current_size
    free_bytes = shutil.disk_usage(destination.parent).free
    if free_bytes < remaining_bytes + MINIMUM_FREE_SPACE_RESERVE_BYTES:
        raise RuntimeError(
            "Недостаточно свободного места: нужно не менее "
            f"{remaining_bytes + MINIMUM_FREE_SPACE_RESERVE_BYTES} байт, "
            f"доступно {free_bytes}"
        )

    headers = {"User-Agent": "aerial-map-measurement/0.1"}
    if current_size:
        headers["Range"] = f"bytes={current_size}-"
    request = Request(url, headers=headers)
    with open_request(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            raise RuntimeError(
                "Сервер вернул HTML вместо бинарного набора данных; "
                "частичный файл оставлен без изменений"
            )
        status = _response_status(response)
        if current_size:
            if status != 206:
                raise RuntimeError(
                    "Сервер не подтвердил продолжение HTTP 206; "
                    "частичный файл оставлен без изменений"
                )
            parse_content_range(
                response.headers.get("Content-Range"),
                expected_start=current_size,
                expected_total=expected_size_bytes,
            )
        elif status == 206:
            parse_content_range(
                response.headers.get("Content-Range"),
                expected_start=0,
                expected_total=expected_size_bytes,
            )
        elif status not in (200, 206):
            raise RuntimeError(f"Неожиданный HTTP-статус: {status}")
        content_length = response.headers.get("Content-Length")
        if (
            content_length is not None
            and int(content_length) != remaining_bytes
        ):
            raise RuntimeError(
                "Content-Length ответа не совпал с ожидаемым остатком файла"
            )

        mode = "ab" if current_size else "wb"
        with partial_path.open(mode) as output_file:
            while chunk := response.read(CHUNK_SIZE_BYTES):
                output_file.write(chunk)

    actual_size = partial_path.stat().st_size
    if actual_size != expected_size_bytes:
        raise RuntimeError(
            "Загрузка прервалась до ожидаемого размера; .part сохранён: "
            f"{actual_size} из {expected_size_bytes} байт"
        )

    actual_sha256 = calculate_sha256(partial_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Загруженный файл имеет неверную SHA-256; .part сохранён для анализа"
        )
    partial_path.replace(destination)
    destination.with_suffix(f"{destination.suffix}.sha256").write_text(
        f"{actual_sha256}  {destination.name}\n",
        encoding="ascii",
    )
    return "загружен; размер и SHA-256 проверены", actual_sha256


def parse_args() -> argparse.Namespace:
    """Требует явный список файлов, чтобы случайно не скачать весь набор."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--sequence",
        action="append",
        required=True,
        help="Имя последовательности; параметр можно повторять",
    )
    parser.add_argument(
        "--source",
        choices=("official_rosbag", "huggingface_mcap"),
        default="official_rosbag",
        help="Явно выбрать официальный ROS bag или зафиксированное MCAP-зеркало",
    )
    return parser.parse_args()


def main() -> None:
    """Последовательно загружает выбранные bag и печатает их контрольные суммы."""

    args = parse_args()
    manifest = load_manifest(args.manifest)
    storage_root = PROJECT_ROOT / manifest["storage_directory"]
    for name in args.sequence:
        sequence = find_sequence(manifest, name)
        if args.source == "official_rosbag":
            artifact = sequence
            storage_directory = storage_root / "bags"
        else:
            artifact = sequence["mirrors"]["huggingface_mcap"]
            storage_directory = storage_root / "mcap"
        expected_sha256 = artifact.get("sha256")
        status, actual_sha256 = download_with_resume(
            url=str(artifact["url"]),
            destination=storage_directory / str(artifact["filename"]),
            expected_size_bytes=int(artifact["size_bytes"]),
            expected_sha256=(
                str(expected_sha256) if expected_sha256 is not None else None
            ),
        )
        print(f"{name} [{args.source}]: {status}")
        print(f"{name}: SHA-256 {actual_sha256}")

    print(f"Лицензионный статус: {manifest['license_audit']['status']}")
    print(manifest["license_audit"]["decision_for_this_experiment"])


if __name__ == "__main__":
    main()
