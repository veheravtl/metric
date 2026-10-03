#!/usr/bin/env python3
"""Загружает и выборочно распаковывает CLOUD UTIAS Field trial 1.

Архив проверяется по зафиксированной SHA-256. Из него извлекаются PNG и CSV,
необходимые для smoke test; дублирующие AVI пропускаются. Внешние данные не
коммитятся в репозиторий.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/manifests/cloud_utiasfield_trial1.json"
CHUNK_SIZE_BYTES = 1024 * 1024
NETWORK_TIMEOUT_SECONDS = 120


def calculate_sha256(path: Path) -> str:
    """Считает контрольную сумму потоково, не загружая архив в память."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    """Читает происхождение, контрольную сумму и правила извлечения."""

    with path.open(encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def download_archive(
    *,
    url: str,
    destination: Path,
    expected_size_bytes: int,
    expected_sha256: str,
    force: bool,
) -> str:
    """Атомарно загружает архив либо проверяет уже существующий файл."""

    if destination.exists() and not force:
        actual_hash = calculate_sha256(destination)
        if (
            destination.stat().st_size == expected_size_bytes
            and actual_hash == expected_sha256
        ):
            return "уже существует; размер и SHA-256 совпадают"
        raise RuntimeError(
            f"Существующий {destination} не совпал с манифестом. "
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
        total_size = 0
        try:
            with urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
                while chunk := response.read(CHUNK_SIZE_BYTES):
                    temporary_file.write(chunk)
                    digest.update(chunk)
                    total_size += len(chunk)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    if total_size != expected_size_bytes or digest.hexdigest() != expected_sha256:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Загруженный архив не совпал с размером или SHA-256 из манифеста"
        )
    temporary_path.replace(destination)
    return "загружен; размер и SHA-256 совпадают"


def _selected_relative_path(
    member_name: str,
    *,
    archive_root: str,
    include_patterns: list[str],
) -> PurePosixPath | None:
    """Убирает корень ZIP и принимает только явно разрешённые пути."""

    member = PurePosixPath(member_name)
    if member.is_absolute() or ".." in member.parts:
        raise RuntimeError(f"Небезопасный путь внутри ZIP: {member_name}")
    if not member.parts or member.parts[0] != archive_root:
        return None
    relative = PurePosixPath(*member.parts[1:])
    if not relative.parts or member_name.endswith("/"):
        return None
    relative_text = relative.as_posix()
    if not any(fnmatch.fnmatch(relative_text, pattern) for pattern in include_patterns):
        return None
    return relative


def extract_selected_files(
    archive_path: Path,
    *,
    destination: Path,
    archive_root: str,
    include_patterns: list[str],
    force: bool,
) -> tuple[int, int]:
    """Извлекает разрешённые файлы атомарно и не перезаписывает их молча."""

    extracted_count = 0
    existing_count = 0
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP повреждён; первый ошибочный файл: {bad_member}")
        for member_info in archive.infolist():
            relative = _selected_relative_path(
                member_info.filename,
                archive_root=archive_root,
                include_patterns=include_patterns,
            )
            if relative is None:
                continue
            output_path = destination.joinpath(*relative.parts)
            if output_path.exists() and not force:
                if output_path.stat().st_size == member_info.file_size:
                    existing_count += 1
                    continue
                raise RuntimeError(
                    f"{output_path} существует, но размер отличается; "
                    "передайте --force только после проверки причины"
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with (
                archive.open(member_info) as source,
                NamedTemporaryFile(
                    dir=output_path.parent,
                    prefix=f".{output_path.name}.",
                    suffix=".extract",
                    delete=False,
                ) as temporary_file,
            ):
                temporary_path = Path(temporary_file.name)
                try:
                    shutil.copyfileobj(source, temporary_file)
                except Exception:
                    temporary_path.unlink(missing_ok=True)
                    raise
            temporary_path.replace(output_path)
            extracted_count += 1
    return extracted_count, existing_count


def parse_args() -> argparse.Namespace:
    """Разбирает путь к манифесту и осознанное разрешение перезаписи."""

    parser = argparse.ArgumentParser(
        description="Загрузить и распаковать CLOUD UTIAS Field trial 1"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Повторно загрузить архив и заменить извлекаемые файлы",
    )
    return parser.parse_args()


def main() -> None:
    """Проверяет архив, извлекает минимальный набор и печатает ограничения."""

    args = parse_args()
    manifest = load_manifest(args.manifest)
    storage_directory = PROJECT_ROOT / manifest["storage_directory"]
    archive_description = manifest["archive"]
    archive_path = storage_directory / archive_description["filename"]
    status = download_archive(
        url=archive_description["url"],
        destination=archive_path,
        expected_size_bytes=int(archive_description["size_bytes"]),
        expected_sha256=archive_description["sha256"],
        force=args.force,
    )
    extraction = manifest["extraction"]
    destination = storage_directory / extraction["directory"]
    extracted_count, existing_count = extract_selected_files(
        archive_path,
        destination=destination,
        archive_root=extraction["directory"],
        include_patterns=list(extraction["include"]),
        force=args.force,
    )

    print(f"{archive_path.relative_to(PROJECT_ROOT)}: {status}")
    print(f"Извлечено файлов: {extracted_count}; уже были полными: {existing_count}")
    print(f"Лицензионный статус: {manifest['license_audit']['status']}")
    print(manifest["license_audit"]["decision_for_this_experiment"])


if __name__ == "__main__":
    main()
