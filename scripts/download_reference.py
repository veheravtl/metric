#!/usr/bin/env python3
"""Загружает эталонный PoC-растр по параметрам из JSON-манифеста.

Скрипт обращается к стандартной операции Export Map ArcGIS REST API, проверяет
геометрию ответа и самостоятельно записывает GeoTIFF. Это надёжнее ручного
снимка экрана: координатная система и аффинное преобразование становятся частью
файла и не зависят от масштаба окна браузера.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile

from aerial_mapper.reference_grid import ProjectedRasterGrid

DEFAULT_MANIFEST = Path("data/manifests/cherkasy_2021_poc.json")
HTTP_TIMEOUT_SECONDS = 60
GEOMETRY_TOLERANCE_M = 1e-6


def load_manifest(path: Path) -> dict[str, Any]:
    """Читает манифест и возвращает его как словарь."""

    with path.open(encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def build_grid(manifest: dict[str, Any]) -> ProjectedRasterGrid:
    """Строит ожидаемую метрическую сетку из параметров манифеста."""

    selection = manifest["selection"]
    center = selection["center_wgs84"]
    return ProjectedRasterGrid.centered_on_wgs84(
        longitude=center["longitude"],
        latitude=center["latitude"],
        output_crs=selection["output_crs"],
        width_m=selection["width_m"],
        height_m=selection["height_m"],
        resolution_m_per_pixel=selection["resolution_m_per_pixel"],
    )


def request_json(url: str) -> dict[str, Any]:
    """Получает JSON по HTTPS с конечным временем ожидания."""

    request = Request(url, headers={"User-Agent": "aerial-map-measurement/0.1"})
    with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return json.load(response)


def request_bytes(url: str) -> bytes:
    """Получает бинарный файл по HTTPS с конечным временем ожидания."""

    request = Request(url, headers={"User-Agent": "aerial-map-measurement/0.1"})
    with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return response.read()


def build_export_url(manifest: dict[str, Any], grid: ProjectedRasterGrid) -> str:
    """Формирует воспроизводимый запрос к операции ArcGIS Export Map."""

    source = manifest["source"]
    epsg_code = grid.crs.removeprefix("EPSG:")
    query = urlencode(
        {
            "bbox": f"{grid.west},{grid.south},{grid.east},{grid.north}",
            "bboxSR": epsg_code,
            "imageSR": epsg_code,
            "size": f"{grid.width_pixels},{grid.height_pixels}",
            "format": "png32",
            "transparent": "false",
            "layers": f"show:{source['image_layer_id']}",
            "f": "json",
        }
    )
    return f"{source['map_service_url'].rstrip('/')}/export?{query}"


def validate_export_response(
    response: dict[str, Any],
    expected_grid: ProjectedRasterGrid,
) -> None:
    """Отклоняет ответ сервера, если его геометрия отличается от запроса."""

    if "error" in response:
        raise RuntimeError(f"ArcGIS вернул ошибку: {response['error']}")

    if response.get("width") != expected_grid.width_pixels:
        raise RuntimeError("Сервер вернул неожиданную ширину изображения")
    if response.get("height") != expected_grid.height_pixels:
        raise RuntimeError("Сервер вернул неожиданную высоту изображения")

    extent = response["extent"]
    expected_extent = {
        "xmin": expected_grid.west,
        "ymin": expected_grid.south,
        "xmax": expected_grid.east,
        "ymax": expected_grid.north,
    }
    for coordinate_name, expected_value in expected_extent.items():
        actual_value = extent[coordinate_name]
        if abs(actual_value - expected_value) > GEOMETRY_TOLERANCE_M:
            raise RuntimeError(
                "ArcGIS изменил запрошенный охват: "
                f"{coordinate_name}={actual_value}, ожидалось {expected_value}"
            )

    response_wkid = extent["spatialReference"].get("latestWkid")
    expected_wkid = int(expected_grid.crs.removeprefix("EPSG:"))
    if response_wkid != expected_wkid:
        raise RuntimeError(
            f"ArcGIS вернул EPSG:{response_wkid}, ожидался EPSG:{expected_wkid}"
        )


def write_geotiff(
    image_bytes: bytes,
    output_path: Path,
    grid: ProjectedRasterGrid,
    manifest: dict[str, Any],
) -> None:
    """Записывает загруженный PNG как геопривязанный сжатый GeoTIFF."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # PNG закономерно не содержит координатной привязки: её даёт отдельный
    # JSON-ответ ArcGIS, уже проверенный перед вызовом этой функции. Поэтому
    # предупреждение GDAL здесь ожидаемо и подавляется только в узком блоке.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        with MemoryFile(image_bytes) as memory_file, memory_file.open() as source:
            if source.width != grid.width_pixels or source.height != grid.height_pixels:
                raise RuntimeError(
                    "Фактический размер PNG не совпадает с ответом ArcGIS"
                )

            if source.colorinterp[0] == ColorInterp.palette:
                # ArcGIS может вернуть индексированный PNG: каждый пиксель
                # хранит номер цвета, а сами RGB-значения лежат в палитре.
                # Простое копирование одного канала превратило бы цветное
                # изображение в серую карту индексов.
                indices = source.read(1)
                color_map = source.colormap(1)
                palette = np.zeros((256, 3), dtype=np.uint8)
                for color_index, (red, green, blue, _alpha) in color_map.items():
                    palette[color_index] = (red, green, blue)
                bands = np.moveaxis(palette[indices], -1, 0)
            else:
                source_bands = source.read()
                if source_bands.shape[0] >= 3:
                    bands = source_bands[:3]
                elif source_bands.shape[0] == 1:
                    # Настоящий одноканальный источник тоже записываем в RGB,
                    # чтобы последующая обработка имела стабильный контракт.
                    bands = np.repeat(source_bands, 3, axis=0)
                else:
                    raise RuntimeError("PNG не содержит каналов изображения")

    profile = {
        "driver": "GTiff",
        "width": grid.width_pixels,
        "height": grid.height_pixels,
        "count": 3,
        "dtype": bands.dtype,
        "crs": grid.crs,
        "transform": grid.affine_transform,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    # Временный файл создаётся рядом с результатом, поэтому атомарная замена не
    # пересекает границы файловых систем.
    with NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=".tmp.tif",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        with rasterio.open(temporary_path, "w", **profile) as destination:
            destination.write(bands)
            destination.update_tags(
                dataset_id=manifest["id"],
                purpose=manifest["purpose"],
                source_title=manifest["source"]["title"],
                source_url=manifest["source"]["map_service_url"],
                attribution=manifest["source"]["attribution"],
            )
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def sha256(path: Path) -> str:
    """Вычисляет SHA-256 файла порциями, не загружая весь GeoTIFF в память."""

    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_runtime_metadata(
    path: Path,
    *,
    manifest: dict[str, Any],
    grid: ProjectedRasterGrid,
    export_url: str,
    source_png_sha256: str,
    geotiff_sha256: str,
) -> None:
    """Сохраняет локальный протокол конкретной загрузки рядом с растром."""

    metadata = {
        "dataset_id": manifest["id"],
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "export_request_url": export_url,
        "crs": grid.crs,
        "bounds": {
            "west": grid.west,
            "south": grid.south,
            "east": grid.east,
            "north": grid.north,
        },
        "width_pixels": grid.width_pixels,
        "height_pixels": grid.height_pixels,
        "resolution_m_per_pixel": grid.resolution_x_m_per_pixel,
        "source_png_sha256": source_png_sha256,
        "geotiff_sha256": geotiff_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        metadata_file.write("\n")


def parse_args() -> argparse.Namespace:
    """Разбирает минимальный интерфейс командной строки."""

    parser = argparse.ArgumentParser(
        description="Загрузить геопривязанный эталонный растр для PoC",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Путь к JSON-манифесту (по умолчанию {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Разрешить замену уже существующего локального GeoTIFF",
    )
    return parser.parse_args()


def main() -> int:
    """Выполняет загрузку, строгую проверку и запись результата."""

    args = parse_args()
    manifest = load_manifest(args.manifest)
    output_path = Path(manifest["storage"]["raster_path"])
    metadata_path = Path(manifest["storage"]["runtime_metadata_path"])

    if output_path.exists() and not args.force:
        print(
            f"Файл {output_path} уже существует. "
            "Для осознанной замены передайте --force.",
            file=sys.stderr,
        )
        return 2

    grid = build_grid(manifest)
    export_url = build_export_url(manifest, grid)
    export_response = request_json(export_url)
    validate_export_response(export_response, grid)

    image_bytes = request_bytes(export_response["href"])
    write_geotiff(image_bytes, output_path, grid, manifest)

    source_png_hash = hashlib.sha256(image_bytes).hexdigest()
    geotiff_hash = sha256(output_path)
    write_runtime_metadata(
        metadata_path,
        manifest=manifest,
        grid=grid,
        export_url=export_url,
        source_png_sha256=source_png_hash,
        geotiff_sha256=geotiff_hash,
    )

    print(f"Сохранён GeoTIFF: {output_path}")
    print(
        "Охват: "
        f"{grid.west:.3f}, {grid.south:.3f}, "
        f"{grid.east:.3f}, {grid.north:.3f} ({grid.crs})"
    )
    print(
        f"Размер: {grid.width_pixels} x {grid.height_pixels} px; "
        f"{grid.resolution_x_m_per_pixel:.3f} м/пиксель"
    )
    print(f"SHA-256 GeoTIFF: {geotiff_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
