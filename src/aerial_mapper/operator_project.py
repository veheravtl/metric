"""Версионированное сохранение операторской разметки в JSON-проект.

Видео и карта остаются внешними файлами. Проект хранит их пути и контрольные
признаки, все маяки, покадровые наблюдения, результаты совмещения и позицию
оператора. Запись через временный файл и os.replace не оставляет на месте
рабочего проекта наполовину записанный JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from aerial_mapper.alignment_session import (
    AlignmentSession,
    FrameAnnotation,
    FrameObservation,
    MachineObservation,
)
from aerial_mapper.frame_quality import FrameImageQuality
from aerial_mapper.manual_alignment import ManualAlignmentResult

PROJECT_SCHEMA_VERSION = 1
PROJECT_FILE_SUFFIX = ".aerial-project.json"


@dataclass(frozen=True)
class OperatorProject:
    """Проверенная рабочая сессия, готовая к восстановлению в GUI."""

    video_source_path: Path
    video_source_type: str
    map_source_path: Path
    video_fingerprint: dict[str, Any]
    map_fingerprint: dict[str, Any]
    current_frame_index: int
    overlay_visible: bool
    session: AlignmentSession
    saved_at_utc: str


def save_operator_project(
    path: str | Path,
    *,
    video_source_path: str | Path,
    video_source_type: str,
    map_source_path: str | Path,
    current_frame_index: int,
    overlay_visible: bool,
    session: AlignmentSession,
) -> Path:
    """Атомарно сохранить проект и вернуть его абсолютный путь."""

    output = Path(path).expanduser().resolve()
    video = Path(video_source_path).expanduser().resolve()
    map_path = Path(map_source_path).expanduser().resolve()
    if video_source_type not in {"video", "flight"}:
        raise ValueError("Тип источника должен быть video или flight")
    if current_frame_index < 0:
        raise ValueError("Индекс текущего кадра не может быть отрицательным")
    document = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "saved_at_utc": datetime.now(UTC).isoformat(),
        "sources": {
            "video": {
                "path": str(video),
                "type": video_source_type,
                "fingerprint": fingerprint_source(video),
            },
            "map": {
                "path": str(map_path),
                "fingerprint": fingerprint_source(map_path),
            },
        },
        "workspace": {
            "current_frame_index": int(current_frame_index),
            "overlay_visible": bool(overlay_visible),
        },
        "alignment_session": _session_to_json(session),
    }
    payload = (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return output


def load_operator_project(path: str | Path) -> OperatorProject:
    """Прочитать и проверить структуру проекта без загрузки изображений."""

    project_path = Path(path).expanduser().resolve()
    try:
        root = json.loads(project_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Файл проекта содержит повреждённый JSON: {error}") from error
    if not isinstance(root, dict):
        raise ValueError("Корень файла проекта должен быть объектом JSON")
    if root.get("schema_version") != PROJECT_SCHEMA_VERSION:
        raise ValueError(
            f"Неподдерживаемая версия проекта {root.get('schema_version')!r}"
        )
    try:
        sources = _mapping(root, "sources")
        video = _mapping(sources, "video")
        map_source = _mapping(sources, "map")
        workspace = _mapping(root, "workspace")
        video_type = _string(video, "type")
        current_frame = int(workspace["current_frame_index"])
        project = OperatorProject(
            video_source_path=Path(_string(video, "path")).expanduser().resolve(),
            video_source_type=video_type,
            map_source_path=Path(_string(map_source, "path")).expanduser().resolve(),
            video_fingerprint=dict(_mapping(video, "fingerprint")),
            map_fingerprint=dict(_mapping(map_source, "fingerprint")),
            current_frame_index=current_frame,
            overlay_visible=bool(workspace["overlay_visible"]),
            session=_session_from_json(_mapping(root, "alignment_session")),
            saved_at_utc=str(root.get("saved_at_utc", "")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Неполная структура файла проекта: {error}") from error
    if video_type not in {"video", "flight"} or current_frame < 0:
        raise ValueError("Проект содержит неверный тип источника или номер кадра")
    return project


def fingerprint_source(path: str | Path) -> dict[str, Any]:
    """Получить быстрый признак файла или каталога пролёта.

    Для большого файла читаются лишь крайние блоки. Это не полный контрольный
    хеш, но он обнаруживает обычную подмену материала без чтения гигабайт.
    """

    source = Path(path).expanduser().resolve()
    if source.is_file():
        stat = source.stat()
        return {
            "kind": "file",
            "size_bytes": stat.st_size,
            "sample_sha256": _sample_sha256(source),
        }
    if source.is_dir():
        candidates = [
            item
            for item in (source / "image_ids.csv", source / "gps.csv")
            if item.is_file()
        ]
        images_dir = source / "images"
        if images_dir.is_dir():
            images = sorted(item for item in images_dir.iterdir() if item.is_file())
            if images:
                candidates.extend([images[0], images[-1]])
        digest = hashlib.sha256()
        for item in dict.fromkeys(candidates):
            digest.update(item.relative_to(source).as_posix().encode("utf-8"))
            digest.update(_sample_sha256(item).encode("ascii"))
        return {
            "kind": "directory",
            "sampled_entry_count": len(dict.fromkeys(candidates)),
            "sample_sha256": digest.hexdigest(),
        }
    raise FileNotFoundError(f"Источник проекта не найден: {source}")


def source_matches_fingerprint(
    path: str | Path, expected: dict[str, Any]
) -> tuple[bool, str]:
    """Сравнить материал с сохранённым признаком и объяснить несовпадение."""

    try:
        actual = fingerprint_source(path)
    except OSError as error:
        return False, str(error)
    if actual.get("kind") != expected.get("kind"):
        return False, "тип источника изменился (файл/каталог)"
    if actual.get("kind") == "file" and actual.get("size_bytes") != expected.get(
        "size_bytes"
    ):
        return False, "размер файла отличается от сохранённого"
    if actual.get("sample_sha256") != expected.get("sample_sha256"):
        return False, "контрольный признак содержимого не совпадает"
    return True, ""


def _sample_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode("ascii"))
    with path.open("rb") as source:
        digest.update(source.read(block_size))
        if size > block_size:
            source.seek(max(0, size - block_size))
            digest.update(source.read(block_size))
    return digest.hexdigest()


def _session_to_json(session: AlignmentSession) -> dict[str, Any]:
    frames = []
    for index in session.frame_indices:
        annotation = session.frame(index)
        if annotation is not None:
            frames.append(_annotation_to_json(annotation))
    return {
        "next_landmark_id": session.next_landmark_id,
        "landmarks": [
            {
                "landmark_id": landmark_id,
                "map_point_px": list(session.landmark(landmark_id).map_point_px),
            }
            for landmark_id in session.landmark_ids
        ],
        "frames": frames,
    }


def _session_from_json(value: dict[str, Any]) -> AlignmentSession:
    landmarks = value.get("landmarks")
    frames = value.get("frames")
    if not isinstance(landmarks, list) or not isinstance(frames, list):
        raise ValueError("Маяки и кадры проекта должны быть списками")
    session = AlignmentSession()
    for item in landmarks:
        session.restore_landmark(
            int(item["landmark_id"]), _point(item["map_point_px"])
        )
    session.reserve_next_landmark_id(int(value.get("next_landmark_id", 0)))
    for item in frames:
        annotation = _annotation_from_json(item)
        session.remember_frame(
            annotation.frame_index,
            annotation.observations,
            alignment=annotation.alignment,
            status=annotation.status,
            suspect_landmark_ids=annotation.suspect_landmark_ids,
            machine_observations=annotation.machine_observations,
            image_quality=annotation.image_quality,
            rejection_reasons=annotation.rejection_reasons,
        )
    return session


def _annotation_to_json(annotation: FrameAnnotation) -> dict[str, Any]:
    return {
        "frame_index": annotation.frame_index,
        "status": annotation.status,
        "observations": [
            {
                "landmark_id": item.landmark_id,
                "frame_point_px": list(item.frame_point_px),
            }
            for item in annotation.observations
        ],
        "suspect_landmark_ids": sorted(annotation.suspect_landmark_ids),
        "alignment": (
            _alignment_to_json(annotation.alignment)
            if annotation.alignment is not None
            else None
        ),
        "machine_observations": [
            _json_safe(asdict(item)) for item in annotation.machine_observations
        ],
        "image_quality": (
            _json_safe(asdict(annotation.image_quality))
            if annotation.image_quality is not None
            else None
        ),
        "rejection_reasons": list(annotation.rejection_reasons),
    }


def _annotation_from_json(value: dict[str, Any]) -> FrameAnnotation:
    alignment = value.get("alignment")
    quality = value.get("image_quality")
    return FrameAnnotation(
        frame_index=int(value["frame_index"]),
        observations=tuple(
            FrameObservation(int(item["landmark_id"]), _point(item["frame_point_px"]))
            for item in value["observations"]
        ),
        alignment=_alignment_from_json(alignment) if alignment is not None else None,
        status=str(value["status"]),  # type: ignore[arg-type]
        suspect_landmark_ids=frozenset(
            int(item) for item in value.get("suspect_landmark_ids", [])
        ),
        machine_observations=tuple(
            MachineObservation(
                source_frame_index=int(item["source_frame_index"]),
                source_frame_point_px=_point(item["source_frame_point_px"]),
                frame_point_px=_point(item["frame_point_px"]),
                map_point_px=_point(item["map_point_px"]),
                forward_backward_error_px=_number(
                    item["forward_backward_error_px"]
                ),
                used_for_alignment=bool(item["used_for_alignment"]),
            )
            for item in value.get("machine_observations", [])
        ),
        image_quality=(
            FrameImageQuality(
                previous_laplacian_variance=_number(
                    quality["previous_laplacian_variance"]
                ),
                current_laplacian_variance=_number(
                    quality["current_laplacian_variance"]
                ),
                sharpness_ratio_to_previous=_number(
                    quality["sharpness_ratio_to_previous"]
                ),
                mean_luminance=_number(quality["mean_luminance"]),
                dark_pixel_fraction=_number(quality["dark_pixel_fraction"]),
                bright_pixel_fraction=_number(quality["bright_pixel_fraction"]),
                warning_reasons=tuple(quality["warning_reasons"]),
                rejection_reasons=tuple(quality["rejection_reasons"]),
            )
            if quality is not None
            else None
        ),
        rejection_reasons=tuple(value.get("rejection_reasons", [])),
    )


_ALIGNMENT_ARRAY_FIELDS = (
    "homography_frame_to_map",
    "frame_points_px",
    "map_points_px",
    "fit_errors_map_px",
    "leave_one_out_errors_map_px",
    "projected_frame_corners_map_px",
)
_ALIGNMENT_NUMBER_FIELDS = ("frame_coverage_fraction", "map_coverage_fraction")


def _alignment_to_json(result: ManualAlignmentResult) -> dict[str, Any]:
    document = {
        name: _json_safe(getattr(result, name)) for name in _ALIGNMENT_ARRAY_FIELDS
    }
    document.update(
        {name: _json_safe(getattr(result, name)) for name in _ALIGNMENT_NUMBER_FIELDS}
    )
    return document


def _alignment_from_json(value: dict[str, Any]) -> ManualAlignmentResult:
    arrays = {
        name: np.asarray(_restore_numbers(value[name]), dtype=np.float64)
        for name in _ALIGNMENT_ARRAY_FIELDS
    }
    numbers = {name: _number(value[name]) for name in _ALIGNMENT_NUMBER_FIELDS}
    return ManualAlignmentResult(**arrays, **numbers)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        return number
    return value


def _restore_numbers(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_numbers(item) for item in value]
    return _number(value)


def _number(value: Any) -> float:
    labels = {"NaN": float("nan"), "Infinity": float("inf"), "-Infinity": float("-inf")}
    if isinstance(value, str) and value in labels:
        return labels[value]
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Нечисловое значение требует явной строковой метки")
    return number


def _point(value: Any) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Точка должна содержать ровно две координаты")
    return _number(value[0]), _number(value[1])


def _mapping(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document[key]
    if not isinstance(value, dict):
        raise TypeError(f"Поле {key!r} должно быть объектом")
    return value


def _string(document: dict[str, Any], key: str) -> str:
    value = document[key]
    if not isinstance(value, str) or not value:
        raise TypeError(f"Поле {key!r} должно быть непустой строкой")
    return value
