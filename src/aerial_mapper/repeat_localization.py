# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Автоматическая цепочка Repeat-кадр → Teach-кадр → растровая карта.

Сохранённый операторский проект уже содержит покадровые гомографии
``Teach → карта``. Поэтому новый пролёт не нужно повторно размечать по карте:
достаточно найти подходящий кадр первичного пролёта и оценить преобразование
``Repeat → Teach``. Композиция двух матриц даёт прямое преобразование
``Repeat → карта`` в исходных пиксельных координатах изображений.

Модуль намеренно поддерживает только извлечённые пролёты MARS-LVIG. Для них
зафиксированы калибровка камеры, подготовка RGB, DINOv2/VLAD-поиск и пороги
геометрической проверки, прошедшие отдельный held-out-эксперимент.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from aerial_mapper.alignment import (
    AlignmentFailure,
    AlignmentResult,
    SiftRansacConfig,
    align_frame_to_reference,
)
from aerial_mapper.alignment_session import AlignmentSession
from aerial_mapper.gui.flight_sequence import MarsLvigFlight
from aerial_mapper.mars_lvig_preprocessing import (
    mars_lvig_camera_calibration,
    prepare_mars_lvig_image,
)
from aerial_mapper.place_retrieval import (
    DinoV2SmallPatchExtractor,
    aggregate_vlad,
    fit_visual_vocabulary,
    rank_database,
)
from aerial_mapper.place_verification import (
    CandidateGeometry,
    VerificationThresholds,
    verify_candidates,
)
from aerial_mapper.quality import analyze_alignment_quality

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/dinov2_vits14.json"
DATASET_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/mars_lvig_hkairport.json"
TORCH_HOME = PROJECT_ROOT / "data/models/torch"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "outputs/cache/mars_lvig_gate"

PREPARED_LONG_SIDE_PIXELS = 672
DINO_INPUT_SIZE_PIXELS = (336, 280)
CLUSTER_COUNT = 8
MAXIMUM_VOCABULARY_DESCRIPTORS = 20_000
MAXIMUM_VISUAL_INDEX_CANDIDATES = 400
TOP_K = 10
RANDOM_SEED = 0

# Эти пороги были выбраны по HKairport02 и до просмотра результата зафиксированы
# для HKairport03. GUI не ослабляет их ради получения красивого ответа.
FROZEN_THRESHOLDS = VerificationThresholds(
    minimum_inlier_count=60,
    minimum_inlier_ratio=0.5,
    minimum_coverage_fraction=0.20,
    maximum_reprojection_p95_px=4.0,
    maximum_stability_p95_corner_shift_px=10.0,
    minimum_winner_to_runner_up_inlier_ratio=1.0,
)

FloatPoints = NDArray[np.float64]
Homography = NDArray[np.float64]


class RepeatLocalizationFailure(RuntimeError):
    """Безопасный отказ, когда Repeat нельзя связать с размеченным Teach."""


@dataclass(frozen=True)
class MappedTeachFrame:
    """Один Teach-кандидат, у которого уже есть путь до карты."""

    database_index: int
    frame_index: int
    image_id: int
    image_path: Path
    homography_teach_to_map: Homography


@dataclass(frozen=True)
class TeachIndexPreparationResult:
    """Результат заблаговременной подготовки визуальной Teach-базы."""

    candidate_count: int
    loaded_from_disk_cache: bool
    processing_time_seconds: float


@dataclass(frozen=True)
class RepeatLocalizationResult:
    """Подтверждённая привязка исходного Repeat-кадра к карте."""

    repeat_frame_index: int
    repeat_image_id: int
    teach_frame_index: int
    teach_image_id: int
    homography_repeat_to_map: Homography
    retrieval_rank: int
    eligible_candidate_count: int
    inlier_count: int
    inlier_ratio: float
    frame_coverage_fraction: float
    reprojection_p95_px: float
    stability_p95_corner_shift_px: float
    processing_time_seconds: float


@dataclass(frozen=True)
class ScaledMapSegmentMeasurement:
    """Отрезок Repeat-кадра, перенесённый на карту с постоянным масштабом."""

    frame_points_px: FloatPoints
    map_points_px: FloatPoints
    length_meters: float


@dataclass
class _PreparedImageStore:
    """Читает исходные кадры и кэширует исправленную 672-px версию."""

    flight: MarsLvigFlight
    cache_directory: Path

    def load(self, frame_index: int) -> NDArray[np.uint8]:
        image_id = int(self.flight.image_ids[frame_index])
        cache_path = self.cache_directory / f"{image_id:06d}.png"
        if cache_path.exists():
            cached_bgr = cv2.imread(str(cache_path), cv2.IMREAD_COLOR)
            if cached_bgr is None:
                raise RepeatLocalizationFailure(
                    f"Не удалось прочитать кэш подготовленного кадра: {cache_path}"
                )
            return cv2.cvtColor(cached_bgr, cv2.COLOR_BGR2RGB)

        source_bgr = self.flight.read_frame_bgr(frame_index)
        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        intrinsics, distortion = mars_lvig_camera_calibration()
        prepared_rgb = prepare_mars_lvig_image(
            source_rgb,
            intrinsics=intrinsics,
            distortion=distortion,
            maximum_side_pixels=PREPARED_LONG_SIDE_PIXELS,
        )
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".png.part")
        success, encoded = cv2.imencode(
            ".png", cv2.cvtColor(prepared_rgb, cv2.COLOR_RGB2BGR)
        )
        if not success:
            raise RepeatLocalizationFailure(
                f"Не удалось закодировать подготовленный кадр {image_id}"
            )
        temporary_path.write_bytes(encoded.tobytes())
        temporary_path.replace(cache_path)
        return prepared_rgb


def compose_repeat_to_map_homography(
    homography_repeat_to_teach: NDArray[np.floating],
    homography_teach_to_map: NDArray[np.floating],
) -> Homography:
    """Составляет две проекции в порядке Repeat → Teach → карта."""

    repeat_to_teach = _validated_homography(
        homography_repeat_to_teach, name="Repeat → Teach"
    )
    teach_to_map = _validated_homography(
        homography_teach_to_map, name="Teach → карта"
    )
    composed = teach_to_map @ repeat_to_teach
    return _validated_homography(composed, name="Repeat → карта")


def measure_scaled_map_segment(
    frame_points_px: NDArray[np.floating] | list[tuple[float, float]],
    homography_repeat_to_map: NDArray[np.floating],
    *,
    map_width_pixels: int,
    map_height_pixels: int,
    map_scale_meters_per_pixel: float,
) -> ScaledMapSegmentMeasurement:
    """Переносит два клика на карту и измеряет длину в локальных метрах.

    Постоянный масштаб допустим для сохранённого плоского снимка карты. Он не
    означает, что тот же масштаб постоянен на перспективном Repeat-кадре:
    перспективу сначала устраняет гомография.
    """

    points = np.asarray(frame_points_px, dtype=np.float64)
    if points.shape != (2, 2) or not np.all(np.isfinite(points)):
        raise RepeatLocalizationFailure(
            "Для отрезка нужны ровно две конечные точки кадра"
        )
    if map_width_pixels <= 1 or map_height_pixels <= 1:
        raise RepeatLocalizationFailure("Размер карты должен быть больше 1 px")
    scale = float(map_scale_meters_per_pixel)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RepeatLocalizationFailure("Масштаб карты должен быть положительным")

    homography = _validated_homography(
        homography_repeat_to_map, name="Repeat → карта"
    )
    mapped = cv2.perspectiveTransform(
        points.reshape(1, -1, 2), homography
    ).reshape(-1, 2)
    if not np.all(np.isfinite(mapped)):
        raise RepeatLocalizationFailure("Измеряемая точка перенесена в бесконечность")
    outside = (
        (mapped[:, 0] < 0.0)
        | (mapped[:, 0] > map_width_pixels - 1.0)
        | (mapped[:, 1] < 0.0)
        | (mapped[:, 1] > map_height_pixels - 1.0)
    )
    if np.any(outside):
        raise RepeatLocalizationFailure(
            "Хотя бы одна точка отрезка перенесена за границы карты"
        )
    length_pixels = float(np.linalg.norm(mapped[1] - mapped[0]))
    return ScaledMapSegmentMeasurement(
        frame_points_px=points,
        map_points_px=mapped.astype(np.float64),
        length_meters=length_pixels * scale,
    )


def mapped_teach_frames(
    session: AlignmentSession,
    flight: MarsLvigFlight,
    *,
    spaced_image_ids: NDArray[np.integer] | list[int],
) -> tuple[MappedTeachFrame, ...]:
    """Оставляет из разреженной базы только кадры с подтверждённой картой."""

    frame_by_image_id = {
        int(image_id): frame_index
        for frame_index, image_id in enumerate(flight.image_ids)
    }
    records: list[MappedTeachFrame] = []
    for image_id_value in spaced_image_ids:
        image_id = int(image_id_value)
        frame_index = frame_by_image_id.get(image_id)
        if frame_index is None:
            continue
        annotation = session.frame(frame_index)
        if annotation is None or annotation.alignment is None:
            continue
        records.append(
            MappedTeachFrame(
                database_index=len(records),
                frame_index=frame_index,
                image_id=image_id,
                image_path=flight.image_paths[frame_index],
                homography_teach_to_map=annotation.alignment.homography_frame_to_map,
            )
        )
    if not records:
        raise RepeatLocalizationFailure(
            "В проекте нет привязанных Teach-кадров из поисковой базы"
        )
    return tuple(records)


def limit_mapped_teach_frames(
    records: tuple[MappedTeachFrame, ...],
    *,
    maximum_count: int = MAXIMUM_VISUAL_INDEX_CANDIDATES,
) -> tuple[MappedTeachFrame, ...]:
    """Равномерно ограничить базу по порядку кадров, не используя координаты.

    Вся Teach-разметка может содержать тысячи почти соседних кадров. Для
    поиска места они дают мало новой информации, но заметно удлиняют первый
    прогрев на CPU. Равномерная выборка сохраняет начало, конец и покрытие
    всего размеченного маршрута. После выбора индексы пересоздаются, чтобы
    строка матрицы поиска однозначно совпадала с записью кандидата.
    """

    if maximum_count <= 0:
        raise ValueError(
            "Максимальное число Teach-кандидатов должно быть положительным"
        )
    if len(records) <= maximum_count:
        selected = records
    else:
        selected_indices = np.linspace(
            0,
            len(records) - 1,
            maximum_count,
            dtype=np.int64,
        )
        selected = tuple(records[int(index)] for index in selected_indices)
    return tuple(
        MappedTeachFrame(
            database_index=database_index,
            frame_index=record.frame_index,
            image_id=record.image_id,
            image_path=record.image_path,
            homography_teach_to_map=record.homography_teach_to_map,
        )
        for database_index, record in enumerate(selected)
    )


class RepeatLocalizationService:
    """Лениво строит Teach-индекс и привязывает выбранный Repeat-кадр."""

    def __init__(
        self,
        *,
        teach_flight_directory: str | Path,
        session: AlignmentSession,
        cache_root: str | Path = DEFAULT_CACHE_ROOT,
    ) -> None:
        self._teach_directory = Path(teach_flight_directory).expanduser().resolve()
        self._session = session
        self._cache_root = Path(cache_root).expanduser().resolve()
        self._teach_flight: MarsLvigFlight | None = None
        self._teach_store: _PreparedImageStore | None = None
        self._extractor: DinoV2SmallPatchExtractor | None = None
        self._vocabulary: NDArray[np.float32] | None = None
        self._database_descriptors: NDArray[np.float32] | None = None
        self._records: tuple[MappedTeachFrame, ...] = ()
        self._preparation_signature = self._camera_signature()
        self._repeat_flights: dict[Path, MarsLvigFlight] = {}
        self._index_loaded_from_disk_cache = False

    @property
    def mapped_candidate_count(self) -> int:
        """Число Teach-кандидатов после построения индекса."""

        return len(self._records)

    def prepare_index(self) -> TeachIndexPreparationResult:
        """Подготовить визуальную поисковую базу до наступления события.

        Метод не читает координаты Teach или Repeat. База состоит только из
        уже привязанных к карте Teach-кадров, а сходство впоследствии считается
        исключительно по содержимому изображений.
        """

        started_at = perf_counter()
        self._ensure_index()
        return TeachIndexPreparationResult(
            candidate_count=len(self._records),
            loaded_from_disk_cache=self._index_loaded_from_disk_cache,
            processing_time_seconds=perf_counter() - started_at,
        )

    def localize(
        self,
        repeat_flight_directory: str | Path,
        repeat_frame_index: int,
    ) -> RepeatLocalizationResult:
        """Найти и проверить Teach для одного выбранного Repeat-кадра."""

        started_at = perf_counter()
        self._ensure_index()
        assert self._extractor is not None
        assert self._vocabulary is not None
        assert self._database_descriptors is not None
        assert self._teach_flight is not None
        assert self._teach_store is not None

        repeat_directory = Path(repeat_flight_directory).expanduser().resolve()
        repeat_flight = self._repeat_flights.get(repeat_directory)
        if repeat_flight is None:
            repeat_flight = MarsLvigFlight.load(
                repeat_directory, load_positions=False
            )
            self._repeat_flights[repeat_directory] = repeat_flight
        if not 0 <= repeat_frame_index < repeat_flight.frame_count:
            raise RepeatLocalizationFailure("Индекс Repeat-кадра вне пролёта")

        repeat_store = _PreparedImageStore(
            repeat_flight,
            self._prepared_cache_directory(repeat_directory.name),
        )
        repeat_image_id = int(repeat_flight.image_ids[repeat_frame_index])
        repeat_prepared = repeat_store.load(repeat_frame_index)
        query_patches = self._load_or_extract_patches(
            repeat_prepared,
            sequence_name=repeat_directory.name,
            image_id=repeat_image_id,
        )
        query_descriptor = aggregate_vlad(query_patches, self._vocabulary)
        retrieval = rank_database(
            self._database_descriptors,
            query_descriptor,
            maximum_results=min(TOP_K, len(self._records)),
        )

        evidence: list[CandidateGeometry] = []
        alignments: dict[int, AlignmentResult] = {}
        for retrieval_rank, (database_index, similarity) in enumerate(
            zip(
                retrieval.database_indices,
                retrieval.similarities,
                strict=True,
            ),
            start=1,
        ):
            record = self._records[int(database_index)]
            teach_prepared = self._teach_store.load(record.frame_index)
            candidate, alignment = self._geometry_evidence(
                teach_prepared,
                repeat_prepared,
                database_index=int(database_index),
                retrieval_rank=retrieval_rank,
                retrieval_similarity=float(similarity),
            )
            evidence.append(candidate)
            if alignment is not None:
                alignments[int(database_index)] = alignment

        decision = verify_candidates(evidence, thresholds=FROZEN_THRESHOLDS)
        if decision.accepted_database_index is None:
            raise RepeatLocalizationFailure(
                "Стоп-кадр не привязан: " + decision.reason
            )
        selected_index = decision.accepted_database_index
        selected_record = self._records[selected_index]

        # Поиск и защитный выбор выполнялись на исправленных уменьшенных RGB,
        # как в held-out-гейте. Для точной композиции с операторской гомографией
        # повторяем SIFT только для выбранной пары в исходных пикселях: обе
        # стороны сняты одной камерой, поэтому получаем прямую raw→raw матрицу.
        teach_raw_bgr = self._teach_flight.read_frame_bgr(
            selected_record.frame_index
        )
        repeat_raw_bgr = repeat_flight.read_frame_bgr(repeat_frame_index)
        try:
            raw_alignment = align_frame_to_reference(
                cv2.cvtColor(teach_raw_bgr, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(repeat_raw_bgr, cv2.COLOR_BGR2RGB),
                config=SiftRansacConfig(),
            )
        except AlignmentFailure as error:
            raise RepeatLocalizationFailure(
                f"Выбранный Teach не подтвердился в исходных пикселях: {error}"
            ) from error
        raw_quality = analyze_alignment_quality(
            raw_alignment,
            frame_width_pixels=repeat_raw_bgr.shape[1],
            frame_height_pixels=repeat_raw_bgr.shape[0],
            random_seed=RANDOM_SEED,
        )
        raw_stability = raw_quality.stability_p95_max_corner_shift_reference_px
        raw_evidence = CandidateGeometry(
            database_index=selected_index,
            retrieval_rank=decision.accepted_retrieval_rank or 1,
            retrieval_similarity=0.0,
            inlier_count=raw_alignment.inlier_count,
            inlier_ratio=raw_alignment.inlier_ratio,
            coverage_fraction=raw_alignment.inlier_spatial_coverage_fraction,
            reprojection_p95_px=raw_quality.inlier_reprojection_p95_reference_px,
            stability_p95_corner_shift_px=raw_stability,
        )
        raw_decision = verify_candidates(
            [raw_evidence], thresholds=FROZEN_THRESHOLDS
        )
        if not raw_decision.accepted:
            raise RepeatLocalizationFailure(
                "Выбранная пара не прошла защитные пороги в исходных пикселях"
            )
        assert raw_stability is not None

        repeat_to_map = compose_repeat_to_map_homography(
            raw_alignment.homography_frame_to_reference,
            selected_record.homography_teach_to_map,
        )
        return RepeatLocalizationResult(
            repeat_frame_index=int(repeat_frame_index),
            repeat_image_id=repeat_image_id,
            teach_frame_index=selected_record.frame_index,
            teach_image_id=selected_record.image_id,
            homography_repeat_to_map=repeat_to_map,
            retrieval_rank=decision.accepted_retrieval_rank or 1,
            eligible_candidate_count=decision.eligible_candidate_count,
            inlier_count=raw_alignment.inlier_count,
            inlier_ratio=raw_alignment.inlier_ratio,
            frame_coverage_fraction=raw_alignment.inlier_spatial_coverage_fraction,
            reprojection_p95_px=(
                raw_quality.inlier_reprojection_p95_reference_px
            ),
            stability_p95_corner_shift_px=raw_stability,
            processing_time_seconds=perf_counter() - started_at,
        )

    def _ensure_index(self) -> None:
        if self._database_descriptors is not None:
            return
        self._teach_flight = MarsLvigFlight.load(
            self._teach_directory, load_positions=False
        )

        # Рабочая локализация не читает GPS. Искать имеет смысл только среди
        # Teach-кадров с подтверждённой гомографией до карты: любой иной
        # кандидат всё равно не позволяет выполнить последующий замер.
        self._records = limit_mapped_teach_frames(
            mapped_teach_frames(
                self._session,
                self._teach_flight,
                spaced_image_ids=self._teach_flight.image_ids,
            )
        )
        self._teach_store = _PreparedImageStore(
            self._teach_flight,
            self._prepared_cache_directory(self._teach_directory.name),
        )

        model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
        self._verify_model_weights(model_manifest)
        model = model_manifest["model"]
        source = model_manifest["source"]
        self._extractor = DinoV2SmallPatchExtractor(
            repository=str(source["torch_hub_repository"]),
            revision=str(source["repository_revision"]),
            model_name=str(model["torch_hub_name"]),
            torch_home=TORCH_HOME,
            landscape_size_pixels=DINO_INPUT_SIZE_PIXELS,
        )
        self._verify_model_weights(model_manifest)

        indexed_image_ids = np.asarray(
            [record.image_id for record in self._records], dtype=np.int64
        )
        index_cache_path = self._visual_index_cache_path(indexed_image_ids)
        cached_index = _load_visual_search_index(
            index_cache_path,
            expected_image_ids=indexed_image_ids,
        )
        if cached_index is not None:
            self._vocabulary, self._database_descriptors = cached_index
            self._index_loaded_from_disk_cache = True
            return

        # Словарь строится только по доступным для измерения Teach-кадрам.
        # Это чисто визуальная база: ни положение камеры, ни GPS Repeat-кадра
        # не входят ни в дескриптор, ни в ранжирование.
        patches_by_image_id: dict[int, NDArray[np.float32]] = {}
        all_patches: list[NDArray[np.float32]] = []
        for record in self._records:
            prepared = self._teach_store.load(record.frame_index)
            patches = self._load_or_extract_patches(
                prepared,
                sequence_name=self._teach_directory.name,
                image_id=record.image_id,
            )
            patches_by_image_id[record.image_id] = patches
            all_patches.append(patches)
        self._vocabulary = fit_visual_vocabulary(
            all_patches,
            cluster_count=CLUSTER_COUNT,
            maximum_training_descriptors=MAXIMUM_VOCABULARY_DESCRIPTORS,
            random_seed=RANDOM_SEED,
        )
        self._database_descriptors = np.stack(
            [
                aggregate_vlad(
                    patches_by_image_id[record.image_id], self._vocabulary
                )
                for record in self._records
            ]
        )
        _save_visual_search_index(
            index_cache_path,
            image_ids=indexed_image_ids,
            vocabulary=self._vocabulary,
            database_descriptors=self._database_descriptors,
        )
        self._index_loaded_from_disk_cache = False

    def _geometry_evidence(
        self,
        teach_rgb: NDArray[np.uint8],
        repeat_rgb: NDArray[np.uint8],
        *,
        database_index: int,
        retrieval_rank: int,
        retrieval_similarity: float,
    ) -> tuple[CandidateGeometry, AlignmentResult | None]:
        try:
            alignment = align_frame_to_reference(
                teach_rgb, repeat_rgb, config=SiftRansacConfig()
            )
        except AlignmentFailure:
            return (
                CandidateGeometry(
                    database_index=database_index,
                    retrieval_rank=retrieval_rank,
                    retrieval_similarity=retrieval_similarity,
                    inlier_count=None,
                    inlier_ratio=None,
                    coverage_fraction=None,
                    reprojection_p95_px=None,
                    stability_p95_corner_shift_px=None,
                ),
                None,
            )
        height, width = repeat_rgb.shape[:2]
        quality = analyze_alignment_quality(
            alignment,
            frame_width_pixels=width,
            frame_height_pixels=height,
            random_seed=RANDOM_SEED,
        )
        return (
            CandidateGeometry(
                database_index=database_index,
                retrieval_rank=retrieval_rank,
                retrieval_similarity=retrieval_similarity,
                inlier_count=alignment.inlier_count,
                inlier_ratio=alignment.inlier_ratio,
                coverage_fraction=alignment.inlier_spatial_coverage_fraction,
                reprojection_p95_px=(
                    quality.inlier_reprojection_p95_reference_px
                ),
                stability_p95_corner_shift_px=(
                    quality.stability_p95_max_corner_shift_reference_px
                ),
            ),
            alignment,
        )

    def _load_or_extract_patches(
        self,
        prepared_rgb: NDArray[np.uint8],
        *,
        sequence_name: str,
        image_id: int,
    ) -> NDArray[np.float32]:
        assert self._extractor is not None
        cache_directory = (
            self._cache_root
            / "dinov2"
            / self._extractor.cache_signature
            / self._preparation_signature
            / sequence_name
        )
        cache_path = cache_directory / f"{image_id:06d}.npy"
        if cache_path.exists():
            descriptors = np.load(cache_path, allow_pickle=False)
            if descriptors.ndim != 2 or not np.all(np.isfinite(descriptors)):
                raise RepeatLocalizationFailure(
                    f"Некорректный DINOv2-кэш: {cache_path}"
                )
            return descriptors.astype(np.float32)
        result = self._extractor.extract(prepared_rgb)
        cache_directory.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, result.patch_descriptors, allow_pickle=False)
        return result.patch_descriptors

    def _visual_index_cache_path(
        self, indexed_image_ids: NDArray[np.int64]
    ) -> Path:
        """Получить имя кэша, зависящее от модели, подготовки и состава базы."""

        assert self._extractor is not None
        signature_payload = {
            "schema_version": 1,
            "sequence_name": self._teach_directory.name,
            "image_ids": indexed_image_ids.tolist(),
            "extractor": self._extractor.cache_signature,
            "preparation": self._preparation_signature,
            "cluster_count": CLUSTER_COUNT,
            "maximum_vocabulary_descriptors": MAXIMUM_VOCABULARY_DESCRIPTORS,
            "random_seed": RANDOM_SEED,
        }
        serialized = json.dumps(
            signature_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(serialized).hexdigest()[:20]
        return (
            self._cache_root
            / "visual_index"
            / self._extractor.cache_signature
            / self._preparation_signature
            / f"{self._teach_directory.name}_{digest}.npz"
        )

    def _prepared_cache_directory(self, sequence_name: str) -> Path:
        return (
            self._cache_root
            / "prepared"
            / self._preparation_signature
            / sequence_name
        )

    @staticmethod
    def _camera_signature() -> str:
        # Повторяем подпись held-out-эксперимента буквально, включая имена
        # полей manifest. Иначе тот же кадр попал бы в новый каталог кэша и
        # дорогое извлечение DINOv2 пришлось бы выполнять заново.
        manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
        serialized = json.dumps(manifest["camera"], sort_keys=True).encode()
        return (
            f"long{PREPARED_LONG_SIDE_PIXELS}_"
            f"{hashlib.sha256(serialized).hexdigest()[:12]}"
        )

    @staticmethod
    def _verify_model_weights(manifest: dict[str, Any]) -> None:
        model = manifest["model"]
        weights_path = TORCH_HOME / "hub/checkpoints" / model["weights_filename"]
        if not weights_path.exists():
            return
        expected_size = int(model["weights_size_bytes"])
        if weights_path.stat().st_size != expected_size:
            raise RepeatLocalizationFailure(
                f"Не совпадает размер весов DINOv2: {weights_path}"
            )
        digest = hashlib.sha256()
        with weights_path.open("rb") as weights_file:
            for block in iter(lambda: weights_file.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != str(model["weights_sha256"]):
            raise RepeatLocalizationFailure(
                f"Не совпадает SHA-256 весов DINOv2: {weights_path}"
            )


def _load_visual_search_index(
    cache_path: Path,
    *,
    expected_image_ids: NDArray[np.int64],
) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
    """Загрузить целый индекс и строго проверить его форму и порядок строк.

    Строка VLAD-базы связана с Teach-кадром только своим индексом. Поэтому
    перестановка идентификаторов опаснее обычной ошибки чтения: она могла бы
    дать правдоподобную, но неверную гомографию. Такой кэш явно отклоняется.
    """

    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            schema_version = np.asarray(archive["schema_version"])
            cached_image_ids = np.asarray(archive["image_ids"], dtype=np.int64)
            vocabulary = np.asarray(archive["vocabulary"], dtype=np.float32)
            database = np.asarray(
                archive["database_descriptors"], dtype=np.float32
            )
    except (OSError, ValueError, KeyError) as error:
        raise RepeatLocalizationFailure(
            f"Не удалось прочитать кэш визуального Teach-индекса: {cache_path}"
        ) from error

    expected_ids = np.asarray(expected_image_ids, dtype=np.int64)
    expected_descriptor_width = (
        CLUSTER_COUNT * vocabulary.shape[1] if vocabulary.ndim == 2 else -1
    )
    valid = (
        schema_version.shape == (1,)
        and int(schema_version[0]) == 1
        and cached_image_ids.ndim == 1
        and np.array_equal(cached_image_ids, expected_ids)
        and vocabulary.ndim == 2
        and vocabulary.shape[0] == CLUSTER_COUNT
        and vocabulary.shape[1] > 0
        and database.shape == (expected_ids.size, expected_descriptor_width)
        and np.all(np.isfinite(vocabulary))
        and np.all(np.isfinite(database))
    )
    if not valid:
        raise RepeatLocalizationFailure(
            f"Некорректный кэш визуального Teach-индекса: {cache_path}"
        )
    return vocabulary, database


def _save_visual_search_index(
    cache_path: Path,
    *,
    image_ids: NDArray[np.int64],
    vocabulary: NDArray[np.float32],
    database_descriptors: NDArray[np.float32],
) -> None:
    """Атомарно сохранить словарь, базу и порядок Teach-кадров.

    Сначала записывается соседний временный файл. Замена конечного имени
    выполняется только после успешного закрытия архива, поэтому аварийное
    завершение не оставляет частично записанный индекс под рабочим именем.
    """

    normalized_ids = np.asarray(image_ids, dtype=np.int64)
    normalized_vocabulary = np.asarray(vocabulary, dtype=np.float32)
    normalized_database = np.asarray(database_descriptors, dtype=np.float32)
    expected_descriptor_width = (
        CLUSTER_COUNT * normalized_vocabulary.shape[1]
        if normalized_vocabulary.ndim == 2
        else -1
    )
    if (
        normalized_ids.ndim != 1
        or normalized_vocabulary.ndim != 2
        or normalized_vocabulary.shape[0] != CLUSTER_COUNT
        or normalized_vocabulary.shape[1] <= 0
        or normalized_database.shape
        != (normalized_ids.size, expected_descriptor_width)
        or not np.all(np.isfinite(normalized_vocabulary))
        or not np.all(np.isfinite(normalized_database))
    ):
        raise RepeatLocalizationFailure(
            "Нельзя сохранить некорректный визуальный Teach-индекс"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".npz.part")
    with temporary_path.open("wb") as temporary_file:
        np.savez_compressed(
            temporary_file,
            schema_version=np.asarray([1], dtype=np.int64),
            image_ids=normalized_ids,
            vocabulary=normalized_vocabulary,
            database_descriptors=normalized_database,
        )
    temporary_path.replace(cache_path)


def _validated_homography(
    values: NDArray[np.floating], *, name: str
) -> Homography:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise RepeatLocalizationFailure(f"Матрица {name} должна быть конечной 3×3")
    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0:
        raise RepeatLocalizationFailure(f"Матрица {name} вырождена")
    normalized = matrix / scale
    determinant = float(np.linalg.det(normalized))
    if not np.isfinite(determinant) or abs(determinant) < 1e-12:
        raise RepeatLocalizationFailure(f"Матрица {name} вырождена")
    if np.isclose(matrix[2, 2], 0.0):
        return normalized
    return matrix / matrix[2, 2]
