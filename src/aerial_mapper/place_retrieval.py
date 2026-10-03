"""Компактный поиск места по DINOv2-S и агрегации VLAD.

DINOv2-S превращает каждый участок изображения размером 14 × 14 пикселей в
384-мерный локальный признак. VLAD (Vector of Locally Aggregated Descriptors —
вектор агрегированных локальных описаний) собирает переменное число таких
признаков в один вектор фиксированной длины для поиска по базе кадров.

Модуль намеренно не содержит телеметрии, порогов принятия и геометрической
проверки. Его единственная задача — выдать ранжированный список визуально
похожих кадров. Контрольная истина и SIFT + RANSAC остаются в эксперименте.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float32]


@dataclass(frozen=True)
class DinoV2ExtractionResult:
    """Локальные признаки одного кадра и измеренное время их вычисления."""

    patch_descriptors: FloatMatrix
    processing_time_seconds: float


@dataclass(frozen=True)
class RetrievalResult:
    """Индексы базы в порядке убывания сходства и соответствующие оценки."""

    database_indices: NDArray[np.int64]
    similarities: FloatMatrix


class DinoV2SmallPatchExtractor:
    """Извлекает нормализованные patch-токены официальной DINOv2-S/14.

    Модель загружается через ``torch.hub`` из зафиксированной ревизии
    официального репозитория. Эксперимент передаёт commit из манифеста, поэтому
    последующие изменения ветки ``main`` не меняют воспроизводимый вход.
    """

    def __init__(
        self,
        *,
        repository: str,
        revision: str,
        model_name: str,
        torch_home: Path,
        landscape_size_pixels: tuple[int, int] = (336, 252),
    ) -> None:
        width, height = landscape_size_pixels
        if width <= 0 or height <= 0 or width % 14 != 0 or height % 14 != 0:
            raise ValueError("Размеры входа DINOv2 должны быть кратны 14")

        self._landscape_size_pixels = landscape_size_pixels
        self._cache_signature = f"{model_name}_{revision[:12]}_{width}x{height}"
        torch_home.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(torch_home / "hub"))
        self._model = torch.hub.load(
            f"{repository}:{revision}",
            model_name,
            trust_repo=True,
        ).eval()

    def cache_signature(self) -> str:
        """Идентифицирует модель и размер входа для безопасного кэша."""

        return self._cache_signature

    def _prepare_tensor(self, image_rgb: NDArray[np.uint8]) -> torch.Tensor:
        """Масштабирует RGB-кадр без смены ориентации и нормализует каналы."""

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("DINOv2 ожидает RGB-изображение с тремя каналами")
        if image_rgb.dtype != np.uint8:
            raise ValueError("DINOv2 ожидает изображение типа uint8")

        landscape_width, landscape_height = self._landscape_size_pixels
        if image_rgb.shape[1] >= image_rgb.shape[0]:
            output_size = (landscape_width, landscape_height)
        else:
            output_size = (landscape_height, landscape_width)
        resized_rgb = cv2.resize(image_rgb, output_size, interpolation=cv2.INTER_AREA)
        contiguous_rgb = np.ascontiguousarray(resized_rgb)
        tensor = torch.from_numpy(contiguous_rgb).permute(2, 0, 1).float().div(255.0)
        channel_mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        channel_std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return ((tensor - channel_mean) / channel_std).unsqueeze(0)

    def extract(self, image_rgb: NDArray[np.uint8]) -> DinoV2ExtractionResult:
        """Возвращает массив ``(число фрагментов, 384)`` для одного кадра."""

        input_tensor = self._prepare_tensor(image_rgb)
        started_at = perf_counter()
        with torch.inference_mode():
            features = self._model.forward_features(input_tensor)
            patch_tokens = features["x_norm_patchtokens"]
        elapsed_seconds = perf_counter() - started_at
        descriptors = patch_tokens.squeeze(0).cpu().numpy().astype(np.float32)
        if descriptors.ndim != 2 or not np.all(np.isfinite(descriptors)):
            raise RuntimeError("DINOv2 вернула некорректные локальные признаки")
        return DinoV2ExtractionResult(
            patch_descriptors=descriptors,
            processing_time_seconds=elapsed_seconds,
        )


def fit_visual_vocabulary(
    descriptor_sets: list[FloatMatrix],
    *,
    cluster_count: int,
    maximum_training_descriptors: int = 20_000,
    random_seed: int = 0,
) -> FloatMatrix:
    """Строит визуальный словарь k-means только по эталонным Teach-признакам.

    Ограниченная детерминированная подвыборка удерживает время и память опыта в
    разумных пределах. Repeat-признаки сюда передавать нельзя: это было бы
    использованием тестовых запросов при построении поисковой базы.
    """

    if cluster_count < 2:
        raise ValueError("Визуальный словарь должен содержать минимум два центра")
    if maximum_training_descriptors < cluster_count:
        raise ValueError("Число обучающих признаков должно быть не меньше центров")
    if not descriptor_sets:
        raise ValueError("Для словаря нужен хотя бы один набор признаков")
    if any(descriptors.ndim != 2 for descriptors in descriptor_sets):
        raise ValueError("Каждый набор локальных признаков должен быть матрицей")
    feature_dimensions = {descriptors.shape[1] for descriptors in descriptor_sets}
    if len(feature_dimensions) != 1:
        raise ValueError("Все локальные признаки должны иметь одну размерность")

    all_descriptors = np.concatenate(descriptor_sets, axis=0).astype(np.float32)
    if not np.all(np.isfinite(all_descriptors)):
        raise ValueError("Локальные признаки содержат NaN или бесконечность")
    random_generator = np.random.default_rng(random_seed)
    if all_descriptors.shape[0] > maximum_training_descriptors:
        selected_indices = random_generator.choice(
            all_descriptors.shape[0],
            size=maximum_training_descriptors,
            replace=False,
        )
        training_descriptors = all_descriptors[selected_indices]
    else:
        training_descriptors = all_descriptors

    cv2.setRNGSeed(random_seed)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-4,
    )
    _, _, centers = cv2.kmeans(
        training_descriptors,
        cluster_count,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    return centers.astype(np.float32)


def aggregate_vlad(
    patch_descriptors: FloatMatrix,
    vocabulary_centers: FloatMatrix,
) -> FloatMatrix:
    """Собирает локальные признаки в один L2-нормализованный VLAD-вектор.

    Для каждого локального признака выбирается ближайший центр словаря. Разница
    между признаком и центром суммируется внутри группы. Каждая группа и затем
    весь объединённый вектор нормализуются, чтобы сравнение отражало структуру,
    а не просто число локальных фрагментов.
    """

    if patch_descriptors.ndim != 2 or vocabulary_centers.ndim != 2:
        raise ValueError("Признаки и центры должны быть двумерными матрицами")
    if patch_descriptors.shape[1] != vocabulary_centers.shape[1]:
        raise ValueError("Размерности локальных признаков и центров не совпадают")
    if patch_descriptors.shape[0] == 0 or vocabulary_centers.shape[0] == 0:
        raise ValueError("Признаки и словарь не могут быть пустыми")

    squared_distances = np.sum(
        (patch_descriptors[:, np.newaxis, :] - vocabulary_centers[np.newaxis, :, :])
        ** 2,
        axis=2,
    )
    assignments = np.argmin(squared_distances, axis=1)
    aggregated = np.zeros_like(vocabulary_centers, dtype=np.float32)
    for cluster_index in range(vocabulary_centers.shape[0]):
        cluster_mask = assignments == cluster_index
        if not np.any(cluster_mask):
            continue
        aggregated[cluster_index] = np.sum(
            patch_descriptors[cluster_mask] - vocabulary_centers[cluster_index],
            axis=0,
        )
        cluster_norm = np.linalg.norm(aggregated[cluster_index])
        if cluster_norm > 0.0:
            aggregated[cluster_index] /= cluster_norm

    flattened = aggregated.reshape(-1)
    vector_norm = np.linalg.norm(flattened)
    if vector_norm == 0.0:
        raise ValueError("VLAD-вектор выродился в ноль")
    return (flattened / vector_norm).astype(np.float32)


def rank_database(
    database_descriptors: FloatMatrix,
    query_descriptor: FloatMatrix,
    *,
    maximum_results: int,
) -> RetrievalResult:
    """Точно ранжирует нормализованные векторы по косинусному сходству."""

    if database_descriptors.ndim != 2 or query_descriptor.ndim != 1:
        raise ValueError("База должна быть матрицей, а запрос — одним вектором")
    if database_descriptors.shape[1] != query_descriptor.shape[0]:
        raise ValueError("Размерности базы и запроса не совпадают")
    if not 1 <= maximum_results <= database_descriptors.shape[0]:
        raise ValueError("Некорректное число запрошенных результатов")

    similarities = database_descriptors @ query_descriptor
    ranked_indices = np.argsort(-similarities, kind="stable")[:maximum_results]
    return RetrievalResult(
        database_indices=ranked_indices.astype(np.int64),
        similarities=similarities[ranked_indices].astype(np.float32),
    )
