"""Тесты математического ядра облегчённого DINOv2 + VLAD-поиска."""

import numpy as np
import pytest

from aerial_mapper.place_retrieval import (
    DinoV2SmallPatchExtractor,
    aggregate_vlad,
    fit_visual_vocabulary,
    rank_database,
    score_similarity_sequences,
)


def test_aggregate_vlad_accumulates_residuals_per_visual_word() -> None:
    """VLAD должен хранить направления отклонений отдельно для двух групп."""

    centers = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    descriptors = np.asarray([[1.0, 0.0], [2.0, 0.0], [9.0, 0.0]], dtype=np.float32)

    result = aggregate_vlad(descriptors, centers)

    expected_scale = 1.0 / np.sqrt(2.0)
    assert result == pytest.approx(
        [expected_scale, 0.0, -expected_scale, 0.0],
        abs=1e-6,
    )
    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_rank_database_returns_exact_cosine_order() -> None:
    """Поиск должен возвращать ближайшие нормализованные векторы без индекса."""

    database = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [np.sqrt(0.5), np.sqrt(0.5)]],
        dtype=np.float32,
    )
    query = np.asarray([1.0, 0.0], dtype=np.float32)

    result = rank_database(database, query, maximum_results=3)

    assert result.database_indices.tolist() == [0, 2, 1]
    assert result.similarities == pytest.approx([1.0, np.sqrt(0.5), 0.0])


def test_visual_vocabulary_is_reproducible() -> None:
    """Фиксированный seed должен давать те же центры на тех же признаках."""

    random_generator = np.random.default_rng(17)
    descriptors = random_generator.normal(size=(200, 4)).astype(np.float32)

    first = fit_visual_vocabulary([descriptors], cluster_count=4, random_seed=3)
    second = fit_visual_vocabulary([descriptors], cluster_count=4, random_seed=3)

    assert first == pytest.approx(second)


def test_sequence_score_suppresses_isolated_false_peak() -> None:
    """Одиночный яркий кандидат не должен победить связную диагональ."""

    similarities = np.zeros((5, 7), dtype=np.float32)
    for query_index in range(5):
        similarities[query_index, query_index + 1] = 0.7
    similarities[2, 5] = 0.99

    result = score_similarity_sequences(
        similarities,
        window_length=3,
        velocity_ratios=(0.0, 1.0),
        directions=(1,),
    )

    center_row = 1  # исходный query index 2 после удаления краёв окна
    assert np.argmax(similarities[2]) == 5
    assert np.argmax(result.similarities[center_row]) == 3
    assert result.similarities[center_row, 3] == pytest.approx(0.7)


def test_sequence_score_recognizes_reverse_route_direction() -> None:
    """Встречный Repeat-проход должен давать направление -1 без помощи GPS."""

    similarities = np.zeros((5, 7), dtype=np.float32)
    for query_index in range(5):
        similarities[query_index, 5 - query_index] = 0.8

    result = score_similarity_sequences(
        similarities,
        window_length=5,
        velocity_ratios=(0.0, 1.0),
    )

    assert result.query_center_indices.tolist() == [2]
    assert np.argmax(result.similarities[0]) == 3
    assert result.best_directions[0, 3] == -1
    assert result.best_velocity_ratios[0, 3] == pytest.approx(1.0)


def test_sequence_score_rejects_even_window() -> None:
    """Чётное окно не имеет одного однозначного центрального запроса."""

    with pytest.raises(ValueError, match="нечётной"):
        score_similarity_sequences(
            np.ones((5, 4), dtype=np.float32),
            window_length=4,
            velocity_ratios=(1.0,),
        )


def test_extractor_exposes_cache_signature_as_value() -> None:
    """Эксперименту нужен текст пути, а не связанный метод объекта."""

    extractor = DinoV2SmallPatchExtractor.__new__(DinoV2SmallPatchExtractor)
    extractor._cache_signature = "model_revision_336x252"

    assert extractor.cache_signature == "model_revision_336x252"
