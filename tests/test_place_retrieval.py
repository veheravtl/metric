"""Тесты математического ядра облегчённого DINOv2 + VLAD-поиска."""

import numpy as np
import pytest

from aerial_mapper.place_retrieval import (
    aggregate_vlad,
    fit_visual_vocabulary,
    rank_database,
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
