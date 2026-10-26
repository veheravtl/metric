"""Тесты RTK-оценки решений MARS-LVIG без запуска тяжёлой DINOv2."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from aerial_mapper.place_verification import CandidateGeometry, VerificationThresholds


def _load_gate_module() -> ModuleType:
    """Загружает эксперимент как модуль, не превращая experiments в пакет."""

    script_path = (
        Path(__file__).parents[1]
        / "experiments/feasibility/mars_lvig_end_to_end_gate.py"
    )
    spec = importlib.util.spec_from_file_location("mars_lvig_gate", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE_MODULE = _load_gate_module()


def _candidate(database_index: int, inliers: int) -> CandidateGeometry:
    """Создаёт геометрически пригодного кандидата с контролируемой силой."""

    return CandidateGeometry(
        database_index=database_index,
        retrieval_rank=database_index + 1,
        retrieval_similarity=0.9 - 0.1 * database_index,
        inlier_count=inliers,
        inlier_ratio=0.8,
        coverage_fraction=0.3,
        reprojection_p95_px=1.0,
        stability_p95_corner_shift_px=2.0,
    )


def _record(normal_inliers: int, hard_inliers: int) -> dict[str, object]:
    """Один близкий нормальный ответ и отдельный кандидат дальше 200 м."""

    return {
        "repeat_image_id": 10,
        "teach_ids": np.asarray([100, 200], dtype=np.int64),
        "distances_meters": np.asarray([5.0, 250.0]),
        "nearest_teach_distance_meters": 5.0,
        "has_positive_in_database": True,
        "positive_in_top_k": True,
        "candidates": [_candidate(0, normal_inliers)],
        "hard_negative_candidates": [_candidate(1, hard_inliers)],
    }


def test_summary_keeps_normal_and_hard_negative_decisions_separate() -> None:
    """Принятие близкого ответа не должно скрыть принятие далёкого контроля."""

    thresholds = VerificationThresholds(4, 0.1, 0.1, 3.0, 10.0, 1.0)

    summary, rows = GATE_MODULE.summarize_decisions(
        [_record(50, 40)],
        thresholds=thresholds,
    )

    assert summary["accepted_within_10m_count"] == 1
    assert summary["catastrophic_false_accept_over_200m_count"] == 0
    assert summary["hard_negative_case_count"] == 1
    assert summary["hard_negative_accept_count"] == 1
    assert rows[0]["hard_negative_accepted"] is True


def test_development_selection_rejects_weak_hard_negative() -> None:
    """Сетка должна предпочесть безопасный порог, сохранив сильный близкий ответ."""

    thresholds, summary, _ = GATE_MODULE.choose_development_thresholds(
        [_record(80, 10)]
    )

    assert thresholds == VerificationThresholds(
        minimum_inlier_count=60,
        minimum_inlier_ratio=0.5,
        minimum_coverage_fraction=0.2,
        maximum_reprojection_p95_px=2.0,
        maximum_stability_p95_corner_shift_px=10.0,
        minimum_winner_to_runner_up_inlier_ratio=1.25,
    )
    assert summary["accepted_within_10m_count"] == 1
    assert summary["hard_negative_accept_count"] == 0
