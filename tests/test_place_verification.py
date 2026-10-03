"""Тесты безопасного выбора кандидата после грубого поиска места."""

import pytest

from aerial_mapper.place_verification import (
    CandidateGeometry,
    VerificationThresholds,
    verify_candidates,
)

THRESHOLDS = VerificationThresholds(
    minimum_inlier_count=20,
    minimum_inlier_ratio=0.5,
    minimum_coverage_fraction=0.2,
    maximum_reprojection_p95_px=3.0,
    maximum_stability_p95_corner_shift_px=10.0,
    minimum_winner_to_runner_up_inlier_ratio=1.2,
)


def candidate(
    database_index: int,
    *,
    inliers: int = 40,
    coverage: float = 0.4,
    similarity: float = 0.8,
) -> CandidateGeometry:
    """Создаёт компактное заведомо пригодное свидетельство для тестов."""

    return CandidateGeometry(
        database_index=database_index,
        retrieval_rank=database_index + 1,
        retrieval_similarity=similarity,
        inlier_count=inliers,
        inlier_ratio=0.7,
        coverage_fraction=coverage,
        reprojection_p95_px=2.0,
        stability_p95_corner_shift_px=4.0,
    )


def test_verifier_prefers_stronger_geometry_over_retrieval_similarity() -> None:
    """Локальная геометрия должна исправлять ошибочный первый результат поиска."""

    decision = verify_candidates(
        [
            candidate(0, inliers=30, similarity=0.95),
            candidate(1, inliers=60, similarity=0.80),
        ],
        thresholds=THRESHOLDS,
    )

    assert decision.accepted_database_index == 1
    assert decision.accepted_retrieval_rank == 2
    assert decision.winner_to_runner_up_inlier_ratio == pytest.approx(2.0)


def test_verifier_rejects_two_similarly_strong_candidates() -> None:
    """Две убедительные версии места безопаснее признать неоднозначными."""

    decision = verify_candidates(
        [candidate(0, inliers=50), candidate(1, inliers=45)],
        thresholds=THRESHOLDS,
    )

    assert not decision.accepted
    assert "неоднозначны" in decision.reason


def test_verifier_rejects_candidate_with_localized_inliers() -> None:
    """Много точек в малой области не должно подтверждать весь кадр."""

    decision = verify_candidates(
        [candidate(0, inliers=100, coverage=0.05)],
        thresholds=THRESHOLDS,
    )

    assert not decision.accepted
    assert decision.eligible_candidate_count == 0


def test_verifier_rejects_missing_stability_measurement() -> None:
    """Отсутствие проверки устойчивости нельзя подменять нулевой ошибкой."""

    incomplete = CandidateGeometry(
        database_index=0,
        retrieval_rank=1,
        retrieval_similarity=0.9,
        inlier_count=50,
        inlier_ratio=0.8,
        coverage_fraction=0.5,
        reprojection_p95_px=2.0,
        stability_p95_corner_shift_px=None,
    )

    decision = verify_candidates([incomplete], thresholds=THRESHOLDS)

    assert not decision.accepted


def test_thresholds_forbid_runner_up_stronger_than_winner() -> None:
    """Отношение меньше единицы меняло бы смысл проверки неоднозначности."""

    invalid = VerificationThresholds(
        minimum_inlier_count=20,
        minimum_inlier_ratio=0.5,
        minimum_coverage_fraction=0.2,
        maximum_reprojection_p95_px=3.0,
        maximum_stability_p95_corner_shift_px=10.0,
        minimum_winner_to_runner_up_inlier_ratio=0.9,
    )

    with pytest.raises(ValueError, match="слабее"):
        verify_candidates([candidate(0)], thresholds=invalid)
