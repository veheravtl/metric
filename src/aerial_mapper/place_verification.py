"""Безопасный выбор одного геометрически подтверждённого эталона.

Грубый поиск места возвращает несколько визуально похожих Teach-кадров. Этот
модуль решает следующую, отдельную задачу: сравнивает доступные без телеметрии
показатели геометрического согласования и либо выбирает один кадр, либо явно
отказывается от ответа.

Модуль намеренно ничего не знает о GPS и правильном номере кадра. Координаты
используются только снаружи, после решения, для честной оценки эксперимента.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateGeometry:
    """Наблюдаемые доказательства для одного кандидата из короткого списка.

    ``retrieval_rank`` начинается с единицы. Остальные величины получают из
    локальных соответствий SIFT и оценки гомографии методом RANSAC. Значение
    ``None`` означает, что матрица не была построена либо показатель нельзя
    было честно вычислить.
    """

    database_index: int
    retrieval_rank: int
    retrieval_similarity: float
    inlier_count: int | None
    inlier_ratio: float | None
    coverage_fraction: float | None
    reprojection_p95_px: float | None
    stability_p95_corner_shift_px: float | None


@dataclass(frozen=True)
class VerificationThresholds:
    """Зафиксированные требования к принимаемому геометрическому ответу.

    Порог отношения оценок победителя и второго кандидата отвечает за
    неоднозначность. Например, значение 1.2 требует, чтобы число согласованных
    соответствий победителя было как минимум на 20 процентов больше.
    """

    minimum_inlier_count: int
    minimum_inlier_ratio: float
    minimum_coverage_fraction: float
    maximum_reprojection_p95_px: float
    maximum_stability_p95_corner_shift_px: float
    minimum_winner_to_runner_up_inlier_ratio: float

    def validate(self) -> None:
        """Не допускает бессмысленные или ослабляющие проверку значения."""

        if self.minimum_inlier_count < 4:
            raise ValueError("Для гомографии нужны минимум четыре inlier-пары")
        if not 0.0 <= self.minimum_inlier_ratio <= 1.0:
            raise ValueError("Доля inlier должна находиться между нулём и единицей")
        if not 0.0 <= self.minimum_coverage_fraction <= 1.0:
            raise ValueError("Покрытие должно находиться между нулём и единицей")
        if self.maximum_reprojection_p95_px <= 0.0:
            raise ValueError("Порог репроекционной ошибки должен быть положительным")
        if self.maximum_stability_p95_corner_shift_px <= 0.0:
            raise ValueError("Порог нестабильности должен быть положительным")
        if self.minimum_winner_to_runner_up_inlier_ratio < 1.0:
            raise ValueError("Победитель не может быть слабее второго кандидата")


@dataclass(frozen=True)
class VerificationDecision:
    """Выбранный кандидат либо мотивированный безопасный отказ."""

    accepted_database_index: int | None
    accepted_retrieval_rank: int | None
    reason: str
    eligible_candidate_count: int
    winner_to_runner_up_inlier_ratio: float | None

    @property
    def accepted(self) -> bool:
        """Сообщает, был ли выдан геометрически подтверждённый ответ."""

        return self.accepted_database_index is not None


def _passes_absolute_thresholds(
    candidate: CandidateGeometry,
    thresholds: VerificationThresholds,
) -> bool:
    """Проверяет одного кандидата без сравнения с конкурентами."""

    values = (
        candidate.inlier_count,
        candidate.inlier_ratio,
        candidate.coverage_fraction,
        candidate.reprojection_p95_px,
        candidate.stability_p95_corner_shift_px,
    )
    if any(value is None for value in values):
        return False
    assert candidate.inlier_count is not None
    assert candidate.inlier_ratio is not None
    assert candidate.coverage_fraction is not None
    assert candidate.reprojection_p95_px is not None
    assert candidate.stability_p95_corner_shift_px is not None
    return (
        candidate.inlier_count >= thresholds.minimum_inlier_count
        and candidate.inlier_ratio >= thresholds.minimum_inlier_ratio
        and candidate.coverage_fraction >= thresholds.minimum_coverage_fraction
        and candidate.reprojection_p95_px <= thresholds.maximum_reprojection_p95_px
        and candidate.stability_p95_corner_shift_px
        <= thresholds.maximum_stability_p95_corner_shift_px
    )


def verify_candidates(
    candidates: list[CandidateGeometry],
    *,
    thresholds: VerificationThresholds,
) -> VerificationDecision:
    """Выбирает самый сильный однозначный кандидат или безопасно отказывает.

    После абсолютных проверок кандидаты сортируются по числу inlier-пар. При
    равенстве используются покрытие кадра, сходство грубого поиска и исходный
    ранг. Такой порядок явно отдаёт приоритет независимому геометрическому
    доказательству, а не нейросетевой похожести изображения.

    Сравнение с занявшим второе место важно для визуально повторяющихся сцен:
    две почти одинаково убедительные матрицы означают неоднозначность, даже если
    каждая по отдельности выглядит качественно.
    """

    thresholds.validate()
    if not candidates:
        return VerificationDecision(None, None, "список кандидатов пуст", 0, None)
    if len({candidate.database_index for candidate in candidates}) != len(candidates):
        raise ValueError("Один индекс базы встретился среди кандидатов несколько раз")

    eligible = [
        candidate
        for candidate in candidates
        if _passes_absolute_thresholds(candidate, thresholds)
    ]
    eligible.sort(
        key=lambda candidate: (
            -(candidate.inlier_count or 0),
            -(candidate.coverage_fraction or 0.0),
            -candidate.retrieval_similarity,
            candidate.retrieval_rank,
        )
    )
    if not eligible:
        return VerificationDecision(
            None,
            None,
            "ни один кандидат не прошёл абсолютные пороги геометрии",
            0,
            None,
        )

    winner = eligible[0]
    if len(eligible) == 1:
        return VerificationDecision(
            winner.database_index,
            winner.retrieval_rank,
            "принят единственный пригодный кандидат",
            1,
            None,
        )

    assert winner.inlier_count is not None
    runner_up = eligible[1]
    assert runner_up.inlier_count is not None
    dominance_ratio = winner.inlier_count / runner_up.inlier_count
    if dominance_ratio < thresholds.minimum_winner_to_runner_up_inlier_ratio:
        return VerificationDecision(
            None,
            None,
            "два лучших кандидата геометрически неоднозначны",
            len(eligible),
            dominance_ratio,
        )
    return VerificationDecision(
        winner.database_index,
        winner.retrieval_rank,
        "принят геометрически доминирующий кандидат",
        len(eligible),
        dominance_ratio,
    )
