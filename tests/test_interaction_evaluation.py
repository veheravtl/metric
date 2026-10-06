"""Проверки классификации продуктового gate для G12."""

import pytest

from aerial_mapper.interaction_evaluation import summarize_product_gate


def test_small_errors_and_correct_signs_are_accepted_correctly() -> None:
    """Gate и скрытая метрика должны согласиться на безопасном ответе."""

    summary = summarize_product_gate(
        [0.04, 0.06, 0.08],
        x_sign_correct=[True, True, True],
        y_sign_correct=[None, None, None],
        gate_accepted=True,
        maximum_vector_error_p95_m=0.25,
    )
    assert summary.classification == "accepted_correct"
    assert summary.product_within_limits is True
    assert summary.vector_error_maximum_m == pytest.approx(0.08)


def test_gate_acceptance_above_product_limit_is_false_accept() -> None:
    """Хорошая внутренняя геометрия не должна скрыть метрическую ошибку."""

    summary = summarize_product_gate(
        [0.10, 0.50],
        x_sign_correct=[True, True],
        y_sign_correct=[True, True],
        gate_accepted=True,
        maximum_vector_error_p95_m=0.25,
    )
    assert summary.classification == "false_accept"
    assert summary.product_within_limits is False


def test_wrong_nonzero_component_sign_is_critical() -> None:
    """Даже малая длина ошибки не оправдывает перестановку лево/право."""

    summary = summarize_product_gate(
        [0.02, 0.03],
        x_sign_correct=[False, True],
        y_sign_correct=[True, True],
        gate_accepted=True,
        maximum_vector_error_p95_m=0.25,
    )
    assert summary.classification == "false_accept"
    assert summary.x_sign_error_count == 1


def test_conservative_rejection_is_distinguished_from_invalid_result() -> None:
    """Отказ от точного ответа и отказ от плохого ответа имеют разный смысл."""

    valid = summarize_product_gate(
        [0.05, 0.08],
        x_sign_correct=[True, True],
        y_sign_correct=[True, True],
        gate_accepted=False,
        maximum_vector_error_p95_m=0.25,
    )
    invalid = summarize_product_gate(
        [0.4, 0.6],
        x_sign_correct=[True, True],
        y_sign_correct=[True, True],
        gate_accepted=False,
        maximum_vector_error_p95_m=0.25,
    )
    assert valid.classification == "rejected_valid"
    assert invalid.classification == "rejected_invalid"


def test_empty_error_sample_is_rejected() -> None:
    """Пустую выборку нельзя превращать в правдоподобный успешный итог."""

    with pytest.raises(ValueError, match="непустой"):
        summarize_product_gate(
            [],
            x_sign_correct=[],
            y_sign_correct=[],
            gate_accepted=True,
            maximum_vector_error_p95_m=0.25,
        )
