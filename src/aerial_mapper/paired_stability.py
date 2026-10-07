# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Парное управление случайной диагностикой устойчивости гомографии.

Функции этого модуля не оценивают изображение и не знают контрольную истину.
Они обеспечивают две воспроизводимые операции для факторных экспериментов:
одинаковые seed для всех вариантов одной пары сцена/поза и раздельный учёт
опасного принятия хотя бы одним либо всеми повторениями диагностики.
"""

from dataclasses import dataclass


def stability_seeds_for_pair(
    *,
    experiment_seed: int,
    surface_number: int,
    repeat_number: int,
    offsets: list[int],
) -> list[int]:
    """Возвращает seed, зависящие от пары сцена/поза, но не от воздействия.

    Один и тот же список нужно использовать для номинала и всех искажённых
    вариантов пары. Иначе изменение физического фактора смешивается со
    случайной подвыборкой inlier-точек в stability-диагностике.
    """

    base = experiment_seed + surface_number * 100_000 + repeat_number * 10_000
    return [base + int(offset) for offset in offsets]


def gate_safety_flags(
    *, metric_within_limit: bool, accepted_by_seed: list[bool]
) -> tuple[bool, bool]:
    """Различает хотя бы одно и устойчивое по всем seed ложное принятие.

    Пустой список означает, что gate не был вычислен, например из-за отказа
    построения alignment. Такой случай не считается принятым и должен
    учитываться вызывающей стороной как отказ, а не как ложный успех.
    """

    if not accepted_by_seed:
        return False, False
    any_false_accept = not metric_within_limit and any(accepted_by_seed)
    all_false_accept = not metric_within_limit and all(accepted_by_seed)
    return any_false_accept, all_false_accept


def failed_alignment_seed_decisions(
    *, seeds: list[int], failure: str
) -> list[dict[str, object]]:
    """Явно записывает одинаковый ранний отказ для каждого seed.

    Если alignment не построен, случайная stability-диагностика не запускается.
    Однако протокол с фиксированным числом решений должен сохранить факт, что
    каждый заранее назначенный seed закончился безопасным отказом, а не исчез
    из знаменателя.
    """

    return [
        {
            "seed": int(seed),
            "gate_accepted": False,
            "gate_failures": [failure],
            "quality": None,
        }
        for seed in seeds
    ]


@dataclass(frozen=True)
class AggregateGateDecision:
    """Одно итоговое решение по нескольким фиксированным stability-проверкам.

    `minimum_accept_count` является инженерной политикой безопасного отказа,
    а не статистически калиброванной вероятностью. Компонентные решения должны
    быть получены без знания контрольной метрической ошибки.
    """

    gate_accepted: bool
    component_accept_count: int
    component_count: int
    minimum_accept_count: int
    gate_failures: tuple[str, ...]


def aggregate_gate_decisions(
    accepted_by_component: list[bool],
    *,
    minimum_accept_count: int,
) -> AggregateGateDecision:
    """Объединяет фиксированный набор проверок по правилу минимального согласия.

    Например, `minimum_accept_count=5` для пяти компонентов реализует
    единогласие: один сомневающийся компонент переводит итог в безопасный отказ.
    Пустой список запрещён, чтобы ранний отказ нельзя было ошибочно принять.
    """

    if not accepted_by_component:
        raise ValueError("Для агрегированного gate нужен хотя бы один компонент")
    component_count = len(accepted_by_component)
    if not 1 <= minimum_accept_count <= component_count:
        raise ValueError(
            "minimum_accept_count должен лежать между 1 и числом компонентов"
        )
    accept_count = sum(bool(value) for value in accepted_by_component)
    accepted = accept_count >= minimum_accept_count
    failures: tuple[str, ...] = ()
    if not accepted:
        failures = (
            "согласие stability "
            f"{accept_count}/{component_count} < "
            f"{minimum_accept_count}/{component_count}",
        )
    return AggregateGateDecision(
        gate_accepted=accepted,
        component_accept_count=accept_count,
        component_count=component_count,
        minimum_accept_count=minimum_accept_count,
        gate_failures=failures,
    )
