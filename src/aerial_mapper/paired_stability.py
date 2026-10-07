# SPDX-FileCopyrightText: 2026 vehera
# SPDX-License-Identifier: GPL-3.0-or-later

"""Парное управление случайной диагностикой устойчивости гомографии.

Функции этого модуля не оценивают изображение и не знают контрольную истину.
Они обеспечивают две воспроизводимые операции для факторных экспериментов:
одинаковые seed для всех вариантов одной пары сцена/поза и раздельный учёт
опасного принятия хотя бы одним либо всеми повторениями диагностики.
"""


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
