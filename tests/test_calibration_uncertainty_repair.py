"""Проверки замороженного repair-протокола G14-A2-R."""

import json
from pathlib import Path

from aerial_mapper.paired_stability import (
    gate_safety_flags,
    stability_seeds_for_pair,
)


def test_stability_seeds_depend_only_on_scene_pose_pair() -> None:
    """Парные варианты обязаны получать один набор случайных подвыборок."""

    seeds = stability_seeds_for_pair(
        experiment_seed=20261403,
        surface_number=2,
        repeat_number=1,
        offsets=[0, 1009, 2027, 4093, 8191],
    )
    assert seeds == [20471403, 20472412, 20473430, 20475496, 20479594]


def test_gate_safety_flags_separate_any_and_all_seed_failures() -> None:
    """Один опасный seed и устойчивое принятие не должны смешиваться."""

    assert gate_safety_flags(
        metric_within_limit=False,
        accepted_by_seed=[False, True, False, False, False],
    ) == (True, False)
    assert gate_safety_flags(
        metric_within_limit=False,
        accepted_by_seed=[True, True, True, True, True],
    ) == (True, True)
    assert gate_safety_flags(
        metric_within_limit=True,
        accepted_by_seed=[True, True, True, True, True],
    ) == (False, False)


def test_g14a2r_protocol_freezes_case_and_seed_counts() -> None:
    """Repair не может удалять физические случаи или неудобный stability seed."""

    repair_path = Path(
        "experiments/configs/synthetic_3d_g14a2r_paired_stability.json"
    )
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    source_path = Path(repair["source_experiment_config"])
    source = json.loads(source_path.read_text(encoding="utf-8"))

    assert repair["stability_seed_offsets"] == [0, 1009, 2027, 4093, 8191]
    assert repair["stability_trials_per_seed"] == 30
    assert repair["stability_subsample_fraction"] == 0.8
    pair_count = len(source["surface_ids"]) * len(source["repeat_camera_ids"])
    cases_per_pair = (
        1
        + len(source["raw_boundary_k1"])
        + len(source["calibration_true_k1"])
        * len(source["calibration_error_k1"])
    )
    assert pair_count * cases_per_pair == 423
    assert pair_count * cases_per_pair * len(repair["stability_seed_offsets"]) == 2115
    assert all(value == 0 for value in repair["preregistered_safety"].values())
