"""Проверки агрегированного gate и смешанной модели G14-C2/G14-D."""

import json
from pathlib import Path

import numpy as np

from aerial_mapper.paired_stability import aggregate_gate_decisions
from aerial_mapper.synthetic_robustness import (
    ImageDegradation,
    apply_image_degradation,
)
from aerial_mapper.video_artifacts import (
    OSD_RANDOM_SEED_OFFSET,
    VideoArtifactProfile,
    apply_osd_occlusion,
    apply_video_artifact_profile,
    jpeg_round_trip,
)


def test_unanimous_gate_rejects_when_one_component_disagrees() -> None:
    """Правило 5/5 обязано безопасно отказать даже при четырёх принятиях."""

    decision = aggregate_gate_decisions(
        [True, True, True, True, False],
        minimum_accept_count=5,
    )
    assert decision.gate_accepted is False
    assert decision.component_accept_count == 4
    assert decision.component_count == 5
    assert decision.gate_failures == ("согласие stability 4/5 < 5/5",)


def test_aggregate_gate_accepts_required_consensus() -> None:
    """Агрегатор поддерживает общий порог, хотя frozen C2 использует 5/5."""

    decision = aggregate_gate_decisions(
        [True, False, True, False, True],
        minimum_accept_count=3,
    )
    assert decision.gate_accepted is True
    assert decision.component_accept_count == 3
    assert decision.gate_failures == ()


def test_aggregate_gate_rejects_empty_or_impossible_rule() -> None:
    """Пустой набор и недостижимый порог не могут стать неявным успехом."""

    for values, minimum in [([], 1), ([True, False], 0), ([True, False], 3)]:
        try:
            aggregate_gate_decisions(values, minimum_accept_count=minimum)
        except ValueError:
            pass
        else:
            raise AssertionError("Некорректное правило должно вызвать ValueError")


def test_identity_video_profile_preserves_every_channel() -> None:
    """Номинал без JPEG обязан быть побитово тождественным."""

    image = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    result = apply_video_artifact_profile(
        image,
        VideoArtifactProfile(),
        random_seed=123,
    )
    assert np.array_equal(result.image_rgb, image)
    assert result.metadata["maximum_channel_difference"] == 0
    assert result.metadata["jpeg_encoded_bytes"] is None
    assert result.metadata["actual_osd_occlusion_fraction"] == 0.0


def test_mixed_video_profile_follows_frozen_operation_order() -> None:
    """Общая функция должна совпадать с явной цепочкой из пяти операций."""

    generator = np.random.default_rng(17)
    image = generator.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    profile = VideoArtifactProfile(
        blur_sigma_px=0.8,
        resolution_scale=0.5,
        noise_standard_deviation=7.0,
        osd_occlusion_fraction=0.08,
        jpeg_quality=40,
    )
    random_seed = 991
    actual = apply_video_artifact_profile(
        image,
        profile,
        random_seed=random_seed,
    )

    optical_sensor = apply_image_degradation(
        image,
        ImageDegradation(
            blur_sigma_px=profile.blur_sigma_px,
            resolution_scale=profile.resolution_scale,
            noise_standard_deviation=profile.noise_standard_deviation,
        ),
        random_seed=random_seed,
    )
    with_osd = apply_osd_occlusion(
        optical_sensor,
        fraction=profile.osd_occlusion_fraction,
        random_seed=random_seed + OSD_RANDOM_SEED_OFFSET,
    )
    expected = jpeg_round_trip(with_osd.image_rgb, quality=40)

    assert np.array_equal(actual.image_rgb, expected.image_rgb)
    assert actual.metadata["artifact_order"] == [
        "blur",
        "resolution",
        "noise",
        "osd",
        "jpeg",
    ]
    assert actual.metadata["jpeg_encoded_bytes"] == expected.encoded_bytes


def test_followup_configs_freeze_expected_case_counts() -> None:
    """Матрицы C2 и D должны совпадать с заранее объявленными знаменателями."""

    c2 = json.loads(
        Path("experiments/configs/synthetic_3d_g14c2_noise_consensus.json").read_text(
            encoding="utf-8"
        )
    )
    d = json.loads(
        Path("experiments/configs/synthetic_3d_g14d_video_mixtures.json").read_text(
            encoding="utf-8"
        )
    )

    pair_count = len(c2["surface_ids"]) * len(c2["repeat_camera_ids"])
    c2_per_pair = 1 + len(c2["noise_standard_deviation_levels"]) * len(
        c2["artifact_replicate_offsets"]
    )
    d_per_pair = 1 + len(d["profiles"]) * len(d["artifact_replicate_offsets"])

    assert pair_count == 9
    assert c2_per_pair == 28
    assert pair_count * c2_per_pair == 252
    assert pair_count * c2_per_pair * 5 == 1260
    assert d_per_pair == 13
    assert pair_count * d_per_pair == 117
    assert pair_count * d_per_pair * 5 == 585
    assert c2["aggregate_gate"]["minimum_accept_count"] == 5
    assert d["aggregate_gate"]["minimum_accept_count"] == 5
