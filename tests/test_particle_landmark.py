"""particle filter のランドマーク観測・再配置テスト。"""

import numpy as np
import pytest

from rikka.common.config import (
    PF_LANDMARK_LIKELIHOOD_FLOOR,
    PF_LANDMARK_MODE,
    PF_LANDMARK_RESET_SIGMA_M,
    PF_LANDMARK_SIGMA_M,
)
from rikka.common.settings import ParticleSettings
from rikka.particle.lib.landmark import (
    landmark_likelihood,
    reset_particles_to_landmark,
)
from rikka.particle.lib.weighting import weight


def test_landmark_likelihood_is_one_at_center_and_approaches_floor() -> None:
    """中心は1、十分遠い粒子は指定した下限になる。"""
    particles = np.asarray([[1.0, 2.0], [1_000.0, 1_000.0]])

    likelihood = landmark_likelihood(particles, (1.0, 2.0), 3.0, 0.05)

    assert likelihood[0] == pytest.approx(1.0)
    assert likelihood[1] == pytest.approx(0.05)


def test_landmark_likelihood_floor_prevents_zero_mass() -> None:
    """全粒子が遠方でも正の尤度を保つ。"""
    particles = np.full((10, 2), 1_000_000.0)

    likelihood = landmark_likelihood(particles, (0.0, 0.0), 1.0, 0.05)

    np.testing.assert_array_equal(likelihood, np.full(10, 0.05))


def test_likelihood_floor_prevents_false_recovery_for_far_particles() -> None:
    """遠方粒子でfloorなしだけが全重み0のrecovery条件になる。"""
    particles = np.full((10, 2), 1_000_000.0)
    prior = np.full(10, 0.1)
    valid = np.ones(10, dtype=bool)
    base = np.ones(10)
    without_floor = weight(
        prior,
        valid,
        base,
        base,
        0.0,
        landmark_likelihood(particles, (0.0, 0.0), 1.0, 0.0),
    )
    with_floor = weight(
        prior,
        valid,
        base,
        base,
        0.0,
        landmark_likelihood(particles, (0.0, 0.0), 1.0, 0.05),
    )

    assert float(without_floor.sum()) == 0.0
    assert float(with_floor.sum()) == pytest.approx(0.05)


def test_weight_without_landmark_preserves_legacy_operation_order() -> None:
    """ランドマーク未指定時は従来式と配列が完全一致する。"""
    prior = np.asarray([0.2, 0.3, 0.5])
    valid = np.asarray([True, False, True])
    stride = np.asarray([0.8, 0.7, 0.6])
    state = np.asarray([0.9, 0.5, 0.4])
    expected = prior * valid.astype(float) * stride * np.power(state, 0.1)

    actual = weight(prior, valid, stride, state, 0.1)

    np.testing.assert_array_equal(actual, expected)


def test_weight_multiplies_landmark_as_fourth_factor() -> None:
    """指定時だけ既存事後重みにランドマーク尤度を掛ける。"""
    prior = np.asarray([0.4, 0.6])
    valid = np.asarray([True, True])
    stride = np.ones(2)
    state = np.ones(2)
    landmark = np.asarray([1.0, 0.25])

    actual = weight(prior, valid, stride, state, 0.0, landmark)

    np.testing.assert_allclose(actual, [0.4, 0.15])


def test_reset_particles_uses_only_walkable_candidates() -> None:
    """reset候補は歩行可能と判定された位置だけを採用する。"""
    particles = reset_particles_to_landmark(
        100,
        (0.0, 0.0),
        1.0,
        np.random.default_rng(0),
        lambda points: points[:, 0] >= 0.0,
    )

    assert np.all(particles[:, 0] >= 0.0)
    assert np.unique(particles, axis=0).shape[0] > 1


def test_particle_landmark_settings_have_provisional_defaults() -> None:
    """PFランドマーク設定がconfigの仮既定値を保持する。"""
    settings = ParticleSettings()

    assert settings.landmark_mode == PF_LANDMARK_MODE == "observation"
    assert settings.landmark_sigma_m == PF_LANDMARK_SIGMA_M == 3.0
    assert settings.landmark_likelihood_floor == PF_LANDMARK_LIKELIHOOD_FLOOR == 0.05
    assert settings.landmark_reset_sigma_m == PF_LANDMARK_RESET_SIGMA_M == 1.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"landmark_mode": "invalid"}, "pf_landmark_mode"),
        ({"landmark_sigma_m": 0.0}, "landmark_sigma_m"),
        ({"landmark_sigma_m": float("nan")}, "landmark_sigma_m"),
        ({"landmark_likelihood_floor": -0.1}, "landmark_likelihood_floor"),
        ({"landmark_likelihood_floor": 1.0}, "landmark_likelihood_floor"),
        ({"landmark_likelihood_floor": float("nan")}, "landmark_likelihood_floor"),
        ({"landmark_reset_sigma_m": 0.0}, "landmark_reset_sigma_m"),
    ],
)
def test_particle_landmark_settings_reject_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    """PFランドマーク設定の選択値と範囲を検証する。"""
    with pytest.raises(ValueError, match=message):
        ParticleSettings(**overrides)  # type: ignore[arg-type]
