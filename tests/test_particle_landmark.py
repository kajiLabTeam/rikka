"""particle filter のランドマーク観測・再配置テスト。"""

import numpy as np
import pytest

from rikka.common.config import (
    BLE_ANCHOR_WARN_JUMP_M,
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    PF_LANDMARK_LIKELIHOOD_FLOOR,
    PF_LANDMARK_MAX_JUMP_M,
    PF_LANDMARK_MODE,
    PF_LANDMARK_RESET_HEADING_SIGMA,
    PF_LANDMARK_RESET_MIN_DISTANCE_M,
    PF_LANDMARK_RESET_SIGMA_M,
    PF_LANDMARK_RESET_SPREAD_RATIO,
    PF_LANDMARK_SIGMA_M,
)
from rikka.common.lib.models import FloorMap, LandmarkDetection, TrajectoryResult
from rikka.common.lib.sensors import load_sensor_data
from rikka.common.settings import BleLandmarkSettings, ParticleSettings, PdrSettings
from rikka.particle.lib.landmark import (
    landmark_likelihood,
    reset_particles_to_landmark,
)
from rikka.particle.lib.weighting import weight
from rikka.particle.pipeline import run_particle
from rikka.pdr.pipeline import run_pdr


@pytest.fixture(scope="module")
def particle_landmark_results() -> dict[str, TrajectoryResult]:
    """固定入力・seedでランドマークなし、none、observationを実行する。"""
    data_dir = "input/sensor_data/natsuki/1turn_rightsidestep_3turn_leftsidestep"
    df_acc, df_gyro = load_sensor_data(data_dir)
    prepared = run_pdr(PdrSettings(), df_acc, df_gyro).prepared
    floormap = FloorMap(FLOORMAP_PATH, FLOORMAP_ORIGIN_PX, FLOORMAP_SCALE)
    landmark_settings = BleLandmarkSettings()
    detection = LandmarkDetection(
        prepared.t_at_steps[20],
        "beacon_1",
        -45.0,
    )
    hybrid_detection = LandmarkDetection(
        prepared.t_at_steps[17],
        "beacon_1",
        -45.0,
    )
    far_detection = LandmarkDetection(
        prepared.t_at_steps[0],
        "beacon_2",
        -45.0,
    )
    base_settings = {"count": 120, "seed": 0}
    return {
        "baseline": run_particle(
            prepared,
            floormap,
            ParticleSettings(landmark_mode="none", **base_settings),
        ),
        "none": run_particle(
            prepared,
            floormap,
            ParticleSettings(landmark_mode="none", **base_settings),
            detections=(detection,),
            landmark_settings=landmark_settings,
        ),
        "observation": run_particle(
            prepared,
            floormap,
            ParticleSettings(landmark_mode="observation", **base_settings),
            detections=(detection,),
            landmark_settings=landmark_settings,
        ),
        "reset": run_particle(
            prepared,
            floormap,
            ParticleSettings(landmark_mode="reset", **base_settings),
            detections=(detection,),
            landmark_settings=landmark_settings,
        ),
        "hybrid": run_particle(
            prepared,
            floormap,
            ParticleSettings(landmark_mode="hybrid", **base_settings),
            detections=(hybrid_detection,),
            landmark_settings=landmark_settings,
        ),
        "reset_skipped": run_particle(
            prepared,
            floormap,
            ParticleSettings(landmark_mode="reset", **base_settings),
            detections=(far_detection,),
            landmark_settings=landmark_settings,
        ),
        "hybrid_near": run_particle(
            prepared,
            floormap,
            ParticleSettings(
                landmark_mode="hybrid",
                landmark_reset_min_distance_m=1000.0,
                count=120,
                seed=0,
            ),
            detections=(hybrid_detection,),
            landmark_settings=landmark_settings,
        ),
        "reset_near": run_particle(
            prepared,
            floormap,
            ParticleSettings(
                landmark_mode="reset",
                landmark_reset_min_distance_m=1000.0,
                count=120,
                seed=0,
            ),
            detections=(detection,),
            landmark_settings=landmark_settings,
        ),
    }


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

    assert settings.landmark_mode == PF_LANDMARK_MODE == "hybrid"
    assert settings.landmark_sigma_m == PF_LANDMARK_SIGMA_M == 1.0
    assert settings.landmark_likelihood_floor == PF_LANDMARK_LIKELIHOOD_FLOOR == 0.05
    assert settings.landmark_reset_sigma_m == PF_LANDMARK_RESET_SIGMA_M == 1.0
    assert settings.landmark_max_jump_m == PF_LANDMARK_MAX_JUMP_M == 5.0
    assert settings.landmark_reset_spread_ratio == PF_LANDMARK_RESET_SPREAD_RATIO == 4.0
    assert (
        settings.landmark_reset_min_distance_m
        == PF_LANDMARK_RESET_MIN_DISTANCE_M
        == 2.0
    )
    assert (
        settings.landmark_reset_heading_sigma == PF_LANDMARK_RESET_HEADING_SIGMA == 0.20
    )
    assert settings.landmark_anchor_warn_jump_m == BLE_ANCHOR_WARN_JUMP_M == 20.0


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
        ({"landmark_max_jump_m": 0.0}, "landmark_max_jump_m"),
        ({"landmark_reset_spread_ratio": 0.0}, "landmark_reset_spread_ratio"),
        ({"landmark_reset_min_distance_m": -0.1}, "landmark_reset_min_distance_m"),
        ({"landmark_reset_heading_sigma": -0.1}, "landmark_reset_heading_sigma"),
        ({"landmark_anchor_warn_jump_m": 0.0}, "landmark_anchor_warn_jump_m"),
    ],
)
def test_particle_landmark_settings_reject_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    """PFランドマーク設定の選択値と範囲を検証する。"""
    with pytest.raises(ValueError, match=message):
        ParticleSettings(**overrides)  # type: ignore[arg-type]


def test_landmark_mode_none_matches_baseline_exactly(
    particle_landmark_results: dict[str, TrajectoryResult],
) -> None:
    """検出を渡してもnoneなら従来PFの全粒子と代表軌跡が完全一致する。"""
    baseline = particle_landmark_results["baseline"]
    disabled = particle_landmark_results["none"]

    np.testing.assert_array_equal(baseline.trajectory, disabled.trajectory)
    assert baseline.particle is not None
    assert disabled.particle is not None
    np.testing.assert_array_equal(
        np.asarray(baseline.particle.all_particles),
        np.asarray(disabled.particle.all_particles),
    )


def test_observation_keeps_step_continuity_around_landmark(
    particle_landmark_results: dict[str, TrajectoryResult],
) -> None:
    """observationのランドマーク歩が通常歩から突出した変位や旋回にならない。"""
    result = particle_landmark_results["observation"]
    assert result.landmark is not None
    event = result.landmark.corrections[0]
    points = np.asarray(result.trajectory, dtype=float)
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    turns = np.abs(
        np.angle(
            np.exp(
                1j
                * np.diff(
                    np.arctan2(vectors[:, 1], vectors[:, 0]),
                )
            )
        )
    )
    event_step = event.step_index

    assert lengths[event_step] <= 2.0 * float(np.median(lengths))
    assert turns[event_step - 1] <= float(np.quantile(turns, 0.95))


def test_observation_does_not_exceed_particle_spread_shift(
    particle_landmark_results: dict[str, TrajectoryResult],
) -> None:
    """observationによる代表点差が粒子群の広がりから大きく逸脱しない。"""
    baseline = particle_landmark_results["none"]
    observed = particle_landmark_results["observation"]
    assert observed.landmark is not None
    assert observed.particle is not None
    event = observed.landmark.corrections[0]
    diagnostic = observed.particle.diagnostics[event.step_index]
    shift = np.linalg.norm(
        np.asarray(observed.trajectory[event.step_index + 1])
        - np.asarray(baseline.trajectory[event.step_index + 1])
    )

    assert shift <= 4.0 * diagnostic.position_spread_rms_m


def test_landmark_step_does_not_trigger_recovery(
    particle_landmark_results: dict[str, TrajectoryResult],
) -> None:
    """ランドマーク尤度の反映だけで地図recoveryを誤作動させない。"""
    result = particle_landmark_results["observation"]
    assert result.landmark is not None
    assert result.particle is not None
    event = result.landmark.corrections[0]
    diagnostic = result.particle.diagnostics[event.step_index]

    assert not diagnostic.recovery_attempted
    assert diagnostic.recovery_mode == "none"


def test_reset_jump_distance_is_bounded(
    particle_landmark_results: dict[str, TrajectoryResult],
) -> None:
    """resetの代表軌跡ジャンプが上限内で直後に大きく折り返さない。"""
    result = particle_landmark_results["reset"]
    assert result.landmark is not None
    event = result.landmark.corrections[0]
    points = np.asarray(result.trajectory, dtype=float)
    vectors = np.diff(points, axis=0)
    jump = float(np.linalg.norm(vectors[event.step_index]))
    headings = np.arctan2(vectors[:, 1], vectors[:, 0])
    next_turn = abs(
        float(
            np.angle(
                np.exp(
                    1j * (headings[event.step_index + 1] - headings[event.step_index])
                )
            )
        )
    )

    assert jump <= ParticleSettings().landmark_max_jump_m
    assert np.degrees(next_turn) < 135.0


def test_hybrid_resets_when_landmark_exceeds_particle_spread(
    particle_landmark_results: dict[str, TrajectoryResult],
) -> None:
    """hybridは上限内で粒子群から遠いランドマークをresetとして反映する。"""
    result = particle_landmark_results["hybrid"]
    assert result.landmark is not None
    assert result.particle is not None
    event = result.landmark.corrections[0]
    diagnostic = result.particle.diagnostics[event.step_index]

    assert event.applied
    assert diagnostic.recovery_mode == "landmark_reset"


def test_hybrid_keeps_observation_for_near_landmark(
    particle_landmark_results: dict[str, TrajectoryResult],
) -> None:
    """ばら撒き幅より誤差が小さいランドマークはresetせず観測尤度で扱う。"""
    result = particle_landmark_results["hybrid_near"]
    assert result.landmark is not None
    assert result.particle is not None
    event = result.landmark.corrections[0]
    diagnostic = result.particle.diagnostics[event.step_index]
    points = np.asarray(result.trajectory, dtype=float)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)

    assert diagnostic.recovery_mode == "none"
    assert not diagnostic.resampled or diagnostic.recovery_mode == "none"
    assert steps[event.step_index] <= 2.0 * float(np.median(steps))


def test_reset_skips_when_error_is_smaller_than_scatter(
    particle_landmark_results: dict[str, TrajectoryResult],
) -> None:
    """reset方式でも誤差が撒き直し幅以下ならresetせず不確かさを増やさない。"""
    result = particle_landmark_results["reset_near"]
    assert result.landmark is not None
    assert result.particle is not None
    event = result.landmark.corrections[0]
    diagnostic = result.particle.diagnostics[event.step_index]

    assert diagnostic.recovery_mode == "none"
    assert not event.applied


def test_reset_over_max_jump_is_skipped(
    particle_landmark_results: dict[str, TrajectoryResult],
) -> None:
    """上限を超えるresetは未適用として残し通常のPF遷移を維持する。"""
    baseline = particle_landmark_results["baseline"]
    result = particle_landmark_results["reset_skipped"]
    assert result.landmark is not None
    assert result.particle is not None
    event = result.landmark.corrections[0]
    diagnostic = result.particle.diagnostics[event.step_index]

    assert not event.applied
    assert diagnostic.recovery_mode == "none"
    np.testing.assert_array_equal(result.trajectory, baseline.trajectory)
