"""particle filterレビューで確認した状態・診断の回帰を検証する。"""

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from rikka.analyze.particle.paths import _unsupported_reversal_count
from rikka.analyze.particle.recovery import (
    _generate_recovery_candidates,
    _replay_from_checkpoint,
)
from rikka.analyze.particle.runner import (
    _adaptive_heading_rejuvenation_sigma,
    run_particle_filter,
)
from rikka.analyze.particle_filter import ParticleFilterStepDiagnostics
from rikka.analyze.pdr.models import StepHeading


def _forward_heading(step_index: int = 1) -> StepHeading:
    """runnerのprepared step入力に使う前進方位を返す。"""
    return StepHeading(
        step_index=step_index,
        timestamp_s=float(step_index - 1),
        gyro_heading=0.0,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=0.0,
        source="trajectory_motion",
        confidence=1.0,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=0.0,
        motion_heading=0.0,
        movement_type="forward",
        forward_displacement=1.0,
        lateral_displacement=0.0,
        motion_confidence=1.0,
        motion_reject_reason=None,
        trajectory_movement_type="forward",
    )


def test_heading_rejuvenation_uses_motion_reliability() -> None:
    assert _adaptive_heading_rejuvenation_sigma(0.009, 0.80) == 0.009
    assert _adaptive_heading_rejuvenation_sigma(0.009, 0.91) == pytest.approx(0.0065)
    assert _adaptive_heading_rejuvenation_sigma(0.009, 0.92) == pytest.approx(0.004)
    assert _adaptive_heading_rejuvenation_sigma(0.009, 1.0) == pytest.approx(0.004)


def test_recovery_keeps_map_detour_in_transient_heading_state() -> None:
    n_particles = 12
    map_gray = np.zeros((31, 31), dtype=float)
    map_gray[:, 15] = 255.0

    result = _generate_recovery_candidates(
        previous_particles=np.zeros((n_particles, 2)),
        previous_heading_correction=np.zeros(n_particles),
        previous_heading_drift=np.zeros(n_particles),
        previous_stride_scale=np.ones(n_particles),
        proposed_motion_state=np.zeros(n_particles, dtype=np.int8),
        previous_weights=np.full(n_particles, 1.0 / n_particles),
        angle_det=0.0,
        step_length=3.0,
        sigma_step_length_ratio=0.0,
        n_particles=n_particles,
        map_gray=map_gray,
        gx_mean=0.0,
        gz_mean=9.8,
        origin_px=(15, 15),
        scale=1.0,
        heading_sigma=np.deg2rad(5.0),
        max_attempts=5,
        rng=np.random.default_rng(42),
        allow_turn_candidates=True,
        preserve_route_branches=False,
    )

    assert result is not None
    assert result.attempts == 2
    np.testing.assert_allclose(result.heading_correction, 0.0)
    assert np.all(np.abs(result.heading_drift) >= np.deg2rad(60.0))
    assert np.all(result.path_log_score_delta <= 0.0)
    np.testing.assert_allclose(
        np.mean(result.path_log_score_delta),
        -0.5 * result.mean_cost,
    )


def test_checkpoint_replay_motion_state_follows_selected_checkpoint_parent() -> None:
    checkpoint_states = np.array([0, 1, 2, 3], dtype=np.int8)
    result = _replay_from_checkpoint(
        checkpoint_particles=np.zeros((4, 2)),
        checkpoint_heading_correction=np.zeros(4),
        checkpoint_heading_drift=np.zeros(4),
        checkpoint_stride_scale=np.ones(4),
        checkpoint_weights=np.full(4, 0.25),
        angles=np.array([0.0, 0.0]),
        step_lengths=np.array([1.0, 1.0]),
        n_particles=4,
        map_gray=np.full((21, 21), 255.0),
        gx_mean=0.0,
        gz_mean=9.8,
        origin_px=(10, 10),
        scale=1.0,
        heading_sigma=np.deg2rad(5.0),
        rng=np.random.default_rng(7),
        checkpoint_motion_state=checkpoint_states,
    )

    assert result is not None
    np.testing.assert_array_equal(
        result.recovery.motion_state,
        checkpoint_states[result.recovery.parent_indices],
    )
    assert result.recovery.path_log_score_delta.shape == (4,)


def test_unsupported_reversal_ignores_stationary_displacement() -> None:
    path = np.array([[0.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])

    count = _unsupported_reversal_count(
        path,
        sensor_headings=np.full(3, np.pi),
        turning_evidence=np.zeros(3, dtype=bool),
    )

    assert count == 0


def test_effective_step_length_diagnostic_matches_particle_displacement(
    tmp_path,
) -> None:
    floormap_path = tmp_path / "open_map.png"
    plt.imsave(
        floormap_path,
        np.ones((41, 41)),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    diagnostics: list[ParticleFilterStepDiagnostics] = []

    result = run_particle_filter(
        np.array([0]),
        pd.DataFrame({"low_angle": [0.0]}),
        pd.DataFrame({"h_y": [0.0], "h_z": [0.0]}),
        gx_mean=0.0,
        gz_mean=9.8,
        floormap_path=floormap_path,
        origin_px=(20, 20),
        scale=1.0,
        n_particles=100,
        sigma_sl_ratio=0.2,
        resample_ess_ratio=0.01,
        prepared_step_headings=[_forward_heading()],
        prepared_step_lengths=[1.0],
        prepared_step_times=[0.0],
        seed=42,
        diagnostics_collector=diagnostics,
    )

    displacements = np.linalg.norm(result[3][1] - result[3][0], axis=1)
    np.testing.assert_allclose(
        diagnostics[0].effective_step_length_mean_m,
        np.mean(displacements),
    )
    np.testing.assert_allclose(
        diagnostics[0].effective_step_length_std_m,
        np.std(displacements),
    )


def test_nonempty_prepared_steps_reject_empty_motion_posteriors(tmp_path) -> None:
    floormap_path = tmp_path / "open_map.png"
    plt.imsave(floormap_path, np.ones((9, 9)), cmap="gray", vmin=0.0, vmax=1.0)

    with np.testing.assert_raises_regex(ValueError, "prepared_motion_posteriors"):
        run_particle_filter(
            np.array([0]),
            pd.DataFrame({"low_angle": [0.0]}),
            pd.DataFrame({"h_y": [0.0], "h_z": [0.0]}),
            gx_mean=0.0,
            gz_mean=9.8,
            floormap_path=floormap_path,
            origin_px=(4, 4),
            scale=1.0,
            n_particles=4,
            prepared_step_headings=[_forward_heading()],
            prepared_step_lengths=[1.0],
            prepared_step_times=[0.0],
            prepared_motion_posteriors=(),
            seed=1,
        )
