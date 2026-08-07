"""particle filterレビューで確認した状態・診断の回帰を検証する。"""

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest

from rikka.common.lib.models import StepHeading
from rikka.particle.lib.initialize import _adaptive_heading_rejuvenation_sigma
from rikka.particle.lib.proposal import _motion_state_transition_matrix
from rikka.particle.lib.recorder import (
    ParticleFilterStepDiagnostics,
    ParticlePathComparison,
    ParticleStepStages,
)
from rikka.particle.lib.recovery.checkpoint import _replay_from_checkpoint
from rikka.particle.lib.recovery.local import _generate_recovery_candidates
from rikka.particle.lib.runner import run_particle_filter as _run_particle_filter
from rikka.particle.lib.sequence_path import _unsupported_reversal_count
from rikka.pdr.lib.motion_state.evidence import (
    build_particle_motion_headings,
    build_step_motion_evidences,
)
from rikka.plot.lib.frames import save_particle_step_frames


def run_particle_filter(*args, **kwargs):
    """確定済みPDR歩列をPF内部回帰テストへ渡す。"""
    for unused_name in (
        "peaks",
        "df_gyro",
        "df_acc",
        "initial_direction",
        "weinberg_k",
        "heading_method",
        "step_segments",
    ):
        kwargs.pop(unused_name, None)
    headings = kwargs.get("prepared_step_headings")
    if headings is not None:
        kwargs.setdefault(
            "prepared_motion_evidences",
            build_step_motion_evidences(headings),
        )
        kwargs.setdefault(
            "prepared_particle_motion_headings",
            build_particle_motion_headings(headings),
        )
    return _run_particle_filter(*args[3:], **kwargs)


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


def test_motion_state_transition_rows_are_probability_distributions() -> None:
    transition = _motion_state_transition_matrix()

    np.testing.assert_allclose(transition.sum(axis=1), np.ones(4))
    assert np.all(transition >= 0.0)


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
    mpimg.imsave(
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
    mpimg.imsave(floormap_path, np.ones((9, 9)), cmap="gray", vmin=0.0, vmax=1.0)

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


def test_stage_collection_preserves_fixed_seed_result_and_shapes(tmp_path) -> None:
    floormap_path = tmp_path / "open_map.png"
    mpimg.imsave(
        floormap_path,
        np.ones((41, 41)),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    headings = [_forward_heading(index) for index in range(1, 4)]
    common_arguments = {
        "peaks": np.arange(3),
        "df_gyro": pd.DataFrame({"low_angle": [0.0, 0.0, 0.0]}),
        "df_acc": pd.DataFrame({"h_y": [0.0, 0.0, 0.0], "h_z": [0.0, 0.0, 0.0]}),
        "gx_mean": 0.0,
        "gz_mean": 9.8,
        "floormap_path": floormap_path,
        "origin_px": (20, 20),
        "scale": 1.0,
        "n_particles": 16,
        "prepared_step_headings": headings,
        "prepared_step_lengths": [1.0, 1.0, 1.0],
        "prepared_step_times": [0.0, 1.0, 2.0],
        "seed": 42,
    }
    diagnostics_without: list[ParticleFilterStepDiagnostics] = []
    result_without = run_particle_filter(
        **common_arguments,
        diagnostics_collector=diagnostics_without,
    )
    diagnostics_with: list[ParticleFilterStepDiagnostics] = []
    stages: list[ParticleStepStages] = []
    path_comparisons: list[ParticlePathComparison] = []
    result_with = run_particle_filter(
        **common_arguments,
        diagnostics_collector=diagnostics_with,
        stage_collector=stages,
        path_comparison_collector=path_comparisons,
    )

    np.testing.assert_array_equal(result_with[3], result_without[3])
    np.testing.assert_array_equal(result_with[0], result_without[0])
    assert result_with[1:3] == result_without[1:3]
    assert result_with[4] == result_without[4]
    assert diagnostics_with == diagnostics_without
    assert len(stages) == 3
    assert len(path_comparisons) == 1
    np.testing.assert_array_equal(
        path_comparisons[0].selected_path,
        np.asarray(result_with[0]),
    )
    for stage, diagnostic in zip(stages, diagnostics_with, strict=True):
        assert stage.before_positions.shape == (16, 2)
        assert stage.proposed_positions.shape == (16, 2)
        assert stage.after_positions.shape == (16, 2)
        assert stage.before_offsets.shape == (16,)
        assert stage.proposed_headings.shape == (16,)
        assert stage.proposed_step_lengths.shape == (16,)
        assert stage.valid_transition.shape == (16,)
        assert stage.posterior_weights.shape == (16,)
        assert stage.parent_indices.shape == (16,)
        assert int(np.count_nonzero(stage.valid_transition)) == diagnostic.valid_count


def test_stage_proposed_heading_contains_particle_offset(tmp_path) -> None:
    floormap_path = tmp_path / "open_map.png"
    mpimg.imsave(
        floormap_path,
        np.ones((21, 21)),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    heading = _forward_heading()._replace(
        selected_heading=0.4,
        body_heading=0.4,
        motion_heading=0.4,
    )
    stages: list[ParticleStepStages] = []

    run_particle_filter(
        np.array([0]),
        pd.DataFrame({"low_angle": [0.0]}),
        pd.DataFrame({"h_y": [0.0], "h_z": [0.0]}),
        gx_mean=0.0,
        gz_mean=9.8,
        floormap_path=floormap_path,
        origin_px=(10, 10),
        scale=1.0,
        n_particles=8,
        sigma_init_heading=0.0,
        sigma_heading=0.0,
        prepared_step_headings=[heading],
        prepared_step_lengths=[1.0],
        prepared_step_times=[0.0],
        seed=7,
        stage_collector=stages,
    )

    np.testing.assert_allclose(stages[0].before_offsets, 0.0)
    np.testing.assert_allclose(stages[0].proposed_headings, 0.4)


def test_step_frame_range_saves_only_requested_steps(tmp_path) -> None:
    floormap_path = tmp_path / "open_map.png"
    mpimg.imsave(
        floormap_path,
        np.ones((41, 41)),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    diagnostics: list[ParticleFilterStepDiagnostics] = []
    stages: list[ParticleStepStages] = []
    headings = [_forward_heading(index) for index in range(1, 4)]
    result = run_particle_filter(
        np.arange(3),
        pd.DataFrame({"low_angle": [0.0, 0.0, 0.0]}),
        pd.DataFrame({"h_y": [0.0, 0.0, 0.0], "h_z": [0.0, 0.0, 0.0]}),
        gx_mean=0.0,
        gz_mean=9.8,
        floormap_path=floormap_path,
        origin_px=(20, 20),
        scale=1.0,
        n_particles=12,
        prepared_step_headings=headings,
        prepared_step_lengths=[1.0, 1.0, 1.0],
        prepared_step_times=[0.0, 1.0, 2.0],
        seed=2,
        diagnostics_collector=diagnostics,
        stage_collector=stages,
    )

    paths = save_particle_step_frames(
        stages,
        diagnostics,
        result[0],
        gx_mean=0.0,
        gz_mean=9.8,
        floormap_path=floormap_path,
        origin_px=(20, 20),
        scale=1.0,
        output_dir=tmp_path / "result",
        step_range=(2, 2),
        arrows=2,
        dpi=30,
    )

    assert [path.name for path in paths] == ["step_002.png"]
    assert paths[0].is_file()
