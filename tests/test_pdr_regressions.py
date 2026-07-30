from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rikka.analyze import pdr
from rikka.analyze.particle_branches import branch_preserving_resample
from rikka.analyze.particle_filter import (
    ParticleFilterStepDiagnostics,
    _effective_sample_size,
    _evaluate_particle_transitions,
    _generate_recovery_candidates,
    _motion_state_headings,
    _normalize_floormap_gray,
    _reconstruct_particle_paths,
    _reconstruct_resampled_paths,
    _replay_from_checkpoint,
    _sample_motion_states,
    _select_reachable_cluster_path,
    _select_reachable_mean_path,
    _snap_trajectory_to_walkable_pixels,
    _systematic_resample,
    run_particle_filter,
)
from rikka.analyze.pdr.adaptive_estimator import estimate_adaptive_pdr
from rikka.analyze.pdr.body_heading import estimate_dynamic_body_headings
from rikka.analyze.pdr.motion_decoder import decode_step_motion_segments
from rikka.analyze.pdr.motion_refinement import (
    refine_step_headings_with_motion_model,
)
from rikka.analyze.sensor_plot import _project_acceleration_to_step_axes
from rikka.cli import commands as pdr_commands
from rikka.pdr.lib.motion_state.evidence import (
    build_step_motion_evidences,
    build_step_motion_observations,
)


def _forward_step_heading(
    step_index: int = 1,
    heading: float = 0.0,
) -> pdr.StepHeading:
    """PF回帰テスト用の確定済み前進ステップを返す。"""
    return pdr.StepHeading(
        step_index=step_index,
        timestamp_s=float(step_index - 1),
        gyro_heading=heading,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=heading,
        source="trajectory_motion",
        confidence=0.0,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=heading,
        motion_heading=heading,
        movement_type="forward",
        forward_displacement=1.0,
        lateral_displacement=0.0,
        motion_confidence=1.0,
        motion_reject_reason=None,
        trajectory_movement_type="forward",
    )


def test_process_sensor_data_resets_index_and_uses_time_delta_for_gyro() -> None:
    df_acc = pd.DataFrame(
        {
            "t": [0.0, 0.1, 0.3],
            "x": [0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0],
            "z": [9.8, 9.8, 9.8],
        },
        index=[10, 11, 12],
    )
    df_gyro = pd.DataFrame(
        {
            "t": [0.0, 0.1, 0.3],
            "x": [0.0, 1.0, 1.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
        },
        index=[100, 101, 102],
    )

    processed_acc, processed_gyro = pdr.process_sensor_data(
        df_acc,
        df_gyro,
        gyro_bias_method="quietest",
    )

    assert processed_acc.index.tolist() == [0, 1, 2]
    assert processed_gyro.index.tolist() == [0, 1, 2]
    np.testing.assert_allclose(
        processed_gyro["angle"].to_numpy(),
        np.array([0.0, 1.0 / 30.0, 0.1]),
    )


def test_estimate_gyro_bias_prewalk_robust_selects_static_subwindow() -> None:
    n_samples = 800
    df_acc = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "low_lin_norm": np.full(n_samples, 0.7),
        }
    )
    df_acc.loc[300:400, "low_lin_norm"] = 0.05
    df_acc.loc[450:540, "low_lin_norm"] = 0.9
    df_acc.loc[[550, 610, 670, 730], "low_lin_norm"] = 3.0
    gyro_x = np.full(n_samples, -0.08)
    gyro_x[300:401] = 0.02
    gyro_x[450:] = 0.4
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "x": gyro_x,
        }
    )

    result = pdr.estimate_gyro_bias(
        df_acc,
        df_gyro,
        method="prewalk_robust",
    )

    assert result.method == "prewalk_robust"
    assert result.calibration_start_s is not None
    assert result.calibration_end_s is not None
    assert 3.0 <= result.calibration_start_s <= 3.1
    assert 3.9 <= result.calibration_end_s <= 4.1
    assert result.search_start_s == 0.5
    assert result.search_end_s == 4.5
    assert result.accel_p95 is not None
    assert result.accel_p95 <= 0.05
    np.testing.assert_allclose(result.bias_rad_s, 0.02, atol=1e-12)


def test_estimate_gyro_bias_static_window_rejects_high_acceleration() -> None:
    n_samples = 800
    df_acc = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "low_lin_norm": np.full(n_samples, 1.4),
        }
    )
    df_acc.loc[200:300, "low_lin_norm"] = 0.08
    df_acc.loc[[550, 610, 670, 730], "low_lin_norm"] = 3.0
    gyro_x = np.full(n_samples, 0.01)
    gyro_x[200:301] = 0.04
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "x": gyro_x,
        }
    )

    result = pdr.estimate_gyro_bias(
        df_acc,
        df_gyro,
        method="prewalk_robust",
    )

    assert result.method == "prewalk_robust"
    assert result.calibration_start_s is not None
    assert 2.0 <= result.calibration_start_s <= 2.1
    np.testing.assert_allclose(result.bias_rad_s, 0.04, atol=1e-12)


def test_estimate_gyro_bias_manual_requires_and_uses_value() -> None:
    df_acc = pd.DataFrame({"low_lin_norm": [0.0]})
    df_gyro = pd.DataFrame({"x": [1.0]})

    result = pdr.estimate_gyro_bias(
        df_acc,
        df_gyro,
        method="manual",
        manual_bias=0.123,
    )

    assert result.method == "manual"
    assert result.bias_rad_s == 0.123
    try:
        pdr.estimate_gyro_bias(df_acc, df_gyro, method="manual")
    except ValueError as exc:
        assert "gyro_bias" in str(exc)
    else:
        raise AssertionError("manual bias without value should fail")


def test_estimate_gyro_bias_zero_does_not_use_sensor_rotation() -> None:
    df_acc = pd.DataFrame({"low_lin_norm": [0.0, 0.0]})
    df_gyro = pd.DataFrame({"x": [0.25, -0.5]})

    result = pdr.estimate_gyro_bias(df_acc, df_gyro, method="zero")

    assert result.method == "zero"
    assert result.bias_rad_s == 0.0
    assert result.sample_count == 0
    assert result.calibration_start_s is None


def test_estimate_gyro_bias_prewalk_guarded_rejects_large_rotation() -> None:
    n_samples = 600
    times = np.arange(n_samples, dtype=float) * 0.01
    low_lin_norm = np.zeros(n_samples)
    low_lin_norm[400::60] = 2.0
    df_acc = pd.DataFrame({"t": times, "low_lin_norm": low_lin_norm})
    df_gyro = pd.DataFrame({"t": times, "x": np.full(n_samples, 0.02)})

    result = pdr.estimate_gyro_bias(
        df_acc,
        df_gyro,
        method="prewalk_guarded",
    )

    assert result.method == "prewalk_guarded"
    assert result.bias_rad_s == 0.0
    assert result.robust_mean is not None
    np.testing.assert_allclose(result.robust_mean, 0.02)
    assert result.fallback_reason is not None
    assert "estimated_bias_exceeds_guard" in result.fallback_reason


def test_estimate_gyro_bias_prewalk_falls_back_to_initial_robust() -> None:
    n_samples = 120
    df_acc = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "low_lin_norm": np.full(n_samples, 2.0),
        }
    )
    gyro_x = np.full(n_samples, -0.03)
    gyro_x[10] = 1.0
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "x": gyro_x,
        }
    )

    result = pdr.estimate_gyro_bias(
        df_acc,
        df_gyro,
        method="prewalk_robust",
    )

    assert result.method == "initial_robust"
    assert result.fallback_reason == "startup_static_unavailable"
    np.testing.assert_allclose(result.bias_rad_s, -0.03, atol=1e-12)


def test_estimate_gyro_bias_quietest_keeps_legacy_window_behavior() -> None:
    n_samples = 100
    df_acc = pd.DataFrame({"low_lin_norm": np.zeros(n_samples)})
    gyro_x = np.concatenate(
        [
            np.linspace(-1.0, 1.0, 40),
            np.full(40, 0.04),
            np.linspace(1.0, -1.0, 20),
        ]
    )
    df_gyro = pd.DataFrame({"x": gyro_x})

    result = pdr.estimate_gyro_bias(
        df_acc,
        df_gyro,
        method="quietest",
    )

    assert result.method == "quietest"
    np.testing.assert_allclose(result.bias_rad_s, 0.04, atol=1e-12)


def test_sample_gyro_angle_interpolates_by_time() -> None:
    df_gyro = pd.DataFrame({"t": [10.0, 20.0], "low_angle": [0.0, 10.0]})

    assert pdr._sample_gyro_angle(df_gyro, sample_index=0, sample_time=15.0) == 5.0


def test_sample_gyro_angle_clips_index_when_time_is_unavailable() -> None:
    df_gyro = pd.DataFrame({"low_angle": [1.0, 2.0, 3.0]})

    assert pdr._sample_gyro_angle(df_gyro, sample_index=-1) == 1.0
    assert pdr._sample_gyro_angle(df_gyro, sample_index=100) == 3.0


def test_sample_gyro_angle_returns_none_for_empty_or_nan_fallback() -> None:
    empty = pd.DataFrame({"low_angle": []})
    with_nan = pd.DataFrame({"low_angle": [np.nan]})

    assert pdr._sample_gyro_angle(empty, sample_index=0) is None
    assert pdr._sample_gyro_angle(with_nan, sample_index=0) is None


def test_create_output_dir_avoids_timestamp_collisions(tmp_path) -> None:
    now = datetime(2026, 5, 15, 12, 0, 0, 123456)

    first = pdr._create_output_dir(tmp_path, now)
    second = pdr._create_output_dir(tmp_path, now)

    assert first != second
    assert first.exists()
    assert second.exists()


def test_build_trajectory_dataframe_adds_elapsed_timestamps() -> None:
    df_trajectory = pdr._build_trajectory_dataframe(
        [[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]],
        [10.5, 11.25],
    )

    assert df_trajectory.columns.tolist() == ["timestamp_s", "x", "y"]
    np.testing.assert_allclose(
        df_trajectory.to_numpy(),
        np.array(
            [
                [0.0, 1.0, 2.0],
                [0.75, 3.0, 4.0],
            ]
        ),
    )


def test_numeric_validators_reject_non_finite_values() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        try:
            pdr._validate_scale(value)
        except ValueError as exc:
            assert "有限" in str(exc)
        else:
            raise AssertionError("_validate_scale should reject non-finite values")

        try:
            pdr._validate_positive_parameter("value", value)
        except ValueError as exc:
            assert "有限" in str(exc)
        else:
            raise AssertionError(
                "_validate_positive_parameter should reject non-finite values"
            )

        try:
            pdr._validate_non_negative_parameter("value", value)
        except ValueError as exc:
            assert "有限" in str(exc)
        else:
            raise AssertionError(
                "_validate_non_negative_parameter should reject non-finite values"
            )


def test_run_does_not_create_output_dir_when_sensor_pair_is_incomplete(
    monkeypatch,
) -> None:
    calls = 0

    def fake_create_output_dir() -> Path:
        nonlocal calls
        calls += 1
        raise AssertionError("_create_output_dir should not be called")

    monkeypatch.setattr(pdr_commands, "_create_output_dir", fake_create_output_dir)

    try:
        pdr.run(df_acc=pd.DataFrame(), df_gyro=None, plot=False)
    except ValueError as exc:
        assert "df_acc と df_gyro" in str(exc)
    else:
        raise AssertionError("incomplete sensor pair should fail")

    assert calls == 0


def test_run_returns_and_saves_timestamped_trajectory_without_steps(
    tmp_path, monkeypatch
) -> None:
    df_acc = pd.DataFrame(
        {
            "t": np.arange(5, dtype=float) * 0.01,
            "x": np.zeros(5),
            "y": np.zeros(5),
            "z": np.full(5, 9.8),
        }
    )
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(5, dtype=float) * 0.01,
            "x": np.zeros(5),
            "y": np.zeros(5),
            "z": np.zeros(5),
        }
    )

    def fake_create_output_dir() -> Path:
        output_dir = tmp_path / "pdr"
        output_dir.mkdir()
        return output_dir

    monkeypatch.setattr(pdr_commands, "_create_output_dir", fake_create_output_dir)
    monkeypatch.setattr(pdr, "detect_steps", lambda _df_acc: np.array([], dtype=int))

    df_trajectory = pdr.run(df_acc=df_acc, df_gyro=df_gyro, plot=False)

    assert df_trajectory.columns.tolist() == ["timestamp_s", "x", "y"]
    assert df_trajectory.empty
    saved = pd.read_csv(tmp_path / "pdr" / "trajectory.csv")
    pd.testing.assert_frame_equal(saved, df_trajectory)


def test_run_particle_filter_seed_makes_particles_deterministic(tmp_path) -> None:
    floormap_path = tmp_path / "map.png"
    plt.imsave(
        floormap_path,
        np.ones((20, 20), dtype=float),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    step_heading = _forward_step_heading()

    first = run_particle_filter(
        np.array([0]),
        pd.DataFrame({"low_angle": [0.0]}),
        pd.DataFrame({"h_y": [0.0], "h_z": [0.0]}),
        gx_mean=0.0,
        gz_mean=9.8,
        floormap_path=floormap_path,
        origin_px=(5, 5),
        scale=1.0,
        n_particles=12,
        prepared_step_headings=[step_heading],
        prepared_step_lengths=[1.0],
        prepared_step_times=[0.0],
        seed=123,
    )
    second = run_particle_filter(
        np.array([0]),
        pd.DataFrame({"low_angle": [0.0]}),
        pd.DataFrame({"h_y": [0.0], "h_z": [0.0]}),
        gx_mean=0.0,
        gz_mean=9.8,
        floormap_path=floormap_path,
        origin_px=(5, 5),
        scale=1.0,
        n_particles=12,
        prepared_step_headings=[step_heading],
        prepared_step_lengths=[1.0],
        prepared_step_times=[0.0],
        seed=123,
    )

    np.testing.assert_allclose(np.asarray(first[0]), np.asarray(second[0]))
    np.testing.assert_allclose(first[3], second[3])


def test_particle_filter_compatibility_facade_exports_existing_symbols() -> None:
    from rikka.analyze import particle_filter

    expected_symbols = (
        "ParticleFilterStepDiagnostics",
        "_effective_sample_size",
        "_evaluate_particle_transitions",
        "_generate_recovery_candidates",
        "_motion_state_headings",
        "_normalize_floormap_gray",
        "_reconstruct_particle_paths",
        "_reconstruct_resampled_paths",
        "_replay_from_checkpoint",
        "_sample_motion_states",
        "_select_reachable_mean_path",
        "_snap_trajectory_to_walkable_pixels",
        "_systematic_resample",
        "plot_particle_filter_trajectory",
        "run_particle_filter",
        "save_particle_animation",
    )

    assert all(hasattr(particle_filter, name) for name in expected_symbols)


def test_particle_filter_diagnostics_field_order_is_stable() -> None:
    assert tuple(ParticleFilterStepDiagnostics.__dataclass_fields__) == (
        "step",
        "timestamp_s",
        "valid_count",
        "valid_ratio",
        "valid_weight_count",
        "valid_weight_mass_before_normalization",
        "ess_before_observation",
        "ess_after_observation",
        "ess_after_resampling",
        "max_weight",
        "unique_parent_count",
        "unique_position_count",
        "position_spread_rms_m",
        "heading_drift_std_deg",
        "heading_total_std_deg",
        "stride_scale_mean",
        "stride_scale_std",
        "effective_step_length_mean_m",
        "effective_step_length_std_m",
        "forward_state_probability",
        "sidestep_left_state_probability",
        "sidestep_right_state_probability",
        "turning_state_probability",
        "representative_motion_state",
        "motion_state_entropy",
        "motion_state_transition_count",
        "motion_reliability",
        "calibration_reliability",
        "resampled",
        "recovery_attempted",
        "recovery_mode",
        "recovery_valid_count",
        "recovery_attempts",
        "recovery_heading_delta_deg",
        "recovery_step_scale",
        "recovery_cost",
        "recovery_checkpoint_step",
        "recovery_replay_steps",
        "trajectory_mode",
        "trajectory_source_index",
        "recovery_candidate_branch_count",
        "recovery_selected_branch_count",
    )


def test_particle_filter_fixed_seed_characterization_snapshot(tmp_path) -> None:
    floormap_path = tmp_path / "open_map.png"
    plt.imsave(
        floormap_path,
        np.ones((20, 20), dtype=float),
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
        origin_px=(5, 5),
        scale=1.0,
        n_particles=12,
        prepared_step_headings=[_forward_step_heading()],
        prepared_step_lengths=[1.0],
        prepared_step_times=[0.0],
        seed=123,
        diagnostics_collector=diagnostics,
    )

    assert len(result) == 5
    np.testing.assert_allclose(
        np.asarray(result[0]),
        np.array([[0.0, 0.0], [1.0175842, 0.00248184]]),
        rtol=1e-7,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result[3][1, :3],
        np.array(
            [
                [1.02581267, -0.0136450707],
                [0.986660999, -0.0158762091],
                [1.02215898, 0.0438210022],
            ]
        ),
        rtol=1e-7,
        atol=1e-8,
    )
    assert result[1] == [1.0]
    assert result[2] == [0.0]
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    np.testing.assert_allclose(
        [
            diagnostic.ess_after_observation,
            diagnostic.position_spread_rms_m,
            diagnostic.heading_drift_std_deg,
            diagnostic.stride_scale_mean,
        ],
        [
            11.999999999999998,
            0.04693755747666554,
            1.3329123268479632,
            1.0172439681875478,
        ],
    )
    assert diagnostic.resampled is False
    assert diagnostic.recovery_mode == "none"
    assert diagnostic.trajectory_mode == "weighted_mean"


def test_effective_sample_size_handles_uniform_and_concentrated_weights() -> None:
    assert _effective_sample_size(np.full(4, 0.25)) == 4.0
    assert _effective_sample_size(np.array([1.0, 0.0, 0.0, 0.0])) == 1.0
    assert _effective_sample_size(np.zeros(4)) == 0.0


def test_systematic_resampling_is_reproducible() -> None:
    indices = _systematic_resample(
        np.array([0.1, 0.2, 0.7]),
        np.random.default_rng(7),
    )

    assert indices.tolist() == [1, 2, 2]


def test_particle_motion_evidence_is_normalized_without_confidence_saturation() -> None:
    evidence = build_step_motion_evidences([_forward_step_heading()])[0]

    np.testing.assert_allclose(
        evidence.forward_likelihood
        + evidence.sidestep_left_likelihood
        + evidence.sidestep_right_likelihood
        + evidence.turning_likelihood,
        1.0,
    )
    assert 0.0 < evidence.motion_reliability < 1.0
    assert 0.0 < evidence.calibration_reliability < 1.0


def test_motion_observation_separates_device_body_and_motion_axis() -> None:
    heading = _forward_step_heading()._replace(
        gyro_heading=np.deg2rad(10.0),
        body_heading=np.deg2rad(25.0),
        motion_heading=np.deg2rad(170.0),
        forward_displacement=0.03,
        lateral_displacement=-0.04,
    )

    observation = build_step_motion_observations([heading])[0]

    np.testing.assert_allclose(observation.device_yaw_heading, np.deg2rad(10.0))
    np.testing.assert_allclose(observation.body_heading_candidate, np.deg2rad(25.0))
    np.testing.assert_allclose(
        observation.directed_motion_heading,
        np.deg2rad(170.0),
    )
    np.testing.assert_allclose(observation.motion_axis_heading, np.deg2rad(-10.0))
    np.testing.assert_allclose(observation.displacement_norm, 0.05)


def test_motion_axis_observation_is_unchanged_by_opposite_direction() -> None:
    forward = _forward_step_heading(heading=np.deg2rad(35.0))
    backward = forward._replace(motion_heading=forward.motion_heading + np.pi)

    forward_observation = build_step_motion_observations([forward])[0]
    backward_observation = build_step_motion_observations([backward])[0]

    np.testing.assert_allclose(
        forward_observation.motion_axis_heading,
        backward_observation.motion_axis_heading,
    )


def test_motion_segment_decoder_confirms_consistent_two_step_side_segment() -> None:
    headings = [
        _forward_step_heading(step_index=1),
        _forward_step_heading(step_index=2)._replace(
            body_heading=0.0,
            motion_heading=np.pi / 2,
            forward_displacement=0.01,
            lateral_displacement=0.10,
        ),
        _forward_step_heading(step_index=3)._replace(
            body_heading=0.0,
            motion_heading=np.pi / 2,
            forward_displacement=0.02,
            lateral_displacement=0.11,
        ),
        _forward_step_heading(step_index=4),
    ]

    result = decode_step_motion_segments(build_step_motion_observations(headings))

    assert result.motion_modes == (
        "forward",
        "sidestep_left",
        "sidestep_left",
        "forward",
    )


def test_motion_segment_decoder_rejects_opposite_side_signs() -> None:
    headings = [
        _forward_step_heading(step_index=1)._replace(
            body_heading=0.0,
            motion_heading=np.pi / 2,
            forward_displacement=0.01,
            lateral_displacement=0.10,
        ),
        _forward_step_heading(step_index=2)._replace(
            body_heading=0.0,
            motion_heading=-np.pi / 2,
            forward_displacement=0.01,
            lateral_displacement=-0.10,
        ),
    ]

    result = decode_step_motion_segments(build_step_motion_observations(headings))

    assert result.motion_modes == ("forward", "forward")


def test_dynamic_body_heading_absorbs_persistent_device_only_rotation() -> None:
    headings = [
        _forward_step_heading(step_index=index)._replace(
            gyro_heading=np.deg2rad(min(30.0, (index - 1) * 5.0)),
            body_heading=np.deg2rad(min(30.0, (index - 1) * 5.0)),
            motion_heading=0.0,
            forward_displacement=0.10,
            lateral_displacement=0.0,
        )
        for index in range(1, 10)
    ]
    observations = build_step_motion_observations(headings)

    estimates = estimate_dynamic_body_headings(
        observations,
        ["forward"] * len(observations),
    )

    assert estimates[-1].updated is True
    assert estimates[-1].body_heading is not None
    assert abs(estimates[-1].body_heading) < np.deg2rad(20.0)
    assert estimates[-1].device_body_offset < 0.0


def test_motion_refinement_suppresses_low_calibration_false_side_run() -> None:
    headings = [
        _forward_step_heading(step_index=index)._replace(
            movement_type="sidestep_left",
            trajectory_movement_type=None,
            body_heading=0.0,
            motion_heading=np.pi / 2,
            forward_displacement=0.01,
            lateral_displacement=0.10,
        )
        for index in range(1, 5)
    ]

    refined = refine_step_headings_with_motion_model(
        headings,
        "clustered",
        "forward",
    )

    assert all(heading.trajectory_movement_type == "forward" for heading in refined)
    assert all(heading.decoded_motion_mode == "forward" for heading in refined)


def test_branch_resampling_protects_small_positive_branch() -> None:
    weights = np.array([0.98, 0.01, 0.01])
    branch_ids = np.array([0, 1, 1])

    parents, resampled_branches, diagnostics = branch_preserving_resample(
        weights,
        branch_ids,
        np.random.default_rng(7),
        output_count=100,
    )

    assert len(parents) == 100
    assert np.count_nonzero(resampled_branches == 1) >= 10
    assert diagnostics.active_branch_count == 2
    assert diagnostics.min_protected_branch_count == 10


def test_recovery_keeps_local_and_turn_route_branches_on_open_map() -> None:
    n_particles = 20
    result = _generate_recovery_candidates(
        previous_particles=np.zeros((n_particles, 2), dtype=float),
        previous_heading_correction=np.zeros(n_particles),
        previous_heading_drift=np.zeros(n_particles),
        previous_stride_scale=np.ones(n_particles),
        proposed_motion_state=np.zeros(n_particles, dtype=np.int8),
        previous_weights=np.full(n_particles, 1.0 / n_particles),
        angle_det=0.0,
        step_length=1.0,
        sigma_step_length_ratio=0.0,
        n_particles=n_particles,
        map_gray=np.full((30, 30), 255.0),
        gx_mean=0.0,
        gz_mean=9.8,
        origin_px=(15, 15),
        scale=1.0,
        heading_sigma=0.08,
        max_attempts=5,
        rng=np.random.default_rng(9),
        allow_turn_candidates=True,
    )

    assert result is not None
    assert set(result.route_branch_ids.tolist()) == {0, 1, 2, 3}
    assert len(result.particles) == n_particles


def test_particle_motion_evidence_preserves_raw_turning_after_smoothing() -> None:
    heading = _forward_step_heading()._replace(
        movement_type="turning_sidestep_right",
        trajectory_movement_type="forward",
        yaw_delta=np.deg2rad(55.0),
    )

    evidence = build_step_motion_evidences([heading])[0]

    assert evidence.turning_likelihood > evidence.forward_likelihood


def test_particle_motion_evidence_prioritizes_confirmed_sidestep_cluster() -> None:
    headings = [_forward_step_heading(step_index=index) for index in range(1, 5)]
    headings.append(
        _forward_step_heading(step_index=5)._replace(
            movement_type="sidestep_right",
            trajectory_movement_type="sidestep_right",
            motion_heading=-np.pi / 2,
            forward_displacement=0.02,
            lateral_displacement=-0.10,
            sidestep_cluster_id=1,
        )
    )

    evidence = build_step_motion_evidences(headings)[-1]

    assert evidence.sidestep_right_likelihood > 0.9
    assert evidence.forward_likelihood < 0.02


def test_particle_motion_state_uses_distinct_forward_and_sidestep_headings() -> None:
    heading = _forward_step_heading()._replace(
        movement_type="sidestep_right",
        trajectory_movement_type="sidestep_right",
        body_heading=0.0,
        motion_heading=np.deg2rad(-70.0),
        sidestep_cluster_id=1,
    )
    evidence = build_step_motion_evidences(
        [_forward_step_heading(step_index=index) for index in range(2, 6)] + [heading]
    )[-1]._replace(calibration_reliability=0.7)

    state_headings = _motion_state_headings(heading, evidence, 0.0)

    np.testing.assert_allclose(state_headings[0], 0.0)
    np.testing.assert_allclose(state_headings[2], np.deg2rad(-70.0))


def test_particle_motion_state_proposal_preserves_sidestep_clusters() -> None:
    previous_states = np.full(10_000, 1, dtype=np.int8)
    states, predictive = _sample_motion_states(
        previous_states,
        np.full(4, 0.25),
        np.random.default_rng(42),
    )

    assert np.mean(states == 1) > 0.79
    np.testing.assert_allclose(predictive, 0.25)


def test_particle_transition_supercover_rejects_thin_walls_and_map_exit() -> None:
    map_gray = np.full((7, 7), 255.0)
    map_gray[3, 3] = 0.0
    previous = np.array([[1.0, 3.0], [1.0, 1.0], [1.0, 5.0], [1.0, 1.0]])
    proposed = np.array([[5.0, 3.0], [5.0, 5.0], [5.0, 5.0], [8.0, 1.0]])

    valid = _evaluate_particle_transitions(
        previous,
        proposed,
        map_gray,
        gx_mean=0.0,
        gz_mean=1.0,
        origin_px=(0, 0),
        scale=1.0,
    )

    assert valid.tolist() == [False, False, True, False]


def test_particle_transition_supercover_treats_corner_touch_as_collision() -> None:
    map_gray = np.full((4, 4), 255.0)
    map_gray[1, 2] = 0.0

    valid = _evaluate_particle_transitions(
        np.array([[1.0, 1.0]]),
        np.array([[2.0, 2.0]]),
        map_gray,
        gx_mean=0.0,
        gz_mean=1.0,
        origin_px=(0, 0),
        scale=1.0,
    )

    assert valid.tolist() == [False]


def test_particle_filter_keeps_weights_without_unneeded_resampling(tmp_path) -> None:
    floormap_path = tmp_path / "map.png"
    plt.imsave(
        floormap_path,
        np.ones((20, 20), dtype=float),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    diagnostics: list[ParticleFilterStepDiagnostics] = []

    run_particle_filter(
        np.array([0]),
        pd.DataFrame({"low_angle": [0.0]}),
        pd.DataFrame({"h_y": [0.0], "h_z": [0.0]}),
        gx_mean=0.0,
        gz_mean=9.8,
        floormap_path=floormap_path,
        origin_px=(5, 5),
        scale=1.0,
        n_particles=12,
        prepared_step_headings=[_forward_step_heading()],
        prepared_step_lengths=[1.0],
        prepared_step_times=[0.0],
        seed=123,
        diagnostics_collector=diagnostics,
    )

    assert len(diagnostics) == 1
    np.testing.assert_allclose(diagnostics[0].ess_after_observation, 12.0)
    assert diagnostics[0].resampled is False
    assert diagnostics[0].unique_parent_count == 12


def test_particle_filter_bounds_heading_drift_in_open_area(tmp_path) -> None:
    floormap_path = tmp_path / "open_map.png"
    plt.imsave(
        floormap_path,
        np.ones((150, 150), dtype=float),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    n_steps = 60
    diagnostics: list[ParticleFilterStepDiagnostics] = []

    result = run_particle_filter(
        np.arange(n_steps),
        pd.DataFrame({"low_angle": np.zeros(n_steps)}),
        pd.DataFrame({"h_y": np.zeros(n_steps), "h_z": np.zeros(n_steps)}),
        gx_mean=0.0,
        gz_mean=9.8,
        floormap_path=floormap_path,
        origin_px=(20, 70),
        scale=1.0,
        n_particles=500,
        prepared_step_headings=[
            _forward_step_heading(step_index=i + 1) for i in range(n_steps)
        ],
        prepared_step_lengths=[1.0] * n_steps,
        prepared_step_times=[float(i) for i in range(n_steps)],
        seed=42,
        diagnostics_collector=diagnostics,
    )

    assert diagnostics[-1].heading_drift_std_deg < 7.5
    assert all(item.trajectory_mode == "weighted_mean" for item in diagnostics)
    trajectory = np.asarray(result[0], dtype=float)
    headings_deg = np.degrees(
        np.arctan2(np.diff(trajectory[:, 1]), np.diff(trajectory[:, 0]))
    )
    assert float(np.mean(np.abs(headings_deg))) < 10.0


def test_particle_filter_learns_and_keeps_persistent_stride_scale(tmp_path) -> None:
    map_gray = np.ones((50, 50), dtype=float)
    map_gray[:, 32] = 0.0
    floormap_path = tmp_path / "stride_constraint_map.png"
    plt.imsave(floormap_path, map_gray, cmap="gray", vmin=0.0, vmax=1.0)
    diagnostics: list[ParticleFilterStepDiagnostics] = []

    run_particle_filter(
        np.arange(3),
        pd.DataFrame({"low_angle": np.zeros(3)}),
        pd.DataFrame({"h_y": np.zeros(3), "h_z": np.zeros(3)}),
        gx_mean=0.0,
        gz_mean=9.8,
        floormap_path=floormap_path,
        origin_px=(10, 25),
        scale=0.1,
        n_particles=500,
        sigma_init_heading=0.0,
        sigma_heading=0.0,
        sigma_sl_ratio=0.0,
        stride_scale_prior_mean=1.0,
        stride_scale_init_sigma=0.15,
        stride_scale_retention=1.0,
        stride_scale_process_sigma=0.0,
        stride_scale_rejuvenation_sigma=0.0,
        resample_ess_ratio=0.99,
        prepared_step_headings=[
            _forward_step_heading(step_index=1, heading=0.0),
            _forward_step_heading(step_index=2, heading=0.0),
            _forward_step_heading(step_index=3, heading=np.pi)._replace(
                yaw_delta=np.pi
            ),
        ],
        prepared_step_lengths=[1.0, 1.0, 1.0],
        prepared_step_times=[0.0, 1.0, 2.0],
        seed=42,
        diagnostics_collector=diagnostics,
    )

    assert diagnostics[1].resampled is True
    assert diagnostics[1].stride_scale_mean < diagnostics[0].stride_scale_mean - 0.03
    np.testing.assert_allclose(
        diagnostics[2].stride_scale_mean,
        diagnostics[1].stride_scale_mean,
        atol=1e-4,
    )
    assert 0.0 < diagnostics[2].effective_step_length_mean_m
    assert (
        diagnostics[2].effective_step_length_mean_m <= diagnostics[2].stride_scale_mean
    )


def test_particle_filter_rejects_invalid_stride_scale_range(tmp_path) -> None:
    floormap_path = tmp_path / "open_map.png"
    plt.imsave(
        floormap_path,
        np.ones((5, 5), dtype=float),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    with np.testing.assert_raises_regex(ValueError, "stride_scale_min"):
        run_particle_filter(
            np.array([], dtype=int),
            pd.DataFrame({"low_angle": []}),
            pd.DataFrame({"h_y": [], "h_z": []}),
            gx_mean=0.0,
            gz_mean=9.8,
            floormap_path=floormap_path,
            origin_px=(2, 2),
            scale=1.0,
            n_particles=4,
            prepared_step_headings=[],
            prepared_step_lengths=[],
            prepared_step_times=[],
            stride_scale_min=1.1,
            stride_scale_max=1.0,
        )


def test_particle_filter_rejects_negative_motion_predictive_weight_power(
    tmp_path,
) -> None:
    floormap_path = tmp_path / "open_map.png"
    plt.imsave(
        floormap_path,
        np.ones((5, 5), dtype=float),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "motion_predictive_weight_power",
    ):
        run_particle_filter(
            np.array([], dtype=int),
            pd.DataFrame({"low_angle": []}),
            pd.DataFrame({"h_y": [], "h_z": []}),
            gx_mean=0.0,
            gz_mean=9.8,
            floormap_path=floormap_path,
            origin_px=(2, 2),
            scale=1.0,
            n_particles=4,
            prepared_step_headings=[],
            prepared_step_lengths=[],
            prepared_step_times=[],
            motion_predictive_weight_power=-0.1,
        )


def test_checkpoint_replay_reconstructs_wall_valid_multi_step_path() -> None:
    map_gray = np.zeros((30, 30), dtype=float)
    for y in range(2, 28):
        for x in range(2, 28):
            if abs(x - y) <= 2:
                map_gray[y, x] = 255.0
    n_particles = 20

    result = _replay_from_checkpoint(
        checkpoint_particles=np.zeros((n_particles, 2), dtype=float),
        checkpoint_heading_correction=np.zeros(n_particles),
        checkpoint_heading_drift=np.zeros(n_particles),
        checkpoint_stride_scale=np.ones(n_particles),
        checkpoint_weights=np.full(n_particles, 1.0 / n_particles),
        angles=np.zeros(2),
        step_lengths=np.full(2, 3.0),
        n_particles=n_particles,
        map_gray=map_gray,
        gx_mean=0.0,
        gz_mean=9.8,
        origin_px=(5, 5),
        scale=1.0,
        heading_sigma=np.deg2rad(5.0),
        rng=np.random.default_rng(42),
    )

    assert result is not None
    assert result.replay_positions.shape == (2, n_particles, 2)
    assert result.recovery.mode == "checkpoint_replay"
    starts = np.concatenate(
        [
            np.zeros((1, n_particles, 2)),
            result.replay_positions[:-1],
        ],
        axis=0,
    )
    assert _evaluate_particle_transitions(
        starts.reshape(-1, 2),
        result.replay_positions.reshape(-1, 2),
        map_gray,
        gx_mean=0.0,
        gz_mean=9.8,
        origin_px=(5, 5),
        scale=1.0,
    ).all()


def test_reachable_mean_path_averages_only_dominant_reachable_cluster() -> None:
    map_gray = np.full((7, 7), 255.0)
    map_gray[3, 3] = 0.0
    upper_path = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 2.0], [5.0, 3.0]])
    lower_path = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0], [4.0, 4.0], [5.0, 3.0]])
    wall_path = np.array([[1.0, 3.0], [2.0, 3.0], [3.0, 3.0], [4.0, 3.0], [5.0, 3.0]])

    selected, modes, sources = _select_reachable_mean_path(
        np.stack([upper_path, lower_path, wall_path]),
        np.array([0.5, 0.5, 0.0]),
        map_gray,
        gx_mean=0.0,
        gz_mean=1.0,
        origin_px=(0, 0),
        scale=1.0,
    )

    assert modes == ["weighted_mean"] * len(upper_path)
    assert sources == [None] * len(upper_path)
    np.testing.assert_allclose(
        selected,
        [[1.0, 3.0], [2.0, 3.0], [3.0, 1.0], [4.0, 3.0], [5.0, 3.0]],
    )
    assert 2 not in sources
    assert _evaluate_particle_transitions(
        selected[:-1],
        selected[1:],
        map_gray,
        gx_mean=0.0,
        gz_mean=1.0,
        origin_px=(0, 0),
        scale=1.0,
    ).all()


def test_reachable_cluster_path_averages_particles_on_same_side_of_wall() -> None:
    map_gray = np.full((7, 7), 255.0)
    map_gray[:, 3] = 0.0
    upper_paths = np.array(
        [
            [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]],
            [[2.0, 1.0], [2.0, 2.0], [2.0, 3.0]],
        ]
    )
    other_side_path = np.array([[[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]]])

    particle_paths = np.concatenate([upper_paths, other_side_path])
    selected, modes, sources = _select_reachable_cluster_path(
        [particle_paths[:, index, :] for index in range(3)],
        [np.array([0.35, 0.35, 0.3]) for _ in range(3)],
        [np.arange(3), np.arange(3)],
        map_gray,
        gx_mean=0.0,
        gz_mean=1.0,
        origin_px=(0, 0),
        scale=1.0,
    )

    np.testing.assert_allclose(selected[:, 0], 1.5)
    np.testing.assert_allclose(selected[:, 1], [1.0, 2.0, 3.0])
    assert modes == ["weighted_mean"] * 3
    assert sources == [None] * 3


def test_reachable_cluster_path_uses_weights_from_each_step() -> None:
    map_gray = np.full((8, 8), 255.0)
    position_history = [
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        np.array([[2.0, 1.0], [2.0, 3.0]]),
        np.array([[3.0, 1.0], [3.0, 3.0]]),
    ]
    weight_history = [
        np.array([0.5, 0.5]),
        np.array([0.8, 0.2]),
        np.array([0.2, 0.8]),
    ]

    selected, modes, sources = _select_reachable_cluster_path(
        position_history,
        weight_history,
        [np.array([0, 1]), np.array([0, 1])],
        map_gray,
        gx_mean=0.0,
        gz_mean=1.0,
        origin_px=(0, 0),
        scale=1.0,
    )

    np.testing.assert_allclose(selected, [[1.0, 1.0], [2.0, 1.4], [3.0, 2.6]])
    assert modes == ["weighted_mean"] * 3
    assert sources == [None] * 3


def test_adaptive_pdr_exposes_normalized_state_and_length_uncertainty() -> None:
    headings = [
        _forward_step_heading(step_index=1),
        _forward_step_heading(step_index=2)._replace(
            movement_type="sidestep_left",
            trajectory_movement_type="sidestep_left",
            motion_heading=np.pi / 2.0,
        ),
    ]
    observations = (
        pdr.StepLengthObservation(1, 1.0, 1.0, 0.65, 2.0, 1.0, 1.0, 0.1, None),
        pdr.StepLengthObservation(2, 1.0, 0.7, 0.65, 1.0, 1.0, 0.9, 0.12, None),
    )
    evidences = (
        pdr.StepMotionEvidence(0.95, 0.02, 0.02, 0.01, 1.0, 1.0),
        pdr.StepMotionEvidence(0.05, 0.9, 0.02, 0.03, 1.0, 1.0),
    )

    result = estimate_adaptive_pdr(
        headings,
        observations,
        evidences,
        smoothing_mode="offline",
    )

    assert len(result.posteriors) == 2
    for posterior in result.posteriors:
        probability_sum = (
            posterior.forward_probability
            + posterior.sidestep_left_probability
            + posterior.sidestep_right_probability
            + posterior.turning_probability
        )
        np.testing.assert_allclose(probability_sum, 1.0)
        assert posterior.heading_std >= 0.0
        assert posterior.length_std_m > 0.0
        assert posterior.source == "adaptive_offline"
    assert (
        result.posteriors[1].sidestep_left_probability
        > result.posteriors[1].sidestep_right_probability
    )
    assert result.step_lengths[1] < observations[1].nominal_length_m


def test_adaptive_recovery_uses_absolute_stride_scale_without_compounding() -> None:
    n_particles = 60
    result = _generate_recovery_candidates(
        previous_particles=np.zeros((n_particles, 2)),
        previous_heading_correction=np.zeros(n_particles),
        previous_heading_drift=np.zeros(n_particles),
        previous_stride_scale=np.full(n_particles, 0.7),
        proposed_motion_state=np.zeros(n_particles, dtype=np.int8),
        previous_weights=np.full(n_particles, 1.0 / n_particles),
        angle_det=0.0,
        step_length=1.0,
        sigma_step_length_ratio=0.0,
        n_particles=n_particles,
        map_gray=np.full((41, 41), 255.0),
        gx_mean=0.0,
        gz_mean=9.8,
        origin_px=(20, 20),
        scale=1.0,
        heading_sigma=np.deg2rad(10.0),
        max_attempts=1,
        rng=np.random.default_rng(42),
        preserve_route_branches=False,
        allow_stride_adaptation=True,
    )

    assert result is not None
    assert float(np.mean(result.stride_scale)) > 0.6
    assert float(np.mean(result.stride_scale)) < 0.9


def test_particle_filter_recovers_with_map_aware_direction_candidates(tmp_path) -> None:
    map_gray = np.zeros((21, 21), dtype=float)
    map_gray[1:20, 10] = 1.0
    floormap_path = tmp_path / "vertical_corridor.png"
    plt.imsave(floormap_path, map_gray, cmap="gray", vmin=0.0, vmax=1.0)
    diagnostics: list[ParticleFilterStepDiagnostics] = []

    result = run_particle_filter(
        np.array([0, 1]),
        pd.DataFrame({"low_angle": [0.0]}),
        pd.DataFrame({"h_y": [0.0], "h_z": [0.0]}),
        gx_mean=0.0,
        gz_mean=9.8,
        floormap_path=floormap_path,
        origin_px=(10, 10),
        scale=1.0,
        n_particles=80,
        sigma_init_heading=0.0,
        sigma_heading=0.0,
        sigma_sl_ratio=0.0,
        stride_scale_init_sigma=0.0,
        stride_scale_process_sigma=0.0,
        stride_scale_rejuvenation_sigma=0.0,
        prepared_step_headings=[
            _forward_step_heading(step_index=1)._replace(
                movement_type="turning",
                trajectory_movement_type="turning",
            ),
            _forward_step_heading(step_index=2)._replace(
                movement_type="turning",
                trajectory_movement_type="turning",
            ),
        ],
        prepared_step_lengths=[3.0, 3.0],
        prepared_step_times=[0.0, 1.0],
        seed=42,
        recovery_valid_ratio=0.5,
        recovery_heading_sigma=0.03,
        diagnostics_collector=diagnostics,
    )

    assert diagnostics[0].recovery_attempted is True
    assert diagnostics[0].recovery_mode in {"local_grid", "turn_grid"}
    assert diagnostics[0].recovery_valid_count > 0
    assert diagnostics[0].ess_after_observation <= 1.0
    np.testing.assert_allclose(diagnostics[0].ess_after_resampling, 80.0)
    # 1歩目のrecovery方位を永続補正として保持するため、2歩目は
    # 同じ通路方向へ通常伝播でき、再度のrecoveryを必要としない。
    assert diagnostics[1].recovery_attempted is True
    assert diagnostics[1].recovery_mode in {"local_grid", "turn_grid"}
    np.testing.assert_allclose(
        [item.stride_scale_mean for item in diagnostics],
        [1.03, 1.03],
        atol=1e-12,
    )
    assert result[0][1] != [0.0, 0.0]
    trajectory = np.asarray(result[0], dtype=float)
    assert _evaluate_particle_transitions(
        trajectory[:-1],
        trajectory[1:],
        _normalize_floormap_gray(plt.imread(floormap_path)),
        gx_mean=0.0,
        gz_mean=9.8,
        origin_px=(10, 10),
        scale=1.0,
    ).all()


def test_particle_filter_rejects_origin_on_wall(tmp_path) -> None:
    floormap_path = tmp_path / "wall.png"
    plt.imsave(floormap_path, np.zeros((5, 5), dtype=float), cmap="gray")

    with np.testing.assert_raises_regex(ValueError, "origin_px"):
        run_particle_filter(
            np.array([], dtype=int),
            pd.DataFrame({"low_angle": []}),
            pd.DataFrame({"h_y": [], "h_z": []}),
            gx_mean=0.0,
            gz_mean=9.8,
            floormap_path=floormap_path,
            origin_px=(2, 2),
            scale=1.0,
            n_particles=4,
            prepared_step_headings=[],
            prepared_step_lengths=[],
            prepared_step_times=[],
            seed=1,
        )


def test_prepare_pdr_steps_returns_shared_step_result_without_steps() -> None:
    df_acc = pd.DataFrame(
        {
            "t": np.arange(5, dtype=float) * 0.01,
            "x": np.zeros(5),
            "y": np.zeros(5),
            "z": np.full(5, 9.8),
        }
    )
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(5, dtype=float) * 0.01,
            "x": np.zeros(5),
            "y": np.zeros(5),
            "z": np.zeros(5),
        }
    )

    prepared = pdr.prepare_pdr_steps(df_acc, df_gyro, gyro_bias_method="quietest")

    assert prepared.step_detection.peaks.tolist() == []
    assert prepared.trajectory == [[0.0, 0.0]]
    assert prepared.step_lengths == []
    assert prepared.t_at_steps == []
    assert prepared.step_headings == []
    assert prepared.motion_evidences == ()
    assert prepared.motion_observations == ()
    assert "low_angle" in prepared.df_gyro.columns


def test_paper_vertical_threshold_detects_step_segments() -> None:
    signal = np.zeros(220)
    signal[[10, 70, 130, 190]] = [4.0, 5.0, 4.5, 5.5]
    df_acc = pd.DataFrame({"v_acc": signal, "low_lin_norm": np.zeros_like(signal)})

    result = pdr.detect_step_result(df_acc, method="paper_vertical_threshold")

    assert result.method == "paper_vertical_threshold"
    assert result.polarity == 1
    assert result.threshold is not None
    np.testing.assert_array_equal(result.peaks, np.array([70, 130, 190]))
    assert result.segments == (
        pdr.StepSegment(start_index=10, end_index=70, contact_index=70),
        pdr.StepSegment(start_index=70, end_index=130, contact_index=130),
        pdr.StepSegment(start_index=130, end_index=190, contact_index=190),
    )


def test_paper_vertical_threshold_handles_negative_contact_polarity() -> None:
    signal = np.zeros(160)
    signal[[20, 80, 140]] = [-4.0, -5.0, -4.5]
    df_acc = pd.DataFrame({"v_acc": signal, "low_lin_norm": np.zeros_like(signal)})

    result = pdr.detect_step_result(df_acc, method="paper_vertical_threshold")

    assert result.polarity == -1
    np.testing.assert_array_equal(result.peaks, np.array([80, 140]))
    assert result.segments == (
        pdr.StepSegment(start_index=20, end_index=80, contact_index=80),
        pdr.StepSegment(start_index=80, end_index=140, contact_index=140),
    )


def test_paper_vertical_threshold_suppresses_close_contacts() -> None:
    signal = np.zeros(180)
    signal[[10, 40, 100, 160]] = [4.0, 8.0, 5.0, 5.0]
    df_acc = pd.DataFrame({"v_acc": signal, "low_lin_norm": np.zeros_like(signal)})

    result = pdr.detect_step_result(df_acc, method="paper_vertical_threshold")

    np.testing.assert_array_equal(result.peaks, np.array([100, 160]))
    assert result.segments == (
        pdr.StepSegment(start_index=40, end_index=100, contact_index=100),
        pdr.StepSegment(start_index=100, end_index=160, contact_index=160),
    )


def test_paper_vertical_threshold_filters_invalid_segment_lengths() -> None:
    signal = np.zeros(260)
    signal[[10, 20, 90, 210]] = [4.0, 5.0, 5.0, 5.0]
    df_acc = pd.DataFrame({"v_acc": signal, "low_lin_norm": np.zeros_like(signal)})

    result = pdr.detect_step_result(df_acc, method="paper_vertical_threshold")

    np.testing.assert_array_equal(result.peaks, np.array([90]))
    assert result.segments == (
        pdr.StepSegment(start_index=20, end_index=90, contact_index=90),
    )


def test_detect_steps_returns_paper_detection_peaks() -> None:
    signal = np.zeros(160)
    signal[[20, 80, 140]] = [4.0, 5.0, 4.5]
    df_acc = pd.DataFrame({"v_acc": signal, "low_lin_norm": np.zeros_like(signal)})

    result = pdr.detect_step_result(df_acc, method="paper_vertical_threshold")

    np.testing.assert_array_equal(
        pdr.detect_steps(df_acc, method="paper_vertical_threshold"),
        result.peaks,
    )


def test_resolve_step_heading_uses_accel_method1_when_requested() -> None:
    n_samples = 100
    df_acc = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "h_y": np.zeros(n_samples),
            "h_z": np.zeros(n_samples),
        }
    )
    df_acc.loc[20, "h_y"] = 2.0
    df_acc.loc[50, "h_y"] = -3.0
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "low_angle": np.full(n_samples, np.pi / 2),
        }
    )

    heading = pdr.resolve_step_heading(
        np.array([10, 70]),
        df_gyro,
        df_acc,
        0,
        initial_direction=0.0,
        heading_method="accel_method1",
    )

    assert heading.source == "accel_method1"
    assert heading.confidence == 1.0
    assert heading.peak1_index == 20
    assert heading.peak2_index == 50
    assert heading.selected_heading is not None
    assert heading.accel_method2_heading is not None
    np.testing.assert_allclose(heading.selected_heading, 0.0, atol=1e-12)
    np.testing.assert_allclose(abs(heading.accel_method2_heading), np.pi, atol=1e-12)


def test_resolve_step_heading_gyro_accel_motion_detects_sidestep() -> None:
    n_samples = 100
    df_acc = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "h_y": np.zeros(n_samples),
            "h_z": np.zeros(n_samples),
        }
    )
    df_acc.loc[20, "h_z"] = 10.0
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "low_angle": np.zeros(n_samples),
        }
    )

    heading = pdr.resolve_step_heading(
        np.array([10, 70]),
        df_gyro,
        df_acc,
        0,
        initial_direction=0.0,
        heading_method="gyro_accel_motion",
        sidestep_min_lateral_displacement=0.001,
    )

    assert heading.source == "gyro_accel_motion"
    assert heading.movement_type == "sidestep_left"
    assert heading.motion_reject_reason is None
    assert heading.selected_heading is not None
    assert heading.motion_heading is not None
    assert heading.lateral_displacement is not None
    np.testing.assert_allclose(heading.body_heading, 0.0, atol=1e-12)
    np.testing.assert_allclose(heading.selected_heading, np.pi / 2, atol=1e-12)
    assert heading.lateral_displacement > 0


def test_estimate_step_motion_uses_sidestep_left_adjustment() -> None:
    n_samples = 100
    df_acc = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "h_y": np.zeros(n_samples),
            "h_z": np.zeros(n_samples),
        }
    )
    df_acc.loc[20, "h_z"] = 10.0
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "low_angle": np.zeros(n_samples),
        }
    )

    heading = pdr.resolve_step_heading(
        np.array([10, 70]),
        df_gyro,
        df_acc,
        0,
        initial_direction=0.0,
        heading_method="gyro",
        sidestep_min_lateral_displacement=0.001,
    )
    motion = pdr.estimate_step_motion(heading, 1.0)

    assert motion is not None
    assert motion.movement_type == "sidestep_left"
    np.testing.assert_allclose(motion.heading, np.pi / 2, atol=1e-12)
    np.testing.assert_allclose(motion.length, 0.8, atol=1e-12)


def test_estimate_step_motion_uses_sidestep_right_adjustment() -> None:
    n_samples = 100
    df_acc = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "h_y": np.zeros(n_samples),
            "h_z": np.zeros(n_samples),
        }
    )
    df_acc.loc[20, "h_z"] = -10.0
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "low_angle": np.zeros(n_samples),
        }
    )

    heading = pdr.resolve_step_heading(
        np.array([10, 70]),
        df_gyro,
        df_acc,
        0,
        initial_direction=0.0,
        heading_method="gyro",
        sidestep_min_lateral_displacement=0.001,
    )
    motion = pdr.estimate_step_motion(heading, 1.0)

    assert motion is not None
    assert motion.movement_type == "sidestep_right"
    np.testing.assert_allclose(motion.heading, -np.pi / 2, atol=1e-12)
    np.testing.assert_allclose(motion.length, pdr.SIDESTEP_LENGTH_SCALE, atol=1e-12)


def test_classify_movement_type_uses_configurable_sidestep_ratio() -> None:
    assert (
        pdr._classify_movement_type(
            forward_displacement=1.0,
            lateral_displacement=1.2,
            sidestep_lateral_ratio=1.0,
            sidestep_min_lateral_displacement=0.1,
        )
        == "sidestep_left"
    )
    assert (
        pdr._classify_movement_type(
            forward_displacement=1.0,
            lateral_displacement=1.2,
            sidestep_lateral_ratio=1.5,
            sidestep_min_lateral_displacement=0.1,
        )
        == "unknown"
    )


def test_classify_movement_type_requires_min_lateral_displacement() -> None:
    assert (
        pdr._classify_movement_type(
            forward_displacement=0.001,
            lateral_displacement=0.02,
            sidestep_lateral_ratio=1.0,
            sidestep_min_lateral_displacement=0.06,
        )
        == "unknown"
    )


def test_estimate_step_motion_uses_turning_adjustment() -> None:
    n_samples = 100
    df_acc = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "h_y": np.zeros(n_samples),
            "h_z": np.zeros(n_samples),
        }
    )
    df_acc.loc[20, "h_y"] = 10.0
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "low_angle": np.linspace(0.0, 1.5, n_samples),
        }
    )

    heading = pdr.resolve_step_heading(
        np.array([10, 70]),
        df_gyro,
        df_acc,
        0,
        initial_direction=0.0,
        heading_method="gyro",
    )
    previous_heading = 0.25
    motion = pdr.estimate_step_motion(heading, 1.0, previous_heading)

    assert motion is not None
    assert motion.movement_type == "turning"
    np.testing.assert_allclose(motion.heading, previous_heading, atol=1e-12)
    np.testing.assert_allclose(motion.length, pdr.TURNING_LENGTH_SCALE, atol=1e-12)


def test_resolve_step_heading_respects_sidestep_ratio_parameter() -> None:
    n_samples = 100
    df_acc = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "h_y": np.zeros(n_samples),
            "h_z": np.zeros(n_samples),
        }
    )
    df_acc.loc[20, "h_z"] = 10.0
    df_acc.loc[30, "h_y"] = 8.0
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "low_angle": np.zeros(n_samples),
        }
    )

    heading = pdr.resolve_step_heading(
        np.array([10, 70]),
        df_gyro,
        df_acc,
        0,
        initial_direction=0.0,
        heading_method="gyro_accel_motion",
        sidestep_lateral_ratio=1.0,
        sidestep_min_lateral_displacement=0.001,
    )

    assert heading.movement_type == "sidestep_left"
    assert heading.sidestep_lateral_ratio == 1.0
    assert heading.sidestep_min_lateral_displacement == 0.001


def test_resolve_step_heading_applies_motion_heading_correction() -> None:
    n_samples = 100
    df_acc = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "h_y": np.zeros(n_samples),
            "h_z": np.zeros(n_samples),
        }
    )
    df_acc.loc[20, "h_z"] = 10.0
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "low_angle": np.zeros(n_samples),
        }
    )

    heading = pdr.resolve_step_heading(
        np.array([10, 70]),
        df_gyro,
        df_acc,
        0,
        initial_direction=0.0,
        heading_method="gyro_accel_motion",
        motion_heading_correction=np.pi / 2,
    )

    assert heading.source == "gyro_accel_motion"
    assert heading.movement_type == "forward"
    assert heading.selected_heading is not None
    assert heading.forward_displacement is not None
    np.testing.assert_allclose(heading.selected_heading, 0.0, atol=1e-12)
    assert heading.forward_displacement > 0


def test_estimate_motion_heading_correction_uses_initial_forward_steps() -> None:
    n_samples = 220
    df_acc = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "h_y": np.zeros(n_samples),
            "h_z": np.zeros(n_samples),
        }
    )
    df_acc.loc[[20, 80, 140], "h_z"] = 10.0
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "low_angle": np.zeros(n_samples),
        }
    )

    correction = pdr._estimate_motion_heading_correction(
        df_acc,
        df_gyro,
        np.array([10, 70, 130, 190]),
        initial_direction=0.0,
    )

    np.testing.assert_allclose(correction, np.pi / 2, atol=1e-12)


def test_resolve_motion_heading_correction_can_be_disabled() -> None:
    n_samples = 220
    df_acc = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "h_y": np.zeros(n_samples),
            "h_z": np.zeros(n_samples),
        }
    )
    df_acc.loc[[20, 80, 140], "h_z"] = 10.0
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "low_angle": np.zeros(n_samples),
        }
    )

    assert (
        pdr._resolve_motion_heading_correction(
            df_acc,
            df_gyro,
            np.array([10, 70, 130, 190]),
            initial_direction=0.0,
            step_segments=(),
            method="none",
        )
        == 0.0
    )


def test_apply_device_orientation_to_horizontal_flips_candidate_axes() -> None:
    h_y = np.array([1.0, -2.0])
    h_z = np.array([3.0, -4.0])

    normal_y, normal_z = pdr._apply_device_orientation_to_horizontal(
        h_y,
        h_z,
        "normal",
    )
    fb_y, fb_z = pdr._apply_device_orientation_to_horizontal(
        h_y,
        h_z,
        "front_back_inverted",
    )
    lr_y, lr_z = pdr._apply_device_orientation_to_horizontal(
        h_y,
        h_z,
        "left_right_inverted",
    )
    rot_y, rot_z = pdr._apply_device_orientation_to_horizontal(
        h_y,
        h_z,
        "rotated_180",
    )

    np.testing.assert_array_equal(normal_y, h_y)
    np.testing.assert_array_equal(normal_z, h_z)
    np.testing.assert_array_equal(fb_y, -h_y)
    np.testing.assert_array_equal(fb_z, h_z)
    np.testing.assert_array_equal(lr_y, h_y)
    np.testing.assert_array_equal(lr_z, -h_z)
    np.testing.assert_array_equal(rot_y, -h_y)
    np.testing.assert_array_equal(rot_z, -h_z)


def test_estimate_device_orientation_mode_prefers_initial_forward_alignment() -> None:
    n_samples = 220
    df_acc = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "h_y": np.zeros(n_samples),
            "h_z": np.zeros(n_samples),
        }
    )
    df_acc.loc[[20, 80, 140], "h_y"] = -10.0
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "low_angle": np.zeros(n_samples),
        }
    )

    mode = pdr._estimate_device_orientation_mode(
        df_acc,
        df_gyro,
        np.array([10, 70, 130, 190]),
        initial_direction=0.0,
    )

    assert mode == "front_back_inverted"


def test_smooth_step_headings_suppresses_isolated_sidestep() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "body_heading": 0.0,
        "motion_heading": 0.0,
        "forward_displacement": 1.0,
        "lateral_displacement": 0.0,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
    }
    headings = [
        pdr.StepHeading(step_index=1, movement_type="forward", **base),
        pdr.StepHeading(step_index=2, movement_type="sidestep_left", **base),
        pdr.StepHeading(step_index=3, movement_type="forward", **base),
    ]

    smoothed = pdr._smooth_step_headings(headings, method="isolated")

    assert [heading.movement_type for heading in smoothed] == [
        "forward",
        "sidestep_left",
        "forward",
    ]
    assert [heading.trajectory_movement_type for heading in smoothed] == [
        None,
        "forward",
        None,
    ]

    unsmoothed = pdr._smooth_step_headings(headings, method="none")
    assert [heading.movement_type for heading in unsmoothed] == [
        "forward",
        "sidestep_left",
        "forward",
    ]


def test_smooth_step_headings_clustered_suppresses_single_sidestep() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "body_heading": 0.0,
        "motion_heading": 0.0,
        "forward_displacement": 1.0,
        "lateral_displacement": 0.0,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
    }
    headings = [
        pdr.StepHeading(step_index=1, movement_type="forward", **base),
        pdr.StepHeading(step_index=2, movement_type="sidestep_left", **base),
        pdr.StepHeading(step_index=3, movement_type="forward", **base),
    ]

    smoothed = pdr._smooth_step_headings(headings, method="clustered")

    assert [heading.trajectory_movement_type for heading in smoothed] == [
        None,
        "forward",
        None,
    ]


def test_smooth_step_headings_clustered_suppresses_separated_same_direction() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "body_heading": 0.0,
        "motion_heading": 0.0,
        "forward_displacement": 1.0,
        "lateral_displacement": 0.0,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
    }
    headings = [
        pdr.StepHeading(step_index=1, movement_type="forward", **base),
        pdr.StepHeading(step_index=2, movement_type="sidestep_left", **base),
        pdr.StepHeading(step_index=3, movement_type="forward", **base),
        pdr.StepHeading(step_index=4, movement_type="sidestep_left", **base),
        pdr.StepHeading(step_index=5, movement_type="forward", **base),
    ]

    smoothed = pdr._smooth_step_headings(headings, method="clustered")

    assert [heading.trajectory_movement_type for heading in smoothed] == [
        None,
        "forward",
        None,
        "forward",
        None,
    ]


def test_smooth_step_headings_clustered_keeps_consecutive_same_direction() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "body_heading": 0.0,
        "motion_heading": 0.0,
        "forward_displacement": 0.02,
        "lateral_displacement": -0.12,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
    }
    headings = [
        pdr.StepHeading(step_index=1, movement_type="forward", **base),
        pdr.StepHeading(step_index=2, movement_type="sidestep_right", **base),
        pdr.StepHeading(step_index=3, movement_type="sidestep_right", **base),
        pdr.StepHeading(step_index=4, movement_type="forward", **base),
    ]

    smoothed = pdr._smooth_step_headings(headings, method="clustered")

    assert [heading.trajectory_movement_type for heading in smoothed] == [
        None,
        "sidestep_right",
        "sidestep_right",
        None,
    ]
    np.testing.assert_allclose(smoothed[1].selected_heading, 0.0, atol=1e-12)
    np.testing.assert_allclose(smoothed[2].selected_heading, 0.0, atol=1e-12)


def test_smooth_step_headings_clustered_does_not_override_sidestep_heading() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "body_heading": 0.0,
        "forward_displacement": 0.02,
        "lateral_displacement": 0.12,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
    }
    headings = [
        pdr.StepHeading(
            step_index=1,
            movement_type="sidestep_left",
            motion_heading=np.deg2rad(170.0),
            **base,
        ),
        pdr.StepHeading(
            step_index=2,
            movement_type="sidestep_left",
            motion_heading=np.deg2rad(-170.0),
            **base,
        ),
    ]

    smoothed = pdr._smooth_step_headings(headings, method="clustered")

    assert [heading.trajectory_movement_type for heading in smoothed] == [
        "sidestep_left",
        "sidestep_left",
    ]
    np.testing.assert_allclose(smoothed[0].selected_heading, 0.0, atol=1e-12)
    np.testing.assert_allclose(smoothed[1].selected_heading, 0.0, atol=1e-12)


def test_smooth_step_headings_clustered_requires_lateral_feature_strength() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "body_heading": 0.0,
        "motion_heading": 0.0,
        "forward_displacement": 0.2,
        "lateral_displacement": -0.02,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
    }
    headings = [
        pdr.StepHeading(step_index=1, movement_type="forward", **base),
        pdr.StepHeading(step_index=2, movement_type="sidestep_right", **base),
        pdr.StepHeading(step_index=3, movement_type="sidestep_right", **base),
        pdr.StepHeading(step_index=4, movement_type="forward", **base),
    ]

    smoothed = pdr._smooth_step_headings(headings, method="clustered")

    assert [heading.trajectory_movement_type for heading in smoothed] == [
        None,
        "forward",
        "forward",
        None,
    ]


def test_smooth_step_headings_clustered_requires_same_direction() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "body_heading": 0.0,
        "motion_heading": 0.0,
        "forward_displacement": 1.0,
        "lateral_displacement": 0.0,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
    }
    headings = [
        pdr.StepHeading(step_index=1, movement_type="forward", **base),
        pdr.StepHeading(step_index=2, movement_type="sidestep_left", **base),
        pdr.StepHeading(step_index=3, movement_type="sidestep_right", **base),
        pdr.StepHeading(step_index=4, movement_type="forward", **base),
    ]

    smoothed = pdr._smooth_step_headings(headings, method="clustered")

    assert [heading.trajectory_movement_type for heading in smoothed] == [
        None,
        "forward",
        "forward",
        None,
    ]


def test_smooth_step_headings_clustered_uses_body_motion_lateral_evidence() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "body_heading": 0.0,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
    }
    headings = [
        pdr.StepHeading(
            step_index=1,
            movement_type="forward",
            motion_heading=np.pi / 2,
            forward_displacement=0.05,
            lateral_displacement=0.08,
            **base,
        ),
        pdr.StepHeading(
            step_index=2,
            movement_type="unknown",
            motion_heading=np.deg2rad(100.0),
            forward_displacement=0.04,
            lateral_displacement=0.07,
            **base,
        ),
    ]

    smoothed = pdr._smooth_step_headings(headings, method="clustered")

    assert [heading.trajectory_movement_type for heading in smoothed] == [
        "sidestep_left",
        "sidestep_left",
    ]
    assert [heading.sidestep_evidence_reason for heading in smoothed] == [
        "body_motion_lateral",
        "body_motion_lateral",
    ]
    assert [heading.sidestep_evidence_direction for heading in smoothed] == [
        "left",
        "left",
    ]
    assert [heading.sidestep_cluster_id for heading in smoothed] == [1, 1]
    np.testing.assert_allclose(smoothed[0].selected_heading, 0.0, atol=1e-12)
    np.testing.assert_allclose(smoothed[1].selected_heading, 0.0, atol=1e-12)


def test_smooth_step_headings_clustered_includes_single_bridge_gap() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "body_heading": 0.0,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
    }
    headings = [
        pdr.StepHeading(
            step_index=1,
            movement_type="forward",
            motion_heading=np.pi / 2,
            forward_displacement=0.05,
            lateral_displacement=-0.08,
            **base,
        ),
        pdr.StepHeading(
            step_index=2,
            movement_type="forward",
            motion_heading=0.0,
            forward_displacement=0.2,
            lateral_displacement=-0.04,
            **base,
        ),
        pdr.StepHeading(
            step_index=3,
            movement_type="unknown",
            motion_heading=np.deg2rad(100.0),
            forward_displacement=0.04,
            lateral_displacement=-0.07,
            **base,
        ),
    ]

    smoothed = pdr._smooth_step_headings(headings, method="clustered")

    assert [heading.trajectory_movement_type for heading in smoothed] == [
        "sidestep_right",
        "sidestep_right",
        "sidestep_right",
    ]
    assert smoothed[1].sidestep_evidence_reason == "bridge_gap"
    np.testing.assert_allclose(smoothed[0].selected_heading, 0.0, atol=1e-12)
    np.testing.assert_allclose(smoothed[1].selected_heading, 0.0, atol=1e-12)
    np.testing.assert_allclose(smoothed[2].selected_heading, 0.0, atol=1e-12)


def test_smooth_step_headings_clustered_marks_strong_single_as_suspect() -> None:
    heading = pdr.StepHeading(
        step_index=1,
        timestamp_s=0.0,
        gyro_heading=0.0,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=0.0,
        source="gyro_accel_motion",
        confidence=0.0,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=0.0,
        motion_heading=np.deg2rad(-100.0),
        movement_type="forward",
        forward_displacement=0.02,
        lateral_displacement=-0.09,
        motion_confidence=1.0,
        motion_reject_reason=None,
    )

    smoothed = pdr._smooth_step_headings(
        [heading],
        method="clustered",
        sidestep_suspect_mode="motion",
    )

    assert smoothed[0].trajectory_movement_type == "sidestep_suspect_right"
    assert smoothed[0].sidestep_evidence_reason == "body_motion_lateral"
    assert smoothed[0].sidestep_cluster_id is None
    np.testing.assert_allclose(smoothed[0].selected_heading, np.deg2rad(-100.0))


def test_smooth_step_headings_keeps_turning_when_side_cluster_is_not_confirmed() -> (
    None
):
    heading = _forward_step_heading()._replace(
        movement_type="turning_sidestep_right",
        forward_displacement=0.02,
        lateral_displacement=-0.09,
        motion_heading=np.deg2rad(-80.0),
        yaw_delta=np.deg2rad(55.0),
    )

    smoothed = pdr._smooth_step_headings([heading], method="clustered")

    assert smoothed[0].trajectory_movement_type == "turning_sidestep_right"


def test_stabilize_trajectory_headings_uses_forward_motion_per_step() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "movement_type": "forward",
        "forward_displacement": 1.0,
        "lateral_displacement": 0.0,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
    }
    headings = [
        pdr.StepHeading(
            step_index=1,
            body_heading=np.deg2rad(80.0),
            motion_heading=np.deg2rad(40.0),
            **base,
        ),
        pdr.StepHeading(
            step_index=2,
            body_heading=np.deg2rad(100.0),
            motion_heading=np.deg2rad(60.0),
            **base,
        ),
    ]

    stabilized = pdr._stabilize_trajectory_headings(
        headings,
        forward_heading_source="motion",
    )

    np.testing.assert_allclose(stabilized[0].body_heading, np.deg2rad(80.0))
    np.testing.assert_allclose(stabilized[1].body_heading, np.deg2rad(100.0))
    np.testing.assert_allclose(stabilized[0].selected_heading, np.deg2rad(40.0))
    np.testing.assert_allclose(stabilized[1].selected_heading, np.deg2rad(60.0))
    assert stabilized[0].source == "trajectory_motion"
    assert stabilized[1].source == "trajectory_motion"


def test_stabilize_trajectory_headings_rejects_initial_forward_outlier() -> None:
    heading = pdr.StepHeading(
        step_index=1,
        timestamp_s=0.0,
        gyro_heading=0.0,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=0.0,
        source="gyro_accel_motion",
        confidence=0.0,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=0.0,
        motion_heading=np.deg2rad(70.0),
        movement_type="forward",
        forward_displacement=1.0,
        lateral_displacement=0.0,
        motion_confidence=1.0,
        motion_reject_reason=None,
    )

    stabilized = pdr._stabilize_trajectory_headings(
        [heading],
        forward_heading_source="motion",
    )

    np.testing.assert_allclose(stabilized[0].selected_heading, 0.0, atol=1e-12)
    assert stabilized[0].source == "trajectory_initial_body_fallback"


def test_stabilize_trajectory_headings_limits_same_type_heading_jump() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "movement_type": "forward",
        "forward_displacement": 1.0,
        "lateral_displacement": 0.0,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
    }
    headings = [
        pdr.StepHeading(
            step_index=1,
            body_heading=np.deg2rad(0.0),
            motion_heading=np.deg2rad(0.0),
            **base,
        ),
        pdr.StepHeading(
            step_index=2,
            body_heading=np.deg2rad(0.0),
            motion_heading=np.deg2rad(60.0),
            **base,
        ),
    ]

    stabilized = pdr._stabilize_trajectory_headings(
        headings,
        forward_heading_source="motion",
    )

    np.testing.assert_allclose(stabilized[0].selected_heading, 0.0, atol=1e-12)
    np.testing.assert_allclose(stabilized[1].selected_heading, np.deg2rad(25.0))
    assert stabilized[1].source == "trajectory_motion_limited"


def test_stabilize_trajectory_headings_can_use_forward_body_source() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "motion_heading": np.deg2rad(50.0),
        "movement_type": "forward",
        "forward_displacement": 1.0,
        "lateral_displacement": 0.0,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
    }
    headings = [
        pdr.StepHeading(step_index=1, body_heading=np.deg2rad(80.0), **base),
        pdr.StepHeading(step_index=2, body_heading=np.deg2rad(100.0), **base),
    ]

    stabilized = pdr._stabilize_trajectory_headings(
        headings,
        forward_heading_source="body",
    )

    np.testing.assert_allclose(stabilized[0].selected_heading, np.deg2rad(80.0))
    np.testing.assert_allclose(stabilized[1].selected_heading, np.deg2rad(100.0))
    assert stabilized[0].source == "trajectory_body"
    assert stabilized[1].source == "trajectory_body"


def test_stabilize_trajectory_headings_uses_sidestep_motion_with_constraint() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "movement_type": "sidestep_right",
        "forward_displacement": 0.0,
        "lateral_displacement": -0.1,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
        "trajectory_movement_type": "sidestep_right",
    }
    headings = [
        pdr.StepHeading(
            step_index=1,
            body_heading=np.deg2rad(0.0),
            motion_heading=np.deg2rad(-80.0),
            **base,
        ),
        pdr.StepHeading(
            step_index=2,
            body_heading=np.deg2rad(0.0),
            motion_heading=np.deg2rad(-70.0),
            **base,
        ),
    ]

    stabilized = pdr._stabilize_trajectory_headings(headings)

    np.testing.assert_allclose(stabilized[0].selected_heading, np.deg2rad(-80.0))
    np.testing.assert_allclose(stabilized[1].selected_heading, np.deg2rad(-70.0))
    assert stabilized[0].source == "trajectory_sidestep_motion"
    assert stabilized[1].source == "trajectory_sidestep_motion"


def test_stabilize_trajectory_headings_falls_back_for_sidestep_motion_outlier() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "movement_type": "sidestep_right",
        "forward_displacement": 0.0,
        "lateral_displacement": -0.1,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
        "trajectory_movement_type": "sidestep_right",
    }
    headings = [
        pdr.StepHeading(
            step_index=1,
            body_heading=0.0,
            motion_heading=np.deg2rad(-80.0),
            **base,
        ),
        pdr.StepHeading(
            step_index=2,
            body_heading=0.0,
            motion_heading=np.deg2rad(0.0),
            **base,
        ),
    ]

    stabilized = pdr._stabilize_trajectory_headings(headings)

    np.testing.assert_allclose(stabilized[0].selected_heading, np.deg2rad(-80.0))
    np.testing.assert_allclose(stabilized[1].selected_heading, np.deg2rad(-80.0))
    assert stabilized[1].source == "trajectory_sidestep_fallback"


def test_classify_movement_type_detects_turning_sidestep() -> None:
    assert (
        pdr._classify_movement_type(
            forward_displacement=0.02,
            lateral_displacement=0.12,
            yaw_delta=np.deg2rad(35.0),
            sidestep_min_lateral_displacement=0.03,
        )
        == "turning_sidestep_left"
    )
    assert (
        pdr._classify_movement_type(
            forward_displacement=0.20,
            lateral_displacement=0.02,
            yaw_delta=np.deg2rad(35.0),
            sidestep_min_lateral_displacement=0.03,
        )
        == "turning"
    )


def test_stabilize_trajectory_headings_prefers_motion_for_turning_sidestep() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "movement_type": "turning_sidestep_right",
        "forward_displacement": 0.01,
        "lateral_displacement": -0.12,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
        "trajectory_movement_type": "turning_sidestep_right",
    }
    heading = pdr.StepHeading(
        step_index=1,
        body_heading=np.deg2rad(0.0),
        motion_heading=np.deg2rad(20.0),
        **base,
    )

    stabilized = pdr._stabilize_trajectory_headings([heading])

    np.testing.assert_allclose(stabilized[0].selected_heading, np.deg2rad(20.0))
    assert stabilized[0].source == "trajectory_turning_sidestep_motion"


def test_stabilize_trajectory_headings_loosens_turning_sidestep_limit() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "movement_type": "turning_sidestep_left",
        "forward_displacement": 0.01,
        "lateral_displacement": 0.12,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
        "trajectory_movement_type": "turning_sidestep_left",
    }
    headings = [
        pdr.StepHeading(
            step_index=1,
            body_heading=0.0,
            motion_heading=np.deg2rad(0.0),
            **base,
        ),
        pdr.StepHeading(
            step_index=2,
            body_heading=0.0,
            motion_heading=np.deg2rad(70.0),
            **base,
        ),
    ]

    stabilized = pdr._stabilize_trajectory_headings(headings)

    np.testing.assert_allclose(stabilized[1].selected_heading, np.deg2rad(70.0))
    assert stabilized[1].source == "trajectory_turning_sidestep_motion"


def test_sidestep_suspect_is_trajectory_sidestep_movement() -> None:
    assert pdr._is_trajectory_sidestep_movement("sidestep_suspect_left")
    assert pdr._is_trajectory_sidestep_movement("sidestep_suspect_right")
    assert pdr._is_trajectory_sidestep_movement("turning_sidestep_left")
    assert pdr._is_trajectory_sidestep_movement("turning_sidestep_right")
    assert pdr._is_sidestep_suspect_movement("sidestep_suspect_left")
    assert pdr._is_sidestep_suspect_movement("sidestep_suspect_right")
    assert not pdr._is_sidestep_suspect_movement("sidestep_left")
    assert not pdr._is_trajectory_sidestep_movement("forward")


def test_smooth_step_headings_clustered_suppresses_alternating_evidence() -> None:
    base = {
        "timestamp_s": 0.0,
        "gyro_heading": 0.0,
        "accel_method1_heading": None,
        "accel_method2_heading": None,
        "selected_heading": 0.0,
        "source": "gyro_accel_motion",
        "confidence": 0.0,
        "angle_diff_method1": None,
        "angle_diff_method2": None,
        "segment_start_index": None,
        "segment_end_index": None,
        "peak1_index": None,
        "peak2_index": None,
        "body_heading": 0.0,
        "motion_confidence": 1.0,
        "motion_reject_reason": None,
    }
    headings = [
        pdr.StepHeading(
            step_index=1,
            movement_type="forward",
            motion_heading=np.deg2rad(100.0),
            forward_displacement=0.02,
            lateral_displacement=0.09,
            **base,
        ),
        pdr.StepHeading(
            step_index=2,
            movement_type="forward",
            motion_heading=np.deg2rad(-100.0),
            forward_displacement=0.02,
            lateral_displacement=-0.09,
            **base,
        ),
    ]

    smoothed = pdr._smooth_step_headings(headings, method="clustered")

    assert [heading.trajectory_movement_type for heading in smoothed] == [
        "forward",
        "forward",
    ]


def test_estimate_step_motion_uses_trajectory_movement_type_override() -> None:
    heading = pdr.StepHeading(
        step_index=1,
        timestamp_s=0.0,
        gyro_heading=0.0,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=np.pi / 2,
        source="gyro_accel_motion",
        confidence=0.0,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=0.0,
        motion_heading=np.pi / 2,
        movement_type="sidestep_left",
        forward_displacement=0.0,
        lateral_displacement=1.0,
        motion_confidence=1.0,
        motion_reject_reason=None,
        trajectory_movement_type="forward",
    )

    motion = pdr.estimate_step_motion(
        heading,
        1.0,
        forward_heading_source="motion",
    )

    assert motion is not None
    assert motion.movement_type == "forward"
    np.testing.assert_allclose(motion.heading, np.pi / 2, atol=1e-12)
    np.testing.assert_allclose(motion.length, 1.0, atol=1e-12)


def test_estimate_step_motion_can_use_motion_heading_for_forward() -> None:
    heading = pdr.StepHeading(
        step_index=1,
        timestamp_s=0.0,
        gyro_heading=0.0,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=np.pi / 2,
        source="gyro_accel_motion",
        confidence=0.0,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=0.0,
        motion_heading=np.pi / 2,
        movement_type="forward",
        forward_displacement=1.0,
        lateral_displacement=0.0,
        motion_confidence=1.0,
        motion_reject_reason=None,
    )

    motion = pdr.estimate_step_motion(
        heading,
        1.0,
        forward_heading_source="motion",
    )

    assert motion is not None
    assert motion.movement_type == "forward"
    np.testing.assert_allclose(motion.heading, np.pi / 2, atol=1e-12)
    np.testing.assert_allclose(motion.length, 1.0, atol=1e-12)


def test_estimate_step_motion_sidestep_defaults_to_motion_heading() -> None:
    heading = pdr.StepHeading(
        step_index=1,
        timestamp_s=0.0,
        gyro_heading=0.0,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=np.pi / 3,
        source="gyro_accel_motion",
        confidence=0.0,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=0.0,
        motion_heading=np.pi / 3,
        movement_type="sidestep_right",
        forward_displacement=0.02,
        lateral_displacement=-0.12,
        motion_confidence=1.0,
        motion_reject_reason=None,
        trajectory_movement_type="sidestep_right",
    )

    motion = pdr.estimate_step_motion(
        heading,
        1.0,
        sidestep_suspect_mode="motion",
    )

    assert motion is not None
    assert motion.movement_type == "sidestep_right"
    np.testing.assert_allclose(motion.heading, np.pi / 3, atol=1e-12)
    np.testing.assert_allclose(motion.length, pdr.SIDESTEP_LENGTH_SCALE, atol=1e-12)


def test_estimate_step_motion_sidestep_can_use_motion_heading() -> None:
    heading = pdr.StepHeading(
        step_index=1,
        timestamp_s=0.0,
        gyro_heading=0.0,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=0.0,
        source="gyro_accel_motion",
        confidence=0.0,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=0.0,
        motion_heading=np.pi / 3,
        movement_type="sidestep_right",
        forward_displacement=0.02,
        lateral_displacement=-0.12,
        motion_confidence=1.0,
        motion_reject_reason=None,
        trajectory_movement_type="sidestep_right",
    )

    motion = pdr.estimate_step_motion(
        heading,
        1.0,
        sidestep_heading_source="motion",
    )

    assert motion is not None
    assert motion.movement_type == "sidestep_right"
    np.testing.assert_allclose(motion.heading, np.pi / 3, atol=1e-12)
    np.testing.assert_allclose(motion.length, pdr.SIDESTEP_LENGTH_SCALE, atol=1e-12)


def test_estimate_step_motion_sidestep_falls_back_without_selected_heading() -> None:
    heading = pdr.StepHeading(
        step_index=1,
        timestamp_s=0.0,
        gyro_heading=0.0,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=None,
        source="gyro_accel_motion",
        confidence=0.0,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=0.0,
        motion_heading=None,
        movement_type="sidestep_right",
        forward_displacement=0.02,
        lateral_displacement=-0.12,
        motion_confidence=1.0,
        motion_reject_reason=None,
        trajectory_movement_type="sidestep_right",
    )

    motion = pdr.estimate_step_motion(heading, 1.0)

    assert motion is not None
    assert motion.movement_type == "sidestep_right"
    np.testing.assert_allclose(motion.heading, -np.pi / 2, atol=1e-12)
    np.testing.assert_allclose(motion.length, pdr.SIDESTEP_LENGTH_SCALE, atol=1e-12)


def test_estimate_step_motion_sidestep_suspect_uses_motion_heading() -> None:
    heading = pdr.StepHeading(
        step_index=1,
        timestamp_s=0.0,
        gyro_heading=0.0,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=None,
        source="gyro_accel_motion",
        confidence=0.0,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=0.0,
        motion_heading=np.pi / 3,
        movement_type="forward",
        forward_displacement=0.02,
        lateral_displacement=-0.12,
        motion_confidence=1.0,
        motion_reject_reason=None,
        trajectory_movement_type="sidestep_suspect_right",
    )

    motion = pdr.estimate_step_motion(
        heading,
        1.0,
        sidestep_suspect_mode="motion",
    )

    assert motion is not None
    assert motion.movement_type == "sidestep_suspect_right"
    np.testing.assert_allclose(motion.heading, np.pi / 3, atol=1e-12)
    np.testing.assert_allclose(motion.length, pdr.SIDESTEP_LENGTH_SCALE, atol=1e-12)


def test_estimate_step_motion_forward_body_falls_back_without_body_heading() -> None:
    heading = pdr.StepHeading(
        step_index=1,
        timestamp_s=0.0,
        gyro_heading=None,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=np.pi / 4,
        source="gyro_accel_motion",
        confidence=0.0,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=None,
        motion_heading=np.pi / 2,
        movement_type="forward",
        forward_displacement=1.0,
        lateral_displacement=0.0,
        motion_confidence=1.0,
        motion_reject_reason=None,
    )

    motion = pdr.estimate_step_motion(
        heading,
        1.0,
        forward_heading_source="body",
    )

    assert motion is not None
    assert motion.movement_type == "forward"
    np.testing.assert_allclose(motion.heading, np.pi / 4, atol=1e-12)
    np.testing.assert_allclose(motion.length, 1.0, atol=1e-12)


def test_estimate_step_motion_unknown_uses_body_heading() -> None:
    heading = pdr.StepHeading(
        step_index=1,
        timestamp_s=0.0,
        gyro_heading=0.0,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=np.pi / 2,
        source="gyro_accel_motion",
        confidence=0.0,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=0.0,
        motion_heading=np.pi / 2,
        movement_type="unknown",
        forward_displacement=0.05,
        lateral_displacement=0.08,
        motion_confidence=1.0,
        motion_reject_reason=None,
    )

    motion = pdr.estimate_step_motion(heading, 1.0)

    assert motion is not None
    assert motion.movement_type == "unknown"
    np.testing.assert_allclose(motion.heading, 0.0, atol=1e-12)
    np.testing.assert_allclose(motion.length, 1.0, atol=1e-12)


def test_estimate_step_motion_unknown_falls_back_without_body_heading() -> None:
    heading = pdr.StepHeading(
        step_index=1,
        timestamp_s=0.0,
        gyro_heading=None,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=np.pi / 4,
        source="gyro_accel_motion",
        confidence=0.0,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=None,
        motion_heading=np.pi / 2,
        movement_type="unknown",
        forward_displacement=0.05,
        lateral_displacement=0.08,
        motion_confidence=1.0,
        motion_reject_reason=None,
    )

    motion = pdr.estimate_step_motion(heading, 1.0)

    assert motion is not None
    assert motion.movement_type == "unknown"
    np.testing.assert_allclose(motion.heading, np.pi / 4, atol=1e-12)
    np.testing.assert_allclose(motion.length, 1.0, atol=1e-12)


def test_resolve_step_heading_gyro_accel_motion_rotates_by_gyro_angle() -> None:
    n_samples = 100
    df_acc = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "h_y": np.zeros(n_samples),
            "h_z": np.zeros(n_samples),
        }
    )
    df_acc.loc[20, "h_y"] = 10.0
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "low_angle": np.full(n_samples, np.pi / 2),
        }
    )

    heading = pdr.resolve_step_heading(
        np.array([10, 70]),
        df_gyro,
        df_acc,
        0,
        initial_direction=0.0,
        heading_method="gyro_accel_motion",
    )

    assert heading.source == "gyro_accel_motion"
    assert heading.movement_type == "forward"
    assert heading.selected_heading is not None
    assert heading.body_heading is not None
    np.testing.assert_allclose(heading.body_heading, np.pi / 2, atol=1e-12)
    np.testing.assert_allclose(heading.selected_heading, np.pi / 2, atol=1e-12)


def test_resolve_step_heading_gyro_accel_motion_falls_back_to_gyro() -> None:
    n_samples = 100
    body_heading = np.deg2rad(30.0)
    df_acc = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "h_y": np.zeros(n_samples),
            "h_z": np.zeros(n_samples),
        }
    )
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(n_samples, dtype=float) * 0.01,
            "low_angle": np.full(n_samples, body_heading),
        }
    )

    heading = pdr.resolve_step_heading(
        np.array([10, 70]),
        df_gyro,
        df_acc,
        0,
        initial_direction=0.0,
        heading_method="gyro_accel_motion",
    )

    assert heading.source == "gyro"
    assert heading.movement_type == "unknown"
    assert heading.motion_reject_reason == "zero_motion"
    assert heading.selected_heading is not None
    np.testing.assert_allclose(heading.selected_heading, body_heading, atol=1e-12)


def test_normalize_floormap_gray_handles_float_and_uint8_images() -> None:
    image_uint8 = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    image_float = image_uint8.astype(float) / 255.0

    np.testing.assert_allclose(
        _normalize_floormap_gray(image_float),
        _normalize_floormap_gray(image_uint8),
    )


def test_reconstruct_resampled_paths_traces_final_particle_ancestors() -> None:
    position_history = [
        np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]),
        np.array([[1.0, 0.0], [11.0, 0.0], [21.0, 0.0]]),
        np.array([[22.0, 0.0], [2.0, 0.0], [12.0, 0.0]]),
    ]
    resample_history = [
        np.array([2, 0, 1]),
        np.array([1, 1, 0]),
    ]

    paths = _reconstruct_resampled_paths(position_history, resample_history)

    np.testing.assert_allclose(
        paths,
        np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[20.0, 0.0], [21.0, 0.0], [22.0, 0.0]],
            ]
        ),
    )


def test_reconstruct_particle_paths_supports_identity_parent_steps() -> None:
    position_history = [
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        np.array([[1.0, 0.0], [11.0, 0.0]]),
        np.array([[12.0, 0.0], [2.0, 0.0]]),
    ]
    parent_history = [
        np.array([0, 1]),
        np.array([1, 0]),
    ]

    paths = _reconstruct_particle_paths(position_history, parent_history)

    np.testing.assert_allclose(
        paths,
        np.array(
            [
                [[10.0, 0.0], [11.0, 0.0], [12.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            ]
        ),
    )


def test_snap_trajectory_to_walkable_pixels_moves_wall_points() -> None:
    map_gray = np.zeros((5, 5), dtype=float)
    map_gray[2, 1] = 255.0
    trajectory = [[1.2, 2.0], [3.0, 2.0], [-1.0, 2.0]]

    snapped = _snap_trajectory_to_walkable_pixels(
        trajectory,
        map_gray,
        gx_mean=0.0,
        gz_mean=1.0,
        origin_px=(0, 0),
        scale=1.0,
    )

    np.testing.assert_allclose(snapped, [[1.2, 2.0], [1.0, 2.0], [1.0, 2.0]])


def test_snap_trajectory_without_walkable_area_is_unchanged() -> None:
    map_gray = np.zeros((5, 5), dtype=float)
    trajectory = [[2.0, 2.0], [3.0, 2.0]]

    snapped = _snap_trajectory_to_walkable_pixels(
        trajectory,
        map_gray,
        gx_mean=0.0,
        gz_mean=1.0,
        origin_px=(0, 0),
        scale=1.0,
    )

    assert snapped == trajectory


def test_project_acceleration_to_step_axes_uses_heading_aligned_forward_axis() -> None:
    samples = (
        np.array([[2.0, 0.0], [4.0, 0.0]], dtype=float),
        np.array([0.0, 1.0], dtype=float),
    )
    df_gyro = pd.DataFrame({"t": [0.0, 1.0], "low_angle": [0.0, 0.0]})

    projected = _project_acceleration_to_step_axes(
        samples,
        df_gyro,
        dx=1.0,
        dy=0.0,
        initial_direction=0.0,
    )

    np.testing.assert_allclose(projected, np.array([[2.0, 0.0], [4.0, 0.0]]))


def test_project_acceleration_to_step_axes_applies_initial_direction() -> None:
    samples = (
        np.array([[2.0, 0.0], [4.0, 0.0]], dtype=float),
        np.array([0.0, 1.0], dtype=float),
    )
    df_gyro = pd.DataFrame({"t": [0.0, 1.0], "low_angle": [0.0, 0.0]})

    projected = _project_acceleration_to_step_axes(
        samples,
        df_gyro,
        dx=0.0,
        dy=1.0,
        initial_direction=90.0,
    )

    np.testing.assert_allclose(
        projected, np.array([[2.0, 0.0], [4.0, 0.0]]), atol=1e-12
    )


def test_project_acceleration_to_step_axes_requires_gyro_angle() -> None:
    samples = (
        np.array([[2.0, 0.0]], dtype=float),
        np.array([0.0], dtype=float),
    )

    assert (
        _project_acceleration_to_step_axes(
            samples,
            None,
            dx=1.0,
            dy=0.0,
        )
        is None
    )
    assert (
        _project_acceleration_to_step_axes(
            samples,
            pd.DataFrame({"t": [0.0], "x": [0.0]}),
            dx=1.0,
            dy=0.0,
        )
        is None
    )


def test_particle_animation_respects_save_animation_flag(tmp_path, monkeypatch) -> None:
    map_path = tmp_path / "map.png"
    plt.imsave(
        map_path,
        np.ones((8, 8)),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    df_acc = pd.DataFrame(
        {
            "t": np.arange(5, dtype=float) * 0.01,
            "x": np.zeros(5),
            "y": np.zeros(5),
            "z": np.full(5, 9.8),
        }
    )
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(5, dtype=float) * 0.01,
            "x": np.zeros(5),
            "y": np.zeros(5),
            "z": np.zeros(5),
        }
    )

    output_dirs = iter([tmp_path / "first", tmp_path / "second"])

    def fake_create_output_dir() -> Path:
        output_dir = next(output_dirs)
        output_dir.mkdir()
        return output_dir

    monkeypatch.setattr(pdr_commands, "_create_output_dir", fake_create_output_dir)
    monkeypatch.setattr(pdr, "detect_steps", lambda _df_acc: np.array([], dtype=int))

    from rikka.plot.lib import animation

    calls = 0

    def fake_save_particle_animation(*_args, **_kwargs) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        animation,
        "save_particle_animation",
        fake_save_particle_animation,
    )

    pdr.run(
        df_acc=df_acc,
        df_gyro=df_gyro,
        plot=False,
        use_particle_filter=True,
        floormap_path=map_path,
        origin_px=(0, 0),
    )
    assert calls == 0
    first_trajectory = pd.read_csv(tmp_path / "first" / "trajectory.csv")
    assert first_trajectory.columns.tolist() == ["timestamp_s", "x", "y"]
    assert first_trajectory.empty
    first_diagnostics = pd.read_csv(tmp_path / "first" / "particle_diagnostics.csv")
    assert first_diagnostics.empty
    assert "ess_after_observation" in first_diagnostics.columns
    assert "stride_scale_mean" in first_diagnostics.columns

    pdr.run(
        df_acc=df_acc,
        df_gyro=df_gyro,
        plot=False,
        use_particle_filter=True,
        save_animation=True,
        floormap_path=map_path,
        origin_px=(0, 0),
    )
    assert calls == 1
