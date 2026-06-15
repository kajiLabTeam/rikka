from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rikka.analyze import pdr
from rikka.analyze.particle_filter import (
    _normalize_floormap_gray,
    _reconstruct_resampled_paths,
    _snap_trajectory_to_walkable_pixels,
)
from rikka.analyze.sensor_plot import _project_acceleration_to_step_axes


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

    monkeypatch.setattr(pdr, "_create_output_dir", fake_create_output_dir)
    monkeypatch.setattr(pdr, "detect_steps", lambda _df_acc: np.array([], dtype=int))

    df_trajectory = pdr.run(df_acc=df_acc, df_gyro=df_gyro, plot=False)

    assert df_trajectory.columns.tolist() == ["timestamp_s", "x", "y"]
    assert df_trajectory.empty
    saved = pd.read_csv(tmp_path / "pdr" / "trajectory.csv")
    pd.testing.assert_frame_equal(saved, df_trajectory)


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
    )

    assert heading.source == "gyro_accel_motion"
    assert heading.movement_type == "sidestep"
    assert heading.motion_reject_reason is None
    assert heading.selected_heading is not None
    assert heading.motion_heading is not None
    assert heading.lateral_displacement is not None
    np.testing.assert_allclose(heading.body_heading, 0.0, atol=1e-12)
    np.testing.assert_allclose(heading.selected_heading, np.pi / 2, atol=1e-12)
    assert heading.lateral_displacement > 0


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
    plt.imsave(map_path, np.ones((8, 8)), cmap="gray")

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

    monkeypatch.setattr(pdr, "_create_output_dir", fake_create_output_dir)
    monkeypatch.setattr(pdr, "detect_steps", lambda _df_acc: np.array([], dtype=int))

    import rikka.analyze.particle_filter as particle_filter

    calls = 0

    def fake_save_particle_animation(*_args, **_kwargs) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        particle_filter,
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
