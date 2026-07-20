"""コードレビューで見つかったPDR不整合の回帰テスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rikka.analyze.pdr import adaptive_estimator, heading, step_length
from rikka.analyze.pdr.adaptive_estimator import estimate_adaptive_pdr
from rikka.analyze.pdr.direction_resolver import resolve_step_directions
from rikka.analyze.pdr.models import (
    StepHeading,
    StepLengthObservation,
    StepMotionEvidence,
    StepMotionObservation,
)


def _step_heading(
    index: int,
    *,
    selected: float = 0.0,
    body: float = 0.0,
    motion: float = 0.0,
    movement: str = "forward",
    confidence: float = 1.0,
) -> StepHeading:
    return StepHeading(
        step_index=index,
        timestamp_s=float(index),
        gyro_heading=body,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=selected,
        source="test",
        confidence=confidence,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=body,
        motion_heading=motion,
        movement_type=movement,
        forward_displacement=1.0 if movement == "forward" else 0.0,
        lateral_displacement=1.0 if movement == "sidestep_left" else 0.0,
        motion_confidence=confidence,
        motion_reject_reason=None,
        trajectory_movement_type=movement,
        yaw_delta=0.0,
    )


def _length_observation(index: int, interval: float = 1.0) -> StepLengthObservation:
    return StepLengthObservation(
        step_index=index,
        nominal_length_m=1.0,
        interval_length_m=interval,
        step_period_s=0.65,
        vertical_amplitude=2.0,
        horizontal_energy=1.0,
        quality=1.0,
        log_length_sigma=0.1,
        fallback_reason=None,
    )


def _evidence(mode: str) -> StepMotionEvidence:
    likelihoods = {
        "forward": (0.999, 0.0003, 0.0003, 0.0004),
        "sidestep_left": (0.0003, 0.999, 0.0003, 0.0004),
    }[mode]
    return StepMotionEvidence(*likelihoods, 1.0, 1.0)


def _angle_delta(left: float, right: float) -> float:
    return float((left - right + np.pi) % (2.0 * np.pi) - np.pi)


def test_adaptive_heading_keeps_validated_sensor_heading() -> None:
    step = _step_heading(
        1,
        selected=0.0,
        body=0.0,
        motion=np.pi / 2.0,
        movement="sidestep_left",
    )

    result = estimate_adaptive_pdr(
        [step],
        (_length_observation(1),),
        (_evidence("sidestep_left"),),
    )

    assert result.posteriors[0].selected_mode == "sidestep_left"
    assert abs(_angle_delta(result.posteriors[0].heading_mean, 0.0)) < 1e-6
    assert result.step_headings[0].selected_heading == result.posteriors[0].heading_mean


def test_adaptive_sidestep_start_guard_prevents_unobserved_quarter_turn() -> None:
    steps = [
        _step_heading(1),
        _step_heading(
            2,
            selected=0.0,
            body=0.0,
            motion=np.pi / 2.0,
            movement="sidestep_left",
        ),
    ]

    result = estimate_adaptive_pdr(
        steps,
        (_length_observation(1), _length_observation(2)),
        (_evidence("forward"), _evidence("sidestep_left")),
    )

    assert result.posteriors[1].selected_mode == "sidestep_left"
    assert abs(_angle_delta(result.posteriors[1].heading_mean, 0.0)) < 1e-6


def test_adaptive_offline_recomputes_length_without_fabricating_heading(
    monkeypatch,
) -> None:
    step = _step_heading(
        1,
        selected=0.0,
        body=0.0,
        motion=np.pi / 2.0,
        movement="sidestep_left",
    )
    inputs = ([step], (_length_observation(1, interval=1.4),), (_evidence("forward"),))
    causal = estimate_adaptive_pdr(*inputs, smoothing_mode="causal")
    monkeypatch.setattr(
        adaptive_estimator,
        "_smooth_mode_probabilities",
        lambda _posteriors: [np.asarray([0.01, 0.97, 0.01, 0.01])],
    )

    offline = estimate_adaptive_pdr(*inputs, smoothing_mode="offline")

    assert offline.posteriors[0].selected_mode == "sidestep_left"
    assert abs(_angle_delta(offline.posteriors[0].heading_mean, 0.0)) < 1e-6
    assert (
        offline.step_headings[0].selected_heading == offline.posteriors[0].heading_mean
    )
    assert not np.isclose(
        offline.posteriors[0].length_mean_m,
        causal.posteriors[0].length_mean_m,
    )
    assert offline.step_lengths[0] == offline.posteriors[0].length_mean_m


def test_robust_direction_tie_prefers_input_selected_heading() -> None:
    step = _step_heading(1, selected=np.pi, body=0.0, motion=0.0, confidence=0.0)
    observation = StepMotionObservation(
        step_index=1,
        timestamp_s=1.0,
        device_yaw_heading=0.0,
        body_heading_candidate=0.0,
        directed_motion_heading=0.0,
        motion_axis_heading=0.0,
        forward_displacement=0.0,
        lateral_displacement=0.0,
        displacement_norm=0.0,
        yaw_delta=0.0,
        motion_confidence=0.0,
        calibration_reliability=0.0,
        raw_movement_type="forward",
        trajectory_movement_type="forward",
        device_orientation_mode="normal",
    )

    resolved, posterior = resolve_step_directions(
        [step],
        [observation],
        smoothing_mode="offline",
    )

    assert abs(_angle_delta(resolved[0].selected_heading or 0.0, np.pi)) < 1e-6
    assert 0.49 <= posterior[0].positive_axis_probability <= 0.51
    assert 0.49 <= posterior[0].negative_axis_probability <= 0.51


def test_device_orientation_keeps_default_when_initial_motion_is_lateral(
    monkeypatch,
) -> None:
    samples = 100
    df_acc = pd.DataFrame(
        {
            "t": np.arange(samples, dtype=float) * 0.01,
            "h_y": np.zeros(samples),
            "h_z": np.zeros(samples),
        }
    )
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(samples, dtype=float) * 0.01,
            "low_angle": np.zeros(samples),
        }
    )

    def lateral_result(*_args, device_orientation_mode="normal", **_kwargs):
        is_rotated = device_orientation_mode == "rotated_180"
        return heading._MotionHeadingResult(
            body_heading=0.0,
            motion_heading=0.0 if is_rotated else np.pi / 2.0,
            movement_type="sidestep_left",
            forward_displacement=1.0 if is_rotated else 0.0,
            lateral_displacement=0.0 if is_rotated else 1.0,
            confidence=1.0,
            reject_reason=None,
            yaw_delta=0.0,
        )

    monkeypatch.setattr(
        heading,
        "_estimate_motion_heading_from_horizontal_accel",
        lateral_result,
    )

    mode = heading._estimate_device_orientation_mode(
        df_acc,
        df_gyro,
        np.asarray([10, 40, 70]),
        initial_direction=0.0,
    )

    assert mode == "normal"


def test_forward_step_length_uses_sensor_timestamps() -> None:
    samples = 41
    acceleration = np.sin(np.linspace(0.0, 2.0 * np.pi, samples))

    def frames(sample_period: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        times = np.arange(samples, dtype=float) * sample_period
        return (
            pd.DataFrame(
                {
                    "t": times,
                    "h_y": acceleration,
                    "h_z": np.zeros(samples),
                }
            ),
            pd.DataFrame({"t": times, "low_angle": np.zeros(samples)}),
        )

    normal_acc, normal_gyro = frames(0.01)
    slow_acc, slow_gyro = frames(0.02)
    peaks = np.asarray([0, 40])
    normal = step_length.estimate_step_length_forward(
        normal_acc,
        normal_gyro,
        peaks,
        0,
        0.0,
    )
    slow = step_length.estimate_step_length_forward(
        slow_acc,
        slow_gyro,
        peaks,
        0,
        0.0,
    )

    assert normal > 0.0
    np.testing.assert_allclose(slow / normal, 4.0, rtol=1e-12)
