"""方向曖昧性解決の回帰テスト。"""

from __future__ import annotations

import numpy as np

from rikka.common.lib.models import StepHeading, StepMotionObservation
from rikka.pdr.lib.fusion.robust import resolve_step_directions


def _heading(
    index: int,
    *,
    selected: float = 0.0,
    motion: float = 0.0,
    body: float = 0.0,
    yaw: float = 0.0,
    confidence: float = 0.8,
    movement: str = "forward",
    lateral: float = 0.0,
) -> StepHeading:
    return StepHeading(
        step_index=index + 1,
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
        forward_displacement=0.02 if lateral == 0.0 else 0.0,
        lateral_displacement=lateral,
        motion_confidence=confidence,
        motion_reject_reason=None,
        trajectory_movement_type=movement,
        yaw_delta=yaw,
    )


def _observation(
    heading: StepHeading,
    *,
    axis: float | None = None,
    calibration: float = 1.0,
) -> StepMotionObservation:
    return StepMotionObservation(
        step_index=heading.step_index,
        timestamp_s=heading.timestamp_s,
        device_yaw_heading=heading.gyro_heading,
        body_heading_candidate=heading.body_heading,
        directed_motion_heading=heading.motion_heading,
        motion_axis_heading=heading.motion_heading if axis is None else axis,
        forward_displacement=heading.forward_displacement,
        lateral_displacement=heading.lateral_displacement,
        displacement_norm=float(
            np.hypot(
                heading.forward_displacement or 0.0,
                heading.lateral_displacement or 0.0,
            )
        ),
        yaw_delta=heading.yaw_delta,
        motion_confidence=heading.motion_confidence,
        calibration_reliability=calibration,
        raw_movement_type=heading.movement_type,
        trajectory_movement_type=(
            heading.trajectory_movement_type or heading.movement_type
        ),
        device_orientation_mode="normal",
    )


def _angle_delta(left: float, right: float) -> float:
    return float((left - right + np.pi) % (2.0 * np.pi) - np.pi)


def test_wrap_at_pi_is_not_resolved_as_reversal() -> None:
    headings = [
        _heading(
            0,
            selected=np.deg2rad(179.0),
            motion=np.deg2rad(179.0),
            body=np.pi,
        ),
        _heading(
            1,
            selected=np.deg2rad(-179.0),
            motion=np.deg2rad(-179.0),
            body=np.pi,
        ),
    ]
    observations = [
        _observation(headings[0], axis=np.deg2rad(179.0)),
        _observation(headings[1], axis=np.deg2rad(-179.0)),
    ]

    resolved, posteriors = resolve_step_directions(
        headings, observations, smoothing_mode="offline"
    )

    assert (
        abs(_angle_delta(resolved[1].selected_heading or 0.0, np.deg2rad(-179))) < 0.1
    )
    assert not posteriors[1].flip_supported


def test_unsupported_180_degree_flip_is_rejected() -> None:
    headings = [
        _heading(index, motion=0.0 if index < 3 else np.pi, confidence=1.0)
        for index in range(4)
    ]
    observations = [_observation(heading, axis=0.0) for heading in headings]

    resolved, posteriors = resolve_step_directions(
        headings, observations, smoothing_mode="offline"
    )

    assert abs(_angle_delta(resolved[-1].selected_heading or 0.0, 0.0)) < 0.1
    assert not posteriors[-1].flip_supported


def test_three_step_yaw_support_allows_real_u_turn() -> None:
    headings: list[StepHeading] = []
    for index in range(6):
        after_turn = index >= 3
        headings.append(
            _heading(
                index,
                motion=np.pi if after_turn else 0.0,
                body=np.pi if after_turn else 0.0,
                yaw=np.deg2rad(20.0) if 1 <= index <= 3 else 0.0,
                confidence=1.0,
            )
        )
    observations = [_observation(heading, axis=0.0) for heading in headings]

    resolved, posteriors = resolve_step_directions(
        headings, observations, smoothing_mode="offline"
    )

    assert (
        abs(abs(_angle_delta(resolved[-1].selected_heading or 0.0, 0.0)) - np.pi) < 0.1
    )
    assert posteriors[3].flip_supported


def test_low_calibration_keeps_both_direction_hypotheses() -> None:
    headings = [_heading(index, confidence=0.2) for index in range(4)]
    observations = [
        _observation(heading, axis=0.0, calibration=0.0) for heading in headings
    ]

    _resolved, posteriors = resolve_step_directions(
        headings, observations, smoothing_mode="offline"
    )

    assert all(
        0.35 <= posterior.positive_axis_probability <= 0.65
        and 0.35 <= posterior.negative_axis_probability <= 0.65
        for posterior in posteriors
    )


def test_lateral_segments_resolve_left_and_right_headings() -> None:
    left_headings = [
        _heading(
            index,
            selected=np.pi / 2.0,
            motion=np.pi / 2.0,
            movement="sidestep_left",
            lateral=0.12,
            confidence=1.0,
        )
        for index in range(3)
    ]
    right_headings = [
        _heading(
            index,
            selected=-np.pi / 2.0,
            motion=-np.pi / 2.0,
            movement="sidestep_right",
            lateral=-0.12,
            confidence=1.0,
        )
        for index in range(3)
    ]

    resolved_left, posterior_left = resolve_step_directions(
        left_headings,
        [_observation(heading, axis=np.pi / 2.0) for heading in left_headings],
        smoothing_mode="offline",
    )
    resolved_right, posterior_right = resolve_step_directions(
        right_headings,
        [_observation(heading, axis=-np.pi / 2.0) for heading in right_headings],
        smoothing_mode="offline",
    )

    assert all(item.selected_motion_mode == "sidestep_left" for item in posterior_left)
    assert all(
        item.selected_motion_mode == "sidestep_right" for item in posterior_right
    )
    assert (
        abs(_angle_delta(resolved_left[-1].selected_heading or 0.0, np.pi / 2.0)) < 0.1
    )
    assert (
        abs(_angle_delta(resolved_right[-1].selected_heading or 0.0, -np.pi / 2.0))
        < 0.1
    )
