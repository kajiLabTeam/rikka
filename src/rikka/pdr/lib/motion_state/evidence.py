"""PDR の横歩き観測と evidence 生成。

役割:
    横歩き観測と evidence 生成を独立した部品として実装する。
依存元:
    common の設定・共有型・角度処理と同じ motion_state 領域の部品を利用する。
利用先:
    pdr pipeline と motion_state の後続処理から使用される。
処理フロー:
    方位候補から観測量、尤度用 evidence、PF 用方位を順に生成する。
"""

from typing import NamedTuple

import numpy as np

from ....common.config import (
    PF_MOTION_CALIBRATION_MIN_STEPS,
    PF_MOTION_DISPLACEMENT_FULL_CONFIDENCE_M,
    TURNING_YAW_DELTA_THRESHOLD_DEG,
)
from ....common.lib.models import (
    StepHeading,
    StepMotionEvidence,
    StepMotionObservation,
)
from ....common.lib.pdr_math import (
    SIDESTEP_BODY_MOTION_ANGLE_THRESHOLD_RAD,
    SIDESTEP_BODY_MOTION_RATIO_THRESHOLD,
    SIDESTEP_STRONG_ANGLE_THRESHOLD_RAD,
    _normalize_angle,
)
from .step_motion import (
    _is_trajectory_sidestep_movement,
    _lateral_forward_ratio,
    _sidestep_direction,
)


def _motion_calibration_reliability(step_headings: list[StepHeading]) -> float:
    """初期の前進候補から端末方位と移動方位の対応安定度を返す。"""
    residuals: list[float] = []
    for heading in step_headings:
        if len(residuals) >= 8:
            break
        if heading.body_heading is None or heading.motion_heading is None:
            continue
        if heading.movement_type not in {"forward", "unknown"}:
            continue
        if heading.yaw_delta is not None and abs(heading.yaw_delta) >= np.deg2rad(20.0):
            continue
        residuals.append(
            _normalize_angle(heading.motion_heading - heading.body_heading)
        )
    if not residuals:
        return 0.0
    concentration = float(
        np.hypot(np.mean(np.sin(residuals)), np.mean(np.cos(residuals)))
    )
    coverage = min(1.0, len(residuals) / PF_MOTION_CALIBRATION_MIN_STEPS)
    return float(np.clip(concentration * coverage, 0.0, 1.0))


def _local_motion_consistency(step_headings: list[StepHeading], index: int) -> float:
    """近傍3歩の移動方位が同じ方向を示す度合いを返す。"""
    angles = [
        heading.motion_heading
        for heading in step_headings[max(0, index - 1) : index + 2]
        if heading.motion_heading is not None
    ]
    if len(angles) <= 1:
        return 0.5
    return float(np.hypot(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def _normalize_motion_axis_heading(angle: float) -> float:
    """方向の正負を決めずに移動軸を [-pi/2, pi/2) へ正規化する。"""
    return float((angle + np.pi / 2.0) % np.pi - np.pi / 2.0)


def build_step_motion_observations(
    step_headings: list[StepHeading],
) -> tuple[StepMotionObservation, ...]:
    """既存の採用方位を変更せず、方位候補と移動軸の観測列を作る。"""
    calibration = _motion_calibration_reliability(step_headings)
    observations: list[StepMotionObservation] = []
    for heading in step_headings:
        motion_axis_heading = (
            None
            if heading.motion_heading is None
            else _normalize_motion_axis_heading(heading.motion_heading)
        )
        displacement_norm = float(
            np.hypot(
                heading.forward_displacement or 0.0,
                heading.lateral_displacement or 0.0,
            )
        )
        observations.append(
            StepMotionObservation(
                step_index=heading.step_index,
                timestamp_s=heading.timestamp_s,
                device_yaw_heading=heading.gyro_heading,
                body_heading_candidate=heading.body_heading,
                directed_motion_heading=heading.motion_heading,
                motion_axis_heading=motion_axis_heading,
                forward_displacement=heading.forward_displacement,
                lateral_displacement=heading.lateral_displacement,
                displacement_norm=displacement_norm,
                yaw_delta=heading.yaw_delta,
                motion_confidence=heading.motion_confidence,
                calibration_reliability=calibration,
                raw_movement_type=heading.movement_type,
                trajectory_movement_type=(
                    heading.trajectory_movement_type or heading.movement_type
                ),
                device_orientation_mode=heading.device_orientation_mode,
            )
        )
    return tuple(observations)


def build_step_motion_evidences(
    step_headings: list[StepHeading],
) -> tuple[StepMotionEvidence, ...]:
    """確定分類を上書きせず、PF向けの運動状態尤度列を生成する。"""
    calibration = _motion_calibration_reliability(step_headings)
    evidences: list[StepMotionEvidence] = []
    turning_threshold = np.deg2rad(TURNING_YAW_DELTA_THRESHOLD_DEG)
    for index, heading in enumerate(step_headings):
        displacement = float(
            np.hypot(
                heading.forward_displacement or 0.0,
                heading.lateral_displacement or 0.0,
            )
        )
        displacement_quality = float(
            np.clip(
                displacement / PF_MOTION_DISPLACEMENT_FULL_CONFIDENCE_M,
                0.0,
                1.0,
            )
        )
        consistency = _local_motion_consistency(step_headings, index)
        reliability = float(
            np.clip(displacement_quality * consistency * calibration, 0.0, 1.0)
        )
        movement = heading.trajectory_movement_type or heading.movement_type
        raw_turning = heading.movement_type == "turning" or (
            heading.movement_type.startswith("turning_sidestep_")
        )
        is_suspect = movement in {
            "sidestep_suspect_left",
            "sidestep_suspect_right",
        }
        left = movement in {
            "sidestep_left",
            "turning_sidestep_left",
            "sidestep_suspect_left",
        }
        right = movement in {
            "sidestep_right",
            "turning_sidestep_right",
            "sidestep_suspect_right",
        }
        yaw_strength = min(
            1.0,
            abs(heading.yaw_delta or 0.0) / max(turning_threshold, 1e-12),
        )
        if movement == "turning":
            forward_likelihood = 0.02
            left_likelihood = 0.01
            right_likelihood = 0.01
            turning = 0.95
        elif left or right:
            if is_suspect:
                side_strength = 0.15 + 0.45 * reliability
                forward_likelihood = 0.6
            elif heading.sidestep_cluster_id is not None and calibration >= 0.5:
                # 複数歩で確定したclusterは、forwardから横歩きへ遷移できる
                # 観測比を与える。校正不良時は従来の連続値へ戻す。
                side_strength = 0.98
                forward_likelihood = 0.01
            else:
                side_strength = float(np.clip((reliability - 0.45) / 0.35, 0.01, 0.99))
                forward_likelihood = max(0.01, 1.0 - side_strength)
            left_likelihood = 0.01 + (side_strength if left else 0.0)
            right_likelihood = 0.01 + (side_strength if right else 0.0)
            turning = (
                0.35 + 0.55 * yaw_strength
                if raw_turning
                else 0.05 + 0.15 * yaw_strength
                if is_suspect
                else 0.01
            )
        else:
            forward_likelihood = 0.15 if raw_turning else 0.95
            left_likelihood = 0.01
            right_likelihood = 0.01
            turning = 0.8 if raw_turning else 0.01
        values = np.asarray(
            [forward_likelihood, left_likelihood, right_likelihood, turning],
            dtype=float,
        )
        values /= values.sum()
        evidences.append(
            StepMotionEvidence(
                forward_likelihood=float(values[0]),
                sidestep_left_likelihood=float(values[1]),
                sidestep_right_likelihood=float(values[2]),
                turning_likelihood=float(values[3]),
                motion_reliability=reliability,
                calibration_reliability=calibration,
            )
        )
    return tuple(evidences)


def build_particle_motion_headings(
    step_headings: list[StepHeading],
) -> tuple[float | None, ...]:
    """通常PDRを変えず、PF向けに横歩き境界を連続化した方位列を返す。"""
    result: list[float | None] = []
    previous_heading: float | None = None
    previous_type: str | None = None
    after_sidestep_steps = 0
    for heading in step_headings:
        movement = heading.trajectory_movement_type or heading.movement_type
        candidate = heading.selected_heading
        if candidate is None:
            result.append(None)
            continue
        yaw_deg = abs(float(np.degrees(heading.yaw_delta or 0.0)))
        raw_sidestep_evidence = (
            "sidestep" in heading.movement_type
            or heading.sidestep_evidence_reason is not None
        )
        use_motion = (
            heading.motion_heading is not None
            and yaw_deg < TURNING_YAW_DELTA_THRESHOLD_DEG
            and (
                raw_sidestep_evidence
                or movement == "forward"
                and after_sidestep_steps > 0
            )
        )
        if use_motion:
            assert heading.motion_heading is not None
            candidate = float(heading.motion_heading)
        genuine_turn = (
            movement == "turning"
            or previous_type == "turning"
            or yaw_deg >= TURNING_YAW_DELTA_THRESHOLD_DEG
        )
        if previous_heading is not None and not genuine_turn:
            limit = np.deg2rad(
                45.0
                if _is_trajectory_sidestep_movement(movement) or raw_sidestep_evidence
                else 25.0
            )
            delta = np.clip(
                _normalize_angle(candidate - previous_heading),
                -limit,
                limit,
            )
            candidate = _normalize_angle(previous_heading + float(delta))
        result.append(candidate)
        if _is_trajectory_sidestep_movement(movement):
            after_sidestep_steps = 2
        elif after_sidestep_steps > 0:
            after_sidestep_steps -= 1
        previous_heading = candidate
        previous_type = movement
    return tuple(result)


class _SidestepEvidence(NamedTuple):
    """横歩き候補の診断情報。"""

    direction: int | None
    reason: str | None
    angle_diff: float | None
    strong: bool


def _mean_finite(values: list[float]) -> float | None:
    """有限値だけの平均を返す。有限値がなければ None。"""
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return None
    return float(np.mean(finite_values))


def _circular_mean_angles(angles: list[float]) -> float | None:
    """角度リストの円平均を返す。有限な角度がなければ None。"""
    finite_angles = [angle for angle in angles if np.isfinite(angle)]
    if not finite_angles:
        return None
    sin_sum = float(np.sum(np.sin(finite_angles)))
    cos_sum = float(np.sum(np.cos(finite_angles)))
    if np.hypot(sin_sum, cos_sum) <= 1e-12:
        return None
    return _normalize_angle(float(np.arctan2(sin_sum, cos_sum)))


def _body_motion_angle_diff(step_heading: StepHeading) -> float | None:
    """端末方位と移動方位の絶対角度差を返す。"""
    if step_heading.body_heading is None or step_heading.motion_heading is None:
        return None
    return abs(
        _normalize_angle(step_heading.motion_heading - step_heading.body_heading)
    )


def _sidestep_evidence(step_heading: StepHeading) -> _SidestepEvidence:
    """1歩が横歩き候補かどうかを判定する。"""
    angle_diff = _body_motion_angle_diff(step_heading)
    direction = _sidestep_direction(step_heading.movement_type)
    if direction is not None:
        ratio = _lateral_forward_ratio(step_heading)
        strong = (
            angle_diff is not None
            and ratio is not None
            and step_heading.lateral_displacement is not None
            and angle_diff >= SIDESTEP_STRONG_ANGLE_THRESHOLD_RAD
            and ratio >= SIDESTEP_BODY_MOTION_RATIO_THRESHOLD
            and abs(step_heading.lateral_displacement)
            >= step_heading.sidestep_min_lateral_displacement
        )
        return _SidestepEvidence(direction, "movement_type", angle_diff, strong)

    if step_heading.movement_type == "turning":
        return _SidestepEvidence(None, None, angle_diff, False)
    if step_heading.lateral_displacement is None:
        return _SidestepEvidence(None, None, angle_diff, False)

    ratio = _lateral_forward_ratio(step_heading)
    lateral_displacement = step_heading.lateral_displacement
    if lateral_displacement is None:
        return _SidestepEvidence(None, None, angle_diff, False)
    lateral_abs = abs(lateral_displacement)
    if (
        angle_diff is None
        or ratio is None
        or lateral_abs < step_heading.sidestep_min_lateral_displacement
        or angle_diff < SIDESTEP_BODY_MOTION_ANGLE_THRESHOLD_RAD
        or ratio < SIDESTEP_BODY_MOTION_RATIO_THRESHOLD
    ):
        return _SidestepEvidence(None, None, angle_diff, False)

    direction = 1 if lateral_displacement > 0 else -1
    strong = (
        angle_diff >= SIDESTEP_STRONG_ANGLE_THRESHOLD_RAD
        and ratio >= SIDESTEP_BODY_MOTION_RATIO_THRESHOLD
        and lateral_abs >= step_heading.sidestep_min_lateral_displacement
    )
    return _SidestepEvidence(direction, "body_motion_lateral", angle_diff, strong)
