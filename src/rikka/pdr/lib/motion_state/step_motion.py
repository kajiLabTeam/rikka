"""PDR の横歩き判定と軌跡方位安定化。

役割:
    方位候補と体軸相対変位から前進、横歩き、横歩き疑い、旋回、後退を整理し、
    軌跡へ採用する方位と歩幅倍率を確定する。
依存元:
    ``config`` の移動別倍率、``common`` の判定閾値・角度処理、``models`` の
    ``StepHeading`` / ``StepMotion`` を利用する。
利用先:
    ``trajectory`` が通常 PDR に使用し、``particle_api`` 経由で
    ``particle_filter`` も同じ平滑化・移動量決定ロジックを使用する。
処理フロー:
    各歩の横方向 evidence を作り、隣接歩をクラスタ平滑化し、急激な方位変化を
    移動種別ごとの上限で抑えた後、最終的な ``StepMotion`` を返す。
"""

from typing import NamedTuple

import numpy as np

from ....analyze.pdr.common import (
    INITIAL_FORWARD_MOTION_BODY_CONSTRAINT_RAD,
    SIDESTEP_BODY_MOTION_ANGLE_THRESHOLD_RAD,
    SIDESTEP_BODY_MOTION_RATIO_THRESHOLD,
    SIDESTEP_HEADING_MAX_STEP_DELTA_RAD,
    SIDESTEP_MOTION_LATERAL_CONSTRAINT_RAD,
    SIDESTEP_STRONG_ANGLE_THRESHOLD_RAD,
    TRAJECTORY_HEADING_MAX_STEP_DELTA_RAD,
    TURNING_SIDESTEP_HEADING_MAX_STEP_DELTA_RAD,
    _normalize_angle,
    _validate_forward_heading_source,
    _validate_sidestep_heading_source,
    _validate_sidestep_smoothing,
    _validate_sidestep_suspect_mode,
)
from ....common.config import (
    BACKWARD_LENGTH_SCALE,
    FORWARD_HEADING_SOURCE,
    PF_MOTION_CALIBRATION_MIN_STEPS,
    PF_MOTION_DISPLACEMENT_FULL_CONFIDENCE_M,
    SIDESTEP_LENGTH_SCALE,
    SIDESTEP_SUSPECT_MODE,
    TURNING_LENGTH_SCALE,
    TURNING_YAW_DELTA_THRESHOLD_DEG,
)
from ....common.lib.models import (
    StepHeading,
    StepMotion,
    StepMotionEvidence,
    StepMotionObservation,
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


def _is_sidestep_movement(movement_type: str) -> bool:
    """横歩き系の移動タイプかどうかを返す。"""
    return movement_type in {
        "sidestep_left",
        "sidestep_right",
        "turning_sidestep_left",
        "turning_sidestep_right",
    }


def _is_trajectory_sidestep_movement(movement_type: str | None) -> bool:
    """軌跡上で横歩きとして扱う移動タイプかどうかを返す。"""
    return movement_type in {
        "sidestep_left",
        "sidestep_right",
        "turning_sidestep_left",
        "turning_sidestep_right",
        "sidestep_suspect_left",
        "sidestep_suspect_right",
    }


def _is_sidestep_suspect_movement(movement_type: str | None) -> bool:
    """横歩き疑いの移動タイプかどうかを返す。"""
    return movement_type in {"sidestep_suspect_left", "sidestep_suspect_right"}


def _movement_type_when_sidestep_is_suppressed(step_heading: StepHeading) -> str:
    """横歩きを抑制しても、生の旋回情報は軌跡用分類に残す。"""
    if step_heading.movement_type.startswith("turning_sidestep_"):
        return step_heading.movement_type
    return "forward"


def _sidestep_body_lateral_heading(
    body_heading: float | None,
    movement_type: str,
) -> float | None:
    """端末方位から見た左右横歩き方向を返す。"""
    if body_heading is None:
        return None
    if movement_type in {
        "sidestep_left",
        "turning_sidestep_left",
        "sidestep_suspect_left",
    }:
        return float(body_heading + np.pi / 2)
    if movement_type in {
        "sidestep_right",
        "turning_sidestep_right",
        "sidestep_suspect_right",
    }:
        return float(body_heading - np.pi / 2)
    return None


def _sidestep_motion_heading(step_heading: StepHeading) -> float | None:
    """横歩き用の実移動方位候補を返す。"""
    if (
        step_heading.source.startswith("trajectory_")
        and step_heading.selected_heading is not None
    ):
        return step_heading.selected_heading
    if step_heading.motion_heading is not None:
        return step_heading.motion_heading
    return step_heading.selected_heading


def _resolve_sidestep_heading(
    step_heading: StepHeading,
    movement_type: str,
    body_heading: float | None,
    heading_source: str,
) -> float | None:
    """指定ソースに応じた横歩き軌跡方位を返す。"""
    selected_heading_source = _validate_sidestep_heading_source(heading_source)
    motion_heading = _sidestep_motion_heading(step_heading)
    body_lateral_heading = _sidestep_body_lateral_heading(body_heading, movement_type)

    if selected_heading_source == "motion":
        return motion_heading if motion_heading is not None else body_lateral_heading
    if selected_heading_source == "body_lateral":
        return (
            body_lateral_heading if body_lateral_heading is not None else motion_heading
        )

    blend_candidates = [
        heading
        for heading in (motion_heading, body_lateral_heading)
        if heading is not None
    ]
    return _circular_mean_angles(blend_candidates)


def _lateral_forward_ratio(step_heading: StepHeading) -> float | None:
    """横方向変位 / 前方向変位 の比を返す。"""
    if (
        step_heading.forward_displacement is None
        or step_heading.lateral_displacement is None
    ):
        return None
    return abs(step_heading.lateral_displacement) / max(
        abs(step_heading.forward_displacement),
        1e-12,
    )


def _sidestep_direction(movement_type: str) -> int | None:
    """横歩き方向を符号で返す。left=+1, right=-1。"""
    if movement_type in {"sidestep_left", "turning_sidestep_left"}:
        return 1
    if movement_type in {"sidestep_right", "turning_sidestep_right"}:
        return -1
    return None


def _sidestep_direction_label(direction: int | None) -> str | None:
    """横歩き方向の符号をCSV向けラベルに変換する。"""
    if direction == 1:
        return "left"
    if direction == -1:
        return "right"
    return None


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


def _is_sidestep_bridge_gap(step_heading: StepHeading, direction: int) -> bool:
    """同方向横歩きclusterに1歩だけ含められる弱い隙間かどうか。"""
    if step_heading.movement_type not in {"forward", "unknown"}:
        return False
    if step_heading.lateral_displacement is None:
        return False
    if (
        abs(step_heading.lateral_displacement)
        < step_heading.sidestep_min_lateral_displacement
    ):
        return False
    return step_heading.lateral_displacement * direction > 0


def _sidestep_cluster_has_lateral_strength(
    step_headings: list[StepHeading],
    indexes: list[int],
    direction: int,
) -> bool:
    """cluster 全体として横方向特徴が十分かどうかを返す。"""
    lateral_values: list[float] = []
    forward_values: list[float] = []
    for index in indexes:
        lateral_displacement = step_headings[index].lateral_displacement
        if lateral_displacement is not None:
            lateral_values.append(lateral_displacement)
        forward_displacement = step_headings[index].forward_displacement
        if forward_displacement is not None:
            forward_values.append(abs(forward_displacement))
    lateral_smoothed = _mean_finite(lateral_values)
    forward_smoothed = _mean_finite(forward_values)
    if lateral_smoothed is None or forward_smoothed is None:
        return False
    if lateral_smoothed * direction <= 0:
        return False
    current = step_headings[indexes[0]]
    lateral_abs = abs(lateral_smoothed)
    return (
        lateral_abs >= current.sidestep_min_lateral_displacement
        and lateral_abs
        >= SIDESTEP_BODY_MOTION_RATIO_THRESHOLD * max(forward_smoothed, 1e-12)
    )


def _has_adjacent_opposite_evidence(
    evidences: list[_SidestepEvidence],
    index: int,
    direction: int,
) -> bool:
    """単発強 evidence の隣に逆方向 evidence があるかを返す。"""
    return (
        index > 0
        and evidences[index - 1].direction == -direction
        or index + 1 < len(evidences)
        and evidences[index + 1].direction == -direction
    )


def _smoothed_step_displacements(
    step_headings: list[StepHeading],
    start: int,
    end: int,
) -> tuple[float | None, float | None]:
    """指定ステップ範囲の横方向・前方向変位特徴を平滑化して返す。"""
    lateral_values = [
        heading.lateral_displacement
        for heading in step_headings[start:end]
        if heading.lateral_displacement is not None
    ]
    forward_values = [
        abs(heading.forward_displacement)
        for heading in step_headings[start:end]
        if heading.forward_displacement is not None
    ]
    return _mean_finite(lateral_values), _mean_finite(forward_values)


def _sidestep_cluster_bounds(
    step_headings: list[StepHeading],
    index: int,
    direction: int,
) -> tuple[int, int]:
    """同方向の横歩き候補が連続する [start, end) を返す。"""
    start = index
    while (
        start > 0
        and _sidestep_direction(step_headings[start - 1].movement_type) == direction
    ):
        start -= 1

    end = index + 1
    while (
        end < len(step_headings)
        and _sidestep_direction(step_headings[end].movement_type) == direction
    ):
        end += 1

    return start, end


def _sidestep_cluster_motion_heading(
    step_headings: list[StepHeading],
    start: int,
    end: int,
) -> float | None:
    """横歩き候補ラン内の motion_heading を円平均して返す。"""
    motion_headings = [
        heading.motion_heading
        for heading in step_headings[start:end]
        if heading.motion_heading is not None
    ]
    return _circular_mean_angles(motion_headings)


def _sidestep_cluster_motion_heading_for_indexes(
    step_headings: list[StepHeading],
    indexes: list[int],
) -> float | None:
    """指定した横歩き evidence 歩の motion_heading を円平均して返す。"""
    motion_headings: list[float] = []
    for index in indexes:
        motion_heading = step_headings[index].motion_heading
        if motion_heading is not None:
            motion_headings.append(motion_heading)
    return _circular_mean_angles(motion_headings)


def _is_confirmed_sidestep_cluster(
    step_headings: list[StepHeading],
    index: int,
) -> bool:
    """平滑化したステップ列特徴から、軌跡へ反映する横歩きか判定する。"""
    current = step_headings[index]
    direction = _sidestep_direction(current.movement_type)
    if direction is None:
        return False

    start, end = _sidestep_cluster_bounds(step_headings, index, direction)
    if end - start < 2:
        return False

    lateral_smoothed, forward_smoothed = _smoothed_step_displacements(
        step_headings,
        start,
        end,
    )
    if lateral_smoothed is None or forward_smoothed is None:
        return False
    if lateral_smoothed * direction <= 0:
        return False

    lateral_abs = abs(lateral_smoothed)
    return (
        lateral_abs >= current.sidestep_min_lateral_displacement
        and lateral_abs >= current.sidestep_lateral_ratio * max(forward_smoothed, 1e-12)
    )


def _smooth_step_headings(
    step_headings: list[StepHeading],
    method: str = "none",
    sidestep_suspect_mode: str = SIDESTEP_SUSPECT_MODE,
) -> list[StepHeading]:
    """横歩き判定の軌跡反映を平滑化する。"""
    selected_method = _validate_sidestep_smoothing(method)
    selected_sidestep_suspect_mode = _validate_sidestep_suspect_mode(
        sidestep_suspect_mode
    )
    if selected_method == "none":
        return step_headings

    smoothed = list(step_headings)
    if selected_method == "isolated":
        if len(step_headings) < 3:
            return step_headings
        for i in range(1, len(step_headings) - 1):
            current = step_headings[i]
            prev_type = step_headings[i - 1].movement_type
            next_type = step_headings[i + 1].movement_type
            if (
                _is_sidestep_movement(current.movement_type)
                and prev_type == "forward"
                and next_type == "forward"
            ):
                smoothed[i] = current._replace(
                    trajectory_movement_type=(
                        _movement_type_when_sidestep_is_suppressed(current)
                    )
                )
        return smoothed

    # 各歩の横歩きらしさを evidence 化し、同方向の連続区間だけを軌跡へ強く反映する。
    evidences = [_sidestep_evidence(heading) for heading in step_headings]
    cluster_id = 0
    i = 0
    while i < len(step_headings):
        evidence = evidences[i]
        direction = evidence.direction
        if direction is None:
            smoothed[i] = step_headings[i]._replace(
                body_motion_angle_diff=evidence.angle_diff,
                sidestep_evidence_direction=None,
                sidestep_evidence_reason=None,
                sidestep_cluster_id=None,
            )
            i += 1
            continue

        members = [i]
        evidence_indexes = [i]
        used_bridge = False
        j = i + 1
        while j < len(step_headings):
            next_evidence = evidences[j]
            if next_evidence.direction == direction:
                members.append(j)
                evidence_indexes.append(j)
                used_bridge = False
                j += 1
                continue
            if (
                next_evidence.direction is None
                and not used_bridge
                and j + 1 < len(step_headings)
                and evidences[j + 1].direction == direction
                and _is_sidestep_bridge_gap(step_headings[j], direction)
            ):
                members.append(j)
                used_bridge = True
                j += 1
                continue
            break

        # 2歩以上かつ横方向変位が十分な cluster だけを確定横歩きとして扱う。
        confirmed = len(evidence_indexes) >= 2
        confirmed = confirmed and _sidestep_cluster_has_lateral_strength(
            step_headings,
            evidence_indexes,
            direction,
        )

        if confirmed:
            cluster_id += 1
            sidestep_type = "sidestep_left" if direction == 1 else "sidestep_right"
            turning_sidestep_type = (
                "turning_sidestep_left" if direction == 1 else "turning_sidestep_right"
            )
            for member_index in members:
                member_evidence = evidences[member_index]
                movement_type = (
                    turning_sidestep_type
                    if step_headings[member_index].movement_type.startswith(
                        "turning_sidestep_"
                    )
                    else sidestep_type
                )
                smoothed[member_index] = step_headings[member_index]._replace(
                    trajectory_movement_type=movement_type,
                    body_motion_angle_diff=member_evidence.angle_diff,
                    sidestep_evidence_direction=_sidestep_direction_label(direction),
                    sidestep_evidence_reason=member_evidence.reason
                    if member_evidence.reason is not None
                    else "bridge_gap",
                    sidestep_cluster_id=cluster_id,
                )
        else:
            for evidence_index in evidence_indexes:
                member_evidence = evidences[evidence_index]
                suspect = (
                    member_evidence.strong
                    and not _has_adjacent_opposite_evidence(
                        evidences,
                        evidence_index,
                        direction,
                    )
                )
                suspect_type = (
                    "sidestep_suspect_left"
                    if direction == 1
                    else "sidestep_suspect_right"
                )
                suspect = suspect and selected_sidestep_suspect_mode != "forward"
                suspect_heading = (
                    step_headings[evidence_index].motion_heading
                    if selected_sidestep_suspect_mode in {"motion", "blend"}
                    else None
                )
                smoothed[evidence_index] = step_headings[evidence_index]._replace(
                    selected_heading=suspect_heading
                    if suspect
                    else step_headings[evidence_index].selected_heading,
                    trajectory_movement_type=(
                        suspect_type
                        if suspect
                        else _movement_type_when_sidestep_is_suppressed(
                            step_headings[evidence_index]
                        )
                    ),
                    body_motion_angle_diff=member_evidence.angle_diff,
                    sidestep_evidence_direction=_sidestep_direction_label(direction),
                    sidestep_evidence_reason=member_evidence.reason,
                    sidestep_cluster_id=None,
                )

        i = j

    return smoothed


def _trajectory_movement_type(step_heading: StepHeading) -> str:
    """軌跡計算に使う移動タイプを返す。"""
    return (
        step_heading.trajectory_movement_type
        if step_heading.trajectory_movement_type is not None
        else step_heading.movement_type
    )


def _limit_heading_change(
    heading: float | None,
    previous_heading: float | None,
    max_delta: float = TRAJECTORY_HEADING_MAX_STEP_DELTA_RAD,
) -> float | None:
    """前回方位からの変化量を制限した方位を返す。"""
    if heading is None or previous_heading is None:
        return heading
    delta = _normalize_angle(heading - previous_heading)
    if abs(delta) <= max_delta:
        return heading
    return _normalize_angle(previous_heading + float(np.sign(delta)) * max_delta)


def _heading_change_limit_for_movement_type(movement_type: str) -> float:
    """移動状態ごとの1歩あたり方位変化上限を返す。"""
    if movement_type.startswith("turning_sidestep_"):
        return float(TURNING_SIDESTEP_HEADING_MAX_STEP_DELTA_RAD)
    if movement_type in {"sidestep_left", "sidestep_right"}:
        return float(SIDESTEP_HEADING_MAX_STEP_DELTA_RAD)
    return float(TRAJECTORY_HEADING_MAX_STEP_DELTA_RAD)


def _resolve_world_motion_heading(
    step_heading: StepHeading,
    movement_type: str,
    previous_heading: float | None,
    previous_movement_type: str | None,
    forward_heading_source: str,
    sidestep_heading_source: str,
) -> tuple[float | None, str | None]:
    """世界座標変換済み移動方位を主候補にして軌跡方位を返す。"""
    body_heading = (
        step_heading.body_heading
        if step_heading.body_heading is not None
        else step_heading.gyro_heading
    )
    candidate: float | None = None
    source: str | None = None

    # 移動状態ごとに、body heading / motion heading / body lateral の
    # どれを使うか決める。
    if movement_type == "forward":
        if forward_heading_source == "body":
            candidate = body_heading
            source = "trajectory_body"
        else:
            candidate = (
                step_heading.motion_heading
                if step_heading.motion_heading is not None
                else body_heading
            )
            source = (
                "trajectory_motion"
                if step_heading.motion_heading is not None
                else "trajectory_body_fallback"
            )
            if (
                previous_heading is None
                and step_heading.motion_heading is not None
                and body_heading is not None
                and abs(_normalize_angle(step_heading.motion_heading - body_heading))
                > INITIAL_FORWARD_MOTION_BODY_CONSTRAINT_RAD
            ):
                candidate = body_heading
                source = "trajectory_initial_body_fallback"
    elif movement_type in {
        "sidestep_left",
        "sidestep_right",
        "turning_sidestep_left",
        "turning_sidestep_right",
    }:
        body_lateral_heading = (
            _sidestep_body_lateral_heading(body_heading, movement_type)
            if body_heading is not None
            else None
        )
        motion_heading = step_heading.motion_heading
        if movement_type.startswith("turning_sidestep_"):
            candidate = (
                motion_heading
                if motion_heading is not None
                else previous_heading
                if previous_heading is not None
                else body_lateral_heading
            )
            source = (
                "trajectory_turning_sidestep_motion"
                if motion_heading is not None
                else "trajectory_turning_sidestep_fallback"
            )
        elif sidestep_heading_source == "body_lateral":
            candidate = (
                body_lateral_heading
                if body_lateral_heading is not None
                else motion_heading
            )
            source = "trajectory_body_lateral"
        elif sidestep_heading_source == "blend":
            if (
                motion_heading is not None
                and body_lateral_heading is not None
                and abs(_normalize_angle(motion_heading - body_lateral_heading))
                <= SIDESTEP_MOTION_LATERAL_CONSTRAINT_RAD
            ):
                candidate = _circular_mean_angles(
                    [motion_heading, body_lateral_heading]
                )
                source = "trajectory_blend"
            else:
                candidate = (
                    previous_heading
                    if previous_heading is not None
                    else body_lateral_heading
                )
                source = "trajectory_sidestep_fallback"
        else:
            if motion_heading is not None and (
                body_lateral_heading is None
                or abs(_normalize_angle(motion_heading - body_lateral_heading))
                <= SIDESTEP_MOTION_LATERAL_CONSTRAINT_RAD
            ):
                candidate = motion_heading
                source = "trajectory_sidestep_motion"
            else:
                candidate = (
                    previous_heading
                    if previous_heading is not None
                    else body_lateral_heading
                )
                source = "trajectory_sidestep_fallback"
    else:
        return None, None

    # 同じ移動状態が続く場合は1歩ごとの急激な方位変化を制限する。
    if movement_type == previous_movement_type:
        limited = _limit_heading_change(
            candidate,
            previous_heading,
            _heading_change_limit_for_movement_type(movement_type),
        )
        if (
            limited is not None
            and candidate is not None
            and abs(_normalize_angle(limited - candidate)) > 1e-12
        ):
            source = f"{source}_limited" if source is not None else "trajectory_limited"
        candidate = limited
    return candidate, source


def _stabilize_trajectory_headings(
    step_headings: list[StepHeading],
    forward_heading_source: str = FORWARD_HEADING_SOURCE,
    sidestep_heading_source: str = "motion",
) -> list[StepHeading]:
    """世界座標変換済み移動方位を制約付きで軌跡方位へ反映する。"""
    selected_forward_heading_source = _validate_forward_heading_source(
        forward_heading_source
    )
    selected_sidestep_heading_source = _validate_sidestep_heading_source(
        sidestep_heading_source
    )
    stabilized = list(step_headings)
    previous_heading: float | None = None
    previous_movement_type: str | None = None
    for index, step_heading in enumerate(step_headings):
        movement_type = _trajectory_movement_type(step_heading)
        selected_heading, source = _resolve_world_motion_heading(
            step_heading,
            movement_type,
            previous_heading,
            previous_movement_type,
            selected_forward_heading_source,
            selected_sidestep_heading_source,
        )
        if selected_heading is not None:
            stabilized[index] = step_heading._replace(
                selected_heading=selected_heading,
                source=source if source is not None else step_heading.source,
            )
            previous_heading = selected_heading
            previous_movement_type = movement_type
        elif movement_type != "turning":
            previous_movement_type = movement_type

    return stabilized


def _stabilize_trajectory_body_headings(
    step_headings: list[StepHeading],
    forward_heading_source: str = FORWARD_HEADING_SOURCE,
    sidestep_heading_source: str = "motion",
) -> list[StepHeading]:
    """互換用: 軌跡方位の区間安定化を返す。"""
    return _stabilize_trajectory_headings(
        step_headings,
        forward_heading_source,
        sidestep_heading_source,
    )


def estimate_step_motion(
    step_heading: StepHeading,
    step_length: float,
    previous_heading: float | None = None,
    forward_heading_source: str = FORWARD_HEADING_SOURCE,
    sidestep_heading_source: str = "motion",
    sidestep_suspect_mode: str = SIDESTEP_SUSPECT_MODE,
) -> StepMotion | None:
    """状態別に1歩の移動方位と歩幅を決める。"""
    selected_forward_heading_source = _validate_forward_heading_source(
        forward_heading_source
    )
    selected_sidestep_heading_source = _validate_sidestep_heading_source(
        sidestep_heading_source
    )
    selected_sidestep_suspect_mode = _validate_sidestep_suspect_mode(
        sidestep_suspect_mode
    )
    body_heading = (
        step_heading.body_heading
        if step_heading.body_heading is not None
        else step_heading.gyro_heading
    )
    fallback_heading = (
        step_heading.selected_heading
        if step_heading.selected_heading is not None
        else body_heading
    )
    if fallback_heading is None:
        return None

    movement_type = (
        step_heading.trajectory_movement_type
        if step_heading.trajectory_movement_type is not None
        else step_heading.movement_type
    )
    heading: float | None
    # trajectory_movement_type を優先し、横歩き・旋回・後退ごとの歩幅補正を適用する。
    if movement_type == "forward":
        if selected_forward_heading_source == "body":
            heading = body_heading if body_heading is not None else fallback_heading
        else:
            heading = (
                step_heading.selected_heading
                if step_heading.source.startswith("trajectory_")
                and step_heading.selected_heading is not None
                else step_heading.motion_heading
                if step_heading.motion_heading is not None
                else fallback_heading
            )
        scale = 1.0
    elif (
        movement_type
        in {
            "sidestep_left",
            "turning_sidestep_left",
            "sidestep_suspect_left",
        }
        and body_heading is not None
    ):
        sidestep_source = (
            "body_lateral"
            if step_heading.trajectory_movement_type is None
            else selected_sidestep_suspect_mode
            if movement_type == "sidestep_suspect_left"
            else selected_sidestep_heading_source
        )
        heading = _resolve_sidestep_heading(
            step_heading,
            movement_type,
            body_heading,
            sidestep_source,
        )
        if heading is None:
            heading = fallback_heading
        scale = SIDESTEP_LENGTH_SCALE
    elif (
        movement_type
        in {
            "sidestep_right",
            "turning_sidestep_right",
            "sidestep_suspect_right",
        }
        and body_heading is not None
    ):
        sidestep_source = (
            "body_lateral"
            if step_heading.trajectory_movement_type is None
            else selected_sidestep_suspect_mode
            if movement_type == "sidestep_suspect_right"
            else selected_sidestep_heading_source
        )
        heading = _resolve_sidestep_heading(
            step_heading,
            movement_type,
            body_heading,
            sidestep_source,
        )
        if heading is None:
            heading = fallback_heading
        scale = SIDESTEP_LENGTH_SCALE
    elif movement_type == "turning":
        heading = previous_heading if previous_heading is not None else fallback_heading
        scale = TURNING_LENGTH_SCALE
    elif movement_type == "backward" and body_heading is not None:
        heading = body_heading + np.pi
        scale = BACKWARD_LENGTH_SCALE
    elif movement_type == "unknown" and body_heading is not None:
        heading = body_heading
        scale = 1.0
    else:
        heading = fallback_heading
        scale = 1.0
        movement_type = "unknown" if movement_type == "backward" else movement_type

    resolved_heading = heading if heading is not None else fallback_heading
    return StepMotion(
        heading=_normalize_angle(float(resolved_heading)),
        length=float(step_length * scale),
        movement_type=movement_type,
        length_scale=float(scale),
    )


# 領域内の別モジュールから利用する helper は public 名で公開する。
smooth_step_headings = _smooth_step_headings
stabilize_trajectory_body_headings = _stabilize_trajectory_body_headings
stabilize_trajectory_headings = _stabilize_trajectory_headings
