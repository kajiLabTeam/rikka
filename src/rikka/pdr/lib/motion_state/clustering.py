"""PDR の横歩きクラスタ判定と平滑化。

役割:
    横歩きクラスタ判定と平滑化を独立した部品として実装する。
依存元:
    common の設定・共有型・角度処理と同じ motion_state 領域の部品を利用する。
利用先:
    pdr pipeline と motion_state の後続処理から使用される。
処理フロー:
    隣接する evidence をクラスタ化し、確定した移動種別を各歩へ反映する。
"""

from ....common.config import (
    SIDESTEP_SUSPECT_MODE,
)
from ....common.lib.models import (
    StepHeading,
)
from ....common.lib.pdr_math import (
    SIDESTEP_BODY_MOTION_RATIO_THRESHOLD,
    _validate_sidestep_smoothing,
    _validate_sidestep_suspect_mode,
)
from .evidence import (
    _circular_mean_angles,
    _mean_finite,
    _sidestep_evidence,
    _SidestepEvidence,
)
from .step_motion import (
    _is_sidestep_movement,
    _movement_type_when_sidestep_is_suppressed,
    _sidestep_direction,
    _sidestep_direction_label,
)


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


smooth_step_headings = _smooth_step_headings
