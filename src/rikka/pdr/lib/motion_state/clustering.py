"""PDR の横歩きクラスタ判定と平滑化。
役割: 横歩きクラスタ判定と平滑化を実装する。依存元: common と同領域部品。
利用先: PDR preparation と motion_state の後続処理から使用される。
処理フロー: 隣接 evidence をクラスタ化し、確定移動種別を各歩へ反映する。
"""

from ....common.config import (
    SIDESTEP_SUSPECT_MODE,
)
from ....common.lib.angles import circular_mean_angles
from ....common.lib.models import (
    StepHeading,
)
from ....common.lib.pdr_math import (
    SIDESTEP_BODY_MOTION_RATIO_THRESHOLD,
)
from .evidence import (
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
    return circular_mean_angles(motion_headings)


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
    return circular_mean_angles(motion_headings)


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


def _smooth_isolated_headings(
    step_headings: list[StepHeading],
) -> list[StepHeading]:
    """前進歩に挟まれた単独横歩きを抑制する。"""
    if len(step_headings) < 3:
        return step_headings
    smoothed = list(step_headings)
    for index in range(1, len(step_headings) - 1):
        current = step_headings[index]
        if (
            _is_sidestep_movement(current.movement_type)
            and step_headings[index - 1].movement_type == "forward"
            and step_headings[index + 1].movement_type == "forward"
        ):
            smoothed[index] = current._replace(
                trajectory_movement_type=(
                    _movement_type_when_sidestep_is_suppressed(current)
                )
            )
    return smoothed


def _find_cluster_members(
    step_headings: list[StepHeading],
    evidences: list[_SidestepEvidence],
    start: int,
    direction: int,
) -> tuple[list[int], list[int], int]:
    """同方向evidenceと1歩のbridge gapからcluster候補を返す。"""
    members = [start]
    evidence_indexes = [start]
    used_bridge = False
    index = start + 1
    while index < len(step_headings):
        evidence = evidences[index]
        if evidence.direction == direction:
            members.append(index)
            evidence_indexes.append(index)
            used_bridge = False
            index += 1
            continue
        if (
            evidence.direction is None
            and not used_bridge
            and index + 1 < len(step_headings)
            and evidences[index + 1].direction == direction
            and _is_sidestep_bridge_gap(step_headings[index], direction)
        ):
            members.append(index)
            used_bridge = True
            index += 1
            continue
        break
    return members, evidence_indexes, index


def _apply_confirmed_cluster(
    smoothed: list[StepHeading],
    step_headings: list[StepHeading],
    evidences: list[_SidestepEvidence],
    members: list[int],
    direction: int,
    cluster_id: int,
) -> None:
    """確定clusterを横歩きとして軌跡へ反映する。"""
    sidestep_type = "sidestep_left" if direction == 1 else "sidestep_right"
    turning_type = (
        "turning_sidestep_left" if direction == 1 else "turning_sidestep_right"
    )
    for index in members:
        evidence = evidences[index]
        movement_type = (
            turning_type
            if step_headings[index].movement_type.startswith("turning_sidestep_")
            else sidestep_type
        )
        smoothed[index] = step_headings[index]._replace(
            trajectory_movement_type=movement_type,
            body_motion_angle_diff=evidence.angle_diff,
            sidestep_evidence_direction=_sidestep_direction_label(direction),
            sidestep_evidence_reason=(
                evidence.reason if evidence.reason is not None else "bridge_gap"
            ),
            sidestep_cluster_id=cluster_id,
        )


def _apply_suspect_cluster(
    smoothed: list[StepHeading],
    step_headings: list[StepHeading],
    evidences: list[_SidestepEvidence],
    evidence_indexes: list[int],
    direction: int,
    suspect_mode: str,
) -> None:
    """未確定clusterをsuspect設定に従って反映または抑制する。"""
    for index in evidence_indexes:
        evidence = evidences[index]
        suspect = evidence.strong and not _has_adjacent_opposite_evidence(
            evidences,
            index,
            direction,
        )
        suspect = suspect and suspect_mode != "forward"
        suspect_type = (
            "sidestep_suspect_left" if direction == 1 else "sidestep_suspect_right"
        )
        suspect_heading = (
            step_headings[index].motion_heading
            if suspect_mode in {"motion", "blend"}
            else None
        )
        smoothed[index] = step_headings[index]._replace(
            selected_heading=(
                suspect_heading if suspect else step_headings[index].selected_heading
            ),
            trajectory_movement_type=(
                suspect_type
                if suspect
                else _movement_type_when_sidestep_is_suppressed(step_headings[index])
            ),
            body_motion_angle_diff=evidence.angle_diff,
            sidestep_evidence_direction=_sidestep_direction_label(direction),
            sidestep_evidence_reason=evidence.reason,
            sidestep_cluster_id=None,
        )


def _smooth_clustered_headings(
    step_headings: list[StepHeading],
    suspect_mode: str,
) -> list[StepHeading]:
    """連続する同方向横歩きevidenceだけをclusterとして確定する。"""
    smoothed = list(step_headings)
    evidences = [_sidestep_evidence(heading) for heading in step_headings]
    cluster_id = 0
    index = 0
    while index < len(step_headings):
        direction = evidences[index].direction
        if direction is None:
            smoothed[index] = step_headings[index]._replace(
                body_motion_angle_diff=evidences[index].angle_diff,
                sidestep_evidence_direction=None,
                sidestep_evidence_reason=None,
                sidestep_cluster_id=None,
            )
            index += 1
            continue
        members, evidence_indexes, next_index = _find_cluster_members(
            step_headings,
            evidences,
            index,
            direction,
        )
        confirmed = len(evidence_indexes) >= 2
        confirmed = confirmed and _sidestep_cluster_has_lateral_strength(
            step_headings,
            evidence_indexes,
            direction,
        )

        if confirmed:
            cluster_id += 1
            _apply_confirmed_cluster(
                smoothed,
                step_headings,
                evidences,
                members,
                direction,
                cluster_id,
            )
        else:
            _apply_suspect_cluster(
                smoothed,
                step_headings,
                evidences,
                evidence_indexes,
                direction,
                suspect_mode,
            )
        index = next_index
    return smoothed


def _smooth_step_headings(
    step_headings: list[StepHeading],
    method: str = "none",
    sidestep_suspect_mode: str = SIDESTEP_SUSPECT_MODE,
) -> list[StepHeading]:
    """横歩き判定の軌跡反映を平滑化する。"""
    if method == "none":
        return step_headings
    if method == "isolated":
        return _smooth_isolated_headings(step_headings)
    return _smooth_clustered_headings(step_headings, sidestep_suspect_mode)


smooth_step_headings = _smooth_step_headings
