"""PDR の軌跡方位の選択と変化量制限。

役割:
    軌跡方位の選択と変化量制限を独立した部品として実装する。
依存元:
    common の設定・共有型・角度処理と同じ motion_state 領域の部品を利用する。
利用先:
    pdr pipeline と motion_state の後続処理から使用される。
処理フロー:
    移動種別ごとに採用方位を選び、連続歩の変化量を制限する。
"""

import numpy as np

from ....common.config import (
    FORWARD_HEADING_SOURCE,
)
from ....common.lib.angles import circular_mean_angles
from ....common.lib.models import (
    StepHeading,
)
from ....common.lib.pdr_math import (
    INITIAL_FORWARD_MOTION_BODY_CONSTRAINT_RAD,
    SIDESTEP_HEADING_MAX_STEP_DELTA_RAD,
    SIDESTEP_MOTION_LATERAL_CONSTRAINT_RAD,
    TRAJECTORY_HEADING_MAX_STEP_DELTA_RAD,
    TURNING_SIDESTEP_HEADING_MAX_STEP_DELTA_RAD,
    _normalize_angle,
)
from .step_motion import _sidestep_body_lateral_heading


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


def _forward_motion_heading(
    step_heading: StepHeading,
    previous_heading: float | None,
    forward_heading_source: str,
) -> tuple[float | None, str | None]:
    """前進歩のbody/motion候補を選ぶ。"""
    body_heading = (
        step_heading.body_heading
        if step_heading.body_heading is not None
        else step_heading.gyro_heading
    )
    if forward_heading_source == "body":
        return body_heading, "trajectory_body"
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
        return body_heading, "trajectory_initial_body_fallback"
    return candidate, source


def _sidestep_motion_heading(
    step_heading: StepHeading,
    movement_type: str,
    previous_heading: float | None,
    sidestep_heading_source: str,
) -> tuple[float | None, str]:
    """横歩きのmotion/body lateral候補を選ぶ。"""
    body_heading = (
        step_heading.body_heading
        if step_heading.body_heading is not None
        else step_heading.gyro_heading
    )
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
            body_lateral_heading if body_lateral_heading is not None else motion_heading
        )
        source = "trajectory_body_lateral"
    elif sidestep_heading_source == "blend":
        if (
            motion_heading is not None
            and body_lateral_heading is not None
            and abs(_normalize_angle(motion_heading - body_lateral_heading))
            <= SIDESTEP_MOTION_LATERAL_CONSTRAINT_RAD
        ):
            candidate = circular_mean_angles([motion_heading, body_lateral_heading])
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
    return candidate, source


def _resolve_world_motion_heading(
    step_heading: StepHeading,
    movement_type: str,
    previous_heading: float | None,
    previous_movement_type: str | None,
    forward_heading_source: str,
    sidestep_heading_source: str,
) -> tuple[float | None, str | None]:
    """移動状態に応じた世界座標方位を選び、連続時の変化量を制限する。"""
    if movement_type == "forward":
        candidate, source = _forward_motion_heading(
            step_heading,
            previous_heading,
            forward_heading_source,
        )
    elif movement_type in {
        "sidestep_left",
        "sidestep_right",
        "turning_sidestep_left",
        "turning_sidestep_right",
    }:
        candidate, source = _sidestep_motion_heading(
            step_heading,
            movement_type,
            previous_heading,
            sidestep_heading_source,
        )
    else:
        return None, None
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
            forward_heading_source,
            sidestep_heading_source,
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


stabilize_trajectory_body_headings = _stabilize_trajectory_body_headings
stabilize_trajectory_headings = _stabilize_trajectory_headings
