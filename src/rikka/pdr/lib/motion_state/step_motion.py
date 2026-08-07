"""PDR の歩行状態別の移動量決定。

役割:
    歩行状態別の移動量決定を独立した部品として実装する。
依存元:
    common の設定・共有型・角度処理と同じ motion_state 領域の部品を利用する。
利用先:
    pdr pipeline と motion_state の後続処理から使用される。
処理フロー:
    移動種別を正規化し、方位と歩幅倍率を適用して StepMotion を返す。
"""

import numpy as np

from ....common.config import (
    BACKWARD_LENGTH_SCALE,
    FORWARD_HEADING_SOURCE,
    SIDESTEP_LENGTH_SCALE,
    SIDESTEP_SUSPECT_MODE,
    TURNING_LENGTH_SCALE,
)
from ....common.lib.angles import circular_mean_angles
from ....common.lib.models import (
    StepHeading,
    StepMotion,
)
from ....common.lib.pdr_math import (
    _normalize_angle,
)

_circular_mean_angles = circular_mean_angles


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
    motion_heading = _sidestep_motion_heading(step_heading)
    body_lateral_heading = _sidestep_body_lateral_heading(body_heading, movement_type)

    if heading_source == "motion":
        return motion_heading if motion_heading is not None else body_lateral_heading
    if heading_source == "body_lateral":
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


def _resolve_motion_heading_and_scale(
    step_heading: StepHeading,
    movement_type: str,
    body_heading: float | None,
    fallback_heading: float,
    previous_heading: float | None,
    forward_heading_source: str,
    sidestep_heading_source: str,
    sidestep_suspect_mode: str,
) -> tuple[float, float, str]:
    """移動状態ごとの方位、歩幅倍率、確定状態を返す。"""
    heading: float | None
    if movement_type == "forward":
        if forward_heading_source == "body":
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
            else sidestep_suspect_mode
            if movement_type == "sidestep_suspect_left"
            else sidestep_heading_source
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
            else sidestep_suspect_mode
            if movement_type == "sidestep_suspect_right"
            else sidestep_heading_source
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
    return (
        heading if heading is not None else fallback_heading,
        scale,
        movement_type,
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
    resolved_heading, scale, movement_type = _resolve_motion_heading_and_scale(
        step_heading,
        movement_type,
        body_heading,
        fallback_heading,
        previous_heading,
        forward_heading_source,
        sidestep_heading_source,
        sidestep_suspect_mode,
    )

    return StepMotion(
        heading=_normalize_angle(float(resolved_heading)),
        length=float(step_length * scale),
        movement_type=movement_type,
        length_scale=float(scale),
    )


# 領域内の別モジュールから利用する helper は public 名で公開する。
