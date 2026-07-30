"""PDR のステップ方位候補の選択。

役割:
    ステップ方位候補の選択を独立した部品として実装する。
依存元:
    common の設定・共有型・時刻処理と同じ heading 領域の部品を利用する。
利用先:
    pdr pipeline または heading resolver から使用される。
処理フロー:
    各推定部品を呼び、指定方式に応じた StepHeading を組み立てる。
"""

import numpy as np
import pandas as pd

from ....common.config import (
    HEADING_METHOD,
    INITIAL_DIRECTION,
    MOTION_HEADING_CONFIDENCE_THRESHOLD,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
)
from ....common.lib.models import StepHeading, StepSegment
from ....common.lib.pdr_math import (
    _abs_angle_diff,
    _normalize_angle,
    _validate_heading_method,
    _validate_non_negative_parameter,
    _validate_positive_parameter,
)
from ....common.lib.time_utils import (
    _sample_gyro_angle,
    _step_mid_index,
    _step_mid_time,
    _step_output_time,
)
from .accel import _estimate_accel_headings
from .motion import _estimate_motion_heading_from_horizontal_accel


def resolve_step_heading(
    peaks: np.ndarray,
    df_gyro: pd.DataFrame,
    df_acc: pd.DataFrame,
    i: int,
    initial_direction: float = INITIAL_DIRECTION,
    heading_method: str = HEADING_METHOD,
    step_segments: tuple[StepSegment, ...] = (),
    motion_heading_correction: float = 0.0,
    sidestep_lateral_ratio: float = SIDESTEP_LATERAL_RATIO,
    sidestep_min_lateral_displacement: float = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    device_orientation_mode: str = "normal",
) -> StepHeading:
    """指定ステップのジャイロ/加速度/移動方向方位を解決する。"""
    selected_method = _validate_heading_method(heading_method)
    sidestep_lateral_ratio = _validate_positive_parameter(
        "sidestep_lateral_ratio",
        sidestep_lateral_ratio,
    )
    sidestep_min_lateral_displacement = _validate_non_negative_parameter(
        "sidestep_min_lateral_displacement",
        sidestep_min_lateral_displacement,
    )
    direction_offset = float(np.deg2rad(initial_direction))
    mid_idx = _step_mid_index(peaks, i)
    mid_time = _step_mid_time(df_acc, peaks, i)
    gyro_base = _sample_gyro_angle(df_gyro, sample_index=mid_idx, sample_time=mid_time)
    gyro_heading = (
        None if gyro_base is None else _normalize_angle(gyro_base + direction_offset)
    )
    accel_heading = _estimate_accel_headings(
        df_acc, peaks, i, direction_offset, step_segments
    )
    motion_heading = _estimate_motion_heading_from_horizontal_accel(
        df_acc,
        df_gyro,
        peaks,
        i,
        gyro_heading,
        direction_offset,
        step_segments,
        motion_heading_correction,
        sidestep_lateral_ratio,
        sidestep_min_lateral_displacement,
        device_orientation_mode,
    )

    # 指定された heading_method に従って候補方位を選ぶ。
    # 失敗時は gyro -> accel の順でフォールバックし、
    # 後段が扱える StepHeading にまとめる。
    selected_heading: float | None
    if selected_method == "accel_method1" and accel_heading.method1_heading is not None:
        selected_heading = accel_heading.method1_heading
        source = "accel_method1"
    elif (
        selected_method == "accel_method2" and accel_heading.method2_heading is not None
    ):
        selected_heading = accel_heading.method2_heading
        source = "accel_method2"
    elif selected_method == "gyro_accel_motion":
        if (
            motion_heading.motion_heading is not None
            and motion_heading.confidence >= MOTION_HEADING_CONFIDENCE_THRESHOLD
        ):
            selected_heading = motion_heading.motion_heading
            source = "gyro_accel_motion"
        else:
            selected_heading = gyro_heading
            source = "gyro" if gyro_heading is not None else "none"
    else:
        selected_heading = gyro_heading
        source = "gyro" if gyro_heading is not None else "none"

    if selected_heading is None and accel_heading.method1_heading is not None:
        selected_heading = accel_heading.method1_heading
        source = "accel_method1"

    return StepHeading(
        step_index=i + 1,
        timestamp_s=_step_output_time(df_acc, peaks, i),
        gyro_heading=gyro_heading,
        accel_method1_heading=accel_heading.method1_heading,
        accel_method2_heading=accel_heading.method2_heading,
        selected_heading=selected_heading,
        source=source,
        confidence=accel_heading.confidence,
        angle_diff_method1=_abs_angle_diff(gyro_heading, accel_heading.method1_heading),
        angle_diff_method2=_abs_angle_diff(gyro_heading, accel_heading.method2_heading),
        segment_start_index=accel_heading.segment_start_index,
        segment_end_index=accel_heading.segment_end_index,
        peak1_index=accel_heading.peak1_index,
        peak2_index=accel_heading.peak2_index,
        body_heading=motion_heading.body_heading,
        motion_heading=motion_heading.motion_heading,
        movement_type=motion_heading.movement_type,
        forward_displacement=motion_heading.forward_displacement,
        lateral_displacement=motion_heading.lateral_displacement,
        motion_confidence=motion_heading.confidence,
        motion_reject_reason=motion_heading.reject_reason,
        yaw_delta=motion_heading.yaw_delta,
        motion_heading_correction=motion_heading_correction,
        sidestep_lateral_ratio=sidestep_lateral_ratio,
        sidestep_min_lateral_displacement=sidestep_min_lateral_displacement,
        device_orientation_mode=device_orientation_mode,
    )
