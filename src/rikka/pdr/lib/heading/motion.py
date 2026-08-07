"""PDR の水平加速度による移動方位推定。

役割:
    水平加速度による移動方位推定を独立した部品として実装する。
依存元:
    common の設定・共有型・時刻処理と同じ heading 領域の部品を利用する。
利用先:
    pdr pipeline または heading resolver から使用される。
処理フロー:
    加速度を積分して変位、移動種別、補正値を返す。
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from ....common.config import (
    MOTION_HEADING_CALIBRATION_STEPS,
    MOTION_HEADING_CONFIDENCE_THRESHOLD,
    MOTION_HEADING_MIN_DISPLACEMENT_M,
    SAMPLING_RATE,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
)
from ....common.lib.models import StepSegment
from ....common.lib.pdr_math import (
    _normalize_angle,
    _score_ratio,
)
from ....common.lib.time_utils import (
    _sample_gyro_angle,
    _step_mid_index,
    _step_mid_time,
    _time_values,
)
from .device_orientation import (
    _apply_device_orientation_to_horizontal,
    _classify_movement_type,
    _rotate_vector,
)
from .gyro import _step_segment_bounds


class _MotionHeadingResult(NamedTuple):
    """ジャイロで世界座標へ回転した水平加速度から推定した移動方向。"""

    body_heading: float | None
    motion_heading: float | None
    movement_type: str
    forward_displacement: float | None
    lateral_displacement: float | None
    confidence: float
    reject_reason: str | None
    yaw_delta: float | None = None


def _dataframe_times_or_sample_index(df: pd.DataFrame) -> np.ndarray:
    """DataFrame の時刻列が使えない場合は固定サンプリング周期の時刻を返す。"""
    times = _time_values(df)
    if times is not None:
        return times
    return np.arange(len(df), dtype=float) / SAMPLING_RATE


def _integrate_motion_with_zero_velocity(
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    times: np.ndarray,
) -> tuple[float, float]:
    """水平加速度を2重積分し、ステップ両端の速度を0に揃える。"""
    if len(acc_x) < 3 or len(acc_y) < 3 or len(times) < 3:
        return 0.0, 0.0
    dt = np.diff(times, prepend=times[0])
    dt[0] = 0.0
    if not np.isfinite(dt).all() or np.any(dt < 0):
        dt = np.full(len(times), 1.0 / SAMPLING_RATE)
        dt[0] = 0.0

    velocity_x = np.cumsum(acc_x * dt)
    velocity_y = np.cumsum(acc_y * dt)
    velocity_x -= np.linspace(velocity_x[0], velocity_x[-1], len(velocity_x))
    velocity_y -= np.linspace(velocity_y[0], velocity_y[-1], len(velocity_y))

    return (
        float(np.sum(velocity_x * dt)),
        float(np.sum(velocity_y * dt)),
    )


def _motion_result_from_samples(
    h_y: np.ndarray,
    h_z: np.ndarray,
    sample_times: np.ndarray,
    gyro_times: np.ndarray,
    low_angle: np.ndarray,
    valid_gyro: np.ndarray,
    body_heading: float,
    direction_offset: float,
    motion_heading_correction: float,
    sidestep_lateral_ratio: float,
    sidestep_min_lateral_displacement: float,
) -> _MotionHeadingResult:
    """有効サンプルを世界座標へ積分し移動状態を返す。"""
    angles = np.interp(sample_times, gyro_times[valid_gyro], low_angle[valid_gyro])
    angles = angles + direction_offset
    yaw_delta = _normalize_angle(float(angles[-1] - angles[0]))
    world_x = h_y * np.cos(angles) - h_z * np.sin(angles)
    world_y = h_y * np.sin(angles) + h_z * np.cos(angles)
    disp_x, disp_y = _integrate_motion_with_zero_velocity(
        world_x,
        world_y,
        sample_times,
    )
    if abs(motion_heading_correction) > 1e-12:
        disp_x, disp_y = _rotate_vector(disp_x, disp_y, -motion_heading_correction)
    displacement_norm = float(np.hypot(disp_x, disp_y))
    if displacement_norm <= 1e-12:
        return _MotionHeadingResult(
            body_heading,
            None,
            "unknown",
            0.0,
            0.0,
            0.0,
            "zero_motion",
            yaw_delta,
        )
    motion_heading = _normalize_angle(float(np.arctan2(disp_y, disp_x)))
    body_axis = np.array([np.cos(body_heading), np.sin(body_heading)], dtype=float)
    lateral_axis = np.array([-body_axis[1], body_axis[0]], dtype=float)
    displacement = np.array([disp_x, disp_y], dtype=float)
    forward_displacement = float(displacement @ body_axis)
    lateral_displacement = float(displacement @ lateral_axis)
    movement_type = _classify_movement_type(
        forward_displacement,
        lateral_displacement,
        yaw_delta,
        sidestep_lateral_ratio,
        sidestep_min_lateral_displacement,
    )
    confidence = _score_ratio(displacement_norm, MOTION_HEADING_MIN_DISPLACEMENT_M)
    return _MotionHeadingResult(
        body_heading=body_heading,
        motion_heading=motion_heading,
        movement_type=movement_type,
        forward_displacement=forward_displacement,
        lateral_displacement=lateral_displacement,
        confidence=confidence,
        reject_reason=(
            None
            if confidence >= MOTION_HEADING_CONFIDENCE_THRESHOLD
            else "low_motion_confidence"
        ),
        yaw_delta=yaw_delta,
    )


def _unknown_motion_result(
    body_heading: float | None,
    reason: str,
) -> _MotionHeadingResult:
    """移動方位を計算できない場合の共通結果を返す。"""
    return _MotionHeadingResult(
        body_heading,
        None,
        "unknown",
        None,
        None,
        0.0,
        reason,
    )


def _estimate_motion_heading_from_horizontal_accel(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    peaks: np.ndarray,
    i: int,
    body_heading: float | None,
    direction_offset: float,
    step_segments: tuple[StepSegment, ...] = (),
    motion_heading_correction: float = 0.0,
    sidestep_lateral_ratio: float = SIDESTEP_LATERAL_RATIO,
    sidestep_min_lateral_displacement: float = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    device_orientation_mode: str = "normal",
) -> _MotionHeadingResult:
    """ジャイロで向きを固定し、水平加速度から世界座標上の移動方向を推定する。"""
    if body_heading is None:
        return _unknown_motion_result(None, "no_gyro")
    if "low_angle" not in df_gyro.columns:
        return _unknown_motion_result(body_heading, "no_gyro_angle")

    bounds = _step_segment_bounds(peaks, i, len(df_acc), step_segments)
    if bounds is None:
        return _unknown_motion_result(body_heading, "no_step_bounds")
    start, end = bounds
    if end - start < 3:
        return _unknown_motion_result(body_heading, "short_segment")

    # 端末座標の水平加速度を取り出し、推定済みの装着向き補正を適用する。
    h_y = np.asarray(pd.to_numeric(df_acc["h_y"].iloc[start:end], errors="coerce"))
    h_z = np.asarray(pd.to_numeric(df_acc["h_z"].iloc[start:end], errors="coerce"))
    sample_times = _dataframe_times_or_sample_index(df_acc)[start:end]
    valid_acc = np.isfinite(h_y) & np.isfinite(h_z) & np.isfinite(sample_times)
    if int(valid_acc.sum()) < 3:
        return _unknown_motion_result(body_heading, "no_horizontal_accel")

    h_y = h_y[valid_acc].astype(float)
    h_z = h_z[valid_acc].astype(float)
    h_y, h_z = _apply_device_orientation_to_horizontal(
        h_y,
        h_z,
        device_orientation_mode,
    )
    sample_times = sample_times[valid_acc].astype(float)

    gyro_times = _dataframe_times_or_sample_index(df_gyro)
    low_angle = np.asarray(
        pd.to_numeric(df_gyro["low_angle"], errors="coerce"),
        dtype=float,
    )
    valid_gyro = np.isfinite(gyro_times) & np.isfinite(low_angle)
    if not valid_gyro.any():
        return _unknown_motion_result(body_heading, "no_gyro_angle")

    return _motion_result_from_samples(
        h_y,
        h_z,
        sample_times,
        gyro_times,
        low_angle,
        valid_gyro,
        body_heading,
        direction_offset,
        motion_heading_correction,
        sidestep_lateral_ratio,
        sidestep_min_lateral_displacement,
    )


def _estimate_motion_heading_correction(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    peaks: np.ndarray,
    initial_direction: float,
    step_segments: tuple[StepSegment, ...] = (),
    device_orientation_mode: str = "normal",
) -> float:
    """歩行開始直後の前進歩行から水平加速度方位の固定ずれを推定する。"""
    if MOTION_HEADING_CALIBRATION_STEPS <= 0:
        return 0.0

    direction_offset = float(np.deg2rad(initial_direction))
    sin_sum = 0.0
    cos_sum = 0.0
    count = 0
    for i in range(min(len(peaks), MOTION_HEADING_CALIBRATION_STEPS)):
        mid_idx = _step_mid_index(peaks, i)
        mid_time = _step_mid_time(df_acc, peaks, i)
        gyro_base = _sample_gyro_angle(
            df_gyro,
            sample_index=mid_idx,
            sample_time=mid_time,
        )
        if gyro_base is None:
            continue
        body_heading = _normalize_angle(gyro_base + direction_offset)
        motion = _estimate_motion_heading_from_horizontal_accel(
            df_acc,
            df_gyro,
            peaks,
            i,
            body_heading,
            direction_offset,
            step_segments,
            device_orientation_mode=device_orientation_mode,
        )
        if (
            motion.motion_heading is None
            or motion.confidence < MOTION_HEADING_CONFIDENCE_THRESHOLD
        ):
            continue
        diff = _normalize_angle(motion.motion_heading - body_heading)
        sin_sum += float(np.sin(diff))
        cos_sum += float(np.cos(diff))
        count += 1

    if count == 0 or float(np.hypot(sin_sum, cos_sum)) <= 1e-12:
        return 0.0
    return _normalize_angle(float(np.arctan2(sin_sum, cos_sum)))


def _resolve_motion_heading_correction(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    peaks: np.ndarray,
    initial_direction: float,
    step_segments: tuple[StepSegment, ...],
    method: str,
    device_orientation_mode: str = "normal",
) -> float:
    """指定モードに応じた水平加速度方位補正角を返す。"""
    if method == "none":
        return 0.0
    return _estimate_motion_heading_correction(
        df_acc,
        df_gyro,
        peaks,
        initial_direction,
        step_segments,
        device_orientation_mode,
    )


estimate_motion_heading_correction = _estimate_motion_heading_correction
estimate_motion_heading_from_horizontal_accel = (
    _estimate_motion_heading_from_horizontal_accel
)
integrate_motion_with_zero_velocity = _integrate_motion_with_zero_velocity
resolve_motion_heading_correction = _resolve_motion_heading_correction
