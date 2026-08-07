"""PDR の端末姿勢推定。

役割:
    端末姿勢推定を独立した部品として実装する。
依存元:
    common の設定・共有型・時刻処理と同じ heading 領域の部品を利用する。
利用先:
    pdr pipeline または heading resolver から使用される。
処理フロー:
    水平加速度軸を補正し、候補ごとの整合度から姿勢モードを推定する。
"""

import numpy as np
import pandas as pd

from ....common.config import (
    MOTION_HEADING_CALIBRATION_STEPS,
    MOTION_HEADING_CONFIDENCE_THRESHOLD,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    TURNING_YAW_DELTA_THRESHOLD_DEG,
)
from ....common.lib.models import StepSegment
from ....common.lib.pdr_math import _normalize_angle
from ....common.lib.time_utils import (
    _sample_gyro_angle,
    _step_mid_index,
    _step_mid_time,
)
from ....common.lib.validation import DEVICE_ORIENTATION_MODES


def _rotate_vector(
    x: float,
    y: float,
    angle: float,
) -> tuple[float, float]:
    """2次元ベクトルを指定角度だけ回転する。"""
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


def _apply_device_orientation_to_horizontal(
    h_y: np.ndarray,
    h_z: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """端末装着向き候補に応じて水平加速度軸を反転する。"""
    if mode == "normal":
        return h_y, h_z
    if mode == "front_back_inverted":
        return -h_y, h_z
    if mode == "left_right_inverted":
        return h_y, -h_z
    if mode == "rotated_180":
        return -h_y, -h_z
    raise ValueError(f"unknown device_orientation_mode: {mode}")


def _classify_movement_type(
    forward_displacement: float,
    lateral_displacement: float,
    yaw_delta: float | None = None,
    sidestep_lateral_ratio: float = SIDESTEP_LATERAL_RATIO,
    sidestep_min_lateral_displacement: float = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
) -> str:
    """体の向きに対する移動タイプを返す。"""
    forward_abs = abs(forward_displacement)
    lateral_abs = abs(lateral_displacement)
    is_lateral_dominant = (
        lateral_abs >= sidestep_min_lateral_displacement
        and lateral_abs >= sidestep_lateral_ratio * max(forward_abs, 1e-12)
    )
    if yaw_delta is not None and abs(yaw_delta) >= np.deg2rad(
        TURNING_YAW_DELTA_THRESHOLD_DEG
    ):
        if is_lateral_dominant:
            return (
                "turning_sidestep_left"
                if lateral_displacement > 0
                else "turning_sidestep_right"
            )
        return "turning"

    if is_lateral_dominant:
        return "sidestep_left" if lateral_displacement > 0 else "sidestep_right"
    if forward_abs >= lateral_abs:
        return "forward"
    return "unknown"


def _estimate_device_orientation_mode(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    peaks: np.ndarray,
    initial_direction: float,
    step_segments: tuple[StepSegment, ...] = (),
) -> str:
    """初期歩行から端末水平軸の装着向き候補を推定する。"""
    if len(peaks) == 0:
        return "normal"

    direction_offset = float(np.deg2rad(initial_direction))
    best_mode = "normal"
    best_score = float("-inf")
    # 複数の装着向き候補を試し、初期歩行が最も「前進らしく」見える向きを採用する。
    for mode in DEVICE_ORIENTATION_MODES:
        score = 0.0
        count = 0
        observed_count = 0
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
            from .motion import _estimate_motion_heading_from_horizontal_accel

            motion = _estimate_motion_heading_from_horizontal_accel(
                df_acc,
                df_gyro,
                peaks,
                i,
                body_heading,
                direction_offset,
                step_segments,
                device_orientation_mode=mode,
            )
            if (
                motion.forward_displacement is None
                or motion.lateral_displacement is None
                or motion.motion_heading is None
            ):
                continue
            observed_count += 1
            if (
                motion.movement_type != "forward"
                or motion.confidence < MOTION_HEADING_CONFIDENCE_THRESHOLD
                or motion.forward_displacement <= 0.0
            ):
                continue
            forward = motion.forward_displacement
            lateral = abs(motion.lateral_displacement)
            diff = abs(_normalize_angle(motion.motion_heading - body_heading))
            score += forward - lateral - max(-forward, 0.0) - 0.25 * diff
            count += 1
        # 初期区間が横歩き・旋回中心なら装着向きを推定せず、既定向きを保つ。
        if count < 2 or count < 0.6 * observed_count:
            continue
        if count > 0:
            score /= count
        if score > best_score:
            best_score = score
            best_mode = mode
    return best_mode


apply_device_orientation_to_horizontal = _apply_device_orientation_to_horizontal
classify_movement_type = _classify_movement_type
estimate_device_orientation_mode = _estimate_device_orientation_mode
rotate_vector = _rotate_vector
