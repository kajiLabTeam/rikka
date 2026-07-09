from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ...config import (
    ACCEL_HEADING_MIN_LINE_LENGTH,
    ACCEL_HEADING_MIN_PEAK_DISTANCE,
    ACCEL_HEADING_MIN_PEAK_NORM,
    BACKWARD_LENGTH_SCALE,
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    FORWARD_HEADING_SOURCE,
    GYRO_BIAS_METHOD,
    HEADING_METHOD,
    INITIAL_DIRECTION,
    MAX_SEG_SAMPLES,
    MIN_SEG_SAMPLES,
    MOTION_HEADING_CALIBRATION_STEPS,
    MOTION_HEADING_CONFIDENCE_THRESHOLD,
    MOTION_HEADING_MIN_DISPLACEMENT_M,
    SAMPLING_RATE,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_LENGTH_SCALE,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    SIDESTEP_SMOOTHING_METHOD,
    STEP_LENGTH_METHOD,
    STEP_LENGTH_WINDOW,
    TURNING_LENGTH_SCALE,
    TURNING_YAW_DELTA_THRESHOLD_DEG,
    USER_HEIGHT_M,
    WEINBERG_K,
    compute_weinberg_k,
)
from ...config import (
    DATA_DIR as DATA_DIR,
)
from ...config import (
    GYRO_BIAS_MIN_CALIBRATION_SECONDS as GYRO_BIAS_MIN_CALIBRATION_SECONDS,
)
from ...config import (
    GYRO_BIAS_OUTLIER_MAD_SCALE as GYRO_BIAS_OUTLIER_MAD_SCALE,
)
from ...config import (
    GYRO_BIAS_STATIC_ACCEL_P95_WEIGHT as GYRO_BIAS_STATIC_ACCEL_P95_WEIGHT,
)
from ...config import (
    GYRO_BIAS_STATIC_GYRO_STD_WEIGHT as GYRO_BIAS_STATIC_GYRO_STD_WEIGHT,
)
from ...config import (
    GYRO_BIAS_STATIC_MAX_ACCEL_P95 as GYRO_BIAS_STATIC_MAX_ACCEL_P95,
)
from ...config import (
    GYRO_BIAS_STATIC_MAX_GYRO_STD as GYRO_BIAS_STATIC_MAX_GYRO_STD,
)
from ...config import (
    GYRO_BIAS_STATIC_SEARCH_END_SECONDS as GYRO_BIAS_STATIC_SEARCH_END_SECONDS,
)
from ...config import (
    GYRO_BIAS_STATIC_SEARCH_START_SECONDS as GYRO_BIAS_STATIC_SEARCH_START_SECONDS,
)
from ...config import (
    GYRO_BIAS_STATIC_WALK_ONSET_MARGIN_S as GYRO_BIAS_STATIC_WALK_ONSET_MARGIN_S,
)
from ...config import (
    GYRO_BIAS_STATIC_WINDOW_SECONDS as GYRO_BIAS_STATIC_WINDOW_SECONDS,
)
from ...config import (
    GYRO_BIAS_STATIC_WINDOW_STEP_SECONDS as GYRO_BIAS_STATIC_WINDOW_STEP_SECONDS,
)
from ...config import (
    GYRO_BIAS_WALK_ONSET_MAX_INTERVAL_S as GYRO_BIAS_WALK_ONSET_MAX_INTERVAL_S,
)
from ...config import (
    GYRO_BIAS_WALK_ONSET_MIN_STEPS as GYRO_BIAS_WALK_ONSET_MIN_STEPS,
)
from ...config import (
    K_FORWARD as K_FORWARD,
)
from ...config import (
    PEAK_DISTANCE as PEAK_DISTANCE,
)
from ...config import (
    PEAK_HEIGHT as PEAK_HEIGHT,
)
from ...config import (
    STEP_DETECTION_METHOD as STEP_DETECTION_METHOD,
)
from ...config import (
    STEP_VERTICAL_SMOOTH_WINDOW as STEP_VERTICAL_SMOOTH_WINDOW,
)
from ...config import (
    STEP_VERTICAL_THRESHOLD_PERCENTILE as STEP_VERTICAL_THRESHOLD_PERCENTILE,
)
from ...config import (
    WINDOW_ACC as WINDOW_ACC,
)
from ...config import (
    WINDOW_GYRO as WINDOW_GYRO,
)
from .gyro_bias import (
    GYRO_BIAS_METHODS as GYRO_BIAS_METHODS,
)
from .gyro_bias import (
    _estimate_gyro_bias_initial_robust as _estimate_gyro_bias_initial_robust,
)
from .gyro_bias import (
    _estimate_gyro_bias_prewalk_robust as _estimate_gyro_bias_prewalk_robust,
)
from .gyro_bias import (
    _estimate_gyro_bias_quietest as _estimate_gyro_bias_quietest,
)
from .gyro_bias import (
    _estimate_gyro_bias_static_window as _estimate_gyro_bias_static_window,
)
from .gyro_bias import (
    _find_static_gyro_bias_candidate as _find_static_gyro_bias_candidate,
)
from .gyro_bias import (
    _find_walk_onset_time as _find_walk_onset_time,
)
from .gyro_bias import (
    _GyroBiasStaticCandidate as _GyroBiasStaticCandidate,
)
from .gyro_bias import (
    _robust_gyro_bias_from_mask as _robust_gyro_bias_from_mask,
)
from .gyro_bias import (
    _startup_static_search_range as _startup_static_search_range,
)
from .gyro_bias import (
    _time_mask as _time_mask,
)
from .gyro_bias import (
    _validate_gyro_bias_method as _validate_gyro_bias_method,
)
from .gyro_bias import (
    estimate_gyro_bias as estimate_gyro_bias,
)
from .models import (
    GyroBiasResult,
    PreparedPdrSteps,
    StepHeading,
    StepMotion,
    StepSegment,
)
from .models import (
    StepDetectionResult as StepDetectionResult,
)
from .outputs import (
    _angle_to_deg as _angle_to_deg,
)
from .outputs import (
    _build_gyro_bias_dataframe as _build_gyro_bias_dataframe,
)
from .outputs import (
    _build_step_headings_dataframe as _build_step_headings_dataframe,
)
from .outputs import (
    _build_step_segments_dataframe as _build_step_segments_dataframe,
)
from .outputs import (
    _build_step_vectors_dataframe as _build_step_vectors_dataframe,
)
from .outputs import (
    _build_trajectory_dataframe as _build_trajectory_dataframe,
)
from .outputs import (
    _create_output_dir as _create_output_dir,
)
from .outputs import (
    _step_plot_signal as _step_plot_signal,
)
from .plotting import (
    _compute_pixel_coords as _compute_pixel_coords,
)
from .plotting import (
    _pixel_vector_from_heading as _pixel_vector_from_heading,
)
from .plotting import (
    _plot_heading_overlay as _plot_heading_overlay,
)
from .plotting import (
    plot_trajectory as plot_trajectory,
)
from .sensors import (
    ACC_COLUMNS as ACC_COLUMNS,
)
from .sensors import (
    GYRO_COLUMNS as GYRO_COLUMNS,
)
from .sensors import (
    load_sensor_data,
    process_sensor_data,
)
from .step_detection import (
    STEP_DETECTION_METHODS as STEP_DETECTION_METHODS,
)
from .step_detection import (
    _detect_steps_by_peak as _detect_steps_by_peak,
)
from .step_detection import (
    _detect_steps_by_vertical_threshold as _detect_steps_by_vertical_threshold,
)
from .step_detection import (
    _suppress_close_contacts as _suppress_close_contacts,
)
from .step_detection import (
    _threshold_groups as _threshold_groups,
)
from .step_detection import (
    _validate_step_detection_method as _validate_step_detection_method,
)
from .step_detection import (
    detect_step_result,
)
from .step_detection import (
    detect_steps as detect_steps,
)
from .step_length import (
    _estimate_initial_forward_angle as _estimate_initial_forward_angle,
)
from .step_length import (
    estimate_step_length as estimate_step_length,
)
from .step_length import (
    estimate_step_length_forward as estimate_step_length_forward,
)
from .time_utils import (
    _gyro_integration_dt as _gyro_integration_dt,
)
from .time_utils import (
    _sample_gyro_angle as _sample_gyro_angle,
)
from .time_utils import (
    _step_mid_index as _step_mid_index,
)
from .time_utils import (
    _step_mid_time as _step_mid_time,
)
from .time_utils import (
    _step_output_time as _step_output_time,
)
from .time_utils import (
    _time_at_index as _time_at_index,
)
from .time_utils import (
    _time_values as _time_values,
)

HEADING_METHODS = (
    "gyro",
    "accel_method1",
    "accel_method2",
    "gyro_accel_motion",
)
MOTION_HEADING_CORRECTION_METHODS = ("auto", "none")
SIDESTEP_SMOOTHING_METHODS = ("none", "isolated", "clustered")
FORWARD_HEADING_SOURCES = ("body", "motion")
SIDESTEP_HEADING_SOURCES = ("motion", "body_lateral", "blend")
SIDESTEP_SUSPECT_MODES = ("motion", "body_lateral", "blend", "forward")
SIDESTEP_BODY_MOTION_ANGLE_THRESHOLD_RAD = np.deg2rad(45.0)
SIDESTEP_BODY_MOTION_RATIO_THRESHOLD = 0.8
SIDESTEP_STRONG_ANGLE_THRESHOLD_RAD = np.deg2rad(75.0)
SIDESTEP_MOTION_LATERAL_CONSTRAINT_RAD = np.deg2rad(45.0)
TRAJECTORY_HEADING_MAX_STEP_DELTA_RAD = np.deg2rad(25.0)
SIDESTEP_HEADING_MAX_STEP_DELTA_RAD = np.deg2rad(45.0)
TURNING_SIDESTEP_HEADING_MAX_STEP_DELTA_RAD = np.deg2rad(90.0)
INITIAL_FORWARD_MOTION_BODY_CONSTRAINT_RAD = np.deg2rad(45.0)
DEVICE_ORIENTATION_MODES = (
    "normal",
    "front_back_inverted",
    "left_right_inverted",
    "rotated_180",
)


def _validate_scale(scale: float) -> None:
    """フロアマップ縮尺が正の値であることを確認する。"""
    if scale <= 0:
        raise ValueError("scale は正の値を指定してください。")


class _AccelHeadingResult(NamedTuple):
    """加速度方位候補の内部計算結果。"""

    method1_heading: float | None
    method2_heading: float | None
    confidence: float
    segment_start_index: int | None
    segment_end_index: int | None
    peak1_index: int | None
    peak2_index: int | None


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


def _validate_heading_method(method: str) -> str:
    """方位推定手法名を検証する。"""
    if method not in HEADING_METHODS:
        allowed = ", ".join(HEADING_METHODS)
        raise ValueError(f"heading_method は {allowed} のいずれかを指定してください。")
    return method


def _normalize_angle(angle: float) -> float:
    """角度を [-pi, pi) に正規化する。"""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _abs_angle_diff(angle_a: float | None, angle_b: float | None) -> float | None:
    """2つの角度差の絶対値を返す。どちらかが None なら None。"""
    if angle_a is None or angle_b is None:
        return None
    return abs(_normalize_angle(angle_a - angle_b))


def _score_ratio(value: float, target: float) -> float:
    """target 以上を 1.0 とする 0..1 スコアを返す。"""
    if target <= 0:
        return 1.0
    return float(np.clip(value / target, 0.0, 1.0))


def _validate_positive_parameter(name: str, value: float) -> float:
    """正の解析パラメータであることを確認する。"""
    if value <= 0:
        raise ValueError(f"{name} は正の値を指定してください。")
    return value


def _validate_non_negative_parameter(name: str, value: float) -> float:
    """0以上の解析パラメータであることを確認する。"""
    if value < 0:
        raise ValueError(f"{name} は0以上の値を指定してください。")
    return value


def _validate_motion_heading_correction(method: str) -> str:
    """水平加速度移動方向の固定ずれ補正モードを検証する。"""
    if method not in MOTION_HEADING_CORRECTION_METHODS:
        allowed = ", ".join(MOTION_HEADING_CORRECTION_METHODS)
        raise ValueError(
            f"motion_heading_correction は {allowed} のいずれかを指定してください。"
        )
    return method


def _validate_sidestep_smoothing(method: str) -> str:
    """横歩き判定の平滑化モードを検証する。"""
    if method not in SIDESTEP_SMOOTHING_METHODS:
        allowed = ", ".join(SIDESTEP_SMOOTHING_METHODS)
        raise ValueError(
            f"sidestep_smoothing は {allowed} のいずれかを指定してください。"
        )
    return method


def _validate_forward_heading_source(source: str) -> str:
    """forward 判定ステップに使う方位ソースを検証する。"""
    if source not in FORWARD_HEADING_SOURCES:
        allowed = ", ".join(FORWARD_HEADING_SOURCES)
        raise ValueError(
            f"forward_heading_source は {allowed} のいずれかを指定してください。"
        )
    return source


def _validate_sidestep_heading_source(source: str) -> str:
    """横歩き確定ステップに使う方位ソースを検証する。"""
    if source not in SIDESTEP_HEADING_SOURCES:
        allowed = ", ".join(SIDESTEP_HEADING_SOURCES)
        raise ValueError(
            f"sidestep_heading_source は {allowed} のいずれかを指定してください。"
        )
    return source


def _validate_sidestep_suspect_mode(mode: str) -> str:
    """横歩き疑いステップの軌跡反映モードを検証する。"""
    if mode not in SIDESTEP_SUSPECT_MODES:
        allowed = ", ".join(SIDESTEP_SUSPECT_MODES)
        raise ValueError(
            f"sidestep_suspect_mode は {allowed} のいずれかを指定してください。"
        )
    return mode


def _step_segment_bounds(
    peaks: np.ndarray,
    i: int,
    n_samples: int,
    step_segments: tuple[StepSegment, ...] = (),
) -> tuple[int, int] | None:
    """加速度方位推定に使うステップ区間を返す。"""
    if i < len(step_segments):
        segment = step_segments[i]
        return segment.start_index, segment.end_index
    if i + 1 < len(peaks):
        return int(peaks[i]), int(peaks[i + 1])
    if i < len(peaks):
        peak = int(peaks[i])
        return (
            max(0, peak - STEP_LENGTH_WINDOW),
            min(n_samples, peak + STEP_LENGTH_WINDOW + 1),
        )
    return None


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
        return _MotionHeadingResult(None, None, "unknown", None, None, 0.0, "no_gyro")
    if "low_angle" not in df_gyro.columns:
        return _MotionHeadingResult(
            body_heading,
            None,
            "unknown",
            None,
            None,
            0.0,
            "no_gyro_angle",
        )

    bounds = _step_segment_bounds(peaks, i, len(df_acc), step_segments)
    if bounds is None:
        return _MotionHeadingResult(
            body_heading,
            None,
            "unknown",
            None,
            None,
            0.0,
            "no_step_bounds",
        )
    start, end = bounds
    if end - start < 3:
        return _MotionHeadingResult(
            body_heading,
            None,
            "unknown",
            None,
            None,
            0.0,
            "short_segment",
        )

    h_y = np.asarray(pd.to_numeric(df_acc["h_y"].iloc[start:end], errors="coerce"))
    h_z = np.asarray(pd.to_numeric(df_acc["h_z"].iloc[start:end], errors="coerce"))
    sample_times = _dataframe_times_or_sample_index(df_acc)[start:end]
    valid_acc = np.isfinite(h_y) & np.isfinite(h_z) & np.isfinite(sample_times)
    if int(valid_acc.sum()) < 3:
        return _MotionHeadingResult(
            body_heading,
            None,
            "unknown",
            None,
            None,
            0.0,
            "no_horizontal_accel",
        )

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
        return _MotionHeadingResult(
            body_heading,
            None,
            "unknown",
            None,
            None,
            0.0,
            "no_gyro_angle",
        )

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
    confidence = _score_ratio(
        displacement_norm,
        MOTION_HEADING_MIN_DISPLACEMENT_M,
    )
    reject_reason = (
        None
        if confidence >= MOTION_HEADING_CONFIDENCE_THRESHOLD
        else "low_motion_confidence"
    )
    return _MotionHeadingResult(
        body_heading=body_heading,
        motion_heading=motion_heading,
        movement_type=movement_type,
        forward_displacement=forward_displacement,
        lateral_displacement=lateral_displacement,
        confidence=confidence,
        reject_reason=reject_reason,
        yaw_delta=yaw_delta,
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
    for mode in DEVICE_ORIENTATION_MODES:
        score = 0.0
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
                device_orientation_mode=mode,
            )
            if (
                motion.forward_displacement is None
                or motion.lateral_displacement is None
                or motion.motion_heading is None
            ):
                continue
            forward = motion.forward_displacement
            lateral = abs(motion.lateral_displacement)
            diff = abs(_normalize_angle(motion.motion_heading - body_heading))
            score += forward - lateral - max(-forward, 0.0) - 0.25 * diff
            count += 1
        if count > 0:
            score /= count
        if score > best_score:
            best_score = score
            best_mode = mode
    return best_mode


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
    selected_method = _validate_motion_heading_correction(method)
    if selected_method == "none":
        return 0.0
    return _estimate_motion_heading_correction(
        df_acc,
        df_gyro,
        peaks,
        initial_direction,
        step_segments,
        device_orientation_mode,
    )


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
    sidestep_suspect_mode: str = "motion",
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
                smoothed[i] = current._replace(trajectory_movement_type="forward")
        return smoothed

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
                    trajectory_movement_type=suspect_type if suspect else "forward",
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
    sidestep_suspect_mode: str = "motion",
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


def _select_two_accel_peaks(norm: np.ndarray) -> tuple[int, int] | None:
    """平面加速度ノルムから方位推定用の2つの極大点を選ぶ。"""
    if len(norm) < 2 or not np.isfinite(norm).any():
        return None

    safe_norm = np.where(np.isfinite(norm), norm, -np.inf)
    peak_indexes, _ = find_peaks(
        safe_norm,
        distance=max(1, ACCEL_HEADING_MIN_PEAK_DISTANCE),
    )
    candidates = list(peak_indexes)

    # 端点にピークが出るデータでは find_peaks が拾えないため、強い点を補助候補にする。
    for index in np.argsort(safe_norm)[::-1]:
        int_index = int(index)
        if safe_norm[int_index] == -np.inf:
            continue
        if int_index not in candidates:
            candidates.append(int_index)
        if len(candidates) >= 4:
            break

    selected: list[int] = []
    for index in sorted(candidates, key=lambda idx: safe_norm[idx], reverse=True):
        if all(index != existing for existing in selected):
            selected.append(int(index))
        if len(selected) == 2:
            break

    if len(selected) < 2:
        return None
    peak_a, peak_b = sorted(selected[:2])
    return peak_a, peak_b


def _estimate_accel_headings(
    df_acc: pd.DataFrame,
    peaks: np.ndarray,
    i: int,
    direction_offset: float,
    step_segments: tuple[StepSegment, ...] = (),
) -> _AccelHeadingResult:
    """論文手法1/2の加速度平面成分方位と信頼度を返す。"""
    bounds = _step_segment_bounds(peaks, i, len(df_acc), step_segments)
    if bounds is None:
        return _AccelHeadingResult(None, None, 0.0, None, None, None, None)

    start, end = bounds
    if end <= start:
        return _AccelHeadingResult(None, None, 0.0, start, end, None, None)

    h_y = df_acc["h_y"].iloc[start:end].to_numpy(dtype=float)
    h_z = df_acc["h_z"].iloc[start:end].to_numpy(dtype=float)
    valid = np.isfinite(h_y) & np.isfinite(h_z)
    if valid.sum() < 3:
        return _AccelHeadingResult(None, None, 0.0, start, end, None, None)

    h_y_safe = np.where(valid, h_y, np.nan)
    h_z_safe = np.where(valid, h_z, np.nan)
    norm = np.sqrt(h_y_safe**2 + h_z_safe**2)
    selected = _select_two_accel_peaks(norm)
    if selected is None:
        return _AccelHeadingResult(None, None, 0.0, start, end, None, None)

    local_peak1, local_peak2 = selected
    peak1_index = start + local_peak1
    peak2_index = start + local_peak2
    point1 = np.array([h_y_safe[local_peak1], h_z_safe[local_peak1]], dtype=float)
    point2 = np.array([h_y_safe[local_peak2], h_z_safe[local_peak2]], dtype=float)
    if not np.isfinite(point1).all() or not np.isfinite(point2).all():
        return _AccelHeadingResult(
            None, None, 0.0, start, end, peak1_index, peak2_index
        )

    norm1 = float(norm[local_peak1])
    norm2 = float(norm[local_peak2])
    line_length = float(np.linalg.norm(point2 - point1))
    peak_distance = abs(local_peak2 - local_peak1)
    seg_len = end - start

    if line_length <= 1e-12:
        return _AccelHeadingResult(
            None, None, 0.0, start, end, peak1_index, peak2_index
        )

    # 手法1: 時間的に早い極大値方向。手法2: ノルムが大きい極大値方向。
    method1_vec = point1 - point2
    method2_vec = point1 - point2 if norm1 >= norm2 else point2 - point1
    method1_heading = _normalize_angle(
        float(np.arctan2(method1_vec[1], method1_vec[0])) + direction_offset
    )
    method2_heading = _normalize_angle(
        float(np.arctan2(method2_vec[1], method2_vec[0])) + direction_offset
    )

    strength_score = _score_ratio(min(norm1, norm2), ACCEL_HEADING_MIN_PEAK_NORM)
    separation_score = _score_ratio(peak_distance, ACCEL_HEADING_MIN_PEAK_DISTANCE)
    line_length_score = _score_ratio(line_length, ACCEL_HEADING_MIN_LINE_LENGTH)
    duration_score = 1.0 if MIN_SEG_SAMPLES <= seg_len <= MAX_SEG_SAMPLES else 0.0
    confidence = strength_score * separation_score * line_length_score * duration_score

    return _AccelHeadingResult(
        method1_heading=method1_heading,
        method2_heading=method2_heading,
        confidence=float(confidence),
        segment_start_index=start,
        segment_end_index=end,
        peak1_index=peak1_index,
        peak2_index=peak2_index,
    )


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


def estimate_trajectory_with_headings(
    peaks: np.ndarray,
    df_gyro: pd.DataFrame,
    df_acc: pd.DataFrame,
    initial_direction: float = INITIAL_DIRECTION,
    weinberg_k: float = WEINBERG_K,
    heading_method: str = HEADING_METHOD,
    step_segments: tuple[StepSegment, ...] = (),
    sidestep_lateral_ratio: float = SIDESTEP_LATERAL_RATIO,
    sidestep_min_lateral_displacement: float = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    motion_heading_correction: str = "auto",
    sidestep_smoothing: str = SIDESTEP_SMOOTHING_METHOD,
    forward_heading_source: str = FORWARD_HEADING_SOURCE,
    sidestep_heading_source: str = "motion",
    sidestep_suspect_mode: str = "motion",
) -> tuple[list[list[float]], list[float], list[float], list[StepHeading]]:
    """ステップピークとジャイロスコープ角度から2次元軌跡を推定する。

    各ステップピーク時刻の平滑化角度（``low_angle``）と
    ``STEP_LENGTH_METHOD`` で選択した手法による歩幅推定をもとに次の座標を計算し，
    軌跡を構築する。原点 [0.0, 0.0] から始まり，ステップごとに座標を追加する。
    ``initial_direction`` を加算することで，歩行開始方向をフロアマップに合わせられる。

    Args:
        peaks (np.ndarray): ステップピークのインデックス配列
        df_gyro (pd.DataFrame): ``low_angle`` 列を含むジャイロスコープDataFrame
        df_acc (pd.DataFrame):
            ``v_acc``・``h_y``・``h_z``・``h_norm`` 列を含む加速度DataFrame
        initial_direction (float): 歩行開始方向のオフセット [度]
            （デフォルト: ``INITIAL_DIRECTION``）
        weinberg_k (float): Weinbergモデルのスケール係数

    Returns:
        tuple[list[list[float]], list[float], list[float]]:
            - 各ステップの [x, y] 座標リスト（原点を含む）
            - 各ステップの推定歩幅リスト [m]
            - 各移動後座標に対応する時刻リスト [s]
    """
    points: list[list[float]] = [[0.0, 0.0]]
    step_lengths: list[float] = []
    t_at_steps: list[float] = []
    step_headings: list[StepHeading] = []
    sidestep_lateral_ratio = _validate_positive_parameter(
        "sidestep_lateral_ratio",
        sidestep_lateral_ratio,
    )
    sidestep_min_lateral_displacement = _validate_non_negative_parameter(
        "sidestep_min_lateral_displacement",
        sidestep_min_lateral_displacement,
    )
    device_orientation_mode = _estimate_device_orientation_mode(
        df_acc,
        df_gyro,
        peaks,
        initial_direction,
        step_segments,
    )
    motion_heading_correction_rad = _resolve_motion_heading_correction(
        df_acc,
        df_gyro,
        peaks,
        initial_direction,
        step_segments,
        motion_heading_correction,
        device_orientation_mode,
    )
    selected_sidestep_smoothing = _validate_sidestep_smoothing(sidestep_smoothing)
    selected_forward_heading_source = _validate_forward_heading_source(
        forward_heading_source
    )
    selected_sidestep_heading_source = _validate_sidestep_heading_source(
        sidestep_heading_source
    )
    selected_sidestep_suspect_mode = _validate_sidestep_suspect_mode(
        sidestep_suspect_mode
    )

    # forward 手法用: 初期前進角をデータから自動推定
    phi_0 = (
        _estimate_initial_forward_angle(df_acc, df_gyro, peaks)
        if STEP_LENGTH_METHOD == "forward"
        else 0.0
    )
    raw_step_headings: list[StepHeading] = []
    raw_step_lengths: list[float] = []
    raw_step_times: list[float] = []
    previous_heading: float | None = None
    for i, p in enumerate(peaks):
        if p >= len(df_acc):
            continue
        if STEP_LENGTH_METHOD == "forward" and i + 1 >= len(peaks):
            continue  # 次ピークなし：区間定義不可のためスキップ
        step_heading = resolve_step_heading(
            peaks,
            df_gyro,
            df_acc,
            i,
            initial_direction=initial_direction,
            heading_method=heading_method,
            step_segments=step_segments,
            motion_heading_correction=motion_heading_correction_rad,
            sidestep_lateral_ratio=sidestep_lateral_ratio,
            sidestep_min_lateral_displacement=sidestep_min_lateral_displacement,
            device_orientation_mode=device_orientation_mode,
        )
        if step_heading.selected_heading is None:
            continue
        if STEP_LENGTH_METHOD == "forward":
            step_length = estimate_step_length_forward(df_acc, df_gyro, peaks, i, phi_0)
        else:
            step_length = estimate_step_length(df_acc, int(p), k=weinberg_k)
        raw_step_headings.append(step_heading)
        raw_step_lengths.append(step_length)
        raw_step_times.append(_step_output_time(df_acc, peaks, i))

    smoothed_step_headings = _smooth_step_headings(
        raw_step_headings,
        selected_sidestep_smoothing,
        selected_sidestep_suspect_mode,
    )
    stabilized_step_headings = _stabilize_trajectory_headings(
        smoothed_step_headings,
        selected_forward_heading_source,
        selected_sidestep_heading_source,
    )

    for step_heading, step_length, step_time in zip(
        stabilized_step_headings,
        raw_step_lengths,
        raw_step_times,
        strict=True,
    ):
        step_motion = estimate_step_motion(
            step_heading,
            step_length,
            previous_heading,
            selected_forward_heading_source,
            selected_sidestep_heading_source,
            selected_sidestep_suspect_mode,
        )
        if step_motion is None:
            continue
        step_heading = step_heading._replace(
            selected_heading=step_motion.heading,
            source=step_heading.source
            if step_heading.source.startswith("trajectory_")
            else "state_motion",
            step_length_scale=step_motion.length_scale,
            trajectory_movement_type=step_motion.movement_type,
            forward_heading_source=selected_forward_heading_source,
        )
        step_lengths.append(step_motion.length)
        t_at_steps.append(step_time)
        step_headings.append(step_heading)
        previous_heading = step_motion.heading
        x = points[-1][0] + step_motion.length * float(np.cos(step_motion.heading))
        y = points[-1][1] + step_motion.length * float(np.sin(step_motion.heading))
        points.append([x, y])

    return points, step_lengths, t_at_steps, step_headings


def estimate_trajectory(
    peaks: np.ndarray,
    df_gyro: pd.DataFrame,
    df_acc: pd.DataFrame,
    initial_direction: float = INITIAL_DIRECTION,
    weinberg_k: float = WEINBERG_K,
) -> tuple[list[list[float]], list[float], list[float]]:
    """従来互換の決定論的PDR軌跡推定を行う。"""
    points, step_lengths, t_at_steps, _step_headings = (
        estimate_trajectory_with_headings(
            peaks,
            df_gyro,
            df_acc,
            initial_direction=initial_direction,
            weinberg_k=weinberg_k,
            heading_method="gyro",
        )
    )
    return points, step_lengths, t_at_steps


def prepare_pdr_steps(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    initial_direction: float = INITIAL_DIRECTION,
    height_m: float = USER_HEIGHT_M,
    step_detection_method: str | None = None,
    heading_method: str | None = None,
    gyro_bias_method: str | None = None,
    gyro_bias: float | None = None,
    sidestep_lateral_ratio: float = SIDESTEP_LATERAL_RATIO,
    sidestep_min_lateral_displacement: float = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    motion_heading_correction: str = "auto",
    sidestep_smoothing: str = SIDESTEP_SMOOTHING_METHOD,
    forward_heading_source: str = FORWARD_HEADING_SOURCE,
    sidestep_heading_source: str = "motion",
    sidestep_suspect_mode: str = "motion",
) -> PreparedPdrSteps:
    """通常PDRとPFが共用するステップ単位の推定結果を作る。"""
    selected_gyro_bias_method = _validate_gyro_bias_method(
        GYRO_BIAS_METHOD if gyro_bias_method is None else gyro_bias_method
    )
    selected_heading_method = _validate_heading_method(
        HEADING_METHOD if heading_method is None else heading_method
    )
    selected_motion_heading_correction = _validate_motion_heading_correction(
        motion_heading_correction
    )
    selected_sidestep_smoothing = _validate_sidestep_smoothing(sidestep_smoothing)
    selected_forward_heading_source = _validate_forward_heading_source(
        forward_heading_source
    )
    selected_sidestep_heading_source = _validate_sidestep_heading_source(
        sidestep_heading_source
    )
    selected_sidestep_suspect_mode = _validate_sidestep_suspect_mode(
        sidestep_suspect_mode
    )
    sidestep_lateral_ratio = _validate_positive_parameter(
        "sidestep_lateral_ratio",
        sidestep_lateral_ratio,
    )
    sidestep_min_lateral_displacement = _validate_non_negative_parameter(
        "sidestep_min_lateral_displacement",
        sidestep_min_lateral_displacement,
    )

    processed_acc, processed_gyro = process_sensor_data(
        df_acc,
        df_gyro,
        gyro_bias_method=selected_gyro_bias_method,
        gyro_bias=gyro_bias,
    )
    step_detection = detect_step_result(processed_acc, step_detection_method)
    weinberg_k = compute_weinberg_k(height_m)
    trajectory, step_lengths, t_at_steps, step_headings = (
        estimate_trajectory_with_headings(
            step_detection.peaks,
            processed_gyro,
            processed_acc,
            initial_direction,
            weinberg_k,
            heading_method=selected_heading_method,
            step_segments=step_detection.segments,
            sidestep_lateral_ratio=sidestep_lateral_ratio,
            sidestep_min_lateral_displacement=sidestep_min_lateral_displacement,
            motion_heading_correction=selected_motion_heading_correction,
            sidestep_smoothing=selected_sidestep_smoothing,
            forward_heading_source=selected_forward_heading_source,
            sidestep_heading_source=selected_sidestep_heading_source,
            sidestep_suspect_mode=selected_sidestep_suspect_mode,
        )
    )

    return PreparedPdrSteps(
        df_acc=processed_acc,
        df_gyro=processed_gyro,
        step_detection=step_detection,
        trajectory=trajectory,
        step_lengths=step_lengths,
        t_at_steps=t_at_steps,
        step_headings=step_headings,
        gx_mean=float(processed_acc["gx"].mean()),
        gz_mean=float(processed_acc["gz"].mean()),
        weinberg_k=weinberg_k,
        heading_method=selected_heading_method,
        motion_heading_correction=selected_motion_heading_correction,
        sidestep_smoothing=selected_sidestep_smoothing,
        forward_heading_source=selected_forward_heading_source,
        sidestep_heading_source=selected_sidestep_heading_source,
        sidestep_suspect_mode=selected_sidestep_suspect_mode,
    )


def run(
    df_acc: pd.DataFrame | None = None,
    df_gyro: pd.DataFrame | None = None,
    plot: bool = True,
    use_particle_filter: bool = False,
    save_animation: bool | None = None,
    floormap_path: str | Path = FLOORMAP_PATH,
    origin_px: tuple[int, int] = FLOORMAP_ORIGIN_PX,
    scale: float = FLOORMAP_SCALE,
    initial_direction: float = INITIAL_DIRECTION,
    height_m: float = USER_HEIGHT_M,
    step_detection_method: str | None = None,
    heading_method: str | None = None,
    gyro_bias_method: str | None = None,
    gyro_bias: float | None = None,
    sidestep_lateral_ratio: float = SIDESTEP_LATERAL_RATIO,
    sidestep_min_lateral_displacement: float = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    motion_heading_correction: str = "auto",
    sidestep_smoothing: str = SIDESTEP_SMOOTHING_METHOD,
    forward_heading_source: str = FORWARD_HEADING_SOURCE,
    sidestep_heading_source: str = "motion",
    sidestep_suspect_mode: str = "motion",
) -> pd.DataFrame:
    """PDRのメインパイプラインを実行する。

    センサーデータの読み込みから軌跡の推定・CSV保存・表示までを一括して実行する。
    処理の流れ: データ読み込み → センサー処理 → ステップ検出 → 軌跡推定 → CSV保存 → 表示

    Args:
        df_acc (pd.DataFrame | None):
            加速度データ（列: t, x, y, z）。省略時は ``DATA_DIR`` の CSV から読み込む。
            渡す場合は ``load_sensor_data()`` によるリネーム後の列名
            （t, x, y, z）を使うこと。
        df_gyro (pd.DataFrame | None):
            ジャイロスコープデータ（列: t, x, y, z）。
            省略時は ``DATA_DIR`` の CSV から読み込む。
            ``df_acc`` と必ずセットで渡すこと。
        plot (bool):
            ``True`` のとき軌跡をプロット表示する。
            バッチ処理やCI環境では ``False`` を指定する。デフォルトは ``True``。
        use_particle_filter (bool):
            ``True`` のときパーティクルフィルタで軌跡を推定する。
            デフォルトは ``False``。
        save_animation (bool | None):
            パーティクルフィルタのアニメーション保存を制御する。
            ``None`` のときは ``plot`` と同じ値を使う。
        floormap_path (str | Path):
            フロアマップ画像のパス。デフォルトは ``FLOORMAP_PATH``。
        origin_px (tuple[int, int]):
            軌跡起点のピクセル座標 ``(x, y)``。デフォルトは ``FLOORMAP_ORIGIN_PX``。
        scale (float):
            1ピクセルあたりのメートル数。デフォルトは ``FLOORMAP_SCALE``。
        initial_direction (float):
            歩行開始方向のオフセット [度]。デフォルトは ``INITIAL_DIRECTION``。
        height_m (float):
            Weinbergモデルのスケール係数を補正するユーザー身長 [m]。
        step_detection_method:
            ステップ検出手法。``None`` のときは設定値を使用する。
        heading_method:
            方位推定手法。``None`` のときは設定値を使用する。
        gyro_bias_method:
            ジャイロバイアス推定手法。``None`` のときは設定値を使用する。
        gyro_bias:
            ``gyro_bias_method="manual"`` のときに使う手動バイアス [rad/s]。
        sidestep_lateral_ratio:
            横歩き判定に使う横方向/前方向の最小比率。
        sidestep_min_lateral_displacement:
            横歩き判定に必要な横方向変位の最小値 [m]。
        motion_heading_correction:
            水平加速度移動方向の固定ずれ補正モード（``"auto"`` or ``"none"``）。
        sidestep_smoothing:
            横歩き判定の平滑化モード
            （``"none"``、``"isolated"``、``"clustered"``）。
        forward_heading_source:
            forward 判定ステップの軌跡方位ソース（``"body"`` or ``"motion"``）。
        sidestep_heading_source:
            確定横歩きステップの軌跡方位ソース。
        sidestep_suspect_mode:
            横歩き疑いステップの軌跡反映モード。
    Returns:
        pd.DataFrame: 軌跡データ（列: timestamp_s, x, y）

    Raises:
        ValueError: ``df_acc`` と ``df_gyro`` の片方だけが渡された場合
    """
    _validate_scale(scale)
    sidestep_lateral_ratio = _validate_positive_parameter(
        "sidestep_lateral_ratio",
        sidestep_lateral_ratio,
    )
    sidestep_min_lateral_displacement = _validate_non_negative_parameter(
        "sidestep_min_lateral_displacement",
        sidestep_min_lateral_displacement,
    )
    selected_motion_heading_correction = _validate_motion_heading_correction(
        motion_heading_correction
    )
    selected_sidestep_smoothing = _validate_sidestep_smoothing(sidestep_smoothing)
    selected_forward_heading_source = _validate_forward_heading_source(
        forward_heading_source
    )
    selected_sidestep_heading_source = _validate_sidestep_heading_source(
        sidestep_heading_source
    )
    selected_sidestep_suspect_mode = _validate_sidestep_suspect_mode(
        sidestep_suspect_mode
    )
    output_dir = _create_output_dir()
    should_save_animation = plot if save_animation is None else save_animation

    if (df_acc is None) != (df_gyro is None):
        raise ValueError("df_acc と df_gyro は両方渡すか、両方省略してください。")

    if df_acc is None and df_gyro is None:
        df_acc, df_gyro = load_sensor_data()

    if df_acc is None or df_gyro is None:
        raise RuntimeError("内部エラー: df_acc または df_gyro が None（到達不能）")

    prepared_steps = prepare_pdr_steps(
        df_acc,
        df_gyro,
        initial_direction=initial_direction,
        height_m=height_m,
        step_detection_method=step_detection_method,
        heading_method=heading_method,
        gyro_bias_method=gyro_bias_method,
        gyro_bias=gyro_bias,
        sidestep_lateral_ratio=sidestep_lateral_ratio,
        sidestep_min_lateral_displacement=sidestep_min_lateral_displacement,
        motion_heading_correction=selected_motion_heading_correction,
        sidestep_smoothing=selected_sidestep_smoothing,
        forward_heading_source=selected_forward_heading_source,
        sidestep_heading_source=selected_sidestep_heading_source,
        sidestep_suspect_mode=selected_sidestep_suspect_mode,
    )
    df_acc = prepared_steps.df_acc
    df_gyro = prepared_steps.df_gyro
    step_detection = prepared_steps.step_detection
    peaks = step_detection.peaks
    weinberg_k = prepared_steps.weinberg_k
    selected_heading_method = prepared_steps.heading_method
    print(f"Weinberg K: {weinberg_k:.3f} (height={height_m:.2f} m)")
    print(f"Heading method: {selected_heading_method}")
    print(
        "Sidestep detection: "
        f"ratio={sidestep_lateral_ratio:.3f} "
        f"min_lateral={sidestep_min_lateral_displacement:.3f} m "
        f"motion_heading_correction={selected_motion_heading_correction} "
        f"smoothing={selected_sidestep_smoothing} "
        f"forward_heading_source={selected_forward_heading_source} "
        f"sidestep_heading_source={selected_sidestep_heading_source} "
        f"sidestep_suspect_mode={selected_sidestep_suspect_mode}"
    )
    bias_result = df_gyro.attrs.get("gyro_bias_result")
    if isinstance(bias_result, GyroBiasResult):
        if bias_result.calibration_start_s is None:
            window_text = "manual"
        else:
            window_text = (
                f"{bias_result.calibration_start_s:.3f}-"
                f"{bias_result.calibration_end_s:.3f}s"
            )
        print(
            "Gyro bias: "
            f"method={bias_result.method} "
            f"bias={bias_result.bias_rad_s:.6f} rad/s "
            f"window={window_text} "
            f"fallback={bias_result.fallback_reason}"
        )
    if step_detection.threshold is None:
        print(f"Step detection: {step_detection.method}")
    else:
        print(
            "Step detection: "
            f"{step_detection.method} "
            f"threshold={step_detection.threshold:.3f} "
            f"polarity={step_detection.polarity}"
        )

    # 重力成分の平均を算出（Y軸反転の自動判定に使用）
    gx_mean = prepared_steps.gx_mean
    gz_mean = prepared_steps.gz_mean
    dominant = "X軸" if abs(gx_mean) > abs(gz_mean) else "Z軸"
    y_flipped = (abs(gx_mean) > abs(gz_mean) and gx_mean > 0) or (
        abs(gz_mean) >= abs(gx_mean) and gz_mean < 0
    )
    print(
        f"重力主成分: {dominant}  gx={gx_mean:.2f}, gz={gz_mean:.2f} m/s²"
        f" → Y軸{'反転' if y_flipped else '非反転'}"
    )

    df_gyro_bias = _build_gyro_bias_dataframe(df_gyro)
    gyro_bias_path = output_dir / "gyro_bias.csv"
    df_gyro_bias.to_csv(gyro_bias_path, index=False)
    print(f"Gyro bias saved to {gyro_bias_path}")

    if use_particle_filter:
        from ..particle_filter import (  # noqa: PLC0415
            plot_particle_filter_trajectory,
            run_particle_filter,
            save_particle_animation,
        )

        (
            trajectory,
            step_lengths,
            t_at_steps,
            all_particles,
            step_headings,
        ) = run_particle_filter(
            peaks,
            df_gyro,
            df_acc,
            gx_mean,
            gz_mean,
            floormap_path=floormap_path,
            origin_px=origin_px,
            scale=scale,
            initial_direction=initial_direction,
            weinberg_k=weinberg_k,
            heading_method=selected_heading_method,
            step_segments=step_detection.segments,
            prepared_step_headings=prepared_steps.step_headings,
            prepared_step_lengths=prepared_steps.step_lengths,
            prepared_step_times=prepared_steps.t_at_steps,
            sidestep_lateral_ratio=sidestep_lateral_ratio,
            sidestep_min_lateral_displacement=sidestep_min_lateral_displacement,
            motion_heading_correction=selected_motion_heading_correction,
            sidestep_smoothing=selected_sidestep_smoothing,
            forward_heading_source=selected_forward_heading_source,
            sidestep_heading_source=selected_sidestep_heading_source,
            sidestep_suspect_mode=selected_sidestep_suspect_mode,
        )

        print(f"Peaks detected: {len(peaks)}")
        print(f"Steps used: {len(step_lengths)}")
        for i, (x, y) in enumerate(trajectory):
            print(f"step {i}: ({x:.3f}, {y:.3f})")

        df_trajectory = _build_trajectory_dataframe(
            trajectory,
            t_at_steps,
        )
        output_path = output_dir / "trajectory.csv"
        df_trajectory.to_csv(output_path, index=False)
        print(f"Trajectory saved to {output_path}")

        df_step_lengths = pd.DataFrame(
            {"step": range(1, len(step_lengths) + 1), "step_length_m": step_lengths}
        )
        step_length_path = output_dir / "step_lengths.csv"
        df_step_lengths.to_csv(step_length_path, index=False)
        print(f"Step lengths saved to {step_length_path}")

        df_step_vectors = _build_step_vectors_dataframe(trajectory)
        step_vector_path = output_dir / "step_vectors.csv"
        df_step_vectors.to_csv(step_vector_path, index=False)
        print(f"Step vectors saved to {step_vector_path}")

        df_step_headings = _build_step_headings_dataframe(step_headings)
        step_heading_path = output_dir / "step_headings.csv"
        df_step_headings.to_csv(step_heading_path, index=False)
        print(f"Step headings saved to {step_heading_path}")

        if step_detection.method == "paper_vertical_threshold":
            df_step_segments = _build_step_segments_dataframe(
                df_acc,
                step_detection.segments,
            )
            step_segment_path = output_dir / "step_segments.csv"
            df_step_segments.to_csv(step_segment_path, index=False)
            print(f"Step segments saved to {step_segment_path}")

        if plot:
            plot_particle_filter_trajectory(
                trajectory,
                gx_mean=gx_mean,
                gz_mean=gz_mean,
                floormap_path=floormap_path,
                origin_px=origin_px,
                scale=scale,
                output_dir=output_dir,
                step_headings=step_headings,
            )
            from ..sensor_plot import (  # noqa: PLC0415
                plot_step_lengths,
                plot_step_vectors,
            )

            t_acc = (
                df_acc["t"].to_numpy()
                if "t" in df_acc.columns
                else np.arange(len(df_acc)) / SAMPLING_RATE
            )
            step_signal, step_signal_label, step_signal_threshold = _step_plot_signal(
                df_acc,
                step_detection,
            )
            plot_step_lengths(
                step_lengths,
                output_dir,
                t_at_steps=t_at_steps,
                t_acc=t_acc,
                step_signal=step_signal,
                step_signal_label=step_signal_label,
                step_signal_threshold=step_signal_threshold,
            )
            plot_step_vectors(
                trajectory,
                output_dir,
                df_acc=df_acc,
                df_gyro=df_gyro,
                peaks=peaks,
                step_headings=step_headings,
                initial_direction=initial_direction,
            )

        if should_save_animation:
            save_particle_animation(
                all_particles,
                trajectory,
                gx_mean=gx_mean,
                gz_mean=gz_mean,
                floormap_path=floormap_path,
                origin_px=origin_px,
                scale=scale,
                output_path=output_dir / "particle_filter.mp4",
            )
    else:
        trajectory = prepared_steps.trajectory
        step_lengths = prepared_steps.step_lengths
        t_at_steps = prepared_steps.t_at_steps
        step_headings = prepared_steps.step_headings

        print(f"Peaks detected: {len(peaks)}")
        print(f"Steps used: {len(step_lengths)}")
        for i, (x, y) in enumerate(trajectory):
            print(f"step {i}: ({x:.3f}, {y:.3f})")

        df_trajectory = _build_trajectory_dataframe(
            trajectory,
            t_at_steps,
        )

        # 軌跡データをoutputフォルダにCSVとして保存
        output_path = output_dir / "trajectory.csv"
        df_trajectory.to_csv(output_path, index=False)
        print(f"Trajectory saved to {output_path}")

        # 歩幅データをoutputフォルダにCSVとして保存
        df_step_lengths = pd.DataFrame(
            {"step": range(1, len(step_lengths) + 1), "step_length_m": step_lengths}
        )
        step_length_path = output_dir / "step_lengths.csv"
        df_step_lengths.to_csv(step_length_path, index=False)
        print(f"Step lengths saved to {step_length_path}")

        df_step_vectors = _build_step_vectors_dataframe(trajectory)
        step_vector_path = output_dir / "step_vectors.csv"
        df_step_vectors.to_csv(step_vector_path, index=False)
        print(f"Step vectors saved to {step_vector_path}")

        df_step_headings = _build_step_headings_dataframe(step_headings)
        step_heading_path = output_dir / "step_headings.csv"
        df_step_headings.to_csv(step_heading_path, index=False)
        print(f"Step headings saved to {step_heading_path}")

        if step_detection.method == "paper_vertical_threshold":
            df_step_segments = _build_step_segments_dataframe(
                df_acc,
                step_detection.segments,
            )
            step_segment_path = output_dir / "step_segments.csv"
            df_step_segments.to_csv(step_segment_path, index=False)
            print(f"Step segments saved to {step_segment_path}")

        if plot:
            plot_trajectory(
                trajectory,
                gx_mean=gx_mean,
                gz_mean=gz_mean,
                floormap_path=floormap_path,
                origin_px=origin_px,
                scale=scale,
                output_dir=output_dir,
                step_headings=step_headings,
            )
            from ..sensor_plot import (  # noqa: PLC0415
                plot_step_lengths,
                plot_step_vectors,
            )

            t_acc = (
                df_acc["t"].to_numpy()
                if "t" in df_acc.columns
                else np.arange(len(df_acc)) / SAMPLING_RATE
            )
            step_signal, step_signal_label, step_signal_threshold = _step_plot_signal(
                df_acc,
                step_detection,
            )
            plot_step_lengths(
                step_lengths,
                output_dir,
                t_at_steps=t_at_steps,
                t_acc=t_acc,
                step_signal=step_signal,
                step_signal_label=step_signal_label,
                step_signal_threshold=step_signal_threshold,
            )
            plot_step_vectors(
                trajectory,
                output_dir,
                df_acc=df_acc,
                df_gyro=df_gyro,
                peaks=peaks,
                step_headings=step_headings,
                initial_direction=initial_direction,
            )

    return df_trajectory
