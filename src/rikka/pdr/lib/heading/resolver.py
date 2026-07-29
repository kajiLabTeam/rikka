"""PDR のステップ方位推定。

役割:
    1歩ごとにジャイロ由来の体方位、加速度ピーク方位、水平加速度を積分した
    移動方位を推定し、候補と信頼度を ``StepHeading`` にまとめる。
依存元:
    ``config`` の閾値、``common`` の角度・検証処理、``models`` の共有型、
    ``time_utils`` の時刻補間を利用し、NumPy、Pandas、SciPy で信号を解析する。
利用先:
    ``trajectory`` と ``particle_api`` がステップ方位候補の生成に使用し、
    ``sidestep`` が候補を移動状態と軌跡方位へ変換する。
処理フロー:
    ステップ区間を決め、端末姿勢と各方式の方位・変位・信頼度を計算し、指定方式に
    応じた候補を返す。横歩きの最終採否と軌跡反映はここでは行わない。
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ....analyze.pdr.common import (
    DEVICE_ORIENTATION_MODES,
    _abs_angle_diff,
    _normalize_angle,
    _score_ratio,
    _validate_heading_method,
    _validate_motion_heading_correction,
    _validate_non_negative_parameter,
    _validate_positive_parameter,
)
from ....common.config import (
    ACCEL_HEADING_MIN_LINE_LENGTH,
    ACCEL_HEADING_MIN_PEAK_DISTANCE,
    ACCEL_HEADING_MIN_PEAK_NORM,
    HEADING_METHOD,
    INITIAL_DIRECTION,
    MAX_SEG_SAMPLES,
    MIN_SEG_SAMPLES,
    MOTION_HEADING_CALIBRATION_STEPS,
    MOTION_HEADING_CONFIDENCE_THRESHOLD,
    MOTION_HEADING_MIN_DISPLACEMENT_M,
    SAMPLING_RATE,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    STEP_LENGTH_WINDOW,
    TURNING_YAW_DELTA_THRESHOLD_DEG,
)
from ....common.lib.models import StepHeading, StepSegment
from ....common.lib.time_utils import (
    _sample_gyro_angle,
    _step_mid_index,
    _step_mid_time,
    _step_output_time,
    _time_values,
)


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

    # 端末座標の水平加速度を取り出し、推定済みの装着向き補正を適用する。
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

    # ジャイロ角を加速度サンプル時刻へ補間し、水平加速度を世界座標へ回す。
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
    # 体の前方軸・左右軸に射影し、forward / sidestep / turning を分類する。
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
