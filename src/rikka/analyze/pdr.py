from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.signal import find_peaks

from ..config import (
    ACCEL_HEADING_MIN_LINE_LENGTH,
    ACCEL_HEADING_MIN_PEAK_DISTANCE,
    ACCEL_HEADING_MIN_PEAK_NORM,
    BACKWARD_LENGTH_SCALE,
    DATA_DIR,
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    FORWARD_HEADING_SOURCE,
    GYRO_BIAS_METHOD,
    GYRO_BIAS_MIN_CALIBRATION_SECONDS,
    GYRO_BIAS_OUTLIER_MAD_SCALE,
    GYRO_BIAS_STATIC_ACCEL_P95_WEIGHT,
    GYRO_BIAS_STATIC_GYRO_STD_WEIGHT,
    GYRO_BIAS_STATIC_MAX_ACCEL_P95,
    GYRO_BIAS_STATIC_MAX_GYRO_STD,
    GYRO_BIAS_STATIC_SEARCH_END_SECONDS,
    GYRO_BIAS_STATIC_SEARCH_START_SECONDS,
    GYRO_BIAS_STATIC_WALK_ONSET_MARGIN_S,
    GYRO_BIAS_STATIC_WINDOW_SECONDS,
    GYRO_BIAS_STATIC_WINDOW_STEP_SECONDS,
    GYRO_BIAS_WALK_ONSET_MAX_INTERVAL_S,
    GYRO_BIAS_WALK_ONSET_MIN_STEPS,
    HEADING_METHOD,
    INITIAL_DIRECTION,
    K_FORWARD,
    MAX_SEG_SAMPLES,
    MIN_SEG_SAMPLES,
    MOTION_HEADING_CALIBRATION_STEPS,
    MOTION_HEADING_CONFIDENCE_THRESHOLD,
    MOTION_HEADING_MIN_DISPLACEMENT_M,
    PEAK_DISTANCE,
    PEAK_HEIGHT,
    SAMPLING_RATE,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_LENGTH_SCALE,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    SIDESTEP_SMOOTHING_METHOD,
    STEP_DETECTION_METHOD,
    STEP_LENGTH_METHOD,
    STEP_LENGTH_WINDOW,
    STEP_VERTICAL_SMOOTH_WINDOW,
    STEP_VERTICAL_THRESHOLD_PERCENTILE,
    TURNING_LENGTH_SCALE,
    TURNING_YAW_DELTA_THRESHOLD_DEG,
    USER_HEIGHT_M,
    WEINBERG_K,
    WINDOW_ACC,
    WINDOW_GYRO,
    compute_weinberg_k,
)

# Column name mappings from phyphox CSV format
ACC_COLUMNS = {
    "Time (s)": "t",
    "Acceleration x (m/s^2)": "x",
    "Acceleration y (m/s^2)": "y",
    "Acceleration z (m/s^2)": "z",
    "X (m/s^2)": "x",
    "Y (m/s^2)": "y",
    "Z (m/s^2)": "z",
}
GYRO_COLUMNS = {
    "Time (s)": "t",
    "Gyroscope x (rad/s)": "x",
    "Gyroscope y (rad/s)": "y",
    "Gyroscope z (rad/s)": "z",
    "X (rad/s)": "x",
    "Y (rad/s)": "y",
    "Z (rad/s)": "z",
}

STEP_DETECTION_METHODS = ("peak", "paper_vertical_threshold")
HEADING_METHODS = (
    "gyro",
    "accel_method1",
    "accel_method2",
    "gyro_accel_motion",
)
GYRO_BIAS_METHODS = ("prewalk_robust", "initial_robust", "quietest", "manual")
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


class StepSegment(NamedTuple):
    """1歩区間を表すインデックス範囲。"""

    start_index: int
    end_index: int
    contact_index: int


class StepDetectionResult(NamedTuple):
    """ステップ検出結果。既存互換用ピーク列と論文方式用区間を併せて持つ。"""

    method: str
    peaks: np.ndarray
    segments: tuple[StepSegment, ...]
    threshold: float | None
    polarity: int | None


class StepHeading(NamedTuple):
    """1歩ごとの方位候補と採用結果。角度はすべてラジアン。"""

    step_index: int
    timestamp_s: float
    gyro_heading: float | None
    accel_method1_heading: float | None
    accel_method2_heading: float | None
    selected_heading: float | None
    source: str
    confidence: float
    angle_diff_method1: float | None
    angle_diff_method2: float | None
    segment_start_index: int | None
    segment_end_index: int | None
    peak1_index: int | None
    peak2_index: int | None
    body_heading: float | None
    motion_heading: float | None
    movement_type: str
    forward_displacement: float | None
    lateral_displacement: float | None
    motion_confidence: float
    motion_reject_reason: str | None
    trajectory_movement_type: str | None = None
    step_length_scale: float = 1.0
    yaw_delta: float | None = None
    motion_heading_correction: float = 0.0
    sidestep_lateral_ratio: float = SIDESTEP_LATERAL_RATIO
    sidestep_min_lateral_displacement: float = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M
    forward_heading_source: str = FORWARD_HEADING_SOURCE
    body_motion_angle_diff: float | None = None
    sidestep_evidence_direction: str | None = None
    sidestep_evidence_reason: str | None = None
    sidestep_cluster_id: int | None = None
    device_orientation_mode: str = "normal"


class GyroBiasResult(NamedTuple):
    """ジャイロバイアス推定結果と診断情報。"""

    method: str
    bias_rad_s: float
    calibration_start_s: float | None
    calibration_end_s: float | None
    sample_count: int
    kept_sample_count: int
    raw_mean: float | None
    robust_mean: float | None
    median: float | None
    mad: float | None
    candidate_score: float | None
    gyro_std: float | None
    accel_p95: float | None
    accel_max: float | None
    search_start_s: float | None
    search_end_s: float | None
    fallback_reason: str | None


class _GyroBiasStaticCandidate(NamedTuple):
    """gyro bias 推定に使う静止候補窓。"""

    start_s: float
    end_s: float
    score: float
    gyro_std: float
    accel_p95: float
    accel_max: float


def load_sensor_data(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CSVファイルから加速度計とジャイロスコープのデータを読み込む。

    ``data_dir`` が指定された場合はそのディレクトリを，
    省略時は ``DATA_DIR`` を使用する。
    ``Accelerometer.csv`` と ``Gyroscope.csv`` を読み込み，
    列名を統一した形式に変換して返す。

    Args:
        data_dir (str | Path | None):
            CSVファイルが格納されたディレクトリパス。省略時は ``DATA_DIR`` を使用。

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - df_acc: 加速度データ（列: t, x, y, z）
            - df_gyro: ジャイロスコープデータ（列: t, x, y, z）
    """
    data_path = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    df_acc = pd.read_csv(data_path / "Accelerometer.csv").rename(columns=ACC_COLUMNS)
    df_gyro = pd.read_csv(data_path / "Gyroscope.csv").rename(columns=GYRO_COLUMNS)

    # 必須列の存在確認（列名揺れや欠損時に後段で KeyError になるのを防ぐ）
    _required = {"x", "y", "z"}
    missing_acc = _required - set(df_acc.columns)
    if missing_acc:
        raise ValueError(f"Accelerometer.csv に必須列がありません: {missing_acc}")
    missing_gyro = _required - set(df_gyro.columns)
    if missing_gyro:
        raise ValueError(f"Gyroscope.csv に必須列がありません: {missing_gyro}")

    return df_acc, df_gyro


def _time_values(df: pd.DataFrame) -> np.ndarray | None:
    """単調増加する時刻配列を返す。利用できない場合は None を返す。"""
    if "t" not in df.columns:
        return None
    times = np.asarray(pd.to_numeric(df["t"], errors="coerce"), dtype=float)
    if len(times) == 0 or not np.isfinite(times).all():
        return None
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        return None
    return times


def _gyro_integration_dt(df_gyro: pd.DataFrame) -> np.ndarray:
    """ジャイロ積分用のサンプル間隔を返す。時刻列がなければ固定周期を使う。"""
    times = _time_values(df_gyro)
    if times is None or len(times) < 2:
        return np.full(len(df_gyro), 1.0 / SAMPLING_RATE)

    dt = np.diff(times, prepend=times[0])
    dt[0] = 0.0
    return dt


def _time_at_index(df: pd.DataFrame, index: int) -> float:
    """DataFrame の index 位置に対応する時刻を返す。"""
    times = _time_values(df)
    if times is None:
        return index / SAMPLING_RATE
    clipped_index = min(max(index, 0), len(times) - 1)
    return float(times[clipped_index])


def _step_mid_index(peaks: np.ndarray, i: int) -> int:
    """現在ステップの方位サンプリングに使う中点 index を返す。"""
    start = int(peaks[i])
    if i + 1 < len(peaks):
        return (start + int(peaks[i + 1])) // 2
    return start


def _step_mid_time(df_acc: pd.DataFrame, peaks: np.ndarray, i: int) -> float:
    """現在ステップの方位サンプリングに使う中点時刻を返す。"""
    start = int(peaks[i])
    if i + 1 < len(peaks):
        end = int(peaks[i + 1])
        return (_time_at_index(df_acc, start) + _time_at_index(df_acc, end)) / 2
    return _time_at_index(df_acc, start)


def _step_output_time(
    df_acc: pd.DataFrame,
    peaks: np.ndarray,
    i: int,
    step_length_method: str | None = None,
) -> float:
    """移動後座標に対応する時刻を返す。"""
    method = STEP_LENGTH_METHOD if step_length_method is None else step_length_method
    if method == "forward" and i + 1 < len(peaks):
        return _time_at_index(df_acc, int(peaks[i + 1]))
    return _time_at_index(df_acc, int(peaks[i]))


def _sample_gyro_angle(
    df_gyro: pd.DataFrame,
    sample_index: int,
    sample_time: float | None = None,
) -> float | None:
    """指定時刻の low_angle を補間して返す。時刻列がなければ index で取得する。"""
    low_angle = df_gyro["low_angle"].to_numpy(dtype=float)
    if sample_time is not None:
        times = _time_values(df_gyro)
        if times is not None:
            valid = np.isfinite(low_angle)
            if valid.any():
                return float(np.interp(sample_time, times[valid], low_angle[valid]))

    if len(low_angle) == 0:
        return None
    clipped_index = min(max(sample_index, 0), len(low_angle) - 1)
    angle = float(low_angle[clipped_index])
    if not np.isfinite(angle):
        return None
    return angle


def _create_output_dir(
    base_dir: str | Path = "output",
    now: datetime | None = None,
) -> Path:
    """衝突しないタイムスタンプ付き出力ディレクトリを作成して返す。"""
    current = now if now is not None else datetime.now()
    timestamp = current.strftime("%Y%m%d_%H%M%S_%f")
    base_path = Path(base_dir)

    for counter in range(1000):
        suffix = "" if counter == 0 else f"_{counter:03d}"
        output_dir = base_path / f"{timestamp}{suffix}"
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return output_dir

    raise FileExistsError(f"出力ディレクトリ名が衝突しました: {base_path / timestamp}")


def _validate_scale(scale: float) -> None:
    """フロアマップ縮尺が正の値であることを確認する。"""
    if scale <= 0:
        raise ValueError("scale は正の値を指定してください。")


def _validate_gyro_bias_method(method: str) -> str:
    """ジャイロバイアス推定手法名を検証する。"""
    if method not in GYRO_BIAS_METHODS:
        allowed = ", ".join(GYRO_BIAS_METHODS)
        raise ValueError(
            f"gyro_bias_method は {allowed} のいずれかを指定してください。"
        )
    return method


def _time_mask(df: pd.DataFrame, start_s: float, end_s: float) -> np.ndarray:
    """指定時刻範囲に含まれる行を表す mask を返す。"""
    times = _time_values(df)
    if times is None:
        times = np.arange(len(df), dtype=float) / SAMPLING_RATE
    return (times >= start_s) & (times <= end_s)


def _robust_gyro_bias_from_mask(
    df_gyro: pd.DataFrame,
    mask: np.ndarray,
    method: str,
    fallback_reason: str | None = None,
    candidate: _GyroBiasStaticCandidate | None = None,
    search_start_s: float | None = None,
    search_end_s: float | None = None,
) -> GyroBiasResult | None:
    """mask で指定した区間から外れ値に強い gyro bias を推定する。"""
    if len(mask) != len(df_gyro) or not mask.any():
        return None

    values = np.asarray(pd.to_numeric(df_gyro["x"], errors="coerce"), dtype=float)
    segment = values[mask]
    segment = segment[np.isfinite(segment)]
    if len(segment) == 0:
        return None

    median = float(np.nanmedian(segment))
    mad = float(np.nanmedian(np.abs(segment - median)))
    if mad <= 1e-12:
        keep = np.isfinite(segment) & (np.abs(segment - median) <= 1e-12)
    else:
        robust_sigma = 1.4826 * mad
        keep = np.abs(segment - median) <= GYRO_BIAS_OUTLIER_MAD_SCALE * robust_sigma
    kept = segment[keep]
    if len(kept) == 0:
        return None

    times = _time_values(df_gyro)
    if times is None:
        times = np.arange(len(df_gyro), dtype=float) / SAMPLING_RATE
    selected_times = times[mask]
    return GyroBiasResult(
        method=method,
        bias_rad_s=float(np.nanmean(kept)),
        calibration_start_s=float(selected_times[0]) if len(selected_times) else None,
        calibration_end_s=float(selected_times[-1]) if len(selected_times) else None,
        sample_count=int(len(segment)),
        kept_sample_count=int(len(kept)),
        raw_mean=float(np.nanmean(segment)),
        robust_mean=float(np.nanmean(kept)),
        median=median,
        mad=mad,
        candidate_score=None if candidate is None else candidate.score,
        gyro_std=None if candidate is None else candidate.gyro_std,
        accel_p95=None if candidate is None else candidate.accel_p95,
        accel_max=None if candidate is None else candidate.accel_max,
        search_start_s=search_start_s,
        search_end_s=search_end_s,
        fallback_reason=fallback_reason,
    )


def _estimate_gyro_bias_quietest(
    df_gyro: pd.DataFrame,
    fallback_reason: str | None = None,
) -> GyroBiasResult:
    """既存方式: 全期間で分散最小の窓から gyro bias を推定する。"""
    rolling_var = df_gyro["x"].rolling(window=WINDOW_GYRO).var()
    values = np.asarray(pd.to_numeric(df_gyro["x"], errors="coerce"), dtype=float)
    times = _time_values(df_gyro)
    if times is None:
        times = np.arange(len(df_gyro), dtype=float) / SAMPLING_RATE

    if rolling_var.notna().any():
        quiet_end = int(rolling_var.idxmin())
        quiet_start = max(0, quiet_end - WINDOW_GYRO + 1)
    else:
        quiet_start = 0
        quiet_end = len(values) - 1
        fallback_reason = "quietest_all_samples"

    segment = values[quiet_start : quiet_end + 1]
    segment = segment[np.isfinite(segment)]
    if len(segment) == 0:
        return GyroBiasResult(
            method="quietest",
            bias_rad_s=0.0,
            calibration_start_s=None,
            calibration_end_s=None,
            sample_count=0,
            kept_sample_count=0,
            raw_mean=None,
            robust_mean=None,
            median=None,
            mad=None,
            candidate_score=None,
            gyro_std=None,
            accel_p95=None,
            accel_max=None,
            search_start_s=None,
            search_end_s=None,
            fallback_reason="no_valid_gyro",
        )

    raw_mean = float(np.nanmean(segment))
    median = float(np.nanmedian(segment))
    mad = float(np.nanmedian(np.abs(segment - median)))
    return GyroBiasResult(
        method="quietest",
        bias_rad_s=raw_mean,
        calibration_start_s=float(times[quiet_start]) if len(times) else None,
        calibration_end_s=float(times[quiet_end]) if len(times) else None,
        sample_count=int(len(segment)),
        kept_sample_count=int(len(segment)),
        raw_mean=raw_mean,
        robust_mean=raw_mean,
        median=median,
        mad=mad,
        candidate_score=None,
        gyro_std=None,
        accel_p95=None,
        accel_max=None,
        search_start_s=None,
        search_end_s=None,
        fallback_reason=fallback_reason,
    )


def _estimate_gyro_bias_initial_robust(
    df_gyro: pd.DataFrame,
    fallback_reason: str | None = None,
) -> GyroBiasResult | None:
    """記録先頭の短い区間から gyro bias をロバスト推定する。"""
    times = _time_values(df_gyro)
    if times is None:
        start_s = 0.0
    elif len(times) == 0:
        return None
    else:
        start_s = float(times[0])
    end_s = start_s + GYRO_BIAS_MIN_CALIBRATION_SECONDS
    return _robust_gyro_bias_from_mask(
        df_gyro,
        _time_mask(df_gyro, start_s, end_s),
        method="initial_robust",
        fallback_reason=fallback_reason,
    )


def _find_walk_onset_time(df_acc: pd.DataFrame) -> float | None:
    """連続したステップ候補の先頭時刻を歩行開始として返す。"""
    if "low_lin_norm" not in df_acc.columns:
        return None
    peaks, _ = find_peaks(
        df_acc["low_lin_norm"].to_numpy(dtype=float),
        distance=PEAK_DISTANCE,
        height=PEAK_HEIGHT,
    )
    if len(peaks) < GYRO_BIAS_WALK_ONSET_MIN_STEPS:
        return None

    peak_times = np.asarray([_time_at_index(df_acc, int(peak)) for peak in peaks])
    window = GYRO_BIAS_WALK_ONSET_MIN_STEPS
    for start in range(0, len(peak_times) - window + 1):
        intervals = np.diff(peak_times[start : start + window])
        if np.all(intervals <= GYRO_BIAS_WALK_ONSET_MAX_INTERVAL_S):
            return float(peak_times[start])
    return None


def _find_static_gyro_bias_candidate(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    search_start_s: float,
    search_end_s: float,
) -> _GyroBiasStaticCandidate | None:
    """指定範囲内から gyro bias 推定用の静止窓を選ぶ。"""
    if "low_lin_norm" not in df_acc.columns:
        return None

    window_s = GYRO_BIAS_STATIC_WINDOW_SECONDS
    if search_end_s - search_start_s < window_s:
        return None

    max_start_s = search_end_s - window_s
    step_s = GYRO_BIAS_STATIC_WINDOW_STEP_SECONDS
    min_samples = max(3, int(GYRO_BIAS_MIN_CALIBRATION_SECONDS * SAMPLING_RATE))
    best: _GyroBiasStaticCandidate | None = None

    for start_s in np.arange(search_start_s, max_start_s + step_s * 0.5, step_s):
        end_s = float(start_s + window_s)
        gyro_mask = _time_mask(df_gyro, float(start_s), end_s)
        acc_mask = _time_mask(df_acc, float(start_s), end_s)
        if int(gyro_mask.sum()) < min_samples or int(acc_mask.sum()) < min_samples:
            continue

        gyro_segment = np.asarray(
            pd.to_numeric(df_gyro.loc[gyro_mask, "x"], errors="coerce"),
            dtype=float,
        )
        accel_segment = np.asarray(
            pd.to_numeric(df_acc.loc[acc_mask, "low_lin_norm"], errors="coerce"),
            dtype=float,
        )
        gyro_segment = gyro_segment[np.isfinite(gyro_segment)]
        accel_segment = accel_segment[np.isfinite(accel_segment)]
        if len(gyro_segment) < min_samples or len(accel_segment) < min_samples:
            continue

        gyro_std = float(np.nanstd(gyro_segment))
        accel_p95 = float(np.nanpercentile(accel_segment, 95))
        accel_max = float(np.nanmax(accel_segment))
        if (
            accel_p95 > GYRO_BIAS_STATIC_MAX_ACCEL_P95
            or gyro_std > GYRO_BIAS_STATIC_MAX_GYRO_STD
        ):
            continue

        score = (
            GYRO_BIAS_STATIC_ACCEL_P95_WEIGHT * accel_p95
            + GYRO_BIAS_STATIC_GYRO_STD_WEIGHT * gyro_std
        )
        candidate = _GyroBiasStaticCandidate(
            start_s=float(start_s),
            end_s=end_s,
            score=float(score),
            gyro_std=gyro_std,
            accel_p95=accel_p95,
            accel_max=accel_max,
        )
        if best is None or candidate.score < best.score:
            best = candidate

    return best


def _estimate_gyro_bias_static_window(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    search_start_s: float,
    search_end_s: float,
    fallback_reason: str | None = None,
) -> GyroBiasResult | None:
    """探索範囲内の最良静止窓から gyro bias を推定する。"""
    candidate = _find_static_gyro_bias_candidate(
        df_acc,
        df_gyro,
        search_start_s,
        search_end_s,
    )
    if candidate is None:
        return None

    return _robust_gyro_bias_from_mask(
        df_gyro,
        _time_mask(df_gyro, candidate.start_s, candidate.end_s),
        method="prewalk_robust",
        fallback_reason=fallback_reason,
        candidate=candidate,
        search_start_s=search_start_s,
        search_end_s=search_end_s,
    )


def _startup_static_search_range(df_gyro: pd.DataFrame) -> tuple[float, float] | None:
    """記録先頭側の静止探索範囲を返す。"""
    times = _time_values(df_gyro)
    if times is None:
        first_time = 0.0
    elif len(times) == 0:
        return None
    else:
        first_time = float(times[0])

    return (
        first_time + GYRO_BIAS_STATIC_SEARCH_START_SECONDS,
        first_time + GYRO_BIAS_STATIC_SEARCH_END_SECONDS,
    )


def _estimate_gyro_bias_prewalk_robust(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
) -> GyroBiasResult | None:
    """歩行開始前の静止サブウィンドウから gyro bias を推定する。"""
    onset_time = _find_walk_onset_time(df_acc)
    if onset_time is None:
        return None

    search_range = _startup_static_search_range(df_gyro)
    if search_range is None:
        return None
    search_start_s, startup_end_s = search_range
    search_end_s = min(
        startup_end_s,
        onset_time - GYRO_BIAS_STATIC_WALK_ONSET_MARGIN_S,
    )

    return _estimate_gyro_bias_static_window(
        df_acc,
        df_gyro,
        search_start_s,
        search_end_s,
    )


def estimate_gyro_bias(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    method: str = GYRO_BIAS_METHOD,
    manual_bias: float | None = None,
) -> GyroBiasResult:
    """指定手法で gyro bias を推定する。"""
    selected_method = _validate_gyro_bias_method(method)
    if selected_method == "manual":
        if manual_bias is None:
            raise ValueError("gyro_bias_method='manual' では gyro_bias が必要です。")
        return GyroBiasResult(
            method="manual",
            bias_rad_s=float(manual_bias),
            calibration_start_s=None,
            calibration_end_s=None,
            sample_count=0,
            kept_sample_count=0,
            raw_mean=None,
            robust_mean=float(manual_bias),
            median=None,
            mad=None,
            candidate_score=None,
            gyro_std=None,
            accel_p95=None,
            accel_max=None,
            search_start_s=None,
            search_end_s=None,
            fallback_reason=None,
        )

    if selected_method == "quietest":
        return _estimate_gyro_bias_quietest(df_gyro)

    if selected_method == "initial_robust":
        result = _estimate_gyro_bias_initial_robust(df_gyro)
        return (
            result
            if result is not None
            else _estimate_gyro_bias_quietest(
                df_gyro,
                fallback_reason="initial_robust_unavailable",
            )
        )

    result = _estimate_gyro_bias_prewalk_robust(df_acc, df_gyro)
    if result is not None:
        return result
    search_range = _startup_static_search_range(df_gyro)
    if search_range is not None:
        result = _estimate_gyro_bias_static_window(
            df_acc,
            df_gyro,
            search_range[0],
            search_range[1],
            fallback_reason="prewalk_static_unavailable",
        )
        if result is not None:
            return result
    result = _estimate_gyro_bias_initial_robust(
        df_gyro,
        fallback_reason="startup_static_unavailable",
    )
    if result is not None:
        return result
    return _estimate_gyro_bias_quietest(
        df_gyro,
        fallback_reason="prewalk_and_initial_unavailable",
    )


def process_sensor_data(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    gyro_bias_method: str | None = None,
    gyro_bias: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生センサーデータからノルム・重力推定・上下/水平加速度・角度を計算する。

    加速度データには以下の列を追加する：
    - ``gx, gy, gz``: 各軸のLPFによる重力ベクトル推定値
    - ``lin_x, lin_y, lin_z``: ベクトル減算による線形加速度
    - ``lin_norm``: 線形加速度ノルム
    - ``low_lin_norm``: 平滑化線形加速度ノルム（ステップ検出用）
    - ``v_acc``: 重力方向へ射影した上下加速度成分（Weinberg歩幅推定用）
    - ``h_x, h_y, h_z``: 重力方向を射影除去した水平加速度成分
    - ``h_norm``: 水平加速度ノルム（forward歩幅推定用）

    ジャイロスコープデータには積算角度（``angle``）・
    移動平均角度（``low_angle``）を追加する。

    Args:
        df_acc (pd.DataFrame): 加速度データ（列: t, x, y, z）
        df_gyro (pd.DataFrame): ジャイロスコープデータ（列: t, x, y, z）

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - df_acc: 上記列を追加した加速度DataFrame
            - df_gyro:
            ``angle``（積算角度）・``low_angle``（平滑化角度）を追加したジャイロDataFrame
    """
    df_acc = df_acc.copy().reset_index(drop=True)
    df_gyro = df_gyro.copy().reset_index(drop=True)

    # 3軸それぞれにLPFをかけて重力ベクトルを推定（スカラーノルムではなくベクトルで推定）
    # center=True で対称ウィンドウを使用し、位相遅れなく重力方向を推定する
    _roll_args = {"window": WINDOW_ACC, "center": True, "min_periods": 1}
    df_acc["gx"] = df_acc["x"].rolling(**_roll_args).mean()
    df_acc["gy"] = df_acc["y"].rolling(**_roll_args).mean()
    df_acc["gz"] = df_acc["z"].rolling(**_roll_args).mean()

    # ベクトル減算で線形加速度を算出（端末傾斜時も物理的に正確）
    df_acc["lin_x"] = df_acc["x"] - df_acc["gx"]
    df_acc["lin_y"] = df_acc["y"] - df_acc["gy"]
    df_acc["lin_z"] = df_acc["z"] - df_acc["gz"]

    # 線形加速度ノルム（垂直バウンド信号を含むためステップ検出に適する）
    df_acc["lin_norm"] = np.sqrt(
        df_acc["lin_x"] ** 2 + df_acc["lin_y"] ** 2 + df_acc["lin_z"] ** 2
    )
    df_acc["low_lin_norm"] = (
        df_acc["lin_norm"].rolling(window=WINDOW_ACC, center=True, min_periods=1).mean()
    )

    # 重力方向単位ベクトル ĝ = g / |g|
    # |g| の最小値を 1e-9 に制限して、ĝ 正規化時のゼロ除算を回避する
    g_norm = np.maximum(
        np.sqrt(df_acc["gx"] ** 2 + df_acc["gy"] ** 2 + df_acc["gz"] ** 2),
        1e-9,
    )
    df_acc["gx_hat"] = df_acc["gx"] / g_norm
    df_acc["gy_hat"] = df_acc["gy"] / g_norm
    df_acc["gz_hat"] = df_acc["gz"] / g_norm

    # 上下加速度: a_v = a_lin · ĝ（重力方向への符号付き射影）
    # Weinbergモデルは上下方向の振幅を使うため、この値を歩幅推定に使用する
    dot = (
        df_acc["lin_x"] * df_acc["gx_hat"]
        + df_acc["lin_y"] * df_acc["gy_hat"]
        + df_acc["lin_z"] * df_acc["gz_hat"]
    )
    df_acc["v_acc"] = dot

    # 水平加速度: a_h = a_lin − (a_lin · ĝ) ĝ（重力方向成分を射影で除去）
    df_acc["h_x"] = df_acc["lin_x"] - dot * df_acc["gx_hat"]
    df_acc["h_y"] = df_acc["lin_y"] - dot * df_acc["gy_hat"]
    df_acc["h_z"] = df_acc["lin_z"] - dot * df_acc["gz_hat"]
    # 水平加速度ノルム: forward手法で前進方向へ射影するための姿勢非依存な水平成分
    df_acc["h_norm"] = np.sqrt(
        df_acc["h_x"] ** 2 + df_acc["h_y"] ** 2 + df_acc["h_z"] ** 2
    )

    bias_result = estimate_gyro_bias(
        df_acc,
        df_gyro,
        method=GYRO_BIAS_METHOD if gyro_bias_method is None else gyro_bias_method,
        manual_bias=gyro_bias,
    )
    gyro_rate = (df_gyro["x"] - bias_result.bias_rad_s).to_numpy(dtype=float)
    df_gyro["gyro_rate"] = gyro_rate
    df_gyro["gyro_bias"] = bias_result.bias_rad_s
    df_gyro["gyro_bias_method"] = bias_result.method
    df_gyro.attrs["gyro_bias_result"] = bias_result
    df_gyro["angle"] = np.cumsum(gyro_rate * _gyro_integration_dt(df_gyro))
    df_gyro["low_angle"] = (
        df_gyro["angle"].rolling(window=WINDOW_GYRO, center=True, min_periods=1).mean()
    )

    return df_acc, df_gyro


def _validate_step_detection_method(method: str) -> str:
    """ステップ検出手法名を検証する。"""
    if method not in STEP_DETECTION_METHODS:
        allowed = ", ".join(STEP_DETECTION_METHODS)
        raise ValueError(
            f"step_detection_method は {allowed} のいずれかを指定してください。"
        )
    return method


def _detect_steps_by_peak(df_acc: pd.DataFrame) -> StepDetectionResult:
    """既存方式: 平滑化線形加速度ノルムからステップピークを検出する。"""
    peaks, _ = find_peaks(
        df_acc["low_lin_norm"].to_numpy(),
        distance=PEAK_DISTANCE,
        height=PEAK_HEIGHT,
    )
    return StepDetectionResult(
        method="peak",
        peaks=np.asarray(peaks),
        segments=(),
        threshold=PEAK_HEIGHT,
        polarity=None,
    )


def _threshold_groups(mask: np.ndarray) -> list[tuple[int, int]]:
    """True が連続する範囲を [start, end) のリストで返す。"""
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            groups.append((start, i))
            start = None
    if start is not None:
        groups.append((start, len(mask)))
    return groups


def _suppress_close_contacts(
    contacts: list[tuple[int, float]],
    min_distance: int = PEAK_DISTANCE,
) -> list[tuple[int, float]]:
    """近すぎる接地候補は強度が大きい方だけ残す。"""
    if not contacts:
        return []

    kept: list[tuple[int, float]] = [contacts[0]]
    for index, strength in contacts[1:]:
        prev_index, prev_strength = kept[-1]
        if index - prev_index < min_distance:
            if strength > prev_strength:
                kept[-1] = (index, strength)
        else:
            kept.append((index, strength))
    return kept


def _detect_steps_by_vertical_threshold(df_acc: pd.DataFrame) -> StepDetectionResult:
    """論文方式に寄せて、上下加速度の接地閾値から1歩区間を抽出する。"""
    values = np.asarray(pd.to_numeric(df_acc["v_acc"], errors="coerce"), dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return StepDetectionResult(
            method="paper_vertical_threshold",
            peaks=np.array([], dtype=int),
            segments=(),
            threshold=None,
            polarity=None,
        )

    filled = values.copy()
    median = float(np.nanmedian(filled[finite]))
    filled[~finite] = median
    smoothed = (
        pd.Series(filled)
        .rolling(window=STEP_VERTICAL_SMOOTH_WINDOW, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )

    positive_span = float(np.nanpercentile(smoothed, 95))
    negative_span = abs(float(np.nanpercentile(smoothed, 5)))
    polarity = -1 if negative_span > positive_span else 1
    raw_contact_signal = filled * polarity
    contact_signal = smoothed * polarity
    finite_contact = contact_signal[np.isfinite(contact_signal)]
    threshold = float(
        np.nanpercentile(finite_contact, STEP_VERTICAL_THRESHOLD_PERCENTILE)
    )
    baseline = float(np.nanmedian(finite_contact))
    signal_max = float(np.nanmax(finite_contact))
    if threshold <= baseline:
        threshold = baseline + (signal_max - baseline) * 0.25
    if signal_max <= baseline:
        return StepDetectionResult(
            method="paper_vertical_threshold",
            peaks=np.array([], dtype=int),
            segments=(),
            threshold=threshold,
            polarity=polarity,
        )

    groups = _threshold_groups(contact_signal >= threshold)
    contacts: list[tuple[int, float]] = []
    for start, end in groups:
        if end <= start:
            continue
        segment = raw_contact_signal[start:end]
        if not np.isfinite(segment).any():
            continue
        local_index = int(np.nanargmax(segment))
        contact_index = start + local_index
        contacts.append((contact_index, float(raw_contact_signal[contact_index])))

    contacts = _suppress_close_contacts(contacts)
    contact_indexes = [index for index, _strength in contacts]

    segments: list[StepSegment] = []
    for start_index, end_index in zip(
        contact_indexes[:-1], contact_indexes[1:], strict=False
    ):
        seg_len = end_index - start_index
        if MIN_SEG_SAMPLES <= seg_len <= MAX_SEG_SAMPLES:
            segments.append(
                StepSegment(
                    start_index=int(start_index),
                    end_index=int(end_index),
                    contact_index=int(end_index),
                )
            )

    peaks = np.asarray([segment.contact_index for segment in segments], dtype=int)
    return StepDetectionResult(
        method="paper_vertical_threshold",
        peaks=peaks,
        segments=tuple(segments),
        threshold=threshold,
        polarity=polarity,
    )


def detect_step_result(
    df_acc: pd.DataFrame,
    method: str | None = None,
) -> StepDetectionResult:
    """指定方式でステップを検出し、互換ピーク列と区間情報を返す。"""
    selected_method = _validate_step_detection_method(
        STEP_DETECTION_METHOD if method is None else method
    )
    if selected_method == "paper_vertical_threshold":
        return _detect_steps_by_vertical_threshold(df_acc)
    return _detect_steps_by_peak(df_acc)


def detect_steps(df_acc: pd.DataFrame, method: str | None = None) -> np.ndarray:
    """指定方式でステップを検出し、既存互換のピーク配列だけを返す。

    ``low_lin_norm`` 列に対してピーク検出を行い，ステップに対応するインデックスを返す。
    ピーク間距離 ``PEAK_DISTANCE`` と最小高さ ``PEAK_HEIGHT`` でフィルタリングする。

    Args:
        df_acc (pd.DataFrame): ``low_lin_norm`` 列を含む加速度DataFrame
        method: ステップ検出手法。省略時は ``STEP_DETECTION_METHOD`` を使用。

    Returns:
        np.ndarray: ステップピークのインデックス配列
    """
    return detect_step_result(df_acc, method).peaks


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


class StepMotion(NamedTuple):
    """状態別補正後の1歩の移動量。"""

    heading: float
    length: float
    movement_type: str
    length_scale: float


def estimate_step_length(
    df_acc: pd.DataFrame,
    peak_index: int,
    window: int = STEP_LENGTH_WINDOW,
    k: float = WEINBERG_K,
) -> float:
    """Weinberg モデルによる単一ステップの歩幅推定。

    ピークインデックスの前後 ``window`` サンプルの範囲内で
    上下加速度成分（``v_acc``）の最大値・最小値を求め，
    Weinberg 式で歩幅を計算する。
    重力方向へ射影した符号付き成分を使うことで、Weinbergモデルの前提である
    歩行中の上下バウンド振幅を反映する。
    ウィンドウがデータ範囲外にかかる場合はクリッピングする。

    Args:
        df_acc (pd.DataFrame): ``v_acc`` 列を含む加速度DataFrame
        peak_index (int): ステップピークのインデックス
        window (int): ピーク前後のサンプル数
        k (float): Weinberg モデルのスケール係数

    Returns:
        float: 推定歩幅（メートル）
    """
    n = len(df_acc)
    start = max(0, peak_index - window)
    end = min(n, peak_index + window + 1)
    # ウィンドウ内の上下加速度成分（NaN除去済み）
    segment = df_acc["v_acc"].iloc[start:end].dropna()
    if segment.empty:
        return 0.0  # データ不足のステップは歩幅 0 として軌跡から実質除外
    acc_max = float(segment.max())  # v_acc 最大値 [m/s²]（上向きバウンド付近）
    acc_min = float(segment.min())  # v_acc 最小値 [m/s²]（下向きバウンド付近）
    return float(k * (acc_max - acc_min) ** 0.25)


def _estimate_initial_forward_angle(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    peaks: np.ndarray,
) -> float:
    """全ステップの変位方向の循環平均から前進方向の初期角度 φ₀ を推定する。

    各ステップの h_y・h_z を2重積分した変位ベクトルを求め、
    そのステップ中点での low_angle を引いてセンサー座標系の角度を取得。
    全ステップの循環平均（sin/cos の平均 → atan2）で外れ値に頑健な推定を行う。

    Returns:
        float: センサー Y-Z 平面における前進方向の角度 [rad]
    """
    dt = 1.0 / SAMPLING_RATE
    sin_sum = 0.0
    cos_sum = 0.0
    count = 0
    for i in range(len(peaks) - 1):
        start = int(peaks[i])
        end = int(peaks[i + 1])
        if not (30 <= end - start <= MAX_SEG_SAMPLES):
            continue
        h_y = df_acc["h_y"].iloc[start:end].to_numpy()
        h_z = df_acc["h_z"].iloc[start:end].to_numpy()
        n = len(h_y)
        v_y = np.cumsum(h_y) * dt
        v_z = np.cumsum(h_z) * dt
        v_y -= np.linspace(v_y[0], v_y[-1], n)
        v_z -= np.linspace(v_z[0], v_z[-1], n)
        dy = float(np.sum(v_y) * dt)
        dz = float(np.sum(v_z) * dt)
        if np.hypot(dy, dz) < 1e-4:
            continue
        # センサー座標系の角度 = 変位方向 − その時点での yaw 角
        mid_idx = (start + end) // 2
        mid_time = (_time_at_index(df_acc, start) + _time_at_index(df_acc, end)) / 2
        angle_at_mid = _sample_gyro_angle(
            df_gyro,
            sample_index=mid_idx,
            sample_time=mid_time,
        )
        if angle_at_mid is None:
            continue
        sensor_angle = np.arctan2(dz, dy) - angle_at_mid
        sin_sum += np.sin(sensor_angle)
        cos_sum += np.cos(sensor_angle)
        count += 1
    if count == 0:
        return 0.0
    return float(np.arctan2(sin_sum, cos_sum))


def estimate_step_length_forward(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    peaks: np.ndarray,
    i: int,
    phi_0: float,
) -> float:
    """方位方向射影による単一ステップの歩幅推定。

    ジャイロから得た前進方向角（φ₀ + low_angle）へ水平加速度を射影し、
    符号付きの前進加速度を直接2重積分する。
    線形ドリフト補正で両端速度を 0 に揃えた後、K_FORWARD を乗じて歩幅を算出する。

    Args:
        df_acc: h_y・h_z 列を含む加速度 DataFrame
        df_gyro: low_angle 列を含むジャイロ DataFrame
        peaks: ステップピークのインデックス配列
        i: 現在のステップインデックス
        phi_0: センサー座標系における初期前進方向角 [rad]

    Returns:
        float: 推定歩幅 [m]（計算不能時は 0.0）
    """
    dt = 1.0 / SAMPLING_RATE
    start = int(peaks[i])
    end = int(peaks[i + 1]) if i + 1 < len(peaks) else start + 1

    seg_len = end - start
    if seg_len < 30 or seg_len > MAX_SEG_SAMPLES:
        return 0.0

    # このステップ中点での前進方向角
    mid_idx = (start + end) // 2
    mid_time = (_time_at_index(df_acc, start) + _time_at_index(df_acc, end)) / 2
    angle_at_mid = _sample_gyro_angle(
        df_gyro,
        sample_index=mid_idx,
        sample_time=mid_time,
    )
    if angle_at_mid is None:
        return 0.0
    angle = angle_at_mid + phi_0

    # 前進方向加速度（符号付き）= h_y・h_z をヨー角で射影
    h_y = df_acc["h_y"].iloc[start:end].to_numpy()
    h_z = df_acc["h_z"].iloc[start:end].to_numpy()
    a_fwd = h_y * np.cos(angle) + h_z * np.sin(angle)

    n = len(a_fwd)
    if n < 3:
        return 0.0

    # 直接2重積分 + 線形ドリフト補正（両端速度を 0 に）
    v = np.cumsum(a_fwd) * dt
    v -= np.linspace(v[0], v[-1], n)
    osc_disp = abs(float(np.sum(v) * dt))

    return K_FORWARD * osc_disp


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


def _compute_pixel_coords(
    xs: np.ndarray,
    ys: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """メートル座標をフロアマップのピクセル座標に変換する。

    重力の主成分軸（|gx| vs |gz|）でY軸反転の判定軸を自動選択する。
    """
    if abs(gx_mean) > abs(gz_mean):
        y_sign = -1 if gx_mean > 0 else 1
    else:
        y_sign = -1 if gz_mean < 0 else 1
    px = origin_px[0] + xs / scale
    py = origin_px[1] + y_sign * ys / scale
    return px, py


def _pixel_vector_from_heading(
    heading: float,
    length_m: float,
    gx_mean: float,
    gz_mean: float,
    scale: float,
) -> tuple[float, float]:
    """メートル座標の方位ベクトルをピクセル座標の差分に変換する。"""
    y_sign = (
        -1
        if (
            (abs(gx_mean) > abs(gz_mean) and gx_mean > 0)
            or (abs(gz_mean) >= abs(gx_mean) and gz_mean < 0)
        )
        else 1
    )
    return (
        length_m * float(np.cos(heading)) / scale,
        y_sign * length_m * float(np.sin(heading)) / scale,
    )


def _plot_heading_overlay(
    ax: Axes,
    trajectory: list[list[float]],
    step_headings: list[StepHeading] | None,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> None:
    """軌跡上に移動方向・体の向き・横歩き判定を重ねて描画する。"""
    if step_headings is None or len(step_headings) == 0 or len(trajectory) < 2:
        return

    points = np.asarray(trajectory, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        return

    px, py = _compute_pixel_coords(
        points[:, 0],
        points[:, 1],
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    step_count = min(len(step_headings), len(points) - 1)
    if step_count <= 0:
        return

    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    positive_lengths = lengths[lengths > 1e-12]
    arrow_length_m = (
        float(np.median(positive_lengths)) * 0.45
        if len(positive_lengths) > 0
        else max(scale * 30.0, 0.3)
    )
    arrow_length_m = max(arrow_length_m, scale * 24.0)

    body_label_added = False
    motion_label_added = False
    sidestep_points: list[tuple[float, float]] = []
    sidestep_suspect_points: list[tuple[float, float]] = []
    for i in range(step_count):
        heading = step_headings[i]
        trajectory_movement_type = (
            heading.trajectory_movement_type
            if heading.trajectory_movement_type is not None
            else heading.movement_type
        )
        start_x = float(px[i])
        start_y = float(py[i])
        if heading.selected_heading is not None:
            dx, dy = _pixel_vector_from_heading(
                heading.selected_heading,
                arrow_length_m,
                gx_mean,
                gz_mean,
                scale,
            )
            ax.arrow(
                start_x,
                start_y,
                dx,
                dy,
                width=1.4,
                head_width=10.0,
                head_length=12.0,
                length_includes_head=True,
                color="dodgerblue",
                alpha=0.85,
                zorder=5,
                label="Move heading" if not motion_label_added else None,
            )
            motion_label_added = True
        if heading.body_heading is not None:
            dx, dy = _pixel_vector_from_heading(
                heading.body_heading,
                arrow_length_m * 1.05,
                gx_mean,
                gz_mean,
                scale,
            )
            ax.arrow(
                start_x,
                start_y,
                dx,
                dy,
                width=1.8,
                head_width=12.0,
                head_length=15.0,
                length_includes_head=True,
                color="orangered",
                alpha=0.95,
                zorder=6,
                label="Body heading" if not body_label_added else None,
            )
            body_label_added = True
        if _is_sidestep_suspect_movement(trajectory_movement_type):
            sidestep_suspect_points.append((float(px[i + 1]), float(py[i + 1])))
        elif _is_trajectory_sidestep_movement(trajectory_movement_type):
            sidestep_points.append((float(px[i + 1]), float(py[i + 1])))

    if sidestep_points:
        sidestep_arr = np.asarray(sidestep_points, dtype=float)
        ax.scatter(
            sidestep_arr[:, 0],
            sidestep_arr[:, 1],
            marker="s",
            s=52,
            facecolors="none",
            edgecolors="lime",
            linewidths=1.8,
            zorder=7,
            label="Sidestep",
        )
    if sidestep_suspect_points:
        sidestep_suspect_arr = np.asarray(sidestep_suspect_points, dtype=float)
        ax.scatter(
            sidestep_suspect_arr[:, 0],
            sidestep_suspect_arr[:, 1],
            marker="D",
            s=46,
            facecolors="none",
            edgecolors="gold",
            linewidths=1.8,
            zorder=7,
            label="Sidestep suspect",
        )


def plot_trajectory(
    trajectory: list[list[float]],
    gx_mean: float = 0.0,
    gz_mean: float = 0.0,
    floormap_path: str | Path = FLOORMAP_PATH,
    origin_px: tuple[int, int] = FLOORMAP_ORIGIN_PX,
    scale: float = FLOORMAP_SCALE,
    output_dir: Path | None = None,
    step_headings: list[StepHeading] | None = None,
) -> None:
    """推定した2次元歩行軌跡をフロアマップ上にプロットする。

    フロアマップ画像を背景として表示し，軌跡をピクセル座標に変換してオーバーレイする。
    重力の主成分軸（|gx| vs |gz| の大小）でY軸反転の判定軸を自動選択する。
    - |gx| > |gz|（端末を縦に持つ持ち方）: gx < 0 のとき反転（X軸が上を向いている）
    - |gz| >= |gx|（端末を平置き）        : gz > 0 のとき反転（画面が上を向いている）

    Args:
        trajectory (list[list[float]]): 各ステップの [x, y] 座標リスト（メートル）
        gx_mean (float): X軸重力成分の全サンプル平均値 [m/s²]
        gz_mean (float): Z軸重力成分の全サンプル平均値 [m/s²]
        floormap_path (str | Path): フロアマップ画像のパス
        origin_px (tuple[int, int]): 軌跡の起点に対応するピクセル座標 (x_px, y_px)
        scale (float): 1ピクセルあたりのメートル数（1px = 1cm = 0.01m）
    """
    df = pd.DataFrame(trajectory, columns=["x", "y"])

    px, py = _compute_pixel_coords(
        df["x"].to_numpy(), df["y"].to_numpy(), gx_mean, gz_mean, origin_px, scale
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    # フロアマップを背景として表示
    map_img = plt.imread(Path(floormap_path))
    ax.imshow(map_img)

    # 軌跡をグラデーション（開始:青 → 終了:赤）で描画
    n = len(px)
    norm = Normalize(vmin=0, vmax=max(n - 1, 1))
    cmap = cm.get_cmap("plasma")
    # 各ステップ間のセグメントに色を付けて LineCollection で描画
    pts = np.column_stack([px, py]).reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segments.tolist(), cmap=cmap, norm=norm, zorder=2)
    lc.set_array(np.arange(n - 1))
    ax.add_collection(lc)
    # 各ステップ点を同じカラーマップで描画
    sc = ax.scatter(px, py, c=np.arange(n), cmap=cmap, norm=norm, s=20, zorder=3)
    fig.colorbar(sc, ax=ax, label="Step")
    # 起点を強調表示
    ax.plot(px[0], py[0], "go", markersize=10, label="Start", zorder=4)
    _plot_heading_overlay(
        ax,
        trajectory,
        step_headings,
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )

    ax.set_title("Walking Trajectory on Floormap")
    ax.legend()
    plt.tight_layout()
    if output_dir is not None:
        # グラフ画像をoutputフォルダに保存
        fig.savefig(output_dir / "trajectory.png", dpi=150, bbox_inches="tight")
    plt.show()


def _build_step_vectors_dataframe(trajectory: list[list[float]]) -> pd.DataFrame:
    """軌跡点列からステップごとの変位ベクトルをDataFrame化する。"""
    points = np.asarray(trajectory, dtype=float)
    if len(points) < 2:
        return pd.DataFrame(
            columns=[
                "step",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "dx",
                "dy",
                "step_length_m",
                "heading_deg",
            ]
        )

    starts = points[:-1]
    ends = points[1:]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    headings = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return pd.DataFrame(
        {
            "step": np.arange(1, len(vectors) + 1),
            "start_x": starts[:, 0],
            "start_y": starts[:, 1],
            "end_x": ends[:, 0],
            "end_y": ends[:, 1],
            "dx": vectors[:, 0],
            "dy": vectors[:, 1],
            "step_length_m": lengths,
            "heading_deg": headings,
        }
    )


def _build_trajectory_dataframe(
    trajectory: list[list[float]],
    t_at_steps: list[float],
) -> pd.DataFrame:
    """軌跡点列と移動後座標の時刻から時刻付きDataFrameを作成する。"""
    points = np.asarray(trajectory, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("trajectory は [x, y] の点列である必要があります。")

    if len(points) != len(t_at_steps) + 1:
        raise ValueError(
            "trajectory と t_at_steps の長さが一致しません: "
            f"len(trajectory)={len(points)}, len(t_at_steps)={len(t_at_steps)}"
        )
    if len(t_at_steps) == 0:
        return pd.DataFrame(columns=["timestamp_s", "x", "y"])

    moved_points = points[1:]
    first_step_time = t_at_steps[0]
    timestamps = [float(t - first_step_time) for t in t_at_steps]
    return pd.DataFrame(
        {
            "timestamp_s": timestamps,
            "x": moved_points[:, 0],
            "y": moved_points[:, 1],
        }
    )


def _build_step_segments_dataframe(
    df_acc: pd.DataFrame,
    segments: tuple[StepSegment, ...],
) -> pd.DataFrame:
    """ステップ区間情報をCSV保存用DataFrameに変換する。"""
    columns = [
        "step",
        "start_index",
        "end_index",
        "contact_index",
        "start_time_s",
        "end_time_s",
        "duration_s",
    ]
    if len(segments) == 0:
        return pd.DataFrame(columns=columns)

    rows = []
    for step, segment in enumerate(segments, start=1):
        start_time = _time_at_index(df_acc, segment.start_index)
        end_time = _time_at_index(df_acc, segment.end_index)
        rows.append(
            {
                "step": step,
                "start_index": segment.start_index,
                "end_index": segment.end_index,
                "contact_index": segment.contact_index,
                "start_time_s": start_time,
                "end_time_s": end_time,
                "duration_s": end_time - start_time,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _angle_to_deg(angle: float | None) -> float | None:
    """ラジアン角を度へ変換する。None はそのまま返す。"""
    if angle is None:
        return None
    return float(np.degrees(angle))


def _build_gyro_bias_dataframe(df_gyro: pd.DataFrame) -> pd.DataFrame:
    """ジャイロバイアス診断情報をCSV保存用DataFrameに変換する。"""
    columns = [
        "method",
        "bias_rad_s",
        "calibration_start_s",
        "calibration_end_s",
        "sample_count",
        "kept_sample_count",
        "raw_mean",
        "robust_mean",
        "median",
        "mad",
        "candidate_score",
        "gyro_std",
        "accel_p95",
        "accel_max",
        "search_start_s",
        "search_end_s",
        "fallback_reason",
    ]
    result = df_gyro.attrs.get("gyro_bias_result")
    if isinstance(result, GyroBiasResult):
        return pd.DataFrame(
            [
                {
                    "method": result.method,
                    "bias_rad_s": result.bias_rad_s,
                    "calibration_start_s": result.calibration_start_s,
                    "calibration_end_s": result.calibration_end_s,
                    "sample_count": result.sample_count,
                    "kept_sample_count": result.kept_sample_count,
                    "raw_mean": result.raw_mean,
                    "robust_mean": result.robust_mean,
                    "median": result.median,
                    "mad": result.mad,
                    "candidate_score": result.candidate_score,
                    "gyro_std": result.gyro_std,
                    "accel_p95": result.accel_p95,
                    "accel_max": result.accel_max,
                    "search_start_s": result.search_start_s,
                    "search_end_s": result.search_end_s,
                    "fallback_reason": result.fallback_reason,
                }
            ],
            columns=columns,
        )

    return pd.DataFrame(columns=columns)


def _build_step_headings_dataframe(step_headings: list[StepHeading]) -> pd.DataFrame:
    """ステップ方位候補と採用結果をCSV保存用DataFrameに変換する。"""
    columns = [
        "step",
        "timestamp_s",
        "gyro_heading_deg",
        "body_heading_deg",
        "accel_method1_heading_deg",
        "accel_method2_heading_deg",
        "motion_heading_deg",
        "selected_heading_deg",
        "source",
        "movement_type",
        "trajectory_movement_type",
        "forward_heading_source",
        "step_length_scale",
        "confidence",
        "motion_confidence",
        "yaw_delta_deg",
        "motion_heading_correction_deg",
        "device_orientation_mode",
        "body_motion_angle_diff_deg",
        "lateral_forward_ratio",
        "sidestep_lateral_ratio",
        "sidestep_min_lateral_displacement",
        "sidestep_evidence_direction",
        "sidestep_evidence_reason",
        "sidestep_cluster_id",
        "angle_diff_method1_deg",
        "angle_diff_method2_deg",
        "forward_displacement",
        "lateral_displacement",
        "motion_reject_reason",
        "segment_start_index",
        "segment_end_index",
        "peak1_index",
        "peak2_index",
    ]
    rows = [
        {
            "step": heading.step_index,
            "timestamp_s": heading.timestamp_s,
            "gyro_heading_deg": _angle_to_deg(heading.gyro_heading),
            "body_heading_deg": _angle_to_deg(heading.body_heading),
            "accel_method1_heading_deg": _angle_to_deg(heading.accel_method1_heading),
            "accel_method2_heading_deg": _angle_to_deg(heading.accel_method2_heading),
            "motion_heading_deg": _angle_to_deg(heading.motion_heading),
            "selected_heading_deg": _angle_to_deg(heading.selected_heading),
            "source": heading.source,
            "movement_type": heading.movement_type,
            "trajectory_movement_type": heading.trajectory_movement_type,
            "forward_heading_source": heading.forward_heading_source,
            "step_length_scale": heading.step_length_scale,
            "confidence": heading.confidence,
            "motion_confidence": heading.motion_confidence,
            "yaw_delta_deg": _angle_to_deg(heading.yaw_delta),
            "motion_heading_correction_deg": _angle_to_deg(
                heading.motion_heading_correction
            ),
            "device_orientation_mode": heading.device_orientation_mode,
            "body_motion_angle_diff_deg": _angle_to_deg(heading.body_motion_angle_diff),
            "lateral_forward_ratio": _lateral_forward_ratio(heading),
            "sidestep_lateral_ratio": heading.sidestep_lateral_ratio,
            "sidestep_min_lateral_displacement": (
                heading.sidestep_min_lateral_displacement
            ),
            "sidestep_evidence_direction": heading.sidestep_evidence_direction,
            "sidestep_evidence_reason": heading.sidestep_evidence_reason,
            "sidestep_cluster_id": heading.sidestep_cluster_id,
            "angle_diff_method1_deg": _angle_to_deg(heading.angle_diff_method1),
            "angle_diff_method2_deg": _angle_to_deg(heading.angle_diff_method2),
            "forward_displacement": heading.forward_displacement,
            "lateral_displacement": heading.lateral_displacement,
            "motion_reject_reason": heading.motion_reject_reason,
            "segment_start_index": heading.segment_start_index,
            "segment_end_index": heading.segment_end_index,
            "peak1_index": heading.peak1_index,
            "peak2_index": heading.peak2_index,
        }
        for heading in step_headings
    ]
    return pd.DataFrame(rows, columns=columns)


def _step_plot_signal(
    df_acc: pd.DataFrame,
    step_detection: StepDetectionResult,
) -> tuple[np.ndarray, str, float | None]:
    """ステップ検出方式に対応する可視化用信号を返す。"""
    if step_detection.method == "paper_vertical_threshold":
        polarity = 1 if step_detection.polarity is None else step_detection.polarity
        return (
            df_acc["v_acc"].to_numpy(dtype=float) * polarity,
            "vertical contact signal",
            step_detection.threshold,
        )
    return df_acc["low_lin_norm"].to_numpy(dtype=float), "low_lin_norm", None


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

    selected_gyro_bias_method = _validate_gyro_bias_method(
        GYRO_BIAS_METHOD if gyro_bias_method is None else gyro_bias_method
    )
    df_acc, df_gyro = process_sensor_data(
        df_acc,
        df_gyro,
        gyro_bias_method=selected_gyro_bias_method,
        gyro_bias=gyro_bias,
    )
    step_detection = detect_step_result(df_acc, step_detection_method)
    peaks = step_detection.peaks
    weinberg_k = compute_weinberg_k(height_m)
    selected_heading_method = _validate_heading_method(
        HEADING_METHOD if heading_method is None else heading_method
    )
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
    gx_mean = float(df_acc["gx"].mean())
    gz_mean = float(df_acc["gz"].mean())
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
        from .particle_filter import (  # noqa: PLC0415
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
            from .sensor_plot import (  # noqa: PLC0415
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
        trajectory, step_lengths, t_at_steps, step_headings = (
            estimate_trajectory_with_headings(
                peaks,
                df_gyro,
                df_acc,
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
            from .sensor_plot import (  # noqa: PLC0415
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
