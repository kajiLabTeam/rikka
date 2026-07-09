"""ジャイロバイアス推定手法。"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ...config import (
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
    PEAK_DISTANCE,
    PEAK_HEIGHT,
    SAMPLING_RATE,
    WINDOW_GYRO,
)
from .models import GyroBiasResult
from .time_utils import _time_at_index, _time_values

GYRO_BIAS_METHODS = ("prewalk_robust", "initial_robust", "quietest", "manual")


class _GyroBiasStaticCandidate(NamedTuple):
    """gyro bias 推定に使う静止候補窓。"""

    start_s: float
    end_s: float
    score: float
    gyro_std: float
    accel_p95: float
    accel_max: float


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
