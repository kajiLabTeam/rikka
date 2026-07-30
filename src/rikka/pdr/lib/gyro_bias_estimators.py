"""ジャイロバイアス推定の基礎部品。

役割:
    候補型、方式検証、時間マスク、ロバスト統計、quietest 推定を実装する。
依存元:
    common の設定・共有型・時刻処理と NumPy、Pandas を利用する。
利用先:
    gyro_bias の各推定方式と互換 shim から使用される。
処理フロー:
    対象区間を抽出し、外れ値を除いた候補を GyroBiasResult として返す。
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from ...common.config import (
    GYRO_BIAS_GUARD_MAX_ABS_RAD_S,
    GYRO_BIAS_OUTLIER_MAD_SCALE,
    SAMPLING_RATE,
    WINDOW_GYRO,
)
from ...common.lib.models import GyroBiasResult
from ...common.lib.time_utils import _time_values

GYRO_BIAS_METHODS = (
    "prewalk_guarded",
    "zero",
    "prewalk_robust",
    "initial_robust",
    "quietest",
    "manual",
)


def _guard_gyro_bias_result(result: GyroBiasResult) -> GyroBiasResult:
    """大きすぎる歩行前推定値を端末回転由来とみなして無補正へ戻す。"""
    rejected = abs(result.bias_rad_s) > GYRO_BIAS_GUARD_MAX_ABS_RAD_S
    reason = result.fallback_reason
    if rejected:
        reason = (
            "estimated_bias_exceeds_guard"
            if reason is None
            else f"{reason};estimated_bias_exceeds_guard"
        )
    return result._replace(
        method="prewalk_guarded",
        bias_rad_s=0.0 if rejected else result.bias_rad_s,
        fallback_reason=reason,
    )


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
