"""PDR の歩幅推定手法。

役割:
    加速度振幅を使う Weinberg モデル、またはジャイロ方位へ射影した加速度の
    二重積分方式で各ステップの歩幅を推定する。
依存元:
    ``config`` から窓幅・係数・区間上限、``time_utils`` から時刻とジャイロ角の
    サンプリングを取得し、NumPy と Pandas で数値積分する。
利用先:
    ``trajectory`` が通常 PDR に使用し、``particle_api`` を通じて
    ``particle_filter`` も同じ決定論的歩幅を使用する。
処理フロー:
    対象ステップ区間を切り出し、方式に応じて加速度振幅または前進方向変位を計算し、
    1歩分の距離をメートルで返す。
"""

import numpy as np
import pandas as pd

from ...common.config import (
    K_FORWARD,
    MAX_SEG_SAMPLES,
    SAMPLING_RATE,
    STEP_LENGTH_WINDOW,
    WEINBERG_K,
)
from ...common.lib.models import StepHeading, StepLengthObservation
from ...common.lib.time_utils import (
    _sample_gyro_angle,
    _time_at_index,
    _time_values,
)


def _integrate_forward_acceleration(
    acceleration: np.ndarray,
    times: np.ndarray,
) -> float:
    """実時刻で加速度を二重積分し、両端速度を0へ補正する。"""
    if len(acceleration) < 3 or len(times) != len(acceleration):
        return 0.0
    dt = np.diff(times, prepend=times[0])
    dt[0] = 0.0
    if not np.isfinite(dt).all() or np.any(dt < 0.0):
        dt = np.full(len(times), 1.0 / SAMPLING_RATE)
        dt[0] = 0.0
    velocity = np.cumsum(acceleration * dt)
    velocity -= np.linspace(velocity[0], velocity[-1], len(velocity))
    return float(np.sum(velocity * dt))


def _segment_times(df_acc: pd.DataFrame, start: int, end: int) -> np.ndarray:
    """積分区間の実時刻、または固定周期の合成時刻を返す。"""
    times = _time_values(df_acc)
    if times is None:
        times = np.arange(len(df_acc), dtype=float) / SAMPLING_RATE
    return times[start:end]


def estimate_step_length(
    df_acc: pd.DataFrame,
    peak_index: int,
    window: int = STEP_LENGTH_WINDOW,
    k: float = WEINBERG_K,
) -> float:
    """Weinberg モデルによる単一ステップの歩幅推定。"""
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


def build_step_length_observation(
    df_acc: pd.DataFrame,
    step_heading: StepHeading,
    nominal_length_m: float,
    k: float = WEINBERG_K,
) -> StepLengthObservation:
    """検出済みの1歩区間から歩幅観測と品質を作る。

    接地境界が利用できる場合は固定幅窓ではなくその区間を使う。区間が短すぎる、
    または列が不足する場合だけ従来の歩幅を採用し、不確かさを大きくする。
    """
    n = len(df_acc)
    start = step_heading.segment_start_index
    end = step_heading.segment_end_index
    fallback_reason: str | None = None
    if start is None or end is None or not (0 <= start < end < n):
        peak = step_heading.peak1_index
        if peak is None:
            peak = min(max(step_heading.step_index, 0), max(n - 1, 0))
        start = max(0, peak - STEP_LENGTH_WINDOW)
        end = min(n - 1, peak + STEP_LENGTH_WINDOW)
        fallback_reason = "step_interval_unavailable"

    vertical = df_acc["v_acc"].iloc[start : end + 1].dropna()
    sample_count = len(vertical)
    if sample_count < 3:
        amplitude = 0.0
        interval_length = nominal_length_m
        fallback_reason = "insufficient_vertical_samples"
    else:
        amplitude = max(float(vertical.max() - vertical.min()), 0.0)
        interval_length = float(
            k * amplitude**0.25 * max(step_heading.step_length_scale, 1e-6)
        )

    period: float | None = None
    try:
        period_value = _time_at_index(df_acc, end) - _time_at_index(df_acc, start)
        if np.isfinite(period_value) and period_value > 0.0:
            period = float(period_value)
    except IndexError:
        fallback_reason = fallback_reason or "step_period_unavailable"
    except KeyError:
        fallback_reason = fallback_reason or "step_period_unavailable"
    except ValueError:
        fallback_reason = fallback_reason or "step_period_unavailable"

    if "h_norm" in df_acc.columns:
        horizontal = df_acc["h_norm"].iloc[start : end + 1].dropna().to_numpy()
        horizontal_energy = (
            float(np.sqrt(np.mean(np.square(horizontal))))
            if len(horizontal) > 0
            else 0.0
        )
    else:
        horizontal_energy = 0.0
        fallback_reason = fallback_reason or "horizontal_energy_unavailable"

    duration_quality = (
        0.0 if period is None else float(np.exp(-0.5 * ((period - 0.65) / 0.35) ** 2))
    )
    sample_quality = min(sample_count / 30.0, 1.0)
    quality = float(np.clip(0.55 * sample_quality + 0.45 * duration_quality, 0.0, 1.0))
    if fallback_reason is not None:
        quality *= 0.65
    log_length_sigma = float(0.10 + 0.24 * (1.0 - quality))

    return StepLengthObservation(
        step_index=step_heading.step_index,
        nominal_length_m=float(nominal_length_m),
        interval_length_m=interval_length,
        step_period_s=period,
        vertical_amplitude=amplitude,
        horizontal_energy=horizontal_energy,
        quality=quality,
        log_length_sigma=log_length_sigma,
        fallback_reason=fallback_reason,
    )


def _estimate_initial_forward_angle(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    peaks: np.ndarray,
) -> float:
    """全ステップの変位方向の循環平均から前進方向の初期角度 φ₀ を推定する。"""
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
        times = _segment_times(df_acc, start, end)
        dy = _integrate_forward_acceleration(h_y, times)
        dz = _integrate_forward_acceleration(h_z, times)
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
    """方位方向射影による単一ステップの歩幅推定。"""
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
    times = _segment_times(df_acc, start, end)
    osc_disp = abs(_integrate_forward_acceleration(a_fwd, times))

    return K_FORWARD * osc_disp
