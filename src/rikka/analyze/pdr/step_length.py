"""PDR の歩幅推定手法。"""

import numpy as np
import pandas as pd

from ...config import (
    K_FORWARD,
    MAX_SEG_SAMPLES,
    SAMPLING_RATE,
    STEP_LENGTH_WINDOW,
    WEINBERG_K,
)
from .time_utils import _sample_gyro_angle, _time_at_index


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


def _estimate_initial_forward_angle(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    peaks: np.ndarray,
) -> float:
    """全ステップの変位方向の循環平均から前進方向の初期角度 φ₀ を推定する。"""
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
    """方位方向射影による単一ステップの歩幅推定。"""
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
