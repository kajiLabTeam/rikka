"""PDR の時刻・サンプル位置ユーティリティ。

役割:
    DataFrame の実時刻とサンプル番号を統一的に扱い、積分刻み、ステップ中央時刻、
    出力時刻、任意時刻のジャイロ角を計算する。
依存元:
    ``config`` からサンプリングレートと歩幅方式を取得し、NumPy と Pandas の配列・
    補間処理を利用する。
利用先:
    ``sensors``、``gyro_bias``、``heading``、``step_length``、``trajectory``、
    ``outputs`` と ``particle_api`` が時刻処理の共通実装として使用する。
処理フロー:
    ``t`` 列があれば実測時刻を優先し、なければ設定レートから時刻を合成して、
    インデックス境界を処理した値を返す。
"""

import numpy as np
import pandas as pd

from ...config import SAMPLING_RATE, STEP_LENGTH_METHOD


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
