"""センサーデータの読み込みと前処理。

役割:
    phyphox 形式の加速度・ジャイロ CSV を標準列名へ正規化し、重力・線形加速度、
    平滑化信号、ジャイロバイアス補正済み積算角を作る。
依存元:
    ``config`` から入力先・窓幅・補正方式、``gyro_bias`` から偏差推定、
    ``time_utils`` から積分刻みを取得し、NumPy と Pandas で信号を処理する。
利用先:
    ``pipeline`` がファイル入力に、``trajectory.prepare_pdr_steps`` が前処理に、
    ``sensor_plot`` が可視化用データ作成に使用する。
処理フロー:
    CSV 読み込みと必須列検証後、加速度を重力成分と線形成分へ分離し、ジャイロの
    定常偏差を除いて時間積分し、解析列を追加した2つの DataFrame を返す。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ...analyze.pdr.gyro_bias import estimate_gyro_bias
from ..config import (
    DATA_DIR,
    GYRO_BIAS_METHOD,
    WINDOW_ACC,
    WINDOW_GYRO,
)
from .time_utils import _gyro_integration_dt

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


def load_sensor_data(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CSVファイルから加速度計とジャイロスコープのデータを読み込む。"""
    data_path = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    df_acc = pd.read_csv(data_path / "Accelerometer.csv").rename(columns=ACC_COLUMNS)
    df_gyro = pd.read_csv(data_path / "Gyroscope.csv").rename(columns=GYRO_COLUMNS)

    # 必須列の存在確認（列名揺れや欠損時に後段で KeyError になるのを防ぐ）
    required = {"x", "y", "z"}
    missing_acc = required - set(df_acc.columns)
    if missing_acc:
        raise ValueError(f"Accelerometer.csv に必須列がありません: {missing_acc}")
    missing_gyro = required - set(df_gyro.columns)
    if missing_gyro:
        raise ValueError(f"Gyroscope.csv に必須列がありません: {missing_gyro}")

    return df_acc, df_gyro


def process_sensor_data(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    gyro_bias_method: str | None = None,
    gyro_bias: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生センサーデータからノルム・重力推定・上下/水平加速度・角度を計算する。"""
    df_acc = df_acc.copy().reset_index(drop=True)
    df_gyro = df_gyro.copy().reset_index(drop=True)

    # 3軸それぞれにLPFをかけて重力ベクトルを推定（スカラーノルムではなくベクトルで推定）
    # center=True で対称ウィンドウを使用し、位相遅れなく重力方向を推定する
    roll_args = {"window": WINDOW_ACC, "center": True, "min_periods": 1}
    df_acc["gx"] = df_acc["x"].rolling(**roll_args).mean()
    df_acc["gy"] = df_acc["y"].rolling(**roll_args).mean()
    df_acc["gz"] = df_acc["z"].rolling(**roll_args).mean()

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
