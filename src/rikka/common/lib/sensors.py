"""センサーデータの読み込みと前処理。

役割:
    phyphox形式またはAndroidロガー形式のCSVを標準列名へ正規化し、重力・線形加速度、
    平滑化信号、ジャイロバイアス補正済み積算角を作る。
依存元:
    ``config`` から入力先・窓幅・補正方式、``gyro_bias`` から偏差推定、
    ``time_utils`` から積分刻みを取得し、NumPy と Pandas で信号を処理する。
    Androidロガーの記録開始時刻は同じ入力ディレクトリのmetadata.jsonから取得する。
利用先:
    ``pipeline`` がファイル入力に、``trajectory.prepare_pdr_steps`` が前処理に、
    ``sensor_plot`` が可視化用データ作成に使用する。
処理フロー:
    CSV読み込みと必須列検証後、前処理する軸の非有限値を拒否する。
    Android形式では開始壁時刻から先頭オフセットを求め、以降を単調時計差で秒へ変換する。
    加速度を重力成分と線形成分へ分離し、ジャイロの
    定常偏差を除いて時間積分し、解析列を追加した2つの DataFrame を返す。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    DATA_DIR,
    GYRO_BIAS_METHOD,
    WINDOW_ACC,
    WINDOW_GYRO,
)
from .gyro_bias import estimate_gyro_bias
from .time_utils import _gyro_integration_dt, _sample_count_for_reference_rate

ACC_COLUMNS = {
    "Time (s)": "t",
    "Experiment Time (s)": "t",
    "Acceleration x (m/s^2)": "x",
    "Acceleration y (m/s^2)": "y",
    "Acceleration z (m/s^2)": "z",
    "Acceleration X (m/s^2)": "x",
    "Acceleration Y (m/s^2)": "y",
    "Acceleration Z (m/s^2)": "z",
    "X (m/s^2)": "x",
    "Y (m/s^2)": "y",
    "Z (m/s^2)": "z",
}
GYRO_COLUMNS = {
    "Time (s)": "t",
    "Experiment Time (s)": "t",
    "Gyroscope x (rad/s)": "x",
    "Gyroscope y (rad/s)": "y",
    "Gyroscope z (rad/s)": "z",
    "Gyroscope X (rad/s)": "x",
    "Gyroscope Y (rad/s)": "y",
    "Gyroscope Z (rad/s)": "z",
    "X (rad/s)": "x",
    "Y (rad/s)": "y",
    "Z (rad/s)": "z",
}


def _load_android_logger(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ロガーの単調時計を、記録開始時刻を原点とする秒へ変換する。"""
    try:
        metadata = json.loads((data_path / "metadata.json").read_text())
        start_wall_ms = float(metadata["recording_start_wall_time_ms"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            "Androidロガー入力にはmetadata.jsonのrecording_start_wall_time_msが必要です"
        ) from exc
    if not np.isfinite(start_wall_ms):
        raise ValueError("metadata.jsonの記録開始時刻は有限値にしてください")
    frames = []
    for filename in ("acce.csv", "gyro.csv"):
        raw = pd.read_csv(data_path / filename)
        missing = {"timestamp_ns", "wall_time_ms", "x", "y", "z"} - set(raw.columns)
        if missing or raw.empty:
            raise ValueError(f"{filename}が空、または必須列がありません: {missing}")
        nanoseconds = pd.to_numeric(raw["timestamp_ns"], errors="raise").to_numpy()
        first_wall_ms = float(raw["wall_time_ms"].iloc[0])
        if (
            not np.all(np.isfinite(nanoseconds))
            or not np.isfinite(first_wall_ms)
            or np.any(np.diff(nanoseconds) <= 0)
        ):
            raise ValueError(f"{filename}の時刻は有限かつ単調増加にしてください")
        frame = raw[["x", "y", "z"]].astype(float).copy()
        # 壁時計の行ごとの丸めを使わず、センサー間の先頭時刻差は保持する。
        frame.insert(
            0,
            "t",
            (nanoseconds - nanoseconds[0]) / 1e9
            + (first_wall_ms - start_wall_ms) / 1000.0,
        )
        frames.append(frame)
    return frames[0], frames[1]


def load_sensor_data(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CSVファイルから加速度計とジャイロスコープのデータを読み込む。"""
    data_path = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    if (
        not (data_path / "Accelerometer.csv").exists()
        and not (data_path / "Gyroscope.csv").exists()
        and (data_path / "acce.csv").is_file()
        and (data_path / "gyro.csv").is_file()
    ):
        return _load_android_logger(data_path)
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

    # 平滑化による欠損の隠蔽と、積分による非有限値の後続全体への伝播を防ぐ。
    # 直接呼び出しではジャイロの x 軸だけを渡す既存の利用方法も維持する。
    for sensor_name, frame in (("加速度", df_acc), ("ジャイロ", df_gyro)):
        for axis in ("x", "y", "z"):
            if axis not in frame.columns:
                continue
            values = frame[axis].to_numpy(dtype=float, na_value=np.nan)
            invalid_rows = np.flatnonzero(~np.isfinite(values))
            if invalid_rows.size:
                raise ValueError(
                    f"{sensor_name}の {axis} 列に非有限値があります "
                    f"(行位置 {int(invalid_rows[0])}、0始まり)"
                )

    # 3軸それぞれにLPFをかけて重力ベクトルを推定（スカラーノルムではなくベクトルで推定）
    # 実時刻で0.8秒相当へ換算した対称ウィンドウを使い、位相遅れを防ぐ
    acc_window = _sample_count_for_reference_rate(df_acc, WINDOW_ACC)
    roll_args = {"window": acc_window, "center": True, "min_periods": 1}
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
    # 歩検出信号は既存100 Hzデータの挙動を変えないサンプル窓を維持する。
    # 高レート入力でこの窓まで0.8秒へ拡大すると実測歩を結合するため、
    # サンプリング周波数に応じたピーク間隔とは別の役割として扱う。
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
    gyro_window = _sample_count_for_reference_rate(df_gyro, WINDOW_GYRO)
    df_gyro["low_angle"] = (
        df_gyro["angle"]
        .rolling(
            window=gyro_window,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    return df_acc, df_gyro
