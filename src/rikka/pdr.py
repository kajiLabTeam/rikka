from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .config import DATA_DIR

# Constants
WINDOW_ACC = 80
WINDOW_GYRO = 40
STEP_LENGTH = 0.3  # meters
ANGLE_SCALE = 1.2
PEAK_DISTANCE = 50
PEAK_HEIGHT = 10
SAMPLING_RATE = 100

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


def load_sensor_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """CSVファイルから加速度計とジャイロスコープのデータを読み込む。

    ``DATA_DIR`` に設定されたディレクトリ内の ``Accelerometer.csv`` と
    ``Gyroscope.csv`` を読み込み，列名を統一した形式に変換して返す。

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - df_acc: 加速度データ（列: t, x, y, z）
            - df_gyro: ジャイロスコープデータ（列: t, x, y, z）
    """
    data_path = Path(DATA_DIR)
    df_acc = pd.read_csv(data_path / "Accelerometer.csv").rename(columns=ACC_COLUMNS)
    df_gyro = pd.read_csv(data_path / "Gyroscope.csv").rename(columns=GYRO_COLUMNS)
    return df_acc, df_gyro


def process_sensor_data(
    df_acc: pd.DataFrame, df_gyro: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生センサーデータからノルム・移動平均・角度を計算する。

    加速度データにはノルムと移動平均（``low_norm``）を，ジャイロスコープデータには
    ノルム・積算角度（``angle``）・移動平均角度（``low_angle``）を追加する。

    Args:
        df_acc (pd.DataFrame): 加速度データ（列: t, x, y, z）
        df_gyro (pd.DataFrame): ジャイロスコープデータ（列: t, x, y, z）

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - df_acc:
            ``norm``（合成加速度）と ``low_norm``（平滑化ノルム）を追加した加速度Data
            - df_gyro:
            ``norm``（合成角速度）・``angle``（積算角度）・``low_angle``（平滑化角度）を追加したジャイロData
    """
    df_acc = df_acc.copy()
    df_gyro = df_gyro.copy()

    df_acc["norm"] = np.sqrt(df_acc["x"] ** 2 + df_acc["y"] ** 2 + df_acc["z"] ** 2)
    df_acc["low_norm"] = df_acc["norm"].rolling(window=WINDOW_ACC).mean()

    df_gyro["norm"] = np.sqrt(df_gyro["x"] ** 2 + df_gyro["y"] ** 2 + df_gyro["z"] ** 2)
    df_gyro["angle"] = np.cumsum(df_gyro["x"]) / SAMPLING_RATE
    df_gyro["low_angle"] = (
        df_gyro["angle"].rolling(window=WINDOW_GYRO, center=True).mean()
    )

    return df_acc, df_gyro


def detect_steps(df_acc: pd.DataFrame) -> np.ndarray:
    """平滑化した加速度ノルムから歩行ステップのピークを検出する。

    ``low_norm`` 列に対してピーク検出を行い，ステップに対応するインデックスを返す。
    ピーク間距離 ``PEAK_DISTANCE`` と最小高さ ``PEAK_HEIGHT`` でフィルタリングする。

    Args:
        df_acc (pd.DataFrame): ``low_norm`` 列を含む加速度DataFrame

    Returns:
        np.ndarray: ステップピークのインデックス配列
    """
    peaks, _ = find_peaks(
        df_acc["low_norm"].dropna(),
        distance=PEAK_DISTANCE,
        height=PEAK_HEIGHT,
    )
    return peaks


def estimate_trajectory(peaks: np.ndarray, df_gyro: pd.DataFrame) -> list[list[float]]:
    """ステップピークとジャイロスコープ角度から2次元軌跡を推定する。

    各ステップピーク時刻の平滑化角度（``low_angle``）をもとに，
    一定のステップ長（``STEP_LENGTH``）で次の座標を計算し，軌跡を構築する。
    原点 [0.0, 0.0] から始まり，ステップごとに座標を追加する。

    Args:
        peaks (np.ndarray): ステップピークのインデックス配列
        df_gyro (pd.DataFrame): ``low_angle`` 列を含むジャイロスコープDataFrame

    Returns:
        list[list[float]]: 各ステップの [x, y] 座標リスト（原点を含む）
    """
    points: list[list[float]] = [[0.0, 0.0]]
    low_angle = df_gyro["low_angle"]

    for p in peaks:
        if p >= len(low_angle):
            continue
        angle = low_angle.iloc[p] * ANGLE_SCALE
        x = points[-1][0] + STEP_LENGTH * float(np.cos(angle))
        y = points[-1][1] + STEP_LENGTH * float(np.sin(angle))
        points.append([x, y])

    return points


def plot_trajectory(trajectory: list[list[float]]) -> None:
    """推定した2次元歩行軌跡をプロットする。

    軌跡の各座標を折れ線グラフで描画し，縦横比を等倍に固定して表示する。

    Args:
        trajectory (list[list[float]]): 各ステップの [x, y] 座標リスト
    """
    df = pd.DataFrame(trajectory, columns=["x", "y"])

    plt.figure(figsize=(8, 8))
    plt.plot(df["x"], df["y"], ".-", label="Estimated trajectory", zorder=1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Walking Trajectory")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run() -> None:
    """PDRのメインパイプラインを実行する。

    センサーデータの読み込みから軌跡の推定・表示までを一括して実行する。
    処理の流れ: データ読み込み → センサー処理 → ステップ検出 → 軌跡推定 → 表示
    """
    df_acc, df_gyro = load_sensor_data()
    df_acc, df_gyro = process_sensor_data(df_acc, df_gyro)
    peaks = detect_steps(df_acc)
    trajectory = estimate_trajectory(peaks, df_gyro)

    print(f"Steps detected: {len(peaks)}")
    for i, (x, y) in enumerate(trajectory):
        print(f"step {i}: ({x:.3f}, {y:.3f})")

    plot_trajectory(trajectory)
