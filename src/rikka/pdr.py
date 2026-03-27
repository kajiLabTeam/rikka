from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .config import DATA_DIR, STEP_LENGTH_WINDOW, WEINBERG_K

# Constants
WINDOW_ACC = 80
WINDOW_GYRO = 40
ANGLE_SCALE = 1.0
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
    # 移動平均（重力推定値）を引いて動的加速度ノルムを算出
    df_acc["linear_norm"] = df_acc["norm"] - df_acc["low_norm"]

    df_gyro["norm"] = np.sqrt(df_gyro["x"] ** 2 + df_gyro["y"] ** 2 + df_gyro["z"] ** 2)
    # 最小分散区間（静止期間）を自動検出してバイアスを推定する
    # 全体 mean はターン動作の信号が混入するため使用しない
    _rolling_var = df_gyro["x"].rolling(window=WINDOW_GYRO).var()
    _quiet_end = int(_rolling_var.idxmin())
    _quiet_start = max(0, _quiet_end - WINDOW_GYRO)
    gyro_bias = float(df_gyro["x"].iloc[_quiet_start:_quiet_end].mean())
    df_gyro["angle"] = np.cumsum(df_gyro["x"] - gyro_bias) / SAMPLING_RATE
    df_gyro["low_angle"] = (
        df_gyro["angle"].rolling(window=WINDOW_GYRO, center=True, min_periods=1).mean()
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
        df_acc["low_norm"].to_numpy(),
        distance=PEAK_DISTANCE,
        height=PEAK_HEIGHT,
    )
    return np.asarray(peaks)


def estimate_step_length(
    df_acc: pd.DataFrame,
    peak_index: int,
    window: int = STEP_LENGTH_WINDOW,
    k: float = WEINBERG_K,
) -> float:
    """Weinberg モデルによる単一ステップの歩幅推定。

    ピークインデックスの前後 ``window`` サンプルの範囲内で
    重力除去済み加速度ノルム（``linear_norm``）の最大値・最小値を求め，
    Weinberg 式で歩幅を計算する。
    ウィンドウがデータ範囲外にかかる場合はクリッピングする。

    Args:
        df_acc (pd.DataFrame): ``linear_norm`` 列を含む加速度DataFrame
        peak_index (int): ステップピークのインデックス
        window (int): ピーク前後のサンプル数
        k (float): Weinberg モデルのスケール係数

    Returns:
        float: 推定歩幅（メートル）
    """
    n = len(df_acc)
    start = max(0, peak_index - window)
    end = min(n, peak_index + window + 1)
    segment = df_acc["linear_norm"].iloc[start:end].dropna()
    acc_max = float(segment.max())
    acc_min = float(segment.min())
    return float(k * (acc_max - acc_min) ** 0.25)


def estimate_trajectory(
    peaks: np.ndarray, df_gyro: pd.DataFrame, df_acc: pd.DataFrame
) -> list[list[float]]:
    """ステップピークとジャイロスコープ角度から2次元軌跡を推定する。

    各ステップピーク時刻の平滑化角度（``low_angle``）と
    Weinberg モデルによる動的歩幅推定をもとに次の座標を計算し，軌跡を構築する。
    原点 [0.0, 0.0] から始まり，ステップごとに座標を追加する。

    Args:
        peaks (np.ndarray): ステップピークのインデックス配列
        df_gyro (pd.DataFrame): ``low_angle`` 列を含むジャイロスコープDataFrame
        df_acc (pd.DataFrame): ``linear_norm`` 列を含む加速度DataFrame

    Returns:
        list[list[float]]: 各ステップの [x, y] 座標リスト（原点を含む）
    """
    points: list[list[float]] = [[0.0, 0.0]]
    low_angle = df_gyro["low_angle"]

    for i, p in enumerate(peaks):
        if p >= len(low_angle):
            continue
        # 次のピークとの中点（mid-swing 相当）でサンプリングして着地衝撃ノイズを除去
        if i + 1 < len(peaks) and peaks[i + 1] < len(low_angle):
            mid_idx = (int(p) + int(peaks[i + 1])) // 2
        else:
            mid_idx = int(p)
        angle = low_angle.iloc[mid_idx] * ANGLE_SCALE
        if np.isnan(angle):
            continue  # NaN のステップを軌跡から除外して伝播を防ぐ
        step_length = estimate_step_length(df_acc, int(p))
        x = points[-1][0] + step_length * float(np.cos(angle))
        y = points[-1][1] + step_length * float(np.sin(angle))
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


def run(
    df_acc: pd.DataFrame | None = None,
    df_gyro: pd.DataFrame | None = None,
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

    Returns:
        pd.DataFrame: 軌跡データ（列: x, y）

    Raises:
        ValueError: ``df_acc`` と ``df_gyro`` の片方だけが渡された場合
    """
    if (df_acc is None) != (df_gyro is None):
        raise ValueError("df_acc と df_gyro は両方渡すか、両方省略してください。")

    if df_acc is None and df_gyro is None:
        df_acc, df_gyro = load_sensor_data()

    assert df_acc is not None
    assert df_gyro is not None
    df_acc, df_gyro = process_sensor_data(df_acc, df_gyro)
    peaks = detect_steps(df_acc)
    trajectory = estimate_trajectory(peaks, df_gyro, df_acc)

    print(f"Steps detected: {len(peaks)}")
    for i, (x, y) in enumerate(trajectory):
        print(f"step {i}: ({x:.3f}, {y:.3f})")

    df_trajectory = pd.DataFrame(trajectory, columns=["x", "y"])

    # 軌跡データをoutputディレクトリにCSVとして保存
    output_path = Path("output/trajectory.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_trajectory.to_csv(output_path, index=False)
    print(f"Trajectory saved to {output_path}")

    plot_trajectory(trajectory)

    return df_trajectory
