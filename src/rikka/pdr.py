from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .config import (
    DATA_DIR,
    PEAK_DISTANCE,
    PEAK_HEIGHT,
    SAMPLING_RATE,
    STEP_LENGTH_WINDOW,
    WEINBERG_K,
    WINDOW_ACC,
    WINDOW_GYRO,
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


def process_sensor_data(
    df_acc: pd.DataFrame, df_gyro: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生センサーデータからノルム・重力推定・水平加速度・角度を計算する。

    加速度データには以下の列を追加する：
    - ``gx, gy, gz``: 各軸のLPFによる重力ベクトル推定値
    - ``lin_x, lin_y, lin_z``: ベクトル減算による線形加速度
    - ``lin_norm``: 線形加速度ノルム
    - ``low_lin_norm``: 平滑化線形加速度ノルム（ステップ検出用）
    - ``h_x, h_y, h_z``: 重力方向を射影除去した水平加速度成分
    - ``h_norm``: 水平加速度ノルム（歩幅推定用）

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
    df_acc = df_acc.copy()
    df_gyro = df_gyro.copy()

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

    # 水平加速度: a_h = a_lin − (a_lin · ĝ) ĝ（重力方向成分を射影で除去）
    # dot = a_lin · ĝ：線形加速度の重力方向成分（スカラー）
    dot = (
        df_acc["lin_x"] * df_acc["gx_hat"]
        + df_acc["lin_y"] * df_acc["gy_hat"]
        + df_acc["lin_z"] * df_acc["gz_hat"]
    )
    df_acc["h_x"] = df_acc["lin_x"] - dot * df_acc["gx_hat"]
    df_acc["h_y"] = df_acc["lin_y"] - dot * df_acc["gy_hat"]
    df_acc["h_z"] = df_acc["lin_z"] - dot * df_acc["gz_hat"]
    # 水平加速度ノルム: 端末傾斜に依らない歩行動作の強度指標 [m/s²]（歩幅推定に使用）
    df_acc["h_norm"] = np.sqrt(
        df_acc["h_x"] ** 2 + df_acc["h_y"] ** 2 + df_acc["h_z"] ** 2
    )

    # 最小分散区間（静止期間）を自動検出してバイアスを推定する
    # 全体 mean はターン動作の信号が混入するため使用しない
    # 分散が最小のウィンドウが最も静止に近い区間
    _rolling_var = df_gyro["x"].rolling(window=WINDOW_GYRO).var()
    if _rolling_var.notna().any():
        # rolling(window=W) のインデックス i は [i-W+1, i] の W 個を表す
        _quiet_end = int(_rolling_var.idxmin())
        _quiet_start = max(0, _quiet_end - WINDOW_GYRO + 1)
        # 静止区間の平均値をドリフトオフセットとして使用 [rad/s]
        gyro_bias = float(df_gyro["x"].iloc[_quiet_start : _quiet_end + 1].mean())
    else:
        # データ長が WINDOW_GYRO 未満で全 NaN になる場合
        # 全サンプルの平均をフォールバックとして使用
        gyro_bias = float(df_gyro["x"].mean())
    df_gyro["angle"] = np.cumsum(df_gyro["x"] - gyro_bias) / SAMPLING_RATE
    df_gyro["low_angle"] = (
        df_gyro["angle"].rolling(window=WINDOW_GYRO, center=True, min_periods=1).mean()
    )

    return df_acc, df_gyro


def detect_steps(df_acc: pd.DataFrame) -> np.ndarray:
    """平滑化した線形加速度ノルムから歩行ステップのピークを検出する。

    ``low_lin_norm`` 列に対してピーク検出を行い，ステップに対応するインデックスを返す。
    ピーク間距離 ``PEAK_DISTANCE`` と最小高さ ``PEAK_HEIGHT`` でフィルタリングする。

    Args:
        df_acc (pd.DataFrame): ``low_lin_norm`` 列を含む加速度DataFrame

    Returns:
        np.ndarray: ステップピークのインデックス配列
    """
    peaks, _ = find_peaks(
        df_acc["low_lin_norm"].to_numpy(),
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
    水平加速度ノルム（``h_norm``）の最大値・最小値を求め，
    Weinberg 式で歩幅を計算する。
    端末傾斜に対してロバストな水平成分を使用することで推定精度を向上させる。
    ウィンドウがデータ範囲外にかかる場合はクリッピングする。

    Args:
        df_acc (pd.DataFrame): ``h_norm`` 列を含む加速度DataFrame
        peak_index (int): ステップピークのインデックス
        window (int): ピーク前後のサンプル数
        k (float): Weinberg モデルのスケール係数

    Returns:
        float: 推定歩幅（メートル）
    """
    n = len(df_acc)
    start = max(0, peak_index - window)
    end = min(n, peak_index + window + 1)
    # ウィンドウ内の水平加速度ノルム（NaN除去済み）
    segment = df_acc["h_norm"].iloc[start:end].dropna()
    if segment.empty:
        return 0.0  # データ不足のステップは歩幅 0 として軌跡から実質除外
    acc_max = float(segment.max())  # h_norm 最大値 [m/s²]（スイング頂点付近）
    acc_min = float(segment.min())  # h_norm 最小値 [m/s²]（接地付近）
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
        df_acc (pd.DataFrame): ``h_norm`` 列を含む加速度DataFrame

    Returns:
        list[list[float]]: 各ステップの [x, y] 座標リスト（原点を含む）
    """
    points: list[list[float]] = [[0.0, 0.0]]
    low_angle = df_gyro["low_angle"]

    for i, p in enumerate(peaks):
        if p >= len(low_angle):
            continue
        # 次のピークとの中点（swing 中盤）でサンプリング
        # 着地衝撃によるジャイロ揺らぎを避け、安定した進行方向角を得るため
        if i + 1 < len(peaks) and peaks[i + 1] < len(low_angle):
            mid_idx = (int(p) + int(peaks[i + 1])) // 2
        else:
            mid_idx = int(p)
        angle = low_angle.iloc[mid_idx]
        if np.isnan(angle):
            # NaN のステップを軌跡から除外して伝播を防ぐ
            # rolling 端部でデータ不足の場合に発生
            continue
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
    plot: bool = True,
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

    Returns:
        pd.DataFrame: 軌跡データ（列: x, y）

    Raises:
        ValueError: ``df_acc`` と ``df_gyro`` の片方だけが渡された場合
    """
    if (df_acc is None) != (df_gyro is None):
        raise ValueError("df_acc と df_gyro は両方渡すか、両方省略してください。")

    if df_acc is None and df_gyro is None:
        df_acc, df_gyro = load_sensor_data()

    if df_acc is None or df_gyro is None:
        raise RuntimeError("内部エラー: df_acc または df_gyro が None（到達不能）")

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

    if plot:
        plot_trajectory(trajectory)

    return df_trajectory
