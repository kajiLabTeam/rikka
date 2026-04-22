from datetime import datetime
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.signal import find_peaks

from .config import (
    DATA_DIR,
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    INITIAL_DIRECTION,
    K_FORWARD,
    MAX_SEG_SAMPLES,
    PEAK_DISTANCE,
    PEAK_HEIGHT,
    SAMPLING_RATE,
    STEP_LENGTH_METHOD,
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


def _estimate_initial_forward_angle(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    peaks: np.ndarray,
) -> float:
    """全ステップの変位方向の循環平均から前進方向の初期角度 φ₀ を推定する。

    各ステップの h_y・h_z を2重積分した変位ベクトルを求め、
    そのステップ中点での low_angle を引いてセンサー座標系の角度を取得。
    全ステップの循環平均（sin/cos の平均 → atan2）で外れ値に頑健な推定を行う。

    Returns:
        float: センサー Y-Z 平面における前進方向の角度 [rad]
    """
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
        mid_idx = min((start + end) // 2, len(df_gyro["low_angle"]) - 1)
        angle_at_mid = float(df_gyro["low_angle"].iloc[mid_idx])
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
    """方位方向射影による単一ステップの歩幅推定。

    ジャイロから得た前進方向角（φ₀ + low_angle）へ水平加速度を射影し、
    符号付きの前進加速度を直接2重積分する。
    線形ドリフト補正で両端速度を 0 に揃えた後、K_FORWARD を乗じて歩幅を算出する。

    Args:
        df_acc: h_y・h_z 列を含む加速度 DataFrame
        df_gyro: low_angle 列を含むジャイロ DataFrame
        peaks: ステップピークのインデックス配列
        i: 現在のステップインデックス
        phi_0: センサー座標系における初期前進方向角 [rad]

    Returns:
        float: 推定歩幅 [m]（計算不能時は 0.0）
    """
    dt = 1.0 / SAMPLING_RATE
    start = int(peaks[i])
    end = int(peaks[i + 1]) if i + 1 < len(peaks) else start + 1

    seg_len = end - start
    if seg_len < 30 or seg_len > MAX_SEG_SAMPLES:
        return 0.0

    # このステップ中点での前進方向角
    mid_idx = min((start + end) // 2, len(df_gyro["low_angle"]) - 1)
    angle = float(df_gyro["low_angle"].iloc[mid_idx]) + phi_0

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


def estimate_trajectory(
    peaks: np.ndarray,
    df_gyro: pd.DataFrame,
    df_acc: pd.DataFrame,
    initial_direction: float = INITIAL_DIRECTION,
) -> tuple[list[list[float]], list[float]]:
    """ステップピークとジャイロスコープ角度から2次元軌跡を推定する。

    各ステップピーク時刻の平滑化角度（``low_angle``）と
    ``STEP_LENGTH_METHOD`` で選択した手法による歩幅推定をもとに次の座標を計算し，
    軌跡を構築する。原点 [0.0, 0.0] から始まり，ステップごとに座標を追加する。
    ``initial_direction`` を加算することで，歩行開始方向をフロアマップに合わせられる。

    Args:
        peaks (np.ndarray): ステップピークのインデックス配列
        df_gyro (pd.DataFrame): ``low_angle`` 列を含むジャイロスコープDataFrame
        df_acc (pd.DataFrame): ``h_y``・``h_z``・``h_norm`` 列を含む加速度DataFrame
        initial_direction (float): 歩行開始方向のオフセット [度]
            （デフォルト: ``INITIAL_DIRECTION``）

    Returns:
        tuple[list[list[float]], list[float]]:
            - 各ステップの [x, y] 座標リスト（原点を含む）
            - 各ステップの推定歩幅リスト [m]
    """
    points: list[list[float]] = [[0.0, 0.0]]
    step_lengths: list[float] = []
    low_angle = df_gyro["low_angle"]
    # 度 → ラジアン変換してオフセットとして使用
    direction_offset = float(np.deg2rad(initial_direction))

    # forward 手法用: 初期前進角をデータから自動推定
    phi_0 = (
        _estimate_initial_forward_angle(df_acc, df_gyro, peaks)
        if STEP_LENGTH_METHOD == "forward"
        else 0.0
    )

    for i, p in enumerate(peaks):
        if p >= len(low_angle):
            continue
        if STEP_LENGTH_METHOD == "forward" and i + 1 >= len(peaks):
            continue  # 次ピークなし：区間定義不可のためスキップ
        # 次のピークとの中点（swing 中盤）でサンプリング
        # 着地衝撃によるジャイロ揺らぎを避け、安定した進行方向角を得るため
        if i + 1 < len(peaks) and peaks[i + 1] < len(low_angle):
            mid_idx = (int(p) + int(peaks[i + 1])) // 2
        else:
            mid_idx = int(p)
        angle = low_angle.iloc[mid_idx] + direction_offset
        if np.isnan(angle):
            # NaN のステップを軌跡から除外して伝播を防ぐ
            # rolling 端部でデータ不足の場合に発生
            continue
        if STEP_LENGTH_METHOD == "forward":
            step_length = estimate_step_length_forward(df_acc, df_gyro, peaks, i, phi_0)
        else:
            step_length = estimate_step_length(df_acc, int(p))
        step_lengths.append(step_length)
        x = points[-1][0] + step_length * float(np.cos(angle))
        y = points[-1][1] + step_length * float(np.sin(angle))
        points.append([x, y])

    return points, step_lengths


def _compute_pixel_coords(
    xs: np.ndarray,
    ys: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """メートル座標をフロアマップのピクセル座標に変換する。

    重力の主成分軸（|gx| vs |gz|）でY軸反転の判定軸を自動選択する。
    """
    if abs(gx_mean) > abs(gz_mean):
        y_sign = -1 if gx_mean > 0 else 1
    else:
        y_sign = -1 if gz_mean < 0 else 1
    px = origin_px[0] + xs / scale
    py = origin_px[1] + y_sign * ys / scale
    return px, py


def plot_trajectory(
    trajectory: list[list[float]],
    gx_mean: float = 0.0,
    gz_mean: float = 0.0,
    floormap_path: str | Path = FLOORMAP_PATH,
    origin_px: tuple[int, int] = FLOORMAP_ORIGIN_PX,
    scale: float = FLOORMAP_SCALE,
    output_dir: Path | None = None,
) -> None:
    """推定した2次元歩行軌跡をフロアマップ上にプロットする。

    フロアマップ画像を背景として表示し，軌跡をピクセル座標に変換してオーバーレイする。
    重力の主成分軸（|gx| vs |gz| の大小）でY軸反転の判定軸を自動選択する。
    - |gx| > |gz|（端末を縦に持つ持ち方）: gx < 0 のとき反転（X軸が上を向いている）
    - |gz| >= |gx|（端末を平置き）        : gz > 0 のとき反転（画面が上を向いている）

    Args:
        trajectory (list[list[float]]): 各ステップの [x, y] 座標リスト（メートル）
        gx_mean (float): X軸重力成分の全サンプル平均値 [m/s²]
        gz_mean (float): Z軸重力成分の全サンプル平均値 [m/s²]
        floormap_path (str | Path): フロアマップ画像のパス
        origin_px (tuple[int, int]): 軌跡の起点に対応するピクセル座標 (x_px, y_px)
        scale (float): 1ピクセルあたりのメートル数（1px = 1cm = 0.01m）
    """
    df = pd.DataFrame(trajectory, columns=["x", "y"])

    px, py = _compute_pixel_coords(
        df["x"].to_numpy(), df["y"].to_numpy(), gx_mean, gz_mean, origin_px, scale
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    # フロアマップを背景として表示
    map_img = plt.imread(Path(floormap_path))
    ax.imshow(map_img)

    # 軌跡をグラデーション（開始:青 → 終了:赤）で描画
    n = len(px)
    norm = Normalize(vmin=0, vmax=max(n - 1, 1))
    cmap = cm.get_cmap("plasma")
    # 各ステップ間のセグメントに色を付けて LineCollection で描画
    pts = np.column_stack([px, py]).reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, zorder=2)  # type: ignore[arg-type]
    lc.set_array(np.arange(n - 1))
    ax.add_collection(lc)
    # 各ステップ点を同じカラーマップで描画
    sc = ax.scatter(px, py, c=np.arange(n), cmap=cmap, norm=norm, s=20, zorder=3)
    fig.colorbar(sc, ax=ax, label="Step")
    # 起点を強調表示
    ax.plot(px[0], py[0], "go", markersize=10, label="Start", zorder=4)

    ax.set_title("Walking Trajectory on Floormap")
    ax.legend()
    plt.tight_layout()
    if output_dir is not None:
        # グラフ画像をoutputフォルダに保存
        fig.savefig(output_dir / "trajectory.png", dpi=150, bbox_inches="tight")
    plt.show()


def run(
    df_acc: pd.DataFrame | None = None,
    df_gyro: pd.DataFrame | None = None,
    plot: bool = True,
    use_particle_filter: bool = False,
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    if (df_acc is None) != (df_gyro is None):
        raise ValueError("df_acc と df_gyro は両方渡すか、両方省略してください。")

    if df_acc is None and df_gyro is None:
        df_acc, df_gyro = load_sensor_data()

    if df_acc is None or df_gyro is None:
        raise RuntimeError("内部エラー: df_acc または df_gyro が None（到達不能）")

    df_acc, df_gyro = process_sensor_data(df_acc, df_gyro)
    peaks = detect_steps(df_acc)

    # 重力成分の平均を算出（Y軸反転の自動判定に使用）
    gx_mean = float(df_acc["gx"].mean())
    gz_mean = float(df_acc["gz"].mean())
    dominant = "X軸" if abs(gx_mean) > abs(gz_mean) else "Z軸"
    y_flipped = (abs(gx_mean) > abs(gz_mean) and gx_mean > 0) or (
        abs(gz_mean) >= abs(gx_mean) and gz_mean < 0
    )
    print(
        f"重力主成分: {dominant}  gx={gx_mean:.2f}, gz={gz_mean:.2f} m/s²"
        f" → Y軸{'反転' if y_flipped else '非反転'}"
    )

    if use_particle_filter:
        from .particle_filter import (  # noqa: PLC0415
            plot_particle_filter_trajectory,
            run_particle_filter,
            save_particle_animation,
        )

        trajectory, step_lengths, all_particles = run_particle_filter(
            peaks, df_gyro, df_acc, gx_mean, gz_mean
        )

        print(f"Steps detected: {len(peaks)}")
        for i, (x, y) in enumerate(trajectory):
            print(f"step {i}: ({x:.3f}, {y:.3f})")

        df_trajectory = pd.DataFrame(trajectory, columns=["x", "y"])
        output_path = output_dir / "trajectory.csv"
        df_trajectory.to_csv(output_path, index=False)
        print(f"Trajectory saved to {output_path}")

        df_step_lengths = pd.DataFrame(
            {"step": range(1, len(step_lengths) + 1), "step_length_m": step_lengths}
        )
        step_length_path = output_dir / "step_lengths.csv"
        df_step_lengths.to_csv(step_length_path, index=False)
        print(f"Step lengths saved to {step_length_path}")

        if plot:
            plot_particle_filter_trajectory(
                trajectory, gx_mean=gx_mean, gz_mean=gz_mean, output_dir=output_dir
            )

        save_particle_animation(
            all_particles,
            trajectory,
            gx_mean=gx_mean,
            gz_mean=gz_mean,
            output_path=output_dir / "particle_filter.mp4",
        )
    else:
        trajectory, step_lengths = estimate_trajectory(peaks, df_gyro, df_acc)

        print(f"Steps detected: {len(peaks)}")
        for i, (x, y) in enumerate(trajectory):
            print(f"step {i}: ({x:.3f}, {y:.3f})")

        df_trajectory = pd.DataFrame(trajectory, columns=["x", "y"])

        # 軌跡データをoutputフォルダにCSVとして保存
        output_path = output_dir / "trajectory.csv"
        df_trajectory.to_csv(output_path, index=False)
        print(f"Trajectory saved to {output_path}")

        # 歩幅データをoutputフォルダにCSVとして保存
        df_step_lengths = pd.DataFrame(
            {"step": range(1, len(step_lengths) + 1), "step_length_m": step_lengths}
        )
        step_length_path = output_dir / "step_lengths.csv"
        df_step_lengths.to_csv(step_length_path, index=False)
        print(f"Step lengths saved to {step_length_path}")

        if plot:
            plot_trajectory(
                trajectory, gx_mean=gx_mean, gz_mean=gz_mean, output_dir=output_dir
            )

    return df_trajectory
