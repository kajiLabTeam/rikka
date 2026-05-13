"""Configuration file for rikka"""

# Data directory path
# Change this path to use different input data

DATA_DIR = "input/87steps_turn_Yamamoto"

# フロアマップ設定
# 背景として表示するフロアマップ画像のパス
FLOORMAP_PATH = "input/Floormap_building14_5floor.png"
# 軌跡の起点（原点）がフロアマップ上で対応するピクセル座標 (x_px, y_px)
# 実際の歩行開始位置に合わせて調整すること
FLOORMAP_ORIGIN_PX: tuple[int, int] = (2050, 400)
# 1ピクセルあたりのメートル数（1px = 1cm = 0.01m）

FLOORMAP_SCALE = 0.01
# 軌跡の初期方向 [度]（0 = 右方向、90 = 上方向、反時計回りが正）
INITIAL_DIRECTION = 90.0

# センサーのサンプリングレート [Hz]、角速度積分・時間換算に使用
SAMPLING_RATE = 100

# 加速度 LPF のウィンドウサイズ（80サンプル = 0.8秒 @ 100Hz）
# 重力ベクトル推定・低周波成分の平滑化に共用
WINDOW_ACC = 80

# ジャイロ LPF・静止区間検出のウィンドウサイズ（40サンプル = 0.4秒）
WINDOW_GYRO = 40

# ステップ間の最小サンプル数（50サンプル = 0.5秒）
# 1ステップの最短継続時間を保証し、重複検出を防ぐ
PEAK_DISTANCE = 50

# ステップ区間の最大サンプル数（80サンプル = 0.8秒 @ 100Hz）
# これを超える区間はピーク未検出による2ステップ合算とみなして除外する
# 通常の歩行ステップは 50〜71 サンプル（0.5〜0.71秒）
MAX_SEG_SAMPLES = 80

# ステップ検出の最小ピーク高さ [m/s²]
# ノイズ・微小な動きとステップを区別する閾値
PEAK_HEIGHT = 1.0

# ユーザー身長 [m]
# Weinbergモデルの歩幅スケールは身長におおむね比例すると仮定して補正する
USER_HEIGHT_M = 1.65

# Weinberg モデルの基準スケール係数（校正が必要なパラメータ）
# 歩幅 = K × (a_max - a_min)^0.25 の K に相当する
# WEINBERG_REFERENCE_HEIGHT_M のユーザーで校正した値として扱う
WEINBERG_REFERENCE_HEIGHT_M = 1.70
WEINBERG_REFERENCE_K = 0.47


def compute_weinberg_k(height_m: float = USER_HEIGHT_M) -> float:
    """身長に応じた Weinberg モデルのスケール係数を返す。"""
    if height_m <= 0:
        raise ValueError("height_m は正の値を指定してください。")
    return WEINBERG_REFERENCE_K * (height_m / WEINBERG_REFERENCE_HEIGHT_M)


WEINBERG_K = compute_weinberg_k(USER_HEIGHT_M)

# 歩幅推定用ウィンドウ（ステップピーク前後のサンプル数）
# ±50サンプル = ±0.5秒 @ 100Hz
# スイング期（足の振り上げ〜接地）の上下動を十分に捕捉できる幅に設定
STEP_LENGTH_WINDOW = 50

# 歩幅推定手法の選択
# "weinberg": Weinbergモデル（経験則による max-min 振幅から推定）
# "forward" : 方位方向射影積分（ジャイロから前進方向を自動推定し符号付き成分を2重積分）
STEP_LENGTH_METHOD = "weinberg"

# 前進方向射影積分のユニバーサルスケール係数
# 歩幅 = K_FORWARD × 振動変位
# 振動変位 ≈ 0.13 m（体装着センサーの直流速度成分が計測不能なため）
# K = v_mean / Δv_peak × 2/π ≈ 定数（v_mean と Δv が歩行速度に比例するため）
K_FORWARD = 9.0

# パーティクルフィルタ設定
PF_NUM_PARTICLES = 500
PF_SIGMA_INIT_HEADING = 0.15  # 初期方向ばらつき [rad]（±9°）
PF_SIGMA_HEADING = 0.05  # ステップごとの方位角ドリフト [rad/step]（±3°）
PF_SIGMA_STEP_LENGTH_RATIO = 0.08  # ステップ長ノイズ比率（±8%）
