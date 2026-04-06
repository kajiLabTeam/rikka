"""Configuration file for rikka"""

# Data directory path
# Change this path to use different input data

DATA_DIR = "input/90steps_turn_elevator2"

# フロアマップ設定
# 背景として表示するフロアマップ画像のパス
FLOORMAP_PATH = "input/Floormap_building14_5floor.png"
# 軌跡の起点（原点）がフロアマップ上で対応するピクセル座標 (x_px, y_px)
# 実際の歩行開始位置に合わせて調整すること
FLOORMAP_ORIGIN_PX: tuple[int, int] = (2050, 900)
# 1ピクセルあたりのメートル数（1px = 1cm = 0.01m）
FLOORMAP_SCALE = 0.01
# 軌跡の初期方向 [度]（0 = 右方向、90 = 上方向、反時計回りが正）
INITIAL_DIRECTION = 90.0
# DATA_DIR = "input/90steps_turn_Yamamoto"

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

# ステップ検出の最小ピーク高さ [m/s²]
# ノイズ・微小な動きとステップを区別する閾値
PEAK_HEIGHT = 1.0

# Weinberg モデルのスケール係数（校正が必要なパラメータ）
# 歩幅 = K × (a_max - a_min)^0.25 の K に相当する
# ユーザーの体格・歩行スタイル・端末装着位置で個人差があるため、
# 実測の数ステップで検証してキャリブレーションすることを推奨
WEINBERG_K = 0.4

# 歩幅推定用ウィンドウ（ステップピーク前後のサンプル数）
# ±50サンプル = ±0.5秒 @ 100Hz
# スイング期（足の振り上げ〜接地）の上下動を十分に捕捉できる幅に設定
STEP_LENGTH_WINDOW = 50
