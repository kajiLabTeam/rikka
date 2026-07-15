"""rikka 全体の既定設定。

役割:
    入力データ、フロアマップ、信号処理、ステップ・方位・歩幅推定、横歩き判定、
    particle filter の既定値を一元管理する。
依存元:
    標準ライブラリの ``math.isfinite`` だけを使用し、他のプロジェクトモジュールには
    依存しない。
利用先:
    CLI とすべての解析モジュールが関数引数の既定値や判定パラメータとして使用する。
    値の変更は CLI の初期挙動にも反映される。
処理フロー:
    定数を宣言し、``compute_weinberg_k`` で利用者の身長から歩幅係数を検証・算出する。
"""

from math import isfinite

# Data directory path
# Change this path to use different input data

DATA_DIR = "input/sensor_data/1turn_rightsidestep_3turn_leftsidestep5"

# フロアマップ設定
# 背景として表示するフロアマップ画像のパス
FLOORMAP_PATH = "input/Floormap_building14_5floor.png"
# 軌跡の起点（原点）がフロアマップ上で対応するピクセル座標 (x_px, y_px)
# 実際の歩行開始位置に合わせて調整すること
FLOORMAP_ORIGIN_PX: tuple[int, int] = (2050, 400)
# 1ピクセルあたりのメートル数（1px = 1cm = 0.01m）

FLOORMAP_SCALE = 0.01
# 軌跡の初期方向 [度]（0 = +X、90 = +Y、反時計回りが正）
# フロアマップ画像上の上下は端末姿勢から決まる Y 軸反転設定に依存する
INITIAL_DIRECTION = 90.0

# センサーのサンプリングレート [Hz]、角速度積分・時間換算に使用
SAMPLING_RATE = 100

# 加速度 LPF のウィンドウサイズ（80サンプル = 0.8秒 @ 100Hz）
# 重力ベクトル推定・低周波成分の平滑化に共用
WINDOW_ACC = 80

# ジャイロ LPF・静止区間検出のウィンドウサイズ（40サンプル = 0.4秒）
WINDOW_GYRO = 40

# ジャイロバイアス推定手法
# "prewalk_robust": 歩行開始直前の区間からロバスト推定
# "initial_robust": 記録先頭の短い区間からロバスト推定
# "quietest"       : 既存方式（全期間で分散最小の窓を使用）
# "manual"         : 指定値をそのまま使用
GYRO_BIAS_METHOD = "prewalk_robust"
GYRO_BIAS_PREWALK_MAX_SECONDS = 5.0
GYRO_BIAS_MIN_CALIBRATION_SECONDS = 0.5
GYRO_BIAS_WALK_ONSET_MIN_STEPS = 4
GYRO_BIAS_WALK_ONSET_MAX_INTERVAL_S = 1.2
GYRO_BIAS_CALIBRATION_MARGIN_S = 0.3
GYRO_BIAS_OUTLIER_MAD_SCALE = 3.0
GYRO_BIAS_STATIC_SEARCH_START_SECONDS = 0.5
GYRO_BIAS_STATIC_SEARCH_END_SECONDS = 4.5
GYRO_BIAS_STATIC_WALK_ONSET_MARGIN_S = 0.7
GYRO_BIAS_STATIC_WINDOW_SECONDS = 1.0
GYRO_BIAS_STATIC_WINDOW_STEP_SECONDS = 0.1
GYRO_BIAS_STATIC_MAX_ACCEL_P95 = 1.0
GYRO_BIAS_STATIC_MAX_GYRO_STD = 0.12
GYRO_BIAS_STATIC_ACCEL_P95_WEIGHT = 1.0
GYRO_BIAS_STATIC_GYRO_STD_WEIGHT = 1.0

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

# ステップ検出手法
# "peak": 既存の線形加速度ノルムピーク検出
# "paper_vertical_threshold": 論文方式に寄せた上下加速度閾値による1歩区間抽出
STEP_DETECTION_METHOD = "peak"

# 論文寄せステップ検出のパラメータ
# 上下加速度の接地インパルス候補を軽く平滑化して閾値判定する
STEP_VERTICAL_SMOOTH_WINDOW = 5
STEP_VERTICAL_THRESHOLD_PERCENTILE = 85.0
# ステップ区間の最小サンプル数（30サンプル = 0.3秒 @ 100Hz）
MIN_SEG_SAMPLES = 30

# ユーザー身長 [m]
# Weinbergモデルの歩幅スケールは身長におおむね比例すると仮定して補正する
# ryuki身長1.7m
# natsuki身長1.68m

USER_HEIGHT_M = 1.68

# Weinberg モデルの基準スケール係数（校正が必要なパラメータ）
# 歩幅 = K × (a_max - a_min)^0.25 の K に相当する
# WEINBERG_REFERENCE_HEIGHT_M のユーザーで校正した値として扱う
WEINBERG_REFERENCE_HEIGHT_M = 1.70
WEINBERG_REFERENCE_K = 0.47


def compute_weinberg_k(height_m: float = USER_HEIGHT_M) -> float:
    """身長に応じた Weinberg モデルのスケール係数を返す。"""
    if not isfinite(height_m) or height_m <= 0:
        raise ValueError("height_m は有限な正の値を指定してください。")
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

# 方位推定手法の選択
# "gyro"          : 既存のジャイロ積分方位
# "accel_method1" : 論文手法1（時間的に早い平面加速度極大値方向）
# "accel_method2" : 論文手法2（ノルムが大きい平面加速度極大値方向）
# "gyro_accel_motion": ジャイロを体の向き、水平加速度を移動方向として分離
HEADING_METHOD = "gyro_accel_motion"

# forward 判定ステップを軌跡へ積むときの方位ソース
# "body"  : ジャイロ由来の体/端末方向を使う
# "motion": 水平加速度から得た移動方向を使う
FORWARD_HEADING_SOURCE = "body"
SIDESTEP_SUSPECT_MODE = "forward"

# 加速度平面成分方位の信頼度パラメータ
ACCEL_HEADING_MIN_PEAK_NORM = 1.0
ACCEL_HEADING_MIN_PEAK_DISTANCE = 10
ACCEL_HEADING_MIN_LINE_LENGTH = 1.0
ACCEL_HEADING_CONFIDENCE_THRESHOLD = 0.6

# ジャイロ基準の水平加速度移動方向推定パラメータ
MOTION_HEADING_MIN_DISPLACEMENT_M = 1e-4
MOTION_HEADING_CONFIDENCE_THRESHOLD = 0.6
SIDESTEP_LATERAL_RATIO = 1.2
SIDESTEP_MIN_LATERAL_DISPLACEMENT_M = 0.03
SIDESTEP_SMOOTHING_METHOD = "clustered"
MOTION_HEADING_CALIBRATION_STEPS = 8
SIDESTEP_LENGTH_SCALE = 1
TURNING_LENGTH_SCALE = 0.3
BACKWARD_LENGTH_SCALE = 1
TURNING_YAW_DELTA_THRESHOLD_DEG = 35.0

# 前進方向射影積分のユニバーサルスケール係数
# 歩幅 = K_FORWARD × 振動変位
# 振動変位 ≈ 0.13 m（体装着センサーの直流速度成分が計測不能なため）
# K = v_mean / Δv_peak × 2/π ≈ 定数（v_mean と Δv が歩行速度に比例するため）
K_FORWARD = 9.0

# パーティクルフィルタ設定
PF_NUM_PARTICLES = 500
PF_SIGMA_INIT_HEADING = 0.03  # 初期方向ばらつき [rad]
PF_SIGMA_HEADING = 0.01  # ステップごとの方位角ドリフト [rad/step]
PF_SIGMA_STEP_LENGTH_RATIO = 0.01  # 永続倍率で説明できない歩ごとの歩幅ノイズ
PF_STRIDE_SCALE_PRIOR_MEAN = 1.03  # 正解軌跡長と決定論的歩幅合計から得た事前中心
PF_STRIDE_SCALE_INIT_SIGMA = 0.05  # 粒子ごとの初期歩幅倍率ばらつき
PF_STRIDE_SCALE_RETENTION = 0.995  # 学習した歩幅倍率偏差の1歩ごとの保持率
PF_STRIDE_SCALE_PROCESS_SIGMA = 0.002  # 歩幅倍率の1歩ごとの変動
PF_STRIDE_SCALE_REJUVENATION_SIGMA = 0.003  # 再標本化後の歩幅倍率多様化
PF_STRIDE_SCALE_MIN = 0.90  # 歩幅倍率の下限
PF_STRIDE_SCALE_MAX = 1.15  # 歩幅倍率の上限
PF_HEADING_DRIFT_RETENTION = 0.85  # 通常方位ドリフトの1歩ごとの保持率
PF_RESAMPLE_ESS_RATIO = 0.5  # ESSが粒子数に占める割合を下回ると再標本化
PF_REJUVENATION_SIGMA_HEADING = 0.02  # 再標本化後の方位多様化 [rad]
PF_RECOVERY_VALID_RATIO = 0.05  # 有効な重み付き粒子率が下回ると復旧
PF_RECOVERY_HEADING_SIGMA = 0.08  # local recoveryの方位分散 [rad]
PF_RECOVERY_MAX_ATTEMPTS = 5  # recovery候補を追加生成する最大回数
PF_MOTION_DISPLACEMENT_FULL_CONFIDENCE_M = 0.08
PF_MOTION_CALIBRATION_MIN_STEPS = 4
PF_MOTION_STATE_TRANSITION_STAY = 0.82
