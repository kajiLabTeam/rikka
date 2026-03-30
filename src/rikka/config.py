"""Configuration file for rikka"""

# Data directory path
# Change this path to use different input data

DATA_DIR = "input/90steps_turn_elevator"
# DATA_DIR = "input/90steps_turn_Yamamoto"

# Weinberg モデルのスケール係数（校正が必要なパラメータ）
# 歩幅 = K × (a_max - a_min)^0.25 の K に相当する
# ユーザーの体格・歩行スタイル・端末装着位置で個人差があるため、
# 実測の数ステップで検証してキャリブレーションすることを推奨
WEINBERG_K = 0.4

# 歩幅推定用ウィンドウ（ステップピーク前後のサンプル数）
# ±50サンプル = ±0.5秒 @ 100Hz
# スイング期（足の振り上げ〜接地）の上下動を十分に捕捉できる幅に設定
STEP_LENGTH_WINDOW = 50
