"""Configuration file for rikka"""

# Data directory path
# Change this path to use different input data
DATA_DIR = "input/90steps_turn_elevator"

# Weinberg モデルのスケール係数（校正が必要なパラメータ）
WEINBERG_K = 0.4

# 歩幅推定用ウィンドウ（ステップピーク前後のサンプル数）
STEP_LENGTH_WINDOW = 50  # ±50サンプル = ±0.5秒 @ 100Hz
