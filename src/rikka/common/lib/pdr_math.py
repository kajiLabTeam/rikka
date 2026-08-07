"""PDR/PF で共有する幾何定数と数値検証ヘルパー。

役割:
    方位閾値、角度の正規化、数値パラメータの検証を一元管理する。
依存元:
    NumPy の有限値・角度計算だけを利用し、他の PDR モジュールには依存しない。
利用先:
    PDR とparticle filterが共通ルールとして使用する。
処理フロー:
    入力値を正規化または数値検証して返し、不正値は ``ValueError`` にする。
    アルゴリズム本体から独立させることでモジュール間の循環 import を避ける。
"""

import numpy as np

from .angles import abs_angle_diff, normalize_angle, score_ratio
from .validation import (
    validate_non_negative_parameter,
    validate_positive_parameter,
    validate_scale,
)

_normalize_angle = normalize_angle
_abs_angle_diff = abs_angle_diff
_score_ratio = score_ratio

__all__ = [
    "INITIAL_FORWARD_MOTION_BODY_CONSTRAINT_RAD",
    "SIDESTEP_BODY_MOTION_ANGLE_THRESHOLD_RAD",
    "SIDESTEP_BODY_MOTION_RATIO_THRESHOLD",
    "SIDESTEP_HEADING_MAX_STEP_DELTA_RAD",
    "SIDESTEP_MOTION_LATERAL_CONSTRAINT_RAD",
    "SIDESTEP_STRONG_ANGLE_THRESHOLD_RAD",
    "TRAJECTORY_HEADING_MAX_STEP_DELTA_RAD",
    "TURNING_SIDESTEP_HEADING_MAX_STEP_DELTA_RAD",
    "_abs_angle_diff",
    "_normalize_angle",
    "_score_ratio",
    "_validate_non_negative_parameter",
    "_validate_positive_parameter",
    "_validate_scale",
]

# 横歩き判定と軌跡安定化のための幾何的な閾値。
SIDESTEP_BODY_MOTION_ANGLE_THRESHOLD_RAD = np.deg2rad(45.0)
SIDESTEP_BODY_MOTION_RATIO_THRESHOLD = 0.8
SIDESTEP_STRONG_ANGLE_THRESHOLD_RAD = np.deg2rad(75.0)
SIDESTEP_MOTION_LATERAL_CONSTRAINT_RAD = np.deg2rad(45.0)
TRAJECTORY_HEADING_MAX_STEP_DELTA_RAD = np.deg2rad(25.0)
SIDESTEP_HEADING_MAX_STEP_DELTA_RAD = np.deg2rad(45.0)
TURNING_SIDESTEP_HEADING_MAX_STEP_DELTA_RAD = np.deg2rad(90.0)
INITIAL_FORWARD_MOTION_BODY_CONSTRAINT_RAD = np.deg2rad(45.0)


def _validate_scale(scale: float) -> None:
    """フロアマップ縮尺が正の値であることを確認する。"""
    validate_scale(scale)


def _validate_positive_parameter(name: str, value: float) -> float:
    """正の解析パラメータであることを確認する。"""
    return validate_positive_parameter(name, value)


def _validate_non_negative_parameter(name: str, value: float) -> float:
    """0以上の解析パラメータであることを確認する。"""
    return validate_non_negative_parameter(name, value)
