"""PDR 共通定数と検証ヘルパー。

役割:
    モード名、方位閾値、角度の正規化、数値・選択値の検証を一元管理する。
依存元:
    NumPy の有限値・角度計算だけを利用し、他の PDR モジュールには依存しない。
利用先:
    ``heading``、``sidestep``、``trajectory``、``pipeline``、``particle_api`` と
    互換 facade が共通ルールとして使用する。
処理フロー:
    入力値を正規化または検証して返し、不正値は ``ValueError`` にする。
    アルゴリズム本体から独立させることでモジュール間の循環 import を避ける。
"""

import numpy as np

HEADING_METHODS = (
    "gyro",
    "accel_method1",
    "accel_method2",
    "gyro_accel_motion",
)

# CLI や config から渡される文字列モードは、ここで許可値を一元管理する。
MOTION_HEADING_CORRECTION_METHODS = ("auto", "none")
SIDESTEP_SMOOTHING_METHODS = ("none", "isolated", "clustered")
FORWARD_HEADING_SOURCES = ("body", "motion")
SIDESTEP_HEADING_SOURCES = ("motion", "body_lateral", "blend")
SIDESTEP_SUSPECT_MODES = ("motion", "body_lateral", "blend", "forward")

# 横歩き判定と軌跡安定化のための幾何的な閾値。
SIDESTEP_BODY_MOTION_ANGLE_THRESHOLD_RAD = np.deg2rad(45.0)
SIDESTEP_BODY_MOTION_RATIO_THRESHOLD = 0.8
SIDESTEP_STRONG_ANGLE_THRESHOLD_RAD = np.deg2rad(75.0)
SIDESTEP_MOTION_LATERAL_CONSTRAINT_RAD = np.deg2rad(45.0)
TRAJECTORY_HEADING_MAX_STEP_DELTA_RAD = np.deg2rad(25.0)
SIDESTEP_HEADING_MAX_STEP_DELTA_RAD = np.deg2rad(45.0)
TURNING_SIDESTEP_HEADING_MAX_STEP_DELTA_RAD = np.deg2rad(90.0)
INITIAL_FORWARD_MOTION_BODY_CONSTRAINT_RAD = np.deg2rad(45.0)
DEVICE_ORIENTATION_MODES = (
    "normal",
    "front_back_inverted",
    "left_right_inverted",
    "rotated_180",
)


def _validate_scale(scale: float) -> None:
    """フロアマップ縮尺が正の値であることを確認する。"""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale は有限な正の値を指定してください。")


def _validate_heading_method(method: str) -> str:
    """方位推定手法名を検証する。"""
    if method not in HEADING_METHODS:
        allowed = ", ".join(HEADING_METHODS)
        raise ValueError(f"heading_method は {allowed} のいずれかを指定してください。")
    return method


def _normalize_angle(angle: float) -> float:
    """角度を [-pi, pi) に正規化する。"""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _abs_angle_diff(angle_a: float | None, angle_b: float | None) -> float | None:
    """2つの角度差の絶対値を返す。どちらかが None なら None。"""
    if angle_a is None or angle_b is None:
        return None
    return abs(_normalize_angle(angle_a - angle_b))


def _score_ratio(value: float, target: float) -> float:
    """target 以上を 1.0 とする 0..1 スコアを返す。"""
    if target <= 0:
        return 1.0
    return float(np.clip(value / target, 0.0, 1.0))


def _validate_positive_parameter(name: str, value: float) -> float:
    """正の解析パラメータであることを確認する。"""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} は有限な正の値を指定してください。")
    return value


def _validate_non_negative_parameter(name: str, value: float) -> float:
    """0以上の解析パラメータであることを確認する。"""
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} は有限な0以上の値を指定してください。")
    return value


def _validate_motion_heading_correction(method: str) -> str:
    """水平加速度移動方向の固定ずれ補正モードを検証する。"""
    if method not in MOTION_HEADING_CORRECTION_METHODS:
        allowed = ", ".join(MOTION_HEADING_CORRECTION_METHODS)
        raise ValueError(
            f"motion_heading_correction は {allowed} のいずれかを指定してください。"
        )
    return method


def _validate_sidestep_smoothing(method: str) -> str:
    """横歩き判定の平滑化モードを検証する。"""
    if method not in SIDESTEP_SMOOTHING_METHODS:
        allowed = ", ".join(SIDESTEP_SMOOTHING_METHODS)
        raise ValueError(
            f"sidestep_smoothing は {allowed} のいずれかを指定してください。"
        )
    return method


def _validate_forward_heading_source(source: str) -> str:
    """forward 判定ステップに使う方位ソースを検証する。"""
    if source not in FORWARD_HEADING_SOURCES:
        allowed = ", ".join(FORWARD_HEADING_SOURCES)
        raise ValueError(
            f"forward_heading_source は {allowed} のいずれかを指定してください。"
        )
    return source


def _validate_sidestep_heading_source(source: str) -> str:
    """横歩き確定ステップに使う方位ソースを検証する。"""
    if source not in SIDESTEP_HEADING_SOURCES:
        allowed = ", ".join(SIDESTEP_HEADING_SOURCES)
        raise ValueError(
            f"sidestep_heading_source は {allowed} のいずれかを指定してください。"
        )
    return source


def _validate_sidestep_suspect_mode(mode: str) -> str:
    """横歩き疑いステップの軌跡反映モードを検証する。"""
    if mode not in SIDESTEP_SUSPECT_MODES:
        allowed = ", ".join(SIDESTEP_SUSPECT_MODES)
        raise ValueError(
            f"sidestep_suspect_mode は {allowed} のいずれかを指定してください。"
        )
    return mode
