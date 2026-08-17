"""設定オブジェクトが利用する値域・選択値検証。

役割:
    解析設定の許可値と数値範囲を一元的に検証する。
依存元:
    NumPy の有限値判定だけを使用する。
利用先:
    ``common.settings`` が構築時に一度だけ呼び出す。
処理フロー:
    値を検証して正規化せず返し、不正値には ``ValueError`` を送出する。
"""

import numpy as np

HEADING_METHODS = (
    "gyro",
    "accel_method1",
    "accel_method2",
    "gyro_accel_motion",
)
MOTION_HEADING_CORRECTION_METHODS = ("auto", "none")
SIDESTEP_SMOOTHING_METHODS = ("none", "isolated", "clustered")
FORWARD_HEADING_SOURCES = ("body", "motion")
SIDESTEP_HEADING_SOURCES = ("motion", "body_lateral", "blend")
SIDESTEP_SUSPECT_MODES = ("motion", "body_lateral", "blend", "forward")
DEVICE_ORIENTATION_MODES = (
    "normal",
    "front_back_inverted",
    "left_right_inverted",
    "rotated_180",
)
MOTION_ESTIMATION_METHODS = ("legacy", "adaptive", "robust")
SMOOTHING_MODES = ("causal", "offline")
PF_PATH_SELECTION_METHODS = ("current", "sequence")
PF_LANDMARK_MODES = ("none", "observation", "reset")
STEP_LENGTH_METHODS = ("weinberg", "forward")
STEP_DETECTION_METHODS = ("peak", "paper_vertical_threshold")
GYRO_BIAS_METHODS = (
    "prewalk_guarded",
    "zero",
    "prewalk_robust",
    "initial_robust",
    "quietest",
    "manual",
)


def validate_scale(scale: float) -> float:
    """フロアマップ縮尺が正の値であることを確認する。"""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale は有限な正の値を指定してください。")
    return scale


def validate_choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    """文字列が許可値のいずれかであることを確認する。"""
    if value not in choices:
        allowed = ", ".join(choices)
        raise ValueError(f"{name} は {allowed} のいずれかを指定してください。")
    return value


def validate_positive_parameter(name: str, value: float) -> float:
    """正の有限値であることを確認する。"""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} は有限な正の値を指定してください。")
    return value


def validate_non_negative_parameter(name: str, value: float) -> float:
    """0以上の有限値であることを確認する。"""
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} は有限な0以上の値を指定してください。")
    return value


def validate_motion_heading_correction(value: str) -> str:
    """移動方位補正方式を検証する。"""
    return validate_choice(
        "motion_heading_correction",
        value,
        MOTION_HEADING_CORRECTION_METHODS,
    )


def validate_sidestep_smoothing(value: str) -> str:
    """横歩き平滑化方式を検証する。"""
    return validate_choice("sidestep_smoothing", value, SIDESTEP_SMOOTHING_METHODS)


def validate_forward_heading_source(value: str) -> str:
    """前進方位ソースを検証する。"""
    return validate_choice("forward_heading_source", value, FORWARD_HEADING_SOURCES)


def validate_sidestep_heading_source(value: str) -> str:
    """横歩き方位ソースを検証する。"""
    return validate_choice(
        "sidestep_heading_source",
        value,
        SIDESTEP_HEADING_SOURCES,
    )


def validate_sidestep_suspect_mode(value: str) -> str:
    """横歩き疑いの処理方式を検証する。"""
    return validate_choice("sidestep_suspect_mode", value, SIDESTEP_SUSPECT_MODES)


def validate_landmarks(
    landmarks: tuple[tuple[str, float, float], ...],
) -> tuple[tuple[str, float, float], ...]:
    """ランドマーク定義の beacon_id 重複とピクセル座標を確認する。"""
    seen: set[str] = set()
    for beacon_id, pixel_x, pixel_y in landmarks:
        if not beacon_id or not beacon_id.strip():
            raise ValueError("beacon_id は空にできません。")
        if beacon_id in seen:
            raise ValueError(f"beacon_id が重複しています: {beacon_id}")
        seen.add(beacon_id)
        if not np.isfinite(pixel_x) or not np.isfinite(pixel_y):
            raise ValueError(
                f"ランドマークのピクセル座標は有限な値を指定してください: {beacon_id}"
            )
    return landmarks
