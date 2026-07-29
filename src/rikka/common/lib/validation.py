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

