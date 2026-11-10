"""角度の正規化・差分・比率スコアを提供する。

役割:
    領域に依存しない角度演算と単純なスコア化を一元管理する。
依存元:
    NumPy の角度・有限値演算だけを使用する。
利用先:
    PDR と particle filter の方位計算から使用する。
処理フロー:
    入力角度を正規化し、角度差または0から1の比率を返す。
"""

import numpy as np


def normalize_angle(angle: float) -> float:
    """角度を [-pi, pi) に正規化する。"""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def abs_angle_diff(
    angle_a: float | None,
    angle_b: float | None,
) -> float | None:
    """2つの角度差の絶対値を返す。"""
    if angle_a is None or angle_b is None:
        return None
    return abs(normalize_angle(angle_a - angle_b))


def score_ratio(value: float, target: float) -> float:
    """target 以上を1.0とする0..1スコアを返す。"""
    if target <= 0:
        return 1.0
    return float(np.clip(value / target, 0.0, 1.0))


def circular_mean_angles(angles: list[float]) -> float | None:
    """有限な角度の円平均を返す。"""
    finite_angles = [angle for angle in angles if np.isfinite(angle)]
    if not finite_angles:
        return None
    sin_sum = float(np.sum(np.sin(finite_angles)))
    cos_sum = float(np.sum(np.cos(finite_angles)))
    if np.hypot(sin_sum, cos_sum) <= 1e-12:
        return None
    return normalize_angle(float(np.arctan2(sin_sum, cos_sum)))


def normalize_angle_array(angle: np.ndarray) -> np.ndarray:
    """角度配列を -pi 以上 pi 未満へ正規化する。"""
    return (angle + np.pi) % (2 * np.pi) - np.pi
