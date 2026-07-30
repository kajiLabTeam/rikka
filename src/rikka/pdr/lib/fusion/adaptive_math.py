"""適応PDRの数値計算部品。

役割:
    円統計、歩幅統計、初期状態、観測更新を実装する。
依存元:
    common の共有型と NumPy を利用する。
利用先:
    fusion.adaptive の estimator から使用される。
処理フロー:
    観測候補を統計量へ変換し、状態平均と分散を更新する。
"""

import numpy as np

from ....common.lib.models import (
    AdaptivePdrState,
    StepHeading,
)

_MODE_NAMES = ("forward", "sidestep_left", "sidestep_right", "turning")
_TRANSITION = np.asarray(
    [
        [0.91, 0.035, 0.035, 0.02],
        [0.08, 0.86, 0.01, 0.05],
        [0.08, 0.01, 0.86, 0.05],
        [0.20, 0.04, 0.04, 0.72],
    ],
    dtype=float,
)


def _normalize_angle(angle: float) -> float:
    """角度を ``[-pi, pi)`` に正規化する。"""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    """重み付き循環平均を返す。"""
    return float(
        np.arctan2(
            np.sum(weights * np.sin(angles)),
            np.sum(weights * np.cos(angles)),
        )
    )


def _circular_std(angles: np.ndarray, weights: np.ndarray) -> float:
    """重み付き角度分布の標準偏差近似を返す。"""
    resultant = float(
        np.hypot(
            np.sum(weights * np.cos(angles)),
            np.sum(weights * np.sin(angles)),
        )
    )
    return float(np.sqrt(max(-2.0 * np.log(max(resultant, 1e-6)), 0.0)))


def _length_moments(
    probabilities: np.ndarray,
    length_means: np.ndarray,
    log_variances: np.ndarray,
) -> tuple[float, float]:
    """状態確率から歩幅の平均と標準偏差を計算する。"""
    length_mean = float(np.sum(probabilities * length_means))
    within_variance = np.square(length_means) * np.maximum(
        np.exp(log_variances) - 1.0,
        0.0,
    )
    length_variance = float(
        np.sum(
            probabilities * (within_variance + np.square(length_means - length_mean))
        )
    )
    return length_mean, float(np.sqrt(max(length_variance, 0.0)))


def _initial_state() -> AdaptivePdrState:
    """弱い事前分布を持つ初期状態を作る。"""
    return AdaptivePdrState(
        heading_mean=None,
        heading_variance=float(np.deg2rad(35.0) ** 2),
        forward_log_scale_mean=0.0,
        forward_log_scale_variance=0.12**2,
        sidestep_log_scale_mean=0.0,
        sidestep_log_scale_variance=0.18**2,
        device_body_offset_mean=0.0,
        device_body_offset_variance=float(np.deg2rad(25.0) ** 2),
        mode_probabilities=(0.88, 0.04, 0.04, 0.04),
        step_count=0,
    )


def _body_heading(step_heading: StepHeading) -> float | None:
    """端末と身体のずれを補正済みの身体方位候補を返す。"""
    if step_heading.body_heading is not None:
        return step_heading.body_heading
    return step_heading.gyro_heading


def _mode_heading_candidates(step_heading: StepHeading) -> np.ndarray:
    """4運動状態それぞれの世界座標移動方位候補を作る。"""
    body = _body_heading(step_heading)
    selected = step_heading.selected_heading
    motion = step_heading.motion_heading
    fallback = next(
        (value for value in (selected, motion, body) if value is not None),
        0.0,
    )
    body_value = fallback if body is None else body
    reliability = float(np.clip(step_heading.motion_confidence, 0.0, 1.0))

    def side_candidate(sign: float) -> float:
        lateral = _normalize_angle(body_value + sign * np.pi / 2.0)
        if motion is None:
            return lateral
        difference = abs(_normalize_angle(motion - lateral))
        if difference > np.deg2rad(70.0):
            return lateral
        weight = 0.65 * reliability
        return _circular_mean(
            np.asarray([lateral, motion]),
            np.asarray([1.0 - weight, weight]),
        )

    turning = next(
        (value for value in (motion, selected, body) if value is not None), 0.0
    )
    return np.asarray(
        [body_value, side_candidate(1.0), side_candidate(-1.0), turning],
        dtype=float,
    )


def _kalman_update(
    mean: float,
    variance: float,
    measurement: float,
    measurement_variance: float,
    responsibility: float,
) -> tuple[float, float]:
    """状態責任度を含む1次元Kalman更新を行う。"""
    process_variance = 0.018**2
    predicted_variance = variance + process_variance
    effective_variance = measurement_variance / max(responsibility, 0.05)
    gain = predicted_variance / (predicted_variance + effective_variance)
    updated_mean = mean + gain * (measurement - mean)
    updated_variance = max((1.0 - gain) * predicted_variance, 1e-6)
    return float(updated_mean), float(updated_variance)
