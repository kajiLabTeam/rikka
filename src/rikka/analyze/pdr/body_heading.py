"""端末方位から時変の身体方位を推定する純粋な時系列処理。

役割:
    端末 yaw と方向未確定の移動軸から、端末と身体の方位オフセットを歩ごとに
    ロバスト推定し、身体方位候補と診断情報を返す。
依存元:
    ``models`` から ``StepMotionObservation`` を取得し、NumPy を角度演算と
    ロバストな窓集約に利用する。
利用先:
    将来 ``trajectory`` が移動状態の区間推定後に呼び出し、通常 PDR と
    particle filter で共有する身体方位候補を作るために使用する。
処理フロー:
    同一移動モードが3歩続いた区間だけを候補とし、modulo pi の移動軸を予測身体
    方位に近い枝へ持ち上げ、Huber 重み付き残差を1歩上限付きでオフセットへ反映する。
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from .models import StepMotionObservation

_STABLE_WINDOW_STEPS = 3
_MIN_MOTION_CONFIDENCE = 0.5
_MIN_DISPLACEMENT_M = 0.02
_FULL_CONFIDENCE_DISPLACEMENT_M = 0.08
_MIN_AXIS_CONCENTRATION = 0.7
_HUBER_DELTA_RAD = float(np.deg2rad(20.0))
_UPDATE_GAIN = 0.25
_MAX_UPDATE_RAD = float(np.deg2rad(5.0))
_SUPPORTED_MOTION_MODES = frozenset({"forward", "sidestep_left", "sidestep_right"})


class DynamicBodyHeadingEstimate(NamedTuple):
    """1歩に対応する動的身体方位推定と更新診断。"""

    body_heading: float | None
    device_body_offset: float
    confidence: float
    updated: bool
    reason: str


def _normalize_angle(angle: float) -> float:
    """角度を [-pi, pi) に正規化する。"""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _axis_residual(observed_axis: float, predicted_heading: float) -> float:
    """modulo pi の観測軸を予測方位に近い枝へ持ち上げた残差を返す。"""
    return float(
        (observed_axis - predicted_heading + np.pi / 2.0) % np.pi - np.pi / 2.0
    )


def _body_axis_for_mode(motion_axis: float, motion_mode: str) -> float:
    """移動モードに対応する modulo pi の身体前後軸を返す。"""
    if motion_mode == "forward":
        return motion_axis
    # 左右の横歩きは符号が異なるが、身体軸は modulo pi では同じになる。
    return float(motion_axis - np.pi / 2.0)


def _observation_quality(observation: StepMotionObservation) -> float:
    """オフセット更新に使う単歩観測の品質を [0, 1] で返す。"""
    displacement_quality = float(
        np.clip(
            observation.displacement_norm / _FULL_CONFIDENCE_DISPLACEMENT_M,
            0.0,
            1.0,
        )
    )
    # 初期区間の校正値だけで動的更新を完全に止めないよう、下限を持たせる。
    calibration_weight = 0.5 + 0.5 * float(
        np.clip(observation.calibration_reliability, 0.0, 1.0)
    )
    return float(
        np.clip(
            observation.motion_confidence * displacement_quality * calibration_weight,
            0.0,
            1.0,
        )
    )


def _invalid_observation_reason(observation: StepMotionObservation) -> str | None:
    """単歩観測を更新に使えない理由を返す。"""
    if observation.device_yaw_heading is None:
        return "no_device_yaw"
    if observation.motion_axis_heading is None:
        return "no_motion_axis"
    if observation.motion_confidence < _MIN_MOTION_CONFIDENCE:
        return "low_motion_confidence"
    if observation.displacement_norm < _MIN_DISPLACEMENT_M:
        return "low_displacement"
    return None


def _axis_concentration(
    observations: Sequence[StepMotionObservation],
    motion_mode: str,
) -> float:
    """窓内の modulo pi 身体軸が集中する度合いを返す。"""
    body_axes = np.asarray(
        [
            _body_axis_for_mode(float(observation.motion_axis_heading), motion_mode)
            for observation in observations
            if observation.motion_axis_heading is not None
        ],
        dtype=float,
    )
    if len(body_axes) != len(observations) or len(body_axes) == 0:
        return 0.0
    return float(
        np.hypot(
            np.mean(np.sin(2.0 * body_axes)),
            np.mean(np.cos(2.0 * body_axes)),
        )
    )


def _huber_weighted_residual(
    residuals: np.ndarray,
    weights: np.ndarray,
) -> float | None:
    """Huber 重みを加えた残差平均を返す。"""
    absolute = np.abs(residuals)
    huber_weights = np.ones_like(absolute)
    nonzero = absolute > _HUBER_DELTA_RAD
    huber_weights[nonzero] = _HUBER_DELTA_RAD / absolute[nonzero]
    combined = weights * huber_weights
    total = float(np.sum(combined))
    if total <= 1e-12:
        return None
    return float(np.sum(combined * residuals) / total)


def _body_heading_for_observation(
    observation: StepMotionObservation,
    device_body_offset: float,
) -> float | None:
    """端末 yaw とオフセットから身体方位を返し、欠損時は既存候補へ戻す。"""
    if observation.device_yaw_heading is None:
        return observation.body_heading_candidate
    return _normalize_angle(observation.device_yaw_heading + device_body_offset)


def estimate_dynamic_body_headings(
    observations: Sequence[StepMotionObservation],
    motion_modes: Sequence[str],
) -> tuple[DynamicBodyHeadingEstimate, ...]:
    """移動状態が安定した因果窓から端末―身体オフセットを推定する。

    世界座標の移動方位は変更せず、端末 yaw から求める身体方位候補だけを返す。
    未対応モード、状態境界、低信頼観測では直前のオフセットを保持する。
    """
    if len(observations) != len(motion_modes):
        raise ValueError("observations and motion_modes must have the same length")

    estimates: list[DynamicBodyHeadingEstimate] = []
    device_body_offset = 0.0
    estimate_confidence = 0.0

    for index, (observation, motion_mode) in enumerate(
        zip(observations, motion_modes, strict=True)
    ):
        updated = False
        if motion_mode not in _SUPPORTED_MOTION_MODES:
            reason = "turning_or_unsupported_mode"
        else:
            invalid_reason = _invalid_observation_reason(observation)
            if invalid_reason is not None:
                reason = invalid_reason
            elif index + 1 < _STABLE_WINDOW_STEPS:
                reason = "mode_boundary"
            else:
                start = index + 1 - _STABLE_WINDOW_STEPS
                window_modes = motion_modes[start : index + 1]
                window = observations[start : index + 1]
                if any(mode != motion_mode for mode in window_modes):
                    reason = "mode_boundary"
                else:
                    window_invalid = [
                        _invalid_observation_reason(item) for item in window
                    ]
                    if any(item is not None for item in window_invalid):
                        reason = "unstable_window"
                    else:
                        concentration = _axis_concentration(window, motion_mode)
                        if concentration < _MIN_AXIS_CONCENTRATION:
                            reason = "low_axis_concentration"
                        else:
                            residuals: list[float] = []
                            weights: list[float] = []
                            for item in window:
                                assert item.device_yaw_heading is not None
                                assert item.motion_axis_heading is not None
                                predicted = _normalize_angle(
                                    item.device_yaw_heading + device_body_offset
                                )
                                observed_body_axis = _body_axis_for_mode(
                                    item.motion_axis_heading,
                                    motion_mode,
                                )
                                residuals.append(
                                    _axis_residual(observed_body_axis, predicted)
                                )
                                weights.append(_observation_quality(item))
                            robust_residual = _huber_weighted_residual(
                                np.asarray(residuals, dtype=float),
                                np.asarray(weights, dtype=float),
                            )
                            if robust_residual is None:
                                reason = "low_window_weight"
                            else:
                                update = float(
                                    np.clip(
                                        _UPDATE_GAIN * robust_residual,
                                        -_MAX_UPDATE_RAD,
                                        _MAX_UPDATE_RAD,
                                    )
                                )
                                device_body_offset = _normalize_angle(
                                    device_body_offset + update
                                )
                                window_quality = float(np.mean(weights))
                                estimate_confidence = float(
                                    np.clip(
                                        (1.0 - _UPDATE_GAIN) * estimate_confidence
                                        + _UPDATE_GAIN * window_quality * concentration,
                                        0.0,
                                        1.0,
                                    )
                                )
                                updated = True
                                reason = "updated"

        estimates.append(
            DynamicBodyHeadingEstimate(
                body_heading=_body_heading_for_observation(
                    observation,
                    device_body_offset,
                ),
                device_body_offset=device_body_offset,
                confidence=estimate_confidence,
                updated=updated,
                reason=reason,
            )
        )

    return tuple(estimates)


__all__ = ["DynamicBodyHeadingEstimate", "estimate_dynamic_body_headings"]
