"""運動状態・方位・歩幅を同時に逐次推定する適応PDR。

役割:
    1歩ごとの運動状態を確率分布として保持し、端末方位の変化を移動方位へ直接
    混入させず、歩幅の局所変化と不確かさを逐次推定する。
依存元:
    ``models`` から方位・歩幅観測・事後分布型、``config`` から横歩きの既定倍率を
    取得し、NumPy で角度分布と小規模なベイズ更新を計算する。
利用先:
    ``trajectory.prepare_pdr_steps`` が adaptive モードの通常PDRを生成し、同じ
    事後分布を ``particle_filter`` が地図制約付き推定に利用する。
処理フロー:
    運動状態の遷移予測、センサー尤度と方位連続性による更新、状態別歩幅倍率の
    更新、方位・歩幅の混合分布生成を各歩で行い、offline 指定時は状態列を後向きに
    平滑化する。
"""

from __future__ import annotations

import numpy as np

from ....common.lib.models import (
    AdaptivePdrResult,
    AdaptivePdrState,
    StepHeading,
    StepLengthObservation,
    StepMotionEvidence,
    StepMotionPosterior,
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


class AdaptivePdrEstimator:
    """ステップ到着ごとに更新可能な適応PDR推定器。"""

    def __init__(self, state: AdaptivePdrState | None = None) -> None:
        self.state = _initial_state() if state is None else state

    def update_step(
        self,
        step_heading: StepHeading,
        length_observation: StepLengthObservation,
        motion_evidence: StepMotionEvidence,
    ) -> StepMotionPosterior:
        """1歩分の観測だけを使い、因果的に状態を更新する。"""
        previous = self.state
        prior = np.asarray(previous.mode_probabilities) @ _TRANSITION
        sensor_likelihood = np.asarray(
            [
                motion_evidence.forward_likelihood,
                motion_evidence.sidestep_left_likelihood,
                motion_evidence.sidestep_right_likelihood,
                motion_evidence.turning_likelihood,
            ],
            dtype=float,
        )
        candidates = _mode_heading_candidates(step_heading)
        if previous.heading_mean is None:
            continuity = np.ones(4, dtype=float)
        else:
            expected_yaw = float(step_heading.yaw_delta or 0.0)
            residuals = np.asarray(
                [
                    _normalize_angle(value - previous.heading_mean - expected_yaw)
                    for value in candidates
                ]
            )
            sigmas = np.deg2rad(np.asarray([24.0, 34.0, 34.0, 70.0]))
            continuity = np.exp(-0.5 * np.square(residuals / sigmas)) + 1e-4
        probabilities = prior * np.maximum(sensor_likelihood, 1e-4) * continuity
        probabilities /= probabilities.sum()

        representative_heading = next(
            (
                value
                for value in (
                    step_heading.selected_heading,
                    step_heading.motion_heading,
                    _body_heading(step_heading),
                )
                if value is not None
            ),
            float(candidates[int(np.argmax(probabilities))]),
        )
        heading_mean = float(representative_heading)
        if previous.heading_mean is not None:
            heading_jump = _normalize_angle(heading_mean - previous.heading_mean)
            yaw_delta = abs(float(step_heading.yaw_delta or 0.0))
            movement_type = (
                step_heading.trajectory_movement_type or step_heading.movement_type
            )
            previous_side_probability = (
                previous.mode_probabilities[1] + previous.mode_probabilities[2]
            )
            if (
                movement_type in {"sidestep_left", "sidestep_right"}
                and previous_side_probability < 0.5
                and abs(heading_jump) > np.deg2rad(45.0)
                and yaw_delta < np.deg2rad(20.0)
                and probabilities[3] < 0.25
            ):
                heading_mean = previous.heading_mean
        heading_std = min(_circular_std(candidates, probabilities), np.pi)

        nominal = max(length_observation.nominal_length_m, 1e-4)
        interval = max(length_observation.interval_length_m, 1e-4)
        ratio_measurement = float(np.log(np.clip(interval / nominal, 0.60, 1.45)))
        measurement_variance = float(length_observation.log_length_sigma**2)
        side_probability = float(probabilities[1] + probabilities[2])
        forward_probability = float(probabilities[0] + probabilities[3])
        forward_mean, forward_variance = _kalman_update(
            previous.forward_log_scale_mean,
            previous.forward_log_scale_variance,
            ratio_measurement,
            measurement_variance,
            forward_probability,
        )
        side_measurement = ratio_measurement
        side_mean, side_variance = _kalman_update(
            previous.sidestep_log_scale_mean,
            previous.sidestep_log_scale_variance,
            side_measurement,
            measurement_variance,
            side_probability,
        )
        mode_scales = np.exp(
            np.asarray([forward_mean, side_mean, side_mean, forward_mean])
        )
        length_means = nominal * mode_scales
        log_variances = (
            np.asarray(
                [forward_variance, side_variance, side_variance, forward_variance]
            )
            + measurement_variance
        )
        length_mean, length_std = _length_moments(
            probabilities,
            length_means,
            log_variances,
        )

        offset_measurement = float(step_heading.device_body_offset)
        offset_variance = float(
            np.deg2rad(
                6.0 + 24.0 * (1.0 - step_heading.dynamic_body_heading_confidence)
            )
            ** 2
        )
        offset_mean, offset_state_variance = _kalman_update(
            previous.device_body_offset_mean,
            previous.device_body_offset_variance,
            offset_measurement,
            offset_variance,
            max(step_heading.dynamic_body_heading_confidence, 0.05),
        )

        selected_mode = _MODE_NAMES[int(np.argmax(probabilities))]
        self.state = AdaptivePdrState(
            heading_mean=heading_mean,
            heading_variance=heading_std**2,
            forward_log_scale_mean=forward_mean,
            forward_log_scale_variance=forward_variance,
            sidestep_log_scale_mean=side_mean,
            sidestep_log_scale_variance=side_variance,
            device_body_offset_mean=offset_mean,
            device_body_offset_variance=offset_state_variance,
            mode_probabilities=(
                float(probabilities[0]),
                float(probabilities[1]),
                float(probabilities[2]),
                float(probabilities[3]),
            ),
            step_count=previous.step_count + 1,
        )
        return StepMotionPosterior(
            step_index=step_heading.step_index,
            forward_probability=float(probabilities[0]),
            sidestep_left_probability=float(probabilities[1]),
            sidestep_right_probability=float(probabilities[2]),
            turning_probability=float(probabilities[3]),
            heading_mean=heading_mean,
            heading_std=heading_std,
            length_mean_m=length_mean,
            length_std_m=length_std,
            device_body_offset_mean=offset_mean,
            device_body_offset_std=float(np.sqrt(offset_state_variance)),
            selected_mode=selected_mode,
            source="adaptive_causal",
        )


def _smooth_mode_probabilities(
    posteriors: list[StepMotionPosterior],
) -> list[np.ndarray]:
    """因果状態確率を後向き遷移で平滑化する。"""
    if not posteriors:
        return []
    smoothed = [
        np.asarray(
            [
                item.forward_probability,
                item.sidestep_left_probability,
                item.sidestep_right_probability,
                item.turning_probability,
            ],
            dtype=float,
        )
        for item in posteriors
    ]
    for index in range(len(smoothed) - 2, -1, -1):
        backward = _TRANSITION @ smoothed[index + 1]
        values = smoothed[index] * np.maximum(backward, 1e-6)
        smoothed[index] = values / values.sum()
    return smoothed


def estimate_adaptive_pdr(
    step_headings: list[StepHeading],
    length_observations: tuple[StepLengthObservation, ...],
    motion_evidences: tuple[StepMotionEvidence, ...],
    smoothing_mode: str = "causal",
) -> AdaptivePdrResult:
    """ステップ列を適応推定し、軌跡生成に使える方位と歩幅を返す。"""
    if smoothing_mode not in {"causal", "offline"}:
        raise ValueError("smoothing_mode は causal または offline を指定してください")
    if not (len(step_headings) == len(length_observations) == len(motion_evidences)):
        raise ValueError("適応PDRへ渡すステップ列の長さが一致しません")

    estimator = AdaptivePdrEstimator()
    posteriors: list[StepMotionPosterior] = []
    mode_length_means: list[np.ndarray] = []
    mode_log_variances: list[np.ndarray] = []
    for heading, length, evidence in zip(
        step_headings,
        length_observations,
        motion_evidences,
        strict=True,
    ):
        posterior = estimator.update_step(heading, length, evidence)
        posteriors.append(posterior)
        nominal = max(length.nominal_length_m, 1e-4)
        state = estimator.state
        mode_length_means.append(
            nominal
            * np.exp(
                np.asarray(
                    [
                        state.forward_log_scale_mean,
                        state.sidestep_log_scale_mean,
                        state.sidestep_log_scale_mean,
                        state.forward_log_scale_mean,
                    ]
                )
            )
        )
        mode_log_variances.append(
            np.asarray(
                [
                    state.forward_log_scale_variance,
                    state.sidestep_log_scale_variance,
                    state.sidestep_log_scale_variance,
                    state.forward_log_scale_variance,
                ]
            )
            + length.log_length_sigma**2
        )
    if smoothing_mode == "offline":
        probabilities = _smooth_mode_probabilities(posteriors)
        offline_posteriors: list[StepMotionPosterior] = []
        for posterior, values, length_means, log_variances in zip(
            posteriors,
            probabilities,
            mode_length_means,
            mode_log_variances,
            strict=True,
        ):
            length_mean, length_std = _length_moments(
                values,
                length_means,
                log_variances,
            )
            offline_posteriors.append(
                posterior._replace(
                    forward_probability=float(values[0]),
                    sidestep_left_probability=float(values[1]),
                    sidestep_right_probability=float(values[2]),
                    turning_probability=float(values[3]),
                    length_mean_m=length_mean,
                    length_std_m=length_std,
                    selected_mode=_MODE_NAMES[int(np.argmax(values))],
                    source="adaptive_offline",
                )
            )
        posteriors = offline_posteriors

    adjusted_headings = [
        heading._replace(
            selected_heading=posterior.heading_mean,
            source=f"trajectory_{posterior.source}",
            trajectory_movement_type=posterior.selected_mode,
            decoded_motion_mode=posterior.selected_mode,
            decoded_motion_confidence=max(
                posterior.forward_probability,
                posterior.sidestep_left_probability,
                posterior.sidestep_right_probability,
                posterior.turning_probability,
            ),
        )
        for heading, posterior in zip(step_headings, posteriors, strict=True)
    ]
    return AdaptivePdrResult(
        step_headings=adjusted_headings,
        step_lengths=[posterior.length_mean_m for posterior in posteriors],
        posteriors=tuple(posteriors),
        final_state=estimator.state,
    )


__all__ = ["AdaptivePdrEstimator", "estimate_adaptive_pdr"]
