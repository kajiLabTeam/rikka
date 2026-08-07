"""移動軸の ``theta`` / ``theta + pi`` 方向曖昧性を区間単位で解決する。

役割:
    方向未確定の移動軸と前進・左右横歩き・旋回状態から、
    物理的に連続する有向方位列をViterbi復号する。
依存元:
    ``models`` から歩単位の方位・運動観測・事後分布型を取得し、
    ``motion_decoder`` のsemi-Markov区間復号を移動状態に利用する。
利用先:
    ``trajectory.prepare_pdr_steps`` が ``motion_estimation=robust`` のとき呼び出し、
    通常PDRとparticle filterで共有する方位列と診断値を生成する。
処理フロー:
    semi-Markov復号で運動区間を決め、各歩の移動軸に対する2方向候補を
    観測尤度と方位連続性で評価する。causalは固定lag、offlineは全列の
    Viterbiと前向後向確率を使い、根拠のない180度反転を抑制する。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ....common.lib.models import (
    StepDirectionPosterior,
    StepHeading,
    StepMotionObservation,
)
from ..motion_state.decoder import decode_step_motion_modes

_REVERSAL_THRESHOLD = np.deg2rad(135.0)
_YAW_SUPPORT_THRESHOLD = np.deg2rad(60.0)
_DIRECTED_SUPPORT_THRESHOLD = np.deg2rad(45.0)
_DIRECTED_CONFIDENCE_THRESHOLD = 0.6
_EPSILON = 1e-12


def _normalize_angle(angle: float) -> float:
    """角度を ``[-pi, pi)`` に正規化する。"""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _finite_angle(value: float | None) -> float | None:
    """有限な角度だけを正規化して返す。"""
    if value is None or not np.isfinite(value):
        return None
    return _normalize_angle(float(value))


def _normalize_mode(mode: str) -> str:
    """旋回付き横歩きを左右状態へ正規化する。"""
    if "sidestep_left" in mode:
        return "sidestep_left"
    if "sidestep_right" in mode:
        return "sidestep_right"
    if mode == "turning":
        return "turning"
    return "forward"


def _mode_sequence(
    observations: Sequence[StepMotionObservation],
) -> tuple[str, ...]:
    """semi-Markov復号と旋回観測から運動状態列を作る。"""
    decoded = decode_step_motion_modes(list(observations))
    modes: list[str] = []
    for observation, decoded_mode in zip(observations, decoded, strict=True):
        raw_mode = _normalize_mode(observation.trajectory_movement_type)
        yaw = abs(float(observation.yaw_delta or 0.0))
        if raw_mode == "turning" or (
            yaw >= np.deg2rad(35.0) and decoded_mode == "forward"
        ):
            modes.append("turning")
        else:
            modes.append(_normalize_mode(decoded_mode))
    return tuple(modes)


def _axis_candidates(
    heading: StepHeading,
    observation: StepMotionObservation,
) -> tuple[float, float]:
    """同じ移動軸を表す反対向きの2候補を返す。"""
    axis = _finite_angle(observation.motion_axis_heading)
    if axis is None:
        selected = _finite_angle(heading.selected_heading)
        axis = 0.0 if selected is None else float(selected % np.pi)
    return axis, _normalize_angle(axis + np.pi)


def _expected_heading(
    observation: StepMotionObservation,
    mode: str,
) -> float | None:
    """身体方位と運動状態から期待移動方位を返す。"""
    body = _finite_angle(observation.body_heading_candidate)
    if body is None:
        body = _finite_angle(observation.device_yaw_heading)
    if body is None:
        return None
    if mode == "sidestep_left":
        return _normalize_angle(body + np.pi / 2.0)
    if mode == "sidestep_right":
        return _normalize_angle(body - np.pi / 2.0)
    return body


def _emission_scores(
    candidates: tuple[float, float],
    observation: StepMotionObservation,
    mode: str,
) -> np.ndarray:
    """身体方位と有向観測の信頼度を反映した尤度を返す。"""
    scores = np.zeros(2, dtype=float)
    calibration = float(np.clip(observation.calibration_reliability, 0.0, 1.0))
    expected = _expected_heading(observation, mode)
    if expected is not None:
        sigma = np.deg2rad(70.0 if mode == "turning" else 32.0)
        for state, candidate in enumerate(candidates):
            residual = _normalize_angle(candidate - expected)
            scores[state] -= calibration * 0.5 * (residual / sigma) ** 2

    directed = _finite_angle(observation.directed_motion_heading)
    directed_weight = (
        float(np.clip(observation.motion_confidence, 0.0, 1.0)) * calibration
    )
    if directed is not None and directed_weight > 0.0:
        sigma = np.deg2rad(38.0)
        for state, candidate in enumerate(candidates):
            residual = _normalize_angle(candidate - directed)
            scores[state] -= directed_weight * 0.5 * (residual / sigma) ** 2
    return scores


def _flip_support(
    index: int,
    candidate: float,
    observations: Sequence[StepMotionObservation],
) -> bool:
    """3歩累yawまたは3歩連続の有向観測が反転を支持するか返す。"""
    start = max(0, index - 2)
    window = observations[start : index + 1]
    if len(window) < 3:
        return False
    accumulated_yaw = abs(sum(float(item.yaw_delta or 0.0) for item in window))
    if accumulated_yaw >= _YAW_SUPPORT_THRESHOLD:
        return True
    return all(
        item.motion_confidence >= _DIRECTED_CONFIDENCE_THRESHOLD
        and (directed := _finite_angle(item.directed_motion_heading)) is not None
        and abs(_normalize_angle(directed - candidate)) <= _DIRECTED_SUPPORT_THRESHOLD
        for item in window
    )


def _transition_score(
    index: int,
    previous_candidate: float,
    candidate: float,
    previous_mode: str,
    mode: str,
    observations: Sequence[StepMotionObservation],
) -> float:
    """方位連続性のスコアを返し、根拠のない180度反転を禁止する。"""
    yaw = float(observations[index].yaw_delta or 0.0)
    physical_delta = abs(_normalize_angle(candidate - previous_candidate))
    if (
        previous_mode == mode
        and physical_delta >= _REVERSAL_THRESHOLD
        and not _flip_support(index, candidate, observations)
    ):
        return float("-inf")
    expected = _normalize_angle(previous_candidate + yaw)
    residual = _normalize_angle(candidate - expected)
    sigma = np.deg2rad(75.0 if mode == "turning" else 30.0)
    return float(-0.5 * (residual / sigma) ** 2)


def _decode_sequence(
    headings: Sequence[StepHeading],
    observations: Sequence[StepMotionObservation],
    modes: Sequence[str],
) -> tuple[tuple[int, ...], tuple[tuple[float, float], ...]]:
    """2方向HMMのViterbi列と前向後向事後確率を返す。"""
    length = len(headings)
    if length == 0:
        return (), ()
    candidates = [
        _axis_candidates(heading, observation)
        for heading, observation in zip(headings, observations, strict=True)
    ]
    emissions = [
        _emission_scores(candidate, observation, mode)
        for candidate, observation, mode in zip(
            candidates, observations, modes, strict=True
        )
    ]
    scores = np.full((length, 2), float("-inf"), dtype=float)
    back = np.zeros((length, 2), dtype=int)
    scores[0] = emissions[0]
    # 校正尤度が同点でも、呼び出し元が既に選んだ初期方向を捨てない。
    # ごく小さい値に留め、センサー根拠がある場合の尤度判断は変えない。
    initial_selected = _finite_angle(headings[0].selected_heading)
    if initial_selected is not None:
        preferred_state = min(
            range(2),
            key=lambda state: abs(
                _normalize_angle(candidates[0][state] - initial_selected)
            ),
        )
        scores[0, preferred_state] += 1e-9
    transitions: list[np.ndarray] = [np.zeros((2, 2), dtype=float)]
    for index in range(1, length):
        matrix = np.full((2, 2), float("-inf"), dtype=float)
        for previous_state in range(2):
            for state in range(2):
                matrix[previous_state, state] = _transition_score(
                    index,
                    candidates[index - 1][previous_state],
                    candidates[index][state],
                    modes[index - 1],
                    modes[index],
                    observations,
                )
        transitions.append(matrix)
        for state in range(2):
            values = scores[index - 1] + matrix[:, state]
            back[index, state] = int(np.argmax(values))
            scores[index, state] = float(np.max(values) + emissions[index][state])

    states = [0] * length
    states[-1] = int(np.argmax(scores[-1]))
    for index in range(length - 1, 0, -1):
        states[index - 1] = int(back[index, states[index]])

    # max-productの前向・後向スコアから、低校正時に2仮説が
    # 残っていることを診断可能な近似事後確率として返す。
    backward = np.zeros((length, 2), dtype=float)
    for index in range(length - 2, -1, -1):
        for state in range(2):
            backward[index, state] = max(
                transitions[index + 1][state, next_state]
                + emissions[index + 1][next_state]
                + backward[index + 1, next_state]
                for next_state in range(2)
            )
    probabilities: list[tuple[float, float]] = []
    for index in range(length):
        combined = scores[index] + backward[index]
        combined -= np.max(combined)
        weights = np.exp(combined)
        weights /= max(float(weights.sum()), _EPSILON)
        probabilities.append((float(weights[0]), float(weights[1])))
    return tuple(states), tuple(probabilities)


def resolve_step_directions(
    step_headings: Sequence[StepHeading],
    observations: Sequence[StepMotionObservation],
    smoothing_mode: str = "causal",
    fixed_lag: int = 5,
) -> tuple[list[StepHeading], tuple[StepDirectionPosterior, ...]]:
    """方向曖昧性を解決した方位列と歩ごとの事後分布を返す。"""
    if smoothing_mode not in {"causal", "offline"}:
        raise ValueError("smoothing_mode は causal または offline を指定してください")
    if fixed_lag < 0:
        raise ValueError("fixed_lag は0以上を指定してください")
    if len(step_headings) != len(observations):
        raise ValueError("方位列と運動観測列の長さが一致しません")
    if not step_headings:
        return [], ()

    modes = _mode_sequence(observations)
    if smoothing_mode == "offline":
        states, probabilities = _decode_sequence(step_headings, observations, modes)
    else:
        selected_states: list[int] = []
        selected_probabilities: list[tuple[float, float]] = []
        for target in range(len(step_headings)):
            end = min(len(step_headings), target + fixed_lag + 1)
            prefix_states, prefix_probabilities = _decode_sequence(
                step_headings[:end],
                observations[:end],
                modes[:end],
            )
            selected_states.append(prefix_states[target])
            selected_probabilities.append(prefix_probabilities[target])
        states = tuple(selected_states)
        probabilities = tuple(selected_probabilities)

    adjusted: list[StepHeading] = []
    posteriors: list[StepDirectionPosterior] = []
    previous_heading: float | None = None
    previous_mode: str | None = None
    for index, (heading, observation, mode, state, probability) in enumerate(
        zip(step_headings, observations, modes, states, probabilities, strict=True)
    ):
        candidates = _axis_candidates(heading, observation)
        selected_state = state
        selected = candidates[selected_state]
        reversal_support = _flip_support(index, selected, observations)
        is_reversal = (
            previous_heading is not None
            and previous_mode == mode
            and abs(_normalize_angle(selected - previous_heading))
            >= _REVERSAL_THRESHOLD
        )
        if is_reversal and not reversal_support:
            selected_state = 1 - selected_state
            selected = candidates[selected_state]
            is_reversal = False
        selected_probability = probability[selected_state]
        adjusted.append(
            heading._replace(
                selected_heading=selected,
                source=f"trajectory_robust_{smoothing_mode}",
                trajectory_movement_type=mode,
                decoded_motion_mode=mode,
                decoded_motion_confidence=max(probability),
            )
        )
        posteriors.append(
            StepDirectionPosterior(
                step_index=heading.step_index,
                positive_axis_probability=probability[0],
                negative_axis_probability=probability[1],
                selected_heading=selected,
                selected_motion_mode=mode,
                confidence=selected_probability,
                source=f"robust_{smoothing_mode}",
                flip_supported=is_reversal and reversal_support,
            )
        )
        previous_heading = selected
        previous_mode = mode
    return adjusted, tuple(posteriors)


__all__ = ["resolve_step_directions"]
