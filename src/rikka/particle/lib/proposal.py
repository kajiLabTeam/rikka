"""粒子ごとの歩行運動状態モデル。

役割:
    前進・左右横歩き・旋回の状態遷移、観測尤度、状態別方位と角度統計を扱う。
依存元:
    ``config`` の状態継続率と ``common`` の歩行方位・観測型を取得する。
利用先:
    粒子フィルタの主ループとrecovery処理が運動状態の提案と方位更新に使用する。
処理フロー:
    前状態と観測から提案分布を作り、状態を標本化し、対応する移動方位を返す。
"""

import numpy as np

from ...common.config import PF_MOTION_STATE_TRANSITION_STAY
from ...common.lib.models import StepHeading, StepMotionEvidence

_MOTION_FORWARD = 0
_MOTION_SIDESTEP_LEFT = 1
_MOTION_SIDESTEP_RIGHT = 2
_MOTION_TURNING = 3
_MOTION_STATE_NAMES = ("forward", "sidestep_left", "sidestep_right", "turning")


def _normalize_angle(angle: np.ndarray) -> np.ndarray:
    """角度配列を -pi 以上 pi 未満へ正規化する。"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _motion_state_transition_matrix() -> np.ndarray:
    """前進・左右横歩き・旋回の継続性を表す遷移行列を返す。"""
    side_stay = PF_MOTION_STATE_TRANSITION_STAY
    side_transition = (1.0 - side_stay) / 3.0
    return np.asarray(
        [
            [0.90, 0.035, 0.035, 0.03],
            [side_transition, side_stay, side_transition, side_transition],
            [side_transition, side_transition, side_stay, side_transition],
            [0.45, 0.08, 0.08, 0.39],
        ],
        dtype=float,
    )


def _sample_motion_states(
    previous_states: np.ndarray,
    observation_likelihoods: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """遷移と観測を使う最適提案分布から次状態と予測尤度を返す。"""
    unnormalized = (
        _motion_state_transition_matrix()[previous_states]
        * observation_likelihoods[None, :]
    )
    predictive_likelihoods = unnormalized.sum(axis=1)
    probabilities = unnormalized / predictive_likelihoods[:, None]
    draws = rng.random(len(previous_states))
    states = np.sum(draws[:, None] > np.cumsum(probabilities, axis=1), axis=1).astype(
        np.int8
    )
    return states, predictive_likelihoods


def _motion_state_headings(
    step_heading: StepHeading,
    evidence: StepMotionEvidence,
    particle_heading: float,
) -> np.ndarray:
    """1歩の運動状態別方位候補を返す。"""
    fallback = particle_heading
    if (
        evidence.calibration_reliability >= 0.85
        or step_heading.sidestep_cluster_id is None
    ):
        return np.full(4, fallback, dtype=float)
    body = (
        step_heading.body_heading if step_heading.body_heading is not None else fallback
    )
    movement = step_heading.trajectory_movement_type or step_heading.movement_type
    motion = (
        step_heading.motion_heading
        if step_heading.motion_heading is not None
        else fallback
    )
    forward = body
    left = body + np.pi / 2
    right = body - np.pi / 2
    if movement in {
        "sidestep_left",
        "turning_sidestep_left",
        "sidestep_suspect_left",
    }:
        left = motion
    elif movement in {
        "sidestep_right",
        "turning_sidestep_right",
        "sidestep_suspect_right",
    }:
        right = motion
    turning = motion if movement.startswith("turning") else fallback
    return _normalize_angle(np.asarray([forward, left, right, turning], dtype=float))


def _motion_state_likelihoods(evidence: StepMotionEvidence) -> np.ndarray:
    """運動観測を粒子状態順の尤度配列へ変換する。"""
    return np.asarray(
        [
            evidence.forward_likelihood,
            evidence.sidestep_left_likelihood,
            evidence.sidestep_right_likelihood,
            evidence.turning_likelihood,
        ],
        dtype=float,
    )


def _weighted_circular_std(angles: np.ndarray, weights: np.ndarray) -> float:
    """重み付き角度分布の円周標準偏差をラジアンで返す。"""
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    cosine = float(np.sum(weights * np.cos(angles)) / total)
    sine = float(np.sum(weights * np.sin(angles)) / total)
    resultant = float(np.clip(np.hypot(cosine, sine), 1e-15, 1.0))
    return float(np.sqrt(max(0.0, -2.0 * np.log(resultant))))
