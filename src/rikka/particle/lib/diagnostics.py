"""パーティクルフィルタの1歩分の診断値構築。

役割:
    粒子更新後の状態、重み、親インデックス、復旧結果から1歩分の診断値を計算する。
依存元:
    ``models`` の診断データ型、``motion`` の角度統計と状態名、PDR bridgeの
    運動観測型、NumPyの配列演算を利用する。
利用先:
    ``particle.runner`` が各歩の履歴更新後に診断collectorへ追記する値を作るために
    使用する。
処理フロー:
    粒子分散、方位分散、歩幅倍率、実効歩幅、運動状態分布を従来と同じ順で計算し、
    軌跡選択前を示すpending状態の不変dataclassを返す。
"""

from typing import Any

import numpy as np

from ...particle.lib.recorder import ParticleFilterStepDiagnostics
from .proposal import _MOTION_STATE_NAMES, _normalize_angle, _weighted_circular_std


def _build_step_diagnostics(**values: Any) -> ParticleFilterStepDiagnostics:
    """粒子更新後の値から従来と同じ演算順で1歩分の診断値を返す。"""
    step_number = values["step_number"]
    step_time = values["step_time"]
    valid_count = values["valid_count"]
    n_particles = values["n_particles"]
    valid_weight_count = values["valid_weight_count"]
    valid_weight_mass = values["valid_weight_mass"]
    ess_before_observation = values["ess_before_observation"]
    ess_after_observation = values["ess_after_observation"]
    ess_after_resampling = values["ess_after_resampling"]
    weights = values["weights"]
    particles = values["particles"]
    heading_drift = values["heading_drift"]
    heading_correction = values["heading_correction"]
    stride_scale = values["stride_scale"]
    effective_step_lengths = values["effective_step_lengths"]
    parent_indices = values["parent_indices"]
    resampled = values["resampled"]
    motion_state = values["motion_state"]
    motion_state_before = values["motion_state_before"]
    motion_evidence = values["motion_evidence"]
    recovery_attempted = values["recovery_attempted"]
    recovery_mode = values["recovery_mode"]
    recovery_valid_count = values["recovery_valid_count"]
    recovery_attempts = values["recovery_attempts"]
    recovery_heading_delta_deg = values["recovery_heading_delta_deg"]
    recovery_step_scale = values["recovery_step_scale"]
    recovery_cost = values["recovery_cost"]
    recovery_checkpoint_step = values["recovery_checkpoint_step"]
    recovery_replay_steps = values["recovery_replay_steps"]
    recovery_candidate_branch_count = values["recovery_candidate_branch_count"]
    recovery_selected_branch_count = values["recovery_selected_branch_count"]
    unique_position_count = int(
        np.unique(np.round(particles, decimals=9), axis=0).shape[0]
    )
    position_center = np.average(particles, axis=0, weights=weights)
    position_spread_rms_m = float(
        np.sqrt(
            np.sum(weights * np.sum(np.square(particles - position_center), axis=1))
        )
    )
    heading_drift_std_deg = float(
        np.degrees(_weighted_circular_std(heading_drift, weights))
    )
    heading_total_std_deg = float(
        np.degrees(
            _weighted_circular_std(
                _normalize_angle(heading_correction + heading_drift),
                weights,
            )
        )
    )
    stride_scale_mean = float(np.sum(weights * stride_scale))
    stride_scale_std = float(
        np.sqrt(np.sum(weights * np.square(stride_scale - stride_scale_mean)))
    )
    effective_step_length_mean_m = float(np.sum(weights * effective_step_lengths))
    effective_step_length_std_m = float(
        np.sqrt(
            np.sum(
                weights
                * np.square(effective_step_lengths - effective_step_length_mean_m)
            )
        )
    )
    state_probabilities = np.asarray(
        [float(np.sum(weights[motion_state == state])) for state in range(4)]
    )
    representative_state_index = int(np.argmax(state_probabilities))
    positive_state_probabilities = state_probabilities[state_probabilities > 0.0]
    motion_state_entropy = float(
        -np.sum(positive_state_probabilities * np.log(positive_state_probabilities))
    )
    parent_states = motion_state_before[parent_indices]
    motion_state_transition_count = int(np.count_nonzero(motion_state != parent_states))
    return ParticleFilterStepDiagnostics(
        step=step_number,
        timestamp_s=float(step_time),
        valid_count=valid_count,
        valid_ratio=valid_count / n_particles,
        valid_weight_count=valid_weight_count,
        valid_weight_mass_before_normalization=valid_weight_mass,
        ess_before_observation=ess_before_observation,
        ess_after_observation=ess_after_observation,
        ess_after_resampling=ess_after_resampling,
        max_weight=float(np.max(weights)),
        unique_parent_count=int(np.unique(parent_indices).size),
        unique_position_count=unique_position_count,
        position_spread_rms_m=position_spread_rms_m,
        heading_drift_std_deg=heading_drift_std_deg,
        heading_total_std_deg=heading_total_std_deg,
        stride_scale_mean=stride_scale_mean,
        stride_scale_std=stride_scale_std,
        effective_step_length_mean_m=effective_step_length_mean_m,
        effective_step_length_std_m=effective_step_length_std_m,
        forward_state_probability=float(state_probabilities[0]),
        sidestep_left_state_probability=float(state_probabilities[1]),
        sidestep_right_state_probability=float(state_probabilities[2]),
        turning_state_probability=float(state_probabilities[3]),
        representative_motion_state=_MOTION_STATE_NAMES[representative_state_index],
        motion_state_entropy=motion_state_entropy,
        motion_state_transition_count=motion_state_transition_count,
        motion_reliability=motion_evidence.motion_reliability,
        calibration_reliability=(motion_evidence.calibration_reliability),
        resampled=resampled,
        recovery_attempted=recovery_attempted,
        recovery_mode=recovery_mode,
        recovery_valid_count=recovery_valid_count,
        recovery_attempts=recovery_attempts,
        recovery_heading_delta_deg=recovery_heading_delta_deg,
        recovery_step_scale=recovery_step_scale,
        recovery_cost=recovery_cost,
        recovery_checkpoint_step=recovery_checkpoint_step,
        recovery_replay_steps=recovery_replay_steps,
        trajectory_mode="pending",
        trajectory_source_index=None,
        recovery_candidate_branch_count=(recovery_candidate_branch_count),
        recovery_selected_branch_count=(recovery_selected_branch_count),
    )
