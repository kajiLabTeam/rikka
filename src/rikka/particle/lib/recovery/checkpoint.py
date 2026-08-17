"""checkpoint から粒子列を再生する復旧戦略。

役割:
    健全だった過去状態から複数歩を再生し、壁を横切らない粒子列を選ぶ。
依存元:
    ``local`` の共通復旧結果、地図遷移判定、角度正規化を利用する。
利用先:
    particle の復旧段階と互換 API から呼び出される。
処理フロー:
    方位差と歩幅倍率を展開し、全再生歩が有効な候補を重み付き抽出する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..diagnostics import _build_step_diagnostics
from ..map_constraints import _evaluate_particle_transitions
from ..proposal import _normalize_angle
from ..recorder import ParticleStepStages
from .local import (
    _LOCAL_OFFSET_DEGREES,
    _recovery_length_factors,
    _RecoveryResult,
)

if TYPE_CHECKING:
    from ..state import ParticleRuntime


@dataclass(frozen=True)
class _CheckpointReplayResult:
    """checkpointから再生した粒子列と最終状態。"""

    recovery: _RecoveryResult
    replay_positions: np.ndarray


def record_checkpoint_replay(ctx: ParticleRuntime) -> None:
    """checkpoint 再生で置換された診断と段階状態を書き戻す。"""
    assert ctx.checkpoint_step is not None
    assert ctx.replay_result is not None
    assert ctx.recovery is not None
    recorder = ctx.recorder

    if recorder.diagnostics_enabled:
        replay_weights = np.full(ctx.n_particles, 1.0 / ctx.n_particles)
        for replay_offset in range(ctx.replay_depth - 1):
            history_step = ctx.checkpoint_step + replay_offset + 1
            replay_parent_indices = (
                ctx.recovery.parent_indices
                if replay_offset == 0
                else np.arange(ctx.n_particles, dtype=int)
            )
            replay_motion_state_before = (
                ctx.motion_state_history[ctx.checkpoint_step]
                if replay_offset == 0
                else ctx.recovery.motion_state
            )
            replay_previous_positions = (
                ctx.position_history[ctx.checkpoint_step][ctx.recovery.parent_indices]
                if replay_offset == 0
                else ctx.replay_result.replay_positions[replay_offset - 1]
            )
            replay_effective_lengths = np.linalg.norm(
                ctx.replay_result.replay_positions[replay_offset]
                - replay_previous_positions,
                axis=1,
            )
            collector_index = ctx.diagnostics_start_index + history_step - 1
            previous_diagnostic = recorder.diagnostics[collector_index]
            recorder.diagnostics[collector_index] = _build_step_diagnostics(
                step_number=history_step,
                step_time=ctx.t_at_steps[history_step - 1],
                valid_count=ctx.n_particles,
                n_particles=ctx.n_particles,
                valid_weight_count=ctx.n_particles,
                valid_weight_mass=1.0,
                ess_before_observation=float(ctx.n_particles),
                ess_after_observation=float(ctx.n_particles),
                ess_after_resampling=float(ctx.n_particles),
                weights=replay_weights,
                particles=ctx.replay_result.replay_positions[replay_offset],
                heading_drift=ctx.recovery.heading_drift,
                heading_correction=ctx.recovery.heading_correction,
                stride_scale=ctx.recovery.stride_scale,
                effective_step_lengths=replay_effective_lengths,
                parent_indices=replay_parent_indices,
                resampled=replay_offset == 0,
                motion_state=ctx.recovery.motion_state,
                motion_state_before=replay_motion_state_before,
                motion_evidence=ctx.motion_evidences[history_step - 1],
                recovery_attempted=True,
                recovery_mode="checkpoint_replayed",
                recovery_valid_count=ctx.recovery.valid_count,
                recovery_attempts=ctx.recovery_attempts,
                recovery_heading_delta_deg=ctx.recovery.heading_delta_deg,
                recovery_step_scale=ctx.recovery.step_scale,
                recovery_cost=ctx.recovery.mean_cost,
                recovery_checkpoint_step=ctx.checkpoint_step,
                recovery_replay_steps=ctx.replay_depth,
                recovery_candidate_branch_count=int(
                    np.unique(ctx.recovery.route_branch_ids).size
                ),
                recovery_selected_branch_count=int(
                    np.unique(ctx.recovery.route_branch_ids).size
                ),
                landmark_beacon_id=previous_diagnostic.landmark_beacon_id,
                landmark_distance_m=previous_diagnostic.landmark_distance_m,
                landmark_likelihood_mean=(previous_diagnostic.landmark_likelihood_mean),
            )

    if recorder.stages_enabled:
        replay_weights = np.full(ctx.n_particles, 1.0 / ctx.n_particles)
        replay_offsets = _normalize_angle(
            ctx.recovery.heading_correction + ctx.recovery.heading_drift
        )
        for replay_offset in range(ctx.replay_depth - 1):
            history_step = ctx.checkpoint_step + replay_offset + 1
            replay_heading = ctx.step_headings[history_step - 1]
            replay_parent_indices = (
                ctx.recovery.parent_indices
                if replay_offset == 0
                else np.arange(ctx.n_particles, dtype=int)
            )
            replay_before_positions = (
                ctx.position_history[ctx.checkpoint_step][ctx.recovery.parent_indices]
                if replay_offset == 0
                else ctx.replay_result.replay_positions[replay_offset - 1]
            )
            replay_after_positions = ctx.replay_result.replay_positions[replay_offset]
            replay_step_lengths = np.linalg.norm(
                replay_after_positions - replay_before_positions,
                axis=1,
            )
            replay_sensor_heading = replay_heading.selected_heading
            replay_proposed_headings = (
                np.full(
                    ctx.n_particles,
                    float(replay_sensor_heading or 0.0),
                )
                + replay_offsets
            )
            collector_index = ctx.stages_start_index + history_step - 1
            recorder.stages[collector_index] = ParticleStepStages(
                step=history_step,
                timestamp_s=ctx.t_at_steps[history_step - 1],
                sensor_heading=replay_sensor_heading,
                sensor_yaw_delta=replay_heading.yaw_delta,
                movement_type=replay_heading.trajectory_movement_type
                or replay_heading.movement_type,
                deterministic_step_length_m=ctx.step_lengths[history_step - 1],
                before_positions=replay_before_positions.copy(),
                before_offsets=replay_offsets.copy(),
                before_weights=replay_weights.copy(),
                before_motion_state=ctx.recovery.motion_state.copy(),
                proposed_positions=replay_after_positions.copy(),
                proposed_headings=replay_proposed_headings.copy(),
                proposed_step_lengths=replay_step_lengths.copy(),
                proposed_motion_state=ctx.recovery.motion_state.copy(),
                valid_transition=np.ones(ctx.n_particles, dtype=bool),
                posterior_weights=replay_weights.copy(),
                ess_before_observation=float(ctx.n_particles),
                ess_after_observation=float(ctx.n_particles),
                parent_indices=replay_parent_indices.copy(),
                resampled=replay_offset == 0,
                recovery_mode="checkpoint_replayed",
                recovery_candidate_headings=(
                    ctx.recovery.candidate_headings.copy()
                    if replay_offset == 0
                    and ctx.recovery.candidate_headings is not None
                    else None
                ),
                recovery_candidate_valid=(
                    ctx.recovery.candidate_valid.copy()
                    if replay_offset == 0 and ctx.recovery.candidate_valid is not None
                    else None
                ),
                recovery_selected_index=(
                    ctx.recovery.selected_candidate_indices.copy()
                    if replay_offset == 0
                    and ctx.recovery.selected_candidate_indices is not None
                    else None
                ),
                after_positions=replay_after_positions.copy(),
                after_offsets=replay_offsets.copy(),
                after_weights=replay_weights.copy(),
                after_motion_state=ctx.recovery.motion_state.copy(),
            )


def _replay_from_checkpoint(
    checkpoint_particles: np.ndarray,
    checkpoint_heading_correction: np.ndarray,
    checkpoint_heading_drift: np.ndarray,
    checkpoint_stride_scale: np.ndarray,
    checkpoint_weights: np.ndarray,
    angles: np.ndarray,
    step_lengths: np.ndarray,
    n_particles: int,
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
    heading_sigma: float,
    rng: np.random.Generator,
    *,
    checkpoint_motion_state: np.ndarray | None = None,
    allow_stride_adaptation: bool = False,
    stride_scale_min: float = 0.5,
    stride_scale_max: float = 1.6,
    capture_candidates: bool = False,
) -> _CheckpointReplayResult | None:
    """同じ小方位差で最大3歩を再生し、壁非交差経路を返す。"""
    if len(angles) == 0 or len(angles) != len(step_lengths):
        return None
    offset_degrees = _LOCAL_OFFSET_DEGREES
    length_factors = _recovery_length_factors(allow_stride_adaptation)
    combinations = np.array(
        [(offset, factor) for offset in offset_degrees for factor in length_factors]
    )
    parent_indices = np.repeat(np.arange(n_particles), len(combinations))
    combination_indices = np.tile(np.arange(len(combinations)), n_particles)
    batch_size = len(parent_indices)
    offsets = np.deg2rad(combinations[combination_indices, 0])
    length_factor = combinations[combination_indices, 1]
    parent_correction = checkpoint_heading_correction[parent_indices]
    parent_drift = checkpoint_heading_drift[parent_indices]
    candidate_stride_scales = checkpoint_stride_scale[parent_indices]
    current = checkpoint_particles[parent_indices].copy()
    replay_positions: list[np.ndarray] = []
    valid = np.ones(batch_size, dtype=bool)
    for angle, step_length in zip(angles, step_lengths, strict=True):
        theta = angle + parent_correction + parent_drift + offsets
        proposed = current.copy()
        effective_stride_scales = (
            length_factor
            if allow_stride_adaptation
            else candidate_stride_scales * length_factor
        )
        lengths = step_length * effective_stride_scales
        proposed[:, 0] += lengths * np.cos(theta)
        proposed[:, 1] += lengths * np.sin(theta)
        valid &= _evaluate_particle_transitions(
            current,
            proposed,
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        current = np.where(valid[:, None], proposed, current)
        replay_positions.append(current.copy())
    if not valid.any():
        return None

    valid &= checkpoint_weights[parent_indices] > 0.0
    if not valid.any():
        return None

    valid_indices = np.flatnonzero(valid)
    heading_cost = np.square(offsets[valid] / max(heading_sigma, np.deg2rad(5.0)))
    length_cost_sigma = 0.22 if allow_stride_adaptation else 0.1
    length_cost_center = (
        candidate_stride_scales[valid] if allow_stride_adaptation else 1.0
    )
    length_cost = np.square(
        (length_factor[valid] - length_cost_center) / length_cost_sigma
    )
    costs = heading_cost + length_cost
    probabilities = checkpoint_weights[parent_indices[valid]] * np.exp(
        -0.5 * (costs - costs.min())
    )
    probabilities /= probabilities.sum()
    selected_local = rng.choice(
        len(valid_indices),
        size=n_particles,
        replace=len(valid_indices) < n_particles,
        p=probabilities,
    )
    select = valid_indices[selected_local]
    selected_offsets = offsets[select]
    selected_factors = length_factor[select]
    selected_costs = costs[selected_local]
    recovery = _RecoveryResult(
        particles=current[select],
        heading_correction=parent_correction[select],
        heading_drift=_normalize_angle(parent_drift[select] + selected_offsets),
        stride_scale=np.clip(
            selected_factors
            if allow_stride_adaptation
            else candidate_stride_scales[select],
            stride_scale_min,
            stride_scale_max,
        ),
        motion_state=(
            np.zeros(n_particles, dtype=np.int8)
            if checkpoint_motion_state is None
            else checkpoint_motion_state[parent_indices[select]]
        ),
        parent_indices=parent_indices[select],
        valid_count=int(np.count_nonzero(valid)),
        attempts=1,
        mode="checkpoint_replay",
        heading_delta_deg=float(np.degrees(np.mean(np.abs(selected_offsets)))),
        step_scale=float(np.mean(selected_factors)),
        mean_cost=float(np.mean(selected_costs)),
        route_branch_ids=np.zeros(n_particles, dtype=np.int8),
        path_log_score_delta=-0.5 * selected_costs,
        candidate_headings=theta.copy() if capture_candidates else None,
        candidate_valid=valid.copy() if capture_candidates else None,
        selected_candidate_indices=select.copy() if capture_candidates else None,
    )
    selected_positions = np.stack(
        [positions[select] for positions in replay_positions],
        axis=0,
    )
    return _CheckpointReplayResult(recovery, selected_positions)
