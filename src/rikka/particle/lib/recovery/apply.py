"""particle filter の復旧結果適用。

役割:
    局所復旧、checkpoint再生、広域fallback、位置保持の各結果を実行状態へ適用する。
依存元:
    ``state`` の型付き実行状態、``local`` と ``checkpoint`` の候補生成、
    ``fallback`` の広域探索を利用する。
利用先:
    ``particle.lib.evaluate_map.resolve_map_constraints`` が地図制約違反時に使用する。
処理フロー:
    局所候補を試し、失敗時はcheckpoint再生、広域fallback、位置保持の順に選び、
    粒子状態・履歴・診断・可視化用候補を対応する結果へ更新する。
"""

import numpy as np

from ..state import ParticleRuntime
from .checkpoint import _replay_from_checkpoint, record_checkpoint_replay
from .fallback import recover_fallback
from .local import _generate_recovery_candidates


def _try_local_recovery(ctx: ParticleRuntime) -> bool:
    """現在歩の局所復旧候補を生成し、成功したかを返す。"""
    ctx.allow_turn_candidates = bool(
        ctx.motion_posterior is not None
        and ctx.motion_posterior.turning_probability >= 0.25
        or ctx.step_heading.trajectory_movement_type == "turning"
        or ctx.step_heading.movement_type == "turning"
        or (
            "sidestep"
            in (
                ctx.step_heading.trajectory_movement_type
                or ctx.step_heading.movement_type
            )
        )
        or (
            bool(ctx.step_headings)
            and (
                ctx.step_headings[-1].trajectory_movement_type == "turning"
                or ctx.step_headings[-1].movement_type == "turning"
            )
        )
    )
    ctx.recovery = _generate_recovery_candidates(
        ctx.particles_before,
        ctx.heading_correction_before,
        ctx.heading_drift_before,
        ctx.stride_scale_before,
        ctx.proposed_motion_state,
        ctx.weights_before,
        ctx.particle_base_headings,
        ctx.particle_step_lengths,
        ctx.sigma_sl_ratio,
        ctx.n_particles,
        ctx.map_gray,
        ctx.gx_mean,
        ctx.gz_mean,
        ctx.origin_px,
        ctx.scale,
        ctx.recovery_heading_sigma,
        ctx.recovery_max_attempts,
        ctx.rng,
        allow_turn_candidates=ctx.allow_turn_candidates,
        preserve_route_branches=ctx.preserve_recovery_branches,
        allow_stride_adaptation=ctx.adaptive_recovery_scale,
        stride_scale_min=ctx.effective_stride_scale_min,
        stride_scale_max=ctx.effective_stride_scale_max,
        capture_candidates=ctx.recorder.stages_enabled,
    )
    ctx.recovery_attempts = (
        ctx.recovery.attempts
        if ctx.recovery is not None
        else min(ctx.recovery_max_attempts, 2 if ctx.allow_turn_candidates else 1)
    )
    return ctx.recovery is not None


def _try_checkpoint_replay(ctx: ParticleRuntime) -> bool:
    """直近の健全checkpointから最大3歩を再生し、成功したかを返す。"""
    ctx.completed_steps = len(ctx.step_lengths)
    ctx.checkpoint_step = next(
        (
            candidate
            for candidate in ctx.healthy_checkpoint_steps
            if 0 < ctx.completed_steps - candidate <= 3
        ),
        None,
    )
    ctx.replay_result = None
    if ctx.checkpoint_step is None:
        return False

    ctx.replay_headings = ctx.step_headings[ctx.checkpoint_step :] + [ctx.step_heading]
    ctx.replay_angles = np.asarray(
        [heading.selected_heading for heading in ctx.replay_headings],
        dtype=float,
    )
    ctx.replay_lengths = np.asarray(
        ctx.step_lengths[ctx.checkpoint_step :] + [ctx.sl_det],
        dtype=float,
    )
    ctx.replay_result = _replay_from_checkpoint(
        ctx.position_history[ctx.checkpoint_step],
        ctx.heading_correction_history[ctx.checkpoint_step],
        ctx.heading_drift_history[ctx.checkpoint_step],
        ctx.stride_scale_history[ctx.checkpoint_step],
        ctx.weight_history[ctx.checkpoint_step],
        ctx.replay_angles,
        ctx.replay_lengths,
        ctx.n_particles,
        ctx.map_gray,
        ctx.gx_mean,
        ctx.gz_mean,
        ctx.origin_px,
        ctx.scale,
        ctx.recovery_heading_sigma,
        ctx.rng,
        checkpoint_motion_state=ctx.motion_state_history[ctx.checkpoint_step],
        allow_stride_adaptation=ctx.adaptive_recovery_scale,
        stride_scale_min=ctx.effective_stride_scale_min,
        stride_scale_max=ctx.effective_stride_scale_max,
        capture_candidates=ctx.recorder.stages_enabled,
    )
    ctx.recovery_attempts += 1
    return ctx.replay_result is not None


def _apply_checkpoint_replay(ctx: ParticleRuntime) -> None:
    """checkpoint再生結果で起点以後の履歴と現在状態を置き換える。"""
    assert ctx.checkpoint_step is not None
    assert ctx.replay_result is not None
    ctx.recovery = ctx.replay_result.recovery
    ctx.replay_depth = ctx.replay_result.replay_positions.shape[0]
    ctx.position_history = ctx.position_history[: ctx.checkpoint_step + 1]
    ctx.position_history.extend(ctx.replay_result.replay_positions[:-1].copy())
    ctx.all_particles_list = ctx.all_particles_list[: ctx.checkpoint_step + 1]
    ctx.all_particles_list.extend(ctx.replay_result.replay_positions[:-1].copy())
    ctx.parent_history = ctx.parent_history[: ctx.checkpoint_step]
    if ctx.replay_depth > 1:
        ctx.parent_history.append(ctx.recovery.parent_indices.copy())
        ctx.parent_history.extend(
            np.arange(ctx.n_particles, dtype=int) for _ in range(ctx.replay_depth - 2)
        )
    ctx.heading_correction_history = ctx.heading_correction_history[
        : ctx.checkpoint_step + 1
    ]
    ctx.heading_drift_history = ctx.heading_drift_history[: ctx.checkpoint_step + 1]
    ctx.motion_state_history = ctx.motion_state_history[: ctx.checkpoint_step + 1]
    ctx.stride_scale_history = ctx.stride_scale_history[: ctx.checkpoint_step + 1]
    ctx.weight_history = ctx.weight_history[: ctx.checkpoint_step + 1]
    ctx.path_log_score_history = ctx.path_log_score_history[: ctx.checkpoint_step + 1]
    ctx.healthy_checkpoint_steps = [
        step for step in ctx.healthy_checkpoint_steps if step <= ctx.checkpoint_step
    ]
    for _ in range(ctx.replay_depth - 1):
        ctx.heading_correction_history.append(ctx.recovery.heading_correction.copy())
        ctx.heading_drift_history.append(ctx.recovery.heading_drift.copy())
        ctx.motion_state_history.append(ctx.recovery.motion_state.copy())
        ctx.stride_scale_history.append(ctx.recovery.stride_scale.copy())
        ctx.weight_history.append(np.full(ctx.n_particles, 1.0 / ctx.n_particles))
        ctx.path_log_score_history.append(
            ctx.path_log_score_history[ctx.checkpoint_step][
                ctx.recovery.parent_indices
            ].copy()
        )
    record_checkpoint_replay(ctx)
    ctx.particles = ctx.recovery.particles
    ctx.heading_correction = ctx.recovery.heading_correction
    ctx.heading_drift = ctx.recovery.heading_drift
    ctx.stride_scale = ctx.recovery.stride_scale
    ctx.motion_state = ctx.recovery.motion_state
    ctx.weights = np.full(ctx.n_particles, 1.0 / ctx.n_particles)
    ctx.parent_indices = (
        np.arange(ctx.n_particles, dtype=int)
        if ctx.replay_depth > 1
        else ctx.recovery.parent_indices
    )
    ctx.diagnostic_motion_state_before = (
        ctx.recovery.motion_state
        if ctx.replay_depth > 1
        else ctx.motion_state_history[ctx.checkpoint_step]
    )
    ctx.effective_step_lengths_for_diagnostics = np.linalg.norm(
        ctx.particles
        - (
            ctx.replay_result.replay_positions[-2]
            if ctx.replay_depth > 1
            else ctx.position_history[ctx.checkpoint_step][ctx.recovery.parent_indices]
        ),
        axis=1,
    )
    ctx.next_path_log_scores = (
        ctx.path_log_score_history[ctx.checkpoint_step][
            ctx.recovery.parent_indices
        ].copy()
        + ctx.recovery.path_log_score_delta
    )
    ctx.recovery_mode = ctx.recovery.mode
    ctx.recovery_valid_count = ctx.recovery.valid_count
    ctx.recovery_heading_delta_deg = ctx.recovery.heading_delta_deg
    ctx.recovery_step_scale = ctx.recovery.step_scale
    ctx.recovery_cost = ctx.recovery.mean_cost
    ctx.recovery_checkpoint_step = ctx.checkpoint_step
    ctx.recovery_replay_steps = ctx.replay_depth
    ctx.recovery_candidate_branch_count = int(
        np.unique(ctx.recovery.route_branch_ids).size
    )
    ctx.recovery_selected_branch_count = ctx.recovery_candidate_branch_count
    ctx.recovery_candidate_headings = ctx.recovery.candidate_headings
    ctx.recovery_candidate_valid = ctx.recovery.candidate_valid
    ctx.recovery_selected_index = ctx.recovery.selected_candidate_indices
    ctx.resampled = True


def _apply_fallback_recovery(ctx: ParticleRuntime) -> bool:
    """広域fallback候補を生成し、成功時は現在状態へ適用する。"""
    ctx.fallback_recovery = recover_fallback(ctx)
    if ctx.fallback_recovery is None:
        return False

    recovery = ctx.fallback_recovery
    ctx.particles = recovery.particles
    ctx.heading_correction = recovery.heading_correction
    ctx.heading_drift = recovery.heading_drift
    ctx.stride_scale = recovery.stride_scale
    ctx.motion_state = recovery.motion_state
    ctx.weights = np.full(ctx.n_particles, 1.0 / ctx.n_particles)
    ctx.parent_indices = recovery.parent_indices
    ctx.effective_step_lengths_for_diagnostics = np.linalg.norm(
        ctx.particles - ctx.particles_before[ctx.parent_indices],
        axis=1,
    )
    ctx.next_path_log_scores = (
        ctx.path_log_scores_before[ctx.parent_indices] + recovery.path_log_score_delta
    )
    ctx.recovery_mode = f"fallback_{recovery.mode}"
    ctx.recovery_valid_count = recovery.valid_count
    ctx.recovery_attempts += recovery.attempts
    ctx.recovery_heading_delta_deg = recovery.heading_delta_deg
    ctx.recovery_step_scale = recovery.step_scale
    ctx.recovery_cost = recovery.mean_cost
    ctx.recovery_candidate_branch_count = int(np.unique(recovery.route_branch_ids).size)
    ctx.recovery_selected_branch_count = ctx.recovery_candidate_branch_count
    ctx.recovery_candidate_headings = recovery.candidate_headings
    ctx.recovery_candidate_valid = recovery.candidate_valid
    ctx.recovery_selected_index = recovery.selected_candidate_indices
    ctx.resampled = True
    return True


def _hold_on_recovery_failure(ctx: ParticleRuntime) -> None:
    """全復旧失敗時に直前状態を保持し、失敗診断を設定する。"""
    ctx.particles = ctx.particles_before
    ctx.heading_correction = ctx.heading_correction_before
    ctx.heading_drift = ctx.heading_drift_before
    ctx.stride_scale = ctx.stride_scale_before
    ctx.motion_state = ctx.proposed_motion_state
    ctx.weights = ctx.weights_before
    ctx.parent_indices = np.arange(ctx.n_particles, dtype=int)
    ctx.effective_step_lengths_for_diagnostics = np.zeros(
        ctx.n_particles,
        dtype=float,
    )
    ctx.next_path_log_scores = ctx.path_log_scores_before.copy()
    ctx.recovery_mode = "failed_hold"
    ctx.recovery_attempts += min(ctx.recovery_max_attempts, 2)


def _apply_local_recovery(ctx: ParticleRuntime) -> None:
    """局所復旧結果を現在状態へ適用する。"""
    assert ctx.recovery is not None
    ctx.particles = ctx.recovery.particles
    ctx.heading_correction = ctx.recovery.heading_correction
    ctx.heading_drift = ctx.recovery.heading_drift
    ctx.stride_scale = ctx.recovery.stride_scale
    ctx.motion_state = ctx.recovery.motion_state
    ctx.weights = np.full(ctx.n_particles, 1.0 / ctx.n_particles)
    ctx.parent_indices = ctx.recovery.parent_indices
    ctx.effective_step_lengths_for_diagnostics = np.linalg.norm(
        ctx.particles - ctx.particles_before[ctx.parent_indices],
        axis=1,
    )
    ctx.next_path_log_scores = (
        ctx.path_log_scores_before[ctx.parent_indices]
        + ctx.recovery.path_log_score_delta
    )
    ctx.recovery_mode = ctx.recovery.mode
    ctx.recovery_valid_count = ctx.recovery.valid_count
    ctx.recovery_heading_delta_deg = ctx.recovery.heading_delta_deg
    ctx.recovery_step_scale = ctx.recovery.step_scale
    ctx.recovery_cost = ctx.recovery.mean_cost
    ctx.recovery_candidate_branch_count = int(
        np.unique(ctx.recovery.route_branch_ids).size
    )
    ctx.recovery_selected_branch_count = ctx.recovery_candidate_branch_count
    ctx.recovery_candidate_headings = ctx.recovery.candidate_headings
    ctx.recovery_candidate_valid = ctx.recovery.candidate_valid
    ctx.recovery_selected_index = ctx.recovery.selected_candidate_indices
    ctx.resampled = True
