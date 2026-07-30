"""particle filter のevaluate_map。

役割:
    evaluate_mapを独立した段階として実装する。
依存元:
    common の共有型・設定と particle/lib の部品、ParticleRuntime を利用する。
利用先:
    particle/lib/runner が元の実行順序どおりに呼び出す。
処理フロー:
    提案された遷移へ地図拘束を適用する。
"""

import numpy as np

from ...particle.lib.proposal import (
    _normalize_angle,
)
from ...particle.lib.recorder import (
    ParticleStepStages,
)
from ...particle.lib.recovery.local import (
    _generate_recovery_candidates,
    _replay_from_checkpoint,
)
from ...particle.lib.resampling import (
    _systematic_resample,
)
from .diagnostics import _build_step_diagnostics
from .state import ParticleRuntime


def evaluate_map(ctx: ParticleRuntime) -> None:
    ctx.recovery_step_scale = None
    ctx.recovery_cost = None
    ctx.recovery_checkpoint_step = None
    ctx.recovery_replay_steps = 0
    ctx.recovery_candidate_branch_count = 0
    ctx.recovery_selected_branch_count = 0
    ctx.recovery_candidate_headings = None
    ctx.recovery_candidate_valid = None
    ctx.recovery_selected_index = None
    ctx.resampled = False
    ctx.next_path_log_scores = np.empty(0, dtype=float)
    if ctx.recovery_attempted:
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
        if ctx.recovery is None:
            ctx.completed_steps = len(ctx.step_lengths)
            ctx.checkpoint_step = next(
                (
                    ctx.candidate
                    for ctx.candidate in ctx.healthy_checkpoint_steps
                    if 0 < ctx.completed_steps - ctx.candidate <= 3
                ),
                None,
            )
            ctx.replay_result = None
            if ctx.checkpoint_step is not None:
                ctx.replay_headings = ctx.step_headings[ctx.checkpoint_step :] + [
                    ctx.step_heading
                ]
                ctx.replay_angles = np.asarray(
                    [
                        ctx.heading.selected_heading
                        for ctx.heading in ctx.replay_headings
                    ],
                    dtype=float,
                )
                ctx.replay_lengths = np.asarray(
                    ctx.step_lengths[ctx.checkpoint_step :] + [ctx.sl_det], dtype=float
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
                    checkpoint_motion_state=ctx.motion_state_history[
                        ctx.checkpoint_step
                    ],
                    allow_stride_adaptation=ctx.adaptive_recovery_scale,
                    stride_scale_min=ctx.effective_stride_scale_min,
                    stride_scale_max=ctx.effective_stride_scale_max,
                    capture_candidates=ctx.recorder.stages_enabled,
                )
                ctx.recovery_attempts += 1
            if ctx.replay_result is None:
                ctx.fallback_recovery = _generate_recovery_candidates(
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
                    allow_turn_candidates=True,
                    preserve_route_branches=ctx.preserve_recovery_branches,
                    allow_stride_adaptation=ctx.adaptive_recovery_scale,
                    stride_scale_min=ctx.effective_stride_scale_min,
                    stride_scale_max=ctx.effective_stride_scale_max,
                    capture_candidates=ctx.recorder.stages_enabled,
                )
                if ctx.fallback_recovery is None:
                    ctx.particles = ctx.particles_before
                    ctx.heading_correction = ctx.heading_correction_before
                    ctx.heading_drift = ctx.heading_drift_before
                    ctx.stride_scale = ctx.stride_scale_before
                    ctx.motion_state = ctx.proposed_motion_state
                    ctx.weights = ctx.weights_before
                    ctx.parent_indices = np.arange(ctx.n_particles, dtype=int)
                    ctx.effective_step_lengths_for_diagnostics = np.zeros(
                        ctx.n_particles, dtype=float
                    )
                    ctx.next_path_log_scores = ctx.path_log_scores_before.copy()
                    ctx.recovery_mode = "failed_hold"
                    ctx.recovery_attempts += min(ctx.recovery_max_attempts, 2)
                else:
                    ctx.particles = ctx.fallback_recovery.particles
                    ctx.heading_correction = ctx.fallback_recovery.heading_correction
                    ctx.heading_drift = ctx.fallback_recovery.heading_drift
                    ctx.stride_scale = ctx.fallback_recovery.stride_scale
                    ctx.motion_state = ctx.fallback_recovery.motion_state
                    ctx.weights = np.full(ctx.n_particles, 1.0 / ctx.n_particles)
                    ctx.parent_indices = ctx.fallback_recovery.parent_indices
                    ctx.effective_step_lengths_for_diagnostics = np.linalg.norm(
                        ctx.particles - ctx.particles_before[ctx.parent_indices], axis=1
                    )
                    ctx.next_path_log_scores = (
                        ctx.path_log_scores_before[ctx.parent_indices]
                        + ctx.fallback_recovery.path_log_score_delta
                    )
                    ctx.recovery_mode = f"fallback_{ctx.fallback_recovery.mode}"
                    ctx.recovery_valid_count = ctx.fallback_recovery.valid_count
                    ctx.recovery_attempts += ctx.fallback_recovery.attempts
                    ctx.recovery_heading_delta_deg = (
                        ctx.fallback_recovery.heading_delta_deg
                    )
                    ctx.recovery_step_scale = ctx.fallback_recovery.step_scale
                    ctx.recovery_cost = ctx.fallback_recovery.mean_cost
                    ctx.recovery_candidate_branch_count = int(
                        np.unique(ctx.fallback_recovery.route_branch_ids).size
                    )
                    ctx.recovery_selected_branch_count = (
                        ctx.recovery_candidate_branch_count
                    )
                    ctx.recovery_candidate_headings = (
                        ctx.fallback_recovery.candidate_headings
                    )
                    ctx.recovery_candidate_valid = ctx.fallback_recovery.candidate_valid
                    ctx.recovery_selected_index = (
                        ctx.fallback_recovery.selected_candidate_indices
                    )
                    ctx.resampled = True
            else:
                ctx.recovery = ctx.replay_result.recovery
                assert ctx.checkpoint_step is not None
                ctx.replay_depth = ctx.replay_result.replay_positions.shape[0]
                ctx.position_history = ctx.position_history[: ctx.checkpoint_step + 1]
                ctx.position_history.extend(
                    ctx.replay_result.replay_positions[:-1].copy()
                )
                ctx.all_particles_list = ctx.all_particles_list[
                    : ctx.checkpoint_step + 1
                ]
                ctx.all_particles_list.extend(
                    ctx.replay_result.replay_positions[:-1].copy()
                )
                ctx.parent_history = ctx.parent_history[: ctx.checkpoint_step]
                if ctx.replay_depth > 1:
                    ctx.parent_history.append(ctx.recovery.parent_indices.copy())
                    ctx.parent_history.extend(
                        np.arange(ctx.n_particles, dtype=int)
                        for ctx._ in range(ctx.replay_depth - 2)
                    )
                ctx.heading_correction_history = ctx.heading_correction_history[
                    : ctx.checkpoint_step + 1
                ]
                ctx.heading_drift_history = ctx.heading_drift_history[
                    : ctx.checkpoint_step + 1
                ]
                ctx.motion_state_history = ctx.motion_state_history[
                    : ctx.checkpoint_step + 1
                ]
                ctx.stride_scale_history = ctx.stride_scale_history[
                    : ctx.checkpoint_step + 1
                ]
                ctx.weight_history = ctx.weight_history[: ctx.checkpoint_step + 1]
                ctx.path_log_score_history = ctx.path_log_score_history[
                    : ctx.checkpoint_step + 1
                ]
                ctx.healthy_checkpoint_steps = [
                    ctx.step
                    for ctx.step in ctx.healthy_checkpoint_steps
                    if ctx.step <= ctx.checkpoint_step
                ]
                for _ in range(ctx.replay_depth - 1):
                    ctx.heading_correction_history.append(
                        ctx.recovery.heading_correction.copy()
                    )
                    ctx.heading_drift_history.append(ctx.recovery.heading_drift.copy())
                    ctx.motion_state_history.append(ctx.recovery.motion_state.copy())
                    ctx.stride_scale_history.append(ctx.recovery.stride_scale.copy())
                    ctx.weight_history.append(
                        np.full(ctx.n_particles, 1.0 / ctx.n_particles)
                    )
                    ctx.path_log_score_history.append(
                        ctx.path_log_score_history[ctx.checkpoint_step][
                            ctx.recovery.parent_indices
                        ].copy()
                    )
                if ctx.recorder.diagnostics_enabled:
                    ctx.replay_weights = np.full(ctx.n_particles, 1.0 / ctx.n_particles)
                    for replay_offset in range(ctx.replay_depth - 1):
                        ctx.replay_offset = replay_offset
                        ctx.history_step = ctx.checkpoint_step + ctx.replay_offset + 1
                        ctx.replay_parent_indices = (
                            ctx.recovery.parent_indices
                            if ctx.replay_offset == 0
                            else np.arange(ctx.n_particles, dtype=int)
                        )
                        ctx.replay_motion_state_before = (
                            ctx.motion_state_history[ctx.checkpoint_step]
                            if ctx.replay_offset == 0
                            else ctx.recovery.motion_state
                        )
                        ctx.replay_previous_positions = (
                            ctx.position_history[ctx.checkpoint_step][
                                ctx.recovery.parent_indices
                            ]
                            if ctx.replay_offset == 0
                            else ctx.replay_result.replay_positions[
                                ctx.replay_offset - 1
                            ]
                        )
                        ctx.replay_effective_lengths = np.linalg.norm(
                            ctx.replay_result.replay_positions[ctx.replay_offset]
                            - ctx.replay_previous_positions,
                            axis=1,
                        )
                        ctx.collector_index = (
                            ctx.diagnostics_start_index + ctx.history_step - 1
                        )
                        ctx.recorder.diagnostics[ctx.collector_index] = (
                            _build_step_diagnostics(
                                step_number=ctx.history_step,
                                step_time=ctx.t_at_steps[ctx.history_step - 1],
                                valid_count=ctx.n_particles,
                                n_particles=ctx.n_particles,
                                valid_weight_count=ctx.n_particles,
                                valid_weight_mass=1.0,
                                ess_before_observation=float(ctx.n_particles),
                                ess_after_observation=float(ctx.n_particles),
                                ess_after_resampling=float(ctx.n_particles),
                                weights=ctx.replay_weights,
                                particles=ctx.replay_result.replay_positions[
                                    ctx.replay_offset
                                ],
                                heading_drift=ctx.recovery.heading_drift,
                                heading_correction=ctx.recovery.heading_correction,
                                stride_scale=ctx.recovery.stride_scale,
                                effective_step_lengths=ctx.replay_effective_lengths,
                                parent_indices=ctx.replay_parent_indices,
                                resampled=ctx.replay_offset == 0,
                                motion_state=ctx.recovery.motion_state,
                                motion_state_before=ctx.replay_motion_state_before,
                                motion_evidence=ctx.motion_evidences[
                                    ctx.history_step - 1
                                ],
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
                            )
                        )
                if ctx.recorder.stages_enabled:
                    ctx.replay_weights = np.full(ctx.n_particles, 1.0 / ctx.n_particles)
                    ctx.replay_offsets = _normalize_angle(
                        ctx.recovery.heading_correction + ctx.recovery.heading_drift
                    )
                    for replay_offset in range(ctx.replay_depth - 1):
                        ctx.replay_offset = replay_offset
                        ctx.history_step = ctx.checkpoint_step + ctx.replay_offset + 1
                        ctx.replay_heading = ctx.step_headings[ctx.history_step - 1]
                        ctx.replay_parent_indices = (
                            ctx.recovery.parent_indices
                            if ctx.replay_offset == 0
                            else np.arange(ctx.n_particles, dtype=int)
                        )
                        ctx.replay_before_positions = (
                            ctx.position_history[ctx.checkpoint_step][
                                ctx.recovery.parent_indices
                            ]
                            if ctx.replay_offset == 0
                            else ctx.replay_result.replay_positions[
                                ctx.replay_offset - 1
                            ]
                        )
                        ctx.replay_after_positions = ctx.replay_result.replay_positions[
                            ctx.replay_offset
                        ]
                        ctx.replay_step_lengths = np.linalg.norm(
                            ctx.replay_after_positions - ctx.replay_before_positions,
                            axis=1,
                        )
                        ctx.replay_sensor_heading = ctx.replay_heading.selected_heading
                        ctx.replay_proposed_headings = (
                            np.full(
                                ctx.n_particles, float(ctx.replay_sensor_heading or 0.0)
                            )
                            + ctx.replay_offsets
                        )
                        ctx.collector_index = (
                            ctx.stages_start_index + ctx.history_step - 1
                        )
                        ctx.recorder.stages[ctx.collector_index] = ParticleStepStages(
                            step=ctx.history_step,
                            timestamp_s=ctx.t_at_steps[ctx.history_step - 1],
                            sensor_heading=ctx.replay_sensor_heading,
                            sensor_yaw_delta=ctx.replay_heading.yaw_delta,
                            movement_type=ctx.replay_heading.trajectory_movement_type
                            or ctx.replay_heading.movement_type,
                            deterministic_step_length_m=ctx.step_lengths[
                                ctx.history_step - 1
                            ],
                            before_positions=ctx.replay_before_positions.copy(),
                            before_offsets=ctx.replay_offsets.copy(),
                            before_weights=ctx.replay_weights.copy(),
                            before_motion_state=ctx.recovery.motion_state.copy(),
                            proposed_positions=ctx.replay_after_positions.copy(),
                            proposed_headings=ctx.replay_proposed_headings.copy(),
                            proposed_step_lengths=ctx.replay_step_lengths.copy(),
                            proposed_motion_state=ctx.recovery.motion_state.copy(),
                            valid_transition=np.ones(ctx.n_particles, dtype=bool),
                            posterior_weights=ctx.replay_weights.copy(),
                            ess_before_observation=float(ctx.n_particles),
                            ess_after_observation=float(ctx.n_particles),
                            parent_indices=ctx.replay_parent_indices.copy(),
                            resampled=ctx.replay_offset == 0,
                            recovery_mode="checkpoint_replayed",
                            recovery_candidate_headings=ctx.recovery.candidate_headings.copy()
                            if ctx.replay_offset == 0
                            and ctx.recovery.candidate_headings is not None
                            else None,
                            recovery_candidate_valid=ctx.recovery.candidate_valid.copy()
                            if ctx.replay_offset == 0
                            and ctx.recovery.candidate_valid is not None
                            else None,
                            recovery_selected_index=ctx.recovery.selected_candidate_indices.copy()
                            if ctx.replay_offset == 0
                            and ctx.recovery.selected_candidate_indices is not None
                            else None,
                            after_positions=ctx.replay_after_positions.copy(),
                            after_offsets=ctx.replay_offsets.copy(),
                            after_weights=ctx.replay_weights.copy(),
                            after_motion_state=ctx.recovery.motion_state.copy(),
                        )
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
                        else ctx.position_history[ctx.checkpoint_step][
                            ctx.recovery.parent_indices
                        ]
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
        else:
            ctx.particles = ctx.recovery.particles
            ctx.heading_correction = ctx.recovery.heading_correction
            ctx.heading_drift = ctx.recovery.heading_drift
            ctx.stride_scale = ctx.recovery.stride_scale
            ctx.motion_state = ctx.recovery.motion_state
            ctx.weights = np.full(ctx.n_particles, 1.0 / ctx.n_particles)
            ctx.parent_indices = ctx.recovery.parent_indices
            ctx.effective_step_lengths_for_diagnostics = np.linalg.norm(
                ctx.particles - ctx.particles_before[ctx.parent_indices], axis=1
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
    else:
        if ctx.valid_weight_mass <= 0.0:
            raise RuntimeError("内部エラー: recoveryせず粒子重みが全滅しました")
        ctx.posterior_weights /= ctx.valid_weight_mass
        if ctx.ess_after_observation < ctx.resample_ess_ratio * ctx.n_particles:
            ctx.indices = _systematic_resample(ctx.posterior_weights, ctx.rng)
            ctx.particles = ctx.proposed_particles[ctx.indices]
            ctx.heading_correction = ctx.proposed_correction[ctx.indices]
            ctx.heading_drift = ctx.proposed_drift[ctx.indices]
            ctx.stride_scale = ctx.proposed_stride_scale[ctx.indices]
            ctx.motion_state = ctx.proposed_motion_state[ctx.indices]
            if ctx.effective_heading_rejuvenation_sigma > 0.0:
                ctx.heading_drift = _normalize_angle(
                    ctx.heading_drift
                    + ctx.rng.normal(
                        0, ctx.effective_heading_rejuvenation_sigma, ctx.n_particles
                    )
                )
            ctx.effective_rejuvenation_sigma = (
                max(ctx.stride_scale_rejuvenation_sigma, 0.02)
                if ctx.adaptive_stride_state
                else ctx.stride_scale_rejuvenation_sigma
            )
            if ctx.effective_rejuvenation_sigma > 0.0:
                ctx.stride_scale = np.clip(
                    ctx.stride_scale
                    + ctx.stride_rng.normal(
                        0.0, ctx.effective_rejuvenation_sigma, ctx.n_particles
                    ),
                    ctx.effective_stride_scale_min,
                    ctx.effective_stride_scale_max,
                )
            ctx.weights = np.full(ctx.n_particles, 1.0 / ctx.n_particles)
            ctx.parent_indices = ctx.indices
            ctx.effective_step_lengths_for_diagnostics = ctx.sl[ctx.indices]
            ctx.next_path_log_scores = ctx.candidate_path_log_scores[ctx.indices]
            ctx.resampled = True
        else:
            ctx.particles = ctx.proposed_particles
            ctx.heading_correction = ctx.proposed_correction
            ctx.heading_drift = ctx.proposed_drift
            ctx.stride_scale = ctx.proposed_stride_scale
            ctx.motion_state = ctx.proposed_motion_state
            ctx.weights = ctx.posterior_weights
            ctx.parent_indices = np.arange(ctx.n_particles, dtype=int)
            ctx.next_path_log_scores = ctx.candidate_path_log_scores
