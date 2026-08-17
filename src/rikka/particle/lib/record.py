"""particle filter のrecord。

役割:
    recordを独立した段階として実装する。
依存元:
    common の共有型・設定と particle/lib の部品、ParticleRuntime を利用する。
利用先:
    particle/lib/runner が元の実行順序どおりに呼び出す。
処理フロー:
    履歴・診断・stageを元と同じ順序で記録する。
"""

import numpy as np

from ...common.lib.models import LandmarkCorrection
from ...particle.lib.proposal import (
    _normalize_angle,
)
from ...particle.lib.recorder import (
    ParticleStepStages,
)
from .diagnostics import _build_step_diagnostics
from .state import ParticleRuntime


def _record_landmark_event(ctx: ParticleRuntime) -> None:
    """現在歩で実際に反映したランドマークの補正前後位置を記録する。"""
    if (
        ctx.landmark_mode == "none"
        or ctx.landmark_detection is None
        or ctx.landmark_xy is None
        or ctx.landmark_before_position is None
    ):
        return
    after = np.average(ctx.particles, axis=0, weights=ctx.weights)
    detection = ctx.landmark_detection
    ctx.landmark_events.append(
        LandmarkCorrection(
            step_index=ctx.step_number - 1,
            timestamp_s=detection.timestamp_s,
            beacon_id=detection.beacon_id,
            rssi_dbm=detection.rssi_dbm,
            before_x=ctx.landmark_before_position[0],
            before_y=ctx.landmark_before_position[1],
            landmark_x=ctx.landmark_xy[0],
            landmark_y=ctx.landmark_xy[1],
            after_x=float(after[0]),
            after_y=float(after[1]),
            applied=ctx.landmark_applied,
        )
    )


def record(ctx: ParticleRuntime) -> None:
    _record_landmark_event(ctx)
    ctx.heading_correction_history.append(ctx.heading_correction.copy())
    ctx.heading_drift_history.append(ctx.heading_drift.copy())
    ctx.motion_state_history.append(ctx.motion_state.copy())
    ctx.stride_scale_history.append(ctx.stride_scale.copy())
    ctx.weight_history.append(ctx.weights.copy())
    ctx.path_log_score_history.append(ctx.next_path_log_scores.copy())
    ctx.step_lengths.append(ctx.sl_det)
    ctx.t_at_steps.append(ctx.step_time)
    ctx.step_headings.append(ctx.step_heading)
    ctx.parent_history.append(ctx.parent_indices)
    ctx.all_particles_list.append(ctx.particles.copy())
    if (
        ctx.recovery_mode == "none"
        and ctx.valid_weight_count > 0
        or (
            ctx.recovery_mode not in {"none", "failed_hold"}
            and ctx.recovery_valid_count > 0
        )
    ):
        ctx.healthy_checkpoint_steps.append(len(ctx.step_lengths))
    if ctx.recorder.diagnostics_enabled:
        ctx.recorder.diagnostics.append(
            _build_step_diagnostics(
                step_number=ctx.step_number,
                step_time=ctx.step_time,
                valid_count=ctx.valid_count,
                n_particles=ctx.n_particles,
                valid_weight_count=ctx.valid_weight_count,
                valid_weight_mass=ctx.valid_weight_mass,
                ess_before_observation=ctx.ess_before_observation,
                ess_after_observation=ctx.ess_after_observation,
                ess_after_resampling=ctx.ess_after_resampling,
                weights=ctx.weights,
                particles=ctx.particles,
                heading_drift=ctx.heading_drift,
                heading_correction=ctx.heading_correction,
                stride_scale=ctx.stride_scale,
                effective_step_lengths=ctx.effective_step_lengths_for_diagnostics,
                parent_indices=ctx.parent_indices,
                resampled=ctx.resampled,
                motion_state=ctx.motion_state,
                motion_state_before=ctx.diagnostic_motion_state_before,
                motion_evidence=ctx.motion_evidence,
                recovery_attempted=ctx.recovery_attempted,
                recovery_mode=ctx.recovery_mode,
                recovery_valid_count=ctx.recovery_valid_count,
                recovery_attempts=ctx.recovery_attempts,
                recovery_heading_delta_deg=ctx.recovery_heading_delta_deg,
                recovery_step_scale=ctx.recovery_step_scale,
                recovery_cost=ctx.recovery_cost,
                recovery_checkpoint_step=ctx.recovery_checkpoint_step,
                recovery_replay_steps=ctx.recovery_replay_steps,
                recovery_candidate_branch_count=ctx.recovery_candidate_branch_count,
                recovery_selected_branch_count=ctx.recovery_selected_branch_count,
                landmark_detection=ctx.landmark_detection,
                landmark_before_position=ctx.landmark_before_position,
                landmark_xy=ctx.landmark_xy,
                landmark_likelihood_mean=ctx.landmark_likelihood_mean,
            )
        )
    if ctx.recorder.stages_enabled:
        assert ctx.posterior_weights_for_stages is not None
        ctx.recorder.stages.append(
            ParticleStepStages(
                step=ctx.step_number,
                timestamp_s=ctx.step_time,
                sensor_heading=ctx.step_heading.selected_heading,
                sensor_yaw_delta=ctx.step_heading.yaw_delta,
                movement_type=ctx.step_heading.trajectory_movement_type
                or ctx.step_heading.movement_type,
                deterministic_step_length_m=ctx.sl_det,
                before_positions=ctx.particles_before.copy(),
                before_offsets=_normalize_angle(
                    ctx.heading_correction_before + ctx.heading_drift_before
                ).copy(),
                before_weights=ctx.weights_before.copy(),
                before_motion_state=ctx.motion_state_before.copy(),
                proposed_positions=ctx.proposed_particles.copy(),
                proposed_headings=_normalize_angle(ctx.theta).copy(),
                proposed_step_lengths=ctx.sl.copy(),
                proposed_motion_state=ctx.proposed_motion_state.copy(),
                valid_transition=ctx.valid_transition.copy(),
                posterior_weights=ctx.posterior_weights_for_stages.copy(),
                ess_before_observation=ctx.ess_before_observation,
                ess_after_observation=ctx.ess_after_observation,
                parent_indices=ctx.parent_indices.copy(),
                resampled=ctx.resampled,
                recovery_mode=ctx.recovery_mode,
                recovery_candidate_headings=ctx.recovery_candidate_headings.copy()
                if ctx.recovery_candidate_headings is not None
                else None,
                recovery_candidate_valid=ctx.recovery_candidate_valid.copy()
                if ctx.recovery_candidate_valid is not None
                else None,
                recovery_selected_index=ctx.recovery_selected_index.copy()
                if ctx.recovery_selected_index is not None
                else None,
                after_positions=ctx.particles.copy(),
                after_offsets=_normalize_angle(
                    ctx.heading_correction + ctx.heading_drift
                ).copy(),
                after_weights=ctx.weights.copy(),
                after_motion_state=ctx.motion_state.copy(),
            )
        )
