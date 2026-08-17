"""particle filter の提案・観測重み計算。

役割:
    1歩の粒子状態を提案し、地図遷移と観測尤度から事後重みを計算する。
依存元:
    common の共有型・設定と particle/lib の部品、ParticleRuntime を利用する。
利用先:
    particle/lib/runner が元の実行順序どおりに呼び出す。
処理フロー:
    歩の観測と乱数を使い提案状態を生成し、遷移の歩行可否、運動状態尤度、
    歩幅尤度を統合してrecovery判定まで行う。
"""

import numpy as np

from ...common.config import (
    SIDESTEP_LENGTH_SCALE,
    TURNING_LENGTH_SCALE,
)
from ...particle.lib.landmark import landmark_likelihood
from ...particle.lib.map_constraints import (
    _evaluate_particle_transitions,
)
from ...particle.lib.proposal import (
    _motion_state_headings,
    _motion_state_likelihoods,
    _normalize_angle,
    _sample_motion_states,
)
from ...particle.lib.resampling import (
    _effective_sample_size,
)
from ...particle.lib.weighting import weight
from .state import ParticleRuntime


def _resolve_landmark_observation(ctx: ParticleRuntime) -> None:
    """現在歩のランドマークと、observation方式の尤度・診断値を解決する。"""
    ctx.landmark_detection = None
    ctx.landmark_xy = None
    ctx.landmark_likelihood = None
    ctx.landmark_likelihood_mean = None
    ctx.landmark_position_spread_rms_m = None
    ctx.landmark_before_position = None
    ctx.landmark_applied = False
    if ctx.landmark_mode == "none":
        return
    ctx.landmark_detection = ctx.landmark_by_step.get(ctx.step_number)
    if ctx.landmark_detection is None:
        return
    ctx.landmark_xy = ctx.landmark_meters[ctx.landmark_detection.beacon_id]
    base_weights = weight(
        ctx.weights_before,
        ctx.valid_transition,
        ctx.stride_observation_likelihood,
        ctx.state_predictive_likelihoods,
        ctx.motion_predictive_weight_power,
    )
    base_mass = float(base_weights.sum())
    normalized = base_weights / base_mass if base_mass > 0.0 else ctx.weights_before
    center = np.average(ctx.proposed_particles, axis=0, weights=normalized)
    ctx.landmark_before_position = (float(center[0]), float(center[1]))
    ctx.landmark_position_spread_rms_m = float(
        np.sqrt(
            np.sum(
                normalized * np.sum(np.square(ctx.proposed_particles - center), axis=1)
            )
        )
    )
    if ctx.landmark_mode not in {"observation", "hybrid"}:
        return
    ctx.landmark_likelihood = landmark_likelihood(
        ctx.proposed_particles,
        ctx.landmark_xy,
        ctx.landmark_sigma_m,
        ctx.landmark_likelihood_floor,
    )
    ctx.landmark_likelihood_mean = float(np.sum(normalized * ctx.landmark_likelihood))
    ctx.landmark_applied = True


def propose(ctx: ParticleRuntime) -> None:
    assert ctx.particle_heading is not None
    ctx.motion_posterior = (
        ctx.prepared_motion_posteriors[ctx.step_number - 1]
        if ctx.prepared_motion_posteriors is not None
        else None
    )
    ctx.step_heading = ctx.step_heading._replace(
        selected_heading=ctx.particle_heading, source="particle_evidence_motion"
    )
    ctx.particles_before = ctx.particles.copy()
    ctx.heading_correction_before = ctx.heading_correction.copy()
    ctx.heading_drift_before = ctx.heading_drift.copy()
    ctx.stride_scale_before = ctx.stride_scale.copy()
    ctx.motion_state_before = ctx.motion_state.copy()
    ctx.diagnostic_motion_state_before = ctx.motion_state_before
    ctx.weights_before = ctx.weights.copy()
    ctx.path_log_scores_before = ctx.path_log_score_history[-1].copy()
    ctx.ess_before_observation = _effective_sample_size(ctx.weights_before)
    ctx.proposed_correction = ctx.heading_correction_before
    ctx.heading_process_sigma = (
        max(ctx.sigma_heading, 0.15 * ctx.motion_posterior.heading_std)
        if ctx.motion_posterior is not None
        else ctx.sigma_heading
    )
    ctx.proposed_drift = _normalize_angle(
        ctx.heading_drift_retention * ctx.heading_drift_before
        + ctx.rng.normal(0, ctx.heading_process_sigma, ctx.n_particles)
    )
    ctx.observation_likelihoods = _motion_state_likelihoods(ctx.motion_evidence)
    ctx.proposed_motion_state, ctx.state_predictive_likelihoods = _sample_motion_states(
        ctx.motion_state_before, ctx.observation_likelihoods, ctx.motion_rng
    )
    ctx.state_headings = _motion_state_headings(
        ctx.step_heading, ctx.motion_evidence, ctx.particle_heading
    )
    ctx.particle_base_headings = ctx.state_headings[ctx.proposed_motion_state]
    ctx.theta = (
        ctx.particle_base_headings + ctx.proposed_correction + ctx.proposed_drift
    )
    ctx.relative_length_uncertainty = (
        ctx.motion_posterior.length_std_m
        / max(ctx.motion_posterior.length_mean_m, 1e-06)
        if ctx.motion_posterior is not None
        else 0.0
    )
    ctx.adaptive_recovery_scale = (
        ctx.motion_posterior is not None and ctx.relative_length_uncertainty >= 0.18
    )
    ctx.step_stride_process_sigma = (
        max(
            ctx.stride_scale_process_sigma,
            min(0.05, 0.12 * ctx.relative_length_uncertainty),
        )
        if ctx.motion_posterior is not None
        else ctx.stride_scale_process_sigma
    )
    ctx.proposed_stride_scale = np.clip(
        ctx.stride_scale_prior_mean
        + ctx.stride_scale_retention
        * (ctx.stride_scale_before - ctx.stride_scale_prior_mean)
        + ctx.stride_rng.normal(0.0, ctx.step_stride_process_sigma, ctx.n_particles),
        ctx.effective_stride_scale_min,
        ctx.effective_stride_scale_max,
    )
    if (
        ctx.motion_evidence.calibration_reliability >= 0.85
        or ctx.step_heading.sidestep_cluster_id is None
    ):
        ctx.particle_step_lengths = np.full(ctx.n_particles, ctx.sl_det, dtype=float)
    else:
        ctx.raw_step_length = ctx.sl_det / max(
            ctx.step_heading.step_length_scale, 1e-12
        )
        ctx.state_length_scales = np.asarray(
            [1.0, SIDESTEP_LENGTH_SCALE, SIDESTEP_LENGTH_SCALE, TURNING_LENGTH_SCALE]
        )
        ctx.particle_step_lengths = (
            ctx.raw_step_length * ctx.state_length_scales[ctx.proposed_motion_state]
        )
    ctx.sl = np.clip(
        ctx.particle_step_lengths
        * ctx.proposed_stride_scale
        * (1 + ctx.rng.normal(0, ctx.sigma_sl_ratio, ctx.n_particles)),
        0,
        None,
    )
    ctx.proposed_particles = ctx.particles_before.copy()
    ctx.proposed_particles[:, 0] += ctx.sl * np.cos(ctx.theta)
    ctx.proposed_particles[:, 1] += ctx.sl * np.sin(ctx.theta)
    ctx.effective_step_lengths_for_diagnostics = ctx.sl.copy()
    ctx.valid_transition = _evaluate_particle_transitions(
        ctx.particles_before,
        ctx.proposed_particles,
        ctx.map_gray,
        ctx.gx_mean,
        ctx.gz_mean,
        ctx.origin_px,
        ctx.scale,
    )
    ctx.valid_count = int(np.count_nonzero(ctx.valid_transition))
    ctx.valid_weight_mask = ctx.valid_transition & (ctx.weights_before > 0.0)
    ctx.valid_weight_count = int(np.count_nonzero(ctx.valid_weight_mask))
    if ctx.motion_posterior is None:
        ctx.stride_observation_likelihood = np.ones(ctx.n_particles, dtype=float)
    else:
        ctx.stride_prior_sigma = float(
            np.clip(1.5 * ctx.relative_length_uncertainty, 0.1, 0.3)
        )
        ctx.stride_observation_likelihood = np.exp(
            -0.5
            * np.square(
                (ctx.proposed_stride_scale - ctx.stride_scale_prior_mean)
                / ctx.stride_prior_sigma
            )
        )
    _resolve_landmark_observation(ctx)
    ctx.posterior_weights = weight(
        ctx.weights_before,
        ctx.valid_transition,
        ctx.stride_observation_likelihood,
        ctx.state_predictive_likelihoods,
        ctx.motion_predictive_weight_power,
        ctx.landmark_likelihood,
    )
    ctx.posterior_weights_for_stages = (
        ctx.posterior_weights.copy() if ctx.recorder.stages_enabled else None
    )
    with np.errstate(divide="ignore"):
        ctx.observation_log_likelihood = np.log(ctx.stride_observation_likelihood)
        if ctx.motion_predictive_weight_power > 0.0:
            ctx.observation_log_likelihood += (
                ctx.motion_predictive_weight_power
                * np.log(ctx.state_predictive_likelihoods)
            )
        if ctx.landmark_likelihood is not None:
            ctx.observation_log_likelihood += np.log(ctx.landmark_likelihood)
    ctx.candidate_path_log_scores = np.where(
        ctx.valid_transition,
        ctx.path_log_scores_before + ctx.observation_log_likelihood,
        -np.inf,
    )
    ctx.valid_weight_mass = float(ctx.posterior_weights.sum())
    ctx.ess_after_observation = (
        _effective_sample_size(ctx.posterior_weights / ctx.valid_weight_mass)
        if ctx.valid_weight_mass > 0.0
        else 0.0
    )
    ctx.recovery_attempted = (
        ctx.valid_weight_mass <= 0.0
        or ctx.valid_weight_count / ctx.n_particles < ctx.recovery_valid_ratio
    )
    ctx.recovery_mode = "none"
    ctx.recovery_valid_count = 0
    ctx.recovery_attempts = 0
    ctx.recovery_heading_delta_deg = None
