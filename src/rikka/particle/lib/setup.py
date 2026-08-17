"""particle filter の入力検証。

役割:
    入力検証を独立した段階として実装する。
依存元:
    common の共有型・設定と particle/lib の部品、ParticleRuntime を利用する。
利用先:
    particle/lib/runner が元の実行順序どおりに呼び出す。
処理フロー:
    設定値を検証し、乱数系列と有効パラメータを準備する。
"""

import numpy as np

from ...common.lib.validation import (
    PF_LANDMARK_MODES,
    validate_choice,
    validate_forward_heading_source,
    validate_motion_heading_correction,
    validate_non_negative_parameter,
    validate_positive_parameter,
    validate_sidestep_heading_source,
    validate_sidestep_smoothing,
    validate_sidestep_suspect_mode,
)
from .state import ParticleRuntime


def setup(ctx: ParticleRuntime) -> None:
    ctx.sidestep_lateral_ratio = validate_positive_parameter(
        "sidestep_lateral_ratio", ctx.sidestep_lateral_ratio
    )
    ctx.sidestep_min_lateral_displacement = validate_non_negative_parameter(
        "sidestep_min_lateral_displacement", ctx.sidestep_min_lateral_displacement
    )
    validate_motion_heading_correction(ctx.motion_heading_correction)
    validate_sidestep_smoothing(ctx.sidestep_smoothing)
    validate_forward_heading_source(ctx.forward_heading_source)
    validate_sidestep_heading_source(ctx.sidestep_heading_source)
    validate_sidestep_suspect_mode(ctx.sidestep_suspect_mode)
    ctx.scale = validate_positive_parameter("scale", ctx.scale)
    if ctx.n_particles <= 0:
        raise ValueError("n_particles は正の整数を指定してください")
    ctx.sigma_sl_ratio = validate_non_negative_parameter(
        "sigma_sl_ratio", ctx.sigma_sl_ratio
    )
    ctx.stride_scale_prior_mean = validate_positive_parameter(
        "stride_scale_prior_mean", ctx.stride_scale_prior_mean
    )
    ctx.stride_scale_init_sigma = validate_non_negative_parameter(
        "stride_scale_init_sigma", ctx.stride_scale_init_sigma
    )
    ctx.stride_scale_process_sigma = validate_non_negative_parameter(
        "stride_scale_process_sigma", ctx.stride_scale_process_sigma
    )
    ctx.stride_scale_retention = validate_non_negative_parameter(
        "stride_scale_retention", ctx.stride_scale_retention
    )
    if ctx.stride_scale_retention > 1.0:
        raise ValueError("stride_scale_retention は1以下を指定してください")
    ctx.stride_scale_rejuvenation_sigma = validate_non_negative_parameter(
        "stride_scale_rejuvenation_sigma", ctx.stride_scale_rejuvenation_sigma
    )
    ctx.stride_scale_min = validate_positive_parameter(
        "stride_scale_min", ctx.stride_scale_min
    )
    ctx.stride_scale_max = validate_positive_parameter(
        "stride_scale_max", ctx.stride_scale_max
    )
    if ctx.stride_scale_min >= ctx.stride_scale_max:
        raise ValueError("stride_scale_min は stride_scale_max 未満を指定してください")
    if not ctx.stride_scale_min <= ctx.stride_scale_prior_mean <= ctx.stride_scale_max:
        raise ValueError(
            "stride_scale_prior_mean は stride_scale_min 以上 "
            "stride_scale_max 以下を指定してください"
        )
    ctx.heading_drift_retention = validate_non_negative_parameter(
        "heading_drift_retention", ctx.heading_drift_retention
    )
    if ctx.heading_drift_retention > 1.0:
        raise ValueError("heading_drift_retention は1以下を指定してください")
    ctx.resample_ess_ratio = validate_positive_parameter(
        "resample_ess_ratio", ctx.resample_ess_ratio
    )
    if ctx.resample_ess_ratio > 1.0:
        raise ValueError("resample_ess_ratio は1以下を指定してください")
    ctx.rejuvenation_sigma_heading = validate_non_negative_parameter(
        "rejuvenation_sigma_heading", ctx.rejuvenation_sigma_heading
    )
    ctx.recovery_valid_ratio = validate_non_negative_parameter(
        "recovery_valid_ratio", ctx.recovery_valid_ratio
    )
    if ctx.recovery_valid_ratio > 1.0:
        raise ValueError("recovery_valid_ratio は1以下を指定してください")
    ctx.recovery_heading_sigma = validate_positive_parameter(
        "recovery_heading_sigma", ctx.recovery_heading_sigma
    )
    ctx.motion_predictive_weight_power = validate_non_negative_parameter(
        "motion_predictive_weight_power", ctx.motion_predictive_weight_power
    )
    if ctx.path_selection not in {"current", "sequence"}:
        raise ValueError("path_selection は current または sequence を指定してください")
    validate_choice("pf_landmark_mode", ctx.landmark_mode, PF_LANDMARK_MODES)
    ctx.landmark_sigma_m = validate_positive_parameter(
        "landmark_sigma_m", ctx.landmark_sigma_m
    )
    ctx.landmark_reset_sigma_m = validate_positive_parameter(
        "landmark_reset_sigma_m", ctx.landmark_reset_sigma_m
    )
    if not 0.0 <= ctx.landmark_likelihood_floor < 1.0:
        raise ValueError(
            "landmark_likelihood_floor は 0 以上 1 未満を指定してください。"
        )
    if ctx.recovery_max_attempts <= 0:
        raise ValueError("recovery_max_attempts は正の整数を指定してください")
    ctx.adaptive_stride_state = bool(
        ctx.prepared_motion_posteriors is not None
        and any(
            posterior.length_std_m / max(posterior.length_mean_m, 1e-06) >= 0.18
            for posterior in ctx.prepared_motion_posteriors
        )
    )
    ctx.effective_stride_scale_min = (
        min(ctx.stride_scale_min, 0.6)
        if ctx.adaptive_stride_state
        else ctx.stride_scale_min
    )
    ctx.effective_stride_scale_max = (
        max(ctx.stride_scale_max, 1.5)
        if ctx.adaptive_stride_state
        else ctx.stride_scale_max
    )
    ctx.effective_stride_init_sigma = (
        max(ctx.stride_scale_init_sigma, 0.12)
        if ctx.adaptive_stride_state
        else ctx.stride_scale_init_sigma
    )
    ctx.rng = np.random.default_rng(ctx.seed)
    ctx.stride_rng = np.random.default_rng(
        None if ctx.seed is None else np.random.SeedSequence([ctx.seed, 1398035017])
    )
    ctx.motion_rng = np.random.default_rng(
        None if ctx.seed is None else np.random.SeedSequence([ctx.seed, 1297044558])
    )
