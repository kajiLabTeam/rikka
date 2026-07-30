"""particle filter の段階実行。

役割:
    段階実行を独立した段階として実装する。
依存元:
    common の共有型・設定と particle/lib の部品、ParticleRuntime を利用する。
利用先:
    particle/lib/runner が元の実行順序どおりに呼び出す。
処理フロー:
    互換引数をruntimeへ束ね、初期化後に5段階を歩ごとに呼び、結果を確定する。
"""

from typing import Any

from ...config import (
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    FORWARD_HEADING_SOURCE,
    INITIAL_DIRECTION,
    PF_HEADING_DRIFT_RETENTION,
    PF_MOTION_PREDICTIVE_WEIGHT_POWER,
    PF_NUM_PARTICLES,
    PF_PATH_SELECTION,
    PF_RECOVERY_HEADING_SIGMA,
    PF_RECOVERY_MAX_ATTEMPTS,
    PF_RECOVERY_VALID_RATIO,
    PF_REJUVENATION_SIGMA_HEADING,
    PF_RESAMPLE_ESS_RATIO,
    PF_SIGMA_HEADING,
    PF_SIGMA_INIT_HEADING,
    PF_SIGMA_STEP_LENGTH_RATIO,
    PF_STRIDE_SCALE_INIT_SIGMA,
    PF_STRIDE_SCALE_MAX,
    PF_STRIDE_SCALE_MIN,
    PF_STRIDE_SCALE_PRIOR_MEAN,
    PF_STRIDE_SCALE_PROCESS_SIGMA,
    PF_STRIDE_SCALE_REJUVENATION_SIGMA,
    PF_STRIDE_SCALE_RETENTION,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    SIDESTEP_SMOOTHING_METHOD,
    SIDESTEP_SUSPECT_MODE,
    WEINBERG_K,
)
from .evaluate_map import evaluate_map
from .finalize import finalize
from .initialize import initialize
from .propose import propose
from .record import record
from .resolve import resolve
from .setup import setup
from .state import ParticleRuntime
from .weight_step import weight_step

_PARAMETERS = (
    "peaks",
    "df_gyro",
    "df_acc",
    "gx_mean",
    "gz_mean",
    "floormap_path",
    "origin_px",
    "scale",
    "initial_direction",
    "n_particles",
    "sigma_init_heading",
    "sigma_heading",
    "sigma_sl_ratio",
    "stride_scale_prior_mean",
    "stride_scale_init_sigma",
    "stride_scale_retention",
    "stride_scale_process_sigma",
    "stride_scale_rejuvenation_sigma",
    "stride_scale_min",
    "stride_scale_max",
    "weinberg_k",
    "heading_method",
    "step_segments",
    "sidestep_lateral_ratio",
    "sidestep_min_lateral_displacement",
    "motion_heading_correction",
    "sidestep_smoothing",
    "forward_heading_source",
    "sidestep_heading_source",
    "sidestep_suspect_mode",
    "prepared_step_headings",
    "prepared_step_lengths",
    "prepared_step_times",
    "prepared_motion_evidences",
    "prepared_particle_motion_headings",
    "prepared_motion_posteriors",
    "seed",
    "heading_drift_retention",
    "resample_ess_ratio",
    "rejuvenation_sigma_heading",
    "recovery_valid_ratio",
    "recovery_heading_sigma",
    "recovery_max_attempts",
    "diagnostics_collector",
    "stage_collector",
    "path_comparison_collector",
    "preserve_recovery_branches",
    "motion_predictive_weight_power",
    "path_selection",
)
_DEFAULTS = {
    "floormap_path": FLOORMAP_PATH,
    "origin_px": FLOORMAP_ORIGIN_PX,
    "scale": FLOORMAP_SCALE,
    "initial_direction": INITIAL_DIRECTION,
    "n_particles": PF_NUM_PARTICLES,
    "sigma_init_heading": PF_SIGMA_INIT_HEADING,
    "sigma_heading": PF_SIGMA_HEADING,
    "sigma_sl_ratio": PF_SIGMA_STEP_LENGTH_RATIO,
    "stride_scale_prior_mean": PF_STRIDE_SCALE_PRIOR_MEAN,
    "stride_scale_init_sigma": PF_STRIDE_SCALE_INIT_SIGMA,
    "stride_scale_retention": PF_STRIDE_SCALE_RETENTION,
    "stride_scale_process_sigma": PF_STRIDE_SCALE_PROCESS_SIGMA,
    "stride_scale_rejuvenation_sigma": PF_STRIDE_SCALE_REJUVENATION_SIGMA,
    "stride_scale_min": PF_STRIDE_SCALE_MIN,
    "stride_scale_max": PF_STRIDE_SCALE_MAX,
    "weinberg_k": WEINBERG_K,
    "heading_method": "gyro",
    "step_segments": (),
    "sidestep_lateral_ratio": SIDESTEP_LATERAL_RATIO,
    "sidestep_min_lateral_displacement": SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    "motion_heading_correction": "auto",
    "sidestep_smoothing": SIDESTEP_SMOOTHING_METHOD,
    "forward_heading_source": FORWARD_HEADING_SOURCE,
    "sidestep_heading_source": "motion",
    "sidestep_suspect_mode": SIDESTEP_SUSPECT_MODE,
    "prepared_step_headings": None,
    "prepared_step_lengths": None,
    "prepared_step_times": None,
    "prepared_motion_evidences": None,
    "prepared_particle_motion_headings": None,
    "prepared_motion_posteriors": None,
    "seed": None,
    "heading_drift_retention": PF_HEADING_DRIFT_RETENTION,
    "resample_ess_ratio": PF_RESAMPLE_ESS_RATIO,
    "rejuvenation_sigma_heading": PF_REJUVENATION_SIGMA_HEADING,
    "recovery_valid_ratio": PF_RECOVERY_VALID_RATIO,
    "recovery_heading_sigma": PF_RECOVERY_HEADING_SIGMA,
    "recovery_max_attempts": PF_RECOVERY_MAX_ATTEMPTS,
    "diagnostics_collector": None,
    "stage_collector": None,
    "path_comparison_collector": None,
    "preserve_recovery_branches": False,
    "motion_predictive_weight_power": PF_MOTION_PREDICTIVE_WEIGHT_POWER,
    "path_selection": PF_PATH_SELECTION,
}


def run_particle_steps(*args: Any, **kwargs: Any) -> Any:
    """準備済み歩列を受け、元と同じ順序でPF段階を実行する。"""
    if len(args) > len(_PARAMETERS):
        raise TypeError("位置引数が多すぎます")
    values = dict(zip(_PARAMETERS, args, strict=False))
    duplicated = set(values) & set(kwargs)
    if duplicated:
        name = sorted(duplicated)[0]
        raise TypeError(f"{name} が重複指定されています")
    values.update(kwargs)
    for name in _PARAMETERS:
        if name not in values and name in _DEFAULTS:
            values[name] = _DEFAULTS[name]
    missing = [name for name in _PARAMETERS if name not in values]
    if missing:
        raise TypeError(f"必須引数が不足しています: {', '.join(missing)}")

    ctx = ParticleRuntime()
    ctx.update(values)
    setup(ctx)
    initialize(ctx)
    for step_number, (
        step_heading,
        sl_det,
        step_time,
        motion_evidence,
        particle_heading,
    ) in enumerate(
        zip(
            ctx.stabilized_step_headings,
            ctx.raw_step_lengths,
            ctx.raw_step_times,
            ctx.motion_evidences,
            ctx.particle_motion_headings,
            strict=True,
        ),
        start=1,
    ):
        ctx.step_number = step_number
        ctx.step_heading = step_heading
        ctx.sl_det = sl_det
        ctx.step_time = step_time
        ctx.motion_evidence = motion_evidence
        ctx.particle_heading = particle_heading
        if ctx.particle_heading is None:
            continue
        propose(ctx)
        evaluate_map(ctx)
        weight_step(ctx)
        resolve(ctx)
        record(ctx)
    return finalize(ctx)


run_particle_filter = run_particle_steps
