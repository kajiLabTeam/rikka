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

from pathlib import Path

import numpy as np

from ...common.config import (
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    FORWARD_HEADING_SOURCE,
    PF_HEADING_DRIFT_RETENTION,
    PF_LANDMARK_LIKELIHOOD_FLOOR,
    PF_LANDMARK_MODE,
    PF_LANDMARK_RESET_SIGMA_M,
    PF_LANDMARK_SIGMA_M,
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
)
from ...common.lib.models import (
    Landmark,
    LandmarkCorrection,
    LandmarkDetection,
    StepHeading,
    StepMotionEvidence,
    StepMotionPosterior,
)
from .evaluate_map import resolve_map_constraints
from .finalize import finalize
from .initialize import initialize
from .propose import propose
from .record import record
from .recorder import (
    ParticleFilterStepDiagnostics,
    ParticlePathComparison,
    ParticleStepStages,
)
from .resampling import _effective_sample_size
from .setup import setup
from .state import ParticleRuntime

ParticleStepsResult = tuple[
    list[list[float]],
    list[float],
    list[float],
    np.ndarray,
    list[StepHeading],
]


def run_particle_steps(
    gx_mean: float,
    gz_mean: float,
    floormap_path: str | Path = FLOORMAP_PATH,
    origin_px: tuple[int, int] = FLOORMAP_ORIGIN_PX,
    scale: float = FLOORMAP_SCALE,
    n_particles: int = PF_NUM_PARTICLES,
    sigma_init_heading: float = PF_SIGMA_INIT_HEADING,
    sigma_heading: float = PF_SIGMA_HEADING,
    sigma_sl_ratio: float = PF_SIGMA_STEP_LENGTH_RATIO,
    stride_scale_prior_mean: float = PF_STRIDE_SCALE_PRIOR_MEAN,
    stride_scale_init_sigma: float = PF_STRIDE_SCALE_INIT_SIGMA,
    stride_scale_retention: float = PF_STRIDE_SCALE_RETENTION,
    stride_scale_process_sigma: float = PF_STRIDE_SCALE_PROCESS_SIGMA,
    stride_scale_rejuvenation_sigma: float = PF_STRIDE_SCALE_REJUVENATION_SIGMA,
    stride_scale_min: float = PF_STRIDE_SCALE_MIN,
    stride_scale_max: float = PF_STRIDE_SCALE_MAX,
    sidestep_lateral_ratio: float = SIDESTEP_LATERAL_RATIO,
    sidestep_min_lateral_displacement: float = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    motion_heading_correction: str = "auto",
    sidestep_smoothing: str = SIDESTEP_SMOOTHING_METHOD,
    forward_heading_source: str = FORWARD_HEADING_SOURCE,
    sidestep_heading_source: str = "motion",
    sidestep_suspect_mode: str = SIDESTEP_SUSPECT_MODE,
    prepared_step_headings: tuple[StepHeading, ...] | list[StepHeading] | None = None,
    prepared_step_lengths: np.ndarray | list[float] | None = None,
    prepared_step_times: np.ndarray | list[float] | None = None,
    prepared_motion_evidences: (
        tuple[StepMotionEvidence, ...] | list[StepMotionEvidence] | None
    ) = None,
    prepared_particle_motion_headings: (
        tuple[float | None, ...] | list[float | None] | None
    ) = None,
    prepared_motion_posteriors: (
        tuple[StepMotionPosterior, ...] | list[StepMotionPosterior] | None
    ) = None,
    seed: int | None = None,
    heading_drift_retention: float = PF_HEADING_DRIFT_RETENTION,
    resample_ess_ratio: float = PF_RESAMPLE_ESS_RATIO,
    rejuvenation_sigma_heading: float = PF_REJUVENATION_SIGMA_HEADING,
    recovery_valid_ratio: float = PF_RECOVERY_VALID_RATIO,
    recovery_heading_sigma: float = PF_RECOVERY_HEADING_SIGMA,
    recovery_max_attempts: int = PF_RECOVERY_MAX_ATTEMPTS,
    diagnostics_collector: list[ParticleFilterStepDiagnostics] | None = None,
    stage_collector: list[ParticleStepStages] | None = None,
    path_comparison_collector: list[ParticlePathComparison] | None = None,
    preserve_recovery_branches: bool = False,
    motion_predictive_weight_power: float = PF_MOTION_PREDICTIVE_WEIGHT_POWER,
    path_selection: str = PF_PATH_SELECTION,
    landmark_detections: tuple[LandmarkDetection, ...] = (),
    landmarks: tuple[Landmark, ...] = (),
    landmark_mode: str = PF_LANDMARK_MODE,
    landmark_sigma_m: float = PF_LANDMARK_SIGMA_M,
    landmark_likelihood_floor: float = PF_LANDMARK_LIKELIHOOD_FLOOR,
    landmark_reset_sigma_m: float = PF_LANDMARK_RESET_SIGMA_M,
    landmark_events_collector: list[LandmarkCorrection] | None = None,
) -> ParticleStepsResult:
    """準備済み歩列を受け、元と同じ順序でPF段階を実行する。"""
    ctx = ParticleRuntime(
        gx_mean=gx_mean,
        gz_mean=gz_mean,
        floormap_path=floormap_path,
        origin_px=origin_px,
        scale=scale,
        n_particles=n_particles,
        sigma_init_heading=sigma_init_heading,
        sigma_heading=sigma_heading,
        sigma_sl_ratio=sigma_sl_ratio,
        stride_scale_prior_mean=stride_scale_prior_mean,
        stride_scale_init_sigma=stride_scale_init_sigma,
        stride_scale_retention=stride_scale_retention,
        stride_scale_process_sigma=stride_scale_process_sigma,
        stride_scale_rejuvenation_sigma=stride_scale_rejuvenation_sigma,
        stride_scale_min=stride_scale_min,
        stride_scale_max=stride_scale_max,
        sidestep_lateral_ratio=sidestep_lateral_ratio,
        sidestep_min_lateral_displacement=sidestep_min_lateral_displacement,
        motion_heading_correction=motion_heading_correction,
        sidestep_smoothing=sidestep_smoothing,
        forward_heading_source=forward_heading_source,
        sidestep_heading_source=sidestep_heading_source,
        sidestep_suspect_mode=sidestep_suspect_mode,
        prepared_step_headings=prepared_step_headings,
        prepared_step_lengths=prepared_step_lengths,
        prepared_step_times=prepared_step_times,
        prepared_motion_evidences=prepared_motion_evidences,
        prepared_particle_motion_headings=prepared_particle_motion_headings,
        prepared_motion_posteriors=prepared_motion_posteriors,
        seed=seed,
        heading_drift_retention=heading_drift_retention,
        resample_ess_ratio=resample_ess_ratio,
        rejuvenation_sigma_heading=rejuvenation_sigma_heading,
        recovery_valid_ratio=recovery_valid_ratio,
        recovery_heading_sigma=recovery_heading_sigma,
        recovery_max_attempts=recovery_max_attempts,
        diagnostics_collector=diagnostics_collector,
        stage_collector=stage_collector,
        path_comparison_collector=path_comparison_collector,
        preserve_recovery_branches=preserve_recovery_branches,
        motion_predictive_weight_power=motion_predictive_weight_power,
        path_selection=path_selection,
        landmark_detections=landmark_detections,
        landmarks=landmarks,
        landmark_mode=landmark_mode,
        landmark_sigma_m=landmark_sigma_m,
        landmark_likelihood_floor=landmark_likelihood_floor,
        landmark_reset_sigma_m=landmark_reset_sigma_m,
        landmark_events_collector=landmark_events_collector,
    )
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
        resolve_map_constraints(ctx)
        ctx.ess_after_resampling = _effective_sample_size(ctx.weights)
        ctx.position_history.append(ctx.particles.copy())
        record(ctx)
    return finalize(ctx)


run_particle_filter = run_particle_steps
