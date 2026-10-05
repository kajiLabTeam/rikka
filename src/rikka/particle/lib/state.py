"""particle filter の型付き実行状態。

役割:
    PFの設定、現在粒子、履歴、1歩ごとの中間値を明示したフィールドで保持する。
依存元:
    commonの共有歩行型、recorderの診断型、NumPy配列を利用する。
利用先:
    particle runnerと各段階関数が、同じ型付き状態を受け渡すために使用する。
処理フロー:
    runnerが検証前の入力フィールドを構築し、各段階が宣言済みの中間フィールドだけを
    更新する。slotsにより未宣言属性への書き込みを実行時にも拒否する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ...common.lib.models import (
    Landmark,
    LandmarkCorrection,
    LandmarkObservation,
    LandmarkRange,
    StepHeading,
    StepMotionEvidence,
    StepMotionPosterior,
)
from .recorder import (
    ParticleFilterStepDiagnostics,
    ParticlePathComparison,
    ParticleRecorder,
    ParticleStepStages,
)

if TYPE_CHECKING:
    from .recovery.checkpoint import _CheckpointReplayResult
    from .recovery.local import _RecoveryResult


@dataclass(slots=True)
class ParticleRuntime:
    """段階関数間で共有する1回のPF実行状態。"""

    gx_mean: float
    gz_mean: float
    floormap_path: str | Path
    origin_px: tuple[int, int]
    scale: float
    n_particles: int
    sigma_init_heading: float
    sigma_heading: float
    sigma_sl_ratio: float
    stride_scale_prior_mean: float
    stride_scale_init_sigma: float
    stride_scale_retention: float
    stride_scale_process_sigma: float
    stride_scale_rejuvenation_sigma: float
    stride_scale_min: float
    stride_scale_max: float
    sidestep_lateral_ratio: float
    sidestep_min_lateral_displacement: float
    motion_heading_correction: str
    sidestep_smoothing: str
    forward_heading_source: str
    sidestep_heading_source: str
    sidestep_suspect_mode: str
    prepared_step_headings: tuple[StepHeading, ...] | list[StepHeading] | None
    prepared_step_lengths: np.ndarray | list[float] | None
    prepared_step_times: np.ndarray | list[float] | None
    prepared_motion_evidences: (
        tuple[StepMotionEvidence, ...] | list[StepMotionEvidence] | None
    )
    prepared_particle_motion_headings: (
        tuple[float | None, ...] | list[float | None] | None
    )
    prepared_motion_posteriors: (
        tuple[StepMotionPosterior, ...] | list[StepMotionPosterior] | None
    )
    seed: int | None
    heading_drift_retention: float
    resample_ess_ratio: float
    rejuvenation_sigma_heading: float
    recovery_valid_ratio: float
    recovery_heading_sigma: float
    recovery_max_attempts: int
    diagnostics_collector: list[ParticleFilterStepDiagnostics] | None
    stage_collector: list[ParticleStepStages] | None
    path_comparison_collector: list[ParticlePathComparison] | None
    preserve_recovery_branches: bool
    motion_predictive_weight_power: float
    path_selection: str
    landmark_detections: tuple[LandmarkObservation, ...]
    landmark_ranging_observations: tuple[LandmarkRange, ...]
    landmarks: tuple[Landmark, ...]
    landmark_mode: str
    landmark_sigma_m: float
    landmark_likelihood_floor: float
    landmark_range_weight_power: float
    landmark_reset_sigma_m: float
    landmark_max_jump_m: float
    landmark_reset_spread_ratio: float
    landmark_reset_min_distance_m: float
    landmark_reset_heading_sigma: float
    landmark_anchor_warn_jump_m: float
    landmark_retrofit: bool
    landmark_events_collector: list[LandmarkCorrection] | None

    adaptive_recovery_scale: bool = field(init=False)
    adaptive_stride_state: bool = field(init=False)
    all_particles: np.ndarray = field(init=False)
    all_particles_list: list[np.ndarray] = field(init=False)
    allow_turn_candidates: bool = field(init=False)
    candidate_path_log_scores: np.ndarray = field(init=False)
    checkpoint_step: int | None = field(init=False)
    collector_index: int = field(init=False)
    completed_steps: int = field(init=False)
    current_modes: list[str] = field(init=False)
    current_path: np.ndarray = field(init=False)
    current_reversals: int = field(init=False)
    current_sources: list[int | None] = field(init=False)
    diagnostic_motion_state_before: np.ndarray = field(init=False)
    diagnostics_start_index: int = field(init=False)
    effective_heading_rejuvenation_sigma: float = field(init=False)
    effective_rejuvenation_sigma: float = field(init=False)
    effective_step_lengths_for_diagnostics: np.ndarray = field(init=False)
    effective_stride_init_sigma: float = field(init=False)
    effective_stride_scale_max: float = field(init=False)
    effective_stride_scale_min: float = field(init=False)
    ess_after_observation: float = field(init=False)
    ess_after_resampling: float = field(init=False)
    ess_before_observation: float = field(init=False)
    fallback_recovery: _RecoveryResult | None = field(init=False)
    heading_correction: np.ndarray = field(init=False)
    heading_correction_before: np.ndarray = field(init=False)
    heading_correction_history: list[np.ndarray] = field(init=False)
    heading_drift: np.ndarray = field(init=False)
    heading_drift_before: np.ndarray = field(init=False)
    heading_drift_history: list[np.ndarray] = field(init=False)
    heading_process_sigma: float = field(init=False)
    healthy_checkpoint_steps: list[int] = field(init=False)
    indices: np.ndarray = field(init=False)
    landmark_before_position: tuple[float, float] | None = field(init=False)
    landmark_applied: bool = field(init=False)
    landmark_by_step: dict[int, LandmarkObservation] = field(init=False)
    landmark_observations_by_step: dict[int, tuple[LandmarkRange, ...]] = field(
        init=False
    )
    landmark_observations: tuple[LandmarkRange, ...] = field(init=False)
    landmark_detection: LandmarkObservation | None = field(init=False)
    landmark_definition: Landmark | None = field(init=False)
    landmark_definitions: dict[str, Landmark] = field(init=False)
    landmark_events: list[LandmarkCorrection] = field(init=False)
    landmark_likelihood: np.ndarray | None = field(init=False)
    landmark_likelihood_mean: float | None = field(init=False)
    landmark_position_spread_rms_m: float | None = field(init=False)
    landmark_anchor_steps: set[int] = field(init=False)
    landmark_meters: dict[str, tuple[float, float]] = field(init=False)
    landmark_xy: tuple[float, float] | None = field(init=False)
    landmark_reset_steps: set[int] = field(init=False)
    map_gray: np.ndarray = field(init=False)
    motion_evidence: StepMotionEvidence = field(init=False)
    motion_evidences: tuple[StepMotionEvidence, ...] | list[StepMotionEvidence] = field(
        init=False
    )
    motion_posterior: StepMotionPosterior | None = field(init=False)
    motion_rng: np.random.Generator = field(init=False)
    motion_state: np.ndarray = field(init=False)
    motion_state_before: np.ndarray = field(init=False)
    motion_state_history: list[np.ndarray] = field(init=False)
    next_path_log_scores: np.ndarray = field(init=False)
    observation_likelihoods: np.ndarray = field(init=False)
    observation_log_likelihood: np.ndarray = field(init=False)
    parent_history: list[np.ndarray] = field(init=False)
    parent_indices: np.ndarray = field(init=False)
    particle_base_headings: np.ndarray = field(init=False)
    particle_heading: float | None = field(init=False)
    particle_motion_headings: tuple[float | None, ...] | list[float | None] = field(
        init=False
    )
    particle_paths: np.ndarray = field(init=False)
    particle_step_lengths: np.ndarray = field(init=False)
    particles: np.ndarray = field(init=False)
    particles_before: np.ndarray = field(init=False)
    path_log_score_history: list[np.ndarray] = field(init=False)
    path_log_scores_before: np.ndarray = field(init=False)
    position_history: list[np.ndarray] = field(init=False)
    posterior_weights: np.ndarray = field(init=False)
    posterior_weights_for_stages: np.ndarray | None = field(init=False)
    proposed_correction: np.ndarray = field(init=False)
    proposed_drift: np.ndarray = field(init=False)
    proposed_motion_state: np.ndarray = field(init=False)
    proposed_particles: np.ndarray = field(init=False)
    proposed_stride_scale: np.ndarray = field(init=False)
    raw_step_length: float = field(init=False)
    raw_step_lengths: np.ndarray | list[float] = field(init=False)
    raw_step_times: np.ndarray | list[float] = field(init=False)
    recorder: ParticleRecorder = field(init=False)
    recording_motion_reliability: float = field(init=False)
    recovery: _RecoveryResult | None = field(init=False)
    recovery_attempted: bool = field(init=False)
    recovery_attempts: int = field(init=False)
    recovery_candidate_branch_count: int = field(init=False)
    recovery_candidate_headings: np.ndarray | None = field(init=False)
    recovery_candidate_valid: np.ndarray | None = field(init=False)
    recovery_checkpoint_step: int | None = field(init=False)
    recovery_cost: float | None = field(init=False)
    recovery_heading_delta_deg: float | None = field(init=False)
    recovery_mode: str = field(init=False)
    recovery_replay_steps: int = field(init=False)
    recovery_selected_branch_count: int = field(init=False)
    recovery_selected_index: np.ndarray | None = field(init=False)
    recovery_step_scale: float | None = field(init=False)
    recovery_valid_count: int = field(init=False)
    relative_length_uncertainty: float = field(init=False)
    replay_angles: np.ndarray = field(init=False)
    replay_depth: int = field(init=False)
    replay_headings: list[StepHeading] = field(init=False)
    replay_lengths: np.ndarray = field(init=False)
    replay_result: _CheckpointReplayResult | None = field(init=False)
    resampled: bool = field(init=False)
    rng: np.random.Generator = field(init=False)
    selected_mode: str = field(init=False)
    selected_path: np.ndarray = field(init=False)
    sensor_headings: np.ndarray = field(init=False)
    sequence_modes: list[str] = field(init=False)
    sequence_path: np.ndarray = field(init=False)
    sequence_reversals: int = field(init=False)
    sequence_sources: list[int | None] = field(init=False)
    sl: np.ndarray = field(init=False)
    sl_det: float = field(init=False)
    stabilized_step_headings: tuple[StepHeading, ...] | list[StepHeading] = field(
        init=False
    )
    stages_start_index: int = field(init=False)
    state_headings: np.ndarray = field(init=False)
    state_length_scales: np.ndarray = field(init=False)
    state_predictive_likelihoods: np.ndarray = field(init=False)
    step_heading: StepHeading = field(init=False)
    step_headings: list[StepHeading] = field(init=False)
    step_lengths: list[float] = field(init=False)
    step_number: int = field(init=False)
    step_stride_process_sigma: float = field(init=False)
    step_time: float = field(init=False)
    stride_observation_likelihood: np.ndarray = field(init=False)
    stride_prior_sigma: float = field(init=False)
    stride_rng: np.random.Generator = field(init=False)
    stride_scale: np.ndarray = field(init=False)
    stride_scale_before: np.ndarray = field(init=False)
    stride_scale_history: list[np.ndarray] = field(init=False)
    t_at_steps: list[float] = field(init=False)
    theta: np.ndarray = field(init=False)
    trajectory_modes: list[str] = field(init=False)
    trajectory_sources: list[int | None] = field(init=False)
    turning_evidence: np.ndarray = field(init=False)
    valid_count: int = field(init=False)
    valid_transition: np.ndarray = field(init=False)
    valid_weight_count: int = field(init=False)
    valid_weight_mask: np.ndarray = field(init=False)
    valid_weight_mass: float = field(init=False)
    weight_history: list[np.ndarray] = field(init=False)
    weights: np.ndarray = field(init=False)
    weights_before: np.ndarray = field(init=False)
