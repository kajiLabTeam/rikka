"""パーティクルフィルタの診断データ型と収集器。

役割:
    粒子フィルタが1歩ごとに記録する診断値の不変データ型を定義する。
依存元:
    Python標準ライブラリの ``dataclasses`` と NumPy を利用する。
利用先:
    粒子フィルタ実行処理と診断CSV生成処理から使用される。
処理フロー:
    実行処理が各フィールドを構築し、呼び出し元が診断結果として参照・保存する。
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ParticleFilterStepDiagnostics:
    """1歩分の粒子健全性と復旧結果。"""

    step: int
    timestamp_s: float
    valid_count: int
    valid_ratio: float
    valid_weight_count: int
    valid_weight_mass_before_normalization: float
    ess_before_observation: float
    ess_after_observation: float
    ess_after_resampling: float
    max_weight: float
    unique_parent_count: int
    unique_position_count: int
    position_spread_rms_m: float
    heading_drift_std_deg: float
    heading_total_std_deg: float
    stride_scale_mean: float
    stride_scale_std: float
    effective_step_length_mean_m: float
    effective_step_length_std_m: float
    forward_state_probability: float
    sidestep_left_state_probability: float
    sidestep_right_state_probability: float
    turning_state_probability: float
    representative_motion_state: str
    motion_state_entropy: float
    motion_state_transition_count: int
    motion_reliability: float
    calibration_reliability: float
    resampled: bool
    recovery_attempted: bool
    recovery_mode: str
    recovery_valid_count: int
    recovery_attempts: int
    recovery_heading_delta_deg: float | None
    recovery_step_scale: float | None
    recovery_cost: float | None
    recovery_checkpoint_step: int | None
    recovery_replay_steps: int
    trajectory_mode: str
    trajectory_source_index: int | None
    recovery_candidate_branch_count: int = 0
    recovery_selected_branch_count: int = 0


@dataclass(frozen=True)
class ParticleStepStages:
    """1歩分の段階別パーティクル状態。可視化専用で、推定には使用しない。"""

    step: int
    timestamp_s: float
    sensor_heading: float | None
    sensor_yaw_delta: float | None
    movement_type: str
    deterministic_step_length_m: float
    before_positions: np.ndarray
    before_offsets: np.ndarray
    before_weights: np.ndarray
    before_motion_state: np.ndarray
    proposed_positions: np.ndarray
    proposed_headings: np.ndarray
    proposed_step_lengths: np.ndarray
    proposed_motion_state: np.ndarray
    valid_transition: np.ndarray
    posterior_weights: np.ndarray
    ess_before_observation: float
    ess_after_observation: float
    parent_indices: np.ndarray
    resampled: bool
    recovery_mode: str
    recovery_candidate_headings: np.ndarray | None
    recovery_candidate_valid: np.ndarray | None
    recovery_selected_index: np.ndarray | None
    after_positions: np.ndarray
    after_offsets: np.ndarray
    after_weights: np.ndarray
    after_motion_state: np.ndarray


@dataclass(frozen=True)
class ParticlePathComparison:
    """代表軌跡候補と個別粒子祖先経路の可視化用スナップショット。"""

    selected_mode: str
    selected_path: np.ndarray
    current_path: np.ndarray
    sequence_path: np.ndarray
    particle_paths: np.ndarray
    current_reversals: int
    sequence_reversals: int


class ParticleRecorder:
    """診断・段階状態・代表経路候補を一元管理する収集器。"""

    def __init__(
        self,
        diagnostics: list[ParticleFilterStepDiagnostics] | None = None,
        stages: list[ParticleStepStages] | None = None,
        paths: list[ParticlePathComparison] | None = None,
    ) -> None:
        self.diagnostics_enabled = diagnostics is not None
        self.stages_enabled = stages is not None
        self.paths_enabled = paths is not None
        self.diagnostics = [] if diagnostics is None else diagnostics
        self.stages = [] if stages is None else stages
        self.paths = [] if paths is None else paths

    def truncate_to(
        self,
        diagnostics_length: int,
        stages_length: int,
    ) -> None:
        """replay 起点以後に収集した診断と段階状態を切り詰める。"""
        del self.diagnostics[diagnostics_length:]
        del self.stages[stages_length:]

    def record_checkpoint_replay(self, ctx: Any) -> None:
        """checkpoint 再生で置換された診断と段階状態を書き戻す。"""
        from .diagnostics import _build_step_diagnostics
        from .proposal import _normalize_angle

        if self.diagnostics_enabled:
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
                    ctx.position_history[ctx.checkpoint_step][
                        ctx.recovery.parent_indices
                    ]
                    if replay_offset == 0
                    else ctx.replay_result.replay_positions[replay_offset - 1]
                )
                replay_effective_lengths = np.linalg.norm(
                    ctx.replay_result.replay_positions[replay_offset]
                    - replay_previous_positions,
                    axis=1,
                )
                collector_index = ctx.diagnostics_start_index + history_step - 1
                self.diagnostics[collector_index] = _build_step_diagnostics(
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
                )

        if self.stages_enabled:
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
                    ctx.position_history[ctx.checkpoint_step][
                        ctx.recovery.parent_indices
                    ]
                    if replay_offset == 0
                    else ctx.replay_result.replay_positions[replay_offset - 1]
                )
                replay_after_positions = ctx.replay_result.replay_positions[
                    replay_offset
                ]
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
                self.stages[collector_index] = ParticleStepStages(
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
                        if replay_offset == 0
                        and ctx.recovery.candidate_valid is not None
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
