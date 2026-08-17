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
    landmark_beacon_id: str | None
    landmark_distance_m: float | None
    landmark_nearest_delta_s: float | None
    landmark_likelihood_mean: float | None
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
