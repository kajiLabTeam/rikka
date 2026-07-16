"""パーティクルフィルタの診断データ型。

役割:
    粒子フィルタが1歩ごとに記録する診断値の不変データ型を定義する。
依存元:
    Python標準ライブラリの ``dataclasses`` だけを利用する。
利用先:
    粒子フィルタ実行処理と診断CSV生成処理から使用される。
処理フロー:
    実行処理が各フィールドを構築し、呼び出し元が診断結果として参照・保存する。
"""

from dataclasses import dataclass


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
