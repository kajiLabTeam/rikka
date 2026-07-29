"""PDR パイプラインで共有するデータ構造。

役割:
    ステップ区間、検出結果、方位候補、ジャイロ補正結果、確定移動量、前処理済み
    PDR 一式を NamedTuple / dataclass として定義する。
依存元:
    ``config`` から ``PreparedPdrSteps`` の既定値を取得し、NumPy と Pandas の型を
    配列・DataFrame フィールドに使用する。
利用先:
    ``pdr`` 配下のほぼ全モジュール、``particle_filter``、``sensor_plot`` が
    モジュール間の受け渡し形式として使用する。
処理フロー:
    計算処理は持たず、各段階の入力・結果を明示的な不変データとして保持する。
"""

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ..config import (
    FORWARD_HEADING_SOURCE,
    MOTION_ESTIMATION,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    SMOOTHING_MODE,
)


class StepSegment(NamedTuple):
    """1歩区間を表すインデックス範囲。"""

    start_index: int
    end_index: int
    contact_index: int


class StepDetectionResult(NamedTuple):
    """ステップ検出結果。既存互換用ピーク列と論文方式用区間を併せて持つ。"""

    method: str
    peaks: np.ndarray
    segments: tuple[StepSegment, ...]
    threshold: float | None
    polarity: int | None


class StepHeading(NamedTuple):
    """1歩ごとの方位候補と採用結果。角度はすべてラジアン。"""

    step_index: int
    timestamp_s: float
    gyro_heading: float | None
    accel_method1_heading: float | None
    accel_method2_heading: float | None
    selected_heading: float | None
    source: str
    confidence: float
    angle_diff_method1: float | None
    angle_diff_method2: float | None
    segment_start_index: int | None
    segment_end_index: int | None
    peak1_index: int | None
    peak2_index: int | None
    body_heading: float | None
    motion_heading: float | None
    movement_type: str
    forward_displacement: float | None
    lateral_displacement: float | None
    motion_confidence: float
    motion_reject_reason: str | None
    trajectory_movement_type: str | None = None
    step_length_scale: float = 1.0
    yaw_delta: float | None = None
    motion_heading_correction: float = 0.0
    sidestep_lateral_ratio: float = SIDESTEP_LATERAL_RATIO
    sidestep_min_lateral_displacement: float = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M
    forward_heading_source: str = FORWARD_HEADING_SOURCE
    body_motion_angle_diff: float | None = None
    sidestep_evidence_direction: str | None = None
    sidestep_evidence_reason: str | None = None
    sidestep_cluster_id: int | None = None
    device_orientation_mode: str = "normal"
    decoded_motion_mode: str | None = None
    decoded_motion_confidence: float = 0.0
    device_body_offset: float = 0.0
    dynamic_body_heading_confidence: float = 0.0
    body_heading_update_reason: str | None = None


class GyroBiasResult(NamedTuple):
    """ジャイロバイアス推定結果と診断情報。"""

    method: str
    bias_rad_s: float
    calibration_start_s: float | None
    calibration_end_s: float | None
    sample_count: int
    kept_sample_count: int
    raw_mean: float | None
    robust_mean: float | None
    median: float | None
    mad: float | None
    candidate_score: float | None
    gyro_std: float | None
    accel_p95: float | None
    accel_max: float | None
    search_start_s: float | None
    search_end_s: float | None
    fallback_reason: str | None


class StepMotion(NamedTuple):
    """状態別補正後の1歩の移動量。"""

    heading: float
    length: float
    movement_type: str
    length_scale: float


class StepMotionEvidence(NamedTuple):
    """PFが利用する1歩ごとの運動状態観測。"""

    forward_likelihood: float
    sidestep_left_likelihood: float
    sidestep_right_likelihood: float
    turning_likelihood: float
    motion_reliability: float
    calibration_reliability: float


class StepMotionObservation(NamedTuple):
    """端末方位、身体方位候補、移動軸を分離した1歩の観測。

    ``motion_axis_heading`` は方向を確定しない modulo pi の軸であり、
    ``directed_motion_heading`` の正負の向きは後段で再検証できるよう別に保持する。
    """

    step_index: int
    timestamp_s: float
    device_yaw_heading: float | None
    body_heading_candidate: float | None
    directed_motion_heading: float | None
    motion_axis_heading: float | None
    forward_displacement: float | None
    lateral_displacement: float | None
    displacement_norm: float
    yaw_delta: float | None
    motion_confidence: float
    calibration_reliability: float
    raw_movement_type: str
    trajectory_movement_type: str
    device_orientation_mode: str


class StepLengthObservation(NamedTuple):
    """1歩区間から得た歩幅の物理観測と不確かさ。"""

    step_index: int
    nominal_length_m: float
    interval_length_m: float
    step_period_s: float | None
    vertical_amplitude: float
    horizontal_energy: float
    quality: float
    log_length_sigma: float
    fallback_reason: str | None


class StepMotionPosterior(NamedTuple):
    """運動状態、方位、歩幅、端末姿勢ずれの1歩ごとの事後分布。"""

    step_index: int
    forward_probability: float
    sidestep_left_probability: float
    sidestep_right_probability: float
    turning_probability: float
    heading_mean: float
    heading_std: float
    length_mean_m: float
    length_std_m: float
    device_body_offset_mean: float
    device_body_offset_std: float
    selected_mode: str
    source: str


class AdaptivePdrState(NamedTuple):
    """逐次更新できる適応PDRの内部状態。"""

    heading_mean: float | None
    heading_variance: float
    forward_log_scale_mean: float
    forward_log_scale_variance: float
    sidestep_log_scale_mean: float
    sidestep_log_scale_variance: float
    device_body_offset_mean: float
    device_body_offset_variance: float
    mode_probabilities: tuple[float, float, float, float]
    step_count: int


@dataclass(frozen=True)
class AdaptivePdrResult:
    """適応PDRが返す因果推定またはオフライン平滑化結果。"""

    step_headings: list[StepHeading]
    step_lengths: list[float]
    posteriors: tuple[StepMotionPosterior, ...]
    final_state: AdaptivePdrState


@dataclass(frozen=True)
class StepDirectionPosterior:
    """移動軸の2方向候補に対する1歩ごとの事後分布。"""

    step_index: int
    positive_axis_probability: float
    negative_axis_probability: float
    selected_heading: float
    selected_motion_mode: str
    confidence: float
    source: str
    flip_supported: bool


@dataclass(frozen=True)
class PreparedPdrSteps:
    """通常PDRとPFで共用するステップ単位の推定結果。"""

    df_acc: pd.DataFrame
    df_gyro: pd.DataFrame
    step_detection: StepDetectionResult
    trajectory: list[list[float]]
    step_lengths: list[float]
    t_at_steps: list[float]
    step_headings: list[StepHeading]
    gx_mean: float
    gz_mean: float
    weinberg_k: float
    heading_method: str
    motion_heading_correction: str
    sidestep_smoothing: str
    forward_heading_source: str
    sidestep_heading_source: str
    sidestep_suspect_mode: str
    motion_evidences: tuple[StepMotionEvidence, ...] = ()
    motion_observations: tuple[StepMotionObservation, ...] = ()
    length_observations: tuple[StepLengthObservation, ...] = ()
    motion_posteriors: tuple[StepMotionPosterior, ...] = ()
    motion_estimation: str = MOTION_ESTIMATION
    smoothing_mode: str = SMOOTHING_MODE
    direction_posteriors: tuple[StepDirectionPosterior, ...] = ()
    particle_motion_headings: tuple[float | None, ...] = ()


@dataclass(frozen=True)
class ParticleFilterResult:
    """particle filter の代表軌跡と描画・診断用状態。"""

    trajectory: list[list[float]]
    all_particles: list[np.ndarray]
    diagnostics: tuple[Any, ...] = ()
    stages: tuple[Any, ...] = ()
    path_comparisons: tuple[Any, ...] = ()


@dataclass(frozen=True)
class FloorMap:
    """particle filter が利用する地図画像と座標変換設定。"""

    path: str
    origin_px: tuple[int, int]
    scale: float


@dataclass(frozen=True)
class TrajectoryResult:
    """PDR/PF と出力領域を結ぶ解析結果。"""

    trajectory: list[list[float]]
    step_lengths: list[float]
    t_at_steps: list[float]
    step_headings: list[StepHeading]
    prepared: PreparedPdrSteps
    particle: ParticleFilterResult | None = None
