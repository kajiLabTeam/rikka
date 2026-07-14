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
from typing import NamedTuple

import numpy as np
import pandas as pd

from ...config import (
    FORWARD_HEADING_SOURCE,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
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
