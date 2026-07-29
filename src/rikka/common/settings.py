"""検証済みの解析設定オブジェクト。

役割:
    CLI と各 pipeline の引数を領域別の不変データへまとめる。
依存元:
    ``common.config`` の既定値と ``common.lib.validation`` の検証関数を使う。
利用先:
    PDR、particle、plot の各 pipeline と CLI アダプタから使用する。
処理フロー:
    dataclass の構築時に値を一度検証し、検証済み設定を下流へ渡す。
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import (
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    FORWARD_HEADING_SOURCE,
    GYRO_BIAS_METHOD,
    HEADING_METHOD,
    INITIAL_DIRECTION,
    MOTION_ESTIMATION,
    PF_MOTION_PREDICTIVE_WEIGHT_POWER,
    PF_NUM_PARTICLES,
    PF_PATH_SELECTION,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    SIDESTEP_SMOOTHING_METHOD,
    SIDESTEP_SUSPECT_MODE,
    SMOOTHING_MODE,
    STEP_DETECTION_METHOD,
    USER_HEIGHT_M,
)
from .lib.validation import (
    FORWARD_HEADING_SOURCES,
    HEADING_METHODS,
    MOTION_HEADING_CORRECTION_METHODS,
    SIDESTEP_HEADING_SOURCES,
    SIDESTEP_SMOOTHING_METHODS,
    SIDESTEP_SUSPECT_MODES,
    validate_choice,
    validate_non_negative_parameter,
    validate_positive_parameter,
    validate_scale,
)


@dataclass(frozen=True)
class SensorSettings:
    """センサー前処理とジャイロバイアスの設定。"""

    gyro_bias_method: str = GYRO_BIAS_METHOD
    gyro_bias: float | None = None

    def __post_init__(self) -> None:
        if self.gyro_bias is not None and not np.isfinite(self.gyro_bias):
            raise ValueError("gyro_bias は有限な値を指定してください。")


@dataclass(frozen=True)
class StepSettings:
    """歩検出と歩幅推定の設定。"""

    detection_method: str = STEP_DETECTION_METHOD
    height_m: float = USER_HEIGHT_M

    def __post_init__(self) -> None:
        validate_positive_parameter("height_m", self.height_m)


@dataclass(frozen=True)
class HeadingSettings:
    """方位推定の設定。"""

    initial_direction: float = INITIAL_DIRECTION
    method: str = HEADING_METHOD

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_direction):
            raise ValueError("initial_direction は有限な値を指定してください。")
        validate_choice("heading_method", self.method, HEADING_METHODS)


@dataclass(frozen=True)
class MotionStateSettings:
    """運動状態推定と方位選択の設定。"""

    sidestep_lateral_ratio: float = SIDESTEP_LATERAL_RATIO
    sidestep_min_lateral_displacement: float = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M
    motion_heading_correction: str = "auto"
    sidestep_smoothing: str = SIDESTEP_SMOOTHING_METHOD
    forward_heading_source: str = FORWARD_HEADING_SOURCE
    sidestep_heading_source: str = "motion"
    sidestep_suspect_mode: str = SIDESTEP_SUSPECT_MODE
    motion_estimation: str = MOTION_ESTIMATION
    smoothing_mode: str = SMOOTHING_MODE

    def __post_init__(self) -> None:
        validate_positive_parameter(
            "sidestep_lateral_ratio",
            self.sidestep_lateral_ratio,
        )
        validate_non_negative_parameter(
            "sidestep_min_lateral_displacement",
            self.sidestep_min_lateral_displacement,
        )
        validate_choice(
            "motion_heading_correction",
            self.motion_heading_correction,
            MOTION_HEADING_CORRECTION_METHODS,
        )
        validate_choice(
            "sidestep_smoothing",
            self.sidestep_smoothing,
            SIDESTEP_SMOOTHING_METHODS,
        )
        validate_choice(
            "forward_heading_source",
            self.forward_heading_source,
            FORWARD_HEADING_SOURCES,
        )
        validate_choice(
            "sidestep_heading_source",
            self.sidestep_heading_source,
            SIDESTEP_HEADING_SOURCES,
        )
        validate_choice(
            "sidestep_suspect_mode",
            self.sidestep_suspect_mode,
            SIDESTEP_SUSPECT_MODES,
        )


@dataclass(frozen=True)
class ParticleSettings:
    """particle filter とフロアマップの設定。"""

    floormap_path: str | Path = FLOORMAP_PATH
    origin_px: tuple[int, int] = FLOORMAP_ORIGIN_PX
    scale: float = FLOORMAP_SCALE
    seed: int | None = None
    count: int = PF_NUM_PARTICLES
    motion_predictive_weight_power: float = PF_MOTION_PREDICTIVE_WEIGHT_POWER
    path_selection: str = PF_PATH_SELECTION

    def __post_init__(self) -> None:
        validate_scale(self.scale)
        if self.count <= 0:
            raise ValueError("particle_count は正の整数を指定してください。")
        validate_non_negative_parameter(
            "motion_predictive_weight_power",
            self.motion_predictive_weight_power,
        )
        validate_choice(
            "pf_path_selection",
            self.path_selection,
            ("current", "sequence"),
        )


@dataclass(frozen=True)
class PdrSettings:
    """通常PDRで使用する設定一式。"""

    sensor: SensorSettings = field(default_factory=SensorSettings)
    step: StepSettings = field(default_factory=StepSettings)
    heading: HeadingSettings = field(default_factory=HeadingSettings)
    motion_state: MotionStateSettings = field(default_factory=MotionStateSettings)
