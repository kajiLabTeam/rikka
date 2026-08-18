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
    BLE_ANCHOR_WARN_JUMP_M,
    BLE_DATA_PATH,
    BLE_LANDMARK_ANCHORS,
    BLE_LANDMARK_ENABLED,
    BLE_LANDMARKS_PX,
    BLE_RSSI_RELEASE_MARGIN_DB,
    BLE_RSSI_RELEASE_STREAK,
    BLE_RSSI_THRESHOLD_DBM,
    BLE_SAMPLE_BASE_RSSI_DBM,
    BLE_SAMPLE_INTERVAL_S,
    BLE_SAMPLE_MIN_RSSI_DBM,
    BLE_SAMPLE_MODE,
    BLE_SAMPLE_NOISE_SIGMA_DB,
    BLE_SAMPLE_PEAK_RSSI_DBM,
    BLE_SAMPLE_PEAK_TIMES_S,
    BLE_SAMPLE_SEED,
    BLE_SAMPLE_SIGMA_M,
    BLE_SAMPLE_SIGMA_S,
    BLE_SYNC_WINDOW_S,
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    FORWARD_HEADING_SOURCE,
    GYRO_BIAS_METHOD,
    HEADING_METHOD,
    INITIAL_DIRECTION,
    MOTION_ESTIMATION,
    PF_LANDMARK_LIKELIHOOD_FLOOR,
    PF_LANDMARK_MAX_JUMP_M,
    PF_LANDMARK_MODE,
    PF_LANDMARK_RESET_HEADING_SIGMA,
    PF_LANDMARK_RESET_MIN_DISTANCE_M,
    PF_LANDMARK_RESET_SIGMA_M,
    PF_LANDMARK_RESET_SPREAD_RATIO,
    PF_LANDMARK_SIGMA_M,
    PF_MOTION_PREDICTIVE_WEIGHT_POWER,
    PF_NUM_PARTICLES,
    PF_PATH_SELECTION,
    PF_STEP_FRAMES_ARROWS,
    PF_STEP_FRAMES_DPI,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    SIDESTEP_SMOOTHING_METHOD,
    SIDESTEP_SUSPECT_MODE,
    SMOOTHING_MODE,
    STEP_DETECTION_METHOD,
    STEP_LENGTH_METHOD,
    USER_HEIGHT_M,
)
from .lib.models import Landmark
from .lib.validation import (
    BLE_SAMPLE_MODES,
    FORWARD_HEADING_SOURCES,
    GYRO_BIAS_METHODS,
    HEADING_METHODS,
    MOTION_ESTIMATION_METHODS,
    MOTION_HEADING_CORRECTION_METHODS,
    PF_LANDMARK_MODES,
    PF_PATH_SELECTION_METHODS,
    SIDESTEP_HEADING_SOURCES,
    SIDESTEP_SMOOTHING_METHODS,
    SIDESTEP_SUSPECT_MODES,
    SMOOTHING_MODES,
    STEP_DETECTION_METHODS,
    STEP_LENGTH_METHODS,
    validate_choice,
    validate_landmarks,
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
        validate_choice("gyro_bias_method", self.gyro_bias_method, GYRO_BIAS_METHODS)
        if self.gyro_bias is not None and not np.isfinite(self.gyro_bias):
            raise ValueError("gyro_bias は有限な値を指定してください。")


@dataclass(frozen=True)
class StepSettings:
    """歩検出と歩幅推定の設定。"""

    detection_method: str = STEP_DETECTION_METHOD
    length_method: str = STEP_LENGTH_METHOD
    height_m: float = USER_HEIGHT_M

    def __post_init__(self) -> None:
        validate_choice(
            "step_detection_method",
            self.detection_method,
            STEP_DETECTION_METHODS,
        )
        validate_choice("step_length_method", self.length_method, STEP_LENGTH_METHODS)
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
        validate_choice(
            "motion_estimation",
            self.motion_estimation,
            MOTION_ESTIMATION_METHODS,
        )
        validate_choice(
            "smoothing_mode",
            self.smoothing_mode,
            SMOOTHING_MODES,
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
    landmark_mode: str = PF_LANDMARK_MODE
    landmark_sigma_m: float = PF_LANDMARK_SIGMA_M
    landmark_likelihood_floor: float = PF_LANDMARK_LIKELIHOOD_FLOOR
    landmark_reset_sigma_m: float = PF_LANDMARK_RESET_SIGMA_M
    landmark_max_jump_m: float = PF_LANDMARK_MAX_JUMP_M
    landmark_reset_spread_ratio: float = PF_LANDMARK_RESET_SPREAD_RATIO
    landmark_reset_min_distance_m: float = PF_LANDMARK_RESET_MIN_DISTANCE_M
    landmark_reset_heading_sigma: float = PF_LANDMARK_RESET_HEADING_SIGMA
    landmark_anchor_warn_jump_m: float = BLE_ANCHOR_WARN_JUMP_M

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
            PF_PATH_SELECTION_METHODS,
        )
        validate_choice("pf_landmark_mode", self.landmark_mode, PF_LANDMARK_MODES)
        validate_positive_parameter("landmark_sigma_m", self.landmark_sigma_m)
        validate_positive_parameter(
            "landmark_reset_sigma_m",
            self.landmark_reset_sigma_m,
        )
        validate_positive_parameter("landmark_max_jump_m", self.landmark_max_jump_m)
        validate_positive_parameter(
            "landmark_reset_spread_ratio",
            self.landmark_reset_spread_ratio,
        )
        validate_non_negative_parameter(
            "landmark_reset_min_distance_m",
            self.landmark_reset_min_distance_m,
        )
        validate_non_negative_parameter(
            "landmark_reset_heading_sigma",
            self.landmark_reset_heading_sigma,
        )
        validate_positive_parameter(
            "landmark_anchor_warn_jump_m",
            self.landmark_anchor_warn_jump_m,
        )
        if not 0.0 <= self.landmark_likelihood_floor < 1.0:
            raise ValueError(
                "landmark_likelihood_floor は 0 以上 1 未満を指定してください。"
            )


def _build_landmarks_with_anchors(
    landmark_rows: tuple[tuple[str, float, float], ...],
    anchor_rows: tuple[tuple[str, float, float | None, float, bool], ...],
) -> tuple[Landmark, ...]:
    """座標定義へ beacon_id が一致する確定情報を結合する。"""
    validated = validate_landmarks(landmark_rows)
    known_ids = {beacon_id for beacon_id, _, _ in validated}
    anchors: dict[str, tuple[float, float | None, float, bool]] = {}
    for (
        beacon_id,
        position_sigma_m,
        heading_deg,
        heading_sigma_deg,
        bidirectional,
    ) in anchor_rows:
        if beacon_id not in known_ids:
            raise ValueError(
                "BLE_LANDMARK_ANCHORS に BLE_LANDMARKS_PX 未定義の "
                f"beacon_id があります: {beacon_id}"
            )
        if beacon_id in anchors:
            raise ValueError(
                f"BLE_LANDMARK_ANCHORS の beacon_id が重複しています: {beacon_id}"
            )
        anchors[beacon_id] = (
            position_sigma_m,
            heading_deg,
            heading_sigma_deg,
            bidirectional,
        )

    landmarks = []
    for beacon_id, pixel_x, pixel_y in validated:
        anchor = anchors.get(beacon_id)
        if anchor is None:
            landmarks.append(Landmark(beacon_id, pixel_x, pixel_y))
            continue
        landmarks.append(
            Landmark(
                beacon_id,
                pixel_x,
                pixel_y,
                position_sigma_m=anchor[0],
                heading_deg=anchor[1],
                heading_sigma_deg=anchor[2],
                heading_bidirectional=anchor[3],
            )
        )
    return tuple(landmarks)


@dataclass(frozen=True)
class BleLandmarkSettings:
    """BLE ランドマーク補正の設定。"""

    enabled: bool = BLE_LANDMARK_ENABLED
    data_path: str | Path = BLE_DATA_PATH
    rssi_threshold_dbm: float = BLE_RSSI_THRESHOLD_DBM
    release_margin_db: float = BLE_RSSI_RELEASE_MARGIN_DB
    release_streak: int = BLE_RSSI_RELEASE_STREAK
    sync_window_s: float = BLE_SYNC_WINDOW_S
    landmarks: tuple[Landmark, ...] = field(
        default_factory=lambda: _build_landmarks_with_anchors(
            BLE_LANDMARKS_PX,
            BLE_LANDMARK_ANCHORS,
        )
    )

    def __post_init__(self) -> None:
        if not np.isfinite(self.rssi_threshold_dbm):
            raise ValueError("rssi_threshold_dbm は有限な値を指定してください。")
        validate_non_negative_parameter("release_margin_db", self.release_margin_db)
        if (
            not isinstance(self.release_streak, int)
            or isinstance(self.release_streak, bool)
            or self.release_streak < 1
        ):
            raise ValueError("release_streak は 1 以上の整数を指定してください。")
        validate_non_negative_parameter("sync_window_s", self.sync_window_s)
        validate_landmarks(
            tuple(
                (item.beacon_id, item.pixel_x, item.pixel_y) for item in self.landmarks
            )
        )
        for item in self.landmarks:
            if item.position_sigma_m is not None:
                validate_positive_parameter(
                    f"{item.beacon_id}.position_sigma_m",
                    item.position_sigma_m,
                )
            if item.heading_deg is not None and not np.isfinite(item.heading_deg):
                raise ValueError(
                    f"{item.beacon_id}.heading_deg は有限な値を指定してください。"
                )
            if item.heading_deg is not None and item.position_sigma_m is None:
                raise ValueError(
                    f"{item.beacon_id}.heading_deg には position_sigma_m が必要です。"
                )
            validate_non_negative_parameter(
                f"{item.beacon_id}.heading_sigma_deg",
                item.heading_sigma_deg,
            )
        if self.enabled and not self.landmarks:
            raise ValueError(
                "BLE ランドマーク補正を有効にする場合は landmarks を 1 件以上"
                "指定してください。"
            )

    def landmark_map(self) -> dict[str, Landmark]:
        """beacon_id からランドマークを引く辞書を返す。"""
        return {item.beacon_id: item for item in self.landmarks}


@dataclass(frozen=True)
class BleSampleSettings:
    """サンプル BLE RSSI 生成の条件。本番のランドマーク測位では使用しない。"""

    mode: str = BLE_SAMPLE_MODE
    peak_times_s: tuple[tuple[str, float], ...] = BLE_SAMPLE_PEAK_TIMES_S
    interval_s: float = BLE_SAMPLE_INTERVAL_S
    base_rssi_dbm: float = BLE_SAMPLE_BASE_RSSI_DBM
    peak_rssi_dbm: float = BLE_SAMPLE_PEAK_RSSI_DBM
    sigma_s: float = BLE_SAMPLE_SIGMA_S
    sigma_m: float = BLE_SAMPLE_SIGMA_M
    noise_sigma_db: float = BLE_SAMPLE_NOISE_SIGMA_DB
    min_rssi_dbm: float = BLE_SAMPLE_MIN_RSSI_DBM
    seed: int = BLE_SAMPLE_SEED

    def __post_init__(self) -> None:
        validate_choice("ble_sample_mode", self.mode, BLE_SAMPLE_MODES)
        validate_positive_parameter("interval_s", self.interval_s)
        validate_positive_parameter("sigma_s", self.sigma_s)
        validate_positive_parameter("sigma_m", self.sigma_m)
        validate_non_negative_parameter("noise_sigma_db", self.noise_sigma_db)
        if self.mode == "time" and not self.peak_times_s:
            raise ValueError("peak_times_s は 1 件以上指定してください。")
        seen: set[str] = set()
        for beacon_id, peak_time in self.peak_times_s:
            if beacon_id in seen:
                raise ValueError(f"beacon_id が重複しています: {beacon_id}")
            seen.add(beacon_id)
            if not np.isfinite(peak_time):
                raise ValueError("peak_times_s の時刻は有限な値を指定してください。")
        if self.peak_rssi_dbm <= self.base_rssi_dbm:
            raise ValueError(
                "peak_rssi_dbm は base_rssi_dbm より大きい値を指定してください。"
            )


@dataclass(frozen=True)
class OutputSettings:
    """CSV・図・particle 診断成果物の出力設定。"""

    plot: bool = True
    save_animation: bool = True
    save_step_frames: bool = False
    step_frames_range: tuple[int, int] | None = None
    step_frames_arrows: int = PF_STEP_FRAMES_ARROWS
    step_frames_dpi: int = PF_STEP_FRAMES_DPI
    save_path_comparison: bool = False

    def __post_init__(self) -> None:
        if self.step_frames_range is not None:
            first_step, last_step = self.step_frames_range
            if first_step < 1 or first_step > last_step:
                raise ValueError(
                    "step_frames_range は 1 <= A <= B を満たす必要があります。"
                )
        if self.step_frames_arrows < 0:
            raise ValueError("step_frames_arrows は0以上を指定してください。")
        if self.step_frames_dpi <= 0:
            raise ValueError("step_frames_dpi は正の整数を指定してください。")


@dataclass(frozen=True)
class PdrSettings:
    """通常PDRで使用する設定一式。"""

    sensor: SensorSettings = field(default_factory=SensorSettings)
    step: StepSettings = field(default_factory=StepSettings)
    heading: HeadingSettings = field(default_factory=HeadingSettings)
    motion_state: MotionStateSettings = field(default_factory=MotionStateSettings)
    landmark: BleLandmarkSettings = field(default_factory=BleLandmarkSettings)
