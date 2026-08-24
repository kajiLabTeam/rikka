"""PDR CLI コマンドの実行処理。

役割:
    Python API と CLI の引数を検証済み設定へ変換し、各pipelineを呼ぶ。
依存元:
    ``common.settings``、PDR/PF/plot の各 pipeline を使用する。
利用先:
    Click CLI とPythonコードからPDRまたはparticle filterを実行するために使われる。
処理フロー:
    設定構築、通常PDR、任意のPF、コンソール要約、成果物保存の順に実行する。
"""

from dataclasses import replace
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from ..ble.pipeline import run_ble_landmark_detection
from ..common.config import (
    BLE_DATA_PATH,
    BLE_LANDMARK_ENABLED,
    BLE_PDR_CORRECTION_MODE,
    BLE_RSSI_RELEASE_MARGIN_DB,
    BLE_RSSI_RELEASE_STREAK,
    BLE_RSSI_THRESHOLD_DBM,
    BLE_SYNC_WINDOW_S,
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    FORWARD_HEADING_SOURCE,
    GYRO_BIAS_METHOD,
    HEADING_METHOD,
    INITIAL_DIRECTION,
    MOTION_ESTIMATION,
    PF_LANDMARK_MODE,
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
from ..common.lib.floormap import is_walkable_cell
from ..common.lib.models import FloorMap, GyroBiasResult, Landmark, PreparedPdrSteps
from ..common.settings import (
    BleLandmarkSettings,
    HeadingSettings,
    MotionStateSettings,
    OutputSettings,
    ParticleSettings,
    PdrSettings,
    SensorSettings,
    StepSettings,
)
from ..particle.lib.map_constraints import (
    _normalize_floormap_gray,
    _validate_floormap_origin,
)
from ..particle.pipeline import run_particle
from ..pdr.pipeline import run_pdr
from ..plot import pipeline as plot_pipeline


def _load_floormap_gray(floormap_path: str | Path) -> np.ndarray:
    """フロアマップを読み込み、0..255の2次元配列に正規化する。"""
    path = Path(floormap_path)
    if not path.exists():
        raise ValueError(f"フロアマップが存在しません: {path}")
    if not path.is_file():
        raise ValueError(f"フロアマップはファイルを指定してください: {path}")
    try:
        map_raw = mpimg.imread(path)
    except (OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"フロアマップを画像として読み込めません: {path}") from exc

    map_gray = _normalize_floormap_gray(map_raw)
    if map_gray.ndim != 2 or map_gray.size == 0:
        raise ValueError(f"フロアマップ画像の形状が不正です: {path}")
    return map_gray


def _validate_particle_floormap(
    floormap_path: str | Path,
    origin_px: tuple[int, int],
) -> None:
    """PF実行前にフロアマップと歩行可能な起点を検証する。"""
    map_gray = _load_floormap_gray(floormap_path)
    _validate_floormap_origin(map_gray, origin_px)


def _validate_landmark_pixels(
    map_gray: np.ndarray,
    landmarks: tuple[Landmark, ...],
) -> None:
    """ランドマークが地図内の歩行可能画素にあることを検証する。"""
    for landmark in landmarks:
        pixel_x = int(np.floor(landmark.pixel_x + 0.5))
        pixel_y = int(np.floor(landmark.pixel_y + 0.5))
        if not is_walkable_cell(map_gray, pixel_x, pixel_y):
            raise ValueError(
                "ランドマークは歩行可能なマップ内画素を指定してください: "
                f"{landmark.beacon_id} ({landmark.pixel_x}, {landmark.pixel_y})"
            )


def run(
    df_acc: pd.DataFrame | None = None,
    df_gyro: pd.DataFrame | None = None,
    plot: bool = True,
    use_particle_filter: bool = False,
    save_animation: bool | None = None,
    save_step_frames: bool = False,
    step_frames_range: tuple[int, int] | None = None,
    step_frames_arrows: int = PF_STEP_FRAMES_ARROWS,
    step_frames_dpi: int = PF_STEP_FRAMES_DPI,
    save_path_comparison: bool = False,
    floormap_path: str | Path = FLOORMAP_PATH,
    origin_px: tuple[int, int] = FLOORMAP_ORIGIN_PX,
    scale: float = FLOORMAP_SCALE,
    initial_direction: float = INITIAL_DIRECTION,
    height_m: float = USER_HEIGHT_M,
    step_detection_method: str | None = None,
    step_length_method: str = STEP_LENGTH_METHOD,
    heading_method: str | None = None,
    gyro_bias_method: str | None = None,
    gyro_bias: float | None = None,
    sidestep_lateral_ratio: float = SIDESTEP_LATERAL_RATIO,
    sidestep_min_lateral_displacement: float = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    motion_heading_correction: str = "auto",
    sidestep_smoothing: str = SIDESTEP_SMOOTHING_METHOD,
    forward_heading_source: str = FORWARD_HEADING_SOURCE,
    sidestep_heading_source: str = "motion",
    sidestep_suspect_mode: str = SIDESTEP_SUSPECT_MODE,
    particle_seed: int | None = None,
    particle_count: int = PF_NUM_PARTICLES,
    motion_estimation: str = MOTION_ESTIMATION,
    smoothing_mode: str = SMOOTHING_MODE,
    motion_predictive_weight_power: float = PF_MOTION_PREDICTIVE_WEIGHT_POWER,
    pf_path_selection: str = PF_PATH_SELECTION,
    pf_landmark_mode: str = PF_LANDMARK_MODE,
    ble_landmark: bool = BLE_LANDMARK_ENABLED,
    ble_data_path: str | Path = BLE_DATA_PATH,
    ble_rssi_threshold: float = BLE_RSSI_THRESHOLD_DBM,
    ble_correction: str = BLE_PDR_CORRECTION_MODE,
    ble_release_margin: float = BLE_RSSI_RELEASE_MARGIN_DB,
    ble_release_streak: int = BLE_RSSI_RELEASE_STREAK,
    ble_sync_window: float = BLE_SYNC_WINDOW_S,
    ble_landmarks: tuple[Landmark, ...] | None = None,
) -> pd.DataFrame:
    """後方互換引数を設定へ変換し、解析と成果物保存を実行する。"""
    if (save_step_frames or save_path_comparison) and not use_particle_filter:
        raise ValueError("粒子可視化の保存には use_particle_filter=True が必要です。")
    floormap = FloorMap(str(floormap_path), origin_px, scale)
    landmark_settings = (
        BleLandmarkSettings(
            enabled=ble_landmark,
            data_path=ble_data_path,
            rssi_threshold_dbm=ble_rssi_threshold,
            release_margin_db=ble_release_margin,
            release_streak=ble_release_streak,
            sync_window_s=ble_sync_window,
            correction_mode=ble_correction,
        )
        if ble_landmarks is None
        else BleLandmarkSettings(
            enabled=ble_landmark,
            data_path=ble_data_path,
            rssi_threshold_dbm=ble_rssi_threshold,
            release_margin_db=ble_release_margin,
            release_streak=ble_release_streak,
            sync_window_s=ble_sync_window,
            correction_mode=ble_correction,
            landmarks=ble_landmarks,
        )
    )
    pdr_settings = PdrSettings(
        sensor=SensorSettings(
            gyro_bias_method=(
                GYRO_BIAS_METHOD if gyro_bias_method is None else gyro_bias_method
            ),
            gyro_bias=gyro_bias,
        ),
        step=StepSettings(
            detection_method=(
                STEP_DETECTION_METHOD
                if step_detection_method is None
                else step_detection_method
            ),
            length_method=step_length_method,
            height_m=height_m,
        ),
        heading=HeadingSettings(
            initial_direction=initial_direction,
            method=HEADING_METHOD if heading_method is None else heading_method,
        ),
        motion_state=MotionStateSettings(
            sidestep_lateral_ratio=sidestep_lateral_ratio,
            sidestep_min_lateral_displacement=sidestep_min_lateral_displacement,
            motion_heading_correction=motion_heading_correction,
            sidestep_smoothing=sidestep_smoothing,
            forward_heading_source=forward_heading_source,
            sidestep_heading_source=sidestep_heading_source,
            sidestep_suspect_mode=sidestep_suspect_mode,
            motion_estimation=motion_estimation,
            smoothing_mode=smoothing_mode,
        ),
        landmark=(
            replace(landmark_settings, enabled=False)
            if use_particle_filter
            else landmark_settings
        ),
    )
    particle_settings = ParticleSettings(
        floormap_path=floormap_path,
        origin_px=origin_px,
        scale=scale,
        seed=particle_seed,
        count=particle_count,
        motion_predictive_weight_power=motion_predictive_weight_power,
        path_selection=pf_path_selection,
        landmark_mode=pf_landmark_mode,
    )
    output_settings = OutputSettings(
        plot=plot,
        save_animation=plot if save_animation is None else save_animation,
        save_step_frames=save_step_frames,
        step_frames_range=step_frames_range,
        step_frames_arrows=step_frames_arrows,
        step_frames_dpi=step_frames_dpi,
        save_path_comparison=save_path_comparison,
    )
    if use_particle_filter:
        _validate_particle_floormap(floormap_path, origin_px)
        if ble_landmark:
            map_gray = _load_floormap_gray(floormap_path)
            _validate_landmark_pixels(map_gray, landmark_settings.landmarks)
    elif landmark_settings.enabled:
        map_gray = _load_floormap_gray(floormap_path)
        _validate_floormap_origin(map_gray, origin_px)
        _validate_landmark_pixels(map_gray, landmark_settings.landmarks)

    particle_detections = (
        run_ble_landmark_detection(landmark_settings) if use_particle_filter else None
    )
    result = run_pdr(pdr_settings, df_acc, df_gyro, floormap)
    if use_particle_filter:
        result = run_particle(
            result.prepared,
            floormap,
            particle_settings,
            detections=particle_detections,
            landmark_settings=landmark_settings,
        )
    _print_prepared_summary(result.prepared, pdr_settings)
    output_dir = plot_pipeline.create_output_dir()
    dataframe = plot_pipeline.write_outputs(result, output_dir)
    plot_pipeline.render(
        result,
        pdr_settings,
        particle_settings,
        output_settings,
        output_dir,
    )
    return dataframe


def _print_prepared_summary(
    prepared: PreparedPdrSteps,
    settings: PdrSettings,
) -> None:
    """従来のセンサー・推定設定の要約を表示する。"""
    motion = settings.motion_state
    print(
        f"Weinberg K: {prepared.weinberg_k:.3f} (height={settings.step.height_m:.2f} m)"
    )
    print(f"Heading method: {prepared.heading_method}")
    print(
        "Sidestep detection: "
        f"ratio={motion.sidestep_lateral_ratio:.3f} "
        f"min_lateral={motion.sidestep_min_lateral_displacement:.3f} m "
        f"motion_heading_correction={prepared.motion_heading_correction} "
        f"smoothing={prepared.sidestep_smoothing} "
        f"forward_heading_source={prepared.forward_heading_source} "
        f"sidestep_heading_source={prepared.sidestep_heading_source} "
        f"sidestep_suspect_mode={prepared.sidestep_suspect_mode}"
    )
    bias = prepared.df_gyro.attrs.get("gyro_bias_result")
    if isinstance(bias, GyroBiasResult):
        window = (
            "manual"
            if bias.calibration_start_s is None
            else f"{bias.calibration_start_s:.3f}-{bias.calibration_end_s:.3f}s"
        )
        print(
            "Gyro bias: "
            f"method={bias.method} bias={bias.bias_rad_s:.6f} rad/s "
            f"window={window} fallback={bias.fallback_reason}"
        )
    detection = prepared.step_detection
    if detection.threshold is None:
        print(f"Step detection: {detection.method}")
    else:
        print(
            f"Step detection: {detection.method} "
            f"threshold={detection.threshold:.3f} polarity={detection.polarity}"
        )
    dominant = "X軸" if abs(prepared.gx_mean) > abs(prepared.gz_mean) else "Z軸"
    y_flipped = (
        abs(prepared.gx_mean) > abs(prepared.gz_mean) and prepared.gx_mean > 0
    ) or (abs(prepared.gz_mean) >= abs(prepared.gx_mean) and prepared.gz_mean < 0)
    print(
        f"重力主成分: {dominant}  gx={prepared.gx_mean:.2f}, "
        f"gz={prepared.gz_mean:.2f} m/s²"
        f" → Y軸{'反転' if y_flipped else '非反転'}"
    )
