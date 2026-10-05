"""通常PDRとPFで共有する歩列を準備する。

役割:
    センサー前処理からステップ検出、運動推定までを実行し ``PreparedPdrSteps`` を作る。
依存元:
    共通設定・モデル、PDR の検出・軌跡・融合部品を利用する。
利用先:
    PDR pipeline と検証コードの ``prepare_pdr_steps`` API から使用される。
処理フロー:
    設定検証、センサー処理、歩列生成、運動推定、PF共有値構築の順に処理する。
"""

import pandas as pd

from ...common.config import (
    FORWARD_HEADING_SOURCE,
    GYRO_BIAS_METHOD,
    HEADING_METHOD,
    INITIAL_DIRECTION,
    MOTION_ESTIMATION,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    SIDESTEP_SMOOTHING_METHOD,
    SIDESTEP_SUSPECT_MODE,
    SMOOTHING_MODE,
    STEP_LENGTH_METHOD,
    USER_HEIGHT_M,
    compute_weinberg_k,
)
from ...common.lib.integrate import integrate_steps
from ...common.lib.models import PreparedPdrSteps
from ...common.lib.sensors import process_sensor_data
from ...common.settings import (
    HeadingSettings,
    MotionStateSettings,
    PdrSettings,
    SensorSettings,
    StepSettings,
)
from .fusion.protocol import MOTION_ESTIMATORS
from .motion_state.evidence import (
    build_particle_motion_headings,
    build_step_motion_evidences,
    build_step_motion_observations,
)
from .step_detection import detect_step_result
from .step_length import build_step_length_observation
from .trajectory import prepare_trajectory_steps


def prepare_pdr_steps(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    initial_direction: float = INITIAL_DIRECTION,
    height_m: float = USER_HEIGHT_M,
    step_detection_method: str | None = None,
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
    motion_refinement: bool = True,
    motion_estimation: str = MOTION_ESTIMATION,
    smoothing_mode: str = SMOOTHING_MODE,
    step_length_method: str = STEP_LENGTH_METHOD,
    direction_fixed_lag: int = 5,
) -> PreparedPdrSteps:
    """通常PDRとPFが共用するステップ単位の推定結果を作る。"""
    settings = PdrSettings(
        sensor=SensorSettings(
            gyro_bias_method=(
                GYRO_BIAS_METHOD if gyro_bias_method is None else gyro_bias_method
            ),
            gyro_bias=gyro_bias,
        ),
        step=StepSettings(
            detection_method=step_detection_method or StepSettings().detection_method,
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
    )
    return prepare_pdr_steps_with_settings(
        df_acc,
        df_gyro,
        settings,
        motion_refinement=motion_refinement,
        direction_fixed_lag=direction_fixed_lag,
    )


def prepare_pdr_steps_with_settings(
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    settings: PdrSettings,
    *,
    motion_refinement: bool = True,
    direction_fixed_lag: int = 5,
) -> PreparedPdrSteps:
    """検証済み設定から共有歩列を作る。"""
    sensor = settings.sensor
    step = settings.step
    heading = settings.heading
    motion = settings.motion_state
    processed_acc, processed_gyro = process_sensor_data(
        df_acc,
        df_gyro,
        gyro_bias_method=sensor.gyro_bias_method,
        gyro_bias=sensor.gyro_bias,
    )
    step_detection = detect_step_result(processed_acc, step.detection_method)
    weinberg_k = compute_weinberg_k(step.height_m)
    step_lengths, times, step_headings = prepare_trajectory_steps(
        step_detection.peaks,
        processed_gyro,
        processed_acc,
        heading.initial_direction,
        weinberg_k,
        heading_method=heading.method,
        step_segments=step_detection.segments,
        sidestep_lateral_ratio=motion.sidestep_lateral_ratio,
        sidestep_min_lateral_displacement=(motion.sidestep_min_lateral_displacement),
        motion_heading_correction=motion.motion_heading_correction,
        sidestep_smoothing=motion.sidestep_smoothing,
        forward_heading_source=motion.forward_heading_source,
        sidestep_heading_source=motion.sidestep_heading_source,
        sidestep_suspect_mode=motion.sidestep_suspect_mode,
        motion_refinement=motion_refinement,
        step_length_method=step.length_method,
    )
    motion_evidences = build_step_motion_evidences(step_headings)
    length_observations = tuple(
        build_step_length_observation(
            processed_acc,
            step_heading,
            step_length,
            weinberg_k,
        )
        for step_heading, step_length in zip(
            step_headings,
            step_lengths,
            strict=True,
        )
    )
    estimation = MOTION_ESTIMATORS[motion.motion_estimation](
        step_headings,
        step_lengths,
        length_observations,
        motion_evidences,
        motion.smoothing_mode,
        direction_fixed_lag,
    )
    step_headings = estimation.step_headings
    step_lengths = estimation.step_lengths
    trajectory = integrate_steps(step_headings, step_lengths)
    return PreparedPdrSteps(
        df_acc=processed_acc,
        df_gyro=processed_gyro,
        step_detection=step_detection,
        trajectory=trajectory,
        step_lengths=step_lengths,
        t_at_steps=times,
        step_headings=step_headings,
        gx_mean=float(processed_acc["gx"].mean()),
        gz_mean=float(processed_acc["gz"].mean()),
        weinberg_k=weinberg_k,
        heading_method=heading.method,
        motion_heading_correction=motion.motion_heading_correction,
        sidestep_smoothing=motion.sidestep_smoothing,
        forward_heading_source=motion.forward_heading_source,
        sidestep_heading_source=motion.sidestep_heading_source,
        sidestep_suspect_mode=motion.sidestep_suspect_mode,
        motion_evidences=estimation.motion_evidences,
        motion_observations=build_step_motion_observations(step_headings),
        length_observations=length_observations,
        motion_posteriors=estimation.motion_posteriors,
        motion_estimation=motion.motion_estimation,
        smoothing_mode=motion.smoothing_mode,
        direction_posteriors=estimation.direction_posteriors,
        particle_motion_headings=build_particle_motion_headings(step_headings),
    )
