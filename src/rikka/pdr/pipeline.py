"""PDR の軌跡生成とステップ準備。

役割:
    センサーデータから歩ごとの方位、歩幅、時刻を揃え、通常 PDR の2次元軌跡と
    particle filter でも共有する ``PreparedPdrSteps`` を生成する。
依存元:
    ``sensors``、``step_detection``、``step_length``、``heading``、``sidestep``、
    ``time_utils`` の各処理と、``common`` の検証、``models`` の共有型を利用する。
利用先:
    ``pipeline.run`` が共通前処理として使用し、互換 facade が従来の軌跡推定関数を
    外部へ公開する。
処理フロー:
    センサー前処理、ステップ検出、方位・歩幅候補生成、横歩き平滑化、方位安定化を
    行い、各歩の移動ベクトルを原点から順に加算する。
"""

import numpy as np
import pandas as pd

from ..common.config import (
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
    WEINBERG_K,
    compute_weinberg_k,
)
from ..common.lib.models import (
    PreparedPdrSteps,
    StepHeading,
    StepSegment,
    TrajectoryResult,
)
from ..common.lib.pdr_math import (
    _validate_forward_heading_source,
    _validate_heading_method,
    _validate_motion_heading_correction,
    _validate_non_negative_parameter,
    _validate_positive_parameter,
    _validate_sidestep_heading_source,
    _validate_sidestep_smoothing,
    _validate_sidestep_suspect_mode,
)
from ..common.lib.sensors import load_sensor_data, process_sensor_data
from ..common.lib.time_utils import _step_output_time
from ..common.settings import PdrSettings
from .lib.fusion.protocol import MOTION_ESTIMATORS
from .lib.gyro_bias_estimators import _validate_gyro_bias_method
from .lib.heading.device_orientation import estimate_device_orientation_mode
from .lib.heading.motion import resolve_motion_heading_correction
from .lib.heading.resolver import (
    resolve_step_heading,
)
from .lib.integrate import integrate_steps
from .lib.motion_state.clustering import smooth_step_headings
from .lib.motion_state.evidence import (
    build_particle_motion_headings,
    build_step_motion_evidences,
    build_step_motion_observations,
)
from .lib.motion_state.heading_policy import stabilize_trajectory_headings
from .lib.motion_state.refinement import refine_step_headings_with_motion_model
from .lib.motion_state.step_motion import (
    estimate_step_motion,
)
from .lib.step_detection import detect_step_result
from .lib.step_length import (
    _estimate_initial_forward_angle,
    build_step_length_observation,
    estimate_step_length,
    estimate_step_length_forward,
)


def estimate_trajectory_with_headings(
    peaks: np.ndarray,
    df_gyro: pd.DataFrame,
    df_acc: pd.DataFrame,
    initial_direction: float = INITIAL_DIRECTION,
    weinberg_k: float = WEINBERG_K,
    heading_method: str = HEADING_METHOD,
    step_segments: tuple[StepSegment, ...] = (),
    sidestep_lateral_ratio: float = SIDESTEP_LATERAL_RATIO,
    sidestep_min_lateral_displacement: float = SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    motion_heading_correction: str = "auto",
    sidestep_smoothing: str = SIDESTEP_SMOOTHING_METHOD,
    forward_heading_source: str = FORWARD_HEADING_SOURCE,
    sidestep_heading_source: str = "motion",
    sidestep_suspect_mode: str = SIDESTEP_SUSPECT_MODE,
    motion_refinement: bool = True,
) -> tuple[list[list[float]], list[float], list[float], list[StepHeading]]:
    """ステップピークとジャイロスコープ角度から2次元軌跡を推定する。

    各ステップピーク時刻の平滑化角度（``low_angle``）と
    ``STEP_LENGTH_METHOD`` で選択した手法による歩幅推定をもとに次の座標を計算し，
    軌跡を構築する。原点 [0.0, 0.0] から始まり，ステップごとに座標を追加する。
    ``initial_direction`` を加算することで，歩行開始方向をフロアマップに合わせられる。

    Args:
        peaks (np.ndarray): ステップピークのインデックス配列
        df_gyro (pd.DataFrame): ``low_angle`` 列を含むジャイロスコープDataFrame
        df_acc (pd.DataFrame):
            ``v_acc``・``h_y``・``h_z``・``h_norm`` 列を含む加速度DataFrame
        initial_direction (float): 歩行開始方向のオフセット [度]
            （デフォルト: ``INITIAL_DIRECTION``）
        weinberg_k (float): Weinbergモデルのスケール係数

    Returns:
        tuple[list[list[float]], list[float], list[float]]:
            - 各ステップの [x, y] 座標リスト（原点を含む）
            - 各ステップの推定歩幅リスト [m]
            - 各移動後座標に対応する時刻リスト [s]
    """
    points: list[list[float]] = [[0.0, 0.0]]
    step_lengths: list[float] = []
    t_at_steps: list[float] = []
    step_headings: list[StepHeading] = []
    sidestep_lateral_ratio = _validate_positive_parameter(
        "sidestep_lateral_ratio",
        sidestep_lateral_ratio,
    )
    sidestep_min_lateral_displacement = _validate_non_negative_parameter(
        "sidestep_min_lateral_displacement",
        sidestep_min_lateral_displacement,
    )
    # 軌跡計算に入る前に、全ステップで共通利用する端末向きと
    # motion heading 補正を決める。
    device_orientation_mode = estimate_device_orientation_mode(
        df_acc,
        df_gyro,
        peaks,
        initial_direction,
        step_segments,
    )
    motion_heading_correction_rad = resolve_motion_heading_correction(
        df_acc,
        df_gyro,
        peaks,
        initial_direction,
        step_segments,
        motion_heading_correction,
        device_orientation_mode,
    )
    selected_sidestep_smoothing = _validate_sidestep_smoothing(sidestep_smoothing)
    selected_forward_heading_source = _validate_forward_heading_source(
        forward_heading_source
    )
    selected_sidestep_heading_source = _validate_sidestep_heading_source(
        sidestep_heading_source
    )
    selected_sidestep_suspect_mode = _validate_sidestep_suspect_mode(
        sidestep_suspect_mode
    )

    # forward 手法用: 初期前進角をデータから自動推定
    phi_0 = (
        _estimate_initial_forward_angle(df_acc, df_gyro, peaks)
        if STEP_LENGTH_METHOD == "forward"
        else 0.0
    )
    raw_step_headings: list[StepHeading] = []
    raw_step_lengths: list[float] = []
    raw_step_times: list[float] = []
    previous_heading: float | None = None
    # まず各ステップの候補 heading / 生の歩幅 / 時刻を揃える。
    for i, p in enumerate(peaks):
        if p >= len(df_acc):
            continue
        if STEP_LENGTH_METHOD == "forward" and i + 1 >= len(peaks):
            continue  # 次ピークなし：区間定義不可のためスキップ
        step_heading = resolve_step_heading(
            peaks,
            df_gyro,
            df_acc,
            i,
            initial_direction=initial_direction,
            heading_method=heading_method,
            step_segments=step_segments,
            motion_heading_correction=motion_heading_correction_rad,
            sidestep_lateral_ratio=sidestep_lateral_ratio,
            sidestep_min_lateral_displacement=sidestep_min_lateral_displacement,
            device_orientation_mode=device_orientation_mode,
        )
        if step_heading.selected_heading is None:
            continue
        if STEP_LENGTH_METHOD == "forward":
            step_length = estimate_step_length_forward(df_acc, df_gyro, peaks, i, phi_0)
        else:
            step_length = estimate_step_length(df_acc, int(p), k=weinberg_k)
        raw_step_headings.append(step_heading)
        raw_step_lengths.append(step_length)
        raw_step_times.append(_step_output_time(df_acc, peaks, i))

    # 横歩き判定を平滑化し、軌跡用 heading として安定化する。
    smoothed_step_headings = (
        refine_step_headings_with_motion_model(
            raw_step_headings,
            selected_sidestep_smoothing,
            selected_sidestep_suspect_mode,
        )
        if motion_refinement
        else smooth_step_headings(
            raw_step_headings,
            selected_sidestep_smoothing,
            selected_sidestep_suspect_mode,
        )
    )
    stabilized_step_headings = stabilize_trajectory_headings(
        smoothed_step_headings,
        selected_forward_heading_source,
        selected_sidestep_heading_source,
    )

    # StepMotion に変換しながら、1歩ずつ座標を積み上げる。
    for step_heading, step_length, step_time in zip(
        stabilized_step_headings,
        raw_step_lengths,
        raw_step_times,
        strict=True,
    ):
        step_motion = estimate_step_motion(
            step_heading,
            step_length,
            previous_heading,
            selected_forward_heading_source,
            selected_sidestep_heading_source,
            selected_sidestep_suspect_mode,
        )
        if step_motion is None:
            continue
        step_heading = step_heading._replace(
            selected_heading=step_motion.heading,
            source=step_heading.source
            if step_heading.source.startswith("trajectory_")
            else "state_motion",
            step_length_scale=step_motion.length_scale,
            trajectory_movement_type=step_motion.movement_type,
            forward_heading_source=selected_forward_heading_source,
        )
        step_lengths.append(step_motion.length)
        t_at_steps.append(step_time)
        step_headings.append(step_heading)
        previous_heading = step_motion.heading
    points = integrate_steps(step_headings, step_lengths)

    return points, step_lengths, t_at_steps, step_headings


def estimate_trajectory(
    peaks: np.ndarray,
    df_gyro: pd.DataFrame,
    df_acc: pd.DataFrame,
    initial_direction: float = INITIAL_DIRECTION,
    weinberg_k: float = WEINBERG_K,
) -> tuple[list[list[float]], list[float], list[float]]:
    """従来互換の決定論的PDR軌跡推定を行う。"""
    points, step_lengths, t_at_steps, _step_headings = (
        estimate_trajectory_with_headings(
            peaks,
            df_gyro,
            df_acc,
            initial_direction=initial_direction,
            weinberg_k=weinberg_k,
            heading_method="gyro",
        )
    )
    return points, step_lengths, t_at_steps


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
    direction_fixed_lag: int = 5,
) -> PreparedPdrSteps:
    """通常PDRとPFが共用するステップ単位の推定結果を作る。"""
    selected_gyro_bias_method = _validate_gyro_bias_method(
        GYRO_BIAS_METHOD if gyro_bias_method is None else gyro_bias_method
    )
    selected_heading_method = _validate_heading_method(
        HEADING_METHOD if heading_method is None else heading_method
    )
    selected_motion_heading_correction = _validate_motion_heading_correction(
        motion_heading_correction
    )
    selected_sidestep_smoothing = _validate_sidestep_smoothing(sidestep_smoothing)
    selected_forward_heading_source = _validate_forward_heading_source(
        forward_heading_source
    )
    selected_sidestep_heading_source = _validate_sidestep_heading_source(
        sidestep_heading_source
    )
    selected_sidestep_suspect_mode = _validate_sidestep_suspect_mode(
        sidestep_suspect_mode
    )
    sidestep_lateral_ratio = _validate_positive_parameter(
        "sidestep_lateral_ratio",
        sidestep_lateral_ratio,
    )
    sidestep_min_lateral_displacement = _validate_non_negative_parameter(
        "sidestep_min_lateral_displacement",
        sidestep_min_lateral_displacement,
    )
    if motion_estimation not in {"legacy", "adaptive", "robust"}:
        raise ValueError(
            "motion_estimation は legacy、adaptive、robust のいずれかを指定してください"
        )
    if smoothing_mode not in {"causal", "offline"}:
        raise ValueError("smoothing_mode は causal または offline を指定してください")

    # PDR と particle filter の両方が同じ前処理・ステップ検出・heading 推定を使う。
    processed_acc, processed_gyro = process_sensor_data(
        df_acc,
        df_gyro,
        gyro_bias_method=selected_gyro_bias_method,
        gyro_bias=gyro_bias,
    )
    step_detection = detect_step_result(processed_acc, step_detection_method)
    weinberg_k = compute_weinberg_k(height_m)
    trajectory, step_lengths, t_at_steps, step_headings = (
        estimate_trajectory_with_headings(
            step_detection.peaks,
            processed_gyro,
            processed_acc,
            initial_direction,
            weinberg_k,
            heading_method=selected_heading_method,
            step_segments=step_detection.segments,
            sidestep_lateral_ratio=sidestep_lateral_ratio,
            sidestep_min_lateral_displacement=sidestep_min_lateral_displacement,
            motion_heading_correction=selected_motion_heading_correction,
            sidestep_smoothing=selected_sidestep_smoothing,
            forward_heading_source=selected_forward_heading_source,
            sidestep_heading_source=selected_sidestep_heading_source,
            sidestep_suspect_mode=selected_sidestep_suspect_mode,
            motion_refinement=motion_refinement,
        )
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
    estimator = MOTION_ESTIMATORS[motion_estimation]
    estimation = estimator(
        step_headings,
        step_lengths,
        length_observations,
        motion_evidences,
        smoothing_mode,
        direction_fixed_lag,
    )
    step_headings = estimation.step_headings
    step_lengths = estimation.step_lengths
    motion_evidences = estimation.motion_evidences
    motion_posteriors = estimation.motion_posteriors
    direction_posteriors = estimation.direction_posteriors
    trajectory = integrate_steps(step_headings, step_lengths)

    return PreparedPdrSteps(
        df_acc=processed_acc,
        df_gyro=processed_gyro,
        step_detection=step_detection,
        trajectory=trajectory,
        step_lengths=step_lengths,
        t_at_steps=t_at_steps,
        step_headings=step_headings,
        gx_mean=float(processed_acc["gx"].mean()),
        gz_mean=float(processed_acc["gz"].mean()),
        weinberg_k=weinberg_k,
        heading_method=selected_heading_method,
        motion_heading_correction=selected_motion_heading_correction,
        sidestep_smoothing=selected_sidestep_smoothing,
        forward_heading_source=selected_forward_heading_source,
        sidestep_heading_source=selected_sidestep_heading_source,
        sidestep_suspect_mode=selected_sidestep_suspect_mode,
        motion_evidences=motion_evidences,
        motion_observations=build_step_motion_observations(step_headings),
        length_observations=length_observations,
        motion_posteriors=motion_posteriors,
        motion_estimation=motion_estimation,
        smoothing_mode=smoothing_mode,
        direction_posteriors=direction_posteriors,
        particle_motion_headings=build_particle_motion_headings(step_headings),
    )


def run_pdr(
    settings: PdrSettings,
    df_acc: pd.DataFrame | None = None,
    df_gyro: pd.DataFrame | None = None,
) -> TrajectoryResult:
    """センサー入力を共有ステップと通常PDR軌跡へ変換する。"""
    if (df_acc is None) != (df_gyro is None):
        raise ValueError("df_acc と df_gyro は両方渡すか、両方省略してください。")
    if df_acc is None and df_gyro is None:
        df_acc, df_gyro = load_sensor_data()
    if df_acc is None or df_gyro is None:
        raise RuntimeError("内部エラー: センサーデータが取得できませんでした。")

    prepared = prepare_pdr_steps(
        df_acc,
        df_gyro,
        initial_direction=settings.heading.initial_direction,
        height_m=settings.step.height_m,
        step_detection_method=settings.step.detection_method,
        heading_method=settings.heading.method,
        gyro_bias_method=settings.sensor.gyro_bias_method,
        gyro_bias=settings.sensor.gyro_bias,
        sidestep_lateral_ratio=settings.motion_state.sidestep_lateral_ratio,
        sidestep_min_lateral_displacement=(
            settings.motion_state.sidestep_min_lateral_displacement
        ),
        motion_heading_correction=(settings.motion_state.motion_heading_correction),
        sidestep_smoothing=settings.motion_state.sidestep_smoothing,
        forward_heading_source=settings.motion_state.forward_heading_source,
        sidestep_heading_source=settings.motion_state.sidestep_heading_source,
        sidestep_suspect_mode=settings.motion_state.sidestep_suspect_mode,
        motion_estimation=settings.motion_state.motion_estimation,
        smoothing_mode=settings.motion_state.smoothing_mode,
    )
    return TrajectoryResult(
        trajectory=prepared.trajectory,
        step_lengths=prepared.step_lengths,
        t_at_steps=prepared.t_at_steps,
        step_headings=prepared.step_headings,
        prepared=prepared,
    )
