"""PDR の軌跡生成とステップ準備。

センサー処理済み DataFrame とステップ検出結果から、歩ごとの heading / step length /
timestamp を揃え、最終的な2次元軌跡へ積み上げる。通常 PDR と particle filter が
同じ決定論的な歩行ステップ情報を共有できるよう、prepare_pdr_steps() もここに置く。
"""

import numpy as np
import pandas as pd

from ...config import (
    FORWARD_HEADING_SOURCE,
    GYRO_BIAS_METHOD,
    HEADING_METHOD,
    INITIAL_DIRECTION,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    SIDESTEP_SMOOTHING_METHOD,
    STEP_LENGTH_METHOD,
    USER_HEIGHT_M,
    WEINBERG_K,
    compute_weinberg_k,
)
from .common import (
    _validate_forward_heading_source,
    _validate_heading_method,
    _validate_motion_heading_correction,
    _validate_non_negative_parameter,
    _validate_positive_parameter,
    _validate_sidestep_heading_source,
    _validate_sidestep_smoothing,
    _validate_sidestep_suspect_mode,
)
from .gyro_bias import _validate_gyro_bias_method
from .heading import (
    _estimate_device_orientation_mode,
    _resolve_motion_heading_correction,
    resolve_step_heading,
)
from .models import PreparedPdrSteps, StepHeading, StepSegment
from .sensors import process_sensor_data
from .sidestep import (
    _smooth_step_headings,
    _stabilize_trajectory_headings,
    estimate_step_motion,
)
from .step_detection import detect_step_result
from .step_length import (
    _estimate_initial_forward_angle,
    estimate_step_length,
    estimate_step_length_forward,
)
from .time_utils import _step_output_time


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
    sidestep_suspect_mode: str = "motion",
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
    device_orientation_mode = _estimate_device_orientation_mode(
        df_acc,
        df_gyro,
        peaks,
        initial_direction,
        step_segments,
    )
    motion_heading_correction_rad = _resolve_motion_heading_correction(
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
    smoothed_step_headings = _smooth_step_headings(
        raw_step_headings,
        selected_sidestep_smoothing,
        selected_sidestep_suspect_mode,
    )
    stabilized_step_headings = _stabilize_trajectory_headings(
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
        x = points[-1][0] + step_motion.length * float(np.cos(step_motion.heading))
        y = points[-1][1] + step_motion.length * float(np.sin(step_motion.heading))
        points.append([x, y])

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
    sidestep_suspect_mode: str = "motion",
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
        )
    )

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
    )
