"""歩ごとの方位・歩幅から通常PDR軌跡を生成する。

役割:
    センサーとステップピークから生の歩候補を集め、運動状態を反映した軌跡へ統合する。
依存元:
    heading、step length、motion state、共通設定と時刻処理を利用する。
利用先:
    PDR preparation と従来互換の trajectory API から使用される。
処理フロー:
    端末向き決定、生候補収集、状態平滑化、歩運動確定、軌跡積分の順に処理する。
"""

import numpy as np
import pandas as pd

from ...common.config import (
    FORWARD_HEADING_SOURCE,
    HEADING_METHOD,
    INITIAL_DIRECTION,
    SIDESTEP_LATERAL_RATIO,
    SIDESTEP_MIN_LATERAL_DISPLACEMENT_M,
    SIDESTEP_SMOOTHING_METHOD,
    SIDESTEP_SUSPECT_MODE,
    STEP_LENGTH_METHOD,
    WEINBERG_K,
)
from ...common.lib.models import StepHeading, StepSegment
from ...common.lib.time_utils import _step_output_time
from ...common.settings import MotionStateSettings
from .heading.device_orientation import estimate_device_orientation_mode
from .heading.motion import resolve_motion_heading_correction
from .heading.resolver import resolve_step_heading
from .integrate import integrate_steps
from .motion_state.clustering import smooth_step_headings
from .motion_state.heading_policy import stabilize_trajectory_headings
from .motion_state.refinement import refine_step_headings_with_motion_model
from .motion_state.step_motion import estimate_step_motion
from .step_length import (
    _estimate_initial_forward_angle,
    estimate_step_length,
    estimate_step_length_forward,
)


def _collect_raw_steps(
    peaks: np.ndarray,
    df_gyro: pd.DataFrame,
    df_acc: pd.DataFrame,
    initial_direction: float,
    weinberg_k: float,
    heading_method: str,
    step_segments: tuple[StepSegment, ...],
    motion: MotionStateSettings,
    device_orientation_mode: str,
    motion_heading_correction_rad: float,
) -> tuple[list[StepHeading], list[float], list[float]]:
    """各ピークから方位候補、生の歩幅、出力時刻を同じ順序で集める。"""
    phi_0 = (
        _estimate_initial_forward_angle(df_acc, df_gyro, peaks)
        if STEP_LENGTH_METHOD == "forward"
        else 0.0
    )
    headings: list[StepHeading] = []
    lengths: list[float] = []
    times: list[float] = []
    for index, peak in enumerate(peaks):
        if peak >= len(df_acc):
            continue
        if STEP_LENGTH_METHOD == "forward" and index + 1 >= len(peaks):
            continue
        heading = resolve_step_heading(
            peaks,
            df_gyro,
            df_acc,
            index,
            initial_direction=initial_direction,
            heading_method=heading_method,
            step_segments=step_segments,
            motion_heading_correction=motion_heading_correction_rad,
            sidestep_lateral_ratio=motion.sidestep_lateral_ratio,
            sidestep_min_lateral_displacement=(
                motion.sidestep_min_lateral_displacement
            ),
            device_orientation_mode=device_orientation_mode,
        )
        if heading.selected_heading is None:
            continue
        length = (
            estimate_step_length_forward(df_acc, df_gyro, peaks, index, phi_0)
            if STEP_LENGTH_METHOD == "forward"
            else estimate_step_length(df_acc, int(peak), k=weinberg_k)
        )
        headings.append(heading)
        lengths.append(length)
        times.append(_step_output_time(df_acc, peaks, index))
    return headings, lengths, times


def _resolve_motion_steps(
    raw_headings: list[StepHeading],
    raw_lengths: list[float],
    raw_times: list[float],
    motion: MotionStateSettings,
    motion_refinement: bool,
) -> tuple[list[list[float]], list[float], list[float], list[StepHeading]]:
    """運動状態を平滑化して歩ごとの移動方位と実効歩幅を確定する。"""
    smoothed = (
        refine_step_headings_with_motion_model(
            raw_headings,
            motion.sidestep_smoothing,
            motion.sidestep_suspect_mode,
        )
        if motion_refinement
        else smooth_step_headings(
            raw_headings,
            motion.sidestep_smoothing,
            motion.sidestep_suspect_mode,
        )
    )
    stabilized = stabilize_trajectory_headings(
        smoothed,
        motion.forward_heading_source,
        motion.sidestep_heading_source,
    )
    step_lengths: list[float] = []
    step_times: list[float] = []
    step_headings: list[StepHeading] = []
    previous_heading: float | None = None
    for heading, length, step_time in zip(
        stabilized,
        raw_lengths,
        raw_times,
        strict=True,
    ):
        step_motion = estimate_step_motion(
            heading,
            length,
            previous_heading,
            motion.forward_heading_source,
            motion.sidestep_heading_source,
            motion.sidestep_suspect_mode,
        )
        if step_motion is None:
            continue
        heading = heading._replace(
            selected_heading=step_motion.heading,
            source=(
                heading.source
                if heading.source.startswith("trajectory_")
                else "state_motion"
            ),
            step_length_scale=step_motion.length_scale,
            trajectory_movement_type=step_motion.movement_type,
            forward_heading_source=motion.forward_heading_source,
        )
        step_lengths.append(step_motion.length)
        step_times.append(step_time)
        step_headings.append(heading)
        previous_heading = step_motion.heading
    return (
        integrate_steps(step_headings, step_lengths),
        step_lengths,
        step_times,
        step_headings,
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
    """ステップピークから状態補正済みの2次元軌跡を推定する。"""
    motion = MotionStateSettings(
        sidestep_lateral_ratio=sidestep_lateral_ratio,
        sidestep_min_lateral_displacement=sidestep_min_lateral_displacement,
        motion_heading_correction=motion_heading_correction,
        sidestep_smoothing=sidestep_smoothing,
        forward_heading_source=forward_heading_source,
        sidestep_heading_source=sidestep_heading_source,
        sidestep_suspect_mode=sidestep_suspect_mode,
    )
    device_mode = estimate_device_orientation_mode(
        df_acc,
        df_gyro,
        peaks,
        initial_direction,
        step_segments,
    )
    correction_rad = resolve_motion_heading_correction(
        df_acc,
        df_gyro,
        peaks,
        initial_direction,
        step_segments,
        motion.motion_heading_correction,
        device_mode,
    )
    raw = _collect_raw_steps(
        peaks,
        df_gyro,
        df_acc,
        initial_direction,
        weinberg_k,
        heading_method,
        step_segments,
        motion,
        device_mode,
        correction_rad,
    )
    return _resolve_motion_steps(*raw, motion, motion_refinement)


def estimate_trajectory(
    peaks: np.ndarray,
    df_gyro: pd.DataFrame,
    df_acc: pd.DataFrame,
    initial_direction: float = INITIAL_DIRECTION,
    weinberg_k: float = WEINBERG_K,
) -> tuple[list[list[float]], list[float], list[float]]:
    """従来互換の決定論的PDR軌跡推定を行う。"""
    points, step_lengths, t_at_steps, _ = estimate_trajectory_with_headings(
        peaks,
        df_gyro,
        df_acc,
        initial_direction=initial_direction,
        weinberg_k=weinberg_k,
        heading_method="gyro",
    )
    return points, step_lengths, t_at_steps
