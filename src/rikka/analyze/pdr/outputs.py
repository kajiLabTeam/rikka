"""PDR の CSV 出力用データ整形。

役割:
    軌跡、歩幅ベクトル、ステップ区間、方位診断、ジャイロ補正結果を保存可能な
    Pandas DataFrame に変換し、タイムスタンプ付き出力ディレクトリを作る。
依存元:
    ``models`` の各結果型と ``time_utils`` の時刻参照を利用し、NumPy、Pandas、
    pathlib、datetime で列値と保存先を構成する。
利用先:
    ``pipeline.run`` が通常 PDR と particle filter の CSV を書き出す際に使用する。
処理フロー:
    解析結果の配列長と時刻を揃え、角度や分類値を出力列へ変換して DataFrame を返す。
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .models import (
    GyroBiasResult,
    StepDetectionResult,
    StepHeading,
    StepLengthObservation,
    StepMotionPosterior,
    StepSegment,
)
from .time_utils import _time_at_index


def _create_output_dir(
    base_dir: str | Path = "output",
    now: datetime | None = None,
) -> Path:
    """衝突しないタイムスタンプ付き出力ディレクトリを作成して返す。"""
    current = now if now is not None else datetime.now()
    timestamp = current.strftime("%Y%m%d_%H%M%S_%f")
    base_path = Path(base_dir)

    for counter in range(1000):
        suffix = "" if counter == 0 else f"_{counter:03d}"
        output_dir = base_path / f"{timestamp}{suffix}"
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return output_dir

    raise FileExistsError(f"出力ディレクトリ名が衝突しました: {base_path / timestamp}")


def _build_step_vectors_dataframe(trajectory: list[list[float]]) -> pd.DataFrame:
    """軌跡点列からステップごとの変位ベクトルをDataFrame化する。"""
    points = np.asarray(trajectory, dtype=float)
    if len(points) < 2:
        return pd.DataFrame(
            columns=[
                "step",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "dx",
                "dy",
                "step_length_m",
                "heading_deg",
            ]
        )

    starts = points[:-1]
    ends = points[1:]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    headings = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return pd.DataFrame(
        {
            "step": np.arange(1, len(vectors) + 1),
            "start_x": starts[:, 0],
            "start_y": starts[:, 1],
            "end_x": ends[:, 0],
            "end_y": ends[:, 1],
            "dx": vectors[:, 0],
            "dy": vectors[:, 1],
            "step_length_m": lengths,
            "heading_deg": headings,
        }
    )


def _build_trajectory_dataframe(
    trajectory: list[list[float]],
    t_at_steps: list[float],
) -> pd.DataFrame:
    """軌跡点列と移動後座標の時刻から時刻付きDataFrameを作成する。"""
    points = np.asarray(trajectory, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("trajectory は [x, y] の点列である必要があります。")

    if len(points) != len(t_at_steps) + 1:
        raise ValueError(
            "trajectory と t_at_steps の長さが一致しません: "
            f"len(trajectory)={len(points)}, len(t_at_steps)={len(t_at_steps)}"
        )
    if len(t_at_steps) == 0:
        return pd.DataFrame(columns=["timestamp_s", "x", "y"])

    moved_points = points[1:]
    first_step_time = t_at_steps[0]
    timestamps = [float(t - first_step_time) for t in t_at_steps]
    return pd.DataFrame(
        {
            "timestamp_s": timestamps,
            "x": moved_points[:, 0],
            "y": moved_points[:, 1],
        }
    )


def _build_step_segments_dataframe(
    df_acc: pd.DataFrame,
    segments: tuple[StepSegment, ...],
) -> pd.DataFrame:
    """ステップ区間情報をCSV保存用DataFrameに変換する。"""
    columns = [
        "step",
        "start_index",
        "end_index",
        "contact_index",
        "start_time_s",
        "end_time_s",
        "duration_s",
    ]
    if len(segments) == 0:
        return pd.DataFrame(columns=columns)

    rows = []
    for step, segment in enumerate(segments, start=1):
        start_time = _time_at_index(df_acc, segment.start_index)
        end_time = _time_at_index(df_acc, segment.end_index)
        rows.append(
            {
                "step": step,
                "start_index": segment.start_index,
                "end_index": segment.end_index,
                "contact_index": segment.contact_index,
                "start_time_s": start_time,
                "end_time_s": end_time,
                "duration_s": end_time - start_time,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _angle_to_deg(angle: float | None) -> float | None:
    """ラジアン角を度へ変換する。None はそのまま返す。"""
    if angle is None:
        return None
    return float(np.degrees(angle))


def _build_gyro_bias_dataframe(df_gyro: pd.DataFrame) -> pd.DataFrame:
    """ジャイロバイアス診断情報をCSV保存用DataFrameに変換する。"""
    columns = [
        "method",
        "bias_rad_s",
        "calibration_start_s",
        "calibration_end_s",
        "sample_count",
        "kept_sample_count",
        "raw_mean",
        "robust_mean",
        "median",
        "mad",
        "candidate_score",
        "gyro_std",
        "accel_p95",
        "accel_max",
        "search_start_s",
        "search_end_s",
        "fallback_reason",
    ]
    result = df_gyro.attrs.get("gyro_bias_result")
    if isinstance(result, GyroBiasResult):
        return pd.DataFrame(
            [
                {
                    "method": result.method,
                    "bias_rad_s": result.bias_rad_s,
                    "calibration_start_s": result.calibration_start_s,
                    "calibration_end_s": result.calibration_end_s,
                    "sample_count": result.sample_count,
                    "kept_sample_count": result.kept_sample_count,
                    "raw_mean": result.raw_mean,
                    "robust_mean": result.robust_mean,
                    "median": result.median,
                    "mad": result.mad,
                    "candidate_score": result.candidate_score,
                    "gyro_std": result.gyro_std,
                    "accel_p95": result.accel_p95,
                    "accel_max": result.accel_max,
                    "search_start_s": result.search_start_s,
                    "search_end_s": result.search_end_s,
                    "fallback_reason": result.fallback_reason,
                }
            ],
            columns=columns,
        )

    return pd.DataFrame(columns=columns)


def _lateral_forward_ratio_for_output(step_heading: StepHeading) -> float | None:
    """出力用に横方向/前方向の比率を計算する。"""
    if (
        step_heading.forward_displacement is None
        or step_heading.lateral_displacement is None
    ):
        return None
    return abs(step_heading.lateral_displacement) / max(
        abs(step_heading.forward_displacement), 1e-12
    )


def _build_step_headings_dataframe(step_headings: list[StepHeading]) -> pd.DataFrame:
    """ステップ方位候補と採用結果をCSV保存用DataFrameに変換する。"""
    columns = [
        "step",
        "timestamp_s",
        "gyro_heading_deg",
        "body_heading_deg",
        "accel_method1_heading_deg",
        "accel_method2_heading_deg",
        "motion_heading_deg",
        "selected_heading_deg",
        "source",
        "movement_type",
        "trajectory_movement_type",
        "forward_heading_source",
        "step_length_scale",
        "confidence",
        "motion_confidence",
        "yaw_delta_deg",
        "motion_heading_correction_deg",
        "device_orientation_mode",
        "decoded_motion_mode",
        "decoded_motion_confidence",
        "device_body_offset_deg",
        "dynamic_body_heading_confidence",
        "body_heading_update_reason",
        "body_motion_angle_diff_deg",
        "lateral_forward_ratio",
        "sidestep_lateral_ratio",
        "sidestep_min_lateral_displacement",
        "sidestep_evidence_direction",
        "sidestep_evidence_reason",
        "sidestep_cluster_id",
        "angle_diff_method1_deg",
        "angle_diff_method2_deg",
        "forward_displacement",
        "lateral_displacement",
        "motion_reject_reason",
        "segment_start_index",
        "segment_end_index",
        "peak1_index",
        "peak2_index",
    ]
    rows = [
        {
            "step": heading.step_index,
            "timestamp_s": heading.timestamp_s,
            "gyro_heading_deg": _angle_to_deg(heading.gyro_heading),
            "body_heading_deg": _angle_to_deg(heading.body_heading),
            "accel_method1_heading_deg": _angle_to_deg(heading.accel_method1_heading),
            "accel_method2_heading_deg": _angle_to_deg(heading.accel_method2_heading),
            "motion_heading_deg": _angle_to_deg(heading.motion_heading),
            "selected_heading_deg": _angle_to_deg(heading.selected_heading),
            "source": heading.source,
            "movement_type": heading.movement_type,
            "trajectory_movement_type": heading.trajectory_movement_type,
            "forward_heading_source": heading.forward_heading_source,
            "step_length_scale": heading.step_length_scale,
            "confidence": heading.confidence,
            "motion_confidence": heading.motion_confidence,
            "yaw_delta_deg": _angle_to_deg(heading.yaw_delta),
            "motion_heading_correction_deg": _angle_to_deg(
                heading.motion_heading_correction
            ),
            "device_orientation_mode": heading.device_orientation_mode,
            "decoded_motion_mode": heading.decoded_motion_mode,
            "decoded_motion_confidence": heading.decoded_motion_confidence,
            "device_body_offset_deg": _angle_to_deg(heading.device_body_offset),
            "dynamic_body_heading_confidence": (
                heading.dynamic_body_heading_confidence
            ),
            "body_heading_update_reason": heading.body_heading_update_reason,
            "body_motion_angle_diff_deg": _angle_to_deg(heading.body_motion_angle_diff),
            "lateral_forward_ratio": _lateral_forward_ratio_for_output(heading),
            "sidestep_lateral_ratio": heading.sidestep_lateral_ratio,
            "sidestep_min_lateral_displacement": (
                heading.sidestep_min_lateral_displacement
            ),
            "sidestep_evidence_direction": heading.sidestep_evidence_direction,
            "sidestep_evidence_reason": heading.sidestep_evidence_reason,
            "sidestep_cluster_id": heading.sidestep_cluster_id,
            "angle_diff_method1_deg": _angle_to_deg(heading.angle_diff_method1),
            "angle_diff_method2_deg": _angle_to_deg(heading.angle_diff_method2),
            "forward_displacement": heading.forward_displacement,
            "lateral_displacement": heading.lateral_displacement,
            "motion_reject_reason": heading.motion_reject_reason,
            "segment_start_index": heading.segment_start_index,
            "segment_end_index": heading.segment_end_index,
            "peak1_index": heading.peak1_index,
            "peak2_index": heading.peak2_index,
        }
        for heading in step_headings
    ]
    return pd.DataFrame(rows, columns=columns)


def _build_step_length_observations_dataframe(
    observations: tuple[StepLengthObservation, ...],
) -> pd.DataFrame:
    """歩幅の物理観測と品質を診断CSV用に変換する。"""
    return pd.DataFrame(
        [observation._asdict() for observation in observations],
        columns=StepLengthObservation._fields,
    )


def _build_motion_posteriors_dataframe(
    posteriors: tuple[StepMotionPosterior, ...],
) -> pd.DataFrame:
    """適応PDRの状態・方位・歩幅事後分布を診断CSV用に変換する。"""
    rows = []
    for posterior in posteriors:
        values = posterior._asdict()
        values["heading_mean_deg"] = _angle_to_deg(posterior.heading_mean)
        values["heading_std_deg"] = _angle_to_deg(posterior.heading_std)
        values["device_body_offset_mean_deg"] = _angle_to_deg(
            posterior.device_body_offset_mean
        )
        values["device_body_offset_std_deg"] = _angle_to_deg(
            posterior.device_body_offset_std
        )
        del values["heading_mean"]
        del values["heading_std"]
        del values["device_body_offset_mean"]
        del values["device_body_offset_std"]
        rows.append(values)
    return pd.DataFrame(rows)


def _step_plot_signal(
    df_acc: pd.DataFrame,
    step_detection: StepDetectionResult,
) -> tuple[np.ndarray, str, float | None]:
    """ステップ検出方式に対応する可視化用信号を返す。"""
    if step_detection.method == "paper_vertical_threshold":
        polarity = 1 if step_detection.polarity is None else step_detection.polarity
        return (
            df_acc["v_acc"].to_numpy(dtype=float) * polarity,
            "vertical contact signal",
            step_detection.threshold,
        )
    return df_acc["low_lin_norm"].to_numpy(dtype=float), "low_lin_norm", None
