"""ランドマーク検出時刻と歩行軌跡の時間整合評価。

役割:
    検出時刻の歩行者位置、近接する局所最接近時刻、経路への最短距離を計算する。
依存元:
    NumPy と、PDR/PF が共有するメートル座標の軌跡・歩時刻を受け取る。
利用先:
    PDR補正、PF診断、agent検証が同じ定義で時間合わせを評価するために使用する。
処理フロー:
    歩軌跡を検出時刻へ線形補間し、距離系列の局所最小点から検出時刻に最も近い
    接近イベントを選び、検出距離・時間差・位置ずれ残差を返す。
"""

from typing import NamedTuple

import numpy as np


class LandmarkTimingMetrics(NamedTuple):
    """1検出についての位置・時刻整合指標。"""

    detection_distance_m: float
    nearest_approach_time_s: float
    nearest_approach_delta_s: float
    nearest_approach_distance_m: float
    position_residual_m: float


def _trajectory_times(
    t_at_steps: list[float] | np.ndarray,
    trajectory_length: int,
) -> np.ndarray:
    """初期点を含む軌跡時刻列を作る。"""
    step_times = np.asarray(t_at_steps, dtype=float)
    if trajectory_length != len(step_times) + 1:
        raise ValueError("trajectory は t_at_steps より1点多い必要があります。")
    if len(step_times) > 1 and np.any(np.diff(step_times) <= 0.0):
        raise ValueError("t_at_steps は狭義単調増加である必要があります。")
    initial_time = 0.0 if len(step_times) == 0 else min(0.0, float(step_times[0]))
    return np.concatenate(([initial_time], step_times))


def _local_minimum_indices(distances: np.ndarray) -> np.ndarray:
    """端点を含む距離系列の局所最小indexを返す。"""
    if len(distances) == 1:
        return np.asarray([0], dtype=int)
    indices: list[int] = []
    if distances[0] <= distances[1]:
        indices.append(0)
    indices.extend(
        index
        for index in range(1, len(distances) - 1)
        if distances[index] <= distances[index - 1]
        and distances[index] <= distances[index + 1]
        and (
            distances[index] < distances[index - 1]
            or distances[index] < distances[index + 1]
        )
    )
    if distances[-1] <= distances[-2]:
        indices.append(len(distances) - 1)
    if not indices:
        indices.append(int(np.argmin(distances)))
    return np.asarray(indices, dtype=int)


def evaluate_landmark_timing(
    trajectory: list[list[float]] | np.ndarray,
    t_at_steps: list[float] | np.ndarray,
    detection_time_s: float,
    landmark_xy: tuple[float, float],
) -> LandmarkTimingMetrics:
    """検出時刻と時間的に対応する局所最接近の指標を返す。"""
    positions = np.asarray(trajectory, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
        raise ValueError("trajectory は1点以上の2次元座標列が必要です。")
    times = _trajectory_times(t_at_steps, len(positions))
    landmark = np.asarray(landmark_xy, dtype=float)
    detection_position = np.asarray(
        [
            np.interp(detection_time_s, times, positions[:, 0]),
            np.interp(detection_time_s, times, positions[:, 1]),
        ],
        dtype=float,
    )
    detection_distance = float(np.linalg.norm(detection_position - landmark))
    distances = np.linalg.norm(positions - landmark, axis=1)
    local_indices = _local_minimum_indices(distances)
    nearest_index = int(
        local_indices[np.argmin(np.abs(times[local_indices] - detection_time_s))]
    )
    nearest_time = float(times[nearest_index])
    nearest_distance = float(distances[nearest_index])
    return LandmarkTimingMetrics(
        detection_distance_m=detection_distance,
        nearest_approach_time_s=nearest_time,
        nearest_approach_delta_s=float(detection_time_s - nearest_time),
        nearest_approach_distance_m=nearest_distance,
        position_residual_m=max(0.0, detection_distance - nearest_distance),
    )
