"""ランドマーク検出と歩の共通割り当て。

役割:
    検出時刻をPDRとparticle filterが共有する歩時刻へ対応付け、方式ごとの
    反映処理が同じ割り当て結果を利用できるようにする。
依存元:
    ``common.lib.models.LandmarkDetection`` と NumPy の時刻検索を利用する。
利用先:
    ``pdr.lib.landmark_correction`` が0始まりの歩indexを使い、将来の
    particle filter が1始まりの歩番号マップを使用する。
処理フロー:
    各検出をその時刻以降で最初の歩へ割り当て、最終歩後は破棄する。PF用では
    未登録検出を除き、同一歩の最後の検出を1始まり歩番号へ対応付ける。
"""

from collections.abc import Sequence

import numpy as np

from ...common.lib.models import LandmarkObservation, LandmarkRange


def assign_detections_to_steps(
    detections: tuple[LandmarkObservation, ...],
    t_at_steps: np.ndarray | Sequence[float],
) -> tuple[dict[int, list[LandmarkObservation]], int]:
    """検出時刻を0始まりの歩indexへ写像し、歩ごとの検出と破棄件数を返す。"""
    if len(t_at_steps) == 0:
        return {}, len(detections)
    times = np.asarray(t_at_steps, dtype=float)
    assigned: dict[int, list[LandmarkObservation]] = {}
    discarded_count = 0
    for detection in detections:
        index = int(np.searchsorted(times, detection.timestamp_s, side="left"))
        if index >= len(times):
            discarded_count += 1
            continue
        assigned.setdefault(index, []).append(detection)
    return assigned, discarded_count


def build_step_landmark_map(
    detections: tuple[LandmarkObservation, ...],
    t_at_steps: np.ndarray | Sequence[float],
    landmark_meters: dict[str, tuple[float, float]],
) -> dict[int, LandmarkObservation]:
    """PFの1始まり歩番号からその歩で反映する最後の登録済み検出への辞書を返す。"""
    assigned, _ = assign_detections_to_steps(detections, t_at_steps)
    result: dict[int, LandmarkObservation] = {}
    for step_index, step_detections in assigned.items():
        registered = [
            detection
            for detection in step_detections
            if detection.beacon_id in landmark_meters
        ]
        if registered:
            result[step_index + 1] = registered[-1]
    return result


def build_step_observation_map(
    observations: tuple[LandmarkRange, ...],
    t_at_steps: np.ndarray | Sequence[float],
    landmark_meters: dict[str, tuple[float, float]],
) -> dict[int, tuple[LandmarkRange, ...]]:
    """全測距観測を歩・ビーコンごとのRSSI中央値へ集約する。"""
    assigned, _ = assign_detections_to_steps(observations, t_at_steps)
    result: dict[int, tuple[LandmarkRange, ...]] = {}
    for step_index, step_observations in assigned.items():
        grouped: dict[str, list[LandmarkRange]] = {}
        for observation in step_observations:
            if (
                isinstance(observation, LandmarkRange)
                and observation.beacon_id in landmark_meters
            ):
                grouped.setdefault(observation.beacon_id, []).append(observation)
        medians = []
        for beacon_id, rows in sorted(grouped.items()):
            medians.append(
                LandmarkRange(
                    timestamp_s=float(np.median([item.timestamp_s for item in rows])),
                    beacon_id=beacon_id,
                    rssi_dbm=float(np.median([item.rssi_dbm for item in rows])),
                    distance_m=float(np.median([item.distance_m for item in rows])),
                    distance_sigma_m=float(
                        np.median([item.distance_sigma_m for item in rows])
                    ),
                    smoothed_rssi_dbm=float(
                        np.median([item.smoothed_rssi_dbm for item in rows])
                    ),
                )
            )
        if medians:
            result[step_index + 1] = tuple(medians)
    return result
