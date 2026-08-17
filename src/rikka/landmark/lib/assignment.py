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

import numpy as np

from ...common.lib.models import LandmarkDetection


def assign_detections_to_steps(
    detections: tuple[LandmarkDetection, ...],
    t_at_steps: list[float],
) -> tuple[dict[int, list[LandmarkDetection]], int]:
    """検出時刻を0始まりの歩indexへ写像し、歩ごとの検出と破棄件数を返す。"""
    if not t_at_steps:
        return {}, len(detections)
    times = np.asarray(t_at_steps, dtype=float)
    assigned: dict[int, list[LandmarkDetection]] = {}
    discarded_count = 0
    for detection in detections:
        index = int(np.searchsorted(times, detection.timestamp_s, side="left"))
        if index >= len(times):
            discarded_count += 1
            continue
        assigned.setdefault(index, []).append(detection)
    return assigned, discarded_count


def build_step_landmark_map(
    detections: tuple[LandmarkDetection, ...],
    t_at_steps: list[float],
    landmark_meters: dict[str, tuple[float, float]],
) -> dict[int, LandmarkDetection]:
    """PFの1始まり歩番号からその歩で反映する最後の登録済み検出への辞書を返す。"""
    assigned, _ = assign_detections_to_steps(detections, t_at_steps)
    result: dict[int, LandmarkDetection] = {}
    for step_index, step_detections in assigned.items():
        registered = [
            detection
            for detection in step_detections
            if detection.beacon_id in landmark_meters
        ]
        if registered:
            result[step_index + 1] = registered[-1]
    return result
