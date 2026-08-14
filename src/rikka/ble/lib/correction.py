"""ランドマーク検出を軌跡座標へ反映する補正処理。

役割:
    検出したランドマークの既知座標へ推定位置を置き換え、その座標を起点に以降の
    歩の変位を積み直した軌跡を作る。
依存元:
    ``common.lib.models`` の ``LandmarkDetection`` / ``LandmarkCorrection`` /
    ``LandmarkCorrectionResult`` と ``common.settings.BleLandmarkSettings`` を使う。
利用先:
    ``ble.pipeline`` が ``pdr.pipeline.run_pdr`` へ渡す補正済み軌跡を作る際に使用する。
    検出処理とは分離しており、将来の重み付き補正へ差し替えられる。
処理フロー:
    検出時刻を歩 index へ写像し、歩ごとの変位を保ったまま逐次積分し、検出のある歩で
    座標をランドマークへ置き換えて補正履歴とともに返す。
"""

import numpy as np

from ...common.lib.models import (
    LandmarkCorrection,
    LandmarkCorrectionResult,
    LandmarkDetection,
)
from ...common.settings import BleLandmarkSettings


def assign_detections_to_steps(
    detections: tuple[LandmarkDetection, ...],
    t_at_steps: list[float],
) -> tuple[dict[int, list[LandmarkDetection]], int]:
    """検出時刻を歩 index へ写像し、歩ごとの検出と破棄件数を返す。"""
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


def apply_landmark_corrections(
    trajectory: list[list[float]],
    t_at_steps: list[float],
    detections: tuple[LandmarkDetection, ...],
    settings: BleLandmarkSettings,
    data_path: str,
) -> LandmarkCorrectionResult:
    """検出に従って軌跡を補正し、補正後の軌跡と履歴を返す。"""
    if not trajectory:
        raise ValueError("trajectory は 1 点以上必要です。")
    known = settings.landmark_map()
    assigned, discarded = assign_detections_to_steps(detections, t_at_steps)
    corrected: list[list[float]] = [list(trajectory[0])]
    corrections: list[LandmarkCorrection] = []

    for step_index in range(len(trajectory) - 1):
        delta_x = trajectory[step_index + 1][0] - trajectory[step_index][0]
        delta_y = trajectory[step_index + 1][1] - trajectory[step_index][1]
        position_x = corrected[step_index][0] + delta_x
        position_y = corrected[step_index][1] + delta_y

        step_detections = assigned.get(step_index, [])
        for order, detection in enumerate(step_detections):
            landmark = known.get(detection.beacon_id)
            if landmark is None:
                continue
            is_last = order == len(step_detections) - 1
            before_x, before_y = position_x, position_y
            if is_last:
                position_x, position_y = landmark.x, landmark.y
            corrections.append(
                LandmarkCorrection(
                    step_index=step_index,
                    timestamp_s=detection.timestamp_s,
                    beacon_id=detection.beacon_id,
                    rssi_dbm=detection.rssi_dbm,
                    before_x=before_x,
                    before_y=before_y,
                    landmark_x=landmark.x,
                    landmark_y=landmark.y,
                    after_x=position_x,
                    after_y=position_y,
                    applied=is_last,
                )
            )
        corrected.append([position_x, position_y])

    return LandmarkCorrectionResult(
        trajectory=corrected,
        raw_trajectory=[list(point) for point in trajectory],
        corrections=tuple(corrections),
        detection_count=len(detections),
        discarded_count=discarded,
        rssi_threshold_dbm=settings.rssi_threshold_dbm,
        data_path=data_path,
        detections=detections,
    )
