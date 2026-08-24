"""ランドマーク検出を軌跡座標へ反映する補正処理。

役割:
    検出したランドマークの既知座標へ推定位置を置き換え、その座標を起点に以降の
    歩の変位を積み直した軌跡を作る。
依存元:
    ``common.lib.models`` の ``LandmarkDetection`` / ``LandmarkCorrection`` /
    ``LandmarkCorrectionResult``、``landmark.lib`` の共通割り当てと座標変換を使う。
利用先:
    ``pdr.pipeline.run_pdr`` が BLE 検出を通常PDR軌跡へ反映する際に使用する。
    検出処理とは分離しており、将来の重み付き補正へ差し替えられる。
処理フロー:
    検出時刻を歩 index へ写像し、歩ごとの変位を保ったまま逐次積分し、検出のある歩で
    座標をランドマークへ置き換えて補正履歴とともに返す。
"""

import numpy as np

from ...common.lib.models import (
    FloorMap,
    Landmark,
    LandmarkCorrection,
    LandmarkCorrectionResult,
    LandmarkObservation,
    LandmarkRange,
)
from ...landmark.lib.assignment import assign_detections_to_steps
from ...landmark.lib.coordinates import build_landmark_meter_map
from ...landmark.lib.timing import evaluate_landmark_timing


def _constraint_position(
    before: tuple[float, float],
    landmark: tuple[float, float],
    estimated_distance_m: float | None,
) -> tuple[float, float]:
    """ビーコン距離円周上で補正前位置に最も近い拘束点を返す。"""
    if estimated_distance_m is None or estimated_distance_m <= 0.0:
        return landmark
    before_array = np.asarray(before, dtype=float)
    landmark_array = np.asarray(landmark, dtype=float)
    delta = before_array - landmark_array
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-12:
        return landmark
    target = landmark_array + estimated_distance_m * delta / norm
    return float(target[0]), float(target[1])


def _apply_translation(
    corrected: list[list[float]],
    raw_trajectory: list[list[float]],
    *,
    endpoint_index: int,
    target: tuple[float, float],
    mode: str,
    warp_start_point: int,
) -> float:
    """拘束残差をsnapまたは累積距離比warpで軌跡へ反映し、区間長を返す。"""
    delta_x = target[0] - corrected[endpoint_index][0]
    delta_y = target[1] - corrected[endpoint_index][1]
    segment_lengths = [
        float(
            np.linalg.norm(
                np.asarray(raw_trajectory[index]) - raw_trajectory[index - 1]
            )
        )
        for index in range(warp_start_point + 1, endpoint_index + 1)
    ]
    span = float(sum(segment_lengths))
    if mode == "warp" and span > 0.0:
        cumulative = 0.0
        for point_index, segment_length in enumerate(
            segment_lengths,
            start=warp_start_point + 1,
        ):
            cumulative += segment_length
            weight = cumulative / span
            corrected[point_index][0] += delta_x * weight
            corrected[point_index][1] += delta_y * weight
    else:
        corrected[endpoint_index][0] += delta_x
        corrected[endpoint_index][1] += delta_y
    for point_index in range(endpoint_index + 1, len(corrected)):
        corrected[point_index][0] += delta_x
        corrected[point_index][1] += delta_y
    return span


def apply_landmark_corrections(
    trajectory: list[list[float]],
    t_at_steps: list[float],
    *,
    detections: tuple[LandmarkObservation, ...],
    landmarks: tuple[Landmark, ...],
    floormap: FloorMap,
    data_path: str,
    rssi_threshold_dbm: float,
    gx_mean: float,
    gz_mean: float,
    correction_mode: str = "snap",
) -> LandmarkCorrectionResult:
    """検出に従って軌跡を補正し、補正後の軌跡と履歴を返す。"""
    if not trajectory:
        raise ValueError("trajectory は 1 点以上必要です。")
    known = build_landmark_meter_map(
        landmarks,
        gx_mean,
        gz_mean,
        floormap,
    )
    definitions = {item.beacon_id: item for item in landmarks}
    assigned, discarded = assign_detections_to_steps(detections, t_at_steps)
    if correction_mode not in {"snap", "warp"}:
        raise ValueError("correction_mode は snap または warp を指定してください。")
    corrected = [list(point) for point in trajectory]
    corrections: list[LandmarkCorrection] = []
    last_constraint_point = 0

    for step_index in range(len(trajectory) - 1):
        step_detections = assigned.get(step_index, [])
        registered = [
            detection for detection in step_detections if detection.beacon_id in known
        ]
        for order, detection in enumerate(registered):
            landmark_position = known.get(detection.beacon_id)
            if landmark_position is None:  # pragma: no cover - 上の絞り込みとの型境界
                raise RuntimeError("内部エラー: 登録済みランドマーク座標がありません。")
            landmark_x, landmark_y = landmark_position
            landmark = definitions[detection.beacon_id]
            is_last = order == len(registered) - 1
            endpoint_index = step_index + 1
            before_x, before_y = corrected[endpoint_index]
            timing = evaluate_landmark_timing(
                trajectory,
                t_at_steps,
                detection.timestamp_s,
                landmark_position,
            )
            estimated_distance = (
                detection.distance_m if isinstance(detection, LandmarkRange) else None
            )
            target = (landmark_x, landmark_y)
            warp_span = None
            warp_start_point = last_constraint_point
            if is_last:
                if correction_mode == "warp":
                    target = _constraint_position(
                        (before_x, before_y),
                        (landmark_x, landmark_y),
                        estimated_distance,
                    )
                warp_span = _apply_translation(
                    corrected,
                    trajectory,
                    endpoint_index=endpoint_index,
                    target=target,
                    mode=correction_mode,
                    warp_start_point=warp_start_point,
                )
                last_constraint_point = endpoint_index
            corrections.append(
                LandmarkCorrection(
                    step_index=step_index,
                    timestamp_s=detection.timestamp_s,
                    beacon_id=detection.beacon_id,
                    rssi_dbm=detection.rssi_dbm,
                    before_x=before_x,
                    before_y=before_y,
                    landmark_x=landmark_x,
                    landmark_y=landmark_y,
                    after_x=corrected[endpoint_index][0],
                    after_y=corrected[endpoint_index][1],
                    applied=is_last,
                    detection_distance_m=(
                        (before_x - landmark_x) ** 2 + (before_y - landmark_y) ** 2
                    )
                    ** 0.5,
                    nearest_approach_delta_s=timing.nearest_approach_delta_s,
                    anchor_position_sigma_m=landmark.position_sigma_m,
                    anchor_heading_deg=landmark.heading_deg,
                    anchor_heading_sigma_deg=(
                        landmark.heading_sigma_deg
                        if landmark.position_sigma_m is not None
                        else None
                    ),
                    anchor_heading_bidirectional=landmark.heading_bidirectional,
                    estimated_distance_m=estimated_distance,
                    correction_mode=correction_mode,
                    warp_start_step=(
                        warp_start_point if correction_mode == "warp" else None
                    ),
                    warp_span_m=warp_span if correction_mode == "warp" else None,
                )
            )

    return LandmarkCorrectionResult(
        trajectory=corrected,
        raw_trajectory=[list(point) for point in trajectory],
        corrections=tuple(corrections),
        detection_count=len(detections),
        discarded_count=discarded,
        rssi_threshold_dbm=rssi_threshold_dbm,
        data_path=data_path,
        detections=detections,
        landmarks=landmarks,
    )
