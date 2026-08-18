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

from ...common.lib.models import (
    FloorMap,
    Landmark,
    LandmarkCorrection,
    LandmarkCorrectionResult,
    LandmarkDetection,
)
from ...landmark.lib.assignment import assign_detections_to_steps
from ...landmark.lib.coordinates import build_landmark_meter_map
from ...landmark.lib.timing import evaluate_landmark_timing


def apply_landmark_corrections(
    trajectory: list[list[float]],
    t_at_steps: list[float],
    *,
    detections: tuple[LandmarkDetection, ...],
    landmarks: tuple[Landmark, ...],
    floormap: FloorMap,
    data_path: str,
    rssi_threshold_dbm: float,
    gx_mean: float,
    gz_mean: float,
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
    corrected: list[list[float]] = [list(trajectory[0])]
    corrections: list[LandmarkCorrection] = []

    for step_index in range(len(trajectory) - 1):
        delta_x = trajectory[step_index + 1][0] - trajectory[step_index][0]
        delta_y = trajectory[step_index + 1][1] - trajectory[step_index][1]
        position_x = corrected[step_index][0] + delta_x
        position_y = corrected[step_index][1] + delta_y

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
            before_x, before_y = position_x, position_y
            timing = evaluate_landmark_timing(
                trajectory,
                t_at_steps,
                detection.timestamp_s,
                landmark_position,
            )
            if is_last:
                position_x, position_y = landmark_x, landmark_y
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
                    after_x=position_x,
                    after_y=position_y,
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
                )
            )
        corrected.append([position_x, position_y])

    return LandmarkCorrectionResult(
        trajectory=corrected,
        raw_trajectory=[list(point) for point in trajectory],
        corrections=tuple(corrections),
        detection_count=len(detections),
        discarded_count=discarded,
        rssi_threshold_dbm=rssi_threshold_dbm,
        data_path=data_path,
        detections=detections,
    )
