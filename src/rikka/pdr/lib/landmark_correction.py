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

import warnings

import numpy as np

from ...common.lib.integrate import integrate_steps
from ...common.lib.models import (
    FloorMap,
    Landmark,
    LandmarkCorrection,
    LandmarkCorrectionResult,
    LandmarkObservation,
    LandmarkRange,
    RangingConsistency,
    StepHeading,
)
from ...landmark.lib.assignment import assign_detections_to_steps
from ...landmark.lib.coordinates import build_landmark_meter_map
from ...landmark.lib.retrofit import (
    SimilarityTransform,
    apply_transform,
    count_walkability_violations,
    damp_transform,
    solve_anchor_similarity,
)
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


def _warp_span(raw_trajectory: list[list[float]], start: int, endpoint: int) -> float:
    """warp対象区間の累積距離を返す。"""
    return float(
        sum(
            np.linalg.norm(
                np.asarray(raw_trajectory[index]) - raw_trajectory[index - 1]
            )
            for index in range(start + 1, endpoint + 1)
        )
    )


def _similarity_candidate(
    points: list[list[float]],
    transform: SimilarityTransform,
    *,
    anchor_index: int,
    endpoint_index: int,
    forward_mode: str,
) -> list[list[float]]:
    """相似変換候補をholdまたはfreezeの範囲へ適用する。"""
    end_index = None if forward_mode == "hold" else endpoint_index
    candidate = apply_transform(points, transform, anchor_index + 1, end_index)
    if forward_mode == "freeze":
        delta_x = candidate[endpoint_index][0] - points[endpoint_index][0]
        delta_y = candidate[endpoint_index][1] - points[endpoint_index][1]
        for point_index in range(endpoint_index + 1, len(candidate)):
            candidate[point_index][0] += delta_x
            candidate[point_index][1] += delta_y
    return candidate


def _rewrite_step_state(
    headings: list[StepHeading],
    lengths: list[float],
    transform: SimilarityTransform,
    *,
    anchor_index: int,
    endpoint_index: int,
    forward_mode: str,
) -> tuple[list[StepHeading], list[float]]:
    """点変換と等価になるよう確定方位と歩幅を書き換える。"""
    rewritten_headings = list(headings)
    rewritten_lengths = list(lengths)
    end_step = len(headings) if forward_mode == "hold" else endpoint_index
    for step_index in range(anchor_index, end_step):
        heading = rewritten_headings[step_index]
        if heading.selected_heading is None:  # pragma: no cover - 積分済み状態の防御
            raise RuntimeError("内部エラー: selected_heading が未確定です。")
        rewritten_headings[step_index] = heading._replace(
            selected_heading=heading.selected_heading + transform.rotation_rad,
            landmark_heading_offset=(
                heading.landmark_heading_offset + transform.rotation_rad
            ),
            landmark_length_scale=heading.landmark_length_scale * transform.scale,
        )
        rewritten_lengths[step_index] *= transform.scale
    return rewritten_headings, rewritten_lengths


def apply_landmark_corrections(
    trajectory: list[list[float]],
    t_at_steps: list[float],
    *,
    step_headings: list[StepHeading] | None = None,
    step_lengths: list[float] | None = None,
    detections: tuple[LandmarkObservation, ...],
    landmarks: tuple[Landmark, ...],
    floormap: FloorMap,
    data_path: str,
    rssi_threshold_dbm: float,
    gx_mean: float,
    gz_mean: float,
    correction_mode: str = "snap",
    max_correction_m: float = float("inf"),
    max_warp_span_m: float = float("inf"),
    retrofit_forward_mode: str = "hold",
    retrofit_max_heading_deg: float = 30.0,
    retrofit_stride_scale_min: float = 0.7,
    retrofit_stride_scale_max: float = 1.4,
    retrofit_min_span_m: float = 3.0,
    retrofit_map_check: str = "warn",
    retrofit_damp_factors: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25),
    map_gray: np.ndarray | None = None,
    ranging_consistency: tuple[RangingConsistency, ...] = (),
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
    if correction_mode not in {"snap", "warp", "similarity"}:
        raise ValueError(
            "correction_mode は snap、warp、similarity のいずれかを指定してください。"
        )
    if correction_mode == "similarity" and (
        step_headings is None or step_lengths is None
    ):
        raise ValueError("similarity には step_headings と step_lengths が必要です。")
    if retrofit_forward_mode not in {"hold", "freeze"}:
        raise ValueError(
            "retrofit_forward_mode は hold または freeze を指定してください。"
        )
    if retrofit_map_check not in {"off", "warn", "enforce"}:
        raise ValueError(
            "retrofit_map_check は off、warn、enforce を指定してください。"
        )
    corrected = [list(point) for point in trajectory]
    corrected_headings = None if step_headings is None else list(step_headings)
    corrected_lengths = None if step_lengths is None else list(step_lengths)
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
            target = _constraint_position(
                (before_x, before_y),
                (landmark_x, landmark_y),
                estimated_distance,
            )
            warp_span = None
            warp_start_point = last_constraint_point
            applied = False
            retrofit_rotation_deg = None
            retrofit_scale = None
            retrofit_damp_factor = None
            retrofit_map_violations = None
            retrofit_reject_reason = None
            if is_last:
                correction_distance = float(
                    np.linalg.norm(
                        np.asarray(target) - np.asarray((before_x, before_y))
                    )
                )
                candidate_span = _warp_span(
                    trajectory, warp_start_point, endpoint_index
                )
                span_exceeded = (
                    correction_mode == "warp" and candidate_span > max_warp_span_m
                )
                if correction_mode == "similarity":
                    transform = solve_anchor_similarity(
                        (
                            corrected[warp_start_point][0],
                            corrected[warp_start_point][1],
                        ),
                        (
                            corrected[endpoint_index][0],
                            corrected[endpoint_index][1],
                        ),
                        target,
                    )
                    if correction_distance > max_correction_m:
                        retrofit_reject_reason = "max_correction_exceeded"
                    elif transform is None:
                        retrofit_reject_reason = "degenerate_span"
                    else:
                        raw_span = float(
                            np.linalg.norm(
                                np.asarray(corrected[endpoint_index])
                                - np.asarray(corrected[warp_start_point])
                            )
                        )
                        retrofit_rotation_deg = float(
                            np.degrees(transform.rotation_rad)
                        )
                        retrofit_scale = transform.scale
                        if raw_span < retrofit_min_span_m:
                            retrofit_reject_reason = "span_too_short"
                        elif abs(retrofit_rotation_deg) > retrofit_max_heading_deg:
                            retrofit_reject_reason = "heading_exceeded"
                        elif not (
                            retrofit_stride_scale_min
                            <= retrofit_scale
                            <= retrofit_stride_scale_max
                        ):
                            retrofit_reject_reason = "stride_scale_exceeded"
                        elif retrofit_map_check == "enforce" and map_gray is None:
                            retrofit_reject_reason = "map_unavailable"
                        else:
                            factors = (
                                retrofit_damp_factors
                                if retrofit_map_check == "enforce"
                                else (1.0,)
                            )
                            accepted_transform = None
                            for factor in factors:
                                candidate_transform = damp_transform(transform, factor)
                                candidate = _similarity_candidate(
                                    corrected,
                                    candidate_transform,
                                    anchor_index=warp_start_point,
                                    endpoint_index=endpoint_index,
                                    forward_mode=retrofit_forward_mode,
                                )
                                violations = (
                                    None
                                    if map_gray is None or retrofit_map_check == "off"
                                    else count_walkability_violations(
                                        candidate,
                                        map_gray=map_gray,
                                        gx_mean=gx_mean,
                                        gz_mean=gz_mean,
                                        origin_px=floormap.origin_px,
                                        scale=floormap.scale,
                                    )
                                )
                                retrofit_map_violations = violations
                                if retrofit_map_check != "enforce" or violations == 0:
                                    accepted_transform = candidate_transform
                                    retrofit_damp_factor = factor
                                    break
                            if accepted_transform is None:
                                retrofit_reject_reason = "map_violation"
                            else:
                                if (
                                    corrected_headings is None
                                    or corrected_lengths is None
                                ):
                                    raise RuntimeError(
                                        "内部エラー: similarity の歩状態がありません。"
                                    )
                                corrected_headings, corrected_lengths = (
                                    _rewrite_step_state(
                                        corrected_headings,
                                        corrected_lengths,
                                        accepted_transform,
                                        anchor_index=warp_start_point,
                                        endpoint_index=endpoint_index,
                                        forward_mode=retrofit_forward_mode,
                                    )
                                )
                                corrected = integrate_steps(
                                    corrected_headings, corrected_lengths
                                )
                                last_constraint_point = endpoint_index
                                applied = True
                                if (
                                    retrofit_map_check == "warn"
                                    and retrofit_map_violations
                                ):
                                    warnings.warn(
                                        "BLE相似補正後の軌跡が"
                                        "歩行不可画素を横切ります: "
                                        f"{detection.beacon_id} "
                                        f"violations={retrofit_map_violations}",
                                        UserWarning,
                                        stacklevel=2,
                                    )
                elif correction_distance <= max_correction_m and not span_exceeded:
                    warp_span = _apply_translation(
                        corrected,
                        trajectory,
                        endpoint_index=endpoint_index,
                        target=target,
                        mode=correction_mode,
                        warp_start_point=warp_start_point,
                    )
                    last_constraint_point = endpoint_index
                    applied = True
                elif correction_mode == "warp":
                    warp_span = candidate_span
                if not applied:
                    if correction_mode == "similarity":
                        message = (
                            "BLE補正を棄却しました: "
                            f"{detection.beacon_id} "
                            f"correction={correction_distance:.3f}m "
                            f"warp_span={candidate_span:.3f}m "
                            f"reason={retrofit_reject_reason}"
                        )
                    else:
                        message = (
                            "BLE補正を上限超過のため棄却しました: "
                            f"{detection.beacon_id} "
                            f"correction={correction_distance:.3f}m "
                            f"warp_span={candidate_span:.3f}m"
                        )
                    warnings.warn(message, UserWarning, stacklevel=2)
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
                    applied=applied,
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
                    retrofit_rotation_deg=retrofit_rotation_deg,
                    retrofit_scale=retrofit_scale,
                    retrofit_damp_factor=retrofit_damp_factor,
                    retrofit_map_violations=retrofit_map_violations,
                    retrofit_reject_reason=retrofit_reject_reason,
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
        ranging_consistency=ranging_consistency,
        step_headings=corrected_headings,
        step_lengths=corrected_lengths,
    )
