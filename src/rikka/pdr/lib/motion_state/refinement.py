"""区間復号と動的身体方位を既存の横歩き平滑化へ統合する。

役割:
    歩単位の運動観測を区間復号し、端末と身体の時変方位差を推定した後、
    世界移動ベクトルを変えずに身体軸成分と横歩き区間を再評価する。
依存元:
    ``body_heading`` から動的方位推定、``motion_decoder`` から区間復号、
    ``heading`` から移動分類、``sidestep`` から既存平滑化を取得する。
利用先:
    ``trajectory`` が生の ``StepHeading`` 列を軌跡用に確定する直前に使用し、
    通常 PDR と particle filter で同じ結果を共有する。
処理フロー:
    初回復号、既存clusterとの合意、動的身体方位推定、体軸再射影、
    再復号を順に行い、高信頼区間の補完と低校正時の偽陽性抑制だけを適用する。
"""

import numpy as np

from ....common.lib.models import StepHeading
from ..heading.body import estimate_dynamic_body_headings
from ..heading.resolver import classify_movement_type
from .decoder import MotionDecodeResult, decode_step_motion_segments
from .step_motion import build_step_motion_observations, smooth_step_headings

_MIN_TRUSTED_CALIBRATION = 0.45


def _is_side_mode(mode: str | None) -> bool:
    """横歩き状態かどうかを返す。"""
    return mode in {"sidestep_left", "sidestep_right"}


def _trajectory_side_mode(heading: StepHeading) -> str | None:
    """既存軌跡分類から旋回付きを除いた横歩き方向を返す。"""
    movement = heading.trajectory_movement_type or heading.movement_type
    if movement.endswith("sidestep_left"):
        return "sidestep_left"
    if movement.endswith("sidestep_right"):
        return "sidestep_right"
    return None


def _decoder_confidences(result: MotionDecodeResult) -> tuple[float, ...]:
    """区間診断を歩ごとの信頼度列へ展開する。"""
    confidences = [0.0] * len(result.motion_modes)
    for segment in result.segments:
        confidences[segment.start_index : segment.end_index] = [segment.confidence] * (
            segment.end_index - segment.start_index
        )
    return tuple(confidences)


def _reproject_to_body_heading(
    heading: StepHeading,
    body_heading: float | None,
) -> StepHeading:
    """世界移動ベクトルを新しい身体前後・左右軸へ再射影する。"""
    if body_heading is None or heading.motion_heading is None:
        return heading._replace(body_heading=body_heading)
    displacement = float(
        np.hypot(
            heading.forward_displacement or 0.0,
            heading.lateral_displacement or 0.0,
        )
    )
    relative_heading = heading.motion_heading - body_heading
    forward = displacement * float(np.cos(relative_heading))
    lateral = displacement * float(np.sin(relative_heading))
    movement = classify_movement_type(
        forward,
        lateral,
        heading.yaw_delta,
        heading.sidestep_lateral_ratio,
        heading.sidestep_min_lateral_displacement,
    )
    return heading._replace(
        body_heading=body_heading,
        forward_displacement=forward,
        lateral_displacement=lateral,
        movement_type=movement,
        trajectory_movement_type=None,
        sidestep_cluster_id=None,
    )


def _offset_motion_modes(
    headings: list[StepHeading],
    decoded: MotionDecodeResult,
    calibration: float,
) -> tuple[str, ...]:
    """動的方位更新に使う保守的な移動状態列を返す。"""
    modes: list[str] = []
    for heading, decoded_mode in zip(headings, decoded.motion_modes, strict=True):
        raw_movement = heading.movement_type
        legacy_side = _trajectory_side_mode(heading)
        if raw_movement == "turning" or raw_movement.startswith("turning_"):
            modes.append("turning")
        elif _is_side_mode(decoded_mode):
            modes.append(decoded_mode)
        elif calibration >= _MIN_TRUSTED_CALIBRATION and legacy_side is not None:
            modes.append(legacy_side)
        else:
            modes.append("forward")
    return tuple(modes)


def _apply_decoded_segments(
    headings: list[StepHeading],
    legacy_headings: list[StepHeading],
    decoded: MotionDecodeResult,
    calibration: float,
) -> list[StepHeading]:
    """高信頼区間を補完し、低校正時の明らかな偽陽性を抑制する。"""
    refined = list(headings)
    confidences = _decoder_confidences(decoded)
    next_cluster_id = max(
        (heading.sidestep_cluster_id or 0 for heading in headings),
        default=0,
    )
    segment_cluster_ids: dict[int, int] = {}
    for segment_index, segment in enumerate(decoded.segments):
        if _is_side_mode(segment.motion_mode):
            next_cluster_id += 1
            segment_cluster_ids[segment_index] = next_cluster_id

    segment_by_step: dict[int, tuple[int, str]] = {}
    for segment_index, segment in enumerate(decoded.segments):
        for index in range(segment.start_index, segment.end_index):
            segment_by_step[index] = (segment_index, segment.motion_mode)

    for index, (heading, legacy_heading) in enumerate(
        zip(headings, legacy_headings, strict=True)
    ):
        segment_index, decoded_mode = segment_by_step[index]
        raw_turning = heading.movement_type == "turning" or (
            heading.movement_type.startswith("turning_")
        )
        legacy_side = _trajectory_side_mode(heading)
        trajectory_type = heading.trajectory_movement_type or heading.movement_type
        cluster_id = heading.sidestep_cluster_id
        evidence_reason = heading.sidestep_evidence_reason
        if _is_side_mode(decoded_mode):
            trajectory_type = f"turning_{decoded_mode}" if raw_turning else decoded_mode
            cluster_id = segment_cluster_ids[segment_index]
            evidence_reason = "motion_segment_decoder"
        elif (
            calibration >= _MIN_TRUSTED_CALIBRATION
            and _trajectory_side_mode(legacy_heading) is not None
        ):
            trajectory_type = (
                legacy_heading.trajectory_movement_type or legacy_heading.movement_type
            )
            cluster_id = legacy_heading.sidestep_cluster_id
            evidence_reason = legacy_heading.sidestep_evidence_reason
        elif calibration < _MIN_TRUSTED_CALIBRATION and legacy_side is not None:
            trajectory_type = "turning" if raw_turning else "forward"
            cluster_id = None
            evidence_reason = "low_calibration_decoder_forward"
        refined[index] = heading._replace(
            trajectory_movement_type=trajectory_type,
            sidestep_cluster_id=cluster_id,
            sidestep_evidence_reason=evidence_reason,
            decoded_motion_mode=decoded_mode,
            decoded_motion_confidence=confidences[index],
        )
    return refined


def refine_step_headings_with_motion_model(
    raw_step_headings: list[StepHeading],
    sidestep_smoothing: str,
    sidestep_suspect_mode: str,
) -> list[StepHeading]:
    """区間復号と動的身体方位を統合した軌跡前のステップ列を返す。"""
    if not raw_step_headings:
        return []
    raw_observations = build_step_motion_observations(raw_step_headings)
    initial_decoded = decode_step_motion_segments(raw_observations)
    legacy = smooth_step_headings(
        raw_step_headings,
        sidestep_smoothing,
        sidestep_suspect_mode,
    )
    calibration = raw_observations[0].calibration_reliability
    estimates = estimate_dynamic_body_headings(
        raw_observations,
        _offset_motion_modes(legacy, initial_decoded, calibration),
    )
    reprojected = [
        _reproject_to_body_heading(heading, estimate.body_heading)._replace(
            device_body_offset=estimate.device_body_offset,
            dynamic_body_heading_confidence=estimate.confidence,
            body_heading_update_reason=estimate.reason,
        )
        for heading, estimate in zip(raw_step_headings, estimates, strict=True)
    ]
    reprojected_observations = build_step_motion_observations(reprojected)
    decoded = decode_step_motion_segments(reprojected_observations)
    smoothed = smooth_step_headings(
        reprojected,
        sidestep_smoothing,
        sidestep_suspect_mode,
    )
    return _apply_decoded_segments(smoothed, legacy, decoded, calibration)


__all__ = ["refine_step_headings_with_motion_model"]
