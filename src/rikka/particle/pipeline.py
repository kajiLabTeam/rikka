"""PreparedPdrSteps を地図制約付き代表軌跡へ変換する pipeline。

役割:
    通常 PDR で確定した歩列と任意のランドマーク検出を受け、particle filter を
    実行して共有結果型へまとめる。
依存元:
    ``common`` の共有型・設定と既存互換 runner の数値実装を使用する。
利用先:
    CLI が ``pdr.pipeline.run_pdr`` の後にPF指定時だけ呼び出す。
処理フロー:
    診断収集器を用意し、準備済み歩列をrunnerへ渡して共有結果型へまとめる。
"""

from dataclasses import replace

from ..common.lib.models import (
    FloorMap,
    LandmarkCorrection,
    LandmarkCorrectionResult,
    LandmarkDetection,
    ParticleFilterResult,
    PreparedPdrSteps,
    TrajectoryResult,
)
from ..common.settings import BleLandmarkSettings, ParticleSettings
from ..landmark.lib.assignment import (
    assign_detections_to_steps,
    build_step_landmark_map,
)
from ..landmark.lib.coordinates import build_landmark_meter_map
from ..landmark.lib.timing import evaluate_landmark_timing
from .lib.recorder import (
    ParticleFilterStepDiagnostics,
    ParticlePathComparison,
    ParticleStepStages,
)
from .lib.runner import run_particle_steps


def _attach_landmark_timing(
    prepared: PreparedPdrSteps,
    floormap: FloorMap,
    landmark_settings: BleLandmarkSettings,
    detections: tuple[LandmarkDetection, ...],
    diagnostics: list[ParticleFilterStepDiagnostics],
    events: list[LandmarkCorrection],
) -> tuple[list[ParticleFilterStepDiagnostics], list[LandmarkCorrection]]:
    """PF診断と補正履歴へ共通の最接近時刻指標を付ける。"""
    landmark_meters = build_landmark_meter_map(
        landmark_settings.landmarks,
        prepared.gx_mean,
        prepared.gz_mean,
        floormap,
    )
    by_step = build_step_landmark_map(
        detections,
        prepared.t_at_steps,
        landmark_meters,
    )
    timing_by_detection: dict[LandmarkDetection, float] = {}
    for detection in detections:
        landmark_xy = landmark_meters.get(detection.beacon_id)
        if landmark_xy is None:
            continue
        timing_by_detection[detection] = evaluate_landmark_timing(
            prepared.trajectory,
            prepared.t_at_steps,
            detection.timestamp_s,
            landmark_xy,
        ).nearest_approach_delta_s

    updated_diagnostics: list[ParticleFilterStepDiagnostics] = []
    for diagnostic in diagnostics:
        step_detection = by_step.get(diagnostic.step)
        updated_diagnostics.append(
            replace(
                diagnostic,
                landmark_nearest_delta_s=timing_by_detection.get(step_detection),
            )
            if step_detection is not None
            else diagnostic
        )
    updated_events = [
        event._replace(
            detection_distance_m=(
                (event.before_x - event.landmark_x) ** 2
                + (event.before_y - event.landmark_y) ** 2
            )
            ** 0.5,
            nearest_approach_delta_s=timing_by_detection.get(
                LandmarkDetection(event.timestamp_s, event.beacon_id, event.rssi_dbm)
            ),
        )
        for event in events
    ]
    return updated_diagnostics, updated_events


def run_particle(
    prepared: PreparedPdrSteps,
    floormap: FloorMap,
    settings: ParticleSettings,
    *,
    detections: tuple[LandmarkDetection, ...] | None = None,
    landmark_settings: BleLandmarkSettings | None = None,
) -> TrajectoryResult:
    """準備済みの歩列へ地図拘束を適用する。"""
    diagnostics: list[ParticleFilterStepDiagnostics] = []
    stages: list[ParticleStepStages] = []
    path_comparisons: list[ParticlePathComparison] = []
    landmark_events: list[LandmarkCorrection] = []
    runtime_detections = () if detections is None else detections
    trajectory, lengths, times, all_particles, headings = run_particle_steps(
        prepared.gx_mean,
        prepared.gz_mean,
        floormap_path=floormap.path,
        origin_px=floormap.origin_px,
        scale=floormap.scale,
        n_particles=settings.count,
        prepared_step_headings=prepared.step_headings,
        prepared_step_lengths=prepared.step_lengths,
        prepared_step_times=prepared.t_at_steps,
        prepared_motion_evidences=prepared.motion_evidences,
        prepared_particle_motion_headings=prepared.particle_motion_headings,
        prepared_motion_posteriors=prepared.motion_posteriors,
        seed=settings.seed,
        motion_predictive_weight_power=settings.motion_predictive_weight_power,
        path_selection=settings.path_selection,
        diagnostics_collector=diagnostics,
        stage_collector=stages,
        path_comparison_collector=path_comparisons,
        landmark_detections=runtime_detections,
        landmarks=(() if landmark_settings is None else landmark_settings.landmarks),
        landmark_mode=settings.landmark_mode,
        landmark_sigma_m=settings.landmark_sigma_m,
        landmark_likelihood_floor=settings.landmark_likelihood_floor,
        landmark_reset_sigma_m=settings.landmark_reset_sigma_m,
        landmark_max_jump_m=settings.landmark_max_jump_m,
        landmark_reset_spread_ratio=settings.landmark_reset_spread_ratio,
        landmark_reset_min_distance_m=settings.landmark_reset_min_distance_m,
        landmark_reset_heading_sigma=settings.landmark_reset_heading_sigma,
        landmark_anchor_warn_jump_m=settings.landmark_anchor_warn_jump_m,
        landmark_events_collector=landmark_events,
    )
    if detections is not None and landmark_settings is not None:
        diagnostics, landmark_events = _attach_landmark_timing(
            prepared,
            floormap,
            landmark_settings,
            detections,
            diagnostics,
            landmark_events,
        )
    particle = ParticleFilterResult(
        trajectory=trajectory,
        all_particles=[item for item in all_particles],
        diagnostics=tuple(diagnostics),
        stages=tuple(stages),
        path_comparisons=tuple(path_comparisons),
    )
    landmark = None
    if detections is not None and landmark_settings is not None:
        _, discarded_count = assign_detections_to_steps(
            detections,
            prepared.t_at_steps,
        )
        landmark = LandmarkCorrectionResult(
            trajectory=trajectory,
            raw_trajectory=[list(point) for point in prepared.trajectory],
            corrections=tuple(landmark_events),
            detection_count=len(detections),
            discarded_count=discarded_count,
            rssi_threshold_dbm=landmark_settings.rssi_threshold_dbm,
            data_path=str(landmark_settings.data_path),
            detections=detections,
        )
    return TrajectoryResult(
        trajectory=trajectory,
        step_lengths=lengths,
        t_at_steps=times,
        step_headings=headings,
        prepared=prepared,
        particle=particle,
        landmark=landmark,
    )
