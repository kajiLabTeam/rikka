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
    明示指定時は、前向き結果を維持したまま粒子履歴から別の経路候補を作る。
    後方検証では共有歩列を反転して独立に実行し、元の歩番号で距離を比較する。
"""

from dataclasses import replace

import numpy as np

from ..ble.lib.ranging_check import run_ranging_preflight
from ..common.lib.models import (
    BackwardParticleResult,
    BidirectionalAgreement,
    FloorMap,
    HardAnchorCorrection,
    HardPositionObservation,
    LandmarkCorrection,
    LandmarkCorrectionResult,
    LandmarkObservation,
    LandmarkRange,
    ParticleConvergenceDiagnostic,
    ParticleFilterResult,
    PreparedPdrSteps,
    RangingConsistency,
    TrajectoryResult,
)
from ..common.settings import BleLandmarkSettings, ParticleSettings
from ..landmark.lib.assignment import (
    assign_detections_to_steps,
    build_step_landmark_map,
)
from ..landmark.lib.coordinates import build_landmark_meter_map
from ..landmark.lib.timing import evaluate_landmark_timing
from .lib.backward import (
    backward_to_forward,
    build_backward_prepared,
    reverse_observation_times,
    reverse_step_time,
)
from .lib.history_correction import build_history_candidate
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
    detections: tuple[LandmarkObservation, ...],
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
    timing_by_detection: dict[tuple[float, str, float], float] = {}
    for detection in detections:
        landmark_xy = landmark_meters.get(detection.beacon_id)
        if landmark_xy is None:
            continue
        timing_by_detection[
            (detection.timestamp_s, detection.beacon_id, detection.rssi_dbm)
        ] = evaluate_landmark_timing(
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
                landmark_nearest_delta_s=timing_by_detection.get(
                    (
                        step_detection.timestamp_s,
                        step_detection.beacon_id,
                        step_detection.rssi_dbm,
                    )
                ),
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
                (event.timestamp_s, event.beacon_id, event.rssi_dbm)
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
    detections: tuple[LandmarkObservation, ...] | None = None,
    ranging_observations: tuple[LandmarkRange, ...] | None = None,
    landmark_settings: BleLandmarkSettings | None = None,
    history_candidate: bool = False,
    hard_positions: tuple[HardPositionObservation, ...] = (),
) -> TrajectoryResult:
    """準備済みの歩列へ地図拘束を適用する。"""
    diagnostics: list[ParticleFilterStepDiagnostics] = []
    stages: list[ParticleStepStages] = []
    path_comparisons: list[ParticlePathComparison] = []
    landmark_events: list[LandmarkCorrection] = []
    hard_events: list[HardAnchorCorrection] = []
    convergence_diagnostics: list[ParticleConvergenceDiagnostic] = []
    runtime_detections = () if detections is None else detections
    runtime_ranging_observations = (
        tuple(item for item in runtime_detections if isinstance(item, LandmarkRange))
        if ranging_observations is None
        else ranging_observations
    )
    consistency: tuple[RangingConsistency, ...] = ()
    if detections is not None and landmark_settings is not None:
        consistency = run_ranging_preflight(
            landmark_settings,
            floormap,
            prepared.trajectory,
            prepared.t_at_steps,
            prepared.gx_mean,
            prepared.gz_mean,
        )
    candidate_count = settings.candidate_count
    assert candidate_count is not None  # 設定構築時に方式別の整数へ解決済み。
    trajectory, lengths, times, all_particles, headings = run_particle_steps(
        prepared.gx_mean,
        prepared.gz_mean,
        floormap_path=floormap.path,
        init_mode=settings.init_mode,
        convergence_cell_m=settings.convergence_cell_m,
        convergence_top_k=settings.convergence_top_k,
        convergence_weight_ratio=settings.convergence_weight_ratio,
        convergence_spread_m=settings.convergence_spread_m,
        convergence_single_steps=settings.convergence_single_steps,
        convergence_restore_normal=settings.convergence_restore_normal,
        convergence_collector=convergence_diagnostics,
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
        landmark_ranging_observations=runtime_ranging_observations,
        landmarks=(() if landmark_settings is None else landmark_settings.landmarks),
        landmark_mode=settings.landmark_mode,
        landmark_sigma_m=settings.landmark_sigma_m,
        landmark_likelihood_floor=settings.landmark_likelihood_floor,
        landmark_range_weight_power=settings.landmark_range_weight_power,
        landmark_reset_sigma_m=settings.landmark_reset_sigma_m,
        landmark_max_jump_m=settings.landmark_max_jump_m,
        landmark_reset_spread_ratio=settings.landmark_reset_spread_ratio,
        landmark_reset_min_distance_m=settings.landmark_reset_min_distance_m,
        landmark_reset_heading_sigma=settings.landmark_reset_heading_sigma,
        landmark_anchor_warn_jump_m=settings.landmark_anchor_warn_jump_m,
        landmark_retrofit=settings.landmark_retrofit,
        landmark_events_collector=landmark_events,
        proposal_method=settings.proposal_method,
        candidate_count=candidate_count,
        hard_positions=hard_positions,
        floormap_floor_id=floormap.floor_id,
        hard_anchor_events_collector=hard_events,
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
    uncorrected_trajectory = [list(point) for point in trajectory]
    for event in hard_events:
        uncorrected_trajectory[event.step_index] = [event.before_x, event.before_y]
    particle = ParticleFilterResult(
        hard_anchor_corrections=tuple(hard_events),
        uncorrected_trajectory=uncorrected_trajectory if hard_events else None,
        correction_steps=tuple(event.step_index for event in hard_events),
        trajectory=trajectory,
        all_particles=[item for item in all_particles],
        diagnostics=tuple(diagnostics),
        convergence_diagnostics=tuple(convergence_diagnostics),
        convergence=next(
            (
                item
                for item in convergence_diagnostics
                if item.convergence_step == item.step_number
            ),
            None,
        ),
        stages=tuple(stages),
        path_comparisons=tuple(path_comparisons),
        history_candidate=(
            build_history_candidate(
                trajectory,
                stages,
                floormap,
                prepared.gx_mean,
                prepared.gz_mean,
                landmark_constraints=any(event.applied for event in landmark_events),
            )
            if history_candidate and not hard_events
            else None
        ),
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
            landmarks=landmark_settings.landmarks,
            ranging_consistency=consistency,
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


def run_backward_particle(
    prepared: PreparedPdrSteps,
    floormap: FloorMap,
    settings: ParticleSettings,
    *,
    detections: tuple[LandmarkObservation, ...] | None = None,
    ranging_observations: tuple[LandmarkRange, ...] | None = None,
    landmark_settings: BleLandmarkSettings | None = None,
    history_candidate: bool = False,
    hard_positions: tuple[HardPositionObservation, ...] = (),
    forward_result: TrajectoryResult | None = None,
) -> BackwardParticleResult:
    """後方PFを独立に実行し、元の歩番号で前向き経路との距離を返す。

    前向き結果を省略した直接呼び出しでは、同じ引数で前向きPFも実行する。
    CLIは既存結果を渡し、前向きPFの重複実行と追加の乱数消費を避ける。
    収束診断の歩番号・時刻は後方実行のまま保持する。
    """
    if forward_result is None:
        forward_result = run_particle(
            prepared,
            floormap,
            settings,
            detections=detections,
            ranging_observations=ranging_observations,
            landmark_settings=landmark_settings,
            history_candidate=history_candidate,
            hard_positions=hard_positions,
        )
    first = prepared.t_at_steps[0] if prepared.t_at_steps else 0.0
    last = prepared.t_at_steps[-1] if prepared.t_at_steps else 0.0
    result = run_particle(
        build_backward_prepared(prepared),
        floormap,
        settings,
        detections=(
            None
            if detections is None
            else reverse_observation_times(detections, first, last)
        ),
        ranging_observations=(
            None
            if ranging_observations is None
            else reverse_observation_times(ranging_observations, first, last)
        ),
        landmark_settings=landmark_settings,
        history_candidate=history_candidate,
        hard_positions=tuple(
            sorted(
                (
                    replace(
                        obs,
                        timestamp_s=reverse_step_time(obs.timestamp_s, first, last),
                        heading_deg=(
                            None
                            if obs.heading_deg is None
                            else (obs.heading_deg + 360.0) % 360.0 - 180.0
                        ),
                    )
                    for obs in hard_positions
                ),
                key=lambda obs: obs.timestamp_s,
            )
        ),
    )
    trajectory = backward_to_forward(result.trajectory)
    # 実行歩だけが履歴に積まれる。逆順の変位を戻すと同じノード列になる。
    # 例: 有効歩が1,3なら、後方の歩1,3の終点は前向きノード1,0に対応する。
    step_numbers = [0] + [
        i
        for i, heading in enumerate(prepared.particle_motion_headings, 1)
        if heading is not None
    ]
    agreement = tuple(
        BidirectionalAgreement(
            step,
            float(f[0]),
            float(f[1]),
            float(b[0]),
            float(b[1]),
            float(np.hypot(f[0] - b[0], f[1] - b[1])),
        )
        for step, f, b in zip(
            step_numbers, forward_result.trajectory, trajectory, strict=True
        )
    )
    distances = [item.distance_m for item in agreement]
    assert result.particle is not None
    return BackwardParticleResult(
        trajectory=trajectory,
        convergence=result.particle.convergence,
        agreement=agreement,
        start_distance_m=distances[0],
        median_distance_m=float(np.median(distances)),
        max_distance_m=max(distances),
    )
