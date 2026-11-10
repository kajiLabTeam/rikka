"""フロアマップの読み込みと初期粒子の生成。1歩目の分布をここで決める。

役割:
    地図画像を歩行可否のグレースケールへ正規化し、粒子の初期位置・初期方位・
    歩幅倍率・重みと、各種履歴リストを作る。
依存元:
    Matplotlib の画像読み込み、``landmark/lib`` の歩割り当てと座標変換、
    ``particle/lib`` の地図判定・運動モデル・一様初期化、``ParticleRuntime``。
利用先:
    ``particle.lib.runner`` が ``setup`` の直後に1回だけ呼び出す。
処理フロー:
    初期分布は ``init_mode`` で分かれる。

    - ``origin``: 起点 (0, 0) に全粒子を置き、地図上の起点画素を検証する。
    - ``uniform``: 歩行可能画素の面積に比例して粒子を一様配置し、初期方位も
      ``[-pi, pi)`` から一様に取る。起点画素の検証は行わない。

    初期方位のばらつきは、減衰しない ``heading_correction`` へ入れる。
    毎歩0へ引き戻される ``heading_drift`` へ入れると数歩で全粒子が同じ方位へ
    collapse するためである（関数内のコメントに詳細）。
"""

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from ...common.lib.models import FloorMap
from ...landmark.lib.assignment import (
    build_step_landmark_map,
    build_step_observation_map,
)
from ...landmark.lib.coordinates import build_landmark_meter_map
from ...particle.lib.map_constraints import (
    _normalize_floormap_gray,
    _validate_floormap_origin,
    _validate_floormap_shape,
)
from ...particle.lib.motion_model import _MOTION_FORWARD
from ...particle.lib.recorder import ParticleRecorder
from .global_init import (
    sample_uniform_headings,
    sample_uniform_walkable_particles,
    walkable_area_m2,
)
from .hard_anchor import initialize_hard_positions
from .landmark import meter_walkable_mask
from .state import ParticleRuntime


def _adaptive_heading_rejuvenation_sigma(
    base_sigma: float,
    motion_reliability: float,
) -> float:
    """高信頼な移動方位観測がある記録では再標本化ノイズを弱める。"""
    reliability = (
        float(np.clip(motion_reliability, 0.0, 1.0))
        if np.isfinite(motion_reliability)
        else 0.0
    )
    high_reliability_excess = max(0.0, reliability - 0.90)
    high_reliability_scale = 4.0 / 9.0
    scale = float(
        np.clip(
            1.0 - (1.0 - high_reliability_scale) * high_reliability_excess / 0.02,
            high_reliability_scale,
            1.0,
        )
    )
    return base_sigma * scale


def initialize(ctx: ParticleRuntime) -> None:
    """地図を読み込み、初期粒子・状態・履歴を ``ctx`` へ構築する。"""
    ctx.map_gray = _normalize_floormap_gray(mpimg.imread(Path(ctx.floormap_path)))
    _validate_floormap_shape(ctx.map_gray)
    ctx.global_phase_active = ctx.init_mode == "uniform"
    if not ctx.global_phase_active:
        _validate_floormap_origin(ctx.map_gray, ctx.origin_px)
    if ctx.initial_particles is not None:
        ctx.particles = np.asarray(ctx.initial_particles, dtype=float).copy()
        if ctx.particles.shape != (ctx.n_particles, 2):
            raise ValueError("initial_particles の形状は (n_particles, 2) が必要です")
    elif ctx.global_phase_active:
        ctx.particles = sample_uniform_walkable_particles(
            ctx.map_gray,
            ctx.n_particles,
            ctx.gx_mean,
            ctx.gz_mean,
            ctx.origin_px,
            ctx.scale,
            ctx.rng,
        )
    else:
        ctx.particles = np.zeros((ctx.n_particles, 2))
    if ctx.initial_heading_correction is not None:
        ctx.heading_correction = np.asarray(
            ctx.initial_heading_correction, dtype=float
        ).copy()
        if ctx.heading_correction.shape != (ctx.n_particles,):
            raise ValueError(
                "initial_heading_correction の形状は (n_particles,) が必要です"
            )
    elif ctx.global_phase_active:
        # 初期方位の不確かさは減衰しない補正項へ入れる。heading_drift は毎歩
        # PF_HEADING_DRIFT_RETENTION で0へ引き戻されるため、一様な初期方位をそこへ
        # 入れると数歩で全粒子が同じ方位へ collapse し、地図制約で全滅する。
        ctx.heading_correction = sample_uniform_headings(ctx.n_particles, ctx.rng)
    else:
        ctx.heading_correction = np.zeros(ctx.n_particles, dtype=float)
    if ctx.initial_heading_drift is not None:
        ctx.heading_drift = np.asarray(ctx.initial_heading_drift, dtype=float).copy()
        if ctx.heading_drift.shape != (ctx.n_particles,):
            raise ValueError("initial_heading_drift の形状は (n_particles,) が必要です")
    else:
        ctx.heading_drift = ctx.rng.normal(0, ctx.sigma_init_heading, ctx.n_particles)
    if ctx.global_phase_active:
        area = walkable_area_m2(ctx.map_gray, ctx.scale)
        print(
            f"一様初期化: 歩行可能面積 {area:.3f} m^2、粒子数 {ctx.n_particles}、"
            f"1粒子あたり {area / ctx.n_particles:.6f} m^2/粒子"
        )
    ctx.motion_state = np.full(ctx.n_particles, _MOTION_FORWARD, dtype=np.int8)
    ctx.stride_scale = np.clip(
        ctx.stride_scale_prior_mean
        + ctx.stride_rng.normal(0.0, ctx.effective_stride_init_sigma, ctx.n_particles),
        ctx.effective_stride_scale_min,
        ctx.effective_stride_scale_max,
    )
    ctx.weights = np.ones(ctx.n_particles) / ctx.n_particles
    ctx.step_lengths = []
    ctx.t_at_steps = []
    ctx.position_history = [ctx.particles.copy()]
    ctx.heading_correction_history = [ctx.heading_correction.copy()]
    ctx.heading_drift_history = [ctx.heading_drift.copy()]
    ctx.motion_state_history = [ctx.motion_state.copy()]
    ctx.stride_scale_history = [ctx.stride_scale.copy()]
    ctx.weight_history = [ctx.weights.copy()]
    ctx.path_log_score_history = [np.zeros(ctx.n_particles, dtype=float)]
    ctx.parent_history = []
    ctx.all_particles_list = [ctx.particles.copy()]
    ctx.step_headings = []
    ctx.healthy_checkpoint_steps = [0]
    ctx.recorder = ParticleRecorder(
        ctx.diagnostics_collector, ctx.stage_collector, ctx.path_comparison_collector
    )
    ctx.diagnostics_start_index = len(ctx.recorder.diagnostics)
    ctx.stages_start_index = len(ctx.recorder.stages)
    if (
        ctx.prepared_step_headings is None
        or ctx.prepared_step_lengths is None
        or ctx.prepared_step_times is None
    ):
        raise ValueError(
            "prepared_step_headings, prepared_step_lengths, "
            "prepared_step_times は必須です"
        )
    if ctx.prepared_motion_evidences is None:
        raise ValueError("prepared_motion_evidences は必須です")
    if ctx.prepared_particle_motion_headings is None:
        raise ValueError("prepared_particle_motion_headings は必須です")
    if (
        not len(ctx.prepared_step_headings)
        == len(ctx.prepared_step_lengths)
        == len(ctx.prepared_step_times)
    ):
        raise ValueError(
            "prepared_step_headings, prepared_step_lengths, "
            "prepared_step_times の長さが一致しません"
        )
    ctx.stabilized_step_headings = ctx.prepared_step_headings
    ctx.raw_step_lengths = ctx.prepared_step_lengths
    ctx.raw_step_times = ctx.prepared_step_times
    ctx.motion_evidences = ctx.prepared_motion_evidences
    ctx.particle_motion_headings = ctx.prepared_particle_motion_headings
    if len(ctx.motion_evidences) != len(ctx.stabilized_step_headings):
        raise ValueError(
            "prepared_motion_evidences と prepared_step_headings の長さが一致しません"
        )
    if len(ctx.particle_motion_headings) != len(ctx.stabilized_step_headings):
        raise ValueError(
            "prepared_particle_motion_headings と "
            "prepared_step_headings の長さが一致しません"
        )
    if ctx.prepared_motion_posteriors is not None and len(
        ctx.prepared_motion_posteriors
    ) != len(ctx.stabilized_step_headings):
        raise ValueError(
            "prepared_motion_posteriors と prepared_step_headings の長さが一致しません"
        )
    ctx.recording_motion_reliability = (
        float(
            np.median(
                [evidence.motion_reliability for evidence in ctx.motion_evidences]
            )
        )
        if ctx.motion_evidences
        else 0.0
    )
    ctx.effective_heading_rejuvenation_sigma = _adaptive_heading_rejuvenation_sigma(
        ctx.rejuvenation_sigma_heading, ctx.recording_motion_reliability
    )
    ctx.landmark_meters = build_landmark_meter_map(
        ctx.landmarks,
        ctx.gx_mean,
        ctx.gz_mean,
        FloorMap(str(ctx.floormap_path), ctx.origin_px, ctx.scale),
    )
    ctx.landmark_definitions = {item.beacon_id: item for item in ctx.landmarks}
    ctx.landmark_by_step = build_step_landmark_map(
        ctx.landmark_detections,
        ctx.raw_step_times,
        ctx.landmark_meters,
    )
    ctx.landmark_observations_by_step = build_step_observation_map(
        ctx.landmark_ranging_observations,
        ctx.raw_step_times,
        ctx.landmark_meters,
    )
    ctx.landmark_events = (
        [] if ctx.landmark_events_collector is None else ctx.landmark_events_collector
    )
    initialize_hard_positions(ctx)
    ctx.landmark_anchor_steps = set()
    ctx.landmark_reset_steps = set()
    if ctx.landmark_mode != "none":
        for beacon_id, landmark_xy in ctx.landmark_meters.items():
            walkable = meter_walkable_mask(
                np.asarray([landmark_xy], dtype=float),
                ctx.map_gray,
                ctx.gx_mean,
                ctx.gz_mean,
                ctx.origin_px,
                ctx.scale,
            )
            if not bool(walkable[0]):
                print(f"警告: ランドマーク {beacon_id} が歩行不可画素にあります。")
