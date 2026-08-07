"""particle filter の初期状態生成。

役割:
    初期状態生成を独立した段階として実装する。
依存元:
    common の共有型・設定と particle/lib の部品、ParticleRuntime を利用する。
利用先:
    particle/lib/runner が元の実行順序どおりに呼び出す。
処理フロー:
    地図と粒子状態、履歴、準備済み歩列を初期化する。
"""

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from ...particle.lib.map_constraints import (
    _normalize_floormap_gray,
    _validate_floormap_origin,
    _validate_floormap_shape,
)
from ...particle.lib.proposal import (
    _MOTION_FORWARD,
)
from ...particle.lib.recorder import (
    ParticleRecorder,
)
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
    ctx.map_gray = _normalize_floormap_gray(mpimg.imread(Path(ctx.floormap_path)))
    _validate_floormap_shape(ctx.map_gray)
    _validate_floormap_origin(ctx.map_gray, ctx.origin_px)
    ctx.particles = np.zeros((ctx.n_particles, 2))
    ctx.heading_correction = np.zeros(ctx.n_particles, dtype=float)
    ctx.heading_drift = ctx.rng.normal(0, ctx.sigma_init_heading, ctx.n_particles)
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
