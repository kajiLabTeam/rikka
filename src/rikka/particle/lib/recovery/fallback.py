"""広域 fallback recovery。

役割:
    局所候補とcheckpoint再生が失敗した場合に、旋回を許す広域候補を生成する。
依存元:
    ``local`` の候補生成と実行時コンテキストを使用する。
利用先:
    ``resolve_map_constraints`` のfallback経路から呼び出される。
処理フロー:
    通常の局所探索と同じ入力・乱数列で、旋回候補を常に有効化して探索する。
"""

from ..state import ParticleRuntime
from .local import _generate_recovery_candidates, _RecoveryResult


def recover_fallback(ctx: ParticleRuntime) -> _RecoveryResult | None:
    """旋回候補を常に有効化し、広域候補を抽出する。"""
    return _generate_recovery_candidates(
        ctx.particles_before,
        ctx.heading_correction_before,
        ctx.heading_drift_before,
        ctx.stride_scale_before,
        ctx.proposed_motion_state,
        ctx.weights_before,
        ctx.particle_base_headings,
        ctx.particle_step_lengths,
        ctx.sigma_sl_ratio,
        ctx.n_particles,
        ctx.map_gray,
        ctx.gx_mean,
        ctx.gz_mean,
        ctx.origin_px,
        ctx.scale,
        ctx.recovery_heading_sigma,
        ctx.recovery_max_attempts,
        ctx.rng,
        allow_turn_candidates=True,
        preserve_route_branches=ctx.preserve_recovery_branches,
        allow_stride_adaptation=ctx.adaptive_recovery_scale,
        stride_scale_min=ctx.effective_stride_scale_min,
        stride_scale_max=ctx.effective_stride_scale_max,
        capture_candidates=ctx.recorder.stages_enabled,
    )
