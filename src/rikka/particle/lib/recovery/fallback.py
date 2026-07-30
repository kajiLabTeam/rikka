"""広域 fallback と位置保持 recovery。

役割:
    前段戦略が失敗した場合の最終候補生成と安全な位置保持を提供する。
依存元:
    ``local`` の候補生成、実行時コンテキスト、``ParticleState`` を使用する。
利用先:
    recovery chain の末尾から呼び出す。
処理フロー:
    旋回候補を含む広域探索を実行し、最後は入力状態をそのまま保持する。
"""

from dataclasses import dataclass

from ..state import ParticleRuntime, ParticleState
from .local import _generate_recovery_candidates, _RecoveryResult
from .protocol import RecoveryOutcome


@dataclass(frozen=True)
class FallbackRecovery:
    """旋回候補を常に有効化して広域復旧を実行する。"""

    def recover(self, ctx: ParticleRuntime) -> _RecoveryResult | None:
        """通常の局所探索と同じ乱数列で広域候補を抽出する。"""
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


def recover_fallback(ctx: ParticleRuntime) -> _RecoveryResult | None:
    """広域 fallback 戦略を実行する。"""
    return FallbackRecovery().recover(ctx)


@dataclass(frozen=True)
class HoldPosition:
    """粒子を動かさずに失敗を明示する最終戦略。"""

    def recover(self, state: ParticleState) -> RecoveryOutcome:
        """入力状態をそのまま保持する。"""
        return RecoveryOutcome("failed_hold", state)
