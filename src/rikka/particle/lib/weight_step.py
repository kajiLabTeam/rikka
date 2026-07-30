"""particle filter のweight_step。

役割:
    weight_stepを独立した段階として実装する。
依存元:
    common の共有型・設定と particle/lib の部品、ParticleRuntime を利用する。
利用先:
    particle/lib/runner が元の実行順序どおりに呼び出す。
処理フロー:
    観測尤度と遷移結果から重みを更新する。
"""

from ...particle.lib.resampling import (
    _effective_sample_size,
)
from .state import ParticleRuntime


def weight_step(ctx: ParticleRuntime) -> None:
    ctx.ess_after_resampling = _effective_sample_size(ctx.weights)
