"""particle filter のresolve。

役割:
    resolveを独立した段階として実装する。
依存元:
    common の共有型・設定と particle/lib の部品、ParticleRuntime を利用する。
利用先:
    particle/lib/runner が元の実行順序どおりに呼び出す。
処理フロー:
    通常更新またはrecovery結果を確定する。
"""

from .state import ParticleRuntime


def resolve(ctx: ParticleRuntime) -> None:
    ctx.position_history.append(ctx.particles.copy())
