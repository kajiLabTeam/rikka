"""広域 fallback と位置保持 recovery。

役割:
    前段戦略が失敗した場合の最終候補生成と安全な位置保持を提供する。
依存元:
    ``protocol`` の共通結果と ``ParticleState`` を使用する。
利用先:
    recovery chain の末尾から呼び出す。
処理フロー:
    callback による広域探索を試し、最後は入力状態をそのまま保持する。
"""

from collections.abc import Callable
from dataclasses import dataclass

from ..state import ParticleState
from .protocol import RecoveryOutcome


@dataclass(frozen=True)
class FallbackRecovery:
    """広域候補 callback を戦略へ適合させる。"""

    callback: Callable[[ParticleState], ParticleState | None]

    def recover(self, state: ParticleState) -> RecoveryOutcome | None:
        """広域探索に成功した場合だけ結果を返す。"""
        recovered = self.callback(state)
        if recovered is None:
            return None
        return RecoveryOutcome("fallback", recovered)


@dataclass(frozen=True)
class HoldPosition:
    """粒子を動かさずに失敗を明示する最終戦略。"""

    def recover(self, state: ParticleState) -> RecoveryOutcome:
        """入力状態をそのまま保持する。"""
        return RecoveryOutcome("failed_hold", state)
