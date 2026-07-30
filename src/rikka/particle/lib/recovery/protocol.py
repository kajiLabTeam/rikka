"""recovery 戦略と逐次実行チェーンの契約。

役割:
    recovery 方法を独立した戦略として同じ順序で試せるようにする。
依存元:
    ``ParticleState`` のみを共有結果として使用する。
利用先:
    particle runner が local、checkpoint、fallback、hold の順に復旧する。
処理フロー:
    各戦略へ同じ状態を渡し、最初の成功結果を返す。
"""

from dataclasses import dataclass
from typing import Protocol

from ..state import ParticleState


@dataclass(frozen=True)
class RecoveryOutcome:
    """成功した recovery の状態と識別名。"""

    mode: str
    state: ParticleState


class RecoveryStrategy(Protocol):
    """1種類の recovery が満たす契約。"""

    def recover(self, state: ParticleState) -> RecoveryOutcome | None:
        """復旧できた場合だけ結果を返す。"""
        ...


@dataclass(frozen=True)
class RecoveryChain:
    """登録順に recovery を試す。"""

    strategies: tuple[RecoveryStrategy, ...]

    def recover(self, state: ParticleState) -> RecoveryOutcome | None:
        """最初に成功した戦略の結果を返す。"""
        for strategy in self.strategies:
            outcome = strategy.recover(state)
            if outcome is not None:
                return outcome
        return None
