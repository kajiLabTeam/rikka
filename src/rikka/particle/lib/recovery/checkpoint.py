"""checkpoint replay recovery。

役割:
    健全だった過去状態からの再生処理を戦略チェーンへ接続する。
依存元:
    ``protocol`` の共通契約と呼び出し元から渡される callback を使用する。
利用先:
    local recovery が失敗した後に試される。
処理フロー:
    callback を実行し、成功時だけ共通の recovery 結果を返す。
"""

from collections.abc import Callable
from dataclasses import dataclass

from ..state import ParticleState
from .protocol import RecoveryOutcome


@dataclass(frozen=True)
class CheckpointReplay:
    """checkpoint replay callback を recovery 戦略へ適合させる。"""

    callback: Callable[[ParticleState], ParticleState | None]

    def recover(self, state: ParticleState) -> RecoveryOutcome | None:
        """再生に成功した場合だけ checkpoint 結果を返す。"""
        replayed = self.callback(state)
        if replayed is None:
            return None
        return RecoveryOutcome("checkpoint_replay", replayed)
