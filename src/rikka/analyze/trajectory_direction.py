"""軌跡終端方向指標の互換shim。

役割:
    従来の ``analyze.trajectory_direction`` importを維持する。
依存元:
    ``common.lib.trajectory_direction`` の実装を再輸出する。
利用先:
    既存のエージェント検証コードと外部利用者から使用される。
処理フロー:
    実装を持たず、共通指標の型と関数をそのまま公開する。
"""

from ..common.lib.trajectory_direction import (
    TerminalDirectionMetrics,
    evaluate_terminal_direction,
    sample_trajectory_by_arclength,
    terminal_consensus_outliers,
    terminal_heading,
)

__all__ = [
    "TerminalDirectionMetrics",
    "evaluate_terminal_direction",
    "sample_trajectory_by_arclength",
    "terminal_consensus_outliers",
    "terminal_heading",
]
