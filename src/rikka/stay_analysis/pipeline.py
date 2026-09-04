"""滞在分析機能を公開する pipeline。

役割:
    滞在判定・セル集計・heatmap artifact生成の公開境界を定義する。
依存元:
    lib配下から処理関数、modelsから公開する結果型を取得する。
利用先:
    Nozomi などの呼び出し側が軌跡を分析するために使用する。
処理フロー:
    呼び出し側の用途に応じて、公開した分析処理を個別に実行する。
"""

from .lib.artifact import assemble_heatmap_artifact
from .lib.enrichment import enrich_trajectory
from .lib.grid import aggregate_trajectory_grid
from .models import StayCell, TrajectoryCells

__all__ = [
    "StayCell",
    "TrajectoryCells",
    "aggregate_trajectory_grid",
    "assemble_heatmap_artifact",
    "enrich_trajectory",
]
