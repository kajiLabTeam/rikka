"""軌跡に対する分析機能を公開する pipeline。

役割:
    滞在分析をはじめとする派生分析機能の公開境界を定義する。
依存元:
    機能別パッケージから処理関数と共有する結果型を取得する。
利用先:
    Nozomi などの呼び出し側が軌跡を分析するために使用する。
処理フロー:
    呼び出し側の用途に応じて、公開した分析処理を個別に実行する。
"""

from .stay_analysis.artifact import assemble_heatmap_artifact
from .stay_analysis.enrichment import enrich_trajectory
from .stay_analysis.grid import aggregate_trajectory_grid
from .stay_analysis.models import StayCell, TrajectoryCells

__all__ = [
    "StayCell",
    "TrajectoryCells",
    "aggregate_trajectory_grid",
    "assemble_heatmap_artifact",
    "enrich_trajectory",
]
