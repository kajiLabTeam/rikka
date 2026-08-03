"""滞在分析ライブラリの公開パッケージ。

役割:
    速度・滞在判定、グリッド集計、artifact 構築の公開境界を定義する。
依存元:
    各処理は同パッケージの ``enrichment``、``grid``、``artifact`` から取得する。
利用先:
    Nozomi が ``rikka.stay_analysis`` だけを import して分析するために使用する。
処理フロー:
    呼び出し側の用途に応じて、公開関数を個別に呼び出す。
"""

from rikka.stay_analysis.artifact import assemble_heatmap_artifact
from rikka.stay_analysis.enrichment import enrich_trajectory
from rikka.stay_analysis.grid import aggregate_trajectory_grid
from rikka.stay_analysis.models import StayCell, TrajectoryCells

__all__ = [
    "StayCell",
    "TrajectoryCells",
    "aggregate_trajectory_grid",
    "assemble_heatmap_artifact",
    "enrich_trajectory",
]
