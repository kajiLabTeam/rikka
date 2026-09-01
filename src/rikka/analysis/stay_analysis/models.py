"""滞在分析で共有するデータ型とversion定数。

役割:
    セル集計とtrajectory集計のJSON互換型、解析・schema versionを定義する。
依存元:
    Python標準のTypedDictだけを使用する。
利用先:
    grid、artifact、analysis pipelineが共通の結果形式を参照する。
処理フロー:
    実行処理は持たず、各処理が生成・検証する値の構造を固定する。
"""

from typing import TypedDict

DEFINITION_VERSION = "original-v1"
SCHEMA_VERSION = "1.0"


class StayCell(TypedDict):
    grid_column: int
    grid_row: int
    stay_cell_visit_count: int


class TrajectoryCells(TypedDict):
    trajectory_id: str
    cells: list[StayCell]
