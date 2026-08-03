"""滞在判定済みtrajectoryのグリッド集計。

役割:
    画像内の滞在点から、開始セルを除いたセル訪問区間数を算出する。
依存元:
    validationのfloor・座標・parameter検証とmodelsのStayCell型を取得する。
利用先:
    Nozomiがtrajectory別のsparse heatmap入力を作るために使用する。
処理フロー:
    画像外・非滞在・開始セルを除き、滞在点列のセル遷移を数えて安定順に返す。
"""

import math

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from rikka.stay_analysis.models import StayCell
from rikka.stay_analysis.validation import (
    REQUIRED_COLUMNS,
    finite_numeric,
    require_columns,
    validate_floor,
    validate_parameter,
    validate_point,
)


def aggregate_trajectory_grid(
    dataframe: pd.DataFrame,
    *,
    map_width_px: int,
    map_height_px: int,
    floor_scale: float,
    grid_size_m: float = 1.0,
    start_x_px: float,
    start_y_px: float,
) -> list[StayCell]:
    """滞在判定済みtrajectoryを非ゼロのセル訪問回数へ集計する。"""
    require_columns(dataframe, (*REQUIRED_COLUMNS, "is_stay"))
    if not is_bool_dtype(dataframe["is_stay"].dtype):
        raise ValueError("is_stay must be boolean")
    scale = validate_floor(
        map_width_px=map_width_px,
        map_height_px=map_height_px,
        floor_scale=floor_scale,
    )
    grid_size = validate_parameter(
        grid_size_m,
        name="grid_size_m",
        minimum=0.1,
        maximum=10,
    )
    start_x = validate_point(start_x_px, name="start_x_px", limit=map_width_px)
    start_y = validate_point(start_y_px, name="start_y_px", limit=map_height_px)
    finite_numeric(dataframe, "x")
    finite_numeric(dataframe, "y")

    inside = dataframe[
        dataframe["x"].ge(0)
        & dataframe["x"].lt(map_width_px)
        & dataframe["y"].ge(0)
        & dataframe["y"].lt(map_height_px)
        & dataframe["is_stay"]
    ].copy()
    if inside.empty:
        return []

    inside["grid_column"] = np.floor(inside["x"] * scale / grid_size).astype(int)
    inside["grid_row"] = np.floor(inside["y"] * scale / grid_size).astype(int)
    start_column = math.floor(start_x * scale / grid_size)
    start_row = math.floor(start_y * scale / grid_size)
    inside = inside[
        (inside["grid_column"] != start_column) | (inside["grid_row"] != start_row)
    ]
    if inside.empty:
        return []

    changed = (inside["grid_column"] != inside["grid_column"].shift()) | (
        inside["grid_row"] != inside["grid_row"].shift()
    )
    counts = (
        inside[changed]
        .groupby(["grid_row", "grid_column"], sort=True)
        .size()
        .reset_index(name="stay_cell_visit_count")
    )
    return [
        {
            "grid_column": int(row.grid_column),
            "grid_row": int(row.grid_row),
            "stay_cell_visit_count": int(row.stay_cell_visit_count),
        }
        for row in counts.itertuples(index=False)
    ]
