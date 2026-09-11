"""複数trajectoryのsparse heatmap artifact生成。

役割:
    trajectory別セル集計と解析条件を、自己完結したJSON互換dictへまとめる。
依存元:
    親階層のmodelsから公開型・version定数、validationから検証処理を取得する。
利用先:
    Nozomiがheatmap JSONをserializationしてobject storageへ保存するために使用する。
処理フロー:
    条件とcellの整合性を検証し、trajectory順を維持してcellだけを安定順に整列する。
"""

import math
from typing import Any

from ..models import (
    DEFINITION_VERSION,
    SCHEMA_VERSION,
    StayCell,
    TrajectoryCells,
)
from .validation import validate_floor, validate_parameter


def _normalize_trajectories(
    trajectories: list[TrajectoryCells],
    *,
    column_count: int,
    row_count: int,
) -> list[TrajectoryCells]:
    normalized: list[TrajectoryCells] = []
    seen_ids: set[str] = set()
    for trajectory in trajectories:
        trajectory_id = trajectory["trajectory_id"]
        if not trajectory_id or trajectory_id in seen_ids:
            raise ValueError("trajectory_id must be non-empty and unique")
        seen_ids.add(trajectory_id)
        seen_cells: set[tuple[int, int]] = set()
        cells: list[StayCell] = []
        for cell in trajectory["cells"]:
            column = cell["grid_column"]
            row = cell["grid_row"]
            count = cell["stay_cell_visit_count"]
            values = (column, row, count)
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values
            ):
                raise ValueError("cell values must be integers")
            if (
                not 0 <= column < column_count
                or not 0 <= row < row_count
                or count <= 0
                or (column, row) in seen_cells
            ):
                raise ValueError("cell values are invalid or duplicated")
            seen_cells.add((column, row))
            cells.append(cell)
        normalized.append(
            {
                "trajectory_id": trajectory_id,
                "cells": sorted(
                    cells,
                    key=lambda cell: (cell["grid_row"], cell["grid_column"]),
                ),
            }
        )
    return normalized


def assemble_heatmap_artifact(
    trajectories: list[TrajectoryCells],
    *,
    speed_threshold_mps: float = 0.5,
    grid_size_m: float = 1.0,
    map_width_px: int,
    map_height_px: int,
    floor_scale: float,
) -> dict[str, Any]:
    """安定順のtrajectory集計を自己完結したsparse artifactにまとめる。"""
    threshold = validate_parameter(
        speed_threshold_mps,
        name="speed_threshold_mps",
        minimum=0,
        maximum=2,
    )
    grid_size = validate_parameter(
        grid_size_m,
        name="grid_size_m",
        minimum=0.1,
        maximum=10,
    )
    scale = validate_floor(
        map_width_px=map_width_px,
        map_height_px=map_height_px,
        floor_scale=floor_scale,
    )
    column_count = math.ceil(map_width_px * scale / grid_size)
    row_count = math.ceil(map_height_px * scale / grid_size)
    normalized = _normalize_trajectories(
        trajectories,
        column_count=column_count,
        row_count=row_count,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "definition_version": DEFINITION_VERSION,
        "parameters": {
            "speed_threshold_mps": threshold,
            "grid_size_m": grid_size,
        },
        "floor_map": {
            "width_px": map_width_px,
            "height_px": map_height_px,
            "scale_m_per_px": scale,
        },
        "grid": {
            "size_m": grid_size,
            "column_count": column_count,
            "row_count": row_count,
        },
        "input_trajectory_count": len(normalized),
        "trajectories": normalized,
    }
