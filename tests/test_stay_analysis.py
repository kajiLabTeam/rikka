"""滞在判定・セル集計・artifact 公開 API の仕様を検証する。"""

import math

import pandas as pd
import pytest

from rikka.analysis.pipeline import (
    aggregate_trajectory_grid,
    assemble_heatmap_artifact,
    enrich_trajectory,
)


def trajectory(
    *,
    timestamps: list[float | None] | None = None,
    rikka_x: list[float] | None = None,
) -> pd.DataFrame:
    times = timestamps or [None, 0.0, 0.5, 1.0, 2.0]
    relative_x = rikka_x or [0.0, 0.0, 0.25, 0.5, 0.75]
    return pd.DataFrame(
        {
            "step_index": range(len(times)),
            "rikka_timestamp_s": times,
            "rikka_x": relative_x,
            "rikka_y": [0.0] * len(times),
            "x": [10.0 + value for value in relative_x],
            "y": [20.0] * len(times),
            "memo": list(range(len(times))),
        }
    )


@pytest.mark.parametrize("row_count", [1, 2])
def test_enrich_short_trajectory_has_no_stay(row_count: int) -> None:
    result = enrich_trajectory(trajectory().iloc[:row_count].copy())

    assert list(result.columns)[-2:] == ["speed_mps", "is_stay"]
    assert result["speed_mps"].isna().all()
    assert result["is_stay"].tolist() == [False] * row_count


def test_enrich_calculates_incoming_speed_and_row_wise_stay() -> None:
    source = trajectory()
    result = enrich_trajectory(source, speed_threshold_mps=0.5)

    assert result is not source
    assert result["memo"].tolist() == source["memo"].tolist()
    assert math.isnan(result.loc[0, "speed_mps"])
    assert math.isnan(result.loc[1, "speed_mps"])
    assert result["speed_mps"].iloc[2:].tolist() == [0.5, 0.5, 0.25]
    assert result["is_stay"].tolist() == [False, False, False, True, True]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda frame: frame.drop(columns="x"), "missing required columns"),
        (lambda frame: frame.assign(step_index=[0, 1, 3, 4, 5]), "step_index"),
        (
            lambda frame: frame.assign(rikka_timestamp_s=[0.0, 0.0, 1.0, 1.5, 2.0]),
            "first rikka_timestamp_s",
        ),
        (
            lambda frame: frame.assign(rikka_timestamp_s=[None, 0.1, 1.0, 1.5, 2.0]),
            "second rikka_timestamp_s",
        ),
        (
            lambda frame: frame.assign(
                rikka_timestamp_s=[None, 0.0, 1.0, float("nan"), 2.0]
            ),
            "invalid value",
        ),
        (
            lambda frame: frame.assign(rikka_timestamp_s=[None, 0.0, 1.0, 1.0, 2.0]),
            "strictly increasing",
        ),
        (
            lambda frame: frame.assign(rikka_timestamp_s=[None, 0.0, 1.0, 0.5, 2.0]),
            "strictly increasing",
        ),
        (lambda frame: frame.assign(rikka_x=[0, 0, 0, float("nan"), 0]), "finite"),
        (lambda frame: frame.assign(y=[0, 0, 0, 0, float("inf")]), "finite"),
    ],
)
def test_enrich_rejects_invalid_trajectory(change, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        enrich_trajectory(change(trajectory()))


@pytest.mark.parametrize(
    "parameter",
    [-0.001, 2.001, float("nan"), float("inf"), 0.1234],
)
def test_enrich_rejects_invalid_speed_threshold(parameter: float) -> None:
    with pytest.raises(ValueError):
        enrich_trajectory(trajectory(), speed_threshold_mps=parameter)


def test_grid_counts_transitions_after_filtering_and_excludes_start() -> None:
    dataframe = pd.DataFrame(
        {
            "step_index": range(12),
            "rikka_timestamp_s": [None, *map(float, range(11))],
            "rikka_x": [0.0] * 12,
            "rikka_y": [0.0] * 12,
            "x": [5, 15, 16, 25, 35, 25, 15, 5, -1, 34, 20, 5],
            "y": [5, 5, 5, 5, 5, 5, 5, 5, 5, 19, 20, 5],
            "is_stay": [
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
        }
    )

    result = aggregate_trajectory_grid(
        dataframe,
        map_width_px=35,
        map_height_px=20,
        floor_scale=1,
        grid_size_m=10,
        start_x_px=5,
        start_y_px=5,
    )

    assert result == [
        {"grid_column": 1, "grid_row": 0, "stay_cell_visit_count": 2},
        {"grid_column": 2, "grid_row": 0, "stay_cell_visit_count": 1},
        {"grid_column": 3, "grid_row": 1, "stay_cell_visit_count": 1},
    ]


def test_grid_keeps_boundary_point_and_rejects_invalid_settings() -> None:
    enriched = enrich_trajectory(trajectory(), speed_threshold_mps=0.5)
    enriched.loc[3, ["x", "y"]] = [20.0, 10.0]
    result = aggregate_trajectory_grid(
        enriched,
        map_width_px=100,
        map_height_px=100,
        floor_scale=1,
        grid_size_m=10,
        start_x_px=0,
        start_y_px=0,
    )
    assert result[0]["grid_column"] == 2
    assert result[0]["grid_row"] == 1

    with pytest.raises(ValueError, match="grid_size_m"):
        aggregate_trajectory_grid(
            enriched,
            map_width_px=100,
            map_height_px=100,
            floor_scale=1,
            grid_size_m=0.1234,
            start_x_px=0,
            start_y_px=0,
        )
    with pytest.raises(ValueError, match="inside"):
        aggregate_trajectory_grid(
            enriched,
            map_width_px=100,
            map_height_px=100,
            floor_scale=1,
            start_x_px=100,
            start_y_px=0,
        )


def test_grid_rejects_nullable_stay_flag_with_missing_value() -> None:
    dataframe = trajectory()
    dataframe["is_stay"] = pd.Series([False, False, True, pd.NA, True], dtype="boolean")

    with pytest.raises(ValueError, match="only boolean values"):
        aggregate_trajectory_grid(
            dataframe,
            map_width_px=100,
            map_height_px=100,
            floor_scale=1,
            start_x_px=0,
            start_y_px=0,
        )


def test_assemble_artifact_preserves_trajectory_order_and_sorts_cells() -> None:
    artifact = assemble_heatmap_artifact(
        [
            {
                "trajectory_id": "second-in-sort-but-first-in-request",
                "cells": [
                    {"grid_column": 1, "grid_row": 1, "stay_cell_visit_count": 1},
                    {"grid_column": 1, "grid_row": 0, "stay_cell_visit_count": 2},
                ],
            },
            {"trajectory_id": "another", "cells": []},
        ],
        speed_threshold_mps=0.5,
        grid_size_m=1,
        map_width_px=101,
        map_height_px=201,
        floor_scale=0.01,
    )

    assert artifact["schema_version"] == "1.0"
    assert artifact["definition_version"] == "original-v1"
    assert artifact["grid"] == {"size_m": 1.0, "column_count": 2, "row_count": 3}
    assert artifact["input_trajectory_count"] == 2
    assert [item["trajectory_id"] for item in artifact["trajectories"]] == [
        "second-in-sort-but-first-in-request",
        "another",
    ]
    assert artifact["trajectories"][0]["cells"][0]["grid_row"] == 0
    assert artifact["trajectories"][1]["cells"] == []


def test_assemble_artifact_rejects_duplicate_cells_and_trajectories() -> None:
    cell = {"grid_column": 1, "grid_row": 2, "stay_cell_visit_count": 1}
    with pytest.raises(ValueError, match="duplicated"):
        assemble_heatmap_artifact(
            [{"trajectory_id": "a", "cells": [cell, cell]}],
            map_width_px=100,
            map_height_px=100,
            floor_scale=0.01,
        )
    with pytest.raises(ValueError, match="unique"):
        assemble_heatmap_artifact(
            [
                {"trajectory_id": "a", "cells": []},
                {"trajectory_id": "a", "cells": []},
            ],
            map_width_px=100,
            map_height_px=100,
            floor_scale=0.01,
        )


def test_assemble_artifact_rejects_cell_outside_grid() -> None:
    with pytest.raises(ValueError, match="invalid"):
        assemble_heatmap_artifact(
            [
                {
                    "trajectory_id": "a",
                    "cells": [
                        {
                            "grid_column": 1,
                            "grid_row": 0,
                            "stay_cell_visit_count": 1,
                        }
                    ],
                }
            ],
            grid_size_m=1,
            map_width_px=100,
            map_height_px=100,
            floor_scale=0.01,
        )
