"""Androidロガーの時刻変換と開始条件の優先順位を検証する。"""

import json

import numpy as np
import pandas as pd
import pytest

from rikka.common.lib.measurement_config import load_measurement_config
from rikka.common.lib.sensors import load_sensor_data


def _recording(path):
    metadata = {
        "recording_start_wall_time_ms": 1000,
        "constraints": [{"point_type": "start", "x": 10, "y": 20, "direction": 90}],
    }
    (path / "metadata.json").write_text(json.dumps(metadata))
    for name, offset in (("acce.csv", 40), ("gyro.csv", 55)):
        pd.DataFrame(
            {
                "timestamp_ns": [800000000000001, 800000010000001, 800000020000001],
                "wall_time_ms": [1000 + offset] * 3,
                "x": [1, 2, 3],
                "y": [4, 5, 6],
                "z": [7, 8, 9],
            }
        ).to_csv(path / name, index=False)


def test_logger_preserves_sensor_offsets_and_uses_monotonic_clock(tmp_path):
    _recording(tmp_path)
    acc, gyro = load_sensor_data(tmp_path)
    np.testing.assert_allclose(acc.t, [0.04, 0.05, 0.06], atol=1e-12)
    np.testing.assert_allclose(gyro.t, [0.055, 0.065, 0.075], atol=1e-12)
    assert list(acc.columns) == ["t", "x", "y", "z"]
    assert acc.x.tolist() == [1, 2, 3]


@pytest.mark.parametrize("problem", ["metadata", "columns", "time"])
def test_logger_rejects_missing_timing_and_nonmonotonic_clock(tmp_path, problem):
    _recording(tmp_path)
    if problem == "metadata":
        (tmp_path / "metadata.json").unlink()
    else:
        frame = pd.read_csv(tmp_path / "acce.csv")
        if problem == "columns":
            frame = frame.drop(columns="timestamp_ns")
        else:
            frame.loc[1, "timestamp_ns"] = frame.loc[0, "timestamp_ns"]
        frame.to_csv(tmp_path / "acce.csv", index=False)
    with pytest.raises(ValueError):
        load_sensor_data(tmp_path)


def test_metadata_start_and_explicit_walk_config_priority(tmp_path):
    _recording(tmp_path)
    config = load_measurement_config(tmp_path, image_size_px=(100, 100))
    assert config.origin_px == (10, 20)
    assert config.initial_direction_deg == 90
    with pytest.raises(ValueError, match="範囲外"):
        load_measurement_config(tmp_path, image_size_px=(10, 10))
    pd.DataFrame(
        [
            {
                "origin_px_x": 5,
                "origin_px_y": 6,
                "initial_direction_deg": 45,
                "user_height_m": 1.68,
            }
        ]
    ).to_csv(tmp_path / "walk_config.csv", index=False)
    config = load_measurement_config(tmp_path)
    assert config.origin_px == (5, 6)
    assert config.initial_direction_deg == 45
