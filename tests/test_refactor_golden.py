"""領域分割リファクタリング中の数値挙動を固定する golden テスト。"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from rikka.cli.commands import run
from rikka.common.lib.sensors import load_sensor_data
from rikka.plot import pipeline as plot_pipeline

ROOT = Path(__file__).parents[1]
GOLDEN_PATH = ROOT / "tests/data/golden/refactor_golden.json"
OPEN_MAP_GOLDEN_PATH = ROOT / "tests/data/golden/refactor_open_map_golden.json"
CASES = (
    "input/sensor_data/natsuki/10steps_stride_length_check_1m-1",
    "input/sensor_data/natsuki/10steps_stride_length_check_50cm-1",
)


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    """固定済みの現行実装出力を読み込む。"""
    with GOLDEN_PATH.open(encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="module")
def open_map_golden() -> dict[str, Any]:
    """旧実装を同じ合成地図で実行した独立のPF参照値を読み込む。"""
    with OPEN_MAP_GOLDEN_PATH.open(encoding="utf-8") as file:
        return json.load(file)["cases"]


@pytest.mark.parametrize("data_dir", CASES)
@pytest.mark.parametrize("use_particle_filter", [False, True])
def test_refactor_golden(
    data_dir: str,
    use_particle_filter: bool,
    golden: dict[str, Any],
    open_map_golden: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_particle_profile,
) -> None:
    """従来の固定peak・legacy PFの軌跡・歩幅・採用方位を固定する。"""
    output_dir = tmp_path / ("pf" if use_particle_filter else "pdr")
    output_dir.mkdir()
    monkeypatch.setattr(plot_pipeline, "create_output_dir", lambda: output_dir)
    df_acc, df_gyro = load_sensor_data(ROOT / data_dir)

    # この回帰は旧ノイズと自由空間の数値演算を固定し、現地図の変更と分離する。
    map_path = tmp_path / "open_map.png"
    plt.imsave(
        map_path, np.ones((2000, 3000), dtype=np.uint8), cmap="gray", vmin=0, vmax=1
    )
    trajectory = run(
        df_acc=df_acc,
        df_gyro=df_gyro,
        plot=False,
        use_particle_filter=use_particle_filter,
        save_animation=False,
        particle_seed=42,
        particle_count=100,
        floormap_path=map_path,
        origin_px=(2050, 400),
        pf_proposal="legacy",
        pf_candidates=1,
        step_detection_method="peak",
    )

    actual_lengths = pd.read_csv(output_dir / "step_lengths.csv")[
        "step_length_m"
    ].to_numpy()
    actual_headings = np.deg2rad(
        pd.read_csv(output_dir / "step_headings.csv")["selected_heading_deg"].to_numpy()
    )
    expected = (
        open_map_golden[data_dir]["pf"]
        if use_particle_filter
        else golden[data_dir]["pdr"]
    )
    np.testing.assert_allclose(
        trajectory[["x", "y"]].to_numpy(),
        expected["trajectory"],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual_lengths,
        expected["step_lengths"],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual_headings,
        expected["selected_headings"],
        rtol=0,
        atol=1e-12,
    )
