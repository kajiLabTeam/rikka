"""入力検証と安全な出力処理の回帰テスト。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from rikka import cli
from rikka.analyze import sensor_plot
from rikka.analyze.pdr.gyro_bias import estimate_gyro_bias
from rikka.analyze.pdr.time_utils import _gyro_integration_dt, _time_values
from rikka.cli import commands as pipeline
from rikka.plot import pipeline as plot_pipeline


@pytest.mark.parametrize(
    "times",
    [
        [0.0, float("nan"), 0.02],
        [0.0, 0.01, 0.01],
        [0.0, 0.02, 0.01],
    ],
)
def test_explicit_invalid_time_column_is_rejected(times: list[float]) -> None:
    with pytest.raises(ValueError, match="t 列"):
        _time_values(pd.DataFrame({"t": times}))


def test_missing_time_column_keeps_fixed_rate_fallback() -> None:
    frame = pd.DataFrame({"x": [0.0, 0.0, 0.0]})

    assert _time_values(frame) is None
    np.testing.assert_allclose(_gyro_integration_dt(frame), np.full(3, 0.01))


@pytest.mark.parametrize(
    ("args", "option_name"),
    [
        (["run", "--direction", "nan"], "direction"),
        (["run", "--height-m", "inf"], "height_m"),
        (["sensor", "--gyro-bias", "nan"], "gyro_bias"),
        (
            ["particle", "--motion-predictive-weight-power", "nan"],
            "motion_predictive_weight_power",
        ),
    ],
)
def test_cli_rejects_non_finite_numeric_options(
    args: list[str],
    option_name: str,
) -> None:
    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 2
    assert option_name in result.output
    assert "有限" in result.output


def test_manual_gyro_bias_api_rejects_non_finite_value() -> None:
    frame = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]})

    with pytest.raises(ValueError, match="gyro_bias.*有限"):
        estimate_gyro_bias(frame, frame, method="manual", manual_bias=float("nan"))


def test_pipeline_rejects_non_finite_direction_before_output_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_created = False

    def fail_if_called() -> Path:
        nonlocal output_created
        output_created = True
        raise AssertionError("output should not be created")

    monkeypatch.setattr(plot_pipeline, "create_output_dir", fail_if_called)

    with pytest.raises(ValueError, match="initial_direction.*有限"):
        pipeline.run(
            pd.DataFrame(),
            pd.DataFrame(),
            plot=False,
            initial_direction=float("inf"),
        )
    assert not output_created


def test_pipeline_rejects_invalid_pf_path_before_output_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_created = False

    def fail_if_called() -> Path:
        nonlocal output_created
        output_created = True
        raise AssertionError("output should not be created")

    monkeypatch.setattr(plot_pipeline, "create_output_dir", fail_if_called)

    with pytest.raises(ValueError, match="pf_path_selection"):
        pipeline.run(
            pd.DataFrame(),
            pd.DataFrame(),
            plot=False,
            pf_path_selection="invalid",
        )
    assert not output_created


def test_particle_map_failure_happens_before_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_created = False

    def fail_if_called() -> Path:
        nonlocal output_created
        output_created = True
        raise AssertionError("output should not be created")

    monkeypatch.setattr(plot_pipeline, "create_output_dir", fail_if_called)
    sensor = pd.DataFrame(
        {
            "t": [0.0],
            "x": [0.0],
            "y": [0.0],
            "z": [9.8],
        }
    )

    with pytest.raises(ValueError, match="フロアマップが存在しません"):
        pipeline.run(
            sensor,
            sensor,
            plot=False,
            use_particle_filter=True,
            floormap_path=tmp_path / "missing.png",
            origin_px=(0, 0),
        )
    assert not output_created


def test_particle_map_validation_rejects_directory_corrupt_image_and_wall_origin(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="ファイルを指定"):
        pipeline._validate_particle_floormap(tmp_path, (0, 0))

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="画像として読み込めません"):
        pipeline._validate_particle_floormap(corrupt, (0, 0))

    floormap = tmp_path / "map.png"
    image = np.ones((4, 4), dtype=float)
    image[2, 2] = 0.0
    plt.imsave(floormap, image, cmap="gray", vmin=0.0, vmax=1.0)

    pipeline._validate_particle_floormap(floormap, (1, 1))
    with pytest.raises(ValueError, match="origin_px"):
        pipeline._validate_particle_floormap(floormap, (2, 2))
    with pytest.raises(ValueError, match="origin_px"):
        pipeline._validate_particle_floormap(floormap, (10, 10))


def test_sensor_plot_uses_numbered_path_without_overwrite(tmp_path: Path) -> None:
    first = tmp_path / "sensor_plot.png"
    second = tmp_path / "sensor_plot_001.png"

    assert sensor_plot._next_sensor_plot_path(tmp_path) == first
    first.touch()
    assert sensor_plot._next_sensor_plot_path(tmp_path) == second
    second.touch()
    assert sensor_plot._next_sensor_plot_path(tmp_path) == (
        tmp_path / "sensor_plot_002.png"
    )
