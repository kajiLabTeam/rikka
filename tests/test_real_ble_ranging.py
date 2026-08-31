"""実測BLE測距、計測設定、warp補正の回帰テスト。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from rikka.ble.lib.detection import detect_landmarks
from rikka.ble.lib.loader import load_ble_landmarks
from rikka.ble.lib.pathloss import (
    estimate_path_loss,
    expected_rssi_dbm,
    rssi_to_distance_m,
)
from rikka.cli import commands as cli_commands
from rikka.cli.options import _resolve_ble_inputs, cli
from rikka.common.lib.measurement_config import load_measurement_config
from rikka.common.lib.models import (
    BleObservation,
    FloorMap,
    Landmark,
    LandmarkRange,
    PathLossModel,
)
from rikka.common.settings import BleLandmarkSettings
from rikka.particle.lib.landmark import landmark_range_likelihood
from rikka.pdr.lib.landmark_correction import apply_landmark_corrections


def test_measurement_config_reads_optional_height_and_validates_map(
    tmp_path: Path,
) -> None:
    """walk_configの1行を読み、地図外起点を拒否する。"""
    pd.DataFrame(
        [
            {
                "origin_px_x": 12,
                "origin_px_y": 34,
                "initial_direction_deg": 260,
                "user_height_m": 1.68,
                "note": "開始点",
            }
        ]
    ).to_csv(tmp_path / "walk_config.csv", index=False)

    config = load_measurement_config(tmp_path, image_size_px=(100, 100))

    assert config is not None
    assert config.origin_px == (12, 34)
    assert config.initial_direction_deg == 260.0
    assert config.user_height_m == 1.68
    with pytest.raises(ValueError, match="範囲外"):
        load_measurement_config(tmp_path, image_size_px=(10, 10))


def test_path_loss_round_trip_and_estimation() -> None:
    """既知モデルのRSSI変換と回帰が元係数を復元する。"""
    model = PathLossModel(-59.0, 2.2, 4.0)
    distances = np.asarray([0.5, 1.0, 2.0, 5.0, 10.0])
    rssi = np.asarray(expected_rssi_dbm(distances, model))

    estimated = estimate_path_loss(distances, rssi)

    assert rssi_to_distance_m(float(rssi[2]), model) == pytest.approx(2.0)
    assert estimated.tx_power_dbm == pytest.approx(model.tx_power_dbm)
    assert estimated.path_loss_n == pytest.approx(model.path_loss_n)
    assert estimated.rssi_sigma_db == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize(
    "model",
    [
        (-59.0, 0.0, 6.0),
        (-59.0, 2.0, 0.0),
        (float("nan"), 2.0, 6.0),
    ],
)
def test_path_loss_model_rejects_non_physical_values(
    model: tuple[float, float, float],
) -> None:
    """PFで除算不能になる非物理係数を共有型の境界で拒否する。"""
    with pytest.raises(ValueError):
        PathLossModel(*model)


def test_ble_position_optional_path_loss_columns_override_defaults(
    tmp_path: Path,
) -> None:
    """BLE_posの任意3列がビーコン別モデルとして保持される。"""
    pd.DataFrame(
        [
            {
                "beacon_id": "b1",
                "device_name": "b1",
                "mac_address": "00:11:22:33:44:55",
                "raw_data_suffix": "abcd",
                "pixel_x": 10,
                "pixel_y": 20,
                "tx_power_dbm": -62,
                "path_loss_n": 2.5,
                "rssi_sigma_db": 5.0,
            }
        ]
    ).to_csv(tmp_path / "BLE_pos.csv", index=False)

    landmark = load_ble_landmarks(tmp_path / "BLE_pos.csv")[0]

    assert landmark.path_loss_model == PathLossModel(-62.0, 2.5, 5.0)


def test_median_peak_selection_rejects_isolated_spike() -> None:
    """孤立した強RSSIより持続する平滑ピークの時刻を選ぶ。"""
    observations = tuple(
        BleObservation(float(index), "b1", rssi)
        for index, rssi in enumerate([-65.0, -64.0, -40.0, -64.0, -63.0])
    )
    settings = BleLandmarkSettings(
        rssi_threshold_dbm=-70.0,
        detect_min_samples=1,
        detect_min_prominence_db=0.0,
        landmarks=(Landmark("b1", 0.0, 0.0),),
    )

    detections = detect_landmarks(observations, settings)

    assert detections[0].timestamp_s == 4.0
    assert detections[0].rssi_dbm == -63.0


def test_robust_detection_replaces_cooldown_peak_and_rejects_weak_peak() -> None:
    """cooldown内は強い接近を残し、閾値から6dB未満の弱い山を棄却する。"""
    observations = tuple(
        BleObservation(timestamp, beacon_id, rssi)
        for timestamp, beacon_id, rssi in (
            (-0.1, "b1", -80.0),
            (0.0, "b1", -65.0),
            (0.1, "b1", -64.0),
            (0.2, "b1", -63.0),
            (0.3, "b1", -80.0),
            (0.4, "b1", -80.0),
            (5.0, "b1", -59.0),
            (5.1, "b1", -58.0),
            (5.2, "b1", -59.0),
            (5.3, "b1", -80.0),
            (5.4, "b1", -80.0),
            (6.0, "b1", -80.0),
            (20.0, "b2", -68.0),
            (20.1, "b2", -68.0),
            (20.2, "b2", -68.0),
            (20.3, "b2", -80.0),
            (20.4, "b2", -80.0),
        )
    )
    settings = BleLandmarkSettings(
        rssi_threshold_dbm=-70.0,
        rssi_smoothing_samples=3,
        landmarks=(Landmark("b1", 0.0, 0.0), Landmark("b2", 1.0, 0.0)),
    )

    detections = detect_landmarks(observations, settings)

    assert len(detections) == 1
    assert detections[0].beacon_id == "b1"
    assert 5.0 <= detections[0].timestamp_s <= 5.2
    assert detections[0].rssi_dbm == -58.0


def test_pdr_warp_distributes_residual_over_past_steps() -> None:
    """warpは直前拘束から検出歩までへ距離比で残差を配分する。"""
    detection = LandmarkRange(2.0, "b1", -59.0, 0.0, 0.1, -59.0)
    result = apply_landmark_corrections(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        [1.0, 2.0],
        detections=(detection,),
        landmarks=(Landmark("b1", 4.0, 0.0),),
        floormap=FloorMap("map.png", (0, 0), 1.0),
        data_path="ble.csv",
        rssi_threshold_dbm=-70.0,
        gx_mean=0.0,
        gz_mean=1.0,
        correction_mode="warp",
    )

    assert np.allclose(result.trajectory, [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    assert result.corrections[0].warp_span_m == pytest.approx(2.0)
    assert result.corrections[0].correction_mode == "warp"


def test_pdr_rejects_correction_over_safety_limit() -> None:
    """補正移動量の上限超過は履歴だけを残し軌跡を変更しない。"""
    result = apply_landmark_corrections(
        [[0.0, 0.0], [1.0, 0.0]],
        [1.0],
        detections=(LandmarkRange(1.0, "b1", -59.0, 0.0, 0.1, -59.0),),
        landmarks=(Landmark("b1", 10.0, 0.0),),
        floormap=FloorMap("map.png", (0, 0), 1.0),
        data_path="ble.csv",
        rssi_threshold_dbm=-70.0,
        gx_mean=0.0,
        gz_mean=1.0,
        max_correction_m=5.0,
    )

    assert result.trajectory == [[0.0, 0.0], [1.0, 0.0]]
    assert not result.corrections[0].applied


def test_ranging_likelihood_uses_weak_rssi_as_far_observation() -> None:
    """弱いRSSIではビーコンから遠い粒子の尤度が高くなる。"""
    particles = np.asarray([[0.3, 0.0], [3.0, 0.0], [10.0, 0.0]])
    likelihood = landmark_range_likelihood(
        particles,
        (0.0, 0.0),
        -79.0,
        PathLossModel(-59.0, 2.0, 6.0),
        0.05,
    )

    assert likelihood[2] > likelihood[1] > likelihood[0]


def test_explicit_logger_ble_requires_position_file(tmp_path: Path) -> None:
    """Thingsup形式を明示した場合はBLE_pos欠落を黙って補わない。"""
    pd.DataFrame(
        columns=["Timestamp", "Device Name", "MAC Address", "RSSI", "Raw Data"]
    ).to_csv(tmp_path / "BLE.csv", index=False)

    with pytest.raises(ValueError, match="BLE_pos.csv"):
        _resolve_ble_inputs(str(tmp_path), True, str(tmp_path / "BLE.csv"))


def test_cli_walk_config_applies_unless_values_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI明示値がwalk_configより優先され、未指定値だけ自動選択される。"""
    pd.DataFrame(
        [
            {
                "origin_px_x": 12,
                "origin_px_y": 34,
                "initial_direction_deg": 260,
                "user_height_m": 1.75,
            }
        ]
    ).to_csv(tmp_path / "walk_config.csv", index=False)
    map_path = tmp_path / "map.png"
    plt.imsave(map_path, np.ones((100, 100)), cmap="gray")
    monkeypatch.setattr(
        "rikka.common.lib.sensors.load_sensor_data",
        lambda _data_dir: (pd.DataFrame(), pd.DataFrame()),
    )
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(cli_commands, "run", lambda **kwargs: captured.append(kwargs))

    runner = CliRunner()
    automatic = runner.invoke(
        cli,
        ["run", "-d", str(tmp_path), "--floormap", str(map_path), "--no-plot"],
    )
    explicit = runner.invoke(
        cli,
        [
            "run",
            "-d",
            str(tmp_path),
            "--floormap",
            str(map_path),
            "--origin-px",
            "1",
            "2",
            "--direction",
            "90",
            "--height-m",
            "1.6",
            "--no-plot",
        ],
    )

    assert automatic.exit_code == 0, automatic.output
    assert explicit.exit_code == 0, explicit.output
    assert captured[0]["origin_px"] == (12, 34)
    assert captured[0]["initial_direction"] == 260.0
    assert captured[0]["height_m"] == 1.75
    assert captured[1]["origin_px"] == (1, 2)
    assert captured[1]["initial_direction"] == 90.0
    assert captured[1]["height_m"] == 1.6
