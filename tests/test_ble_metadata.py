"""Android計測のBLE配置読み込みとCLI入力解決を検証する。"""

import json
from pathlib import Path

import pandas as pd
import pytest

from rikka.ble.lib.loader import load_ble_metadata_landmarks, load_ble_observations
from rikka.cli import options
from rikka.common.lib.models import Landmark


def _beacon(**changes: object) -> dict[str, object]:
    return {
        "beacon_id": "b1",
        "pixel_x": 2088.37,
        "pixel_y": 3141.7,
        "floor_id": "floor-5",
        "enabled": True,
        **changes,
    }


def _metadata(directory: Path, beacons: list[dict[str, object]]) -> Path:
    path = directory / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "recording_context": {"floor_id": "floor-5"},
                "ble": {"configuration": {"beacons": beacons}},
            }
        ),
        encoding="utf-8",
    )
    return path


def _ble(directory: Path) -> Path:
    path = directory / "ble.csv"
    path.write_text("timestamp_s,beacon_id,rssi_dbm\n1.0,b1,-65\n")
    return path


def test_metadata_preserves_coordinates_without_hard_anchor(tmp_path: Path) -> None:
    path = _metadata(tmp_path, [_beacon(), _beacon(enabled=False, pixel_x=None)])

    assert load_ble_metadata_landmarks(path) == (Landmark("b1", 2088.37, 3141.7),)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"pixel_x": None}, "pixel_x"),
        ({"pixel_y": float("nan")}, "有限"),
        ({"pixel_y": float("inf")}, "有限"),
        ({"pixel_x": -1.0}, "0以上"),
        ({"pixel_x": True}, "数値"),
        ({"beacon_id": ""}, "beacon_id"),
        ({"enabled": "false"}, "真偽値"),
        ({"floor_id": "floor-4"}, "floor_id"),
    ],
)
def test_invalid_active_beacon_is_rejected(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_ble_metadata_landmarks(_metadata(tmp_path, [_beacon(**changes)]))


def test_duplicate_metadata_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="重複"):
        load_ble_metadata_landmarks(_metadata(tmp_path, [_beacon(), _beacon()]))


@pytest.mark.parametrize("contents", ["{", "[]", '{"ble": {}}'])
def test_malformed_metadata_is_rejected(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(contents)
    with pytest.raises(ValueError, match="metadata.jsonのBLE配置"):
        load_ble_metadata_landmarks(path)


def test_metadata_without_ble_keeps_legacy_fallback(tmp_path: Path) -> None:
    data = _ble(tmp_path)
    (tmp_path / "metadata.json").write_text("{}")
    assert options._resolve_ble_inputs(str(tmp_path), True, str(data)) == (
        str(data),
        None,
    )


def test_local_lowercase_ble_and_metadata_are_used(tmp_path: Path) -> None:
    data = _ble(tmp_path)
    _metadata(tmp_path, [_beacon()])

    resolved, landmarks = options._resolve_ble_inputs(
        str(tmp_path), True, options._BLE_DATA_DEFAULT
    )

    assert Path(resolved) == data
    assert landmarks == (Landmark("b1", 2088.37, 3141.7),)
    assert load_ble_observations(resolved)[0].beacon_id == landmarks[0].beacon_id


def test_explicit_ble_uses_its_own_metadata(tmp_path: Path) -> None:
    measurement = tmp_path / "measurement"
    explicit = tmp_path / "explicit"
    measurement.mkdir()
    explicit.mkdir()
    _ble(measurement)
    _metadata(measurement, [_beacon(pixel_x=1.0)])
    data = _ble(explicit)
    _metadata(explicit, [_beacon(pixel_x=2.0)])

    resolved, landmarks = options._resolve_ble_inputs(str(measurement), True, str(data))

    assert resolved == str(data)
    assert landmarks == (Landmark("b1", 2.0, 3141.7),)


def test_position_csv_has_priority_over_metadata(tmp_path: Path) -> None:
    data = _ble(tmp_path)
    _metadata(tmp_path, [_beacon()])
    pd.DataFrame(
        [
            {
                "beacon_id": "b1",
                "device_name": "beacon",
                "mac_address": "AA",
                "raw_data_suffix": "aa",
                "pixel_x": 10.0,
                "pixel_y": 20.0,
            }
        ]
    ).to_csv(tmp_path / "BLE_pos.csv", index=False)

    assert options._resolve_ble_inputs(str(tmp_path), True, str(data))[1] == (
        Landmark("b1", 10.0, 20.0),
    )


def test_disabled_ble_does_not_read_invalid_metadata(tmp_path: Path) -> None:
    data = _ble(tmp_path)
    (tmp_path / "metadata.json").write_text("invalid")
    assert options._resolve_ble_inputs(str(tmp_path), False, str(data)) == (
        str(data),
        None,
    )


def test_empty_active_configuration_does_not_use_global_defaults(
    tmp_path: Path,
) -> None:
    data = _ble(tmp_path)
    _metadata(tmp_path, [_beacon(enabled=False)])
    with pytest.raises(ValueError, match="有効なBLE配置がありません"):
        options._resolve_ble_inputs(str(tmp_path), True, str(data))
