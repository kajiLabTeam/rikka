"""BLE ランドマーク測位の単体テスト。"""

from pathlib import Path

import pandas as pd
import pytest

from rikka.ble.lib.loader import group_by_timestamp, load_ble_observations
from rikka.common.lib.models import BleObservation


def _write_ble_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_ble_observations_sorts_by_timestamp(tmp_path: Path) -> None:
    """時刻が前後した CSV でも昇順に整列されることを確認する。"""
    path = _write_ble_csv(
        tmp_path / "ble.csv",
        [
            {"timestamp_s": 2.0, "beacon_id": "b2", "rssi_dbm": -60.0},
            {"timestamp_s": 1.0, "beacon_id": "b1", "rssi_dbm": -50.0},
        ],
    )

    observations = load_ble_observations(path)

    assert [item.timestamp_s for item in observations] == [1.0, 2.0]


def test_load_ble_observations_keeps_same_timestamp_rows(tmp_path: Path) -> None:
    """同一時刻の複数ビーコン行が保持されることを確認する。"""
    path = _write_ble_csv(
        tmp_path / "ble.csv",
        [
            {"timestamp_s": 1.0, "beacon_id": "b2", "rssi_dbm": -60.0},
            {"timestamp_s": 1.0, "beacon_id": "b1", "rssi_dbm": -50.0},
        ],
    )

    observations = load_ble_observations(path)

    assert [item.beacon_id for item in observations] == ["b2", "b1"]


def test_load_ble_observations_rejects_missing_column(tmp_path: Path) -> None:
    """必須列が欠けた CSV が ValueError になることを確認する。"""
    path = _write_ble_csv(
        tmp_path / "ble.csv",
        [{"timestamp_s": 1.0, "beacon_id": "b1"}],
    )

    with pytest.raises(ValueError, match="必須列"):
        load_ble_observations(path)


def test_load_ble_observations_rejects_non_finite_rssi(tmp_path: Path) -> None:
    """RSSI に非有限値がある CSV が ValueError になることを確認する。"""
    path = _write_ble_csv(
        tmp_path / "ble.csv",
        [{"timestamp_s": 1.0, "beacon_id": "b1", "rssi_dbm": float("nan")}],
    )

    with pytest.raises(ValueError, match="rssi_dbm"):
        load_ble_observations(path)


def test_load_ble_observations_reports_missing_file(tmp_path: Path) -> None:
    """存在しないパスで日本語メッセージの ValueError になることを確認する。"""
    with pytest.raises(ValueError, match="BLE データが存在しません"):
        load_ble_observations(tmp_path / "missing.csv")


def test_group_by_timestamp_groups_same_time() -> None:
    """同一時刻の観測が 1 グループにまとまることを確認する。"""
    observations = (
        BleObservation(1.0, "b1", -50.0),
        BleObservation(1.0, "b2", -60.0),
        BleObservation(2.0, "b1", -70.0),
    )

    groups = group_by_timestamp(observations)

    assert [timestamp for timestamp, _ in groups] == [1.0, 2.0]
    assert [len(group) for _, group in groups] == [2, 1]
