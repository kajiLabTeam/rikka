"""BLE ランドマーク測位の単体テスト。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rikka.ble.lib.detection import detect_landmarks
from rikka.ble.lib.loader import group_by_timestamp, load_ble_observations
from rikka.ble.lib.sample import generate_sample_observations, write_sample_csv
from rikka.common.lib.models import BleObservation, Landmark
from rikka.common.settings import BleLandmarkSettings, BleSampleSettings


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


def _landmark_settings(**overrides: object) -> BleLandmarkSettings:
    values: dict[str, object] = {
        "landmarks": (
            Landmark("beacon_1", 1.0, 2.0),
            Landmark("beacon_2", 3.0, 4.0),
            Landmark("beacon_3", 5.0, 6.0),
        )
    }
    values.update(overrides)
    return BleLandmarkSettings(**values)  # type: ignore[arg-type]


def _observations(rssi_values: list[float]) -> tuple[BleObservation, ...]:
    return tuple(
        BleObservation(float(index), "beacon_1", rssi)
        for index, rssi in enumerate(rssi_values)
    )


def test_detect_landmarks_triggers_once_per_approach() -> None:
    """閾値超過が連続しても検出は 1 回だけになることを確認する。"""
    detections = detect_landmarks(
        _observations([-60, -50, -48, -49, -51, -50, -60]),
        _landmark_settings(),
    )

    assert len(detections) == 1


def test_detect_landmarks_redetects_after_release() -> None:
    """解除後に再び閾値以上になれば再検出することを確認する。"""
    detections = detect_landmarks(
        _observations([-50, -59, -60, -50]),
        _landmark_settings(),
    )

    assert [item.timestamp_s for item in detections] == [0.0, 3.0]


def test_detect_landmarks_release_margin_prevents_chattering() -> None:
    """閾値直下で揺らぐ RSSI では再検出しないことを確認する。"""
    detections = detect_landmarks(
        _observations([-50, -56, -54, -59, -60, -50]),
        _landmark_settings(),
    )

    assert [item.timestamp_s for item in detections] == [0.0, 5.0]


def test_detect_landmarks_selects_strongest_beacon() -> None:
    """同一時刻に複数が閾値を超えたとき最大 RSSI を採用する。"""
    observations = (
        BleObservation(1.0, "beacon_1", -52.0),
        BleObservation(1.0, "beacon_2", -45.0),
        BleObservation(1.0, "beacon_3", -70.0),
    )

    detections = detect_landmarks(observations, _landmark_settings())

    assert [item.beacon_id for item in detections] == ["beacon_2"]


def test_detect_landmarks_ignores_unregistered_beacon() -> None:
    """config に無い beacon_id が無視されることを確認する。"""
    detections = detect_landmarks(
        (BleObservation(1.0, "unknown", -40.0),),
        _landmark_settings(),
    )

    assert detections == ()


def test_detect_landmarks_returns_empty_without_strong_rssi() -> None:
    """閾値を超えないデータでは検出が 0 件になることを確認する。"""
    detections = detect_landmarks(
        _observations([-90, -70, -56]),
        _landmark_settings(),
    )

    assert detections == ()


def test_detect_landmarks_on_sample_data_detects_each_beacon_once(
    tmp_path: Path,
) -> None:
    """サンプル CSV で 3 ビーコンがそれぞれ 1 回だけ検出される。"""
    sample_settings = BleSampleSettings()
    observations = generate_sample_observations(
        np.arange(0.0, 72.3, sample_settings.interval_s),
        sample_settings,
    )
    path = write_sample_csv(observations, tmp_path / "sample.csv")

    detections = detect_landmarks(
        load_ble_observations(path),
        BleLandmarkSettings(),
    )

    assert len(detections) == 3
    assert {item.beacon_id for item in detections} == {
        "beacon_1",
        "beacon_2",
        "beacon_3",
    }
