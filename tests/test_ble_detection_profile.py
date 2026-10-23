"""実測・合成の検出条件をCSV形式と独立に選ぶことを検証する。"""

import json

import pytest

from rikka.ble.pipeline import run_ble_ranging_input
from rikka.common.lib.models import Landmark
from rikka.common.settings import BleLandmarkSettings


def _settings(tmp_path, profile="auto", *, metadata=False):
    path = tmp_path / "ble.csv"
    path.write_text("timestamp_s,beacon_id,rssi_dbm\n0,b,-50\n", encoding="utf-8")
    if metadata:
        (tmp_path / "metadata.json").write_text(
            json.dumps({"ble": {"configuration": {"beacons": []}}}),
            encoding="utf-8",
        )
    return BleLandmarkSettings(
        enabled=True,
        data_path=path,
        landmarks=(Landmark("b", 10, 10),),
        detection_profile=profile,
        rssi_smoothing_samples=1,
        detect_min_samples=3,
        detect_cooldown_s=10,
        detect_min_prominence_db=6,
    )


@pytest.mark.parametrize("profile", ["auto", "synthetic"])
def test_standard_csv_keeps_legacy_synthetic_detection(tmp_path, profile):
    result = run_ble_ranging_input(_settings(tmp_path, profile))
    assert result is not None
    assert len(result.detections) == 1
    assert len(result.observations) == 1


@pytest.mark.parametrize("profile,metadata", [("measured", False), ("auto", True)])
def test_standard_measured_data_does_not_relax_detection(tmp_path, profile, metadata):
    result = run_ble_ranging_input(_settings(tmp_path, profile, metadata=metadata))
    assert result is not None
    assert not result.detections
    # 検出条件で除外しても測距観測は失わない。
    assert len(result.observations) == 1


def test_explicit_synthetic_overrides_metadata(tmp_path):
    result = run_ble_ranging_input(_settings(tmp_path, "synthetic", metadata=True))
    assert result is not None
    assert len(result.detections) == 1


def test_explicit_profile_uses_requested_thresholds_without_mutating_settings(tmp_path):
    settings = _settings(tmp_path, "measured")
    run_ble_ranging_input(settings)
    assert settings.detect_min_samples == 3
    assert settings.detect_min_prominence_db == 6
    assert settings.detect_cooldown_s == 10


def test_detection_profile_rejects_unknown_value():
    with pytest.raises(ValueError, match="ble_detection_profile"):
        BleLandmarkSettings(detection_profile="unknown")


def test_disabled_ble_does_not_read_profile_inputs(tmp_path):
    assert (
        run_ble_ranging_input(
            BleLandmarkSettings(enabled=False, data_path=tmp_path / "missing.csv")
        )
        is None
    )


def test_profile_does_not_revalidate_overridden_metadata_coordinates(tmp_path):
    settings = _settings(tmp_path, metadata=True)
    # 配置は既に呼び出し元が指定している。検出profile判定で別の配置を再検証しない。
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "recording_context": {"floor_id": "f5"},
                "ble": {
                    "configuration": {
                        "beacons": [
                            {"beacon_id": "b", "floor_id": "other", "pixel_x": None}
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    result = run_ble_ranging_input(settings)
    assert result is not None
    assert not result.detections
    assert len(result.observations) == 1
