"""BLE ランドマーク測位の単体テスト。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from rikka.ble.lib.correction import (
    apply_landmark_corrections,
    assign_detections_to_steps,
)
from rikka.ble.lib.detection import detect_landmarks
from rikka.ble.lib.loader import group_by_timestamp, load_ble_observations
from rikka.ble.lib.sample import generate_sample_observations, write_sample_csv
from rikka.ble.pipeline import run_landmark_correction
from rikka.cli.options import cli
from rikka.common.lib.models import BleObservation, Landmark, LandmarkDetection
from rikka.common.lib.sensors import load_sensor_data
from rikka.common.settings import BleLandmarkSettings, BleSampleSettings, PdrSettings
from rikka.pdr.pipeline import run_pdr
from rikka.plot.lib.outputs import _build_landmark_corrections_dataframe


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


def _detection(
    timestamp_s: float,
    beacon_id: str = "beacon_1",
) -> LandmarkDetection:
    return LandmarkDetection(timestamp_s, beacon_id, -50.0)


def test_apply_landmark_corrections_moves_position_to_landmark() -> None:
    """補正が起きた歩の座標がランドマーク座標に一致する。"""
    result = apply_landmark_corrections(
        [[0.0, 0.0], [1.0, 0.0]],
        [1.0],
        (_detection(1.0),),
        _landmark_settings(),
        "ble.csv",
    )

    assert result.trajectory[-1] == [1.0, 2.0]


def test_apply_landmark_corrections_continues_from_corrected_position() -> None:
    """補正後の歩は元 PDR の変位を保って補正座標から続く。"""
    result = apply_landmark_corrections(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        [1.0, 2.0, 3.0],
        (_detection(2.0),),
        _landmark_settings(
            landmarks=(Landmark("beacon_1", 10.0, 5.0),),
        ),
        "ble.csv",
    )

    assert result.trajectory == [
        [0.0, 0.0],
        [1.0, 0.0],
        [10.0, 5.0],
        [11.0, 5.0],
    ]


def test_apply_landmark_corrections_keeps_raw_trajectory() -> None:
    """raw_trajectory が補正前を保持し、入力を破壊しない。"""
    trajectory = [[0.0, 0.0], [1.0, 0.0]]

    result = apply_landmark_corrections(
        trajectory,
        [1.0],
        (_detection(1.0),),
        _landmark_settings(),
        "ble.csv",
    )

    assert result.raw_trajectory == [[0.0, 0.0], [1.0, 0.0]]
    assert trajectory == [[0.0, 0.0], [1.0, 0.0]]


def test_apply_landmark_corrections_without_detection_is_identity() -> None:
    """検出が無い場合は元軌跡と一致する。"""
    trajectory = [[0.0, 0.0], [1.0, 0.0]]

    result = apply_landmark_corrections(
        trajectory,
        [1.0],
        (),
        _landmark_settings(),
        "ble.csv",
    )

    assert result.trajectory == trajectory


def test_assign_detections_to_steps_uses_first_step_at_or_after() -> None:
    """検出時刻以降で最も早い歩に割り当てられる。"""
    assigned, discarded = assign_detections_to_steps(
        (_detection(1.5),),
        [1.0, 2.0, 3.0],
    )

    assert assigned == {1: [_detection(1.5)]}
    assert discarded == 0


def test_assign_detections_to_steps_discards_after_last_step() -> None:
    """最終歩より後の検出を破棄して件数を数える。"""
    assigned, discarded = assign_detections_to_steps(
        (_detection(4.0),),
        [1.0, 2.0, 3.0],
    )

    assert assigned == {}
    assert discarded == 1


def test_apply_landmark_corrections_marks_last_detection_applied() -> None:
    """同じ歩に複数検出があるとき最後だけ applied になる。"""
    result = apply_landmark_corrections(
        [[0.0, 0.0], [1.0, 0.0]],
        [1.0],
        (_detection(0.5, "beacon_1"), _detection(0.7, "beacon_2")),
        _landmark_settings(),
        "ble.csv",
    )

    assert [item.applied for item in result.corrections] == [False, True]
    assert result.trajectory[-1] == [3.0, 4.0]


def test_run_landmark_correction_returns_none_when_disabled() -> None:
    """無効時は BLE ファイルを読まず None を返す。"""
    result = run_landmark_correction(
        [[0.0, 0.0]],
        [],
        BleLandmarkSettings(enabled=False, data_path="missing.csv"),
    )

    assert result is None


def test_run_pdr_without_ble_matches_prepared_trajectory() -> None:
    """BLE 無効時の run_pdr 軌跡が共有済み軌跡と一致する。"""
    df_acc, df_gyro = load_sensor_data()

    result = run_pdr(PdrSettings(), df_acc, df_gyro)

    assert result.landmark is None
    assert result.trajectory == result.prepared.trajectory


def test_build_landmark_corrections_dataframe_has_diagnostic_columns() -> None:
    """補正履歴が必要な診断列を持つ DataFrame になる。"""
    result = apply_landmark_corrections(
        [[0.0, 0.0], [1.0, 0.0]],
        [1.0],
        (_detection(1.0),),
        _landmark_settings(),
        "ble.csv",
    )

    dataframe = _build_landmark_corrections_dataframe(result)

    assert dataframe.to_dict("records") == [
        {
            "step": 0,
            "timestamp_s": 1.0,
            "beacon_id": "beacon_1",
            "rssi_dbm": -50.0,
            "detected": True,
            "applied": True,
            "before_x": 1.0,
            "before_y": 0.0,
            "landmark_x": 1.0,
            "landmark_y": 2.0,
            "after_x": 1.0,
            "after_y": 2.0,
        }
    ]


def test_run_help_includes_ble_landmark_option() -> None:
    """run --help に BLE ランドマークのオプションが出る。"""
    result = CliRunner().invoke(cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--ble-landmark" in result.output
    assert "--ble-data" in result.output
    assert "--ble-rssi-threshold" in result.output


def test_particle_help_includes_ble_landmark_option() -> None:
    """particle --help にも BLE ランドマークのオプションが出る。"""
    result = CliRunner().invoke(cli, ["particle", "--help"])

    assert result.exit_code == 0
    assert "--ble-landmark" in result.output
    assert "--ble-data" in result.output
    assert "--ble-rssi-threshold" in result.output


def test_ble_sample_help_lists_options() -> None:
    """ble-sample --help が output と seed を表示する。"""
    result = CliRunner().invoke(cli, ["ble-sample", "--help"])

    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--seed" in result.output
