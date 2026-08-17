"""BLE ランドマーク測位の単体テスト。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from rikka.ble.lib.detection import detect_landmarks
from rikka.ble.lib.loader import group_by_timestamp, load_ble_observations
from rikka.ble.lib.sample import (
    build_sample_times,
    generate_sample_observations,
    map_truth_to_step_times,
    resolve_walker_positions,
    write_sample_csv,
)
from rikka.ble.pipeline import run_ble_landmark_detection
from rikka.cli import commands as cli_commands
from rikka.cli.commands import run as run_command
from rikka.cli.options import cli
from rikka.common.config import FLOORMAP_PATH
from rikka.common.lib.floormap import compute_meter_coords, compute_pixel_coords
from rikka.common.lib.models import (
    BleObservation,
    FloorMap,
    Landmark,
    LandmarkCorrectionResult,
    LandmarkDetection,
)
from rikka.common.lib.sensors import load_sensor_data
from rikka.common.settings import (
    BleLandmarkSettings,
    BleSampleSettings,
    PdrSettings,
)
from rikka.landmark.lib.assignment import (
    assign_detections_to_steps,
    build_step_landmark_map,
)
from rikka.landmark.lib.timing import evaluate_landmark_timing
from rikka.pdr.lib.landmark_correction import apply_landmark_corrections
from rikka.pdr.pipeline import run_pdr
from rikka.plot import pipeline as plot_pipeline
from rikka.plot.lib.animation import plot_particle_filter_trajectory
from rikka.plot.lib.outputs import _build_landmark_corrections_dataframe
from rikka.plot.lib.trajectory import plot_trajectory


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


def test_load_ble_observations_wraps_empty_csv_error(tmp_path: Path) -> None:
    """0バイトのCSVを日本語の ValueError として報告する。"""
    path = tmp_path / "empty.csv"
    path.touch()

    with pytest.raises(ValueError, match="CSV として読み込めません"):
        load_ble_observations(path)


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


def test_group_by_timestamp_uses_window_start_as_anchor() -> None:
    """許容窓が時刻差の連鎖で伸びないことを確認する。"""
    observations = (
        BleObservation(0.00, "b1", -50.0),
        BleObservation(0.04, "b2", -60.0),
        BleObservation(0.08, "b3", -70.0),
    )

    groups = group_by_timestamp(observations, window_s=0.05)

    assert [timestamp for timestamp, _ in groups] == [0.00, 0.08]
    assert [len(group) for _, group in groups] == [2, 1]


def _landmark_settings(**overrides: object) -> BleLandmarkSettings:
    values: dict[str, object] = {
        "landmarks": (
            Landmark("beacon_1", 1.0, 2.0),
            Landmark("beacon_2", 3.0, 4.0),
            Landmark("beacon_3", 5.0, 6.0),
        ),
    }
    values.update(overrides)
    return BleLandmarkSettings(**values)  # type: ignore[arg-type]


def _apply_corrections(
    trajectory: list[list[float]],
    t_at_steps: list[float],
    detections: tuple[LandmarkDetection, ...],
    settings: BleLandmarkSettings,
    data_path: str,
    gx_mean: float,
    gz_mean: float,
    *,
    floormap: FloorMap | None = None,
) -> LandmarkCorrectionResult:
    """テスト設定をPDR固有のランドマーク補正境界へ展開する。"""
    return apply_landmark_corrections(
        trajectory,
        t_at_steps,
        detections=detections,
        landmarks=settings.landmarks,
        floormap=floormap or FloorMap("map.png", (0, 0), 1.0),
        data_path=data_path,
        rssi_threshold_dbm=settings.rssi_threshold_dbm,
        gx_mean=gx_mean,
        gz_mean=gz_mean,
    )


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


def test_detect_landmarks_reports_peak_timestamp_not_rising_edge() -> None:
    """検出時刻には閾値の立ち上がりではなく区間内最大 RSSI の時刻を使う。"""
    detections = detect_landmarks(
        _observations([-60, -54, -50, -46, -50, -54, -60, -61]),
        _landmark_settings(),
    )

    assert detections == (LandmarkDetection(3.0, "beacon_1", -46.0),)


def test_detect_landmarks_flushes_latched_beacon_at_end_of_data() -> None:
    """記録終端でラッチ中のビーコンも最大 RSSI 時刻で確定する。"""
    detections = detect_landmarks(
        _observations([-60, -54, -48, -46]),
        _landmark_settings(),
    )

    assert detections == (LandmarkDetection(3.0, "beacon_1", -46.0),)


def test_detect_landmarks_returns_time_sorted_detections() -> None:
    """解除による確定順が前後しても検出列はピーク時刻順になる。"""
    observations = (
        BleObservation(0.0, "beacon_1", -50.0),
        BleObservation(1.0, "beacon_2", -50.0),
        BleObservation(2.0, "beacon_2", -60.0),
        BleObservation(3.0, "beacon_2", -61.0),
        BleObservation(4.0, "beacon_1", -60.0),
        BleObservation(5.0, "beacon_1", -61.0),
    )

    detections = detect_landmarks(observations, _landmark_settings())

    assert [item.beacon_id for item in detections] == ["beacon_1", "beacon_2"]


def test_assign_detections_accepts_ndarray_step_times() -> None:
    """歩時刻に ndarray を渡しても検出を割り当てられる。"""
    assigned, discarded = assign_detections_to_steps(
        (_detection(1.5),),
        np.asarray([1.0, 2.0]),
    )

    assert assigned == {1: [_detection(1.5)]}
    assert discarded == 0


def test_detect_landmarks_redetects_after_release() -> None:
    """解除後に再び閾値以上になれば再検出することを確認する。"""
    detections = detect_landmarks(
        _observations([-50, -59, -60, -50]),
        _landmark_settings(),
    )

    assert [item.timestamp_s for item in detections] == [0.0, 3.0]


def test_detect_landmarks_release_streak_one_releases_immediately() -> None:
    """release_streak=1 なら1サンプルの低下後に再検出できる。"""
    detections = detect_landmarks(
        _observations([-50, -59, -50]),
        _landmark_settings(release_streak=1),
    )

    assert [item.timestamp_s for item in detections] == [0.0, 2.0]


def test_landmark_settings_rejects_invalid_release_streak() -> None:
    """解除連続回数は1以上の整数だけを受け付ける。"""
    with pytest.raises(ValueError, match="release_streak"):
        _landmark_settings(release_streak=0)


def test_default_landmarks_are_walkable_map_pixels() -> None:
    """既定ランドマークが実フロアマップの歩行可能画素にある。"""
    map_gray = cli_commands._load_floormap_gray(FLOORMAP_PATH)
    landmarks = BleLandmarkSettings().landmarks

    assert landmarks == (
        Landmark("beacon_1", 2056, 2400),
        Landmark("beacon_2", 750, 1479),
        Landmark("beacon_3", 2056, 700),
    )
    cli_commands._validate_landmark_pixels(map_gray, landmarks)


@pytest.mark.parametrize(("gx_mean", "gz_mean"), [(0.0, 1.0), (0.0, -1.0)])
def test_floormap_meter_conversion_is_pixel_conversion_inverse(
    gx_mean: float,
    gz_mean: float,
) -> None:
    """共有メートル変換が画素変換の逆変換になる。"""
    xs = np.array([-2.0, 0.0, 3.5])
    ys = np.array([1.0, -4.0, 2.5])
    pixel_xs, pixel_ys = compute_pixel_coords(
        xs,
        ys,
        gx_mean,
        gz_mean,
        origin_px=(100, 200),
        scale=0.1,
    )

    actual_xs, actual_ys = compute_meter_coords(
        pixel_xs,
        pixel_ys,
        gx_mean,
        gz_mean,
        origin_px=(100, 200),
        scale=0.1,
    )

    np.testing.assert_allclose(actual_xs, xs)
    np.testing.assert_allclose(actual_ys, ys)


def test_landmark_pixel_validation_rejects_wall_and_out_of_bounds() -> None:
    """壁上と地図範囲外のランドマークを拒否する。"""
    map_gray = np.full((4, 4), 255.0)
    map_gray[2, 2] = 0.0

    with pytest.raises(ValueError, match=r"wall.*\(2.0, 2.0\)"):
        cli_commands._validate_landmark_pixels(
            map_gray,
            (Landmark("wall", 2.0, 2.0),),
        )
    with pytest.raises(ValueError, match="outside"):
        cli_commands._validate_landmark_pixels(
            map_gray,
            (Landmark("outside", 10.0, 10.0),),
        )


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


def test_detect_landmarks_selects_strongest_within_sync_window() -> None:
    """受信時刻が数十 ms ずれても最大 RSSI のビーコンだけを検出する。"""
    observations = (
        BleObservation(10.000, "beacon_1", -52.0),
        BleObservation(10.013, "beacon_2", -45.0),
        BleObservation(10.027, "beacon_3", -70.0),
    )

    detections = detect_landmarks(observations, _landmark_settings())

    assert detections == (LandmarkDetection(10.013, "beacon_2", -45.0),)


def test_detect_landmarks_does_not_merge_separate_rounds() -> None:
    """同期窓より離れた受信周期が別グループになることを確認する。"""
    observations = (
        BleObservation(1.0, "beacon_1", -50.0),
        BleObservation(1.1, "beacon_2", -45.0),
    )

    detections = detect_landmarks(observations, _landmark_settings())

    assert [item.beacon_id for item in detections] == ["beacon_1", "beacon_2"]


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
    sample_settings = BleSampleSettings(mode="time")
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


def _distance_sample_settings(**overrides: object) -> BleSampleSettings:
    values: dict[str, object] = {
        "mode": "distance",
        "sigma_m": 1.0,
        "noise_sigma_db": 0.0,
    }
    values.update(overrides)
    return BleSampleSettings(**values)  # type: ignore[arg-type]


def test_distance_sample_peaks_when_walker_is_nearest() -> None:
    """距離方式の RSSI 最大時刻はビーコンへの最接近時刻に一致する。"""
    times = np.arange(5.0)
    walker = np.column_stack((times, np.zeros_like(times)))

    observations = generate_sample_observations(
        times,
        _distance_sample_settings(),
        walker_positions=walker,
        landmark_positions={"beacon_1": (2.0, 0.0)},
    )

    peak = max(observations, key=lambda item: item.rssi_dbm)
    assert peak.timestamp_s == 2.0


def test_build_sample_times_clamps_small_negative_sensor_start() -> None:
    """センサー開始時刻が負でも BLE CSV の時刻は0以上にする。"""
    times = build_sample_times(
        pd.DataFrame({"t": [-0.003, 0.097, 0.197]}),
        _distance_sample_settings(interval_s=0.1),
    )

    np.testing.assert_allclose(times, [0.0, 0.1])


def test_distance_sample_creates_two_peaks_for_revisited_beacon() -> None:
    """同じビーコンを再訪する軌跡では2回の検出が自然に生じる。"""
    times = np.arange(6.0)
    walker = np.asarray(
        [[0.0, 0.0], [5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [0.0, 0.0]]
    )
    observations = generate_sample_observations(
        times,
        _distance_sample_settings(),
        walker_positions=walker,
        landmark_positions={"beacon_1": (0.0, 0.0)},
    )

    detections = detect_landmarks(observations, _landmark_settings())

    assert [item.timestamp_s for item in detections] == [0.0, 5.0]


def test_distance_sample_yields_no_detection_for_far_beacon() -> None:
    """軌跡が近づかないビーコンの RSSI は検出閾値を超えない。"""
    times = np.arange(5.0)
    walker = np.zeros((len(times), 2))
    observations = generate_sample_observations(
        times,
        _distance_sample_settings(),
        walker_positions=walker,
        landmark_positions={"beacon_1": (100.0, 0.0)},
    )

    assert detect_landmarks(observations, _landmark_settings()) == ()


def test_distance_sample_is_deterministic_for_same_seed() -> None:
    """距離方式も同一 seed と入力から同一観測列を生成する。"""
    times = np.arange(5.0)
    walker = np.column_stack((times, np.zeros_like(times)))
    kwargs = {
        "walker_positions": walker,
        "landmark_positions": {"beacon_1": (2.0, 0.0)},
    }

    first = generate_sample_observations(
        times,
        _distance_sample_settings(noise_sigma_db=1.5, seed=7),
        **kwargs,  # type: ignore[arg-type]
    )
    second = generate_sample_observations(
        times,
        _distance_sample_settings(noise_sigma_db=1.5, seed=7),
        **kwargs,  # type: ignore[arg-type]
    )

    assert first == second


def test_resolve_walker_positions_interpolates_step_trajectory() -> None:
    """歩間の BLE サンプル位置を線形補間する。"""
    actual = resolve_walker_positions(
        np.asarray([0.0, 0.5, 1.0, 1.5, 2.0]),
        [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]],
        [1.0, 2.0],
    )

    np.testing.assert_allclose(
        actual,
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
    )


def test_map_truth_to_step_times_uses_normalized_arclength() -> None:
    """正解軌跡を参照軌跡の正規化弧長位置へ写像する。"""
    mapped, times = map_truth_to_step_times(
        np.asarray([[0.0, 0.0], [0.0, 10.0]]),
        [1.0, 2.0],
        [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
    )

    np.testing.assert_allclose(mapped, [[0.0, 0.0], [0.0, 10.0 / 3.0], [0.0, 10.0]])
    assert times == [0.0, 1.0, 2.0]


def test_landmark_timing_matches_nearest_revisited_approach() -> None:
    """同一ランドマーク再訪時は検出時刻に近い局所最接近と対応付ける。"""
    metrics = evaluate_landmark_timing(
        [[0.0, 0.0], [5.0, 0.0], [0.0, 0.0]],
        [5.0, 10.0],
        9.8,
        (0.0, 0.0),
    )

    assert metrics.nearest_approach_time_s == 10.0
    assert metrics.nearest_approach_delta_s == pytest.approx(-0.2)
    assert metrics.detection_distance_m == pytest.approx(0.2)


def _detection(
    timestamp_s: float,
    beacon_id: str = "beacon_1",
) -> LandmarkDetection:
    return LandmarkDetection(timestamp_s, beacon_id, -50.0)


def test_apply_landmark_corrections_moves_position_to_landmark() -> None:
    """補正が起きた歩の座標が変換後のランドマーク座標に一致する。"""
    result = _apply_corrections(
        [[0.0, 0.0], [1.0, 0.0]],
        [1.0],
        (_detection(1.0),),
        _landmark_settings(),
        "ble.csv",
        0.0,
        1.0,
    )

    assert result.trajectory[-1] == [1.0, 2.0]


def test_apply_landmark_corrections_converts_pixel_landmark_to_meter() -> None:
    """ピクセル座標を起点・縮尺・画素Y方向でPDR座標へ変換する。"""
    result = _apply_corrections(
        [[0.0, 0.0], [1.0, 0.0]],
        [1.0],
        (_detection(1.0),),
        _landmark_settings(
            landmarks=(Landmark("beacon_1", 130.0, 160.0),),
        ),
        "ble.csv",
        0.0,
        -1.0,
        floormap=FloorMap("map.png", (100, 200), 0.1),
    )

    assert result.trajectory[-1] == pytest.approx([3.0, 4.0])
    assert result.corrections[0].landmark_x == pytest.approx(3.0)
    assert result.corrections[0].landmark_y == pytest.approx(4.0)


def test_apply_landmark_corrections_continues_from_corrected_position() -> None:
    """補正後の歩は元 PDR の変位を保って補正座標から続く。"""
    result = _apply_corrections(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        [1.0, 2.0, 3.0],
        (_detection(2.0),),
        _landmark_settings(
            landmarks=(Landmark("beacon_1", 10.0, 5.0),),
        ),
        "ble.csv",
        0.0,
        1.0,
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

    result = _apply_corrections(
        trajectory,
        [1.0],
        (_detection(1.0),),
        _landmark_settings(),
        "ble.csv",
        0.0,
        1.0,
    )

    assert result.raw_trajectory == [[0.0, 0.0], [1.0, 0.0]]
    assert trajectory == [[0.0, 0.0], [1.0, 0.0]]


def test_apply_landmark_corrections_without_detection_is_identity() -> None:
    """検出が無い場合は元軌跡と一致する。"""
    trajectory = [[0.0, 0.0], [1.0, 0.0]]

    result = _apply_corrections(
        trajectory,
        [1.0],
        (),
        _landmark_settings(),
        "ble.csv",
        0.0,
        1.0,
    )

    assert result.trajectory == trajectory


def test_apply_landmark_corrections_rejects_empty_trajectory() -> None:
    """空軌跡は日本語の ValueError にする。"""
    with pytest.raises(ValueError, match="trajectory は 1 点以上"):
        _apply_corrections([], [], (), _landmark_settings(), "ble.csv", 0.0, 1.0)


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


def test_build_step_landmark_map_uses_one_based_last_registered_detection() -> None:
    """PF用マップは1始まり歩番号で、同一歩の最後の登録済み検出を使う。"""
    detections = (
        _detection(0.5, "unknown"),
        _detection(0.6, "beacon_1"),
        _detection(0.7, "beacon_2"),
    )

    result = build_step_landmark_map(
        detections,
        [1.0],
        {"beacon_1": (1.0, 2.0), "beacon_2": (3.0, 4.0)},
    )

    assert result == {1: _detection(0.7, "beacon_2")}


def test_apply_landmark_corrections_marks_last_detection_applied() -> None:
    """同じ歩に複数検出があるとき最後だけ applied になる。"""
    result = _apply_corrections(
        [[0.0, 0.0], [1.0, 0.0]],
        [1.0],
        (_detection(0.5, "beacon_1"), _detection(0.7, "beacon_2")),
        _landmark_settings(),
        "ble.csv",
        0.0,
        1.0,
    )

    assert [item.applied for item in result.corrections] == [False, True]
    assert result.trajectory[-1] == [3.0, 4.0]


def test_run_ble_landmark_detection_returns_none_when_disabled() -> None:
    """無効時は BLE ファイルを読まず None を返す。"""
    result = run_ble_landmark_detection(
        BleLandmarkSettings(enabled=False, data_path="missing.csv"),
    )

    assert result is None


def test_run_ble_landmark_detection_returns_detection_only(tmp_path: Path) -> None:
    """BLE pipeline が軌跡に依存せず共有検出型だけを返す。"""
    path = _write_ble_csv(
        tmp_path / "ble.csv",
        [{"timestamp_s": 1.0, "beacon_id": "beacon_1", "rssi_dbm": -50.0}],
    )

    result = run_ble_landmark_detection(
        _landmark_settings(enabled=True, data_path=path)
    )

    assert result == (LandmarkDetection(1.0, "beacon_1", -50.0),)


def test_run_pdr_without_ble_matches_prepared_trajectory() -> None:
    """BLE 無効時の run_pdr 軌跡が共有済み軌跡と一致する。"""
    df_acc, df_gyro = load_sensor_data()

    result = run_pdr(PdrSettings(), df_acc, df_gyro)

    assert result.landmark is None
    assert result.trajectory == result.prepared.trajectory


def test_run_pdr_with_ble_requires_floormap() -> None:
    """BLE補正有効時は共有地図設定を必須にする。"""
    settings = PdrSettings(
        landmark=BleLandmarkSettings(enabled=True, data_path="missing.csv")
    )

    with pytest.raises(ValueError, match="floormap が必要"):
        run_pdr(settings, pd.DataFrame(), pd.DataFrame())


def test_run_with_particle_filter_loads_ble_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PF のランドマーク有効時も BLE CSV を検出入力として読み込む。"""
    times = np.arange(5, dtype=float) * 0.01
    df_acc = pd.DataFrame(
        {
            "t": times,
            "x": np.zeros(5),
            "y": np.zeros(5),
            "z": np.full(5, 9.8),
        }
    )
    df_gyro = pd.DataFrame(
        {
            "t": times,
            "x": np.zeros(5),
            "y": np.zeros(5),
            "z": np.zeros(5),
        }
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(plot_pipeline, "create_output_dir", lambda: output_dir)

    with pytest.raises(ValueError, match="BLE データが存在しません"):
        run_command(
            df_acc=df_acc,
            df_gyro=df_gyro,
            plot=False,
            use_particle_filter=True,
            floormap_path=FLOORMAP_PATH,
            particle_seed=0,
            ble_landmark=True,
            ble_data_path=tmp_path / "missing.csv",
        )


def test_build_landmark_corrections_dataframe_has_diagnostic_columns() -> None:
    """補正履歴が必要な診断列を持つ DataFrame になる。"""
    result = _apply_corrections(
        [[0.0, 0.0], [1.0, 0.0]],
        [1.0],
        (_detection(1.0),),
        _landmark_settings(),
        "ble.csv",
        0.0,
        1.0,
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
            "detection_distance_m": 2.0,
            "nearest_approach_delta_s": 0.0,
        }
    ]


def test_build_landmark_corrections_dataframe_keeps_discarded_detection() -> None:
    """最終歩より後の検出も未適用行として診断CSVに残す。"""
    result = _apply_corrections(
        [[0.0, 0.0]],
        [],
        (_detection(4.0),),
        _landmark_settings(),
        "ble.csv",
        0.0,
        1.0,
    )

    dataframe = _build_landmark_corrections_dataframe(result)

    assert len(dataframe) == 1
    assert dataframe.loc[0, "step"] == -1
    assert dataframe.loc[0, "applied"] == np.False_
    assert pd.isna(dataframe.loc[0, "before_x"])


def test_plot_trajectory_labels_corrected_path_with_landmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BLE補正時は補正後軌跡を凡例へ追加する。"""
    map_path = tmp_path / "map.png"
    plt.imsave(map_path, np.ones((8, 8)), cmap="gray", vmin=0.0, vmax=1.0)
    landmark = _apply_corrections(
        [[0.0, 0.0], [1.0, 0.0]],
        [1.0],
        (_detection(1.0),),
        _landmark_settings(),
        "ble.csv",
        0.0,
        1.0,
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_trajectory(
        landmark.trajectory,
        floormap_path=map_path,
        output_dir=tmp_path,
        landmark=landmark,
    )

    labels = [artist.get_label() for artist in plt.gcf().axes[0].collections]
    assert "補正後軌跡" in labels
    plt.close("all")


def test_plot_particle_trajectory_overlays_landmark_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PF軌跡図にも通常PDRと同じランドマーク診断を重ねる。"""
    map_path = tmp_path / "map.png"
    plt.imsave(map_path, np.ones((8, 8)), cmap="gray", vmin=0.0, vmax=1.0)
    landmark = _apply_corrections(
        [[0.0, 0.0], [1.0, 0.0]],
        [1.0],
        (_detection(1.0),),
        _landmark_settings(),
        "ble.csv",
        0.0,
        1.0,
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_particle_filter_trajectory(
        landmark.trajectory,
        floormap_path=map_path,
        output_dir=tmp_path,
        landmark=landmark,
    )

    labels = [artist.get_label() for artist in plt.gcf().axes[0].collections]
    assert "ランドマーク反映後軌跡" in labels
    assert (tmp_path / "pf_trajectory.png").exists()
    plt.close("all")


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
    assert "--pf-landmark-mode" in result.output


def test_ble_sample_help_lists_options() -> None:
    """ble-sample --help が生成方式・位置ソース・出力設定を表示する。"""
    result = CliRunner().invoke(cli, ["ble-sample", "--help"])

    assert result.exit_code == 0
    assert "--mode" in result.output
    assert "--source" in result.output
    assert "--truth-csv" in result.output
    assert "--output" in result.output
    assert "--seed" in result.output
