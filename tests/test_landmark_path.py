"""ランドマークのみの軌跡と、初期位置・初期方位なしPDRの検証。"""

import json
from pathlib import Path
from unittest.mock import Mock

import matplotlib.image as mpimg
import numpy as np
import pytest
from click.testing import CliRunner

from rikka.cli import commands as cli_commands
from rikka.cli import options as cli_options
from rikka.cli.options import _parse_landmark_specs, cli
from rikka.common.config import BLE_LANDMARKS_PX
from rikka.common.lib import sensors as sensors_module
from rikka.common.lib.models import Landmark
from rikka.common.settings import OutputSettings
from rikka.landmark.lib.path import build_landmark_path
from rikka.landmark.pipeline import prepare_landmark_path
from rikka.plot.lib.outputs import _build_landmark_path_dataframe


def _write_floormap(path: Path, width: int = 200, height: int = 120) -> Path:
    """全画素が歩行可能な検証用フロアマップを作成する。"""
    mpimg.imsave(path, np.ones((height, width)), cmap="gray", vmin=0.0, vmax=1.0)
    return path


def test_build_landmark_path_connects_landmarks_in_given_order() -> None:
    """区間距離と累積距離が指定順の直線距離になる。"""
    path = build_landmark_path(
        [
            Landmark("lm_1", 100.0, 100.0),
            Landmark("lm_2", 200.0, 100.0),
            Landmark("lm_3", 200.0, 200.0),
        ],
        scale=0.05,
    )

    assert [point.beacon_id for point in path.points] == ["lm_1", "lm_2", "lm_3"]
    assert [point.order for point in path.points] == [0, 1, 2]
    assert path.points[0].segment_length_m == 0.0
    assert path.points[1].segment_length_m == pytest.approx(5.0)
    assert path.points[2].segment_length_m == pytest.approx(5.0)
    assert path.total_length_m == pytest.approx(10.0)
    assert path.points[-1].cumulative_length_m == pytest.approx(10.0)


def test_build_landmark_path_uses_first_landmark_as_meter_origin() -> None:
    """メートル座標は先頭ランドマーク基準で、画像下方向が -Y になる。"""
    path = build_landmark_path(
        [Landmark("lm_1", 100.0, 100.0), Landmark("lm_2", 140.0, 160.0)],
        scale=0.1,
    )

    assert (path.points[0].meter_x, path.points[0].meter_y) == (0.0, 0.0)
    assert path.points[1].meter_x == pytest.approx(4.0)
    assert path.points[1].meter_y == pytest.approx(-6.0)


@pytest.mark.parametrize(
    ("landmarks", "scale", "message"),
    [
        ([Landmark("lm_1", 1.0, 1.0)], 0.05, "2点以上"),
        ([Landmark("lm_1", 1.0, 1.0), Landmark("lm_2", 2.0, 2.0)], 0.0, "正の値"),
        (
            [Landmark("lm_1", 1.0, 1.0), Landmark("lm_2", float("nan"), 2.0)],
            0.05,
            "有限値",
        ),
    ],
)
def test_build_landmark_path_rejects_invalid_input(landmarks, scale, message) -> None:
    """点数、縮尺、座標の不正は構築時に検出する。"""
    with pytest.raises(ValueError, match=message):
        build_landmark_path(landmarks, scale=scale)


def test_landmark_path_dataframe_numbers_points_from_one() -> None:
    """CSV の order は1始まりで、区間距離と累積距離を併せて出力する。"""
    path = build_landmark_path(
        [Landmark("lm_1", 0.0, 0.0), Landmark("lm_2", 100.0, 0.0)],
        scale=0.05,
    )

    frame = _build_landmark_path_dataframe(path)

    assert list(frame.columns) == [
        "order",
        "beacon_id",
        "pixel_x",
        "pixel_y",
        "meter_x",
        "meter_y",
        "segment_length_m",
        "cumulative_length_m",
    ]
    assert frame["order"].tolist() == [1, 2]
    assert frame["cumulative_length_m"].tolist() == pytest.approx([0.0, 5.0])


def test_run_landmark_path_writes_csv_without_plot(tmp_path) -> None:
    """--no-plot 相当の実行でも経路CSVを指定ディレクトリへ保存する。"""
    floormap = _write_floormap(tmp_path / "map.png")
    output_dir = tmp_path / "out"

    result_dir = cli_commands.run_landmark_path(
        (("lm_1", 10.0, 10.0), ("lm_2", 110.0, 10.0)),
        floormap_path=floormap,
        scale=0.05,
        plot=False,
        output_dir=output_dir,
    )

    assert result_dir == output_dir
    assert (output_dir / "landmark_path.csv").is_file()
    assert not (output_dir / "landmark_path.png").exists()


def test_run_landmark_path_plots_and_saves_png(tmp_path) -> None:
    """描画ありの実行では経路図も保存する。"""
    floormap = _write_floormap(tmp_path / "map.png")
    output_dir = tmp_path / "out"

    cli_commands.run_landmark_path(
        (("lm_1", 10.0, 10.0), ("lm_2", 110.0, 60.0), ("lm_3", 150.0, 20.0)),
        floormap_path=floormap,
        scale=0.05,
        plot=True,
        output_dir=output_dir,
    )

    assert (output_dir / "landmark_path.png").is_file()


def test_run_landmark_path_rejects_landmarks_outside_floormap(tmp_path) -> None:
    """地図外のランドマークは実行前に拒否する。"""
    floormap = _write_floormap(tmp_path / "map.png")

    with pytest.raises(ValueError, match="フロアマップ内の画素"):
        cli_commands.run_landmark_path(
            (("lm_1", 10.0, 10.0), ("lm_2", 10_000.0, 10.0)),
            floormap_path=floormap,
            scale=0.05,
            plot=False,
            output_dir=tmp_path / "out",
        )


def test_parse_landmark_specs_requires_id_and_two_coordinates() -> None:
    """CLI指定は beacon_id:X:Y の形式だけを受け付ける。"""
    assert _parse_landmark_specs(("lm_1:10:20",)) == (("lm_1", 10.0, 20.0),)

    with pytest.raises(cli_options.click.UsageError):
        _parse_landmark_specs(("lm_1:10",))
    with pytest.raises(cli_options.click.UsageError):
        _parse_landmark_specs(("lm_1:10:abc",))
    with pytest.raises(cli_options.click.UsageError):
        _parse_landmark_specs((":10:20",))


def test_landmark_path_prefers_measurement_metadata_over_config(tmp_path) -> None:
    """計測フォルダに配置があれば設定既定より優先し、RSSIピーク順に並べる。"""
    metadata = {
        "recording_context": {"floor_id": "floor-1"},
        "ble": {
            "configuration": {
                "beacons": [
                    {
                        "beacon_id": "far",
                        "floor_id": "floor-1",
                        "pixel_x": 150.0,
                        "pixel_y": 20.0,
                        "enabled": True,
                    },
                    {
                        "beacon_id": "near",
                        "floor_id": "floor-1",
                        "pixel_x": 20.0,
                        "pixel_y": 20.0,
                        "enabled": True,
                    },
                ]
            }
        },
    }
    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    # near を先に、far を後に通過した記録として扱う。
    (tmp_path / "ble.csv").write_text(
        "timestamp_s,beacon_id,rssi_dbm\n"
        "1.0,near,-50\n2.0,near,-45\n3.0,near,-60\n"
        "4.0,far,-80\n5.0,far,-70\n6.0,far,-55\n",
        encoding="utf-8",
    )

    landmarks, source, ble_data_path = cli_options._resolve_landmark_path_inputs(
        str(tmp_path), ()
    )
    order_times = cli_options._resolve_landmark_path_order(ble_data_path, landmarks)
    _, path = prepare_landmark_path(landmarks, 0.05, order_times)

    assert "metadata.json" in source
    assert {item[0] for item in landmarks} == {"near", "far"}
    assert [point.beacon_id for point in path.points] == ["near", "far"]


def test_landmark_path_falls_back_to_config_without_measurement_placement(
    tmp_path,
) -> None:
    """配置を持たない計測フォルダでは設定既定を使い、出所を明示する。"""
    landmarks, source, ble_data_path = cli_options._resolve_landmark_path_inputs(
        str(tmp_path), ()
    )

    assert landmarks == BLE_LANDMARKS_PX
    assert "BLE_LANDMARKS_PX" in source
    assert ble_data_path is None


def test_landmark_specs_override_measurement_placement(tmp_path) -> None:
    """--landmark を指定したときは計測フォルダの配置を読まない。"""
    landmarks, source, ble_data_path = cli_options._resolve_landmark_path_inputs(
        str(tmp_path), ("lm_1:10:20", "lm_2:30:40")
    )

    assert landmarks == (("lm_1", 10.0, 20.0), ("lm_2", 30.0, 40.0))
    assert source == "--landmark の指定"
    assert ble_data_path is None


def test_prepare_landmark_path_drops_landmarks_without_passage_time() -> None:
    """通過時刻が無いランドマークは経路へ含めない。"""
    _, path = prepare_landmark_path(
        (("a", 0.0, 0.0), ("b", 100.0, 0.0), ("c", 200.0, 0.0)),
        0.05,
        {"c": 1.0, "a": 2.0},
    )

    assert [point.beacon_id for point in path.points] == ["c", "a"]


def test_landmark_path_command_requires_two_landmarks(tmp_path) -> None:
    """単一ランドマーク指定は経路にならないため使用法エラーにする。"""
    floormap = _write_floormap(tmp_path / "map.png")

    result = CliRunner().invoke(
        cli,
        [
            "landmark-path",
            "--floormap",
            str(floormap),
            "--data-dir",
            str(tmp_path),
            "--landmark",
            "lm_1:10:20",
        ],
    )

    assert result.exit_code == 2


def test_landmark_path_requires_two_received_landmarks(tmp_path) -> None:
    """RSSIピーク順で受信ありが2点未満になる計測は使用法エラーにする。"""
    metadata = {
        "recording_context": {"floor_id": "floor-1"},
        "ble": {
            "configuration": {
                "beacons": [
                    {
                        "beacon_id": "received",
                        "floor_id": "floor-1",
                        "pixel_x": 20.0,
                        "pixel_y": 20.0,
                        "enabled": True,
                    },
                    {
                        "beacon_id": "silent",
                        "floor_id": "floor-1",
                        "pixel_x": 60.0,
                        "pixel_y": 20.0,
                        "enabled": True,
                    },
                ]
            }
        },
    }
    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    # silent は一度も受信していないため経路から外れる。
    (tmp_path / "ble.csv").write_text(
        "timestamp_s,beacon_id,rssi_dbm\n1.0,received,-50\n2.0,received,-45\n",
        encoding="utf-8",
    )
    floormap = _write_floormap(tmp_path / "map.png")

    result = CliRunner().invoke(
        cli,
        [
            "landmark-path",
            "--floormap",
            str(floormap),
            "--data-dir",
            str(tmp_path),
            "--no-plot",
        ],
    )

    assert result.exit_code == 2
    assert "silent" in result.output


def test_landmark_path_order_distinguishes_missing_log_from_no_reception(
    tmp_path,
) -> None:
    """BLEログがあって既知ビーコンの受信が0件なら、空の順序として区別する。"""
    (tmp_path / "ble.csv").write_text(
        "timestamp_s,beacon_id,rssi_dbm\n1.0,other,-50\n2.0,other,-45\n",
        encoding="utf-8",
    )
    landmarks = (("a", 10.0, 10.0), ("b", 40.0, 10.0))

    # ログが無い場合だけ None（指定順へのフォールバック）になる。
    assert cli_options._resolve_landmark_path_order(None, landmarks) is None
    assert (
        cli_options._resolve_landmark_path_order(str(tmp_path / "ble.csv"), landmarks)
        == {}
    )


def test_landmark_path_rejects_log_without_any_reception(tmp_path) -> None:
    """既知ビーコンの受信が0件のログは、指定順へ黙って落とさず使用法エラーにする。"""
    metadata = {
        "recording_context": {"floor_id": "floor-1"},
        "ble": {
            "configuration": {
                "beacons": [
                    {
                        "beacon_id": "a",
                        "floor_id": "floor-1",
                        "pixel_x": 20.0,
                        "pixel_y": 20.0,
                        "enabled": True,
                    },
                    {
                        "beacon_id": "b",
                        "floor_id": "floor-1",
                        "pixel_x": 60.0,
                        "pixel_y": 20.0,
                        "enabled": True,
                    },
                ]
            }
        },
    }
    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    # 配置に無いビーコンだけが記録されたログ。
    (tmp_path / "ble.csv").write_text(
        "timestamp_s,beacon_id,rssi_dbm\n1.0,other,-50\n2.0,other,-45\n",
        encoding="utf-8",
    )
    floormap = _write_floormap(tmp_path / "map.png")

    result = CliRunner().invoke(
        cli,
        [
            "landmark-path",
            "--floormap",
            str(floormap),
            "--data-dir",
            str(tmp_path),
            "--no-plot",
        ],
    )

    assert result.exit_code == 2
    assert "--order given" in result.output


def test_landmark_path_reports_landmarks_without_reception(capsys) -> None:
    """受信が無いランドマークは、経路を中止せず除外として知らせる。"""
    landmarks = (("a", 10.0, 10.0), ("b", 40.0, 10.0), ("c", 70.0, 10.0))

    cli_options._report_landmarks_without_passage(
        landmarks, {"a": 1.0, "b": 2.0}, "metadata.json"
    )

    assert "BLE受信が無いため経路から除外: c" in capsys.readouterr().out


def test_free_origin_runs_pdr_from_origin_without_initial_direction(
    tmp_path, monkeypatch
) -> None:
    """--free-origin は (0, 0) 起点・方位0度でメートル座標の軌跡図を選ぶ。"""
    run_mock = Mock()
    monkeypatch.setattr(cli_options, "_resolve_measurement_settings", Mock())
    monkeypatch.setattr(
        sensors_module, "load_sensor_data", Mock(return_value=(None, None))
    )
    monkeypatch.setattr(cli_commands, "run", run_mock)

    result = CliRunner().invoke(
        cli, ["pdr", "--free-origin", "--data-dir", str(tmp_path), "--no-plot"]
    )

    assert result.exit_code == 0, result.output
    kwargs = run_mock.call_args.kwargs
    assert kwargs["initial_direction"] == 0.0
    assert kwargs["origin_px"] == (0, 0)
    assert kwargs["trajectory_plot"] == "meter"
    cli_options._resolve_measurement_settings.assert_not_called()


@pytest.mark.parametrize(
    "extra_args",
    [["--direction", "10"], ["--origin-px", "10", "20"], ["--ble-landmark"]],
)
def test_free_origin_conflicts_with_map_based_options(
    tmp_path, monkeypatch, extra_args
) -> None:
    """地図座標を前提とする指定との併用は使用法エラーにする。"""
    monkeypatch.setattr(cli_commands, "run", Mock())
    monkeypatch.setattr(
        sensors_module, "load_sensor_data", Mock(return_value=(None, None))
    )

    result = CliRunner().invoke(
        cli,
        ["pdr", "--free-origin", "--data-dir", str(tmp_path), *extra_args],
    )

    assert result.exit_code == 2
    assert "--free-origin" in result.output


def test_output_settings_rejects_unknown_trajectory_plot() -> None:
    """軌跡図の種類は floormap か meter に限定する。"""
    with pytest.raises(ValueError, match="trajectory_plot"):
        OutputSettings(trajectory_plot="pixel")


def test_free_origin_reads_only_user_height_from_measurement_config(
    tmp_path, monkeypatch
) -> None:
    """起点・初期方位を持たない計測設定でも身長だけを反映して実行できる。"""
    (tmp_path / "walk_config.csv").write_text("user_height_m\n1.80\n", encoding="utf-8")
    run_mock = Mock()
    monkeypatch.setattr(cli_commands, "run", run_mock)
    monkeypatch.setattr(
        sensors_module, "load_sensor_data", Mock(return_value=(None, None))
    )

    result = CliRunner().invoke(
        cli, ["pdr", "--free-origin", "--data-dir", str(tmp_path), "--no-plot"]
    )

    assert result.exit_code == 0, result.output
    assert run_mock.call_args.kwargs["height_m"] == pytest.approx(1.80)


def test_pdr_without_free_origin_keeps_floormap_trajectory_plot(
    tmp_path, monkeypatch
) -> None:
    """--free-origin を付けない従来実行は地図上の軌跡図のままにする。"""
    run_mock = Mock()
    monkeypatch.setattr(
        cli_options,
        "_resolve_measurement_settings",
        Mock(return_value=((10, 20), 90.0, 1.70)),
    )
    monkeypatch.setattr(cli_commands, "run", run_mock)
    monkeypatch.setattr(
        sensors_module, "load_sensor_data", Mock(return_value=(None, None))
    )

    result = CliRunner().invoke(cli, ["pdr", "--data-dir", str(tmp_path), "--no-plot"])

    assert result.exit_code == 0, result.output
    kwargs = run_mock.call_args.kwargs
    assert kwargs["trajectory_plot"] == "floormap"
    assert kwargs["origin_px"] == (10, 20)
    assert kwargs["initial_direction"] == 90.0


def test_output_settings_defaults_to_floormap_trajectory_plot() -> None:
    """既定の出力設定は従来どおり地図上の軌跡図を選ぶ。"""
    assert OutputSettings().trajectory_plot == "floormap"


def test_render_trajectory_keeps_existing_plots_for_default_settings(
    monkeypatch,
) -> None:
    """既定設定では通常PDRもPFも従来の描画へ進む。"""
    from rikka.plot import pipeline as plot_pipeline

    meter_plot = Mock()
    floormap_plot = Mock()
    particle_plot = Mock()
    monkeypatch.setattr(plot_pipeline, "plot_trajectory_meters", meter_plot)
    monkeypatch.setattr(plot_pipeline, "plot_trajectory", floormap_plot)
    monkeypatch.setattr(plot_pipeline, "plot_particle_filter_trajectory", particle_plot)

    plot_pipeline._render_trajectory(
        Mock(particle=None), Mock(), OutputSettings(), Path("output")
    )
    floormap_plot.assert_called_once()
    particle_plot.assert_not_called()

    # PF結果は meter 指定でも地図上の描画を使う。
    plot_pipeline._render_trajectory(
        Mock(), Mock(), OutputSettings(trajectory_plot="meter"), Path("output")
    )
    particle_plot.assert_called_once()
    meter_plot.assert_not_called()


def test_render_trajectory_uses_meter_plot_for_free_origin(
    tmp_path, monkeypatch
) -> None:
    """meter 指定の通常PDRでは地図へ合わせ込む描画を呼ばず、背景だけ渡す。"""
    from rikka.plot import pipeline as plot_pipeline

    floormap = _write_floormap(tmp_path / "map.png")
    meter_plot = Mock()
    floormap_plot = Mock()
    monkeypatch.setattr(plot_pipeline, "plot_trajectory_meters", meter_plot)
    monkeypatch.setattr(plot_pipeline, "plot_trajectory", floormap_plot)
    result = Mock(particle=None, trajectory=[[0.0, 0.0], [1.0, 0.0]])

    plot_pipeline._render_trajectory(
        result,
        Mock(floormap_path=str(floormap), scale=0.05),
        OutputSettings(trajectory_plot="meter"),
        Path("output"),
    )

    meter_plot.assert_called_once()
    assert meter_plot.call_args.kwargs["floormap_path"] == floormap
    floormap_plot.assert_not_called()


def test_render_trajectory_skips_background_when_floormap_missing(
    tmp_path, monkeypatch
) -> None:
    """地図ファイルが無くても (0, 0) 起点の軌跡だけで描画できる。"""
    from rikka.plot import pipeline as plot_pipeline

    meter_plot = Mock()
    monkeypatch.setattr(plot_pipeline, "plot_trajectory_meters", meter_plot)

    plot_pipeline._render_trajectory(
        Mock(particle=None, trajectory=[[0.0, 0.0], [1.0, 0.0]]),
        Mock(floormap_path=str(tmp_path / "missing.png"), scale=0.05),
        OutputSettings(trajectory_plot="meter"),
        Path("output"),
    )

    assert meter_plot.call_args.kwargs["floormap_path"] is None


@pytest.mark.parametrize(
    ("gx_mean", "gz_mean", "expected_bottom_m", "expected_inverted"),
    [(0.0, -9.8, -6.0, False), (-9.8, 0.0, 6.0, True)],
)
def test_plot_trajectory_meters_overlays_floormap_from_pixel_origin(
    tmp_path, gx_mean, gz_mean, expected_bottom_m, expected_inverted
) -> None:
    """端末姿勢の符号によらず、地図画素(0, 0)を起点に重ねた図を保存する。"""
    import matplotlib.pyplot as plt

    from rikka.plot.lib.trajectory import plot_trajectory_meters

    floormap = _write_floormap(tmp_path / "map.png", width=200, height=120)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    plot_trajectory_meters(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        output_dir=output_dir,
        floormap_path=floormap,
        scale=0.05,
        gx_mean=gx_mean,
        gz_mean=gz_mean,
    )

    # 画素(0, 0)が原点、画素行の増加方向は y_sign に従う向きで敷かれる。
    axes = plt.gcf().axes[0]
    image = axes.images[0]
    assert tuple(image.get_extent()) == (0.0, 10.0, expected_bottom_m, 0.0)
    assert image.origin == "upper"
    # 地図が画面上で上下反転しないよう、縦軸の向きまで固定する。
    assert bool(axes.yaxis_inverted()) is expected_inverted
    assert (output_dir / "trajectory_meter.png").is_file()
    plt.close("all")
