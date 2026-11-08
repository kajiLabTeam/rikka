"""合成地図・確定済み歩列で、検証専用後方PFと既定出力の互換性を検証する。"""

from dataclasses import replace
from hashlib import sha256
from unittest.mock import Mock

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from test_free_origin_particle import _forward_heading

from rikka.cli import commands
from rikka.cli.options import cli
from rikka.common.lib import sensors
from rikka.common.lib.models import (
    FloorMap,
    LandmarkDetection,
    LandmarkRange,
    PreparedPdrSteps,
    StepDetectionResult,
    StepDirectionPosterior,
    StepLengthObservation,
    StepMotionEvidence,
    StepMotionObservation,
    StepMotionPosterior,
)
from rikka.common.settings import OutputSettings, ParticleSettings, PdrSettings
from rikka.landmark.lib.assignment import assign_detections_to_steps
from rikka.particle import pipeline
from rikka.particle.lib.backward import (
    backward_to_forward,
    build_backward_prepared,
    reverse_observation_times,
    reverse_step_time,
)
from rikka.plot.pipeline import render, write_outputs


@pytest.fixture
def synthetic(tmp_path):
    """既存の一様初期化テストと同じ40画素の合成地図に閉じた歩列を置く。"""
    gray = np.full((40, 40), 255.0)
    gray[0, 0] = 0.0
    path = tmp_path / "map.png"
    mpimg.imsave(path, gray, cmap="gray", vmin=0.0, vmax=255.0)
    angles = [0.0, np.pi / 2, np.pi, -np.pi / 2]
    times = [3.0, 4.0, 6.5, 10.0]
    headings = [
        _forward_heading(i + 1, h)._replace(timestamp_s=times[i])
        for i, h in enumerate(angles)
    ]
    prepared = PreparedPdrSteps(
        df_acc=pd.DataFrame(),
        df_gyro=pd.DataFrame(),
        step_detection=StepDetectionResult("peak", np.arange(4), (), None, None),
        trajectory=[[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0], [0.0, 0.0]],
        step_lengths=[0.5, 1.0, 0.5, 1.0],
        t_at_steps=times,
        step_headings=headings,
        gx_mean=0.0,
        gz_mean=1.0,
        weinberg_k=0.39,
        heading_method="gyro",
        motion_heading_correction="none",
        sidestep_smoothing="none",
        forward_heading_source="body",
        sidestep_heading_source="motion",
        sidestep_suspect_mode="motion",
        motion_evidences=(StepMotionEvidence(1.0, 0.02, 0.03, 0.01, 1.0, 1.0),) * 4,
        particle_motion_headings=tuple(angles),
        motion_posteriors=tuple(
            StepMotionPosterior(
                i + 1,
                1.0,
                0.0,
                0.0,
                0.0,
                h,
                0.0,
                length,
                0.0,
                0.0,
                0.0,
                "forward",
                "synthetic",
            )
            for i, (h, length) in enumerate(
                zip(angles, [0.5, 1.0, 0.5, 1.0], strict=True)
            )
        ),
    )
    return prepared, FloorMap(str(path), (20, 20), 0.5)


def test_reverse_time_preserves_endpoints_intervals_and_observation_assignment():
    """不均等な歩間隔も、同時刻にある検出の歩への割当ても反転で保つ。"""
    times = [3.0, 4.0, 6.5, 10.0]
    reversed_times = [reverse_step_time(t, 3.0, 10.0) for t in reversed(times)]
    assert reversed_times[0] == times[0]
    assert reversed_times[-1] == times[-1]
    np.testing.assert_array_equal(np.diff(reversed_times), np.diff(times)[::-1])
    obs = (
        LandmarkDetection(4.0, "a", -50.0),
        LandmarkRange(6.5, "b", -60.0, 2.0, 1.0, -60.0),
    )
    reversed_obs = reverse_observation_times(obs, 3.0, 10.0)
    assert [o.beacon_id for o in reversed_obs] == ["b", "a"]
    assert isinstance(reversed_obs[0], LandmarkRange)
    original, _ = assign_detections_to_steps(obs, times)
    backward, _ = assign_detections_to_steps(reversed_obs, reversed_times)
    assert list(backward) == [3 - index for index in reversed(original)]
    assert reverse_observation_times(reversed_obs, 3.0, 10.0) == obs


def test_build_backward_rotates_all_absolute_angles_without_swapping_labels(synthetic):
    """欠測・左右・角度差を保持し、入力の可変コンテナも破壊しない。"""
    prepared, _ = synthetic
    prepared = replace(
        prepared,
        particle_motion_headings=(0.0, None, np.pi, -np.pi / 2),
        step_headings=[
            h._replace(
                movement_type="sidestep_left",
                trajectory_movement_type="sidestep_left",
                sidestep_cluster_id=7,
                angle_diff_method1=0.2,
            )
            for h in prepared.step_headings
        ],
        motion_observations=(
            StepMotionObservation(
                1,
                3.0,
                0.0,
                0.1,
                0.2,
                0.3,
                1.0,
                0.0,
                1.0,
                0.4,
                1.0,
                1.0,
                "forward",
                "forward",
                "normal",
            ),
        ),
        length_observations=(
            StepLengthObservation(1, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.1, None),
        ),
        direction_posteriors=(
            StepDirectionPosterior(
                1, 0.7, 0.3, 0.4, "forward", 0.7, "synthetic", False
            ),
        ),
    )
    original_headings = prepared.step_headings.copy()
    result = build_backward_prepared(prepared)
    assert result is not prepared
    assert result.df_acc is prepared.df_acc
    assert result.step_detection is prepared.step_detection
    for i, (actual, original) in enumerate(
        zip(result.step_headings, original_headings[::-1], strict=True), 1
    ):
        assert actual.step_index == i
        assert actual.timestamp_s == result.t_at_steps[i - 1]
        for name in (
            "selected_heading",
            "body_heading",
            "motion_heading",
            "gyro_heading",
        ):
            assert np.cos(
                getattr(actual, name) - getattr(original, name)
            ) == pytest.approx(-1.0)
        assert actual.accel_method1_heading is None
        assert actual.angle_diff_method1 == 0.2
        assert actual.movement_type == "sidestep_left"
        assert actual.trajectory_movement_type == "sidestep_left"
        assert actual.sidestep_cluster_id == 7
    assert result.particle_motion_headings[2] is None
    assert result.particle_motion_headings[-1] == -np.pi
    assert result.motion_evidences == prepared.motion_evidences[::-1]
    assert result.step_lengths == prepared.step_lengths[::-1]
    assert result.motion_posteriors[0].heading_mean == pytest.approx(np.pi / 2)
    assert result.motion_observations[0].yaw_delta == 0.4
    assert result.direction_posteriors[0].selected_heading == pytest.approx(0.4 - np.pi)
    assert result.gx_mean == prepared.gx_mean and result.gz_mean == prepared.gz_mean
    assert prepared.step_headings == original_headings
    result.trajectory[0][0] = 99.0
    result.step_lengths[0] = 99.0
    assert prepared.trajectory[-1][0] == 0.0
    assert prepared.step_lengths[-1] == 1.0
    assert backward_to_forward([[1.0, 2.0], [3.0, 4.0]]) == [[3.0, 4.0], [1.0, 2.0]]


@pytest.mark.parametrize("seed", [0, 42, 100])
@pytest.mark.parametrize("mode", ["origin", "uniform"])
def test_closed_walk_forward_backward_agree(synthetic, monkeypatch, seed, mode):
    """雑音ゼロの閉路で両方向のPFを実行し、全ノードの対応を確認する。"""
    prepared, floormap = synthetic
    original = pipeline.run_particle_steps

    def noiseless(*args, **kwargs):
        return original(
            *args,
            **(
                kwargs
                | dict(
                    sigma_init_heading=0.0,
                    sigma_heading=0.0,
                    sigma_sl_ratio=0.0,
                    stride_scale_init_sigma=0.0,
                    stride_scale_process_sigma=0.0,
                    stride_scale_rejuvenation_sigma=0.0,
                    initial_particles=np.zeros((12, 2)),
                    initial_heading_correction=np.zeros(12),
                    initial_heading_drift=np.zeros(12),
                )
            ),
        )

    monkeypatch.setattr(pipeline, "run_particle_steps", noiseless)
    settings = ParticleSettings(
        count=12, seed=seed, init_mode=mode, proposal_method="legacy"
    )
    forward = pipeline.run_particle(prepared, floormap, settings)
    backward = pipeline.run_backward_particle(
        prepared, floormap, settings, forward_result=forward
    )
    np.testing.assert_allclose(backward.trajectory, forward.trajectory, atol=1e-14)
    assert [item.step_number for item in backward.agreement] == [0, 1, 2, 3, 4]
    assert backward.max_distance_m < 1e-14
    assert backward.start_distance_m < 1e-14
    assert len(set(tuple(p) for p in forward.trajectory)) >= 4


def test_skip_mapping_and_existing_output_bytes(synthetic, tmp_path):
    """欠測歩を除いた対応を使い、後方を付けても既存CSVを一切変えない。"""
    prepared, floormap = synthetic
    prepared = replace(
        prepared, particle_motion_headings=(0.0, None, np.pi, -np.pi / 2)
    )
    settings = ParticleSettings(
        count=12,
        seed=42,
        init_mode="uniform",
        floormap_path=floormap.path,
        origin_px=floormap.origin_px,
        scale=floormap.scale,
    )
    forward = pipeline.run_particle(prepared, floormap, settings)
    assert forward.particle.backward is None
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    write_outputs(forward, before)
    expected = {p.name: p.read_bytes() for p in before.iterdir()}
    backward = pipeline.run_backward_particle(
        prepared, floormap, settings, forward_result=forward
    )
    assert [a.step_number for a in backward.agreement] == [0, 1, 3, 4]
    assert (
        backward.trajectory
        == pipeline.run_backward_particle(prepared, floormap, settings).trajectory
    )
    attached = replace(forward, particle=replace(forward.particle, backward=backward))
    write_outputs(attached, after)
    for name, content in expected.items():
        assert (after / name).read_bytes() == content
    assert pd.read_csv(after / "backward_trajectory.csv").columns.tolist() == [
        "timestamp_s",
        "x",
        "y",
    ]
    agreement = pd.read_csv(after / "bidirectional_agreement.csv")
    assert agreement.step_number.tolist() == [0, 1, 3, 4]
    render(
        attached,
        PdrSettings(),
        settings,
        OutputSettings(plot=False, save_animation=False),
        after,
    )
    assert mpimg.imread(after / "bidirectional_comparison.png").shape[0] > 100


@pytest.mark.parametrize(
    "flag, expected",
    [([], False), (["--pf-backward"], True), (["--no-pf-backward"], False)],
)
def test_cli_backward_defaults_off(tmp_path, monkeypatch, flag, expected):
    """CLIの既定値と明示指定を互換APIへ渡す。"""
    monkeypatch.setattr(
        sensors, "load_sensor_data", Mock(return_value=(Mock(), Mock()))
    )
    run = Mock()
    monkeypatch.setattr(commands, "run", run)
    result = CliRunner().invoke(
        cli, ["particle", "--data-dir", str(tmp_path), "--no-plot", *flag]
    )
    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs["pf_backward"] is expected
    help_result = CliRunner().invoke(cli, ["particle", "--help"])
    assert "--pf-backward" in help_result.output


@pytest.mark.parametrize("enabled", [False, True])
def test_commands_backward_is_opt_in(synthetic, tmp_path, monkeypatch, enabled):
    """互換APIは既定で後方を呼ばず、指定時だけ前向き結果へ追加する。"""
    prepared, floormap = synthetic
    settings = ParticleSettings(count=12, seed=42, init_mode="uniform")
    forward = pipeline.run_particle(prepared, floormap, settings)
    monkeypatch.setattr(commands, "run_pdr", Mock(return_value=forward))
    monkeypatch.setattr(commands, "run_particle", Mock(return_value=forward))
    backward = Mock(wraps=pipeline.run_backward_particle)
    monkeypatch.setattr(commands, "run_backward_particle", backward)
    from rikka.plot import pipeline as plot_pipeline

    monkeypatch.setattr(plot_pipeline, "create_output_dir", lambda: tmp_path)
    captured = Mock()
    monkeypatch.setattr(plot_pipeline, "render", captured)
    kwargs = dict(
        df_acc=prepared.df_acc,
        df_gyro=prepared.df_gyro,
        use_particle_filter=True,
        plot=False,
        save_animation=False,
        floormap_path=floormap.path,
        origin_px=floormap.origin_px,
        scale=floormap.scale,
        particle_seed=42,
        particle_count=12,
        init_mode="uniform",
    )
    if enabled:
        kwargs["pf_backward"] = True
    commands.run(**kwargs)
    result = captured.call_args.args[0]
    assert result.trajectory is forward.trajectory
    if enabled:
        backward.assert_called_once()
        assert backward.call_args.kwargs["forward_result"] is forward
        assert result.particle.backward is not None
        assert (tmp_path / "backward_trajectory.csv").exists()
    else:
        backward.assert_not_called()
        assert result.particle.backward is None
        assert not (tmp_path / "backward_trajectory.csv").exists()


def test_default_csv_bytes_match_pre_backward_baseline(synthetic, tmp_path):
    """652cc43の前向きPFで取得した全9既存CSVのバイト列を固定する。"""
    prepared, floormap = synthetic
    result = pipeline.run_particle(
        prepared, floormap, ParticleSettings(count=12, seed=42, init_mode="uniform")
    )
    assert result.particle.backward is None
    output_dir = tmp_path / "baseline"
    output_dir.mkdir()
    write_outputs(result, output_dir)
    files = sorted(output_dir.iterdir())
    assert len(files) == 9
    digest = sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    assert digest.hexdigest() == (
        "4a729d5edf6b8a9bf297b14973e342f2751406d847ef48c3335fe5fb861a807f"
    )
