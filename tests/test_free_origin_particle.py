"""初期位置・初期方位を与えないPFの一様初期化を検証する。"""

from importlib import import_module
from unittest.mock import Mock

import matplotlib.image as mpimg
import numpy as np
import pytest
from click.testing import CliRunner

from rikka.cli import commands as cli_commands
from rikka.cli.options import cli
from rikka.common.lib import sensors as sensors_module
from rikka.common.lib.floormap import compute_pixel_coords
from rikka.common.lib.models import StepHeading, StepMotionEvidence
from rikka.common.settings import ParticleSettings
from rikka.particle.lib import runner as runner_module
from rikka.particle.lib.global_init import (
    sample_uniform_headings,
    sample_uniform_walkable_particles,
    walkable_area_m2,
)
from rikka.particle.lib.landmark import meter_walkable_mask
from rikka.particle.lib.runner import run_particle_steps


def _forward_heading(index: int, heading: float) -> StepHeading:
    """確定済みの前進歩情報を作る。"""
    return StepHeading(
        step_index=index,
        timestamp_s=float(index - 1),
        gyro_heading=heading,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=heading,
        source="trajectory_motion",
        confidence=0.0,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=heading,
        motion_heading=heading,
        movement_type="forward",
        forward_displacement=1.0,
        lateral_displacement=0.0,
        motion_confidence=1.0,
        motion_reject_reason=None,
        trajectory_movement_type="forward",
    )


@pytest.fixture
def pf_inputs(tmp_path):
    """既定起点は中央、一様初期化の座標原点は壁上になる地図を用意する。"""
    gray = np.full((40, 40), 255.0)
    gray[0, 0] = 0.0
    path = tmp_path / "map.png"
    mpimg.imsave(path, gray, cmap="gray", vmin=0.0, vmax=255.0)
    headings = [0.0, 0.2, 0.5]
    return dict(
        gx_mean=0.0,
        gz_mean=1.0,
        floormap_path=path,
        origin_px=(20, 20),
        scale=0.5,
        n_particles=12,
        seed=42,
        prepared_step_headings=[
            _forward_heading(i + 1, h) for i, h in enumerate(headings)
        ],
        prepared_step_lengths=[0.5] * 3,
        prepared_step_times=[0.0, 1.0, 2.0],
        prepared_motion_evidences=[StepMotionEvidence(0.95, 0.02, 0.02, 0.01, 1.0, 1.0)]
        * 3,
        prepared_particle_motion_headings=headings,
    )


@pytest.mark.parametrize("gravity", [(0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0)])
def test_uniform_samples_only_walkable_cells(gravity) -> None:
    """孤立画素や地図端でも既存の画素中心規約に一致する。"""
    gray = np.array([[255, 0, 128], [0, 255, 0], [129, 0, 255]])
    particles = sample_uniform_walkable_particles(
        gray, 4000, *gravity, (2, 1), 0.2, np.random.default_rng(10)
    )
    assert particles.shape == (4000, 2)
    assert meter_walkable_mask(particles, gray, *gravity, (2, 1), 0.2).all()
    xs, ys = compute_pixel_coords(
        particles[:, 0], particles[:, 1], *gravity, (2, 1), 0.2
    )
    cells = np.floor(ys + 0.5).astype(int) * 3 + np.floor(xs + 0.5).astype(int)
    counts = np.bincount(cells, minlength=9)[[0, 4, 6, 8]]
    assert np.all((counts > 850) & (counts < 1150))
    assert walkable_area_m2(gray, 0.2) == pytest.approx(0.16)


def test_uniform_samples_reject_empty_walkable_area() -> None:
    """歩行可能画素がなければ理由付きで失敗する。"""
    with pytest.raises(ValueError, match="歩行可能画素"):
        sample_uniform_walkable_particles(
            np.zeros((2, 2)), 10, 0.0, 1.0, (0, 0), 1.0, np.random.default_rng(0)
        )


def test_uniform_headings_cover_full_circle() -> None:
    """初期方位は全周を覆い [-π, π) に収まる。"""
    headings = sample_uniform_headings(4000, np.random.default_rng(0))
    assert np.all((headings >= -np.pi) & (headings < np.pi))
    assert abs(np.mean(np.exp(1j * headings))) < 0.05


@pytest.mark.parametrize(
    "extra", [["--origin-px", "0", "0"], ["--direction", "0"], ["--ble-landmark"]]
)
def test_uniform_cli_rejects_initial_conditions(tmp_path, extra) -> None:
    """既定と同じ値でも明示された起点・方位やBLEを拒否する。"""
    result = CliRunner().invoke(
        cli, ["particle", "--pf-init", "uniform", "--data-dir", str(tmp_path), *extra]
    )
    assert result.exit_code == 2
    assert "--pf-init uniform" in result.output
    assert extra[0] in result.output


def test_uniform_cli_reads_only_height(tmp_path, monkeypatch) -> None:
    """起点と方位のない計測設定から身長だけを読み込む。"""
    (tmp_path / "walk_config.csv").write_text(
        "origin_x_px,origin_y_px,initial_direction_deg,user_height_m\n"
        "invalid,invalid,invalid,1.75\n"
    )
    sensor_load = Mock(return_value=(Mock(), Mock()))
    run = Mock()
    monkeypatch.setattr(sensors_module, "load_sensor_data", sensor_load)
    monkeypatch.setattr(cli_commands, "run", run)
    result = CliRunner().invoke(
        cli,
        ["particle", "--pf-init", "uniform", "--data-dir", str(tmp_path), "--no-plot"],
    )
    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs["init_mode"] == "uniform"
    assert run.call_args.kwargs["origin_px"] == (0, 0)
    assert run.call_args.kwargs["initial_direction"] == 0.0
    assert run.call_args.kwargs["height_m"] == 1.75


@pytest.mark.parametrize("seed", [0, 1, 2, 10, 42, 100])
@pytest.mark.parametrize("proposal", ["legacy", "no-relocation", "candidate-bank"])
def test_uniform_runner_returns_delayed_particle_path(
    pf_inputs, monkeypatch, seed, proposal
) -> None:
    """分散した履歴にクラスタ経路選択や局所再配置を適用しない。"""
    finalize_module = import_module("rikka.particle.lib.finalize")
    propose_module = import_module("rikka.particle.lib.propose")
    forbidden = Mock(side_effect=AssertionError("一様初期化では呼べません"))
    monkeypatch.setattr(finalize_module, "_select_sequence_map_path", forbidden)
    monkeypatch.setattr(finalize_module, "_select_reachable_cluster_path", forbidden)
    monkeypatch.setattr(propose_module, "relocate_invalid_particles", forbidden)
    stages, diagnostics, paths = [], [], []
    result = run_particle_steps(
        **(pf_inputs | dict(origin_px=(0, 0), seed=seed)),
        init_mode="uniform",
        proposal_method=proposal,
        candidate_count=3 if proposal == "candidate-bank" else 1,
        stage_collector=stages,
        diagnostics_collector=diagnostics,
        path_comparison_collector=paths,
    )
    assert len(result[0]) == 4
    assert result[3].shape == (4, 12, 2)
    for stage in stages:
        assert stage.recovery_mode == "none"
        assert np.all(stage.posterior_weights[~stage.valid_transition] == 0.0)
    assert any(
        np.array_equal(result[0], candidate) for candidate in paths[0].particle_paths
    )
    assert paths[0].selected_mode == "uniform_delayed"
    assert all(d.trajectory_mode == "uniform_delayed" for d in diagnostics)
    assert all(d.trajectory_source_index is None for d in diagnostics)


def test_uniform_all_dead_raises_without_recovery(pf_inputs, monkeypatch) -> None:
    """全滅時はlegacy再配置・全復旧経路を呼ばず明示的に失敗する。"""
    evaluate = import_module("rikka.particle.lib.evaluate_map")
    forbidden = Mock(side_effect=AssertionError("復旧してはいけません"))
    for name in (
        "_try_local_recovery",
        "_try_checkpoint_replay",
        "_apply_fallback_recovery",
        "_hold_on_recovery_failure",
    ):
        monkeypatch.setattr(evaluate, name, forbidden)
    with pytest.raises(RuntimeError, match="一様初期化の全粒子が地図制約で無効"):
        run_particle_steps(
            **(pf_inputs | dict(prepared_step_lengths=[1000.0] * 3)),
            init_mode="uniform",
        )
    forbidden.assert_not_called()


def test_origin_fixed_seed_matches_pre_change_baseline(pf_inputs) -> None:
    """変更前の固定seedの代表経路を保持し、初期乱数消費も確認する。"""
    stages = []
    result = run_particle_steps(**pf_inputs, stage_collector=stages)
    np.testing.assert_allclose(
        result[0],
        [
            [0.0, 0.0],
            [0.47548451971064365, -0.005104733583323384],
            [0.9438767007325499, 0.09008872666137073],
            [1.3718967606315453, 0.3111069440690287],
        ],
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_array_equal(stages[0].before_positions, np.zeros((12, 2)))
    expected = np.random.default_rng(42).normal(0.0, 0.26, 12)
    expected = (expected + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_array_equal(stages[0].before_offsets, expected)


@pytest.mark.parametrize("mode", ["origin", "uniform"])
def test_explicit_initial_state_has_priority(pf_inputs, mode) -> None:
    """指定した位置と方位のコピーを初期状態に採用する。

    方位オフセットは減衰しない補正項と減衰するドリフト項の和なので、
    後方PFのように初期状態を外から与える場合は両方を指定する契約にする。
    """
    positions = np.full((12, 2), 1.0)
    corrections = np.linspace(-1.0, 1.0, 12)
    drifts = np.linspace(-0.2, 0.2, 12)
    stages = []
    result = run_particle_steps(
        **pf_inputs,
        init_mode=mode,
        initial_particles=positions,
        initial_heading_correction=corrections,
        initial_heading_drift=drifts,
        stage_collector=stages,
    )
    np.testing.assert_array_equal(result[3][0], positions)
    np.testing.assert_allclose(
        stages[0].before_offsets, corrections + drifts, atol=1e-15
    )
    np.testing.assert_array_equal(positions, np.ones((12, 2)))


@pytest.mark.parametrize(
    "initial",
    [
        dict(initial_particles=np.zeros((11, 2))),
        dict(initial_heading_drift=np.zeros((12, 1))),
    ],
)
def test_explicit_initial_state_rejects_wrong_shape(pf_inputs, initial) -> None:
    """初期状態の粒子数と配列次元を検証する。"""
    with pytest.raises(ValueError, match="形状"):
        run_particle_steps(**pf_inputs, init_mode="uniform", **initial)


def test_particle_settings_reject_unknown_init_mode() -> None:
    """未知の初期化方式は設定段階で拒否する。"""
    with pytest.raises(ValueError, match="init_mode"):
        ParticleSettings(init_mode="unknown")


@pytest.mark.parametrize("ess_ratio,full_resample", [(0.2, False), (0.5, True)])
def test_global_phase_keeps_survivors_without_recovery(
    pf_inputs, monkeypatch, ess_ratio, full_resample
) -> None:
    """低生存率でも復旧せず、大域探索専用ESS閾値だけで再標本化を選ぶ。"""
    propose_module = import_module("rikka.particle.lib.propose")
    evaluate_module = import_module("rikka.particle.lib.evaluate_map")
    monkeypatch.setattr(
        propose_module,
        "_evaluate_particle_transitions",
        lambda *args: np.arange(12) < 4,
    )
    forbidden = Mock(side_effect=AssertionError("低生存率でも復旧しません"))
    monkeypatch.setattr(evaluate_module, "_try_local_recovery", forbidden)
    inputs = pf_inputs | {
        name: value[:1]
        for name, value in pf_inputs.items()
        if name.startswith("prepared_")
    }
    stages, diagnostics = [], []
    run_particle_steps(
        **inputs,
        init_mode="uniform",
        recovery_valid_ratio=1.0,
        global_resample_ess_ratio=ess_ratio,
        initial_particles=np.zeros((12, 2)),
        initial_heading_drift=np.zeros(12),
        stage_collector=stages,
        diagnostics_collector=diagnostics,
    )
    stage = stages[0]
    assert not diagnostics[0].recovery_attempted
    if full_resample:
        np.testing.assert_array_equal(stage.after_weights, np.full(12, 1.0 / 12))
    else:
        np.testing.assert_array_equal(stage.parent_indices[:4], np.arange(4))
        counts = np.bincount(stage.parent_indices, minlength=12)
        for parent in range(4):
            assert stage.after_weights[
                stage.parent_indices == parent
            ].sum() == pytest.approx(0.25)
            assert counts[parent] >= 1
    forbidden.assert_not_called()


def test_grid_clusters_ignore_zero_weights_and_rank_by_mass() -> None:
    """離れた分布を分離し、ゼロ重みの橋を連結に含めない。"""
    from rikka.particle.lib.convergence import cluster_distribution

    positions = np.array([[0.1, 0.1], [0.3, 0.1], [6.1, 0.1], [3.1, 0.1]])
    record, indices = cluster_distribution(positions, np.array([2.0, 1.0, 1.0, 0.0]))
    assert record.cluster_count == 2
    assert record.dominant_weight_ratio == pytest.approx(0.75)
    assert record.surviving_particle_count == 3
    assert [c.weight_ratio for c in record.top_clusters] == pytest.approx([0.75, 0.25])
    assert record.dominant_centroid_x_m == pytest.approx(1 / 6)
    assert record.dominant_spread_rms_m == pytest.approx(np.sqrt(2 / 225))
    np.testing.assert_array_equal(indices, [0, 1])
    assert record.position_spread_rms_m > 2.0


def test_grid_single_cluster_and_empty_distribution() -> None:
    """小さい集中分布と全ゼロ分布の両方を診断できる。"""
    from rikka.particle.lib.convergence import cluster_distribution

    positions = np.array([[-0.1, 0.1], [0.1, 0.1], [0.2, 0.1]])
    record, _ = cluster_distribution(positions, np.ones(3))
    assert record.cluster_count == 1
    assert record.dominant_weight_ratio == 1.0
    assert record.dominant_spread_rms_m < 0.2
    empty, indices = cluster_distribution(positions, np.zeros(3))
    assert empty.cluster_count == 0
    assert empty.convergence_step is None
    assert not len(indices)


@pytest.mark.parametrize("converges", [True, False])
def test_online_convergence_switches_recovery_path(
    pf_inputs, monkeypatch, converges
) -> None:
    """収束した次歩から通常の地図評価へ戻り、元の歩番号でKを保持する。"""
    runner = import_module("rikka.particle.lib.runner")
    resolve = runner.resolve_map_constraints
    finalize = runner.finalize
    phases = []
    captured = []

    def observe_phase(ctx):
        phases.append(ctx.global_phase_active)
        resolve(ctx)

    def observe_final(ctx):
        captured.append(ctx)
        return finalize(ctx)

    monkeypatch.setattr(runner, "resolve_map_constraints", observe_phase)
    monkeypatch.setattr(runner, "finalize", observe_final)
    records = []
    result = run_particle_steps(
        **pf_inputs,
        init_mode="uniform",
        initial_particles=np.full((12, 2), 5.0),
        initial_heading_correction=np.zeros(12),
        convergence_cell_m=100.0,
        convergence_spread_m=100.0 if converges else 0.0,
        convergence_single_steps=2,
        convergence_restore_normal=True,
        convergence_collector=records,
    )
    assert phases == ([True, True, False] if converges else [True] * 3)
    assert [r.convergence_step for r in records] == (
        [None, 2, 2] if converges else [None] * 3
    )
    ctx = captured[0]
    assert ctx.global_phase_active is not converges
    assert (ctx.convergence is not None) is converges
    from rikka.particle.lib.convergence import cluster_distribution

    _, indices = cluster_distribution(ctx.particles, ctx.weights, 100.0)
    best = indices[np.argmax(ctx.path_log_score_history[-1][indices])]
    np.testing.assert_array_equal(result[0], ctx.particle_paths[best])
    if converges:
        assert ctx.convergence.step_number == 2
        assert ctx.convergence == ctx.convergence_diagnostics[1]


def test_convergence_streak_resets_and_never_reopens_global_phase() -> None:
    """連続条件の中断と、確定後に分散しても大域探索へ戻さないことを確認する。"""
    from types import SimpleNamespace

    from rikka.particle.lib.convergence import record_convergence

    ctx = SimpleNamespace(
        particles=np.zeros((2, 2)),
        weights=np.ones(2),
        convergence_cell_m=1.5,
        convergence_top_k=3,
        convergence_weight_ratio=0.6,
        convergence_spread_m=0.1,
        convergence_single_steps=2,
        convergence_streak=0,
        convergence=None,
        convergence_diagnostics=[],
        convergence_collector=None,
        convergence_restore_normal=True,
        global_phase_active=True,
        step_number=0,
        step_time=0.0,
    )
    for step, spread in enumerate([0.0, 10.0, 0.0, 0.0, 10.0], start=1):
        ctx.step_number = step
        ctx.particles[1, 0] = spread
        record_convergence(ctx)
    assert [r.convergence_step for r in ctx.convergence_diagnostics] == [
        None,
        None,
        None,
        4,
        4,
    ]
    assert ctx.convergence.step_number == 4
    assert not ctx.global_phase_active


def test_origin_does_not_measure_convergence(pf_inputs, monkeypatch) -> None:
    """起点既知では収束処理を呼ばず、不使用の判定設定も影響させない。"""
    runner = import_module("rikka.particle.lib.runner")
    forbidden = Mock(side_effect=AssertionError("originでは収束処理を呼べません"))
    monkeypatch.setattr(runner, "record_convergence", forbidden)
    records = []
    run_particle_steps(
        **pf_inputs, convergence_collector=records, convergence_cell_m=0.0
    )
    assert records == []
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "parameters",
    [
        {"convergence_cell_m": 0.0},
        {"convergence_cell_m": float("nan")},
        {"convergence_top_k": 0},
        {"convergence_top_k": True},
        {"convergence_weight_ratio": 1.1},
        {"convergence_spread_m": -1.0},
        {"convergence_single_steps": 1.5},
    ],
)
def test_invalid_convergence_settings(pf_inputs, parameters) -> None:
    """直接runnerと設定経由の不正値を同じ基準で拒否する。"""
    with pytest.raises(ValueError, match="convergence"):
        ParticleSettings(init_mode="uniform", **parameters)
    with pytest.raises(ValueError, match="convergence"):
        run_particle_steps(**pf_inputs, init_mode="uniform", **parameters)


def test_uniform_pipeline_saves_convergence_with_no_plot(pf_inputs, tmp_path) -> None:
    """合成歩列で設定の伝搬、CSV列、非表示でのPNG保存を一緒に検証する。"""
    import pandas as pd

    from rikka.common.lib.models import (
        FloorMap,
        PreparedPdrSteps,
        StepDetectionResult,
        StepMotionPosterior,
    )
    from rikka.common.settings import OutputSettings, PdrSettings
    from rikka.particle.pipeline import run_particle
    from rikka.plot.pipeline import render, write_outputs

    prepared = PreparedPdrSteps(
        df_acc=pd.DataFrame(),
        df_gyro=pd.DataFrame(),
        step_detection=StepDetectionResult("peak", np.arange(3), (), None, None),
        trajectory=[[0.0, 0.0]] * 4,
        step_lengths=pf_inputs["prepared_step_lengths"],
        t_at_steps=pf_inputs["prepared_step_times"],
        step_headings=pf_inputs["prepared_step_headings"],
        gx_mean=0.0,
        gz_mean=1.0,
        weinberg_k=0.39,
        heading_method="gyro",
        motion_heading_correction="none",
        sidestep_smoothing="none",
        forward_heading_source="body",
        sidestep_heading_source="motion",
        sidestep_suspect_mode="motion",
        motion_evidences=tuple(pf_inputs["prepared_motion_evidences"]),
        particle_motion_headings=(0.0, 0.2, 0.5),
        motion_posteriors=tuple(
            StepMotionPosterior(
                i + 1,
                0.95,
                0.02,
                0.02,
                0.01,
                h,
                0.1,
                0.5,
                0.05,
                0.0,
                0.1,
                "forward",
                "synthetic",
            )
            for i, h in enumerate((0.0, 0.2, 0.5))
        ),
    )
    settings = ParticleSettings(
        init_mode="uniform",
        count=12,
        seed=42,
        convergence_cell_m=100.0,
        convergence_spread_m=100.0,
        convergence_single_steps=2,
    )
    result = run_particle(
        prepared,
        FloorMap(str(pf_inputs["floormap_path"]), (0, 0), 0.5),
        settings,
    )
    assert result.particle.convergence.step_number == 2
    write_outputs(result, tmp_path)
    render(
        result,
        PdrSettings(),
        settings,
        OutputSettings(plot=False, save_animation=False),
        tmp_path,
    )
    csv = pd.read_csv(tmp_path / "convergence_diagnostics.csv")
    assert list(csv.step_number) == [1, 2, 3]
    assert list(csv.cluster_count) == [1, 1, 1]
    np.testing.assert_allclose(csv["cluster_1_weight_ratio"], 1.0, atol=1e-14)
    assert np.isnan(csv.convergence_step.iloc[0])
    assert list(csv.convergence_step.iloc[1:]) == [2, 2]
    image = mpimg.imread(tmp_path / "convergence_history.png")
    assert image.shape[0] > 100 and image.shape[1] > 100


def test_convergence_uses_original_step_numbers_after_skip(pf_inputs) -> None:
    """実行されない歩は連続判定に数えず、Kは元の歩番号で保存する。"""
    records = []
    run_particle_steps(
        **(pf_inputs | {"prepared_particle_motion_headings": [0.0, None, 0.5]}),
        init_mode="uniform",
        initial_particles=np.full((12, 2), 5.0),
        initial_heading_correction=np.zeros(12),
        convergence_cell_m=100.0,
        convergence_spread_m=100.0,
        convergence_single_steps=2,
        convergence_collector=records,
    )
    assert [r.step_number for r in records] == [1, 3]
    assert [r.convergence_step for r in records] == [None, 3]


def test_convergence_cli_options_forward_values(tmp_path, monkeypatch) -> None:
    """新しい5オプションが互換アダプタへ渡ることを検証する。"""
    monkeypatch.setattr(
        sensors_module, "load_sensor_data", Mock(return_value=(Mock(), Mock()))
    )
    run = Mock()
    monkeypatch.setattr(cli_commands, "run", run)
    result = CliRunner().invoke(
        cli,
        [
            "particle",
            "--pf-init",
            "uniform",
            "--data-dir",
            str(tmp_path),
            "--pf-converge-cell",
            "2.0",
            "--pf-converge-top-k",
            "2",
            "--pf-converge-weight-ratio",
            "0.8",
            "--pf-converge-spread",
            "1.2",
            "--pf-converge-steps",
            "4",
            "--no-plot",
        ],
    )
    assert result.exit_code == 0, result.output
    for key, value in dict(
        convergence_cell_m=2.0,
        convergence_top_k=2,
        convergence_weight_ratio=0.8,
        convergence_spread_m=1.2,
        convergence_single_steps=4,
    ).items():
        assert run.call_args.kwargs[key] == value


def _spy_global_phase(monkeypatch) -> list[bool]:
    """runner が参照する収束判定を包み、各歩の大域フェーズ状態を集める。"""
    captured: list[bool] = []
    original = runner_module.record_convergence

    def spy(ctx) -> None:
        original(ctx)
        captured.append(ctx.global_phase_active)

    monkeypatch.setattr(runner_module, "record_convergence", spy)
    return captured


def test_convergence_keeps_global_phase_by_default(pf_inputs, monkeypatch) -> None:
    """既定では収束後も大域フェーズを維持する。

    4号館1024で収束後に通常PFへ戻すと、ESS閾値が 0.2 から 0.5 へ上がって
    全体再標本化が頻発し、正解系譜が失われて代表経路の平均誤差が
    0.78 m から 24.91 m へ悪化した。
    """
    captured = _spy_global_phase(monkeypatch)
    run_particle_steps(
        **pf_inputs,
        init_mode="uniform",
        convergence_weight_ratio=0.0001,
        convergence_spread_m=1e9,
        convergence_single_steps=1,
    )
    assert captured, "収束判定が1歩も呼ばれていません"
    assert all(captured), "既定で大域フェーズが解除されました"


def test_convergence_restores_normal_phase_when_requested(
    pf_inputs, monkeypatch
) -> None:
    """明示指定時は収束後に通常PFへ戻す。"""
    captured = _spy_global_phase(monkeypatch)
    run_particle_steps(
        **pf_inputs,
        init_mode="uniform",
        convergence_weight_ratio=0.0001,
        convergence_spread_m=1e9,
        convergence_single_steps=1,
        convergence_restore_normal=True,
    )
    assert captured and not captured[-1], "収束後も大域フェーズが解除されませんでした"
