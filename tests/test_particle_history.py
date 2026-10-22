"""前向き推定へ干渉しない過去軌跡候補の回帰を確認する。"""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from rikka import cli
from rikka.cli import commands
from rikka.common.lib.models import (
    FloorMap,
    PreparedPdrSteps,
    StepDetectionResult,
    StepHeading,
    StepMotionPosterior,
)
from rikka.common.settings import ParticleSettings
from rikka.particle.lib.history_correction import (
    _candidate_indices,
    build_history_candidate,
)
from rikka.particle.lib.map_constraints import (
    _evaluate_particle_transitions,
    _normalize_floormap_gray,
)
from rikka.particle.lib.recorder import ParticleStepStages
from rikka.particle.pipeline import run_particle
from rikka.pdr.lib.motion_state.evidence import (
    build_particle_motion_headings,
    build_step_motion_evidences,
)
from rikka.plot.lib.history import save_history_comparison, save_history_csv


def _stage(step, before, after, weights=None):
    before, after = np.asarray(before, float), np.asarray(after, float)
    n = len(after)
    weights = np.full(n, 1 / n) if weights is None else np.asarray(weights, float)
    delta = after - before
    return ParticleStepStages(
        step=step,
        timestamp_s=float(step),
        sensor_heading=0,
        sensor_yaw_delta=0,
        movement_type="forward",
        deterministic_step_length_m=1,
        before_positions=before.copy(),
        before_offsets=np.zeros(n),
        before_weights=weights.copy(),
        before_motion_state=np.zeros(n, int),
        proposed_positions=after.copy(),
        proposed_headings=np.arctan2(delta[:, 1], delta[:, 0]),
        proposed_step_lengths=np.linalg.norm(delta, axis=1),
        proposed_motion_state=np.zeros(n, int),
        valid_transition=np.ones(n, bool),
        posterior_weights=weights.copy(),
        ess_before_observation=float(n),
        ess_after_observation=float(n),
        parent_indices=np.arange(n),
        resampled=False,
        recovery_mode="none",
        recovery_candidate_headings=None,
        recovery_candidate_valid=None,
        recovery_selected_index=None,
        after_positions=after.copy(),
        after_offsets=np.zeros(n),
        after_weights=weights.copy(),
        after_motion_state=np.zeros(n, int),
        after_stride_scale=np.ones(n),
    )


@pytest.fixture
def floor(tmp_path):
    path = tmp_path / "map.png"
    plt.imsave(path, np.ones((40, 40)), cmap="gray", vmin=0, vmax=1)
    return FloorMap(str(path), (10, 10), 1.0)


def test_history_uses_supported_positions_and_keeps_inputs_and_boundaries(floor):
    original = [[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [6.0, 0.0]]
    stages = [
        _stage(1, [[0, 0]], [[2, 0]]),
        _stage(2, [[2, 0]], [[4, 0]]),
        _stage(3, [[4, 0]], [[6, 0]]),
    ]
    positions = [stage.after_positions.copy() for stage in stages]
    result = build_history_candidate(original, stages, floor, 0, 9.8)
    assert result.status == "candidate"
    np.testing.assert_array_equal(result.trajectory, [[0, 0], [2, 0], [4, 0], [6, 0]])
    assert result.objective_candidate < result.objective_original
    assert result.steps[1].source_particle == 0
    assert original == [[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [6.0, 0.0]]
    for stage, saved in zip(stages, positions, strict=True):
        np.testing.assert_array_equal(stage.after_positions, saved)


def test_history_candidate_budget_keeps_spatially_distinct_particles():
    stage = _stage(1, [[0, 0]] * 3, [[2, 0], [2, 0], [4, 0]], [0.6, 0.3, 0.1])
    indices = _candidate_indices(stage, np.array([2.0, 0.0]), 5.0, 2)
    assert indices.tolist() == [0, 2]


def test_history_does_not_claim_parent_edges_for_checkpoint_replay(floor):
    stages = [
        replace(_stage(1, [[0, 0]], [[2, 0]]), recovery_mode="checkpoint_replayed"),
        _stage(2, [[2, 0]], [[4, 0]]),
        _stage(3, [[4, 0]], [[6, 0]]),
    ]
    result = build_history_candidate(
        [[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [6.0, 0.0]], stages, floor, 0, 9.8
    )
    assert result.steps[1].source_particle == 0
    assert result.steps[1].connection == "replay_edge_unknown"


def test_history_rejects_wall_shortcuts(floor):
    gray = np.ones((40, 40))
    gray[:15, 15] = 0
    plt.imsave(floor.path, gray, cmap="gray", vmin=0, vmax=1)
    original = [[0.0, 0.0], [0.0, 8.0], [10.0, 8.0], [10.0, 0.0]]
    stages = [
        _stage(1, [[0, 0]], [[8, 0]]),
        _stage(2, [[8, 0]], [[8, 0]]),
        _stage(3, [[8, 0]], [[10, 0]]),
    ]
    result = build_history_candidate(
        original, stages, floor, 0, 9.8, search_radius_m=20
    )
    points = np.asarray(result.trajectory)
    assert _evaluate_particle_transitions(
        points[:-1],
        points[1:],
        _normalize_floormap_gray(plt.imread(floor.path)),
        0,
        9.8,
        floor.origin_px,
        1,
    ).all()
    assert result.trajectory[1] != [8.0, 0.0]


def test_history_does_not_select_zero_weight_particles(floor):
    original = [[0.0, 0.0], [4.0, 0.0], [1.0, 0.0], [6.0, 0.0]]
    stages = [
        _stage(1, [[0, 0], [0, 0]], [[2, 0], [4, 0]], [0, 1]),
        _stage(2, [[2, 0], [4, 0]], [[4, 0], [1, 0]], [0, 1]),
        _stage(3, [[4, 0], [1, 0]], [[6, 0], [6, 0]], [0, 1]),
    ]
    result = build_history_candidate(original, stages, floor, 0, 9.8)
    assert result.trajectory == original
    assert all(row.source_particle != 0 for row in result.steps)


def test_history_marks_unavailable_and_skips_landmark_constraints(floor):
    gray = np.ones((40, 40))
    gray[:, 15] = 0
    plt.imsave(floor.path, gray, cmap="gray", vmin=0, vmax=1)
    original = [[0.0, 0.0], [10.0, 0.0]]
    stages = [_stage(1, [[0, 0]], [[10, 0]])]
    result = build_history_candidate(original, stages, floor, 0, 9.8)
    assert result.status == "unavailable_map_path"
    assert result.trajectory == original and result.objective_candidate is None
    result = build_history_candidate(
        original, stages, floor, 0, 9.8, landmark_constraints=True
    )
    assert result.status == "skipped_landmark" and result.trajectory == original


def test_history_records_relocation_separately_from_cloning_and_sparse_steps(floor):
    stage = _stage(5, [[0, 0], [0, 0]], [[4, 0], [1, 0]])
    stage = replace(
        stage,
        proposed_step_lengths=np.array([0.7, 1.0]),
        parent_indices=np.array([0, 0]),
    )
    result = build_history_candidate([[0.0, 0.0], [4.0, 0.0]], [stage], floor, 0, 9.8)
    assert result.steps[-1].step == 5
    assert len(result.relocations) == 1
    event = result.relocations[0]
    assert event.particle == 0 and event.descendants == 2
    assert event.proposed_x_m == pytest.approx(0.7)
    assert event.replacement_m == pytest.approx(3.3)
    assert not build_history_candidate(
        [[0.0, 0.0], [4.0, 0.0]],
        [replace(stage, recovery_mode="checkpoint_replayed")],
        floor,
        0,
        9.8,
    ).relocations


def test_history_rejects_invalid_history(floor):
    stage = _stage(1, [[0, 0]], [[1, 0]])
    with pytest.raises(ValueError, match="歩数"):
        build_history_candidate([[0.0, 0.0]], [stage], floor, 0, 9.8)
    with pytest.raises(ValueError, match="有限"):
        build_history_candidate(
            [[0.0, 0.0], [1.0, 0.0]],
            [replace(stage, after_weights=np.array([np.nan]))],
            floor,
            0,
            9.8,
        )


def test_history_exports_without_opening_window(floor, tmp_path, monkeypatch):
    result = build_history_candidate(
        [[0.0, 0.0], [1.0, 0.0]], [_stage(1, [[0, 0]], [[1, 0]])], floor, 0, 9.8
    )

    def forbidden_show(*args, **kwargs):
        raise AssertionError("表示ウィンドウを開かない")

    monkeypatch.setattr(plt, "show", forbidden_show)
    save_history_csv(result, tmp_path)
    save_history_comparison(result, floor, 0, 9.8, tmp_path / "compare.png")
    assert len(pd.read_csv(tmp_path / "particle_history_candidate.csv")) == 2
    assert "particle" in pd.read_csv(tmp_path / "particle_relocations.csv").columns
    assert (tmp_path / "compare.png").is_file()
    assert not plt.get_fignums()


def test_history_flag_is_particle_only():
    assert (
        "--history-candidate" in CliRunner().invoke(cli, ["particle", "--help"]).output
    )
    with pytest.raises(ValueError, match="use_particle_filter"):
        commands.run(plot=False, history_candidate=True)


def test_history_pipeline_preserves_forward_results(floor):
    headings = []
    for step in range(1, 4):
        values = {
            name: None
            for name in StepHeading._fields
            if name not in StepHeading._field_defaults
        }
        values.update(
            step_index=step,
            timestamp_s=float(step),
            gyro_heading=0.0,
            selected_heading=0.0,
            body_heading=0.0,
            motion_heading=0.0,
            source="test",
            confidence=1.0,
            movement_type="forward",
            forward_displacement=1.0,
            lateral_displacement=0.0,
            motion_confidence=1.0,
        )
        headings.append(StepHeading(**values))
    posteriors = tuple(
        StepMotionPosterior(
            i, 1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 1.0, 0.1, 0.0, 0.1, "forward", "test"
        )
        for i in range(1, 4)
    )
    prepared = PreparedPdrSteps(
        df_acc=pd.DataFrame(),
        df_gyro=pd.DataFrame(),
        step_detection=StepDetectionResult("test", np.arange(3), (), None, None),
        trajectory=[[0, 0]] * 4,
        step_lengths=[1.0] * 3,
        t_at_steps=[1.0, 2.0, 3.0],
        step_headings=headings,
        gx_mean=0.0,
        gz_mean=9.8,
        weinberg_k=0.4,
        heading_method="test",
        motion_heading_correction="test",
        sidestep_smoothing="test",
        forward_heading_source="test",
        sidestep_heading_source="test",
        sidestep_suspect_mode="test",
        motion_evidences=build_step_motion_evidences(headings),
        particle_motion_headings=build_particle_motion_headings(headings),
        motion_posteriors=posteriors,
    )
    settings = ParticleSettings(
        floormap_path=floor.path, origin_px=floor.origin_px, scale=1.0, seed=7, count=12
    )
    baseline = run_particle(prepared, floor, settings)
    candidate = run_particle(prepared, floor, settings, history_candidate=True)
    np.testing.assert_array_equal(baseline.trajectory, candidate.trajectory)
    np.testing.assert_array_equal(
        baseline.particle.all_particles, candidate.particle.all_particles
    )
    assert baseline.particle.diagnostics == candidate.particle.diagnostics
    assert baseline.particle.history_candidate is None
    assert candidate.particle.history_candidate is not None
