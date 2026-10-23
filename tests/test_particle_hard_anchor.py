"""確定位置による区間再開始と入力拒否を検証する。"""

import json
from dataclasses import replace
from types import SimpleNamespace

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest

from rikka.common.lib.models import HardPositionObservation, StepHeading
from rikka.landmark.lib.hard_anchor import load_hard_positions
from rikka.particle.lib.runner import run_particle_steps
from rikka.pdr.lib.motion_state.evidence import (
    build_particle_motion_headings,
    build_step_motion_evidences,
)
from rikka.plot.lib.animation import plot_particle_filter_trajectory
from rikka.plot.pipeline import _write_hard_anchor_outputs


@pytest.fixture
def arguments(tmp_path):
    path = tmp_path / "map.png"
    mpimg.imsave(path, np.ones((81, 81)), cmap="gray", vmin=0, vmax=1)
    headings = [
        StepHeading(
            i,
            float(i),
            0.0,
            None,
            None,
            0.0,
            "trajectory_motion",
            1.0,
            None,
            None,
            None,
            None,
            None,
            None,
            0.0,
            0.0,
            "forward",
            1.0,
            0.0,
            1.0,
            None,
            "forward",
        )
        for i in range(4)
    ]
    return dict(
        gx_mean=0.0,
        gz_mean=9.8,
        floormap_path=path,
        origin_px=(40, 40),
        scale=1.0,
        n_particles=24,
        prepared_step_headings=headings,
        prepared_step_lengths=[1.0] * 4,
        prepared_step_times=[0.0, 1.0, 2.0, 3.0],
        prepared_motion_evidences=build_step_motion_evidences(headings),
        prepared_particle_motion_headings=build_particle_motion_headings(headings),
        proposal_method="candidate-bank",
        candidate_count=8,
        landmark_mode="none",
        seed=42,
    )


def observation(**kwargs):
    return replace(
        HardPositionObservation("known", 1.0, 60.0, 50.0, "manual", "到達記録"),
        **kwargs,
    )


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_exact_anchor_preserves_previous_nodes_and_states(arguments, seed):
    baseline_stages, anchored_stages, paths, events = [], [], [], []
    args = {**arguments, "seed": seed}
    baseline = run_particle_steps(**args, stage_collector=baseline_stages)
    anchored = run_particle_steps(
        **args,
        stage_collector=anchored_stages,
        path_comparison_collector=paths,
        hard_positions=(observation(),),
        hard_anchor_events_collector=events,
    )
    np.testing.assert_array_equal(baseline[3][:2], anchored[3][:2])
    np.testing.assert_array_equal(anchored[3][2], np.tile([20.0, 10.0], (24, 1)))
    np.testing.assert_array_equal(
        baseline_stages[1].after_offsets, anchored_stages[1].after_offsets
    )
    np.testing.assert_array_equal(
        baseline_stages[1].after_stride_scale, anchored_stages[1].after_stride_scale
    )
    np.testing.assert_array_equal(paths[0].current_path[2], [20.0, 10.0])
    np.testing.assert_array_equal(paths[0].sequence_path[2], [20.0, 10.0])
    np.testing.assert_array_equal(anchored[0][2], [20.0, 10.0])
    assert len(events) == 1 and events[0].step_index == 2
    assert np.linalg.norm(anchored[3][3] - anchored[3][2], axis=1).max() < 2.0


@pytest.mark.parametrize("time", [0.5, -1.0, 9.0])
def test_ambiguous_or_outside_time_rejected(arguments, time):
    with pytest.raises(ValueError, match="歩境界"):
        run_particle_steps(**arguments, hard_positions=(observation(timestamp_s=time),))


def test_duplicates_and_outside_map_rejected(arguments):
    with pytest.raises(ValueError, match="重複"):
        run_particle_steps(**arguments, hard_positions=(observation(), observation()))
    with pytest.raises(ValueError, match="歩行可能"):
        run_particle_steps(**arguments, hard_positions=(observation(pixel_x=1000.0),))


def test_multiple_anchors_and_last_step(arguments):
    result = run_particle_steps(
        **arguments,
        hard_positions=(
            observation(),
            observation(event_id="second", timestamp_s=3.0, pixel_x=65.0),
        ),
    )
    np.testing.assert_array_equal(result[0][2], [20.0, 10.0])
    np.testing.assert_array_equal(result[0][4], [25.0, 10.0])
    assert len(result[0]) == 5


def test_json_requires_evidence(tmp_path):
    path = tmp_path / "anchors.json"
    path.write_text(
        json.dumps(
            [dict(event_id="x", timestamp_s=0, pixel_x=1, pixel_y=1, source="manual")]
        )
    )
    with pytest.raises(ValueError, match="形式"):
        load_hard_positions(path)


def test_output_separates_position_correction_from_walking(arguments, tmp_path):
    events = []
    result = run_particle_steps(
        **arguments,
        hard_positions=(observation(),),
        hard_anchor_events_collector=events,
    )
    raw = [point.copy() for point in result[0]]
    event = events[0]
    raw[event.step_index] = [event.before_x, event.before_y]
    value = SimpleNamespace(
        trajectory=result[0],
        t_at_steps=result[2],
        particle=SimpleNamespace(
            hard_anchor_corrections=tuple(events),
            uncorrected_trajectory=raw,
            correction_steps=(event.step_index,),
        ),
    )
    _write_hard_anchor_outputs(value, tmp_path)
    segments = pd.read_csv(tmp_path / "trajectory_segments.csv")
    assert segments.walking_distance_m.max() < 2.0
    assert segments.position_correction_m.sum() > 15.0
    assert segments.is_position_correction.sum() == 1
    assert segments.walking_distance_m.sum() < 6.0
    corrections = pd.read_csv(tmp_path / "hard_anchor_corrections.csv")
    assert corrections.loc[0, "evidence"] == "到達記録"
    assert corrections.loc[0, "target_x"] == 20.0


def test_plot_marks_correction_as_separate_dashed_line(arguments, tmp_path):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    events = []
    result = run_particle_steps(
        **arguments,
        hard_positions=(observation(),),
        hard_anchor_events_collector=events,
    )
    plot_particle_filter_trajectory(
        result[0],
        gx_mean=0.0,
        gz_mean=9.8,
        floormap_path=arguments["floormap_path"],
        origin_px=arguments["origin_px"],
        scale=1.0,
        hard_corrections=tuple(events),
        output_dir=tmp_path,
    )
    ax = plt.gcf().axes[0]
    lines = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(lines) == 1
    assert "歩行ではない" in lines[0].get_label()
    walk = next(item for item in ax.collections if isinstance(item, LineCollection))
    lengths = [
        np.linalg.norm(segment[1] - segment[0]) for segment in walk.get_segments()
    ]
    assert max(lengths) < 2.0
    assert (tmp_path / "pf_trajectory.png").is_file()
    plt.close("all")


def test_anchor_preserves_completed_prefix(arguments):
    short_args = dict(arguments)
    for name in (
        "prepared_step_headings",
        "prepared_step_lengths",
        "prepared_step_times",
        "prepared_motion_evidences",
        "prepared_particle_motion_headings",
    ):
        short_args[name] = short_args[name][:2]
    short = run_particle_steps(**short_args, hard_positions=(observation(),))
    long = run_particle_steps(**arguments, hard_positions=(observation(),))
    np.testing.assert_array_equal(short[0], long[0][:3])


def test_floor_must_be_explicit_and_matching(arguments):
    anchor = observation(floor_id="floor5")
    for floor_id in (None, "floor6"):
        with pytest.raises(ValueError, match="floor_id"):
            run_particle_steps(
                **arguments, hard_positions=(anchor,), floormap_floor_id=floor_id
            )
    run_particle_steps(
        **arguments, hard_positions=(anchor,), floormap_floor_id="floor5"
    )


def test_wall_target_is_rejected_without_snapping(arguments):
    pixels = np.ones((81, 81))
    pixels[50, 60] = 0
    mpimg.imsave(arguments["floormap_path"], pixels, cmap="gray", vmin=0, vmax=1)
    with pytest.raises(ValueError, match="歩行可能"):
        run_particle_steps(**arguments, hard_positions=(observation(),))


def test_heading_is_only_fixed_when_explicit(arguments):
    stages = []
    run_particle_steps(
        **arguments,
        hard_positions=(observation(heading_deg=90.0),),
        stage_collector=stages,
    )
    np.testing.assert_allclose(stages[1].after_offsets, np.pi / 2, rtol=0, atol=1e-14)


def test_hard_restart_is_not_reported_as_a_normal_candidate_edge(arguments):
    stages = []
    run_particle_steps(
        **arguments,
        hard_positions=(observation(),),
        stage_collector=stages,
    )
    assert not stages[1].proposal_matches_final_history
    assert stages[1].hard_anchor_event_id == "known"
    assert stages[1].segment_id == 1
    assert stages[0].hard_anchor_event_id is None
    assert stages[0].segment_id == 0
    assert stages[2].hard_anchor_event_id is None
    assert stages[2].segment_id == 1


def test_correction_before_is_selected_arrival(arguments):
    for mode in ("current", "sequence"):
        before_paths, events = [], []
        short_args = dict(arguments)
        for name in (
            "prepared_step_headings",
            "prepared_step_lengths",
            "prepared_step_times",
            "prepared_motion_evidences",
            "prepared_particle_motion_headings",
        ):
            short_args[name] = short_args[name][:2]
        run_particle_steps(
            **short_args, path_selection=mode, path_comparison_collector=before_paths
        )
        run_particle_steps(
            **arguments,
            path_selection=mode,
            hard_positions=(observation(),),
            hard_anchor_events_collector=events,
        )
        expected = getattr(before_paths[0], f"{mode}_path")[-1]
        np.testing.assert_array_equal(
            [events[0].before_x, events[0].before_y], expected
        )
