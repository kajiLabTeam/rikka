"""PFの単一祖先系列による代表軌跡選択を検証する。"""

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from rikka.common.lib.models import StepHeading
from rikka.particle.lib.map_constraints import _evaluate_particle_transitions
from rikka.particle.lib.runner import run_particle_filter as _run_particle_filter
from rikka.particle.lib.sequence_path import _select_sequence_map_path
from rikka.pdr.lib.motion_state.evidence import (
    build_particle_motion_headings,
    build_step_motion_evidences,
)


def run_particle_filter(*args, **kwargs):
    """確定済みPDR歩列をPF内部回帰テストへ渡す。"""
    headings = kwargs.get("prepared_step_headings")
    if headings is not None:
        kwargs.setdefault(
            "prepared_motion_evidences",
            build_step_motion_evidences(headings),
        )
        kwargs.setdefault(
            "prepared_particle_motion_headings",
            build_particle_motion_headings(headings),
        )
    return _run_particle_filter(*args, **kwargs)


def _select(
    paths: np.ndarray,
    scores: np.ndarray,
    map_gray: np.ndarray,
    headings: np.ndarray | None = None,
    turning: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], list[int | None]]:
    return _select_sequence_map_path(
        paths,
        scores,
        map_gray,
        gx_mean=0.0,
        gz_mean=1.0,
        origin_px=(0, 0),
        scale=1.0,
        sensor_headings=headings,
        turning_evidence=turning,
    )


def _forward_heading(step_index: int) -> StepHeading:
    """固定seedのrunner検証に使う前進ステップを返す。"""
    return StepHeading(
        step_index=step_index,
        timestamp_s=float(step_index - 1),
        gyro_heading=0.0,
        accel_method1_heading=None,
        accel_method2_heading=None,
        selected_heading=0.0,
        source="trajectory_motion",
        confidence=1.0,
        angle_diff_method1=None,
        angle_diff_method2=None,
        segment_start_index=None,
        segment_end_index=None,
        peak1_index=None,
        peak2_index=None,
        body_heading=0.0,
        motion_heading=0.0,
        movement_type="forward",
        forward_displacement=1.0,
        lateral_displacement=0.0,
        motion_confidence=1.0,
        motion_reject_reason=None,
        trajectory_movement_type="forward",
    )


def test_sequence_selection_is_fixed_and_uses_maximum_score() -> None:
    rng = np.random.default_rng(42)
    paths = np.stack(
        [
            np.column_stack((np.arange(5), np.full(5, offset)))
            for offset in rng.permutation([1.0, 2.0, 3.0])
        ]
    )
    scores = np.array([-2.0, -0.1, -1.0])

    first, modes, sources = _select(paths, scores, np.full((8, 8), 255.0))
    second, _, _ = _select(paths, scores, np.full((8, 8), 255.0))

    np.testing.assert_array_equal(first, paths[1])
    np.testing.assert_array_equal(first, second)
    assert set(modes) == {"sequence_map_ancestry"}
    assert set(sources) == {1}


def test_sequence_selection_avoids_mean_path_through_wall() -> None:
    map_gray = np.full((7, 7), 255.0)
    map_gray[3, 3] = 0.0
    upper = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 2.0], [5.0, 3.0]])
    lower = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0], [4.0, 4.0], [5.0, 3.0]])
    paths = np.stack([upper, lower])

    selected, _, _ = _select(paths, np.array([-0.1, -0.2]), map_gray)

    np.testing.assert_array_equal(selected, upper)
    assert _evaluate_particle_transitions(
        selected[:-1], selected[1:], map_gray, 0.0, 1.0, (0, 0), 1.0
    ).all()
    mean_path = np.mean(paths, axis=0)
    assert not _evaluate_particle_transitions(
        mean_path[:-1], mean_path[1:], map_gray, 0.0, 1.0, (0, 0), 1.0
    ).all()


def test_sequence_selection_penalizes_unsupported_reversal() -> None:
    reverse = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [2.0, 1.0]])
    straight = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 2.0]])

    selected, _, sources = _select(
        np.stack([reverse, straight]),
        np.array([0.0, -1.0]),
        np.full((7, 7), 255.0),
        headings=np.zeros(3),
        turning=np.zeros(3, dtype=bool),
    )

    np.testing.assert_array_equal(selected, straight)
    assert sources[-1] == 1


def test_sequence_selection_allows_supported_u_turn() -> None:
    u_turn = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [2.0, 1.0]])
    straight = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 2.0]])

    selected, _, sources = _select(
        np.stack([u_turn, straight]),
        np.array([0.0, -1.0]),
        np.full((7, 7), 255.0),
        headings=np.array([0.0, 0.0, np.pi]),
        turning=np.array([False, False, True]),
    )

    np.testing.assert_array_equal(selected, u_turn)
    assert sources[-1] == 0


def test_particle_filter_sequence_mode_is_reproducible(tmp_path) -> None:
    floormap_path = tmp_path / "open_map.png"
    mpimg.imsave(floormap_path, np.ones((20, 20)), cmap="gray", vmin=0.0, vmax=1.0)
    kwargs = {
        "peaks": np.arange(3),
        "df_gyro": pd.DataFrame({"low_angle": np.zeros(3)}),
        "df_acc": pd.DataFrame({"h_y": np.zeros(3), "h_z": np.zeros(3)}),
        "gx_mean": 0.0,
        "gz_mean": 9.8,
        "floormap_path": floormap_path,
        "origin_px": (5, 5),
        "scale": 1.0,
        "n_particles": 20,
        "prepared_step_headings": [_forward_heading(i + 1) for i in range(3)],
        "prepared_step_lengths": [1.0] * 3,
        "prepared_step_times": [0.0, 1.0, 2.0],
        "seed": 42,
        "path_selection": "sequence",
    }

    first = run_particle_filter(**kwargs)[0]
    second = run_particle_filter(**kwargs)[0]

    np.testing.assert_array_equal(first, second)
