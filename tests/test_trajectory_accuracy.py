"""PFの緩やかな反転をセンサー上の旋回と区別する回帰テスト。"""

import numpy as np
import pytest

from rikka.particle.lib.sequence_path import _unsupported_reversal_count


@pytest.mark.parametrize("sensor_turns", [False, True])
def test_gradual_reversal_requires_sensor_evidence(sensor_turns: bool) -> None:
    headings = np.deg2rad(np.arange(0.0, 181.0, 30.0))
    path = np.vstack(
        [np.zeros((1, 2)), np.cumsum(np.c_[np.cos(headings), np.sin(headings)], axis=0)]
    )
    count = _unsupported_reversal_count(
        path,
        headings if sensor_turns else np.zeros(len(headings)),
        np.zeros(len(headings), dtype=bool),
    )
    assert count == (0 if sensor_turns else 1)


def test_gradual_reversal_keeps_supported_turn() -> None:
    headings = np.deg2rad(np.arange(0.0, 181.0, 45.0))
    path = np.vstack(
        [np.zeros((1, 2)), np.cumsum(np.c_[np.cos(headings), np.sin(headings)], axis=0)]
    )
    assert (
        _unsupported_reversal_count(
            path, np.zeros(len(headings)), np.ones(len(headings), dtype=bool)
        )
        == 0
    )


def test_gradual_reversal_handles_angle_wrap_without_double_counting() -> None:
    headings = np.deg2rad([170, 200, 230, 260, 290, 320, 350, 350, 350])
    path = np.vstack(
        [np.zeros((1, 2)), np.cumsum(np.c_[np.cos(headings), np.sin(headings)], axis=0)]
    )
    assert (
        _unsupported_reversal_count(
            path,
            np.full(len(headings), np.deg2rad(170)),
            np.zeros(len(headings), dtype=bool),
        )
        == 1
    )
