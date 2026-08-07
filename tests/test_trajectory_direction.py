"""終端方向の逆走検出と複数計測コンセンサスを検証する。"""

import numpy as np
import pytest

from rikka.common.lib.trajectory_direction import (
    evaluate_terminal_direction,
    sample_trajectory_by_arclength,
    terminal_consensus_outliers,
)


def _corner(final_y: float) -> np.ndarray:
    """東進後に南北どちらかへ曲がる簡易軌跡を作る。"""
    return np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [10.0, final_y]])


def test_terminal_direction_detects_smooth_wrong_turn() -> None:
    """局所的な180度ジャンプがなくても逆側への終端旋回を検出する。"""
    truth = sample_trajectory_by_arclength(_corner(-10.0))
    wrong = sample_trajectory_by_arclength(_corner(10.0))

    metrics = evaluate_terminal_direction(wrong, truth)

    assert metrics.direction_error_deg == pytest.approx(180.0)
    assert metrics.progress_cosine == pytest.approx(-1.0)
    assert metrics.opposed_fraction == pytest.approx(1.0)
    assert metrics.failure


def test_terminal_direction_accepts_same_direction_with_angle_wrap() -> None:
    """±180度境界をまたぐ同方向を逆走と誤判定しない。"""
    reference = sample_trajectory_by_arclength(np.asarray([[0.0, 0.0], [-10.0, 0.1]]))
    candidate = sample_trajectory_by_arclength(np.asarray([[0.0, 0.0], [-10.0, -0.1]]))

    metrics = evaluate_terminal_direction(candidate, reference)

    assert metrics.direction_error_deg < 2.0
    assert metrics.progress_cosine > 0.99
    assert not metrics.failure


def test_terminal_consensus_excludes_opposite_pf_branches() -> None:
    """計測と方式を等重みにし、多数計測と逆向きのPF候補を除外する。"""
    data_names: list[str] = []
    families: list[str] = []
    sampled: list[np.ndarray] = []
    for data_index in range(5):
        pdr_south = data_index != 1
        data_names.append(f"data{data_index}")
        families.append("pdr")
        sampled.append(
            sample_trajectory_by_arclength(_corner(-10.0 if pdr_south else -2.0))
        )
        for _seed in range(3):
            pf_south = data_index < 3
            data_names.append(f"data{data_index}")
            families.append("pf")
            sampled.append(
                sample_trajectory_by_arclength(_corner(-10.0 if pf_south else 10.0))
            )

    outliers, consensus, errors = terminal_consensus_outliers(
        np.stack(sampled), data_names, families
    )

    assert np.degrees(consensus) < 0.0
    assert all(errors[index] > 90.0 for index in outliers)
    assert len(outliers) == 6
