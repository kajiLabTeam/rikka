"""ランドマーク相似変換の純粋関数を検証する。"""

import numpy as np
import pytest

from rikka.landmark.lib.retrofit import (
    SimilarityTransform,
    apply_transform,
    damp_transform,
    evaluate_transform_walkability,
    solve_anchor_similarity,
)


def test_solve_anchor_similarity_maps_endpoint_and_keeps_pivot() -> None:
    """解いた変換が固定点を保ち、補正前終点を目標へ写す。"""
    transform = solve_anchor_similarity((1.0, 1.0), (3.0, 1.0), (1.0, 5.0))

    assert transform is not None
    transformed = apply_transform(
        [[1.0, 1.0], [3.0, 1.0]], transform, start_index=0, end_index=1
    )
    assert transformed[0] == pytest.approx([1.0, 1.0])
    assert transformed[1] == pytest.approx([1.0, 5.0])
    assert transform.rotation_rad == pytest.approx(np.pi / 2.0)
    assert transform.scale == pytest.approx(2.0)


def test_damp_transform_uses_logarithmic_scale_interpolation() -> None:
    """半減衰で回転半分、倍率平方根になる。"""
    damped = damp_transform(SimilarityTransform((0.0, 0.0), np.pi, 4.0), 0.5)

    assert damped.rotation_rad == pytest.approx(np.pi / 2.0)
    assert damped.scale == pytest.approx(2.0)


def test_solve_anchor_similarity_rejects_degenerate_vector() -> None:
    """固定点と終点が一致する縮退区間は解かない。"""
    assert solve_anchor_similarity((1.0, 1.0), (1.0, 1.0), (2.0, 1.0)) is None


def test_walkability_evaluation_counts_transformed_wall_crossing() -> None:
    """変換後に壁画素を通る辺を違反として数える。"""
    map_gray = np.full((8, 8), 255.0)
    map_gray[:, 3] = 0.0
    transform = SimilarityTransform((0.0, 0.0), 0.0, 2.0)

    violations = evaluate_transform_walkability(
        [[0.0, 0.0], [2.0, 0.0]],
        transform,
        1,
        1,
        map_gray=map_gray,
        gx_mean=0.0,
        gz_mean=1.0,
        origin_px=(0, 0),
        scale=1.0,
    )

    assert violations == 1
