"""フロアマップ座標変換と粒子遷移の壁制約。

役割:
    マップ画像を正規化し、メートル・画素座標を対応付け、移動線分が触れる全画素の
    歩行可否を判定する。必要時には軌跡点を最近傍の歩行可能画素へ寄せる。
依存元:
    ``plot.lib`` の画素座標変換、NumPy、SciPy の距離変換を利用する。
利用先:
    粒子フィルタの伝播・recovery・代表軌跡選択処理が地図制約の適用に使用する。
処理フロー:
    座標を画素へ変換し、supercover方式で線分通過画素を検査して真偽配列を返す。
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

from ...common.lib.floormap import (
    compute_meter_coords,
    compute_pixel_coords,
)
from ...common.lib.floormap import (
    normalize_floormap_gray as _normalize_floormap_gray,
)
from ...common.lib.floormap import (
    segment_crosses_only_walkable_cells as _segment_crosses_only_walkable_cells,
)
from ...common.lib.floormap import (
    validate_floormap_origin as _validate_floormap_origin,
)
from ...common.lib.floormap import (
    validate_floormap_shape as _validate_floormap_shape,
)

__all__ = [
    "_normalize_floormap_gray",
    "_segment_crosses_only_walkable_cells",
    "_validate_floormap_origin",
    "_validate_floormap_shape",
]


def _evaluate_particle_transitions(
    previous_particles: np.ndarray,
    proposed_particles: np.ndarray,
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> np.ndarray:
    """各粒子の移動線分が全て歩行可能画素だけを通るか判定する。"""
    if previous_particles.shape != proposed_particles.shape:
        raise ValueError("遷移前後の粒子配列 shape が一致しません")
    previous_px, previous_py = compute_pixel_coords(
        previous_particles[:, 0],
        previous_particles[:, 1],
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    proposed_px, proposed_py = compute_pixel_coords(
        proposed_particles[:, 0],
        proposed_particles[:, 1],
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    return np.asarray(
        [
            _segment_crosses_only_walkable_cells(x0, y0, x1, y1, map_gray)
            for x0, y0, x1, y1 in zip(
                previous_px,
                previous_py,
                proposed_px,
                proposed_py,
                strict=True,
            )
        ],
        dtype=bool,
    )


def _snap_trajectory_to_walkable_pixels(
    trajectory: list[list[float]],
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> list[list[float]]:
    """壁上・範囲外の軌跡点を最近傍の歩行可能画素へ寄せる。"""
    if len(trajectory) == 0:
        return trajectory

    walkable = map_gray > 128
    if not walkable.any():
        return trajectory

    map_h, map_w = walkable.shape
    points = np.asarray(trajectory, dtype=float)
    px_f, py_f = compute_pixel_coords(
        points[:, 0], points[:, 1], gx_mean, gz_mean, origin_px, scale
    )
    pxi = np.round(px_f).astype(int)
    pyi = np.round(py_f).astype(int)

    in_bounds = (0 <= pxi) & (pxi < map_w) & (0 <= pyi) & (pyi < map_h)
    needs_snap = ~in_bounds.copy()
    if in_bounds.any():
        needs_snap[in_bounds] = ~walkable[pyi[in_bounds], pxi[in_bounds]]

    if not needs_snap.any():
        return trajectory

    _, nearest_indices = distance_transform_edt(~walkable, return_indices=True)
    nearest_y = nearest_indices[0]
    nearest_x = nearest_indices[1]

    query_x = pxi.clip(0, map_w - 1)
    query_y = pyi.clip(0, map_h - 1)
    snap_x = query_x.copy()
    snap_y = query_y.copy()
    snap_x[needs_snap] = nearest_x[query_y[needs_snap], query_x[needs_snap]]
    snap_y[needs_snap] = nearest_y[query_y[needs_snap], query_x[needs_snap]]

    xs, ys = compute_meter_coords(
        snap_x.astype(float),
        snap_y.astype(float),
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    snapped_points = points.copy()
    snapped_points[needs_snap, 0] = xs[needs_snap]
    snapped_points[needs_snap, 1] = ys[needs_snap]
    return [[float(x), float(y)] for x, y in snapped_points]
