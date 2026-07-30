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

from ...common.lib.floormap import compute_pixel_coords


def _normalize_floormap_gray(map_raw: np.ndarray) -> np.ndarray:
    """フロアマップ画像を 0..255 のグレースケール配列に正規化する。"""
    map_arr: np.ndarray = np.asarray(map_raw, dtype=float)
    if map_arr.ndim == 3:
        map_arr = np.mean(map_arr[:, :, :3], axis=2)
    if map_arr.size == 0:
        return map_arr
    if float(np.nanmax(map_arr)) <= 1.0:
        map_arr = map_arr * 255.0
    return np.asarray(np.clip(map_arr, 0.0, 255.0), dtype=float)


def _pixel_y_sign(gx_mean: float, gz_mean: float) -> int:
    """メートル座標とピクセル座標のY軸向きを返す。"""
    if abs(gx_mean) > abs(gz_mean):
        return -1 if gx_mean > 0 else 1
    return -1 if gz_mean < 0 else 1


def _compute_meter_coords(
    px: np.ndarray,
    py: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """フロアマップのピクセル座標をメートル座標へ戻す。"""
    y_sign = _pixel_y_sign(gx_mean, gz_mean)
    xs = (px - origin_px[0]) * scale
    ys = (py - origin_px[1]) * scale / y_sign
    return xs, ys


def _is_walkable_cell(map_gray: np.ndarray, x: int, y: int) -> bool:
    """指定画素がマップ内の歩行可能画素かを返す。"""
    map_h, map_w = map_gray.shape
    return 0 <= x < map_w and 0 <= y < map_h and bool(map_gray[y, x] > 128)


def _segment_crosses_only_walkable_cells(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    map_gray: np.ndarray,
) -> bool:
    """線分が触れる全画素を保守的に調べ、歩行可能かを返す。"""
    if not np.isfinite([x0, y0, x1, y1]).all():
        return False

    cell_x = int(np.floor(x0 + 0.5))
    cell_y = int(np.floor(y0 + 0.5))
    end_x = int(np.floor(x1 + 0.5))
    end_y = int(np.floor(y1 + 0.5))
    if not _is_walkable_cell(map_gray, cell_x, cell_y):
        return False
    if cell_x == end_x and cell_y == end_y:
        return True

    dx = x1 - x0
    dy = y1 - y0
    step_x = 1 if dx > 0 else -1 if dx < 0 else 0
    step_y = 1 if dy > 0 else -1 if dy < 0 else 0
    t_delta_x = np.inf if step_x == 0 else 1.0 / abs(dx)
    t_delta_y = np.inf if step_y == 0 else 1.0 / abs(dy)
    next_boundary_x = cell_x + 0.5 if step_x > 0 else cell_x - 0.5
    next_boundary_y = cell_y + 0.5 if step_y > 0 else cell_y - 0.5
    t_max_x = np.inf if step_x == 0 else (next_boundary_x - x0) / dx
    t_max_y = np.inf if step_y == 0 else (next_boundary_y - y0) / dy

    max_cells = abs(end_x - cell_x) + abs(end_y - cell_y) + 2
    for _ in range(max_cells):
        if cell_x == end_x and cell_y == end_y:
            return True
        if abs(t_max_x - t_max_y) <= 1e-12:
            next_x = cell_x + step_x
            next_y = cell_y + step_y
            # 画素角に触れる遷移は、隣接する両画素も通過したものとして扱う。
            if not _is_walkable_cell(map_gray, next_x, cell_y):
                return False
            if not _is_walkable_cell(map_gray, cell_x, next_y):
                return False
            cell_x = next_x
            cell_y = next_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            cell_x += step_x
            t_max_x += t_delta_x
        else:
            cell_y += step_y
            t_max_y += t_delta_y
        if not _is_walkable_cell(map_gray, cell_x, cell_y):
            return False
    return False


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

    xs, ys = _compute_meter_coords(
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
