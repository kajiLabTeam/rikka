"""フロアマップ座標変換の共有部品。

役割:
    メートル座標とフロアマップの画素座標を相互変換し、方位ベクトルを
    画素座標の差分へ変換する。
依存元:
    NumPy の配列・三角関数だけを利用する。
利用先:
    particle の地図拘束、BLE ランドマーク補正、plot の軌跡・診断描画から
    使用される。
処理フロー:
    端末姿勢から画素 Y 軸の符号を決め、起点と縮尺を適用する。
"""

import numpy as np


def pixel_y_sign(gx_mean: float, gz_mean: float) -> int:
    """端末姿勢に応じたフロアマップ画素Y軸の符号を返す。"""
    if abs(gx_mean) > abs(gz_mean):
        return -1 if gx_mean > 0 else 1
    return -1 if gz_mean < 0 else 1


def compute_pixel_coords(
    xs: np.ndarray,
    ys: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """メートル座標をフロアマップのピクセル座標に変換する。"""
    y_sign = pixel_y_sign(gx_mean, gz_mean)
    return origin_px[0] + xs / scale, origin_px[1] + y_sign * ys / scale


def compute_meter_coords(
    pixel_xs: np.ndarray,
    pixel_ys: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """フロアマップのピクセル座標を歩行開始点基準のメートル座標に変換する。"""
    y_sign = pixel_y_sign(gx_mean, gz_mean)
    return (
        (pixel_xs - origin_px[0]) * scale,
        (pixel_ys - origin_px[1]) * scale * y_sign,
    )


def is_walkable_cell(map_gray: np.ndarray, x: int, y: int) -> bool:
    """指定画素がマップ内の歩行可能画素かを返す。"""
    map_h, map_w = map_gray.shape
    return 0 <= x < map_w and 0 <= y < map_h and bool(map_gray[y, x] > 128)


def normalize_floormap_gray(map_raw: np.ndarray) -> np.ndarray:
    """フロアマップ画像を 0..255 のグレースケール配列に正規化する。"""
    map_arr: np.ndarray = np.asarray(map_raw, dtype=float)
    if map_arr.ndim == 3:
        map_arr = np.mean(map_arr[:, :, :3], axis=2)
    if map_arr.size == 0:
        return map_arr
    if float(np.nanmax(map_arr)) <= 1.0:
        map_arr = map_arr * 255.0
    return np.asarray(np.clip(map_arr, 0.0, 255.0), dtype=float)


def validate_floormap_origin(
    map_gray: np.ndarray,
    origin_px: tuple[int, int],
) -> None:
    """起点が検証済み2次元マップ内の歩行可能画素であることを検証する。"""
    origin_x, origin_y = origin_px
    if not is_walkable_cell(map_gray, origin_x, origin_y):
        raise ValueError("origin_px は歩行可能なマップ内画素を指定してください")


def validate_floormap_shape(map_gray: np.ndarray) -> None:
    """フロアマップが空でない2次元配列であることを検証する。"""
    if map_gray.ndim != 2 or map_gray.size == 0:
        raise ValueError("フロアマップは空でない2次元画像を指定してください")


def segment_crosses_only_walkable_cells(
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
    if not is_walkable_cell(map_gray, cell_x, cell_y):
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
            if not is_walkable_cell(map_gray, next_x, cell_y):
                return False
            if not is_walkable_cell(map_gray, cell_x, next_y):
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
        if not is_walkable_cell(map_gray, cell_x, cell_y):
            return False
    return False


def pixel_vector_from_heading(
    heading: float,
    length_m: float,
    gx_mean: float,
    gz_mean: float,
    scale: float,
) -> tuple[float, float]:
    """メートル座標の方位ベクトルをピクセル座標の差分に変換する。"""
    y_sign = pixel_y_sign(gx_mean, gz_mean)
    return (
        length_m * float(np.cos(heading)) / scale,
        y_sign * length_m * float(np.sin(heading)) / scale,
    )
