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
