"""ランドマーク座標の共通変換。

役割:
    フロアマップ上のランドマークピクセル座標を、PDR と particle filter が
    使用できる歩行開始点基準のメートル座標へ変換する。
依存元:
    ``common.lib.models.Landmark`` / ``FloorMap`` から既知ピクセル座標と地図設定、
    ``common.lib.floormap.compute_meter_coords`` から地図座標変換を取得する。
利用先:
    ``pdr.lib.landmark_correction`` が完全座標補正に使用し、将来は
    particle filter のランドマーク観測尤度からも利用する。
処理フロー:
    ランドマーク列から画素 X/Y を配列化し、起点・縮尺・端末向きを適用して
    beacon_id からメートル座標への辞書を返す。
"""

import numpy as np

from ...common.lib.floormap import compute_meter_coords
from ...common.lib.models import FloorMap, Landmark


def build_landmark_meter_map(
    landmarks: tuple[Landmark, ...],
    gx_mean: float,
    gz_mean: float,
    floormap: FloorMap,
) -> dict[str, tuple[float, float]]:
    """beacon_id から歩行開始点基準のメートル座標への辞書を返す。"""
    meter_xs, meter_ys = compute_meter_coords(
        np.asarray([item.pixel_x for item in landmarks], dtype=float),
        np.asarray([item.pixel_y for item in landmarks], dtype=float),
        gx_mean,
        gz_mean,
        floormap.origin_px,
        floormap.scale,
    )
    return {
        item.beacon_id: (float(x), float(y))
        for item, x, y in zip(landmarks, meter_xs, meter_ys, strict=True)
    }
