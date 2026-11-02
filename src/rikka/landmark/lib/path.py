"""ランドマークのみで構成する経路の生成。

役割:
    既知ピクセル座標のランドマークを指定順に直線で結ぶ経路を作り、区間距離と
    累積距離を付与する。PDR の推測航法を使わない参照軌跡として利用する。
依存元:
    ``common.lib.models`` の ``Landmark`` / ``LandmarkPath`` 共有型と、
    ``common.config`` の地図縮尺既定値、NumPy の距離計算を利用する。
利用先:
    ``cli.commands.run_landmark_path()`` が経路を作り、``plot`` 領域が
    フロアマップ上への描画と CSV 出力に使用する。
処理フロー:
    入力順を経路順として検証し、ピクセル距離へ縮尺を掛けた区間距離を求め、
    先頭ランドマークを原点とするメートル座標とともに不変型へ格納する。
"""

from collections.abc import Sequence

import numpy as np

from ...common.config import FLOORMAP_SCALE
from ...common.lib.models import Landmark, LandmarkPath, LandmarkPathPoint

# 画像座標は下方向が正のため、メートル座標では上方向を +Y として符号を反転する。
_PIXEL_Y_SIGN = -1.0


def build_landmark_path(
    landmarks: Sequence[Landmark],
    scale: float = FLOORMAP_SCALE,
) -> LandmarkPath:
    """ランドマークを与えられた順に結ぶ経路を作る。"""
    if len(landmarks) < 2:
        raise ValueError("ランドマーク経路には2点以上のランドマークが必要です。")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale は有限な正の値を指定してください。")
    for landmark in landmarks:
        if not np.isfinite([landmark.pixel_x, landmark.pixel_y]).all():
            raise ValueError(
                f"ランドマーク座標は有限値を指定してください: {landmark.beacon_id}"
            )

    origin_x = float(landmarks[0].pixel_x)
    origin_y = float(landmarks[0].pixel_y)
    points: list[LandmarkPathPoint] = []
    cumulative = 0.0
    for order, landmark in enumerate(landmarks):
        pixel_x = float(landmark.pixel_x)
        pixel_y = float(landmark.pixel_y)
        if order == 0:
            segment_length = 0.0
        else:
            previous = landmarks[order - 1]
            segment_length = scale * float(
                np.hypot(
                    pixel_x - float(previous.pixel_x),
                    pixel_y - float(previous.pixel_y),
                )
            )
        cumulative += segment_length
        points.append(
            LandmarkPathPoint(
                order=order,
                beacon_id=landmark.beacon_id,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                meter_x=(pixel_x - origin_x) * scale,
                meter_y=(pixel_y - origin_y) * scale * _PIXEL_Y_SIGN,
                segment_length_m=segment_length,
                cumulative_length_m=cumulative,
            )
        )
    return LandmarkPath(
        points=tuple(points),
        scale=float(scale),
        total_length_m=cumulative,
    )
