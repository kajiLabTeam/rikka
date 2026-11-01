"""ランドマーク測位の実行入口。

役割:
    ランドマーク領域の処理を他領域から呼ぶための入口をまとめる。現在は既知
    ピクセル座標の列から、指定順に直線で結ぶ経路を組み立てる。
依存元:
    ``common.config`` の地図縮尺既定値、``common.lib.models`` の共有型、
    同領域 ``lib.path`` の経路生成を利用する。
利用先:
    ``cli.commands.run_landmark_path()`` が経路生成の入口として使用する。
処理フロー:
    ``(beacon_id, ピクセルX, ピクセルY)`` の列を共有型 ``Landmark`` へ変換し、
    通過時刻が与えられていればその順へ並べ替えて、
    ``lib.path.build_landmark_path`` で区間距離付きの経路を返す。
"""

from collections.abc import Mapping, Sequence

from ..common.config import FLOORMAP_SCALE
from ..common.lib.models import Landmark, LandmarkPath
from .lib.path import build_landmark_path


def prepare_landmark_path(
    landmarks: Sequence[tuple[str, float, float]],
    scale: float = FLOORMAP_SCALE,
    order_times: Mapping[str, float] | None = None,
) -> tuple[tuple[Landmark, ...], LandmarkPath]:
    """ピクセル座標の指定列から、検証に使う共有型と経路を作る。

    ``order_times`` を渡すと、その時刻順を経路順として並べ替える。時刻を持たない
    ランドマークは通過したと言えないため経路から除く。
    """
    landmark_items = tuple(
        Landmark(
            beacon_id=str(beacon_id),
            pixel_x=float(pixel_x),
            pixel_y=float(pixel_y),
        )
        for beacon_id, pixel_x, pixel_y in landmarks
    )
    if order_times is not None:
        landmark_items = tuple(
            sorted(
                (item for item in landmark_items if item.beacon_id in order_times),
                key=lambda item: order_times[item.beacon_id],
            )
        )
    return landmark_items, build_landmark_path(landmark_items, scale)
