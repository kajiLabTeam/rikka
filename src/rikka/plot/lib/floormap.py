"""PDR/PF 共通のフロアマップ座標変換。

役割:
    メートル座標と方位をフロアマップ上の画素座標へ変換する。
依存元:
    ``trajectory`` の確定済み座標変換実装を取得する。
利用先:
    軌跡図、particle frame、animation から共通利用する。
処理フロー:
    座標列または方位ベクトルを既存と同じ加算順で変換する。
"""

from .trajectory import (
    _compute_pixel_coords as compute_pixel_coords,
)
from .trajectory import (
    _pixel_vector_from_heading as pixel_vector_from_heading,
)
from .trajectory import (
    _plot_heading_overlay as plot_heading_overlay,
)

__all__ = [
    "compute_pixel_coords",
    "pixel_vector_from_heading",
    "plot_heading_overlay",
]
