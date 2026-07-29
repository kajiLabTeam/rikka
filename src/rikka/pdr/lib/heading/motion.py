"""水平加速度移動方位の公開部品。"""

from .resolver import (
    estimate_motion_heading_correction,
    resolve_motion_heading_correction,
)
from .resolver import (
    estimate_motion_heading_from_horizontal_accel as estimate_motion_heading,
)

__all__ = [
    "estimate_motion_heading",
    "estimate_motion_heading_correction",
    "resolve_motion_heading_correction",
]
