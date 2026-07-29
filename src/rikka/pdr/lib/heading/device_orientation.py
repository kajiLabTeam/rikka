"""端末姿勢モード推定の公開部品。"""

from .resolver import (
    _apply_device_orientation_to_horizontal as apply_device_orientation_to_horizontal,
)
from .resolver import (
    _estimate_device_orientation_mode as estimate_device_orientation_mode,
)

__all__ = [
    "apply_device_orientation_to_horizontal",
    "estimate_device_orientation_mode",
]

