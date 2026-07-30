"""旧方位推定 import の互換 shim。"""

from typing import Any

from ...pdr.lib.heading import (
    accel,
    device_orientation,
    gyro,
    motion,
    resolver,
)

_MODULES = (device_orientation, gyro, accel, motion, resolver)


def __getattr__(name: str) -> Any:
    """分割後の所有モジュールから互換シンボルを取得する。"""
    for module in _MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)
