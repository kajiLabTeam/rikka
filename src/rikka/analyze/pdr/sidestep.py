"""旧運動状態・横歩き推定 import の互換 shim。"""

from typing import Any

from ...pdr.lib.motion_state import (
    clustering,
    evidence,
    heading_policy,
    step_motion,
)

_MODULES = (evidence, clustering, heading_policy, step_motion)


def __getattr__(name: str) -> Any:
    """分割後の所有モジュールから互換シンボルを取得する。"""
    for module in _MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)
