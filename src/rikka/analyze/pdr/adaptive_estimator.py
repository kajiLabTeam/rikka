"""旧 adaptive 推定 import の互換 shim。"""
# ruff: noqa: F401

import sys
from typing import TYPE_CHECKING

from ...pdr.lib.fusion import adaptive as _implementation

__all__ = [
    "AdaptivePdrEstimator",
    "_smooth_mode_probabilities",
    "estimate_adaptive_pdr",
]

if TYPE_CHECKING:
    from ...pdr.lib.fusion.adaptive import (
        AdaptivePdrEstimator,
        _smooth_mode_probabilities,
        estimate_adaptive_pdr,
    )
else:
    sys.modules[__name__] = _implementation
