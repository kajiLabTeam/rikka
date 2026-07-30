"""旧 resampling import の互換 shim。"""
# ruff: noqa: F401

from ...particle.lib.resampling import (
    _effective_sample_size,
    _systematic_resample,
)

__all__ = [
    "_effective_sample_size",
    "_systematic_resample",
]
