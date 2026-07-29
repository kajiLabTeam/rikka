"""旧方位推定 import の互換 shim。"""
# ruff: noqa: F401

import sys
from typing import TYPE_CHECKING

from ...pdr.lib.heading import resolver as _implementation

__all__ = [
    "_AccelHeadingResult",
    "_apply_device_orientation_to_horizontal",
    "_classify_movement_type",
    "_dataframe_times_or_sample_index",
    "_estimate_accel_headings",
    "_estimate_device_orientation_mode",
    "_estimate_motion_heading_correction",
    "_estimate_motion_heading_from_horizontal_accel",
    "_integrate_motion_with_zero_velocity",
    "_MotionHeadingResult",
    "_resolve_motion_heading_correction",
    "_rotate_vector",
    "_select_two_accel_peaks",
    "_step_segment_bounds",
    "resolve_step_heading",
]

if TYPE_CHECKING:
    from ...pdr.lib.heading.resolver import (
        _AccelHeadingResult,
        _apply_device_orientation_to_horizontal,
        _classify_movement_type,
        _dataframe_times_or_sample_index,
        _estimate_accel_headings,
        _estimate_device_orientation_mode,
        _estimate_motion_heading_correction,
        _estimate_motion_heading_from_horizontal_accel,
        _integrate_motion_with_zero_velocity,
        _MotionHeadingResult,
        _resolve_motion_heading_correction,
        _rotate_vector,
        _select_two_accel_peaks,
        _step_segment_bounds,
        resolve_step_heading,
    )
else:
    sys.modules[__name__] = _implementation
