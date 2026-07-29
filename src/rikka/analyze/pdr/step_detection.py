"""旧ステップ検出 import の互換 shim。"""

from ...pdr.lib.step_detection import (
    STEP_DETECTION_METHODS,
    _detect_steps_by_peak,
    _detect_steps_by_vertical_threshold,
    _suppress_close_contacts,
    _threshold_groups,
    _validate_step_detection_method,
    detect_step_result,
    detect_steps,
)

__all__ = [
    "STEP_DETECTION_METHODS",
    "_detect_steps_by_peak",
    "_detect_steps_by_vertical_threshold",
    "_suppress_close_contacts",
    "_threshold_groups",
    "_validate_step_detection_method",
    "detect_step_result",
    "detect_steps",
]
