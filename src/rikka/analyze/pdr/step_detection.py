"""旧ステップ検出 import の互換 shim。"""

from ...common.lib.validation import STEP_DETECTION_METHODS
from ...pdr.lib.step_detection import detect_step_result, detect_steps

__all__ = [
    "STEP_DETECTION_METHODS",
    "detect_step_result",
    "detect_steps",
]
