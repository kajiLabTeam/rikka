"""旧運動状態復号 import の互換 shim。"""

from ...pdr.lib.motion_state.decoder import (
    DecodedMotionSegment,
    MotionDecodeResult,
    decode_step_motion_modes,
    decode_step_motion_segments,
)

__all__ = [
    "DecodedMotionSegment",
    "MotionDecodeResult",
    "decode_step_motion_modes",
    "decode_step_motion_segments",
]
