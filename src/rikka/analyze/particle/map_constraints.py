"""旧地図拘束 import の互換 shim。"""
# ruff: noqa: F401

from ...particle.lib.map_constraints import (
    _evaluate_particle_transitions,
    _normalize_floormap_gray,
    _snap_trajectory_to_walkable_pixels,
)

__all__ = [
    "_evaluate_particle_transitions",
    "_normalize_floormap_gray",
    "_snap_trajectory_to_walkable_pixels",
]
