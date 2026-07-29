"""旧 particle 描画 import の互換 shim。"""

from ...plot.lib.animation import (
    plot_particle_filter_trajectory,
    save_particle_animation,
)

__all__ = ["plot_particle_filter_trajectory", "save_particle_animation"]
