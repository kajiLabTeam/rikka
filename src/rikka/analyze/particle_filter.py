"""従来の particle filter API を維持する薄い互換 shim。

実装は ``particle`` と ``plot`` 領域にあり、このモジュールは既存の import パスへ
必要なシンボルを再輸出するだけである。
"""

from ..particle.lib.map_constraints import (
    _evaluate_particle_transitions,
    _normalize_floormap_gray,
    _snap_trajectory_to_walkable_pixels,
)
from ..particle.lib.path_history import (
    _reconstruct_particle_paths,
    _reconstruct_resampled_paths,
)
from ..particle.lib.path_selection import (
    _select_reachable_cluster_path,
    _select_reachable_mean_path,
)
from ..particle.lib.proposal import _motion_state_headings, _sample_motion_states
from ..particle.lib.recorder import (
    ParticleFilterStepDiagnostics,
    ParticlePathComparison,
    ParticleStepStages,
)
from ..particle.lib.recovery.checkpoint import _replay_from_checkpoint
from ..particle.lib.recovery.local import _generate_recovery_candidates
from ..particle.lib.resampling import _effective_sample_size, _systematic_resample
from ..particle.lib.runner import run_particle_filter
from ..particle.lib.sequence_path import (
    _select_sequence_map_path,
)
from ..plot.lib.animation import (
    plot_particle_filter_trajectory,
    save_particle_animation,
)

__all__ = [
    "ParticleFilterStepDiagnostics",
    "ParticlePathComparison",
    "ParticleStepStages",
    "_effective_sample_size",
    "_evaluate_particle_transitions",
    "_generate_recovery_candidates",
    "_motion_state_headings",
    "_normalize_floormap_gray",
    "_reconstruct_particle_paths",
    "_reconstruct_resampled_paths",
    "_replay_from_checkpoint",
    "_sample_motion_states",
    "_select_reachable_cluster_path",
    "_select_reachable_mean_path",
    "_select_sequence_map_path",
    "_snap_trajectory_to_walkable_pixels",
    "_systematic_resample",
    "plot_particle_filter_trajectory",
    "run_particle_filter",
    "save_particle_animation",
]
