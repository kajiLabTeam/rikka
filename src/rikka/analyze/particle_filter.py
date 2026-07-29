"""パーティクルフィルタの後方互換facade。

役割:
    責務別に分割した粒子フィルタ内部実装から、従来の公開関数と既存利用中のprivate
    helperを同じモジュールパスで再輸出する。
依存元:
    ``particle`` パッケージの実行、診断型、再標本化、運動、地図、経路、復旧、描画
    モジュールから互換対象シンボルを取得する。
利用先:
    ``pdr.pipeline.run`` の遅延import、CLI、テスト、診断スクリプトが従来どおり
    ``rikka.analyze.particle_filter`` から使用する。
処理フロー:
    import時に互換シンボルを各所有モジュールへ束ね、計算や描画は移動先の実装へ委譲する。
"""

from .particle.map_constraints import (
    _evaluate_particle_transitions,
    _normalize_floormap_gray,
    _snap_trajectory_to_walkable_pixels,
)
from .particle.models import (
    ParticleFilterStepDiagnostics,
    ParticlePathComparison,
    ParticleStepStages,
)
from .particle.motion import _motion_state_headings, _sample_motion_states
from .particle.paths import (
    _reconstruct_particle_paths,
    _reconstruct_resampled_paths,
    _select_reachable_cluster_path,
    _select_reachable_mean_path,
    _select_sequence_map_path,
)
from .particle.plotting import plot_particle_filter_trajectory, save_particle_animation
from .particle.recovery import (
    _generate_recovery_candidates,
    _replay_from_checkpoint,
)
from .particle.resampling import _effective_sample_size, _systematic_resample
from .particle.runner import run_particle_filter

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
