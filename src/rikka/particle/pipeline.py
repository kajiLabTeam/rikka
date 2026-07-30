"""PreparedPdrSteps を地図制約付き代表軌跡へ変換する pipeline。

役割:
    通常 PDR で確定した歩列だけを受け、particle filter を実行する。
依存元:
    ``common`` の共有型・設定と既存互換 runner の数値実装を使用する。
利用先:
    CLI が ``pdr.pipeline.run_pdr`` の後にPF指定時だけ呼び出す。
処理フロー:
    診断収集器を用意し、準備済み歩列をrunnerへ渡して共有結果型へまとめる。
"""

from ..common.lib.models import (
    FloorMap,
    ParticleFilterResult,
    PreparedPdrSteps,
    TrajectoryResult,
)
from ..common.settings import ParticleSettings
from .lib.engine import _run_particle_steps
from .lib.recorder import (
    ParticleFilterStepDiagnostics,
    ParticlePathComparison,
    ParticleStepStages,
)


def run_particle(
    prepared: PreparedPdrSteps,
    floormap: FloorMap,
    settings: ParticleSettings,
) -> TrajectoryResult:
    """準備済みの歩列へ地図拘束を適用する。"""
    diagnostics: list[ParticleFilterStepDiagnostics] = []
    stages: list[ParticleStepStages] = []
    path_comparisons: list[ParticlePathComparison] = []
    trajectory, lengths, times, all_particles, headings = _run_particle_steps(
        prepared.step_detection.peaks,
        prepared.df_gyro,
        prepared.df_acc,
        prepared.gx_mean,
        prepared.gz_mean,
        floormap_path=floormap.path,
        origin_px=floormap.origin_px,
        scale=floormap.scale,
        n_particles=settings.count,
        prepared_step_headings=prepared.step_headings,
        prepared_step_lengths=prepared.step_lengths,
        prepared_step_times=prepared.t_at_steps,
        prepared_motion_evidences=prepared.motion_evidences,
        prepared_particle_motion_headings=prepared.particle_motion_headings,
        prepared_motion_posteriors=prepared.motion_posteriors,
        seed=settings.seed,
        motion_predictive_weight_power=settings.motion_predictive_weight_power,
        path_selection=settings.path_selection,
        diagnostics_collector=diagnostics,
        stage_collector=stages,
        path_comparison_collector=path_comparisons,
    )
    particle = ParticleFilterResult(
        trajectory=trajectory,
        all_particles=[item for item in all_particles],
        diagnostics=tuple(diagnostics),
        stages=tuple(stages),
        path_comparisons=tuple(path_comparisons),
    )
    return TrajectoryResult(
        trajectory=trajectory,
        step_lengths=lengths,
        t_at_steps=times,
        step_headings=headings,
        prepared=prepared,
        particle=particle,
    )
