"""一様初期化PFの粗グリッドクラスタとオンライン収束判定。

役割:
    正の重みを持つ粒子の空間的な集中を観測し、大域探索の終了を判定する。
依存元:
    common.lib.models の共有診断型、NumPy、scipy.ndimage.label と
    ParticleRuntime の状態を利用する。到達可能クラスタの線分判定は行わない。
利用先:
    runner が各歩の記録後に呼び、finalize が最終クラスタの粒子選択に使う。
処理フロー:
    占有セルを4近傍で連結成分化し、重み和と重心・RMSを集計する。
    重み比と広がりの連続条件を満たすと収束位置を保持する。
    通常PFへ戻すかは convergence_restore_normal で選び、既定では戻さない。
"""

from dataclasses import replace

import numpy as np
from scipy.ndimage import label

from ...common.config import PF_CONVERGENCE_CELL_M, PF_CONVERGENCE_TOP_K
from ...common.lib.models import ParticleClusterSummary, ParticleConvergenceDiagnostic
from .state import ParticleRuntime


def cluster_distribution(
    positions: np.ndarray,
    weights: np.ndarray,
    cell_m: float = PF_CONVERGENCE_CELL_M,
    top_k: int = PF_CONVERGENCE_TOP_K,
    *,
    step_number: int = 0,
    step_time: float = 0.0,
) -> tuple[ParticleConvergenceDiagnostic, np.ndarray]:
    """分布の診断と最大重みクラスタの元粒子添字を返す。ゼロ重みは除く。"""
    alive = np.flatnonzero(weights > 0.0)
    if not len(alive):
        return (
            ParticleConvergenceDiagnostic(
                step_number,
                step_time,
                0,
                0.0,
                float("inf"),
                None,
                None,
                float("inf"),
                0,
            ),
            alive,
        )
    points = positions[alive]
    mass = weights[alive]
    cells = np.floor(points / cell_m).astype(np.int64)
    cells -= cells.min(axis=0)
    occupied = np.zeros(tuple(cells.max(axis=0) + 1), dtype=bool)
    occupied[cells[:, 0], cells[:, 1]] = True
    components, count = label(occupied)
    particle_labels = components[cells[:, 0], cells[:, 1]]
    sums = np.bincount(particle_labels, weights=mass, minlength=count + 1)
    centroids = np.zeros((count + 1, 2))
    for axis in range(2):
        centroids[1:, axis] = (
            np.bincount(
                particle_labels, weights=mass * points[:, axis], minlength=count + 1
            )[1:]
            / sums[1:]
        )
    residuals = np.sum((points - centroids[particle_labels]) ** 2, axis=1)
    variances = np.bincount(
        particle_labels, weights=mass * residuals, minlength=count + 1
    )
    # 同重みなら成分番号で決め、固定seedの再現性を保つ。
    ranked = np.argsort(-sums[1:], kind="stable") + 1
    total_mass = sums[1:].sum()
    top = tuple(
        ParticleClusterSummary(
            float(centroids[index, 0]),
            float(centroids[index, 1]),
            float(sums[index] / total_mass),
            float(np.sqrt(variances[index] / sums[index])),
        )
        for index in ranked[:top_k]
    )
    dominant = top[0]
    center = np.average(points, axis=0, weights=mass)
    spread = float(
        np.sqrt(np.average(np.sum((points - center) ** 2, axis=1), weights=mass))
    )
    return (
        ParticleConvergenceDiagnostic(
            step_number,
            step_time,
            count,
            dominant.weight_ratio,
            dominant.spread_rms_m,
            dominant.centroid_x_m,
            dominant.centroid_y_m,
            spread,
            len(alive),
            top,
        ),
        alive[particle_labels == ranked[0]],
    )


def record_convergence(ctx: ParticleRuntime) -> None:
    """収束を一度だけ確定し、確定後も分布診断を記録する。"""
    diagnostic, _ = cluster_distribution(
        ctx.particles,
        ctx.weights,
        ctx.convergence_cell_m,
        ctx.convergence_top_k,
        step_number=ctx.step_number,
        step_time=ctx.step_time,
    )
    if ctx.convergence is None:
        matches = (
            diagnostic.dominant_weight_ratio >= ctx.convergence_weight_ratio
            and diagnostic.dominant_spread_rms_m <= ctx.convergence_spread_m
        )
        ctx.convergence_streak = ctx.convergence_streak + 1 if matches else 0
        if ctx.convergence_streak >= ctx.convergence_single_steps:
            ctx.convergence = replace(diagnostic, convergence_step=ctx.step_number)
            # 既定では大域フェーズのまま続ける。4号館1024で収束後に通常PFへ戻すと
            # ESS閾値が0.2から0.5へ上がって全体再標本化が頻発し、正解系譜が失われて
            # 代表経路の平均誤差が 0.78 m から 24.91 m へ悪化した。
            if ctx.convergence_restore_normal:
                ctx.global_phase_active = False
    diagnostic = replace(
        diagnostic,
        convergence_step=(
            None if ctx.convergence is None else ctx.convergence.step_number
        ),
    )
    ctx.convergence_diagnostics.append(diagnostic)
    if ctx.convergence_collector is not None:
        ctx.convergence_collector.append(diagnostic)
