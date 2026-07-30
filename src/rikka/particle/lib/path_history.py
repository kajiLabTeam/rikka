"""粒子祖先経路の復元と到達可能クラスタの抽出。

役割:
    親インデックス履歴から粒子ごとの経路を復元し、相互に到達可能な
    粒子のうち合計重み最大のクラスタを返す。
依存元:
    地図遷移判定と NumPy の配列演算を利用する。
利用先:
    path selection と particle 結果確定段階から使用される。
処理フロー:
    終端から祖先を逆向きに追跡し、重み最大の粒子を起点にクラスタ化する。
"""

import numpy as np

from .map_constraints import _evaluate_particle_transitions


def _reconstruct_resampled_paths(
    position_history: list[np.ndarray],
    resample_history: list[np.ndarray],
) -> np.ndarray:
    """リサンプリング祖先をたどって最終粒子群の経路を復元する。"""
    if not position_history:
        return np.empty((0, 0, 2), dtype=float)

    n_particles = position_history[0].shape[0]
    n_steps = len(position_history) - 1
    if len(resample_history) != n_steps:
        raise ValueError("position_history と resample_history の長さが一致しません")

    paths = np.empty((n_particles, n_steps + 1, 2), dtype=float)
    if n_steps == 0:
        paths[:, 0, :] = position_history[0]
        return paths

    lineage = resample_history[-1].astype(int, copy=True)
    paths[:, n_steps, :] = position_history[n_steps][lineage]
    for step in range(n_steps - 1, 0, -1):
        lineage = resample_history[step - 1][lineage]
        paths[:, step, :] = position_history[step][lineage]
    paths[:, 0, :] = position_history[0][lineage]
    return paths


def _reconstruct_particle_paths(
    position_history: list[np.ndarray],
    parent_history: list[np.ndarray],
) -> np.ndarray:
    """各ステップ後の親インデックスから最終粒子の有効経路を復元する。"""
    if not position_history:
        return np.empty((0, 0, 2), dtype=float)
    n_particles = position_history[0].shape[0]
    n_steps = len(position_history) - 1
    if len(parent_history) != n_steps:
        raise ValueError("position_history と parent_history の長さが一致しません")

    paths = np.empty((n_particles, n_steps + 1, 2), dtype=float)
    lineage = np.arange(n_particles, dtype=int)
    for step in range(n_steps, 0, -1):
        paths[:, step, :] = position_history[step][lineage]
        lineage = parent_history[step - 1][lineage]
    paths[:, 0, :] = position_history[0][lineage]
    return paths


def _dominant_reachable_cluster(
    positions: np.ndarray,
    weights: np.ndarray,
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> np.ndarray:
    """直線到達できる粒子を分け、合計重み最大のクラスタを返す。"""
    if positions.shape != (len(weights), 2):
        raise ValueError("positions と weights の粒子数が一致しません")
    remaining = weights > 0.0
    clusters: list[np.ndarray] = []
    while remaining.any():
        remaining_indices = np.flatnonzero(remaining)
        seed_local_index = int(np.argmax(weights[remaining_indices]))
        seed_index = int(remaining_indices[seed_local_index])
        seed_positions = np.repeat(
            positions[seed_index : seed_index + 1],
            len(remaining_indices),
            axis=0,
        )
        reachable = _evaluate_particle_transitions(
            seed_positions,
            positions[remaining_indices],
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        # seed自身は歩行可能なはずだが、不正な入力でも必ず処理を前進させる。
        reachable[seed_local_index] = True
        cluster = remaining_indices[reachable]
        clusters.append(cluster)
        remaining[cluster] = False

    cluster_weights = np.asarray(
        [float(weights[cluster].sum()) for cluster in clusters],
        dtype=float,
    )
    return clusters[int(np.argmax(cluster_weights))]
