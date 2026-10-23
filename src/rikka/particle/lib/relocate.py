"""歩行不可能領域へ落ちた粒子の円内再配置。

役割:
    提案で壁や地図外へ出た粒子だけを、前歩位置を中心とする正円の歩行可能範囲へ
    一様分布で引き直し、重み0で捨てる前に復帰の機会を与える。
依存元:
    ``map_constraints._evaluate_particle_transitions`` で到達可否を判定し、
    NumPy の乱数と配列演算を利用する。
利用先:
    ``particle.lib.propose`` が地図判定の直後に呼び、再配置後の位置と
    更新済みの遷移可否を受け取る。
処理フロー:
    無効粒子を抽出し、円内一様分布の候補を生成して前歩位置からの線分が
    歩行可能画素だけを通る候補を採用する。試行上限まで未成立の粒子は
    元の提案位置と無効判定を維持する。
"""

import numpy as np

from .map_constraints import _evaluate_particle_transitions


def _sample_uniform_disk(
    centers: np.ndarray,
    radius_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """中心ごとに半径 ``radius_m`` の正円内の一様分布点を返す。

    面積一様にするため半径方向は ``radius_m * sqrt(U)`` を使う。角度へ一様な
    半径を掛けると中心付近へ偏り、円内一様にならない。
    """
    count = len(centers)
    angles = rng.uniform(0.0, 2.0 * np.pi, count)
    radii = radius_m * np.sqrt(rng.uniform(0.0, 1.0, count))
    offsets = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    return centers + offsets


def relocate_invalid_particles(
    previous_particles: np.ndarray,
    proposed_particles: np.ndarray,
    valid_transition: np.ndarray,
    radius_m: float,
    max_attempts: int,
    rng: np.random.Generator,
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """無効粒子を前歩位置中心の円内へ再配置し、位置と遷移可否を返す。

    戻り値は (再配置後の提案位置, 更新後の遷移可否, 再配置に成功した粒子数)。
    ``radius_m`` が 0 以下、または無効粒子が無い場合は入力をそのまま返す。
    """
    if radius_m <= 0.0 or max_attempts <= 0:
        return proposed_particles, valid_transition, 0
    pending = ~np.asarray(valid_transition, dtype=bool)
    if not pending.any():
        return proposed_particles, valid_transition, 0

    relocated = np.array(proposed_particles, dtype=float, copy=True)
    updated_valid = np.array(valid_transition, dtype=bool, copy=True)
    relocated_count = 0
    for _ in range(max_attempts):
        pending_indices = np.flatnonzero(pending)
        if len(pending_indices) == 0:
            break
        centers = previous_particles[pending_indices]
        candidates = _sample_uniform_disk(centers, radius_m, rng)
        # 円内の点でも壁を越える移動は採用しない。地図拘束を保ったまま
        # 同一の歩行可能領域内へ戻せた粒子だけを復帰させる。
        accepted = _evaluate_particle_transitions(
            centers,
            candidates,
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        accepted_indices = pending_indices[accepted]
        relocated[accepted_indices] = candidates[accepted]
        updated_valid[accepted_indices] = True
        pending[accepted_indices] = False
        relocated_count += int(len(accepted_indices))
    return relocated, updated_valid, relocated_count
