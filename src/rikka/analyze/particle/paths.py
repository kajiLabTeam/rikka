"""粒子祖先経路の復元と到達可能な代表軌跡の選択。

役割:
    粒子の親インデックス履歴から全経路を復元し、重み付き平均を優先しつつ地図上で
    到達可能な代表軌跡を構成する。
依存元:
    ``map_constraints`` の粒子遷移判定とNumPyの配列演算を利用する。
利用先:
    粒子フィルタの実行処理が最終軌跡と各時点の採用元を決定するために使用する。
処理フロー:
    祖先を終端から逆向きに追跡し、平均状態と同一祖先状態の動的計画法で経路を選ぶ。
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


def _select_reachable_mean_path(
    particle_paths: np.ndarray,
    final_weights: np.ndarray,
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> tuple[np.ndarray, list[str], list[int | None]]:
    """平均を優先し、壁区間だけ同じ粒子祖先へ退避した経路を返す。"""
    positive_indices = np.flatnonzero(final_weights > 0.0)
    if positive_indices.size == 0:
        raise RuntimeError("正の最終重みを持つ粒子がありません")

    positive_paths = particle_paths[positive_indices]
    positive_weights = final_weights[positive_indices].astype(float, copy=True)
    positive_weights /= positive_weights.sum()
    mean_path = np.average(positive_paths, axis=0, weights=positive_weights)

    n_paths, n_times, _ = positive_paths.shape
    n_states = n_paths + 1  # 0は平均、1以降は同じ祖先を追う粒子状態
    costs = np.zeros(n_states, dtype=float)
    backpointers = np.full((n_times, n_states), -1, dtype=int)

    for time_index in range(1, n_times):
        next_costs = np.full(n_states, np.inf, dtype=float)

        mean_to_mean = bool(
            _evaluate_particle_transitions(
                mean_path[time_index - 1 : time_index],
                mean_path[time_index : time_index + 1],
                map_gray,
                gx_mean,
                gz_mean,
                origin_px,
                scale,
            )[0]
        )
        if mean_to_mean and np.isfinite(costs[0]):
            next_costs[0] = costs[0]
            backpointers[time_index, 0] = 0

        particle_to_mean = _evaluate_particle_transitions(
            positive_paths[:, time_index - 1, :],
            np.repeat(mean_path[None, time_index, :], n_paths, axis=0),
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        mean_predecessor_costs = np.where(particle_to_mean, costs[1:], np.inf)
        best_particle = int(np.argmin(mean_predecessor_costs))
        if mean_predecessor_costs[best_particle] < next_costs[0]:
            next_costs[0] = mean_predecessor_costs[best_particle]
            backpointers[time_index, 0] = best_particle + 1

        deviations = np.sum(
            np.square(positive_paths[:, time_index, :] - mean_path[time_index]),
            axis=1,
        )
        # 同じ祖先を追い続ける遷移は、PF伝播時に壁判定済みである。
        same_path_costs = costs[1:] + deviations
        next_costs[1:] = same_path_costs
        backpointers[time_index, 1:] = np.arange(1, n_states)

        mean_to_particle = _evaluate_particle_transitions(
            np.repeat(mean_path[None, time_index - 1, :], n_paths, axis=0),
            positive_paths[:, time_index, :],
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        enter_costs = np.where(mean_to_particle, costs[0] + deviations, np.inf)
        use_mean_predecessor = enter_costs < next_costs[1:]
        particle_state_indices = np.flatnonzero(use_mean_predecessor) + 1
        next_costs[particle_state_indices] = enter_costs[use_mean_predecessor]
        backpointers[time_index, particle_state_indices] = 0

        if not np.isfinite(next_costs).any():
            raise RuntimeError("到達可能な平均・粒子経路を構成できません")
        costs = next_costs

    state = int(np.argmin(costs))
    states = np.empty(n_times, dtype=int)
    states[-1] = state
    for time_index in range(n_times - 1, 0, -1):
        state = int(backpointers[time_index, state])
        if state < 0:
            raise RuntimeError("軌跡の親状態を復元できません")
        states[time_index - 1] = state

    selected_path = np.empty_like(mean_path)
    modes: list[str] = []
    sources: list[int | None] = []
    for time_index, selected_state in enumerate(states):
        if selected_state == 0:
            selected_path[time_index] = mean_path[time_index]
            modes.append("weighted_mean")
            sources.append(None)
        else:
            local_index = selected_state - 1
            selected_path[time_index] = positive_paths[local_index, time_index]
            modes.append("particle_fallback")
            sources.append(int(positive_indices[local_index]))

    if (
        n_times > 1
        and not _evaluate_particle_transitions(
            selected_path[:-1],
            selected_path[1:],
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        ).all()
    ):
        raise RuntimeError("構成した軌跡に壁またはマップ外遷移が含まれます")
    return selected_path, modes, sources
