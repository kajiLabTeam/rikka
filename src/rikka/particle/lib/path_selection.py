"""到達可能クラスタによる代表軌跡の選択。

役割:
    壁越しに混ざらない粒子クラスタの重み付き平均を優先しつつ、
    地図上で到達可能な代表軌跡を構成する。
依存元:
    ``map_constraints`` の粒子遷移判定とNumPyの配列演算を利用する。
利用先:
    particle の結果確定段階が代表軌跡を決定するために使用する。
処理フロー:
    各時点で相互に直線到達できる粒子をクラスタ化し、合計重み最大の
    クラスタ平均と祖先状態から動的計画法で経路を選ぶ。
"""

import numpy as np

from .map_constraints import _evaluate_particle_transitions
from .path_history import _dominant_reachable_cluster


def _select_reachable_cluster_path(
    position_history: list[np.ndarray],
    weight_history: list[np.ndarray],
    parent_history: list[np.ndarray],
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> tuple[np.ndarray, list[str], list[int | None]]:
    """各時点の有力クラスタ平均から、壁を横切らない軌跡を構成する。"""
    if not position_history:
        raise ValueError("position_history は1時点以上必要です")
    if len(weight_history) != len(position_history):
        raise ValueError("position_history と weight_history の長さが一致しません")
    if len(parent_history) != len(position_history) - 1:
        raise ValueError("position_history と parent_history の長さが一致しません")

    n_times = len(position_history)
    n_particles = len(position_history[0])
    means = np.empty((n_times, 2), dtype=float)
    memberships = np.zeros((n_times, n_particles), dtype=bool)
    positive_memberships = np.zeros((n_times, n_particles), dtype=bool)
    mean_is_walkable = np.zeros(n_times, dtype=bool)
    for time_index, (positions, weights) in enumerate(
        zip(position_history, weight_history, strict=True)
    ):
        if positions.shape != (n_particles, 2):
            raise ValueError("position_history 内の粒子配列 shape が一致しません")
        if weights.shape != (n_particles,):
            raise ValueError("weight_history 内の重み配列 shape が一致しません")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("weight_history には有限な非負値を指定してください")
        if float(weights.sum()) <= 0.0:
            raise RuntimeError("正の重みを持つ粒子がありません")
        positive_memberships[time_index] = weights > 0.0

        cluster = _dominant_reachable_cluster(
            positions,
            weights,
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        memberships[time_index, cluster] = True
        cluster_weights = weights[cluster].astype(float, copy=True)
        cluster_weights /= cluster_weights.sum()
        means[time_index] = np.average(
            positions[cluster],
            axis=0,
            weights=cluster_weights,
        )
        mean_is_walkable[time_index] = bool(
            _evaluate_particle_transitions(
                means[time_index : time_index + 1],
                means[time_index : time_index + 1],
                map_gray,
                gx_mean,
                gz_mean,
                origin_px,
                scale,
            )[0]
        )

    n_states = n_particles + 1
    costs = np.full(n_states, np.inf, dtype=float)
    if mean_is_walkable[0]:
        costs[0] = 0.0
    initial_deviations = np.sum(
        np.square(position_history[0] - means[0]),
        axis=1,
    )
    # 優勢クラスタ外の系譜も高いコストで保持し、クラスタ交代時の行き止まりを防ぐ。
    outside_cluster_penalty = 1e6
    initial_penalties = np.where(memberships[0], 0.0, outside_cluster_penalty)
    costs[1:] = np.where(
        positive_memberships[0],
        initial_deviations + initial_penalties,
        np.inf,
    )
    backpointers = np.full((n_times, n_states), -1, dtype=int)

    for time_index in range(1, n_times):
        previous_positions = position_history[time_index - 1]
        current_positions = position_history[time_index]
        parents = np.asarray(parent_history[time_index - 1], dtype=int)
        if parents.shape != (n_particles,) or np.any(
            (parents < 0) | (parents >= n_particles)
        ):
            raise ValueError("parent_history に範囲外の親インデックスがあります")

        next_costs = np.full(n_states, np.inf, dtype=float)
        if mean_is_walkable[time_index]:
            mean_predecessor_positions = np.vstack(
                [means[time_index - 1], previous_positions]
            )
            mean_targets = np.repeat(
                means[time_index : time_index + 1],
                n_states,
                axis=0,
            )
            reachable_to_mean = _evaluate_particle_transitions(
                mean_predecessor_positions,
                mean_targets,
                map_gray,
                gx_mean,
                gz_mean,
                origin_px,
                scale,
            )
            predecessor_costs = np.where(reachable_to_mean, costs, np.inf)
            best_predecessor = int(np.argmin(predecessor_costs))
            if np.isfinite(predecessor_costs[best_predecessor]):
                next_costs[0] = predecessor_costs[best_predecessor]
                backpointers[time_index, 0] = best_predecessor

        deviations = np.sum(
            np.square(current_positions - means[time_index]),
            axis=1,
        )
        cluster_penalties = np.where(
            memberships[time_index],
            0.0,
            outside_cluster_penalty,
        )
        lineage_costs = costs[parents + 1] + deviations + cluster_penalties
        valid_lineages = positive_memberships[time_index] & np.isfinite(lineage_costs)
        next_costs[1:][valid_lineages] = lineage_costs[valid_lineages]
        backpointers[time_index, 1:][valid_lineages] = parents[valid_lineages] + 1

        mean_starts = np.repeat(
            means[time_index - 1 : time_index],
            n_particles,
            axis=0,
        )
        mean_to_particles = _evaluate_particle_transitions(
            mean_starts,
            current_positions,
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        enter_costs = costs[0] + deviations + cluster_penalties
        enter_from_mean = (
            positive_memberships[time_index]
            & mean_to_particles
            & (enter_costs < next_costs[1:])
        )
        next_costs[1:][enter_from_mean] = enter_costs[enter_from_mean]
        backpointers[time_index, 1:][enter_from_mean] = 0

        if not np.isfinite(next_costs).any():
            raise RuntimeError("到達可能クラスタから合法な代表軌跡を構成できません")
        costs = next_costs

    state = int(np.argmin(costs))
    states = np.empty(n_times, dtype=int)
    states[-1] = state
    for time_index in range(n_times - 1, 0, -1):
        state = int(backpointers[time_index, state])
        if state < 0:
            raise RuntimeError("クラスタ軌跡の親状態を復元できません")
        states[time_index - 1] = state

    selected_path = np.empty_like(means)
    modes: list[str] = []
    sources: list[int | None] = []
    for time_index, selected_state in enumerate(states):
        if selected_state == 0:
            selected_path[time_index] = means[time_index]
            modes.append("weighted_mean")
            sources.append(None)
        else:
            particle_index = selected_state - 1
            selected_path[time_index] = position_history[time_index][particle_index]
            modes.append("particle_fallback")
            sources.append(particle_index)

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
        raise RuntimeError("構成したクラスタ軌跡に壁またはマップ外遷移が含まれます")
    return selected_path, modes, sources
