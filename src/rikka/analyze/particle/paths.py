"""粒子祖先経路の復元と到達可能クラスタによる代表軌跡の選択。

役割:
    粒子の親インデックス履歴から全経路を復元し、壁越しに混ざらない粒子クラスタの
    重み付き平均を優先しつつ、地図上で到達可能な代表軌跡を構成する。
依存元:
    ``map_constraints`` の粒子遷移判定とNumPyの配列演算を利用する。
利用先:
    粒子フィルタの実行処理が最終軌跡と各時点の採用元を決定するために使用する。
処理フロー:
    祖先を終端から逆向きに追跡し、各時点で相互に直線到達できる粒子をクラスタ化する。
    合計重み最大のクラスタ平均と、そのクラスタ内の祖先状態から動的計画法で経路を
    選ぶ。
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


def _select_reachable_mean_path(
    particle_paths: np.ndarray,
    final_weights: np.ndarray,
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> tuple[np.ndarray, list[str], list[int | None]]:
    """有力な到達可能クラスタの平均を優先した合法な経路を返す。"""
    positive_indices = np.flatnonzero(final_weights > 0.0)
    if positive_indices.size == 0:
        raise RuntimeError("正の最終重みを持つ粒子がありません")

    positive_paths = particle_paths[positive_indices]
    positive_weights = final_weights[positive_indices].astype(float, copy=True)
    positive_weights /= positive_weights.sum()
    n_paths, n_times, _ = positive_paths.shape
    mean_path = np.empty((n_times, 2), dtype=float)
    cluster_membership = np.zeros((n_times, n_paths), dtype=bool)
    mean_is_walkable = np.zeros(n_times, dtype=bool)
    for time_index in range(n_times):
        cluster_indices = _dominant_reachable_cluster(
            positive_paths[:, time_index, :],
            positive_weights,
            map_gray,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        cluster_membership[time_index, cluster_indices] = True
        cluster_weights = positive_weights[cluster_indices]
        cluster_weights /= cluster_weights.sum()
        mean_path[time_index] = np.average(
            positive_paths[cluster_indices, time_index, :],
            axis=0,
            weights=cluster_weights,
        )
        mean_is_walkable[time_index] = bool(
            _evaluate_particle_transitions(
                mean_path[time_index : time_index + 1],
                mean_path[time_index : time_index + 1],
                map_gray,
                gx_mean,
                gz_mean,
                origin_px,
                scale,
            )[0]
        )

    n_states = n_paths + 1  # 0は平均、1以降は同じ祖先を追う粒子状態
    costs = np.full(n_states, np.inf, dtype=float)
    if mean_is_walkable[0]:
        costs[0] = 0.0
    costs[1:][cluster_membership[0]] = np.sum(
        np.square(positive_paths[cluster_membership[0], 0, :] - mean_path[0]),
        axis=1,
    )
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
        if mean_is_walkable[time_index] and mean_to_mean and np.isfinite(costs[0]):
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
        mean_predecessor_costs = np.where(
            particle_to_mean & mean_is_walkable[time_index],
            costs[1:],
            np.inf,
        )
        best_particle = int(np.argmin(mean_predecessor_costs))
        if mean_predecessor_costs[best_particle] < next_costs[0]:
            next_costs[0] = mean_predecessor_costs[best_particle]
            backpointers[time_index, 0] = best_particle + 1

        deviations = np.sum(
            np.square(positive_paths[:, time_index, :] - mean_path[time_index]),
            axis=1,
        )
        # 同じ祖先を追い続ける遷移は、PF伝播時に壁判定済みである。
        same_path_costs = np.where(
            cluster_membership[time_index],
            costs[1:] + deviations,
            np.inf,
        )
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
        enter_costs = np.where(
            mean_to_particle & cluster_membership[time_index],
            costs[0] + deviations,
            np.inf,
        )
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


def _unsupported_reversal_count(
    path: np.ndarray,
    sensor_headings: np.ndarray,
    turning_evidence: np.ndarray,
    *,
    window_steps: int = 5,
) -> int:
    """センサー上の旋回根拠がない進行方向反転を数える。"""
    displacements = np.diff(path, axis=0)
    if len(displacements) < 2:
        return 0
    moving_step_indices = np.flatnonzero(np.linalg.norm(displacements, axis=1) > 1e-6)
    if len(moving_step_indices) < 2:
        return 0
    moving_displacements = displacements[moving_step_indices]
    path_headings = np.arctan2(
        moving_displacements[:, 1],
        moving_displacements[:, 0],
    )
    path_deltas = np.abs(
        np.arctan2(
            np.sin(np.diff(path_headings)),
            np.cos(np.diff(path_headings)),
        )
    )
    sensor_deltas = np.abs(
        np.arctan2(
            np.sin(np.diff(sensor_headings)),
            np.cos(np.diff(sensor_headings)),
        )
    )
    unsupported_count = 0
    for moving_index, path_delta in enumerate(path_deltas, start=1):
        if path_delta < np.deg2rad(135.0):
            continue
        current_step = int(moving_step_indices[moving_index])
        window_start = max(0, current_step - window_steps + 1)
        recent_yaw = float(np.nansum(sensor_deltas[window_start:current_step]))
        recent_turning = bool(np.any(turning_evidence[window_start : current_step + 1]))
        if recent_yaw < np.deg2rad(60.0) and not recent_turning:
            unsupported_count += 1
    return unsupported_count


def _select_sequence_map_path(
    particle_paths: np.ndarray,
    cumulative_log_scores: np.ndarray,
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
    sensor_headings: np.ndarray | None = None,
    turning_evidence: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], list[int | None]]:
    """累積事後スコア最大の合法な単一祖先経路を返す。"""
    if particle_paths.ndim != 3 or particle_paths.shape[2:] != (2,):
        raise ValueError("particle_paths は shape=(N, T, 2) を指定してください")
    n_paths, n_times, _ = particle_paths.shape
    scores = np.asarray(cumulative_log_scores, dtype=float)
    if scores.shape != (n_paths,):
        raise ValueError("cumulative_log_scores は粒子数と同じ長さにしてください")
    if sensor_headings is None:
        sensor_headings = np.zeros(max(0, n_times - 1), dtype=float)
    else:
        sensor_headings = np.asarray(sensor_headings, dtype=float)
    if turning_evidence is None:
        turning_evidence = np.zeros(max(0, n_times - 1), dtype=bool)
    else:
        turning_evidence = np.asarray(turning_evidence, dtype=bool)
    expected_steps = max(0, n_times - 1)
    if sensor_headings.shape != (expected_steps,):
        raise ValueError("sensor_headings は経路のステップ数と同じ長さにしてください")
    if turning_evidence.shape != (expected_steps,):
        raise ValueError("turning_evidence は経路のステップ数と同じ長さにしてください")

    penalized_scores = scores.copy()
    for path_index, path in enumerate(particle_paths):
        reversal_count = _unsupported_reversal_count(
            path,
            sensor_headings,
            turning_evidence,
        )
        # 根拠のない反転1回につき事後確率を1/10相当にtemperingする。
        penalized_scores[path_index] += reversal_count * np.log(0.1)

    for selected_index in np.argsort(-penalized_scores, kind="stable"):
        selected_path = particle_paths[int(selected_index)]
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
            continue
        return (
            selected_path.copy(),
            ["sequence_map_ancestry"] * n_times,
            [int(selected_index)] * n_times,
        )
    raise RuntimeError("合法な単一祖先経路を選択できません")
