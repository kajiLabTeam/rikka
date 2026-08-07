"""粒子フィルタの経路枝を保つ再標本化処理。

役割:
    粒子ごとの経路枝IDと重みを集計し、少数だが有効な枝へ最低粒子数を
    割り当てながら、枝内で系統リサンプリングを行う。
依存元:
    NumPyから配列演算と乱数生成器を取得する。particle filter本体の状態や
    設定値には依存せず、呼び出し側から重み、枝ID、乱数生成器を受け取る。
利用先:
    ``particle_filter`` がESS低下時の再標本化で利用し、経路分岐の少数枝が
    全体リサンプリングだけで消失することを防ぐために使用する。
処理フロー:
    正の重みを持つ枝を集計し、質量上位の枝へ最低quotaと質量比例分を配分し、
    各枝の条件付き重みに対して系統リサンプリングを行って親indexを返す。
"""

from typing import NamedTuple

import numpy as np


class BranchResamplingDiagnostics(NamedTuple):
    """枝保持リサンプリングの集計値。"""

    active_branch_count: int
    branch_entropy: float
    dominant_branch_probability: float
    min_protected_branch_count: int
    pruned_branch_count: int


def _systematic_resample_count(
    weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """正規化済み重みから指定個数の親indexを系統抽出する。"""
    if count == 0:
        return np.empty(0, dtype=int)
    positions = (np.arange(count, dtype=float) + rng.uniform(0.0, 1.0)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="left")


def _allocate_counts(
    branch_masses: np.ndarray,
    total_count: int,
    minimum_count: int,
) -> np.ndarray:
    """各枝へ最低quotaと質量比例分を割り当てる。"""
    branch_count = len(branch_masses)
    effective_minimum = min(minimum_count, total_count // branch_count)
    counts = np.full(branch_count, effective_minimum, dtype=int)
    remaining = total_count - int(counts.sum())
    if remaining == 0:
        return counts

    normalized_masses = branch_masses / branch_masses.sum()
    exact_additions = remaining * normalized_masses
    additions = np.floor(exact_additions).astype(int)
    counts += additions
    leftover = total_count - int(counts.sum())
    if leftover > 0:
        remainders = exact_additions - additions
        order = np.argsort(-remainders, kind="stable")
        counts[order[:leftover]] += 1
    return counts


def branch_preserving_resample(
    weights: np.ndarray,
    branch_ids: np.ndarray,
    rng: np.random.Generator,
    *,
    max_branches: int = 4,
    min_protected_count: int | None = None,
    output_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, BranchResamplingDiagnostics]:
    """有効な上位枝を最低quota付きで枝内再標本化する。

    正の重みを持つ枝だけを有効枝として扱う。有効枝が1つの場合は通常の
    systematic resamplingと同じ抽出を行う。枝が複数ある場合は質量上位の
    ``max_branches`` 枝を残し、各枝へ最低quotaを割り当てた後、残りを枝質量に
    比例して配分する。返す親index数は常に入力粒子数と一致する。
    """
    weights_array = np.asarray(weights, dtype=float)
    branch_ids_array = np.asarray(branch_ids)
    if weights_array.ndim != 1 or branch_ids_array.ndim != 1:
        raise ValueError("weights と branch_ids は1次元配列を指定してください")
    if len(weights_array) == 0:
        raise ValueError("weights は空でない配列を指定してください")
    if len(weights_array) != len(branch_ids_array):
        raise ValueError("weights と branch_ids の長さが一致しません")
    if not np.isfinite(weights_array).all() or np.any(weights_array < 0.0):
        raise ValueError("weights は有限な非負値を指定してください")
    total_weight = float(weights_array.sum())
    if total_weight <= 0.0:
        raise ValueError("weights の合計は正の値を指定してください")
    if not isinstance(max_branches, int) or max_branches <= 0:
        raise ValueError("max_branches は正の整数を指定してください")
    if min_protected_count is not None and (
        not isinstance(min_protected_count, int) or min_protected_count < 0
    ):
        raise ValueError("min_protected_count は0以上の整数を指定してください")

    normalized_weights = weights_array / total_weight
    unique_ids, first_indices, inverse = np.unique(
        branch_ids_array,
        return_index=True,
        return_inverse=True,
    )
    masses = np.bincount(
        inverse,
        weights=normalized_weights,
        minlength=len(unique_ids),
    )
    active_indices = np.flatnonzero(masses > 0.0)
    active_masses = masses[active_indices]
    active_probabilities = active_masses / active_masses.sum()
    positive_probabilities = active_probabilities[active_probabilities > 0.0]
    entropy = float(-np.sum(positive_probabilities * np.log(positive_probabilities)))

    ranked_active_indices = sorted(
        active_indices.tolist(),
        key=lambda index: (-masses[index], int(first_indices[index])),
    )
    retained_indices = np.asarray(ranked_active_indices[:max_branches], dtype=int)
    retained_masses = masses[retained_indices]
    retained_masses /= retained_masses.sum()
    particle_count = len(weights_array) if output_count is None else output_count
    if not isinstance(particle_count, int) or particle_count <= 0:
        raise ValueError("output_count は正の整数を指定してください")
    requested_minimum = (
        int(np.ceil(np.sqrt(particle_count)))
        if min_protected_count is None
        else min_protected_count
    )
    effective_minimum = min(
        requested_minimum,
        particle_count // len(retained_indices),
    )

    if len(active_indices) == 1:
        parent_indices = _systematic_resample_count(
            normalized_weights,
            particle_count,
            rng,
        )
    else:
        allocated_counts = _allocate_counts(
            retained_masses,
            particle_count,
            effective_minimum,
        )
        selected_parents: list[np.ndarray] = []
        for branch_index, count in zip(
            retained_indices,
            allocated_counts,
            strict=True,
        ):
            member_indices = np.flatnonzero(inverse == branch_index)
            conditional_weights = normalized_weights[member_indices]
            conditional_weights /= conditional_weights.sum()
            local_parents = _systematic_resample_count(
                conditional_weights,
                int(count),
                rng,
            )
            selected_parents.append(member_indices[local_parents])
        parent_indices = np.concatenate(selected_parents)

    resampled_branch_ids = branch_ids_array[parent_indices].copy()
    diagnostics = BranchResamplingDiagnostics(
        active_branch_count=len(active_indices),
        branch_entropy=entropy,
        dominant_branch_probability=float(np.max(active_probabilities)),
        min_protected_branch_count=(
            effective_minimum if len(retained_indices) > 1 else 0
        ),
        pruned_branch_count=len(active_indices) - len(retained_indices),
    )
    return parent_indices, resampled_branch_ids, diagnostics
