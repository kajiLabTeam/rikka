"""前向きPFの粒子履歴から独立した過去経路候補を構成する。

役割:
    現行の再配置と死亡枠補充を維持し、実行後の粒子位置から地図に整合する候補を選ぶ。
依存元:
    commonの共有結果型・地図変換、recorderの段階履歴、NumPy、画像読込を利用する。
利用先:
    particle.pipelineが明示指定時だけ実行し、plot.pipelineへ比較用結果を渡す。
処理フロー:
    再配置イベントを抽出、各時点の近傍粒子を候補化し、地図制約付き動的計画法で
    変位の二乗と位置・重みの支持から経路を選ぶ。元軌跡・粒子・乱数は変更しない。
"""

from collections.abc import Sequence

import numpy as np
from matplotlib.image import imread

from ...common.lib.models import (
    FloorMap,
    ParticleHistoryCandidate,
    ParticleHistoryStep,
    ParticleRelocationEvent,
)
from .map_constraints import _evaluate_particle_transitions, _normalize_floormap_gray
from .recorder import ParticleStepStages


def _relocations(
    stages: Sequence[ParticleStepStages],
) -> tuple[ParticleRelocationEvent, ...]:
    """通常提案と再配置後の位置を比較し、復旧・複製とは区別する。"""
    events = []
    for stage in stages:
        if stage.recovery_mode != "none":
            continue
        proposed = (
            stage.before_positions
            + np.column_stack(
                (np.cos(stage.proposed_headings), np.sin(stage.proposed_headings))
            )
            * stage.proposed_step_lengths[:, None]
        )
        changes = np.linalg.norm(stage.proposed_positions - proposed, axis=1)
        counts = np.bincount(stage.parent_indices, minlength=len(proposed))
        for index in np.flatnonzero((changes > 1e-8) & stage.valid_transition):
            previous = stage.before_positions[index]
            relocated = stage.proposed_positions[index]
            events.append(
                ParticleRelocationEvent(
                    step=stage.step,
                    particle=int(index),
                    previous_x_m=float(previous[0]),
                    previous_y_m=float(previous[1]),
                    proposed_x_m=float(proposed[index, 0]),
                    proposed_y_m=float(proposed[index, 1]),
                    relocated_x_m=float(relocated[0]),
                    relocated_y_m=float(relocated[1]),
                    replacement_m=float(changes[index]),
                    realized_displacement_m=float(np.linalg.norm(relocated - previous)),
                    descendants=int(counts[index]),
                )
            )
    return tuple(events)


def _candidate_indices(
    stage: ParticleStepStages, reference: np.ndarray, radius: float, count: int
) -> np.ndarray:
    """重みと空間的な広がりを使って、実在する近傍の生存粒子を間引く。"""
    positions, weights = stage.after_positions, stage.after_weights
    eligible = np.flatnonzero(
        (weights > 0) & (np.linalg.norm(positions - reference, axis=1) <= radius)
    )
    selected: list[int] = []
    distances = np.full(len(eligible), np.inf)
    while len(selected) < count and len(eligible):
        score = weights[eligible] if not selected else distances
        local = int(np.argmax(score))
        if selected and distances[local] < 1e-12:
            break
        index = int(eligible[local])
        selected.append(index)
        distances = np.minimum(
            distances, np.sum((positions[eligible] - positions[index]) ** 2, axis=1)
        )
    return np.asarray(selected, dtype=int)


def build_history_candidate(
    trajectory: list[list[float]],
    stages: Sequence[ParticleStepStages],
    floormap: FloorMap,
    gx_mean: float,
    gz_mean: float,
    *,
    search_radius_m: float = 5.0,
    max_candidates: int = 24,
    landmark_constraints: bool = False,
) -> ParticleHistoryCandidate:
    """始終点を仮境界として保持し、粒子支持のある独立した経路候補を返す。

    センサーの歩幅・方位による候補の制限は行わない。移動量の二乗による
    正則化は滑らかさの仮説であり、正解軌跡や歩行速度の保証ではない。
    """
    original = np.asarray(trajectory, dtype=float)
    if (
        original.ndim != 2
        or original.shape[1] != 2
        or not len(original)
        or not np.isfinite(original).all()
    ):
        raise ValueError("trajectoryには有限な座標を1点以上指定してください")
    if len(stages) != len(original) - 1:
        raise ValueError("歩数と粒子履歴が一致しません")
    if not np.isfinite(search_radius_m) or search_radius_m <= 0 or max_candidates < 1:
        raise ValueError("探索半径と候補数は正の値にしてください")
    for i, stage in enumerate(stages, start=1):
        if stage.step <= (
            stages[i - 2].step if i > 1 else 0
        ) or stage.after_positions.shape != (
            len(stage.after_weights),
            2,
        ):
            raise ValueError("粒子履歴の歩番号またはshapeが不正です")
        if (
            not np.isfinite(stage.after_positions).all()
            or not np.isfinite(stage.after_weights).all()
            or np.any(stage.after_weights < 0)
            or stage.after_weights.sum() <= 0
        ):
            raise ValueError("粒子履歴には有限な位置と非負で合計が正の重みが必要です")
    events = _relocations(stages)
    gray = _normalize_floormap_gray(imread(floormap.path))

    def valid(previous: np.ndarray, following: np.ndarray) -> bool:
        return bool(
            _evaluate_particle_transitions(
                previous[None, :],
                following[None, :],
                gray,
                gx_mean,
                gz_mean,
                floormap.origin_px,
                floormap.scale,
            )[0]
        )

    nodes = [original[:1].copy()]
    sources = [np.array([-1], dtype=int)]
    unary = [np.zeros(1)]
    for i, stage in enumerate(stages, start=1):
        indices = (
            _candidate_indices(stage, original[i], search_radius_m, max_candidates)
            if i < len(stages)
            else np.empty(0, dtype=int)
        )
        positions = np.vstack((original[i], stage.after_positions[indices]))
        costs = np.zeros(len(positions))
        if len(indices):
            # 各時点の重みを使う。最終重みを過去全体へ適用しない。
            relative = stage.after_weights[indices] / stage.after_weights.max()
            costs[1:] = np.sum(
                (positions[1:] - original[i]) ** 2, axis=1
            ) / search_radius_m**2 - 0.1 * np.log(relative)
        nodes.append(positions)
        sources.append(np.r_[-1, indices])
        unary.append(costs)

    status = "candidate"
    if landmark_constraints or any(
        "landmark" in stage.recovery_mode for stage in stages
    ):
        status = "skipped_landmark"
    costs = np.array([0.0 if valid(original[0], original[0]) else np.inf])
    pointers = [np.array([-1], dtype=int)]
    if status == "candidate":
        for i in range(1, len(nodes)):
            current_costs = np.full(len(nodes[i]), np.inf)
            parents = np.full(len(nodes[i]), -1, dtype=int)
            for j, point in enumerate(nodes[i]):
                proposals = costs + np.sum((nodes[i - 1] - point) ** 2, axis=1)
                for parent in np.argsort(proposals, kind="stable"):
                    if not np.isfinite(proposals[parent]):
                        break
                    if valid(nodes[i - 1][parent], point):
                        current_costs[j] = proposals[parent] + unary[i][j]
                        parents[j] = parent
                        break
            costs = current_costs
            pointers.append(parents)
        if not np.isfinite(costs[0]):
            status = "unavailable_map_path"
    chosen = np.zeros(len(original), dtype=int)
    objective = float(costs[0]) if status == "candidate" else None
    if status == "candidate":
        for i in range(len(original) - 1, 0, -1):
            chosen[i - 1] = pointers[i][chosen[i]]
    candidate = np.array([nodes[i][j] for i, j in enumerate(chosen)])
    corrections = np.linalg.norm(candidate - original, axis=1)
    if status == "candidate" and not np.any(corrections > 1e-8):
        status = "unchanged"
    original_valid = all(valid(p, p) for p in original) and all(
        valid(a, b) for a, b in zip(original[:-1], original[1:], strict=True)
    )
    baseline = float(np.sum(np.diff(original, axis=0) ** 2)) if original_valid else None
    records = []
    for i, point in enumerate(candidate):
        source = int(sources[i][chosen[i]])
        record_stage = stages[i - 1] if i else None
        connection = "forward_reference"
        if i and not np.allclose(candidate[i - 1], original[i - 1], atol=1e-8, rtol=0):
            connection = "hypothesis_switch"
        if source >= 0 and record_stage is not None:
            parent = int(record_stage.parent_indices[source])
            connection = (
                "parent_edge"
                if (i == 1 or int(sources[i - 1][chosen[i - 1]]) == parent)
                and np.allclose(
                    record_stage.before_positions[parent],
                    candidate[i - 1],
                    atol=1e-8,
                    rtol=0,
                )
                else "hypothesis_switch"
            )
            # 再生区間のbefore_positionsは既に親で並べ替え済みなので、
            # 通常歩と同じ親インデックスの照合では系譜を確定できない。
            if "checkpoint" in record_stage.recovery_mode:
                connection = "replay_edge_unknown"
        records.append(
            ParticleHistoryStep(
                step=0 if record_stage is None else record_stage.step,
                timestamp_s=0.0 if record_stage is None else record_stage.timestamp_s,
                original_x_m=float(original[i, 0]),
                original_y_m=float(original[i, 1]),
                candidate_x_m=float(point[0]),
                candidate_y_m=float(point[1]),
                correction_m=float(corrections[i]),
                original_displacement_m=0.0
                if not i
                else float(np.linalg.norm(original[i] - original[i - 1])),
                candidate_displacement_m=0.0
                if not i
                else float(np.linalg.norm(point - candidate[i - 1])),
                source_particle=None if source < 0 else source,
                source_weight=None
                if source < 0 or record_stage is None
                else float(record_stage.after_weights[source]),
                source_offset_rad=None
                if source < 0 or record_stage is None
                else float(record_stage.after_offsets[source]),
                source_stride_scale=None
                if source < 0
                or record_stage is None
                or record_stage.after_stride_scale is None
                else float(record_stage.after_stride_scale[source]),
                source_motion_state=None
                if source < 0 or record_stage is None
                else int(record_stage.after_motion_state[source]),
                connection=connection,
            )
        )
    return ParticleHistoryCandidate(
        trajectory=candidate.tolist(),
        steps=tuple(records),
        relocations=events,
        status=status,
        objective_original=baseline,
        objective_candidate=objective,
        search_radius_m=search_radius_m,
        max_candidates=max_candidates,
    )
