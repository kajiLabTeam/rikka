"""累積事後スコアから単一祖先経路を選ぶ。

役割:
    センサー上の旋回根拠がない反転を抑え、合法な単一祖先経路を選択する。
依存元:
    地図遷移判定と NumPy の配列演算を利用する。
利用先:
    particle 結果確定段階が sequence 選択モードで使用する。
処理フロー:
    各経路の急な反転と区間内に累積した未支持反転を数えてスコアを補正し、
    降順に合法性を検証する。
"""

import numpy as np

from .map_constraints import _evaluate_particle_transitions


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
    gradual_reversal_active = False
    for moving_index, path_delta in enumerate(path_deltas, start=1):
        current_step = int(moving_step_indices[moving_index])
        window_start = max(0, current_step - window_steps)
        recent_yaw = float(np.nansum(sensor_deltas[window_start:current_step]))
        recent_turning = bool(np.any(turning_evidence[window_start : current_step + 1]))
        supported = recent_yaw >= np.deg2rad(60.0) or recent_turning
        if path_delta >= np.deg2rad(135.0) and not supported:
            unsupported_count += 1
            gradual_reversal_active = True
            continue

        # 1歩ずつの曲がりが小さくても、同じ期間のセンサー変化で説明できない
        # 折り返しは区間単位で検出する。停止歩は方位の基準点に使わない。
        first_moving = int(np.searchsorted(moving_step_indices, window_start))
        interval_delta = float(
            np.arctan2(
                np.sin(path_headings[moving_index] - path_headings[first_moving]),
                np.cos(path_headings[moving_index] - path_headings[first_moving]),
            )
        )
        gradual_reversal = abs(interval_delta) >= np.deg2rad(135.0) and not supported
        if gradual_reversal and not gradual_reversal_active:
            unsupported_count += 1
        gradual_reversal_active = gradual_reversal
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
    allowed_jump_steps: set[int] | None = None,
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
        if n_times > 1:
            valid_transitions = _evaluate_particle_transitions(
                selected_path[:-1],
                selected_path[1:],
                map_gray,
                gx_mean,
                gz_mean,
                origin_px,
                scale,
            )
            if allowed_jump_steps:
                jump_indices = np.asarray(
                    [step - 1 for step in allowed_jump_steps], dtype=int
                )
                jump_indices = jump_indices[
                    (jump_indices >= 0) & (jump_indices < len(valid_transitions))
                ]
                valid_transitions[jump_indices] = True
            if not valid_transitions.all():
                continue
        return (
            selected_path.copy(),
            ["sequence_map_ancestry"] * n_times,
            [int(selected_index)] * n_times,
        )
    raise RuntimeError("合法な単一祖先経路を選択できません")
