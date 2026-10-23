"""実データのPF軌跡異常を、製品処理を変更せず段階別に診断する。

役割: 円内救済・重み更新・再標本化・代表経路の移動量と系譜を対応付ける。
入力: --datasets の1019Experiment計測、現行CLI既定値、--seeds。
出力: 新規 --output-dir に段階CSV、メタデータ、集計、診断レポート、図。
処理フロー: CLIの保存前に共有歩列を取得し、通常実行と読み取りラッパー付き
    実行を照合する。ラッパーは既存関数を各1回だけ呼び乱数を追加消費しない。
    親対応の変位と最終履歴を集計し、製品とは独立した診断ファイルへ保存する。
"""

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from rikka.cli import commands
from rikka.common import config
from rikka.common.lib.floormap import pixel_y_sign
from rikka.common.lib.sensors import load_sensor_data
from rikka.particle.lib.map_constraints import _evaluate_particle_transitions
from rikka.particle.lib.sequence_path import _unsupported_reversal_count

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-6


def json_default(value):
    """診断値をJSON互換型へ変換する。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    """既存の成果物を上書きせずJSONを保存する。"""
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)


def digest(path):
    """入力やコードの同一性確認用ハッシュを返す。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def angle_difference(left, right):
    """最小絶対方位差を度で返す。"""
    return np.abs(np.degrees(np.arctan2(np.sin(left - right), np.cos(left - right))))


def safe_ratio(distance, nominal):
    """極小歩幅では欠損にする。"""
    return (
        np.asarray(distance) / nominal
        if nominal > EPS
        else np.full_like(np.asarray(distance, dtype=float), np.nan)
    )


def assert_same(left, right):
    """配列を含む既存の診断・経路を完全一致で再帰比較する。"""
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, strict=True)
    elif is_dataclass(left):
        for field in fields(left):
            assert_same(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same(a, b)
    elif isinstance(left, float) and np.isnan(left):
        assert np.isnan(right)
    else:
        assert left == right, (left, right)


class _PreparedOnlyError(Exception):
    """通常成果物を保存する前にCLIを停止する。"""


def prepare(data_dir):
    """既存CLIと同じ準備処理を用い、保存前に引数を取り出す。"""
    acc, gyro = load_sensor_data(data_dir)
    captured = {}

    def capture(prepared, floormap, settings, **kwargs):
        captured.update(prepared=prepared, floormap=floormap, settings=settings)
        raise _PreparedOnlyError

    with patch.object(commands, "run_particle", capture):
        try:
            commands.run(
                df_acc=acc,
                df_gyro=gyro,
                use_particle_filter=True,
                plot=False,
                save_animation=False,
                ble_landmark=False,
            )
        except _PreparedOnlyError:
            pass
    return captured, acc, gyro


def instrumented_run(prepared, floor, settings):
    """同じ関数・同じ引数を1回ずつ実行し、追加乱数なしで状態を複写する。"""
    runner = importlib.import_module("rikka.particle.lib.runner")
    propose_module = importlib.import_module("rikka.particle.lib.propose")
    old_propose = runner.propose
    old_relocate = propose_module.relocate_invalid_particles
    old_resolve = runner.resolve_map_constraints
    old_finalize = runner.finalize
    records, final = [], {}
    active = {}

    def relocate(*args, **kwargs):
        active["predicted"] = args[1].copy()
        active["initial_valid"] = args[2].copy()
        result = old_relocate(*args, **kwargs)
        active["relocated"] = result[0].copy()
        active["relocated_valid"] = result[1].copy()
        active["relocated_count"] = result[2]
        return result

    def propose(ctx):
        active.clear()
        old_propose(ctx)
        if ctx.candidate_bank is not None:
            active.update(
                predicted=ctx.proposed_particles.copy(),
                relocated=ctx.proposed_particles.copy(),
                initial_valid=ctx.valid_transition.copy(),
                relocated_valid=ctx.valid_transition.copy(),
                relocated_count=0,
            )
        snap = dict(active)
        snap["candidate_bank"] = deepcopy(ctx.candidate_bank)
        snap["proposal_method"] = ctx.proposal_method
        snap["candidate_count"] = ctx.candidate_count
        snap["proposal_history_version"] = ctx.proposal_history_version
        for name in (
            "particles_before",
            "weights_before",
            "theta",
            "sl",
            "proposed_stride_scale",
            "proposed_drift",
            "proposed_correction",
            "proposed_motion_state",
            "motion_state_before",
            "posterior_weights",
            "stride_observation_likelihood",
            "state_predictive_likelihoods",
        ):
            snap[name] = getattr(ctx, name).copy()
        snap.update(
            step=ctx.step_number,
            timestamp_s=float(ctx.step_time),
            pdr_step_m=float(ctx.sl_det),
            pdr_heading=float(ctx.particle_heading),
        )
        records.append(snap)

    def resolve(ctx):
        old_resolve(ctx)
        snap = records[-1]
        for name in (
            "particles",
            "weights",
            "parent_indices",
            "heading_drift",
            "heading_correction",
            "stride_scale",
            "motion_state",
        ):
            snap[f"after_{name}"] = getattr(ctx, name).copy()
        snap.update(
            recovery_mode=ctx.recovery_mode,
            resolved_history_version=ctx.history_version,
            recovery_attempted=ctx.recovery_attempted,
            ess_before_observation=ctx.ess_before_observation,
            ess_after_observation=ctx.ess_after_observation,
            resampled=ctx.resampled,
        )

    def finalize(ctx):
        result = old_finalize(ctx)
        final["recorded_stages"] = list(ctx.recorder.stages)
        for name in (
            "position_history",
            "weight_history",
            "parent_history",
            "current_modes",
            "current_sources",
            "sequence_sources",
            "trajectory_modes",
        ):
            final[name] = list(getattr(ctx, name))
        for name in ("current_path", "sequence_path", "selected_path"):
            final[name] = getattr(ctx, name).copy()
        final.update(
            selected_mode=ctx.selected_mode,
            current_reversals=ctx.current_reversals,
            sequence_reversals=ctx.sequence_reversals,
            turning_evidence=ctx.turning_evidence.copy(),
            sensor_headings=ctx.sensor_headings.copy(),
        )
        return result

    with (
        patch.object(runner, "propose", propose),
        patch.object(propose_module, "relocate_invalid_particles", relocate),
        patch.object(runner, "resolve_map_constraints", resolve),
        patch.object(runner, "finalize", finalize),
    ):
        result = commands.run_particle(prepared, floor, settings)
    return result, records, final


def ancestry_count(parents, time_index, lag, count):
    """確定後の親履歴をたどり、指定歩数前の祖先数を求める。"""
    if time_index < lag:
        return np.nan
    lineage = np.arange(count)
    for index in range(time_index - 1, time_index - lag - 1, -1):
        lineage = parents[index][lineage]
    return int(np.unique(lineage).size)


def lineage_indices(parents, terminal):
    """単一祖先経路の各時点の粒子スロットを求める。"""
    lineage = np.empty(len(parents) + 1, dtype=int)
    lineage[-1] = terminal
    for index in range(len(parents) - 1, -1, -1):
        lineage[index] = parents[index][lineage[index + 1]]
    return lineage


def stats(values, prefix):
    """有限な診断サンプルだけで要約統計を返す。"""
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return {
        f"{prefix}_{name}": float(function(finite)) if len(finite) else np.nan
        for name, function in (
            ("mean", np.mean),
            ("median", np.median),
            ("max", np.max),
            ("p05", lambda a: np.quantile(a, 0.05)),
            ("p95", lambda a: np.quantile(a, 0.95)),
        )
    }


def analyze_run(result, records, final, prepared, floor, run_dir, dataset, seed):
    """粒子・歩・代表経路を親対応で集計し、CSVを分割保存する。"""
    gray = commands._load_floormap_gray(floor.path)

    def valid(before, after):
        return _evaluate_particle_transitions(
            before,
            after,
            gray,
            prepared.gx_mean,
            prepared.gz_mean,
            floor.origin_px,
            floor.scale,
        )

    n = len(records[0]["particles_before"])
    paths = {
        "pdr": np.asarray(prepared.trajectory),
        "current": final["current_path"],
        "sequence": final["sequence_path"],
        "final": np.asarray(result.trajectory),
    }
    steps = len(records)
    assert all(len(path) == steps + 1 for path in paths.values())
    trajectory = pd.DataFrame(
        {"step": np.arange(steps + 1), "timestamp_s": [np.nan] + result.t_at_steps}
    )
    for name, path in paths.items():
        trajectory[f"{name}_x"], trajectory[f"{name}_y"] = path.T
        trajectory[f"{name}_step_m"] = np.r_[
            np.nan, np.linalg.norm(np.diff(path, axis=0), axis=1)
        ]
    trajectory["pdr_step_m"] = np.r_[np.nan, result.step_lengths]
    trajectory["current_mode"] = final["current_modes"]
    trajectory["current_source"] = final["current_sources"]
    trajectory["selected_mode"] = final["selected_mode"]
    trajectory["current_mode_switch"] = trajectory.current_mode.ne(
        trajectory.current_mode.shift()
    )
    trajectory.loc[0, "current_mode_switch"] = False
    actual_heading = np.arctan2(*np.diff(paths["final"], axis=0).T[::-1])
    final_diff = angle_difference(actual_heading, final["sensor_headings"])
    final_diff[np.linalg.norm(np.diff(paths["final"], axis=0), axis=1) <= EPS] = np.nan
    trajectory["final_pdr_angle_diff_deg"] = np.r_[np.nan, final_diff]
    sequence_slots = lineage_indices(
        final["parent_history"], int(final["sequence_sources"][-1])
    )
    trajectory["sequence_particle_id"] = sequence_slots
    trajectory["sequence_relocated"] = False
    particle_frames, step_rows = [], []
    for offset, snap in enumerate(records):
        t = offset + 1
        assert snap["step"] == t
        before, relocated = snap["particles_before"], snap["relocated"]
        displacement = relocated - before
        lengths = np.linalg.norm(displacement, axis=1)
        headings = np.arctan2(displacement[:, 1], displacement[:, 0])
        collisions = ~snap["initial_valid"]
        success = collisions & snap["relocated_valid"]
        assert np.count_nonzero(success) == snap["relocated_count"]
        internal_diff = angle_difference(headings, snap["theta"])
        pdr_diff = angle_difference(headings, snap["pdr_heading"])
        internal_diff[lengths <= EPS] = np.nan
        pdr_diff[lengths <= EPS] = np.nan
        ratios = safe_ratio(lengths, snap["pdr_step_m"])
        parents = snap["after_parent_indices"]
        after = snap["after_particles"]
        after_displacement = after - before[parents]
        after_lengths = np.linalg.norm(after_displacement, axis=1)
        after_headings = np.arctan2(after_displacement[:, 1], after_displacement[:, 0])
        regular = snap["recovery_mode"] == "none"
        if regular:
            np.testing.assert_array_equal(after, relocated[parents])
        history_matches = np.array_equal(after, final["position_history"][t])
        seq_slot = sequence_slots[t]
        seq_parent = sequence_slots[t - 1]
        seq_match = history_matches and parents[seq_slot] == seq_parent
        seq_relocated = bool(success[seq_parent]) if seq_match and regular else None
        trajectory.loc[t, "sequence_relocated"] = seq_relocated
        posterior = snap["posterior_weights"]
        normalized = (
            posterior / posterior.sum() if posterior.sum() > 0 else np.full(n, np.nan)
        )
        full = (
            regular and snap["ess_after_observation"] < config.PF_RESAMPLE_ESS_RATIO * n
        )
        partial = int(np.count_nonzero(posterior == 0)) if regular and not full else 0
        frame = pd.DataFrame(
            {
                "dataset": dataset,
                "seed": seed,
                "step": t,
                "timestamp_s": snap["timestamp_s"],
                "particle_id": np.arange(n),
                "parent_particle_id": parents,
                "before_weight": snap["weights_before"],
                "posterior_weight": posterior,
                "normalized_posterior_weight": normalized,
                "after_weight": snap["after_weights"],
                "collision": collisions,
                "proposal_method": snap["proposal_method"],
                "candidate_count": snap["candidate_count"],
                "relocation_attempted": collisions
                & (snap["proposal_method"] == "legacy")
                & (config.PF_INVALID_RELOCATION_RADIUS_M > 0),
                "relocation_success": success,
                "relocation_failed": collisions & ~success,
                "pdr_step_m": snap["pdr_step_m"],
                "pdr_heading_rad": snap["pdr_heading"],
                "internal_heading_rad": snap["theta"],
                "internal_step_m": snap["sl"],
                "internal_stride_scale": snap["proposed_stride_scale"],
                "internal_drift_rad": snap["proposed_drift"],
                "internal_correction_rad": snap["proposed_correction"],
                "realized_distance_m": lengths,
                "realized_heading_rad": headings,
                "distance_ratio": ratios,
                "pdr_angle_diff_deg": pdr_diff,
                "internal_angle_diff_deg": internal_diff,
                "stride_likelihood": snap["stride_observation_likelihood"],
                "state_predictive_likelihood": snap["state_predictive_likelihoods"],
                "motion_state": snap["proposed_motion_state"],
                "after_motion_state": snap["after_motion_state"],
                "after_stride_scale": snap["after_stride_scale"],
                "after_internal_offset_rad": snap["after_heading_drift"]
                + snap["after_heading_correction"],
                "after_parent_distance_m": after_lengths,
                "after_parent_heading_rad": after_headings,
                "after_parent_internal_heading_rad": snap["theta"][parents]
                if regular
                else np.full(n, np.nan),
                "after_parent_internal_angle_diff_deg": angle_difference(
                    after_headings, snap["theta"][parents]
                )
                if regular
                else np.full(n, np.nan),
                "pdr_forward_projection_m": displacement[:, 0]
                * np.cos(snap["pdr_heading"])
                + displacement[:, 1] * np.sin(snap["pdr_heading"]),
                "pdr_lateral_projection_m": -displacement[:, 0]
                * np.sin(snap["pdr_heading"])
                + displacement[:, 1] * np.cos(snap["pdr_heading"]),
                "after_parent_relocated": success[parents]
                if regular
                else np.full(n, False),
                "recovery_mode": snap["recovery_mode"],
                "collective_recovery": not regular,
                "proposal_history_version": snap["proposal_history_version"],
                "resolved_history_version": snap["resolved_history_version"],
                "final_history_version": final["recorded_stages"][
                    offset
                ].history_version,
                "proposal_matches_final_history": final["recorded_stages"][
                    offset
                ].proposal_matches_final_history,
                "online_matches_final_history": history_matches,
            }
        )
        bank = snap["candidate_bank"]
        if bank is not None:
            selected = bank.selected_indices
            selected_g = bank.factors[np.arange(n), np.maximum(selected, 0)]
            frame["candidate_selected_index"] = selected
            frame["candidate_mean_factor"] = bank.mean_factors
            frame["candidate_selected_factor"] = np.where(
                selected >= 0, selected_g, 0.0
            )
            frame["candidate_parent_survived"] = selected >= 0
            frame["candidate_map_pass_fraction"] = bank.valid_transition.mean(axis=1)
        for name, points in (
            ("before", before),
            ("predicted", snap["predicted"]),
            ("relocated", relocated),
            ("after", after),
            ("after_parent_before", before[parents]),
        ):
            frame[f"{name}_x"], frame[f"{name}_y"] = points.T
        particle_frames.append(frame)
        weights = snap["after_weights"]
        final_weights = final["weight_history"][t]
        center_before = np.average(before, axis=0, weights=snap["weights_before"])
        center_prior_relocated = np.average(
            relocated, axis=0, weights=snap["weights_before"]
        )
        center_post = (
            np.average(relocated, axis=0, weights=normalized)
            if np.isfinite(normalized).all()
            else np.full(2, np.nan)
        )
        center_after = np.average(after, axis=0, weights=weights)
        row = dict(
            proposal_method=snap["proposal_method"],
            candidate_count=snap["candidate_count"],
            dataset=dataset,
            seed=seed,
            step=t,
            timestamp_s=snap["timestamp_s"],
            pdr_step_m=snap["pdr_step_m"],
            collision_count=int(collisions.sum()),
            collision_ratio=float(collisions.mean()),
            relocation_count=int(success.sum()),
            relocation_ratio=float(success.mean()),
            relocation_failed_count=int((collisions & ~success).sum()),
            ratio_missing_count=int(np.count_nonzero(~np.isfinite(ratios))),
            valid_after_relocation_count=int(snap["relocated_valid"].sum()),
            surviving_weight_count=int(np.count_nonzero(posterior > 0)),
            full_resampling=full,
            partial_replacement_count=partial,
            ess_before_observation=snap["ess_before_observation"],
            ess_after_observation=snap["ess_after_observation"],
            ess_after_resampling=float(1 / np.sum(weights**2)),
            unique_parent_count=int(np.unique(parents).size),
            unique_position_count=int(np.unique(np.round(after, 9), axis=0).shape[0]),
            duplicate_position_count=n
            - int(np.unique(np.round(after, 9), axis=0).shape[0]),
            position_spread_rms_m=float(
                np.sqrt(np.sum(weights * np.sum((after - center_after) ** 2, axis=1)))
            ),
            position_variance_trace_m2=float(
                np.sum(weights * np.sum((after - center_after) ** 2, axis=1))
            ),
            ancestors_5=ancestry_count(final["parent_history"], t, 5, n),
            ancestors_10=ancestry_count(final["parent_history"], t, 10, n),
            recovery_mode=snap["recovery_mode"],
            online_matches_final_history=history_matches,
            particle_parent_distance_weighted_m=float(np.sum(weights * after_lengths)),
            final_history_particle_distance_weighted_m=float(
                np.sum(
                    final_weights
                    * np.linalg.norm(
                        final["position_history"][t]
                        - final["position_history"][t - 1][
                            final["parent_history"][t - 1]
                        ],
                        axis=1,
                    )
                )
            ),
            prior_weighted_mean_move_m=float(
                np.linalg.norm(center_prior_relocated - center_before)
            ),
            prior_weighted_particle_distance_m=float(
                np.sum(snap["weights_before"] * lengths)
            ),
            posterior_weighted_mean_move_m=float(
                np.linalg.norm(center_post - center_before)
            ),
            resampling_mean_shift_m=float(np.linalg.norm(center_after - center_post)),
            final_step_m=trajectory.loc[t, "final_step_m"],
            current_step_m=trajectory.loc[t, "current_step_m"],
            sequence_step_m=trajectory.loc[t, "sequence_step_m"],
            final_x=paths["final"][t, 0],
            final_y=paths["final"][t, 1],
            current_mode=final["current_modes"][t],
            sequence_relocated=seq_relocated,
            final_pdr_angle_diff_deg=final_diff[offset],
        )
        for label, center in (
            ("before_mean", center_before),
            ("relocated_prior_mean", center_prior_relocated),
            ("posterior_mean", center_post),
            ("resampled_mean", center_after),
        ):
            row[f"{label}_x"], row[f"{label}_y"] = center
        if bank is not None:
            bank_displacements = bank.positions - before[:, None, :]
            bank_distances = np.linalg.norm(bank_displacements, axis=2)
            bank_angles = np.arctan2(
                bank_displacements[:, :, 1], bank_displacements[:, :, 0]
            )
            bank_angle_diff = angle_difference(bank_angles, snap["pdr_heading"])
            bank_angle_diff[bank_distances <= EPS] = np.nan
            row.update(stats(bank_distances.ravel(), "candidate_all_distance_m"))
            row.update(
                stats(
                    safe_ratio(bank_distances, snap["pdr_step_m"]).ravel(),
                    "candidate_all_distance_ratio",
                )
            )
            row.update(stats(bank_angle_diff.ravel(), "candidate_all_pdr_angle_deg"))
            row["candidate_map_pass_fraction"] = float(bank.valid_transition.mean())
            row["candidate_positive_factor_fraction"] = float((bank.factors > 0).mean())
            row["candidate_parent_survival_fraction"] = float(
                (bank.selected_indices >= 0).mean()
            )
            row["candidate_mean_factor_mean"] = float(bank.mean_factors.mean())
        row["relocated_forward_projection_mean_m"] = (
            float(frame.loc[success, "pdr_forward_projection_m"].mean())
            if success.any()
            else np.nan
        )
        row["unrelocated_forward_projection_mean_m"] = (
            float(frame.loc[~collisions, "pdr_forward_projection_m"].mean())
            if (~collisions).any()
            else np.nan
        )
        for label, mask in (
            ("all_particle_distance_m", np.ones(n, bool)),
            ("relocated_distance", success),
        ):
            row.update(stats(lengths[mask], label))
        for name in ("mean", "median", "max", "p05", "p95"):
            row[f"relocated_distance_{name}_m"] = row.pop(f"relocated_distance_{name}")
        row.update(
            stats(
                (lengths - snap["pdr_step_m"])[success],
                "relocated_pdr_distance_error_m",
            )
        )
        row.update(stats(pdr_diff[success], "relocated_pdr_angle_deg"))
        row.update(stats(internal_diff[success], "relocated_internal_angle_deg"))
        for label, mask in (
            ("lt_0_5", ratios < 0.5),
            ("0_5_to_1_5", (ratios >= 0.5) & (ratios <= 1.5)),
            ("1_5_to_3", (ratios > 1.5) & (ratios <= 3)),
            ("gt_3", ratios > 3),
        ):
            row[f"relocated_ratio_{label}_count"] = int(
                np.count_nonzero(mask & success)
            )
        row["relocated_pdr_gt90_count"] = int(
            np.count_nonzero(success & (pdr_diff > 90))
        )
        row["relocated_pdr_gt135_count"] = int(
            np.count_nonzero(success & (pdr_diff > 135))
        )
        row["relocated_internal_gt90_count"] = int(
            np.count_nonzero(success & (internal_diff > 90))
        )
        row["relocated_internal_gt135_count"] = int(
            np.count_nonzero(success & (internal_diff > 135))
        )
        # 円内救済が同じ重みのまま与える平均変位への寄与を分離する。
        delta = relocated - snap["predicted"]
        row["relocation_prior_mean_shift_m"] = float(
            np.linalg.norm(np.sum(snap["weights_before"][:, None] * delta, axis=0))
        )
        row["live_parent_wall_crossings"] = int(
            np.count_nonzero(~valid(before[parents], after) & (weights > 0))
        )
        step_rows.append(row)
    particles = pd.concat(particle_frames, ignore_index=True)
    diagnostics = pd.DataFrame(step_rows)
    trajectory.insert(0, "seed", seed)
    trajectory.insert(0, "dataset", dataset)
    events = []
    for name in ("current", "sequence", "final"):
        distances = trajectory[f"{name}_step_m"].to_numpy()[1:]
        vectors = np.diff(paths[name], axis=0)
        directions = np.arctan2(vectors[:, 1], vectors[:, 0])
        diff = angle_difference(directions, final["sensor_headings"])
        for k in range(steps):
            candidates = []
            if distances[k] > 2.0:
                candidates.append(("jump_gt_2m", distances[k]))
            if distances[k] > 1.5 * result.step_lengths[k]:
                candidates.append(
                    (
                        "distance_gt_1_5pdr",
                        distances[k] / max(result.step_lengths[k], EPS),
                    )
                )
            if distances[k] < 0.5 * result.step_lengths[k]:
                candidates.append(
                    (
                        "short_lt_0_5pdr",
                        1 - distances[k] / max(result.step_lengths[k], EPS),
                    )
                )
            if distances[k] > EPS and diff[k] > 90:
                candidates.append(("opposed_pdr_gt90", diff[k]))
            if distances[k] > EPS and diff[k] > 135:
                candidates.append(("opposed_pdr_gt135", diff[k]))
            if (
                k > 0
                and distances[k] > EPS
                and distances[k - 1] > EPS
                and angle_difference(directions[k], directions[k - 1]) > 135
            ):
                candidates.append(
                    (
                        "path_turn_gt135",
                        float(angle_difference(directions[k], directions[k - 1])),
                    )
                )
            for event, score in candidates:
                events.append(
                    dict(
                        dataset=dataset,
                        seed=seed,
                        step=k + 1,
                        timestamp_s=result.t_at_steps[k],
                        path=name,
                        event=event,
                        score=float(score),
                        distance_m=distances[k],
                        pdr_step_m=result.step_lengths[k],
                        pdr_angle_diff_deg=diff[k],
                        relocation_ratio=diagnostics.iloc[k].relocation_ratio,
                        current_mode=final["current_modes"][k + 1],
                        selected_mode=final["selected_mode"],
                    )
                )
    events = pd.DataFrame(events)
    particles.to_csv(
        run_dir / "particle_stages.csv.gz", index=False, compression="gzip"
    )
    diagnostics.to_csv(run_dir / "step_diagnostics.csv", index=False)
    trajectory.to_csv(run_dir / "trajectory_comparison.csv", index=False)
    events.to_csv(run_dir / "anomaly_events.csv", index=False)
    pd.DataFrame(result.trajectory, columns=["x", "y"]).assign(
        step=np.arange(steps + 1), timestamp_s=[np.nan] + result.t_at_steps
    ).to_csv(run_dir / "trajectory.csv", index=False)
    pd.DataFrame([asdict(d) for d in result.particle.diagnostics]).to_csv(
        run_dir / "existing_particle_diagnostics.csv", index=False
    )
    relocated_particles = particles[particles.relocation_success]
    ratios = relocated_particles.distance_ratio.dropna()
    summary = dict(
        proposal_method=records[0]["proposal_method"],
        candidate_count=records[0]["candidate_count"],
        dataset=dataset,
        seed=seed,
        steps=steps,
        particles=n,
        pdr_length_m=float(sum(result.step_lengths)),
        final_length_ratio=float(
            trajectory.final_step_m.sum() / sum(result.step_lengths)
        ),
        parent_distance_weighted_sum_m=float(
            diagnostics.particle_parent_distance_weighted_m.sum()
        ),
        final_history_parent_distance_sum_m=float(
            diagnostics.final_history_particle_distance_weighted_m.sum()
        ),
        relocated_particle_instances=len(relocated_particles),
        collision_particle_instances=int(particles.collision.sum()),
        relocation_particle_fraction=float(particles.relocation_success.mean()),
        relocation_step_fraction=float((diagnostics.relocation_count > 0).mean()),
        relocation_failed=int(particles.relocation_failed.sum()),
        ratio_missing_count=int(relocated_particles.distance_ratio.isna().sum()),
        relocated_gt3_ratio=float((ratios > 3).mean()),
        relocated_pdr_gt90_fraction=float(
            (relocated_particles.pdr_angle_diff_deg > 90).mean()
        ),
        relocated_pdr_gt135_fraction=float(
            (relocated_particles.pdr_angle_diff_deg > 135).mean()
        ),
        relocated_internal_gt90_fraction=float(
            (relocated_particles.internal_angle_diff_deg > 90).mean()
        ),
        relocated_internal_gt135_fraction=float(
            (relocated_particles.internal_angle_diff_deg > 135).mean()
        ),
        full_resampling_steps=int(diagnostics.full_resampling.sum()),
        partial_replacement_total=int(diagnostics.partial_replacement_count.sum()),
        collective_recovery_steps=int((diagnostics.recovery_mode != "none").sum()),
        failed_hold_steps=int((diagnostics.recovery_mode == "failed_hold").sum()),
        ess_after_min=float(diagnostics.ess_after_resampling.min()),
        ess_after_median=float(diagnostics.ess_after_resampling.median()),
        ancestors5_min=float(diagnostics.ancestors_5.min()),
        ancestors10_min=float(diagnostics.ancestors_10.min()),
        selected_mode=final["selected_mode"],
        current_reversals=final["current_reversals"],
        sequence_reversals=final["sequence_reversals"],
        sequence_relocated_steps=int(trajectory.sequence_relocated.fillna(False).sum()),
        current_mode_switches=int(trajectory.current_mode_switch.sum()),
        mean_resampling_shift_m=float(diagnostics.resampling_mean_shift_m.mean()),
        max_resampling_shift_m=float(diagnostics.resampling_mean_shift_m.max()),
        live_parent_wall_crossings=int(diagnostics.live_parent_wall_crossings.sum()),
        rewritten_online_steps=int((~diagnostics.online_matches_final_history).sum()),
    )
    for name, path in paths.items():
        distance = np.linalg.norm(np.diff(path, axis=0), axis=1)
        summary.update(
            {
                f"{name}_length_m": float(distance.sum()),
                f"{name}_max_step_m": float(distance.max()),
                f"{name}_jump_gt2m_count": int((distance > 2).sum()),
                f"{name}_wall_crossings": int(
                    np.count_nonzero(~valid(path[:-1], path[1:]))
                ),
                f"{name}_unsupported_reversals": _unsupported_reversal_count(
                    path, final["sensor_headings"], final["turning_evidence"]
                ),
            }
        )
    for label, mask in (
        ("lt0_5", ratios < 0.5),
        ("0_5to1_5", (ratios >= 0.5) & (ratios <= 1.5)),
        ("1_5to3", (ratios > 1.5) & (ratios <= 3)),
        ("gt3", ratios > 3),
    ):
        summary[f"relocated_ratio_{label}_fraction"] = float(mask.mean())
    summary.update(
        stats(relocated_particles.realized_distance_m, "relocated_distance_m")
    )
    summary.update(
        stats(relocated_particles.pdr_angle_diff_deg, "relocated_pdr_angle_deg")
    )
    # 同じ粒子集合・同じ親重みでの相殺率。再重み付けやクラスタ変更を混ぜない。
    diagnostics["directional_cancellation_ratio"] = (
        diagnostics.prior_weighted_mean_move_m
        / diagnostics.prior_weighted_particle_distance_m
    )
    summary["directional_cancellation_ratio_median"] = float(
        diagnostics.directional_cancellation_ratio.median()
    )
    summary["prior_weighted_particle_distance_sum_m"] = float(
        diagnostics.prior_weighted_particle_distance_m.sum()
    )
    summary["prior_weighted_mean_move_sum_m"] = float(
        diagnostics.prior_weighted_mean_move_m.sum()
    )
    summary["relocated_forward_projection_mean_m"] = float(
        relocated_particles.pdr_forward_projection_m.mean()
    )
    summary["unrelocated_forward_projection_mean_m"] = float(
        particles.loc[~particles.collision, "pdr_forward_projection_m"].mean()
    )
    diagnostics.to_csv(run_dir / "step_diagnostics.csv", index=False)
    intervals = []
    scores = trajectory.final_step_m / trajectory.pdr_step_m
    centers = {
        int(trajectory.final_step_m.idxmax()),
        int(scores.iloc[1:].idxmin()),
        int(trajectory.final_pdr_angle_diff_deg.idxmax()),
    }
    for center in sorted(centers):
        start, end = max(1, center - 2), min(steps, center + 2)
        sub = diagnostics[diagnostics.step.between(start, end)]
        intervals.append(
            dict(
                dataset=dataset,
                seed=seed,
                center_step=center,
                start_step=start,
                end_step=end,
                **{
                    f"{name}_length_m": float(
                        trajectory.loc[start:end, f"{name}_step_m"].sum()
                    )
                    for name in paths
                },
                parent_weighted_distance_m=float(
                    sub.particle_parent_distance_weighted_m.sum()
                ),
                relocation_count=int(sub.relocation_count.sum()),
            )
        )
    pd.DataFrame(intervals).to_csv(run_dir / "interval_comparison.csv", index=False)
    write_json(run_dir / "summary.json", summary)
    return summary, diagnostics, trajectory, events, intervals


def augment_saved_outputs(output):
    """保存CSVだけを読み、通常PDR方位との比較・段階差分・独立検算を追加する。

    PFを再実行せず、今回新規作成した診断成果物だけに派生列を追記する。
    """
    metadata_path = output / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    manifest = pd.read_csv(output / "particle_stages.csv")
    checks, summaries, trajectories, effects, switches = [], [], [], [], []
    for entry in manifest.itertuples(index=False):
        particle_path = output / entry.path
        run_dir = particle_path.parent
        particles = pd.read_csv(particle_path)
        steps = pd.read_csv(run_dir / "step_diagnostics.csv")
        trajectory = pd.read_csv(run_dir / "trajectory_comparison.csv")
        normal_vectors = np.diff(trajectory[["pdr_x", "pdr_y"]].to_numpy(), axis=0)
        normal_headings = np.arctan2(normal_vectors[:, 1], normal_vectors[:, 0])
        normal_headings[np.linalg.norm(normal_vectors, axis=1) <= EPS] = np.nan
        particles["normal_pdr_heading_rad"] = particles.step.map(
            dict(enumerate(normal_headings, 1))
        )
        particles["normal_pdr_angle_diff_deg"] = angle_difference(
            particles.realized_heading_rad, particles.normal_pdr_heading_rad
        )
        particles.loc[
            particles.realized_distance_m <= EPS, "normal_pdr_angle_diff_deg"
        ] = np.nan
        particles["after_parent_normal_pdr_angle_diff_deg"] = angle_difference(
            particles.after_parent_heading_rad, particles.normal_pdr_heading_rad
        )
        particles["after_internal_heading_rad"] = np.nan
        for _, group in particles.groupby("step", sort=False):
            group = group.sort_values("particle_id")
            if (group.recovery_mode == "none").all():
                parents = group.parent_particle_id.to_numpy(dtype=int)
                base_heading = (
                    group.internal_heading_rad.to_numpy()
                    - group.internal_correction_rad.to_numpy()
                    - group.internal_drift_rad.to_numpy()
                )
                heading = (
                    base_heading[parents] + group.after_internal_offset_rad.to_numpy()
                )
                particles.loc[group.index, "after_internal_heading_rad"] = np.arctan2(
                    np.sin(heading), np.cos(heading)
                )
        particles.loc[
            particles.after_parent_distance_m <= EPS,
            [
                "after_parent_normal_pdr_angle_diff_deg",
                "after_parent_internal_angle_diff_deg",
            ],
        ] = np.nan
        trajectory["normal_pdr_heading_rad"] = np.r_[np.nan, normal_headings]
        final_vectors = np.diff(trajectory[["final_x", "final_y"]].to_numpy(), axis=0)
        final_angles = np.arctan2(final_vectors[:, 1], final_vectors[:, 0])
        normal_final_diff = angle_difference(final_angles, normal_headings)
        normal_final_diff[np.linalg.norm(final_vectors, axis=1) <= EPS] = np.nan
        trajectory["final_normal_pdr_angle_diff_deg"] = np.r_[np.nan, normal_final_diff]
        for index in range(1, len(trajectory)):
            row = trajectory.iloc[index]
            previous = trajectory.iloc[index - 1]
            if row.current_mode_switch or row.current_step_m > 2:
                switches.append(
                    dict(
                        dataset=entry.dataset,
                        seed=entry.seed,
                        step=int(row.step),
                        timestamp_s=row.timestamp_s,
                        previous_mode=previous.current_mode,
                        current_mode=row.current_mode,
                        mode_switch=bool(row.current_mode_switch),
                        previous_source=previous.current_source,
                        current_source=row.current_source,
                        current_step_m=row.current_step_m,
                        sequence_step_m=row.sequence_step_m,
                        final_step_m=row.final_step_m,
                        selected_mode=row.selected_mode,
                    )
                )
        for row in steps.itertuples(index=False):
            group = particles[particles.step == row.step].sort_values("particle_id")
            parents = group.parent_particle_id.to_numpy(dtype=int)
            before = group[["before_x", "before_y"]].to_numpy()
            predicted = group[["predicted_x", "predicted_y"]].to_numpy()
            relocated = group[["relocated_x", "relocated_y"]].to_numpy()
            after = group[["after_x", "after_y"]].to_numpy()
            np.testing.assert_allclose(
                group[["after_parent_before_x", "after_parent_before_y"]],
                before[parents],
                atol=1e-12,
                rtol=0,
            )
            np.testing.assert_allclose(
                np.linalg.norm(after - before[parents], axis=1),
                group.after_parent_distance_m,
                atol=1e-12,
                rtol=0,
            )
            np.testing.assert_allclose(
                before
                + group.internal_step_m.to_numpy()[:, None]
                * np.column_stack(
                    (
                        np.cos(group.internal_heading_rad),
                        np.sin(group.internal_heading_rad),
                    )
                ),
                predicted,
                atol=1e-12,
                rtol=0,
            )
            if row.recovery_mode == "none":
                np.testing.assert_allclose(
                    after, relocated[parents], atol=1e-12, rtol=0
                )
                terminal_slot = int(
                    trajectory.loc[
                        trajectory.step == row.step, "sequence_particle_id"
                    ].iloc[0]
                )
                sequence_point = trajectory.loc[
                    trajectory.step == row.step, ["sequence_x", "sequence_y"]
                ].to_numpy()[0]
                if row.online_matches_final_history:
                    np.testing.assert_allclose(
                        after[terminal_slot], sequence_point, atol=1e-12, rtol=0
                    )
            valid = (~group.collision) | group.relocation_success
            expected = (
                group.before_weight
                * valid.astype(float)
                * group.stride_likelihood
                * group.state_predictive_likelihood
                ** metadata["pf_config"]["PF_MOTION_PREDICTIVE_WEIGHT_POWER"]
            )
            if metadata.get("proposal_method", "legacy") == "candidate-bank":
                expected = group.before_weight * group.candidate_mean_factor
            np.testing.assert_allclose(
                group.posterior_weight, expected, atol=1e-15, rtol=1e-12
            )
            prior = group.before_weight.to_numpy()
            post = group.normalized_posterior_weight.to_numpy()
            means = [
                np.average(points, axis=0, weights=weight)
                for points, weight in (
                    (before, prior),
                    (predicted, prior),
                    (relocated, prior),
                    (relocated, post),
                    (after, group.after_weight),
                )
            ]
            heading = group.pdr_heading_rad.iloc[0]
            for phase, start, end in zip(
                ["prediction", "relocation", "weighting", "resampling"],
                means[:-1],
                means[1:],
                strict=True,
            ):
                change = end - start
                effects.append(
                    dict(
                        dataset=entry.dataset,
                        seed=entry.seed,
                        step=row.step,
                        timestamp_s=row.timestamp_s,
                        phase=phase,
                        start_x=start[0],
                        start_y=start[1],
                        end_x=end[0],
                        end_y=end[1],
                        dx=change[0],
                        dy=change[1],
                        shift_m=float(np.linalg.norm(change)),
                        pf_forward_shift_m=float(
                            change @ np.array([np.cos(heading), np.sin(heading)])
                        ),
                        collective_recovery=row.recovery_mode != "none",
                    )
                )
            checks.append(
                dict(
                    dataset=entry.dataset,
                    seed=entry.seed,
                    step=row.step,
                    parent_positions_verified=True,
                    prediction_verified=True,
                    posterior_weights_verified=True,
                    normal_step_m=float(np.linalg.norm(normal_vectors[row.step - 1])),
                )
            )
        success = particles[particles.relocation_success]
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["baseline_exact_equal"] = True
        for angle in (90, 135):
            summary[f"relocated_normal_pdr_gt{angle}_fraction"] = float(
                (success.normal_pdr_angle_diff_deg > angle).mean()
            )
        summary["normal_pf_input_heading_diff_max_deg"] = float(
            angle_difference(
                particles.pdr_heading_rad, particles.normal_pdr_heading_rad
            ).max()
        )
        event_frame = pd.read_csv(run_dir / "anomaly_events.csv")
        for event_name in (
            "short_lt_0_5pdr",
            "opposed_pdr_gt90",
            "opposed_pdr_gt135",
            "path_turn_gt135",
        ):
            summary[f"final_{event_name}_count"] = int(
                (
                    (event_frame.path == "final") & (event_frame.event == event_name)
                ).sum()
            )
        np.testing.assert_allclose(
            trajectory.final_step_m.sum(), summary["final_length_m"], atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            trajectory.pdr_step_m.sum(), summary["pdr_length_m"], atol=1e-10, rtol=0
        )
        particles.to_csv(particle_path, index=False, compression="gzip")
        trajectory.to_csv(run_dir / "trajectory_comparison.csv", index=False)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=json_default)
        )
        summaries.append(summary)
        trajectories.append(trajectory)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.concat(trajectories, ignore_index=True).to_csv(
        output / "trajectory_comparison.csv", index=False
    )
    pd.DataFrame(effects).to_csv(output / "phase_mean_effects.csv", index=False)
    pd.DataFrame(switches).to_csv(output / "selection_switch_events.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "validation_checks.csv", index=False)
    metadata["diagnostic_code_sha256"] = {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted((ROOT / "agent").glob("agent_*pf_trajectory*.py"))
    }
    metadata["angle_definitions"] = {
        "pdr_angle_diff_deg": "PF専用に連続化されたPDR由来入力方位との最小絶対角差",
        "normal_pdr_angle_diff_deg": "通常PDR軌跡の座標差分方位との最小絶対角差",
        "internal_angle_diff_deg": "各入力粒子の提案thetaとの最小絶対角差",
    }
    metadata["saved_csv_validation_passed"] = True
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=json_default)
    )


def main():
    """固定した現行設定で通常・診断の完全一致を確認し、診断成果物を保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["18-52-40", "19-17-08"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 10, 42, 100])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--pf-proposal",
        choices=["legacy", "no-relocation", "candidate-bank"],
        default="legacy",
    )
    parser.add_argument("--pf-candidates", type=int, default=1)
    args = parser.parse_args()
    if args.pf_candidates < 1:
        parser.error("--pf-candidates は正の整数を指定してください。")
    if args.pf_proposal != "candidate-bank" and args.pf_candidates != 1:
        parser.error("--pf-candidates != 1 は candidate-bank の場合だけ指定できます。")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_paths = sorted((ROOT / "src").rglob("*.py"))
    before_sources = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    metadata = dict(
        created_at=datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        branch=subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
        initial_git_status=subprocess.check_output(
            ["git", "status", "--short"], text=True
        ),
        command=[sys.executable, *sys.argv],
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        pf_config={k: v for k, v in vars(config).items() if k.startswith("PF_")},
        source_sha256=before_sources,
        proposal_method=args.pf_proposal,
        candidate_count=args.pf_candidates,
        pdr_methods={
            k: getattr(config, k)
            for k in (
                "STEP_LENGTH_METHOD",
                "HEADING_METHOD",
                "MOTION_ESTIMATION",
                "SMOOTHING_MODE",
                "GYRO_BIAS_METHOD",
                "INITIAL_DIRECTION",
                "USER_HEIGHT_M",
                "FORWARD_HEADING_SOURCE",
            )
        },
        ble_enabled=False,
        history_candidate=False,
        ratio_denominator_min_m=EPS,
        provisional_diagnostics={
            "jump_m": 2.0,
            "short_ratio": 0.5,
            "long_ratio": 1.5,
            "opposed_deg": 90,
            "reversal_deg": 135,
        },
        selection_reason="EXP-067とagent/docs-archive/pf-history-prototype-20261022に大きな歩間変位が保存された2計測。現行設定の再現であり過去と同じ軌跡とは断定しない。",
        existing_output_entries=sorted(str(p) for p in (ROOT / "output").iterdir()),
        runs=[],
    )
    summaries, step_frames, trajectories, anomalies, interval_rows, manifest = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for dataset in args.datasets:
        data_dir = (
            ROOT
            / "input/1019Experiment"
            / f"PDR Sensor Logger (Research) 2026-10-19 {dataset}"
        )
        captured, acc, gyro = prepare(data_dir)
        prepared, floor = captured["prepared"], captured["floormap"]
        input_paths = [
            data_dir / "Accelerometer.csv",
            data_dir / "Gyroscope.csv",
            Path(floor.path).resolve(),
        ]
        hashes = {str(path): digest(path) for path in input_paths}
        for seed in args.seeds:
            run_dir = output / f"{dataset}_seed{seed}"
            run_dir.mkdir()
            settings = replace(
                captured["settings"],
                seed=seed,
                proposal_method=args.pf_proposal,
                candidate_count=args.pf_candidates,
            )
            baseline = commands.run_particle(prepared, floor, settings)
            result, records, final = instrumented_run(prepared, floor, settings)
            assert_same(baseline.trajectory, result.trajectory)
            assert_same(baseline.step_lengths, result.step_lengths)
            assert_same(baseline.step_headings, result.step_headings)
            assert_same(baseline.particle, result.particle)
            summary, steps, trajectory, events, intervals = analyze_run(
                result,
                records,
                final,
                prepared,
                floor,
                run_dir,
                dataset,
                seed,
            )
            summary["baseline_exact_equal"] = True
            run_meta = dict(
                dataset=dataset,
                seed=seed,
                proposal_method=args.pf_proposal,
                candidate_count=args.pf_candidates,
                source_sha256=before_sources,
                input_data_dir=str(data_dir),
                input_sha256=hashes,
                floormap_path=str(Path(floor.path).resolve()),
                origin_px=floor.origin_px,
                scale_m_per_px=floor.scale,
                y_sign=pixel_y_sign(prepared.gx_mean, prepared.gz_mean),
                gx_mean=prepared.gx_mean,
                gz_mean=prepared.gz_mean,
                initial_position_m=[0.0, 0.0],
                initial_heading_deg=config.INITIAL_DIRECTION,
                coordinate_transform="px=origin_x+x/scale; py=origin_y+y_sign*y/scale",
                particle_settings=asdict(settings),
                sensor_acc_time_range_s=[float(acc.iloc[0, 0]), float(acc.iloc[-1, 0])],
                sensor_gyro_time_range_s=[
                    float(gyro.iloc[0, 0]),
                    float(gyro.iloc[-1, 0]),
                ],
                accelerometer_duration_s=float(acc.iloc[-1, 0] - acc.iloc[0, 0]),
                first_step_time_s=float(prepared.t_at_steps[0]),
                last_step_time_s=float(prepared.t_at_steps[-1]),
                first_to_last_step_duration_s=float(
                    prepared.t_at_steps[-1] - prepared.t_at_steps[0]
                ),
                steps=len(result.step_lengths),
                gyro_bias=prepared.df_gyro.attrs["gyro_bias_result"]._asdict(),
                baseline_exact_equal=True,
                equality_scope=(
                    "trajectory, step_lengths, step_headings, all_particles; "
                    "all stage fields, diagnostics, path comparisons; "
                    "exact equality including dtype"
                ),
                selected_mode=final["selected_mode"],
                selection_reason="sequence反転数がcurrentより少ない場合にsequence、同数ならcurrent。BLEなし。",
                commit=metadata["commit"],
            )
            write_json(run_dir / "run_metadata.json", run_meta)
            metadata["runs"].append(run_meta)
            summaries.append(summary)
            step_frames.append(steps)
            trajectories.append(trajectory)
            anomalies.append(events)
            interval_rows.extend(intervals)
            manifest.append(
                dict(
                    dataset=dataset,
                    seed=seed,
                    rows=len(records) * settings.count,
                    path=str((run_dir / "particle_stages.csv.gz").relative_to(output)),
                )
            )
            print(json.dumps(summary, default=json_default), flush=True)
        assert all(
            digest(Path(path)) == hash_value for path, hash_value in hashes.items()
        )
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.concat(step_frames, ignore_index=True).to_csv(
        output / "step_diagnostics.csv", index=False
    )
    pd.concat(trajectories, ignore_index=True).to_csv(
        output / "trajectory_comparison.csv", index=False
    )
    pd.concat(anomalies, ignore_index=True).to_csv(
        output / "anomaly_events.csv", index=False
    )
    pd.DataFrame(interval_rows).to_csv(output / "interval_comparison.csv", index=False)
    pd.DataFrame(manifest).to_csv(output / "particle_stages.csv", index=False)
    assert before_sources == {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    metadata["source_unchanged"] = True
    metadata["input_hashes_unchanged"] = True
    write_json(output / "run_metadata.json", metadata)
    augment_saved_outputs(output)
    if not args.skip_figures:
        from agent_plot_pf_trajectory_diagnosis import plot_run, plot_summary

        figure_runs = {(run["dataset"], args.seeds[0]) for run in metadata["runs"]}
        summary_frame = pd.DataFrame(summaries)
        for column in ("current_max_step_m", "final_max_step_m"):
            worst = summary_frame.loc[summary_frame[column].idxmax()]
            figure_runs.add((worst.dataset, int(worst.seed)))
        for run in metadata["runs"]:
            if (run["dataset"], run["seed"]) in figure_runs:
                plot_run(output / f"{run['dataset']}_seed{run['seed']}", run)
        plot_summary(output)
    if args.pf_proposal == "legacy":
        from agent_report_pf_trajectory_diagnosis import write_report

        write_report(output)
    else:
        (output / "diagnostic_report.md").write_text(
            "# PF候補生成の比較診断\n\n"
            f"方式: `{args.pf_proposal}`、候補数: {args.pf_candidates}。\n\n"
            "通常実行と診断実行の完全一致、保存CSVの重み更新・親対応検算を確認した。\n"
            "円内再配置は無効。candidate-bankのpredicted/relocated列は採用提案を示し、"
            "全候補不採用の親には候補0を診断用に保持する。collision列は採用提案の無効を示す。\n"
            "candidate_*列に全候補の距離・角度・地図通過率・親生存率と平均因子を保存する。\n"
            "relocation_failed列は旧形式互換の無効提案数であり、円内救済試行の失敗を意味しない。\n"
            "正解経路との比較は実施していないため、測位精度の向上を示す結果ではない。\n"
            "比較にはsummary.csv、step_diagnostics.csv、trajectory_comparison.csvを使用する。\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
