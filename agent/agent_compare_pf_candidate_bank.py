"""フェーズ0〜2のPF候補生成を小さい成果物で比較する。

役割: 基準診断を再現し、候補方式ごとの歩行モデル・系譜・復旧を比較する。
入力: 1019Experimentの2計測、seed、方式、読取専用の既存基準診断。
出力: 新規出力先へ経路CSV、歩集計CSV、実行要約JSON、比較図、メタデータ。
処理フロー: 共有prepareとinstrumented_runを再利用し、全粒子CSVを書かずに
    状態を要約する。基準経路・重み式とソース/入力不変を検算する。
    正解経路はないため測位精度の順位は付けない。
"""

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from agent_diagnose_pf_trajectory_stages import (
    EPS,
    ROOT,
    ancestry_count,
    angle_difference,
    digest,
    instrumented_run,
    prepare,
    safe_ratio,
    stats,
    write_json,
)

from rikka.cli import commands
from rikka.particle.lib.map_constraints import _evaluate_particle_transitions
from rikka.particle.lib.sequence_path import _unsupported_reversal_count

METHODS = {
    "legacy": ("legacy", 1),
    "no-relocation": ("no-relocation", 1),
    "bank4": ("candidate-bank", 4),
    "bank8": ("candidate-bank", 8),
    "bank16": ("candidate-bank", 16),
}


def summarize(result, records, final, prepared, floor, settings, run_dir):
    """オンライン提案と確定履歴の由来を分けて要約する。"""
    gray = commands._load_floormap_gray(floor.path)
    paths = {
        "pdr": np.asarray(prepared.trajectory),
        "current": final["current_path"],
        "sequence": final["sequence_path"],
        "final": np.asarray(result.trajectory),
    }
    trajectory = pd.DataFrame({"step": np.arange(len(records) + 1)})
    pdr_length = float(np.sum(result.step_lengths))
    summary = {"pdr_length_m": pdr_length, "selected_mode": final["selected_mode"]}
    for name, path in paths.items():
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        trajectory[f"{name}_x"], trajectory[f"{name}_y"] = path.T
        trajectory[f"{name}_step_m"] = np.r_[np.nan, distances]
        walls = _evaluate_particle_transitions(
            path[:-1],
            path[1:],
            gray,
            prepared.gx_mean,
            prepared.gz_mean,
            floor.origin_px,
            floor.scale,
        )
        summary.update(
            {
                f"{name}_length_m": float(distances.sum()),
                f"{name}_max_step_m": float(distances.max()),
                f"{name}_jump_gt2m_count": int((distances > 2).sum()),
                f"{name}_wall_crossings": int((~walls).sum()),
                f"{name}_length_ratio": float(distances.sum() / pdr_length),
                f"{name}_unsupported_reversals": _unsupported_reversal_count(
                    path, final["sensor_headings"], final["turning_evidence"]
                ),
            }
        )
    rows = []
    for index, (snap, stage) in enumerate(
        zip(records, result.particle.stages, strict=True)
    ):
        t = index + 1
        n = len(snap["particles_before"])
        bank = snap["candidate_bank"]
        displacement = snap["relocated"] - snap["particles_before"]
        distance = np.linalg.norm(displacement, axis=1)
        heading = np.arctan2(displacement[:, 1], displacement[:, 0])
        internal_error = angle_difference(heading, snap["theta"])
        internal_error[distance <= EPS] = np.nan
        matches_positions = np.array_equal(
            snap["after_particles"], final["position_history"][t]
        )
        matches_parents = np.array_equal(
            snap["after_parent_indices"], final["parent_history"][index]
        )
        history_matches = bool(stage.proposal_matches_final_history)
        row = dict(
            step=t,
            timestamp_s=snap["timestamp_s"],
            recovery_mode=snap["recovery_mode"],
            recovery_attempted=snap["recovery_attempted"],
            ess_before=snap["ess_before_observation"],
            ess_after=snap["ess_after_observation"],
            ancestors_5=ancestry_count(final["parent_history"], t, 5, n),
            ancestors_10=ancestry_count(final["parent_history"], t, 10, n),
            online_matches_final_positions=matches_positions,
            online_matches_final_parents=matches_parents,
            proposal_history_version=stage.proposal_history_version,
            history_version=stage.history_version,
            proposal_matches_final_history=history_matches,
            final_edge_candidate_id_trusted=history_matches
            and matches_positions
            and matches_parents,
            relocation_count=snap["relocated_count"],
            resampled=snap["resampled"],
        )
        accepted = (
            (bank.selected_indices >= 0)
            if bank is not None
            else snap["relocated_valid"]
        )
        row["accepted_parent_fraction"] = float(accepted.mean())
        row.update(stats(distance[accepted], "selected_distance_m"))
        row.update(
            stats(
                safe_ratio(distance[accepted], snap["pdr_step_m"]),
                "selected_distance_ratio",
            )
        )
        row.update(stats(internal_error[accepted], "selected_internal_angle_deg"))
        row.update(
            stats(
                np.abs(distance[accepted] - snap["sl"][accepted]),
                "selected_internal_distance_error_m",
            )
        )
        if bank is not None:
            all_distance = np.linalg.norm(
                bank.positions - snap["particles_before"][:, None], axis=2
            )
            all_displacement = bank.positions - snap["particles_before"][:, None]
            all_heading = np.arctan2(
                all_displacement[:, :, 1], all_displacement[:, :, 0]
            )
            angles = angle_difference(all_heading, snap["pdr_heading"])
            angles[all_distance <= EPS] = np.nan
            row.update(stats(all_distance.ravel(), "candidate_distance_m"))
            row.update(
                stats(
                    safe_ratio(all_distance, snap["pdr_step_m"]).ravel(),
                    "candidate_distance_ratio",
                )
            )
            row.update(stats(angles.ravel(), "candidate_pdr_angle_deg"))
            row.update(
                candidate_map_pass_fraction=float(bank.valid_transition.mean()),
                candidate_parent_survival_fraction=float(
                    (bank.selected_indices >= 0).mean()
                ),
                candidate_positive_factor_fraction=float((bank.factors > 0).mean()),
                candidate_mean_factor_mean=float(bank.mean_factors.mean()),
            )
            expected = snap["weights_before"] * bank.mean_factors
            np.testing.assert_allclose(
                snap["posterior_weights"], expected, atol=1e-15, rtol=1e-12
            )
            row["candidate_weight_formula_verified"] = True
        rows.append(row)
    steps = pd.DataFrame(rows)
    summary.update(
        collective_recovery_steps=int((steps.recovery_mode != "none").sum()),
        failed_hold_steps=int((steps.recovery_mode == "failed_hold").sum()),
        ess_after_min=float(steps.ess_after.min()),
        ess_after_median=float(steps.ess_after.median()),
        ancestors5_min=float(steps.ancestors_5.min()),
        ancestors10_min=float(steps.ancestors_10.min()),
        online_final_positions_mismatch_steps=int(
            (~steps.online_matches_final_positions).sum()
        ),
        online_final_parent_mismatch_steps=int(
            (~steps.online_matches_final_parents).sum()
        ),
        untrusted_final_edge_steps=int((~steps.final_edge_candidate_id_trusted).sum()),
        max_history_version=int(steps.history_version.max()),
        relocation_count=int(steps.relocation_count.sum()),
    )
    for col in steps.columns:
        if col.startswith("candidate_") or col.startswith("selected_"):
            if pd.api.types.is_numeric_dtype(steps[col]):
                summary[f"{col}_step_mean"] = float(steps[col].mean())
    trajectory.to_csv(run_dir / "trajectory_comparison.csv", index=False)
    steps.to_csv(run_dir / "step_summary.csv", index=False)
    return summary, trajectory


def plot_comparison(output, summaries):
    """精度順位と解釈しない比較図を保存する。"""
    import matplotlib.pyplot as plt

    for dataset in summaries.dataset.unique():
        subset = summaries[summaries.dataset == dataset]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for label in subset.method.unique():
            group = subset[subset.method == label]
            axes[0].scatter([label] * len(group), group.final_max_step_m, label=label)
            axes[1].scatter([label] * len(group), group.final_length_ratio)
            axes[2].scatter([label] * len(group), group.ancestors10_min)
        for ax, title in zip(
            axes,
            [
                "Maximum final edge (m)",
                "Final / PDR distance",
                "Minimum 10-step ancestors",
            ],
            strict=True,
        ):
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=35)
            ax.grid(alpha=0.2)
        fig.suptitle(f"{dataset}: diagnostics without ground truth")
        fig.tight_layout()
        fig.savefig(output / f"comparison_{dataset}.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 6))
        for label in subset.method.unique():
            first = subset[subset.method == label].iloc[0]
            path = pd.read_csv(
                output
                / f"{dataset}_seed{int(first.seed)}_{label}"
                / "trajectory_comparison.csv"
            )
            ax.plot(
                path.final_x,
                path.final_y,
                label=f"{label}, seed {first.seed}",
                alpha=0.8,
            )
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title(f"{dataset}: trajectory overlay, no ground truth")
        fig.tight_layout()
        fig.savefig(output / f"trajectories_{dataset}.png", dpi=150)
        plt.close(fig)


def main():
    """同じ入力・seedの方式比較と基準再現を実行する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["18-52-40", "19-17-08"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 10, 42, 100])
    parser.add_argument(
        "--methods", nargs="+", choices=list(METHODS), default=list(METHODS)
    )
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "docs/pf-trajectory-diagnosis-20261022"
    )
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted((ROOT / "src").rglob("*.py"))
    }
    metadata = dict(
        created_at=datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        command=[sys.executable, *sys.argv],
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256=source_hashes,
        input_sha256={},
        runs=[],
        existing_output_entries=sorted(str(p) for p in (ROOT / "output").iterdir()),
        baseline=str(args.baseline),
        accuracy_claim="正解経路なし。精度順位は付けない。",
    )
    baseline_summary = pd.read_csv(args.baseline / "summary.csv")
    baseline_trajectory = pd.read_csv(args.baseline / "trajectory_comparison.csv")
    summaries = []
    for dataset in args.datasets:
        data_dir = (
            ROOT
            / "input/1019Experiment"
            / f"PDR Sensor Logger (Research) 2026-10-19 {dataset}"
        )
        captured, _, _ = prepare(data_dir)
        prepared, floor = captured["prepared"], captured["floormap"]
        for path in [
            data_dir / "Accelerometer.csv",
            data_dir / "Gyroscope.csv",
            Path(floor.path).resolve(),
        ]:
            metadata["input_sha256"][str(path)] = digest(path)
        for seed in args.seeds:
            for method in args.methods:
                proposal, count = METHODS[method]
                settings = replace(
                    captured["settings"],
                    seed=seed,
                    proposal_method=proposal,
                    candidate_count=count,
                )
                run_dir = output / f"{dataset}_seed{seed}_{method}"
                run_dir.mkdir()
                result, records, final = instrumented_run(prepared, floor, settings)
                summary, trajectory = summarize(
                    result, records, final, prepared, floor, settings, run_dir
                )
                summary.update(
                    dataset=dataset,
                    seed=seed,
                    method=method,
                    proposal_method=proposal,
                    candidate_count=count,
                )
                if method == "legacy":
                    old = baseline_trajectory[
                        (baseline_trajectory.dataset == dataset)
                        & (baseline_trajectory.seed == seed)
                    ]
                    columns = [
                        f"{name}_{axis}"
                        for name in ["pdr", "current", "sequence", "final"]
                        for axis in ["x", "y"]
                    ]
                    np.testing.assert_allclose(
                        trajectory[columns], old[columns], atol=1e-12, rtol=0
                    )
                    old_summary = baseline_summary[
                        (baseline_summary.dataset == dataset)
                        & (baseline_summary.seed == seed)
                    ].iloc[0]
                    for key in [
                        f"{name}_{metric}"
                        for name in ["current", "sequence", "final"]
                        for metric in [
                            "length_m",
                            "max_step_m",
                            "jump_gt2m_count",
                            "wall_crossings",
                            "unsupported_reversals",
                        ]
                    ]:
                        np.testing.assert_allclose(
                            summary[key], old_summary[key], atol=1e-12, rtol=0
                        )
                    summary["saved_baseline_reproduced"] = True
                write_json(run_dir / "summary.json", summary)
                summaries.append(summary)
                metadata["runs"].append(
                    dict(
                        dataset=dataset,
                        seed=seed,
                        method=method,
                        settings=asdict(settings),
                    )
                )
                pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
                print(
                    json.dumps(
                        {
                            k: summary[k]
                            for k in [
                                "dataset",
                                "seed",
                                "method",
                                "final_max_step_m",
                                "final_length_ratio",
                            ]
                        }
                    ),
                    flush=True,
                )
    assert source_hashes == {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted((ROOT / "src").rglob("*.py"))
    }
    assert all(
        digest(Path(path)) == value for path, value in metadata["input_sha256"].items()
    )
    metadata.update(source_unchanged=True, input_hashes_unchanged=True)
    write_json(output / "run_metadata.json", metadata)
    if not args.skip_figures:
        plot_comparison(output, pd.DataFrame(summaries))


if __name__ == "__main__":
    main()
