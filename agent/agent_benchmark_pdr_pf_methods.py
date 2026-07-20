"""通常PDRとparticle filterの現行手法を一括比較する評価スクリプト。

役割:
    共通の正解軌跡を持つ反復計測データについて、通常PDRの3方式とPFの
    運動推定・平滑化の組み合わせを同一指標で比較する。
入力:
    ``--data-dir`` の各phyphox CSV、``--truth-csv`` のx/y正解軌跡、
    フロアマップ、origin、scale、PFのseedとノイズ設定を使用する。
出力:
    データ・方式・seedごとの指標と、方式別のRMSE・終点誤差・地図違反・
    recovery failure・seed間ばらつきの集計をJSONまたはCSVで出力する。
処理フロー:
    正解軌跡と地図を読み、各データの共有PDRステップを方式別に準備し、
    通常PDRとPFを評価した後、行ごとの指標と方式別集計を生成する。
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rikka.analyze.particle_filter import (
    ParticleFilterStepDiagnostics,
    _evaluate_particle_transitions,
    _normalize_floormap_gray,
    run_particle_filter,
)
from rikka.analyze.pdr.models import PreparedPdrSteps
from rikka.analyze.pdr.sensors import load_sensor_data
from rikka.analyze.pdr.trajectory import prepare_pdr_steps
from rikka.analyze.trajectory_direction import evaluate_terminal_direction
from rikka.config import (
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    PF_SIGMA_HEADING,
    PF_SIGMA_INIT_HEADING,
    PF_SIGMA_STEP_LENGTH_RATIO,
)

DEFAULT_DATA_DIRS = tuple(
    Path("input/sensor_data") / f"1turn_rightsidestep_3turn_leftsidestep{suffix}"
    for suffix in ("", "5", "6", "7", "8")
)
DEFAULT_TRUTH_CSV = Path(
    "input/correct_path/1turn_rightsidestep_3turn_leftsidestep/walk_trace (3).csv"
)
PDR_METHODS = (
    "legacy",
    "adaptive-causal",
    "adaptive-offline",
    "robust-causal",
    "robust-offline",
)
PF_METHODS = (
    "legacy-causal",
    "legacy-offline",
    "adaptive-causal",
    "adaptive-offline",
    "robust-causal",
    "robust-offline",
)


def _sample_by_arclength(points: np.ndarray, count: int = 300) -> np.ndarray:
    """既存評価と同じ方法で正規化弧長上の等間隔座標を返す。"""
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
    filtered = points[keep]
    distances = np.r_[
        0.0,
        np.cumsum(np.linalg.norm(np.diff(filtered, axis=0), axis=1)),
    ]
    targets = np.linspace(0.0, distances[-1], count)
    return np.column_stack(
        [
            np.interp(targets, distances, filtered[:, axis])
            for axis in range(filtered.shape[1])
        ]
    )


def _parse_args() -> argparse.Namespace:
    """評価対象、方式、PF設定、出力形式を受け取る。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", nargs="+", type=Path, default=DEFAULT_DATA_DIRS)
    parser.add_argument("--truth-csv", type=Path, default=DEFAULT_TRUTH_CSV)
    parser.add_argument("--floormap", type=Path, default=Path(FLOORMAP_PATH))
    parser.add_argument("--origin-px", nargs=2, type=int, default=FLOORMAP_ORIGIN_PX)
    parser.add_argument("--scale", type=float, default=FLOORMAP_SCALE)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 10, 42, 100])
    parser.add_argument(
        "--pdr-methods", nargs="+", choices=PDR_METHODS, default=PDR_METHODS
    )
    parser.add_argument(
        "--pf-methods", nargs="+", choices=PF_METHODS, default=PF_METHODS
    )
    parser.add_argument(
        "--sigma-init-heading", type=float, default=PF_SIGMA_INIT_HEADING
    )
    parser.add_argument("--sigma-heading", type=float, default=PF_SIGMA_HEADING)
    parser.add_argument(
        "--sigma-step-length-ratio",
        type=float,
        default=PF_SIGMA_STEP_LENGTH_RATIO,
    )
    parser.add_argument(
        "--motion-predictive-weight-powers",
        nargs="+",
        type=float,
        default=[0.0],
    )
    parser.add_argument(
        "--pf-path-selections",
        nargs="+",
        choices=("current", "sequence"),
        default=["current"],
    )
    parser.add_argument("--direction-fixed-lag", type=int, default=5)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        help="候補軌跡CSVとmanifestを保存するディレクトリ",
    )
    parser.add_argument("--skip-pdr", action="store_true")
    parser.add_argument("--skip-pf", action="store_true")
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _split_method(method: str) -> tuple[str, str]:
    """方式名を運動推定方式と平滑化方式へ分解する。"""
    if method == "legacy":
        return "legacy", "causal"
    motion_estimation, smoothing = method.split("-", maxsplit=1)
    return motion_estimation, smoothing


def _trajectory_metrics(
    trajectory: np.ndarray,
    truth: np.ndarray,
    sampled_truth: np.ndarray,
    truth_length: float,
) -> dict[str, Any]:
    """既存評価と同じ正規化弧長RMSEと距離指標を計算する。"""
    sampled = _sample_by_arclength(trajectory)
    errors = np.linalg.norm(sampled - sampled_truth, axis=1)
    terminal = evaluate_terminal_direction(sampled, sampled_truth)
    return {
        "arc_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "endpoint_error_m": float(np.linalg.norm(trajectory[-1] - truth[-1])),
        "estimated_length_m": float(
            np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum()
        ),
        "truth_length_m": truth_length,
        "terminal_direction_error_deg": terminal.direction_error_deg,
        "terminal_progress_cosine": terminal.progress_cosine,
        "terminal_opposed_fraction": terminal.opposed_fraction,
        "terminal_direction_failure": terminal.failure,
    }


def _get_prepared(
    method: str,
    df_acc: pd.DataFrame,
    df_gyro: pd.DataFrame,
    cache: dict[tuple[str, str], PreparedPdrSteps],
    direction_fixed_lag: int,
) -> PreparedPdrSteps:
    """方式に対応する共有PDRステップを準備し、データ内で再利用する。"""
    motion_estimation, smoothing = _split_method(method)
    key = (motion_estimation, smoothing)
    if key not in cache:
        cache[key] = prepare_pdr_steps(
            df_acc,
            df_gyro,
            motion_estimation=motion_estimation,
            smoothing_mode=smoothing,
            direction_fixed_lag=direction_fixed_lag,
        )
    return cache[key]


def _write_candidate_trajectory(
    candidate_dir: Path | None,
    candidate_id: str,
    trajectory: np.ndarray,
) -> str | None:
    """候補軌跡をmanifestから参照できるCSVとして保存する。"""
    if candidate_dir is None:
        return None
    trajectory_dir = candidate_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    safe_name = candidate_id.replace(":", "__").replace("/", "_")
    path = trajectory_dir / f"{safe_name}.csv"
    pd.DataFrame(trajectory, columns=["x", "y"]).to_csv(path, index=False)
    return str(path.resolve())


def _evaluate_pdr(
    data_name: str,
    method: str,
    prepared: PreparedPdrSteps,
    truth: np.ndarray,
    sampled_truth: np.ndarray,
    truth_length: float,
    candidate_dir: Path | None,
) -> dict[str, Any]:
    """準備済みPDR結果を正解軌跡と比較する。"""
    motion_estimation, smoothing = _split_method(method)
    trajectory = np.asarray(prepared.trajectory, dtype=float)
    candidate_id = f"{data_name}:pdr:{method}:none"
    row: dict[str, Any] = {
        "record_type": "measurement",
        "family": "pdr",
        "data": data_name,
        "method": method,
        "motion_estimation": motion_estimation,
        "smoothing": smoothing,
        "seed": None,
        "candidate_id": candidate_id,
        "trajectory_csv": _write_candidate_trajectory(
            candidate_dir, candidate_id, trajectory
        ),
        "weight": 1.0,
        "eligible": True,
        "steps": len(prepared.step_lengths),
        "wall_crossings": None,
        "recovery_failures": None,
        "checkpoint_replays": None,
        "max_position_spread_m": None,
    }
    row.update(_trajectory_metrics(trajectory, truth, sampled_truth, truth_length))
    return row


def _evaluate_pf(
    data_name: str,
    method: str,
    seed: int,
    prepared: PreparedPdrSteps,
    truth: np.ndarray,
    sampled_truth: np.ndarray,
    truth_length: float,
    map_gray: np.ndarray,
    args: argparse.Namespace,
    motion_predictive_weight_power: float,
    path_selection: str,
) -> dict[str, Any]:
    """準備済みステップでPFを実行し、精度と地図制約を評価する。"""
    diagnostics: list[ParticleFilterStepDiagnostics] = []
    result = run_particle_filter(
        prepared.step_detection.peaks,
        prepared.df_gyro,
        prepared.df_acc,
        prepared.gx_mean,
        prepared.gz_mean,
        floormap_path=args.floormap,
        origin_px=tuple(args.origin_px),
        scale=args.scale,
        prepared_step_headings=prepared.step_headings,
        prepared_step_lengths=prepared.step_lengths,
        prepared_step_times=prepared.t_at_steps,
        prepared_motion_evidences=prepared.motion_evidences,
        prepared_motion_posteriors=prepared.motion_posteriors,
        sigma_init_heading=args.sigma_init_heading,
        sigma_heading=args.sigma_heading,
        sigma_sl_ratio=args.sigma_step_length_ratio,
        seed=seed,
        diagnostics_collector=diagnostics,
        motion_predictive_weight_power=motion_predictive_weight_power,
        path_selection=path_selection,
    )
    trajectory = np.asarray(result[0], dtype=float)
    valid = _evaluate_particle_transitions(
        trajectory[:-1],
        trajectory[1:],
        map_gray,
        prepared.gx_mean,
        prepared.gz_mean,
        tuple(args.origin_px),
        args.scale,
    )
    motion_estimation, smoothing = _split_method(method)
    method_label = (
        method
        if motion_predictive_weight_power == 0.0
        else f"{method}-motion-power-{motion_predictive_weight_power:g}"
    )
    if path_selection != "current":
        method_label = f"{method_label}-path-{path_selection}"
    candidate_id = f"{data_name}:pf:{method_label}:{seed}"
    row: dict[str, Any] = {
        "record_type": "measurement",
        "family": "pf",
        "data": data_name,
        "method": method_label,
        "motion_estimation": motion_estimation,
        "smoothing": smoothing,
        "seed": seed,
        "candidate_id": candidate_id,
        "trajectory_csv": _write_candidate_trajectory(
            args.candidate_dir, candidate_id, trajectory
        ),
        "weight": 1.0,
        "eligible": True,
        "motion_predictive_weight_power": motion_predictive_weight_power,
        "pf_path_selection": path_selection,
        "steps": len(prepared.step_lengths),
        "wall_crossings": int(np.count_nonzero(~valid)),
        "recovery_failures": sum(
            item.recovery_mode == "failed_hold" for item in diagnostics
        ),
        "checkpoint_replays": sum(
            item.recovery_mode == "checkpoint_replay" for item in diagnostics
        ),
        "max_position_spread_m": max(
            item.position_spread_rms_m for item in diagnostics
        ),
    }
    row.update(_trajectory_metrics(trajectory, truth, sampled_truth, truth_length))
    return row


def _summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """方式ごとの精度・安全性・データ内seedばらつきを集計する。"""
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for (family, method), group in frame.groupby(["family", "method"], sort=False):
        is_pf = family == "pf"
        seed_rmse_stds: list[float] = []
        seed_endpoint_stds: list[float] = []
        if is_pf:
            for _, data_group in group.groupby("data", sort=False):
                seed_rmse_stds.append(float(data_group["arc_rmse_m"].std(ddof=0)))
                seed_endpoint_stds.append(
                    float(data_group["endpoint_error_m"].std(ddof=0))
                )
        summaries.append(
            {
                "record_type": "summary",
                "family": family,
                "method": method,
                "runs": len(group),
                "datasets": int(group["data"].nunique()),
                "arc_rmse_median_m": float(group["arc_rmse_m"].median()),
                "arc_rmse_max_m": float(group["arc_rmse_m"].max()),
                "endpoint_error_median_m": float(group["endpoint_error_m"].median()),
                "endpoint_error_max_m": float(group["endpoint_error_m"].max()),
                "wall_crossings_total": int(group["wall_crossings"].sum())
                if is_pf
                else None,
                "wall_crossings_max": int(group["wall_crossings"].max())
                if is_pf
                else None,
                "recovery_failures_total": int(group["recovery_failures"].sum())
                if is_pf
                else None,
                "recovery_failures_max": int(group["recovery_failures"].max())
                if is_pf
                else None,
                "seed_arc_rmse_std_median_m": float(np.median(seed_rmse_stds))
                if seed_rmse_stds
                else None,
                "seed_arc_rmse_std_max_m": float(np.max(seed_rmse_stds))
                if seed_rmse_stds
                else None,
                "seed_endpoint_error_std_median_m": float(np.median(seed_endpoint_stds))
                if seed_endpoint_stds
                else None,
                "seed_endpoint_error_std_max_m": float(np.max(seed_endpoint_stds))
                if seed_endpoint_stds
                else None,
            }
        )
    return summaries


def _write_result(
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    output_format: str,
    output: Path | None,
) -> None:
    """JSONまたは単一CSVとして標準出力または指定先へ書き出す。"""
    if output_format == "json":
        content = json.dumps(
            {"measurements": rows, "method_summaries": summaries},
            ensure_ascii=False,
            indent=2,
        )
    else:
        content = pd.concat(
            [pd.DataFrame(rows), pd.DataFrame(summaries)], ignore_index=True
        ).to_csv(index=False)
    if output is None:
        print(content)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(output, file=sys.stderr)


def main() -> None:
    """指定された全データと全方式を評価して集計する。"""
    args = _parse_args()
    if args.skip_pdr and args.skip_pf:
        raise ValueError("skip-pdr と skip-pf は同時に指定できません")
    truth_frame = pd.read_csv(args.truth_csv)
    truth = truth_frame[["x", "y"]].to_numpy(dtype=float)
    truth -= truth[0]
    sampled_truth = _sample_by_arclength(truth)
    truth_length = float(np.linalg.norm(np.diff(truth, axis=0), axis=1).sum())
    map_gray = _normalize_floormap_gray(plt.imread(args.floormap))

    rows: list[dict[str, Any]] = []
    for data_dir in args.data_dir:
        df_acc, df_gyro = load_sensor_data(data_dir)
        prepared_cache: dict[tuple[str, str], PreparedPdrSteps] = {}

        if not args.skip_pdr:
            for method in args.pdr_methods:
                rows.append(
                    _evaluate_pdr(
                        data_dir.name,
                        method,
                        _get_prepared(
                            method,
                            df_acc,
                            df_gyro,
                            prepared_cache,
                            args.direction_fixed_lag,
                        ),
                        truth,
                        sampled_truth,
                        truth_length,
                        args.candidate_dir,
                    )
                )
        if not args.skip_pf:
            for method in args.pf_methods:
                prepared = _get_prepared(
                    method,
                    df_acc,
                    df_gyro,
                    prepared_cache,
                    args.direction_fixed_lag,
                )
                for motion_power in args.motion_predictive_weight_powers:
                    for path_selection in args.pf_path_selections:
                        for seed in args.seeds:
                            rows.append(
                                _evaluate_pf(
                                    data_dir.name,
                                    method,
                                    seed,
                                    prepared,
                                    truth,
                                    sampled_truth,
                                    truth_length,
                                    map_gray,
                                    args,
                                    motion_power,
                                    path_selection,
                                )
                            )

    _write_result(rows, _summarize(rows), args.format, args.output)
    if args.candidate_dir is not None:
        args.candidate_dir.mkdir(parents=True, exist_ok=True)
        manifest_columns = [
            "candidate_id",
            "data",
            "family",
            "method",
            "seed",
            "trajectory_csv",
            "weight",
            "eligible",
            "wall_crossings",
            "recovery_failures",
        ]
        pd.DataFrame(rows, columns=manifest_columns).to_csv(
            args.candidate_dir / "candidates.csv", index=False
        )


if __name__ == "__main__":
    main()
