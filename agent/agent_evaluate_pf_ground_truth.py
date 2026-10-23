"""PF軌跡を同期していない正解軌跡と形状比較する検証スクリプト。

役割:
    指定したセンサーデータを複数seedでPF実行し、開始点を合わせた正解軌跡と
    正規化弧長で比較する。
入力:
    ``--data-dir``のAccelerometer/Gyroscope CSV、``--truth-csv``のx/y列、
    フロアマップ、origin、scaleを使用する。``--ble-data``指定時は同じ歩列へ
    ランドマーク検出を割り当て、指定したPF反映方式も比較する。
出力:
    seedごとのRMSE、終点誤差、壁交差、recovery結果をJSONで標準出力し、
    ``--plot-path``指定時は正解軌跡とPF軌跡の比較画像を保存する。
処理フロー:
    共通PDRステップを準備し、seedごとにPF実行、弧長正規化、地図遷移検査、
    指標集計、任意の重ね描きを行う。
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rikka.ble.pipeline import run_ble_landmark_detection
from rikka.common.config import (
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    GYRO_BIAS_METHOD,
    MOTION_ESTIMATION,
    PF_LANDMARK_LIKELIHOOD_FLOOR,
    PF_LANDMARK_MAX_JUMP_M,
    PF_LANDMARK_RESET_HEADING_SIGMA,
    PF_LANDMARK_RESET_SIGMA_M,
    PF_LANDMARK_RESET_SPREAD_RATIO,
    PF_LANDMARK_SIGMA_M,
    PF_MOTION_PREDICTIVE_WEIGHT_POWER,
    PF_PATH_SELECTION,
    PF_SIGMA_HEADING,
    PF_SIGMA_INIT_HEADING,
    PF_SIGMA_STEP_LENGTH_RATIO,
    SMOOTHING_MODE,
)
from rikka.common.lib.sensors import load_sensor_data
from rikka.common.lib.trajectory_direction import evaluate_terminal_direction
from rikka.common.settings import BleLandmarkSettings
from rikka.particle.lib.map_constraints import (
    _evaluate_particle_transitions,
    _normalize_floormap_gray,
)
from rikka.particle.lib.recorder import (
    ParticleFilterStepDiagnostics,
    ParticlePathComparison,
)
from rikka.particle.lib.runner import run_particle_filter
from rikka.pdr.lib.preparation import prepare_pdr_steps


def _sample_by_arclength(points: np.ndarray, count: int = 300) -> np.ndarray:
    """重複点を除き、正規化弧長上の等間隔座標を返す。"""
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
    """検証用コマンドライン引数を返す。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pf-proposal",
        choices=("legacy", "no-relocation", "candidate-bank"),
        default="legacy",
    )
    parser.add_argument("--pf-candidates", type=int, default=1)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "input/sensor_data/natsuki/1turn_rightsidestep_3turn_leftsidestep"
        ),
    )
    parser.add_argument("--ble-data", type=Path)
    parser.add_argument(
        "--pf-landmark-mode",
        choices=("none", "observation", "reset", "hybrid"),
        default="none",
    )
    parser.add_argument(
        "--landmark-sigma-m",
        type=float,
        default=PF_LANDMARK_SIGMA_M,
    )
    parser.add_argument("--anchor-beacon-id")
    parser.add_argument("--anchor-position-sigma-m", type=float, default=0.3)
    parser.add_argument("--anchor-heading-deg", type=float)
    parser.add_argument("--anchor-heading-sigma-deg", type=float, default=15.0)
    parser.add_argument("--anchor-bidirectional", action="store_true")
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=Path(
            "input/correct_path/1turn_rightsidestep_3turn_leftsidestep/"
            "walk_trace (3).csv"
        ),
    )
    parser.add_argument("--floormap", type=Path, default=Path(FLOORMAP_PATH))
    parser.add_argument("--origin-px", nargs=2, type=int, default=FLOORMAP_ORIGIN_PX)
    parser.add_argument("--scale", type=float, default=FLOORMAP_SCALE)
    parser.add_argument(
        "--gyro-bias-method",
        choices=(
            "prewalk_guarded",
            "zero",
            "prewalk_robust",
            "initial_robust",
            "quietest",
            "manual",
        ),
        default=GYRO_BIAS_METHOD,
    )
    parser.add_argument("--gyro-bias", type=float)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 10, 42, 100])
    parser.add_argument(
        "--sigma-init-heading", type=float, default=PF_SIGMA_INIT_HEADING
    )
    parser.add_argument("--sigma-heading", type=float, default=PF_SIGMA_HEADING)
    parser.add_argument(
        "--sigma-step-length-ratio",
        type=float,
        default=PF_SIGMA_STEP_LENGTH_RATIO,
    )
    parser.add_argument("--plot-path", type=Path)
    parser.add_argument("--enable-recovery-branches", action="store_true")
    parser.add_argument(
        "--motion-predictive-weight-power",
        type=float,
        default=PF_MOTION_PREDICTIVE_WEIGHT_POWER,
    )
    parser.add_argument(
        "--pf-path-selection",
        choices=("current", "sequence"),
        default=PF_PATH_SELECTION,
    )
    parser.add_argument(
        "--motion-estimation",
        choices=("legacy", "adaptive", "robust"),
        default=MOTION_ESTIMATION,
    )
    parser.add_argument(
        "--smoothing",
        choices=("causal", "offline"),
        default=SMOOTHING_MODE,
    )
    return parser.parse_args()


def main() -> None:
    """複数seedのPF軌跡を正解軌跡と比較する。"""
    args = _parse_args()
    truth_frame = pd.read_csv(args.truth_csv)
    truth = truth_frame[["x", "y"]].to_numpy(dtype=float)
    truth -= truth[0]
    sampled_truth = _sample_by_arclength(truth)
    truth_length = float(np.linalg.norm(np.diff(truth, axis=0), axis=1).sum())
    map_gray = _normalize_floormap_gray(plt.imread(args.floormap))

    df_acc, df_gyro = load_sensor_data(args.data_dir)
    prepared = prepare_pdr_steps(
        df_acc,
        df_gyro,
        gyro_bias_method=args.gyro_bias_method,
        gyro_bias=args.gyro_bias,
        motion_estimation=args.motion_estimation,
        smoothing_mode=args.smoothing,
    )
    landmark_settings = BleLandmarkSettings(
        enabled=args.ble_data is not None,
        data_path="" if args.ble_data is None else args.ble_data,
    )
    if args.anchor_beacon_id is not None:
        if args.anchor_beacon_id not in landmark_settings.landmark_map():
            raise ValueError(f"未定義の anchor beacon_id です: {args.anchor_beacon_id}")
        landmark_settings = replace(
            landmark_settings,
            landmarks=tuple(
                replace(
                    item,
                    position_sigma_m=args.anchor_position_sigma_m,
                    heading_deg=args.anchor_heading_deg,
                    heading_sigma_deg=args.anchor_heading_sigma_deg,
                    heading_bidirectional=args.anchor_bidirectional,
                )
                if item.beacon_id == args.anchor_beacon_id
                else item
                for item in landmark_settings.landmarks
            ),
        )
    detections = run_ble_landmark_detection(landmark_settings) or ()
    rows: list[dict[str, float | int | str]] = []
    trajectories: list[np.ndarray] = []
    for seed in args.seeds:
        diagnostics: list[ParticleFilterStepDiagnostics] = []
        path_comparisons: list[ParticlePathComparison] = []
        result = run_particle_filter(
            prepared.gx_mean,
            prepared.gz_mean,
            floormap_path=args.floormap,
            origin_px=tuple(args.origin_px),
            scale=args.scale,
            prepared_step_headings=prepared.step_headings,
            prepared_step_lengths=prepared.step_lengths,
            prepared_step_times=prepared.t_at_steps,
            prepared_motion_evidences=prepared.motion_evidences,
            prepared_particle_motion_headings=prepared.particle_motion_headings,
            prepared_motion_posteriors=prepared.motion_posteriors,
            preserve_recovery_branches=args.enable_recovery_branches,
            motion_predictive_weight_power=args.motion_predictive_weight_power,
            path_selection=args.pf_path_selection,
            proposal_method=args.pf_proposal,
            candidate_count=args.pf_candidates,
            sigma_init_heading=args.sigma_init_heading,
            sigma_heading=args.sigma_heading,
            sigma_sl_ratio=args.sigma_step_length_ratio,
            seed=seed,
            diagnostics_collector=diagnostics,
            path_comparison_collector=path_comparisons,
            landmark_detections=detections,
            landmarks=landmark_settings.landmarks,
            landmark_mode=args.pf_landmark_mode,
            landmark_sigma_m=args.landmark_sigma_m,
            landmark_likelihood_floor=PF_LANDMARK_LIKELIHOOD_FLOOR,
            landmark_reset_sigma_m=PF_LANDMARK_RESET_SIGMA_M,
            landmark_max_jump_m=PF_LANDMARK_MAX_JUMP_M,
            landmark_reset_spread_ratio=PF_LANDMARK_RESET_SPREAD_RATIO,
            landmark_reset_heading_sigma=PF_LANDMARK_RESET_HEADING_SIGMA,
        )
        trajectory = np.asarray(result[0], dtype=float)
        trajectories.append(trajectory)
        sampled_trajectory = _sample_by_arclength(trajectory)
        errors = np.linalg.norm(sampled_trajectory - sampled_truth, axis=1)
        terminal = evaluate_terminal_direction(sampled_trajectory, sampled_truth)
        valid = _evaluate_particle_transitions(
            trajectory[:-1],
            trajectory[1:],
            map_gray,
            prepared.gx_mean,
            prepared.gz_mean,
            tuple(args.origin_px),
            args.scale,
        )
        allowed_landmark_jump_steps = [
            item.step - 1
            for item in diagnostics
            if item.recovery_mode
            in {"landmark_reset", "landmark_anchor", "landmark_anchor_heading"}
        ]
        if allowed_landmark_jump_steps:
            valid[np.asarray(allowed_landmark_jump_steps, dtype=int)] = True
        reset_steps = [
            item.step - 1
            for item in diagnostics
            if item.recovery_mode == "landmark_reset"
        ]
        anchor_steps = [
            item.step - 1
            for item in diagnostics
            if item.recovery_mode in {"landmark_anchor", "landmark_anchor_heading"}
        ]
        landmark_distances = [
            item.landmark_distance_m
            for item in diagnostics
            if item.landmark_distance_m is not None
        ]
        rows.append(
            {
                "seed": seed,
                "motion_estimation": args.motion_estimation,
                "smoothing": args.smoothing,
                "motion_predictive_weight_power": args.motion_predictive_weight_power,
                "pf_path_selection": args.pf_path_selection,
                "pf_proposal": args.pf_proposal,
                "pf_candidates": args.pf_candidates,
                "pf_landmark_mode": args.pf_landmark_mode,
                "anchor_beacon_id": args.anchor_beacon_id or "none",
                "anchor_heading_deg": (
                    args.anchor_heading_deg
                    if args.anchor_heading_deg is not None
                    else "none"
                ),
                "selected_path_mode": (
                    path_comparisons[0].selected_mode
                    if path_comparisons
                    else args.pf_path_selection
                ),
                "landmark_sigma_m": args.landmark_sigma_m,
                "arc_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
                "endpoint_error_m": float(np.linalg.norm(trajectory[-1] - truth[-1])),
                "terminal_direction_error_deg": terminal.direction_error_deg,
                "terminal_progress_cosine": terminal.progress_cosine,
                "terminal_opposed_fraction": terminal.opposed_fraction,
                "terminal_direction_failure": terminal.failure,
                "estimated_length_m": float(
                    np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum()
                ),
                "truth_length_m": truth_length,
                "wall_crossings": int(np.count_nonzero(~valid)),
                "recovery_failures": sum(
                    item.recovery_mode == "failed_hold" for item in diagnostics
                ),
                "checkpoint_replays": sum(
                    item.recovery_mode == "checkpoint_replay" for item in diagnostics
                ),
                "recoveries": sum(
                    item.recovery_mode not in {"none", "failed_hold"}
                    for item in diagnostics
                ),
                "turn_grid_recoveries": sum(
                    item.recovery_mode == "turn_grid" for item in diagnostics
                ),
                "representative_sidestep_steps": sum(
                    item.representative_motion_state.startswith("sidestep")
                    for item in diagnostics
                ),
                "mean_sidestep_probability": float(
                    np.mean(
                        [
                            item.sidestep_left_state_probability
                            + item.sidestep_right_state_probability
                            for item in diagnostics
                        ]
                    )
                ),
                "representative_turning_steps": sum(
                    item.representative_motion_state == "turning"
                    for item in diagnostics
                ),
                "max_position_spread_m": max(
                    item.position_spread_rms_m for item in diagnostics
                ),
                "mean_landmark_distance_m": (
                    float(np.mean(landmark_distances)) if landmark_distances else 0.0
                ),
                "max_landmark_distance_m": (
                    max(landmark_distances) if landmark_distances else 0.0
                ),
                "landmark_resets": len(reset_steps),
                "landmark_anchors": len(anchor_steps),
            }
        )

    if args.plot_path is not None:
        columns = min(3, len(args.seeds))
        rows_count = int(np.ceil(len(args.seeds) / columns))
        figure, axes = plt.subplots(rows_count, columns, figsize=(15, 5 * rows_count))
        for axis, seed, trajectory, metrics in zip(
            np.atleast_1d(axes).flat,
            args.seeds,
            trajectories,
            rows,
            strict=False,
        ):
            axis.plot(truth[:, 0], truth[:, 1], "--", color="deepskyblue")
            axis.plot(trajectory[:, 0], trajectory[:, 1], color="crimson")
            axis.set_title(f"seed {seed} / RMSE {metrics['arc_rmse_m']:.2f} m")
            axis.set_aspect("equal")
            axis.grid(alpha=0.3)
        figure.tight_layout()
        args.plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot_path, dpi=160)
        plt.close(figure)

    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
