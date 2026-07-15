"""PF軌跡を同期していない正解軌跡と形状比較する検証スクリプト。

役割:
    指定したセンサーデータを複数seedでPF実行し、開始点を合わせた正解軌跡と
    正規化弧長で比較する。
入力:
    ``--data-dir``のAccelerometer/Gyroscope CSV、``--truth-csv``のx/y列、
    フロアマップ、origin、scaleを使用する。
出力:
    seedごとのRMSE、終点誤差、壁交差、recovery結果をJSONで標準出力し、
    ``--plot-path``指定時は正解軌跡とPF軌跡の比較画像を保存する。
処理フロー:
    共通PDRステップを準備し、seedごとにPF実行、弧長正規化、地図遷移検査、
    指標集計、任意の重ね描きを行う。
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rikka.analyze.particle_filter import (
    ParticleFilterStepDiagnostics,
    _evaluate_particle_transitions,
    _normalize_floormap_gray,
    run_particle_filter,
)
from rikka.analyze.pdr.sensors import load_sensor_data
from rikka.analyze.pdr.trajectory import prepare_pdr_steps
from rikka.config import (
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    PF_SIGMA_HEADING,
    PF_SIGMA_INIT_HEADING,
    PF_SIGMA_STEP_LENGTH_RATIO,
)


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
        "--data-dir",
        type=Path,
        default=Path("input/sensor_data/1turn_rightsidestep_3turn_leftsidestep"),
    )
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
    prepared = prepare_pdr_steps(df_acc, df_gyro)
    rows: list[dict[str, float | int]] = []
    trajectories: list[np.ndarray] = []
    for seed in args.seeds:
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
            sigma_init_heading=args.sigma_init_heading,
            sigma_heading=args.sigma_heading,
            sigma_sl_ratio=args.sigma_step_length_ratio,
            seed=seed,
            diagnostics_collector=diagnostics,
        )
        trajectory = np.asarray(result[0], dtype=float)
        trajectories.append(trajectory)
        sampled_trajectory = _sample_by_arclength(trajectory)
        errors = np.linalg.norm(sampled_trajectory - sampled_truth, axis=1)
        valid = _evaluate_particle_transitions(
            trajectory[:-1],
            trajectory[1:],
            map_gray,
            prepared.gx_mean,
            prepared.gz_mean,
            tuple(args.origin_px),
            args.scale,
        )
        rows.append(
            {
                "seed": seed,
                "arc_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
                "endpoint_error_m": float(np.linalg.norm(trajectory[-1] - truth[-1])),
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
