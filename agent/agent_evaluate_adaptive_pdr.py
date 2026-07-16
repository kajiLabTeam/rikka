"""通常PDRのlegacy/adaptive軌跡を共通正解ルートと比較する検証スクリプト。

役割:
    同じルートを反復計測した複数センサーデータについて、運動推定方式と平滑化方式を
    揃えて実行し、正規化弧長上の形状誤差と推定距離を比較する。
入力:
    ``--data-dir`` の各phyphox CSVと ``--truth-csv`` のx/y正解軌跡を使用する。
出力:
    データ・方式ごとのRMSE、終点誤差、推定距離をJSONで標準出力し、
    ``--plot-path`` 指定時は方式別の軌跡比較画像を保存する。
処理フロー:
    センサー読み込み、共通PDR準備、正規化弧長補間、正解との誤差集計、任意の
    比較描画の順に実行する。
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rikka.analyze.pdr.sensors import load_sensor_data
from rikka.analyze.pdr.trajectory import prepare_pdr_steps


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
    """評価対象と出力先を受け取る。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--truth-csv",
        type=Path,
        default=Path(
            "input/correct_path/1turn_rightsidestep_3turn_leftsidestep/"
            "walk_trace (3).csv"
        ),
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=("legacy", "adaptive-causal", "adaptive-offline"),
        default=("legacy", "adaptive-causal", "adaptive-offline"),
    )
    parser.add_argument("--plot-path", type=Path)
    return parser.parse_args()


def main() -> None:
    """指定データを全方式で評価する。"""
    args = _parse_args()
    truth_frame = pd.read_csv(args.truth_csv)
    truth = truth_frame[["x", "y"]].to_numpy(dtype=float)
    truth -= truth[0]
    sampled_truth = _sample_by_arclength(truth)
    rows: list[dict[str, float | int | str]] = []
    trajectories: list[tuple[str, str, np.ndarray]] = []
    for data_dir in args.data_dir:
        df_acc, df_gyro = load_sensor_data(data_dir)
        for mode in args.mode:
            adaptive = mode.startswith("adaptive")
            smoothing = "offline" if mode.endswith("offline") else "causal"
            prepared = prepare_pdr_steps(
                df_acc,
                df_gyro,
                motion_estimation="adaptive" if adaptive else "legacy",
                smoothing_mode=smoothing,
            )
            trajectory = np.asarray(prepared.trajectory, dtype=float)
            sampled = _sample_by_arclength(trajectory)
            errors = np.linalg.norm(sampled - sampled_truth, axis=1)
            rows.append(
                {
                    "data": data_dir.name,
                    "mode": mode,
                    "steps": len(prepared.step_lengths),
                    "arc_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
                    "endpoint_error_m": float(
                        np.linalg.norm(trajectory[-1] - truth[-1])
                    ),
                    "estimated_length_m": float(
                        np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum()
                    ),
                    "mean_length_std_m": float(
                        np.mean(
                            [
                                posterior.length_std_m
                                for posterior in prepared.motion_posteriors
                            ]
                        )
                    )
                    if prepared.motion_posteriors
                    else 0.0,
                }
            )
            trajectories.append((data_dir.name, mode, trajectory))

    if args.plot_path is not None:
        figure, axes = plt.subplots(
            len(args.data_dir),
            len(args.mode),
            figsize=(5 * len(args.mode), 5 * len(args.data_dir)),
            squeeze=False,
        )
        for axis, (data_name, mode, trajectory), metrics in zip(
            axes.flat,
            trajectories,
            rows,
            strict=True,
        ):
            axis.plot(truth[:, 0], truth[:, 1], "--", color="deepskyblue")
            axis.plot(trajectory[:, 0], trajectory[:, 1], color="crimson")
            axis.set_title(f"{data_name}\n{mode} / RMSE {metrics['arc_rmse_m']:.2f} m")
            axis.set_aspect("equal")
            axis.grid(alpha=0.3)
        figure.tight_layout()
        args.plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot_path, dpi=160)
        plt.close(figure)

    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
