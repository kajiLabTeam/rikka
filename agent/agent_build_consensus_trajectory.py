"""複数計測・複数方式の候補から完全な代表軌跡を選ぶツール。

役割:
    PDR/PFの候補群を比較し、各計測の寄与を等しくした重み付きmedoidにより、
    座標平均ではなく実在する合法な完全軌跡を代表として選択する。
入力:
    data、family、method、seed、trajectory_csvを持つmanifest、任意の診断JSON、
    評価専用の正解軌跡CSVを受け取る。weightや地図・recovery指標も利用できる。
出力:
    選択元を保持したbest_trajectory.csv、選択根拠のreport.json、候補比較画像を
    指定output-dirへ保存する。
処理フロー:
    不適格候補を除き、各候補を正規化弧長300点へ揃える。計測ごとの総重みを1に
    正規化して全候補への重み付き距離和が最小の候補を選び、正解は選択後の評価と
    可視化にだけ使用する。
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rikka.analyze.trajectory_direction import (
    sample_trajectory_by_arclength,
    terminal_consensus_outliers,
)

SAMPLE_COUNT = 300


def _parse_args() -> argparse.Namespace:
    """候補manifest、除外条件、出力先を受け取る。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--diagnostics-json", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument(
        "--exclude-reversals", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--exclude-terminal-outliers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _load_manifest(path: Path) -> pd.DataFrame:
    """候補一覧を読み、相対軌跡パスと省略可能な評価列を補う。"""
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(raw["candidates"] if isinstance(raw, dict) else raw)
    else:
        frame = pd.read_csv(path)
    required = {"data", "family", "method", "trajectory_csv"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifestに必要な列がありません: {', '.join(missing)}")
    frame = frame.copy()
    defaults: dict[str, Any] = {
        "seed": pd.NA,
        "weight": 1.0,
        "eligible": True,
        "wall_crossings": 0,
        "recovery_failures": 0,
    }
    for column, value in defaults.items():
        if column not in frame:
            frame[column] = value
    if "candidate_id" not in frame:
        frame["candidate_id"] = [
            f"{row.data}:{row.family}:{row.method}:"
            f"{int(row.seed) if pd.notna(row.seed) else 'none'}"
            for row in frame.itertuples(index=False)
        ]
    if frame["candidate_id"].duplicated().any():
        raise ValueError("candidate_idは一意である必要があります")
    base = path.resolve().parent
    frame["trajectory_csv"] = frame["trajectory_csv"].map(
        lambda value: str(
            (base / Path(str(value))).resolve()
            if not Path(str(value)).is_absolute()
            else Path(str(value)).resolve()
        )
    )
    return frame


def _load_trajectory(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """元CSVと有限なx/y座標を読み込む。"""
    frame = pd.read_csv(path)
    if not {"x", "y"}.issubset(frame.columns):
        raise ValueError(f"{path}にx/y列がありません")
    points = frame[["x", "y"]].to_numpy(dtype=float)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError(f"{path}は有限な3点以上の軌跡である必要があります")
    return frame, points


def _sample_by_arclength(points: np.ndarray, count: int) -> np.ndarray:
    """軌跡を始点基準の正規化弧長上へ補間する。"""
    return sample_trajectory_by_arclength(points, count)


def _terminal_outlier_indices(
    candidates: list[dict[str, Any]], sampled: np.ndarray
) -> tuple[set[int], float, np.ndarray]:
    """複数計測の多数方向と90度以上異なる候補indexを返す。"""
    return terminal_consensus_outliers(
        sampled,
        [str(item["data"]) for item in candidates],
        [str(item["family"]) for item in candidates],
    )


def _as_bool(value: Any) -> bool:
    """CSV由来の真偽値を安全に解釈する。"""
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "off"}
    return bool(value)


def _diagnostic_reversals(path: Path | None) -> dict[str, int]:
    """診断JSONから候補別の持続反転数を取得する。"""
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["candidate_id"]): int(item["sustained_reversal_count"])
        for item in payload.get("candidates", [])
    }


def _eligibility_reason(
    row: dict[str, Any], reversals: dict[str, int], exclude_reversals: bool
) -> str | None:
    """候補を除外する最初の理由を返す。"""
    if not _as_bool(row["eligible"]):
        return "manifest_ineligible"
    wall_crossings = 0 if pd.isna(row["wall_crossings"]) else int(row["wall_crossings"])
    recovery_failures = (
        0 if pd.isna(row["recovery_failures"]) else int(row["recovery_failures"])
    )
    if wall_crossings > 0:
        return "wall_crossing"
    if recovery_failures > 0:
        return "recovery_failure"
    if exclude_reversals and reversals.get(str(row["candidate_id"]), 0) > 0:
        return "sustained_reversal"
    weight = float(row["weight"])
    if not np.isfinite(weight) or weight <= 0:
        return "invalid_weight"
    return None


def _select_medoid(
    candidates: list[dict[str, Any]], sampled: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray]:
    """計測ごとの総重みを等しくして重み付きmedoidを選ぶ。"""
    distances = np.sqrt(
        np.mean(
            np.sum(np.square(sampled[:, np.newaxis] - sampled[np.newaxis, :]), axis=3),
            axis=2,
        )
    )
    data_names = sorted({str(item["data"]) for item in candidates})
    weights = np.zeros(len(candidates), dtype=float)
    for data_name in data_names:
        indices = [
            index
            for index, item in enumerate(candidates)
            if str(item["data"]) == data_name
        ]
        raw = np.asarray([float(candidates[index]["weight"]) for index in indices])
        weights[indices] = raw / raw.sum() / len(data_names)
    scores = distances @ weights
    return int(np.argmin(scores)), scores, weights


def _truth_metrics(sampled: np.ndarray, sampled_truth: np.ndarray) -> dict[str, float]:
    """選択後の正規化弧長RMSEと終点誤差を返す。"""
    errors = np.linalg.norm(sampled - sampled_truth, axis=1)
    return {
        "arc_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "endpoint_error_m": float(np.linalg.norm(sampled[-1] - sampled_truth[-1])),
    }


def main() -> None:
    """候補群から代表軌跡を選び、CSV・JSON・比較画像を保存する。"""
    args = _parse_args()
    if args.sample_count < 10:
        raise ValueError("sample-countは10以上にしてください")
    manifest = _load_manifest(args.manifest)
    reversals = _diagnostic_reversals(args.diagnostics_json)

    excluded: list[dict[str, str]] = []
    candidates: list[dict[str, Any]] = []
    trajectories: list[pd.DataFrame] = []
    sampled_list: list[np.ndarray] = []
    for row in manifest.to_dict(orient="records"):
        reason = _eligibility_reason(row, reversals, args.exclude_reversals)
        if reason is not None:
            excluded.append(
                {"candidate_id": str(row["candidate_id"]), "reason": reason}
            )
            continue
        trajectory_frame, points = _load_trajectory(Path(row["trajectory_csv"]))
        candidates.append(row)
        trajectories.append(trajectory_frame)
        sampled_list.append(_sample_by_arclength(points, args.sample_count))
    if not candidates:
        raise ValueError("採用可能な候補軌跡がありません")

    sampled = np.stack(sampled_list)
    terminal_consensus = None
    terminal_errors = np.zeros(len(candidates), dtype=float)
    if (
        args.exclude_terminal_outliers
        and len({str(x["data"]) for x in candidates}) >= 3
    ):
        outliers, terminal_consensus, terminal_errors = _terminal_outlier_indices(
            candidates, sampled
        )
        if outliers and len(outliers) < len(candidates):
            keep = [index for index in range(len(candidates)) if index not in outliers]
            excluded.extend(
                {
                    "candidate_id": str(candidates[index]["candidate_id"]),
                    "reason": "terminal_direction_outlier",
                }
                for index in sorted(outliers)
            )
            candidates = [candidates[index] for index in keep]
            trajectories = [trajectories[index] for index in keep]
            sampled = sampled[keep]
            terminal_errors = terminal_errors[keep]
    selected_index, scores, normalized_weights = _select_medoid(candidates, sampled)
    selected = candidates[selected_index]
    ranking = sorted(
        [
            {
                "candidate_id": str(item["candidate_id"]),
                "data": str(item["data"]),
                "family": str(item["family"]),
                "method": str(item["method"]),
                "seed": None if pd.isna(item["seed"]) else int(item["seed"]),
                "normalized_weight": float(normalized_weights[index]),
                "medoid_score_m": float(scores[index]),
                "terminal_consensus_error_deg": float(terminal_errors[index]),
            }
            for index, item in enumerate(candidates)
        ],
        key=lambda item: item["medoid_score_m"],
    )

    truth_metrics = None
    sampled_truth = None
    if args.truth_csv is not None:
        _, truth_points = _load_trajectory(args.truth_csv)
        sampled_truth = _sample_by_arclength(truth_points, args.sample_count)
        truth_metrics = _truth_metrics(sampled[selected_index], sampled_truth)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_trajectory.csv"
    report_path = args.output_dir / "report.json"
    plot_path = args.output_dir / "comparison.png"
    trajectories[selected_index].to_csv(best_path, index=False)
    report = {
        "selected": ranking[
            next(
                index
                for index, item in enumerate(ranking)
                if item["candidate_id"] == str(selected["candidate_id"])
            )
        ],
        "truth_metrics": truth_metrics,
        "candidate_count": len(candidates),
        "data_count": len({str(item["data"]) for item in candidates}),
        "sample_count": args.sample_count,
        "truth_used_for_selection": False,
        "terminal_consensus_heading_deg": (
            None
            if terminal_consensus is None
            else float(np.degrees(terminal_consensus))
        ),
        "ranking": ranking,
        "excluded": excluded,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(8, 6))
    for index, points in enumerate(sampled):
        axis.plot(
            points[:, 0],
            points[:, 1],
            color="0.75",
            alpha=0.35,
            linewidth=0.8,
            label="Candidates" if index == 0 else None,
        )
    axis.plot(
        sampled[selected_index, :, 0],
        sampled[selected_index, :, 1],
        color="crimson",
        linewidth=2.5,
        label="Selected trajectory",
    )
    if sampled_truth is not None:
        axis.plot(
            sampled_truth[:, 0],
            sampled_truth[:, 1],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Ground truth (evaluation only)",
        )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    print(best_path)
    print(report_path)
    print(plot_path)


if __name__ == "__main__":
    main()
