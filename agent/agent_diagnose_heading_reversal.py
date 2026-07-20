"""候補軌跡の角度wrapと実際の進行方向反転を切り分ける診断ツール。

役割:
    複数計測・PDR/PF方式・seedから得た候補軌跡を共通の正規化弧長へ揃え、
    表示上の角度wrap、持続する180度反転、正解旋回方向との不一致を検出する。
入力:
    候補一覧manifest（data、family、method、seed、trajectory_csv）と、任意の
    x/y正解軌跡CSVを受け取る。標準評価では5計測とPFの6seedを検査できる。
出力:
    候補別の集計CSV、区間別イベントCSV、全情報を含むJSONを指定先へ保存する。
処理フロー:
    各軌跡を始点基準に変換して正規化弧長300点へ補間し、方位差をwrapした値と
    生値で比較する。過去方位に対する持続的な逆行区間と、正解の旋回区間で符号が
    逆になる区間を抽出し、候補メタデータとともに集計する。
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rikka.analyze.trajectory_direction import (
    evaluate_terminal_direction,
    sample_trajectory_by_arclength,
)

SAMPLE_COUNT = 300
DEFAULT_SEEDS = (0, 1, 2, 10, 42, 100)


def _parse_args() -> argparse.Namespace:
    """入力manifestと診断閾値を受け取る。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--reversal-angle-deg", type=float, default=135.0)
    parser.add_argument("--reversal-min-points", type=int, default=2)
    parser.add_argument("--turn-threshold-deg", type=float, default=2.0)
    parser.add_argument("--require-standard-coverage", action="store_true")
    parser.add_argument("--expected-data-count", type=int, default=5)
    parser.add_argument(
        "--expected-pf-seeds", nargs="+", type=int, default=DEFAULT_SEEDS
    )
    return parser.parse_args()


def _load_manifest(path: Path) -> pd.DataFrame:
    """CSVまたはJSONの候補一覧を読み、相対パスをmanifest基準で解決する。"""
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw["candidates"] if isinstance(raw, dict) else raw
        frame = pd.DataFrame(rows)
    else:
        frame = pd.read_csv(path)
    required = {"data", "family", "method", "trajectory_csv"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifestに必要な列がありません: {', '.join(missing)}")
    frame = frame.copy()
    if "seed" not in frame:
        frame["seed"] = pd.NA
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


def _validate_standard_coverage(
    frame: pd.DataFrame, expected_data_count: int, expected_seeds: set[int]
) -> None:
    """標準の5反復データとPF方式ごとの6seedが揃っているか確認する。"""
    actual_count = int(frame["data"].nunique())
    if actual_count != expected_data_count:
        raise ValueError(f"data数が{actual_count}です（期待値: {expected_data_count}）")
    pf = frame[frame["family"].astype(str).str.lower() == "pf"]
    if pf.empty:
        raise ValueError("PF候補がありません")
    for (data, method), group in pf.groupby(["data", "method"], sort=False):
        actual = {int(seed) for seed in group["seed"].dropna()}
        if actual != expected_seeds:
            raise ValueError(
                f"{data}/{method}のseedが不完全です: "
                f"actual={sorted(actual)}, expected={sorted(expected_seeds)}"
            )


def _load_trajectory(path: Path) -> np.ndarray:
    """x/y列を有限な2次元軌跡として読み込む。"""
    frame = pd.read_csv(path)
    if not {"x", "y"}.issubset(frame.columns):
        raise ValueError(f"{path}にx/y列がありません")
    points = frame[["x", "y"]].to_numpy(dtype=float)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError(f"{path}は有限な3点以上の軌跡である必要があります")
    return points


def _sample_by_arclength(points: np.ndarray, count: int) -> np.ndarray:
    """始点を原点へ移し、弧長に沿って指定点数へ補間する。"""
    return sample_trajectory_by_arclength(points, count)


def _wrap_angle(values: np.ndarray) -> np.ndarray:
    """角度を[-pi, pi)へ正規化する。"""
    return np.arctan2(np.sin(values), np.cos(values))


def _runs(mask: np.ndarray, minimum: int = 1) -> list[tuple[int, int]]:
    """真偽配列からminimum以上続く閉区間を返す。"""
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if end - start + 1 >= minimum
    ]


def _circular_mean(values: np.ndarray) -> float:
    """角度列の円平均を返す。"""
    return float(np.arctan2(np.sin(values).mean(), np.cos(values).mean()))


def _diagnose_candidate(
    sampled: np.ndarray,
    sampled_truth: np.ndarray | None,
    reversal_angle: float,
    reversal_min_points: int,
    turn_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """1候補のwrap、反転、旋回方向不一致を抽出する。"""
    headings = np.arctan2(np.diff(sampled[:, 1]), np.diff(sampled[:, 0]))
    raw_changes = np.diff(headings)
    wrapped_changes = _wrap_angle(raw_changes)
    wrap_mask = (np.abs(raw_changes) > np.pi) & (np.abs(wrapped_changes) < np.pi / 2)

    reverse_mask = np.zeros(len(headings), dtype=bool)
    reversal_angles = np.zeros(len(headings), dtype=float)
    history = 5
    for index in range(history, len(headings)):
        reference = _circular_mean(headings[index - history : index])
        reversal_angles[index] = abs(float(_wrap_angle(headings[index] - reference)))
        reverse_mask[index] = reversal_angles[index] >= reversal_angle
    reversal_runs = _runs(reverse_mask, reversal_min_points)

    events: list[dict[str, Any]] = []
    for start, end in _runs(wrap_mask):
        events.append(
            {
                "event_type": "angle_wrap",
                "start_index": start + 1,
                "end_index": end + 1,
                "max_angle_deg": float(
                    np.degrees(np.max(np.abs(raw_changes[start : end + 1])))
                ),
            }
        )
    for start, end in reversal_runs:
        events.append(
            {
                "event_type": "sustained_reversal",
                "start_index": start,
                "end_index": end + 1,
                "max_angle_deg": float(
                    np.degrees(np.max(reversal_angles[start : end + 1]))
                ),
            }
        )

    mismatch_runs: list[tuple[int, int]] = []
    turn_samples = 0
    terminal_metrics = None
    if sampled_truth is not None:
        truth_headings = np.arctan2(
            np.diff(sampled_truth[:, 1]), np.diff(sampled_truth[:, 0])
        )
        truth_turn = _wrap_angle(np.diff(truth_headings))
        candidate_turn = wrapped_changes
        truth_turn_mask = np.abs(truth_turn) >= turn_threshold
        mismatch = truth_turn_mask & (np.sign(truth_turn) != np.sign(candidate_turn))
        mismatch_runs = _runs(mismatch)
        turn_samples = int(np.count_nonzero(truth_turn_mask))
        for start, end in mismatch_runs:
            events.append(
                {
                    "event_type": "turn_direction_mismatch",
                    "start_index": start + 1,
                    "end_index": end + 2,
                    "max_angle_deg": float(
                        np.degrees(np.max(np.abs(truth_turn[start : end + 1])))
                    ),
                }
            )
        terminal_metrics = evaluate_terminal_direction(sampled, sampled_truth)
        if terminal_metrics.failure:
            events.append(
                {
                    "event_type": "terminal_direction_failure",
                    "start_index": int(np.floor(0.85 * (len(sampled) - 1))),
                    "end_index": len(sampled) - 1,
                    "max_angle_deg": terminal_metrics.direction_error_deg,
                }
            )

    summary = {
        "sample_count": len(sampled),
        "angle_wrap_count": int(np.count_nonzero(wrap_mask)),
        "sustained_reversal_count": len(reversal_runs),
        "sustained_reversal_points": int(
            sum(end - start + 1 for start, end in reversal_runs)
        ),
        "truth_turn_sample_count": turn_samples if sampled_truth is not None else None,
        "turn_direction_mismatch_count": len(mismatch_runs)
        if sampled_truth is not None
        else None,
        "turn_direction_mismatch_points": int(
            sum(end - start + 1 for start, end in mismatch_runs)
        )
        if sampled_truth is not None
        else None,
        "terminal_direction_error_deg": (
            terminal_metrics.direction_error_deg
            if terminal_metrics is not None
            else None
        ),
        "terminal_progress_cosine": (
            terminal_metrics.progress_cosine if terminal_metrics is not None else None
        ),
        "terminal_opposed_fraction": (
            terminal_metrics.opposed_fraction if terminal_metrics is not None else None
        ),
        "terminal_direction_failure": (
            terminal_metrics.failure if terminal_metrics is not None else None
        ),
    }
    return summary, events


def main() -> None:
    """全候補を診断し、JSONと2種類のCSVを出力する。"""
    args = _parse_args()
    if args.sample_count < 10:
        raise ValueError("sample-countは10以上にしてください")
    manifest = _load_manifest(args.manifest)
    if args.require_standard_coverage:
        _validate_standard_coverage(
            manifest, args.expected_data_count, set(args.expected_pf_seeds)
        )
    sampled_truth = None
    if args.truth_csv is not None:
        sampled_truth = _sample_by_arclength(
            _load_trajectory(args.truth_csv), args.sample_count
        )

    summaries: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for row in manifest.to_dict(orient="records"):
        sampled = _sample_by_arclength(
            _load_trajectory(Path(row["trajectory_csv"])), args.sample_count
        )
        summary, candidate_events = _diagnose_candidate(
            sampled,
            sampled_truth,
            np.radians(args.reversal_angle_deg),
            args.reversal_min_points,
            np.radians(args.turn_threshold_deg),
        )
        metadata = {
            "candidate_id": str(row["candidate_id"]),
            "data": str(row["data"]),
            "family": str(row["family"]),
            "method": str(row["method"]),
            "seed": None if pd.isna(row["seed"]) else int(row["seed"]),
            "trajectory_csv": str(row["trajectory_csv"]),
        }
        summaries.append(metadata | summary)
        events.extend(metadata | event for event in candidate_events)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "reversal_summary.csv"
    events_path = args.output_dir / "reversal_events.csv"
    json_path = args.output_dir / "reversal_diagnostics.json"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    event_columns = [
        "candidate_id",
        "data",
        "family",
        "method",
        "seed",
        "trajectory_csv",
        "event_type",
        "start_index",
        "end_index",
        "max_angle_deg",
    ]
    pd.DataFrame(events, columns=event_columns).to_csv(events_path, index=False)
    payload = {
        "settings": {
            "sample_count": args.sample_count,
            "reversal_angle_deg": args.reversal_angle_deg,
            "reversal_min_points": args.reversal_min_points,
            "turn_threshold_deg": args.turn_threshold_deg,
        },
        "candidates": summaries,
        "events": events,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json_path)
    print(summary_path)
    print(events_path)


if __name__ == "__main__":
    main()
