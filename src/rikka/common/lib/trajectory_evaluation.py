"""推定軌跡と正解軌跡を比較する評価処理。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrajectoryEvaluationSummary:
    """軌跡評価の集計結果。"""

    schema_version: str
    alignment: str
    coordinate_system: str
    floor_scale: float | None
    sample_count: int
    candidate_count: int
    evaluated_count: int
    point_coverage_rate: float
    mse_m2: float
    rmse_m: float
    mae_m: float
    p95_m: float
    max_error_m: float
    within_tolerance_rate: float
    tolerance_m: float


REQUIRED_COLUMNS = {"timestamp_s", "x", "y"}
EVALUATION_COLUMNS = [
    "step_index",
    "timestamp_s",
    "x",
    "y",
    "ground_truth_x",
    "ground_truth_y",
    "dx_m",
    "dy_m",
    "distance_error_m",
    "squared_error_m2",
    "evaluated",
]


def _read_trajectory_csv(path: str | Path, *, estimated: bool) -> pd.DataFrame:
    """軌跡CSVを読み込み、数値列を検証する。"""
    label = "推定軌跡" if estimated else "正解軌跡"
    try:
        dataframe = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"{label}CSVを読み込めません: {path}") from exc

    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"{label}CSVに必須列がありません: {missing_columns}")
    if dataframe.empty:
        raise ValueError(f"{label}CSVに軌跡点がありません")

    for column in ("x", "y"):
        values = pd.to_numeric(dataframe[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{label}CSVの{column}列には有限な数値が必要です")
        dataframe[column] = values

    timestamps = pd.to_numeric(dataframe["timestamp_s"], errors="coerce")
    if estimated:
        # 既存のrikka出力では先頭の初期位置だけ時刻が空になる。
        if timestamps.iloc[0] is not pd.NA and not pd.isna(timestamps.iloc[0]):
            raise ValueError("推定軌跡CSVの先頭timestamp_sは空である必要があります")
        timestamps = timestamps.iloc[1:]
    elif timestamps.isna().any():
        raise ValueError("正解軌跡CSVのtimestamp_sには空欄を指定できません")

    if not np.isfinite(timestamps.to_numpy(dtype=float)).all():
        raise ValueError(f"{label}CSVのtimestamp_sには有限な数値が必要です")
    if (timestamps < 0).any():
        raise ValueError(f"{label}CSVのtimestamp_sには0以上の値が必要です")
    if len(timestamps) > 1 and not (np.diff(timestamps.to_numpy()) > 0).all():
        raise ValueError(f"{label}CSVのtimestamp_sは単調増加である必要があります")
    return dataframe


def evaluate_trajectory_csv(
    estimated_csv: str | Path,
    truth_csv: str | Path,
    *,
    coordinate_system: str = "pixel",
    floor_scale: float | None = None,
    tolerance_m: float = 1.0,
) -> tuple[pd.DataFrame, TrajectoryEvaluationSummary]:
    """推定軌跡を正解軌跡と時刻で比較する。"""
    if coordinate_system not in {"pixel", "meter"}:
        raise ValueError("coordinate_systemはpixelまたはmeterを指定してください")
    if coordinate_system == "pixel" and (
        floor_scale is None or not np.isfinite(floor_scale) or floor_scale <= 0
    ):
        raise ValueError("pixel座標ではfloor_scaleに正の値が必要です")
    if tolerance_m < 0 or not np.isfinite(tolerance_m):
        raise ValueError("tolerance_mには0以上の有限値が必要です")

    estimated = _read_trajectory_csv(estimated_csv, estimated=True)
    truth = _read_trajectory_csv(truth_csv, estimated=False)
    estimated_times = pd.to_numeric(estimated["timestamp_s"], errors="coerce")
    truth_times = truth["timestamp_s"].to_numpy(dtype=float)
    truth_x = truth["x"].to_numpy(dtype=float)
    truth_y = truth["y"].to_numpy(dtype=float)

    evaluated = estimated_times.notna().to_numpy().copy()
    evaluated &= estimated_times.to_numpy(dtype=float) >= truth_times[0]
    evaluated &= estimated_times.to_numpy(dtype=float) <= truth_times[-1]
    evaluation = pd.DataFrame(
        {
            "step_index": estimated.get(
                "step_index", pd.Series(range(len(estimated)), index=estimated.index)
            ),
            "timestamp_s": estimated["timestamp_s"],
            "x": estimated["x"],
            "y": estimated["y"],
            "ground_truth_x": np.nan,
            "ground_truth_y": np.nan,
            "dx_m": np.nan,
            "dy_m": np.nan,
            "distance_error_m": np.nan,
            "squared_error_m2": np.nan,
            "evaluated": evaluated,
        }
    )

    # 初期位置（先頭のtimestamp_sが空の行）は評価候補から除外する。
    candidate_count = int(estimated_times.notna().sum())
    evaluated_count = int(evaluated.sum())
    if evaluated_count:
        comparison_times = estimated_times.to_numpy(dtype=float)[evaluated]
        truth_at_estimated = np.column_stack(
            (
                np.interp(comparison_times, truth_times, truth_x),
                np.interp(comparison_times, truth_times, truth_y),
            )
        )
        estimated_xy = estimated.loc[evaluated, ["x", "y"]].to_numpy(dtype=float)
        delta = estimated_xy - truth_at_estimated
        scale = floor_scale if coordinate_system == "pixel" else 1.0
        # floor_scaleは1pxあたりのメートル数なので、pixel差分に掛ける。
        error_xy_m = delta * float(scale)
        distances = np.linalg.norm(error_xy_m, axis=1)
        squared = distances**2
        positions = np.flatnonzero(evaluated)
        evaluation.loc[positions, "ground_truth_x"] = truth_at_estimated[:, 0]
        evaluation.loc[positions, "ground_truth_y"] = truth_at_estimated[:, 1]
        evaluation.loc[positions, "dx_m"] = error_xy_m[:, 0]
        evaluation.loc[positions, "dy_m"] = error_xy_m[:, 1]
        evaluation.loc[positions, "distance_error_m"] = distances
        evaluation.loc[positions, "squared_error_m2"] = squared
        mse = float(np.mean(squared))
        summary = TrajectoryEvaluationSummary(
            schema_version="1",
            alignment="timestamp_linear_interpolation",
            coordinate_system=coordinate_system,
            floor_scale=floor_scale,
            sample_count=len(estimated),
            candidate_count=candidate_count,
            evaluated_count=evaluated_count,
            point_coverage_rate=evaluated_count / candidate_count * 100,
            mse_m2=mse,
            rmse_m=float(np.sqrt(mse)),
            mae_m=float(np.mean(distances)),
            p95_m=float(np.percentile(distances, 95)),
            max_error_m=float(np.max(distances)),
            within_tolerance_rate=float(np.mean(distances <= tolerance_m) * 100),
            tolerance_m=tolerance_m,
        )
    else:
        raise ValueError("正解軌跡と時刻が重なる推定点がありません")

    return evaluation[EVALUATION_COLUMNS], summary


def write_evaluation_outputs(
    evaluation: pd.DataFrame,
    summary: TrajectoryEvaluationSummary,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """評価CSVと集計JSONを書き出す。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "evaluation.csv"
    json_path = directory / "evaluation.json"
    evaluation.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path
