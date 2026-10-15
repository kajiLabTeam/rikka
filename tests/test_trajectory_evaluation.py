from pathlib import Path

import pytest

from rikka.common.lib.trajectory_evaluation import (
    evaluate_trajectory_csv,
    write_evaluation_failure,
)


def _write_csv(path: Path, rows: str) -> Path:
    path.write_text(rows, encoding="utf-8")
    return path


def test_evaluate_csv_ignores_initial_position_and_interpolates_truth(
    tmp_path: Path,
) -> None:
    estimated = _write_csv(
        tmp_path / "estimated.csv",
        "step_index,timestamp_s,x,y\n0,,100,200\n1,1,110,200\n2,2,120,200\n",
    )
    truth = _write_csv(
        tmp_path / "truth.csv",
        "timestamp_s,x,y\n0,100,200\n2,100,220\n",
    )

    evaluation, summary = evaluate_trajectory_csv(
        estimated,
        truth,
        coordinate_system="pixel",
        floor_scale=0.1,
    )

    assert not evaluation.loc[0, "evaluated"]
    assert evaluation.loc[1, "ground_truth_x"] == pytest.approx(100.0)
    assert evaluation.loc[1, "distance_error_m"] == pytest.approx(2**0.5)
    assert evaluation.loc[2, "distance_error_m"] == pytest.approx(2**1.5)
    assert summary.candidate_count == 2
    assert summary.evaluated_count == 2
    assert summary.point_coverage_rate == pytest.approx(100.0)
    assert summary.rmse_m == pytest.approx(5**0.5)


def test_evaluate_meter_coordinates_and_excludes_out_of_range_points(
    tmp_path: Path,
) -> None:
    estimated = _write_csv(
        tmp_path / "estimated.csv",
        "timestamp_s,x,y\n,0,0\n1,1,0\n3,3,0\n",
    )
    truth = _write_csv(
        tmp_path / "truth.csv",
        "timestamp_s,x,y\n0,0,0\n2,0,0\n",
    )

    _, summary = evaluate_trajectory_csv(
        estimated,
        truth,
        coordinate_system="meter",
    )

    assert summary.candidate_count == 2
    assert summary.evaluated_count == 1
    assert summary.point_coverage_rate == pytest.approx(50.0)
    assert summary.mae_m == pytest.approx(1.0)


def test_evaluate_rejects_truth_without_timestamp(tmp_path: Path) -> None:
    estimated = _write_csv(
        tmp_path / "estimated.csv",
        "timestamp_s,x,y\n,0,0\n1,1,1\n",
    )
    truth = _write_csv(
        tmp_path / "truth.csv",
        "timestamp_s,x,y\n,0,0\n1,1,1\n",
    )

    with pytest.raises(ValueError, match="正解軌跡CSVのtimestamp_sには空欄"):
        evaluate_trajectory_csv(estimated, truth, coordinate_system="meter")


def test_write_evaluation_failure_creates_failure_summary(tmp_path: Path) -> None:
    path = write_evaluation_failure(tmp_path, "正解軌跡と時刻が重なりません")

    assert path.read_text(encoding="utf-8") == (
        '{\n  "schema_version": "1",\n  "status": "failed",\n'
        '  "error": "正解軌跡と時刻が重なりません"\n}\n'
    )
