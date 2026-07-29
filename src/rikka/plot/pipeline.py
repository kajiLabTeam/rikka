"""解析結果からCSV・図一式を書き出す pipeline。

役割:
    PDR/PF の共通結果を受け、成果物の整形・保存・描画を順に実行する。
依存元:
    ``common`` の共有型と設定、同領域 ``lib`` の出力・描画部品を使用する。
利用先:
    CLI が PDR または particle pipeline の完了後に呼び出す。
処理フロー:
    結果概要を表示し、共通CSVを保存した後、要求された場合だけ軌跡図を描画する。
"""

from pathlib import Path

import pandas as pd

from ..common.lib.models import TrajectoryResult
from ..common.settings import ParticleSettings
from .lib.console import print_trajectory_summary
from .lib.outputs import (
    _build_step_headings_dataframe,
    _build_step_vectors_dataframe,
    _build_trajectory_dataframe,
)
from .lib.sensor import plot_sensor_data
from .lib.trajectory import plot_trajectory


def write_outputs(result: TrajectoryResult, output_dir: Path) -> None:
    """通常PDR/PFで共通する4種類のCSVを書き出す。"""
    print_trajectory_summary(result)
    _build_trajectory_dataframe(result.trajectory, result.t_at_steps).to_csv(
        output_dir / "trajectory.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "step": range(1, len(result.step_lengths) + 1),
            "step_length_m": result.step_lengths,
        }
    ).to_csv(output_dir / "step_lengths.csv", index=False)
    _build_step_vectors_dataframe(result.trajectory).to_csv(
        output_dir / "step_vectors.csv",
        index=False,
    )
    _build_step_headings_dataframe(result.step_headings).to_csv(
        output_dir / "step_headings.csv",
        index=False,
    )


def render(
    result: TrajectoryResult,
    settings: ParticleSettings,
    output_dir: Path,
) -> None:
    """解析軌跡をフロアマップ上へ描画する。"""
    plot_trajectory(
        result.trajectory,
        gx_mean=result.prepared.gx_mean,
        gz_mean=result.prepared.gz_mean,
        floormap_path=settings.floormap_path,
        origin_px=settings.origin_px,
        scale=settings.scale,
        output_dir=output_dir,
        step_headings=result.step_headings,
    )


def render_sensor(
    data_dir: str | Path,
    step_detection_method: str | None = None,
    gyro_bias_method: str | None = None,
    gyro_bias: float | None = None,
) -> None:
    """sensor コマンドの描画処理を実行する。"""
    plot_sensor_data(
        data_dir,
        step_detection_method=step_detection_method,
        gyro_bias_method=gyro_bias_method,
        gyro_bias=gyro_bias,
    )
