"""解析結果からCSV・図一式を書き出す pipeline。

役割:
    PDR/PF の共通結果を受け、成果物の整形・保存・描画を一元化する。
依存元:
    ``common`` の共有型と設定、同領域 ``lib`` の出力・描画部品を使用する。
利用先:
    CLI が PDR または particle pipeline の完了後に必ず呼び出す。
処理フロー:
    出力先を作成し、共通・診断CSV、任意の図、PF animation の順に保存する。
"""

from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.config import SAMPLING_RATE
from ..common.lib.models import TrajectoryResult
from ..common.settings import OutputSettings, ParticleSettings, PdrSettings
from ..particle.lib.recorder import ParticleFilterStepDiagnostics
from .lib.animation import plot_particle_filter_trajectory, save_particle_animation
from .lib.console import print_trajectory_summary
from .lib.frames import (
    generated_files_size,
    save_particle_path_comparison,
    save_particle_step_frames,
)
from .lib.outputs import (
    _build_direction_posteriors_dataframe,
    _build_gyro_bias_dataframe,
    _build_landmark_corrections_dataframe,
    _build_motion_posteriors_dataframe,
    _build_step_headings_dataframe,
    _build_step_length_observations_dataframe,
    _build_step_segments_dataframe,
    _build_step_vectors_dataframe,
    _build_trajectory_dataframe,
    _create_output_dir,
    _step_plot_signal,
)
from .lib.sensor import plot_sensor_data, plot_step_lengths, plot_step_vectors
from .lib.trajectory import plot_trajectory


def create_output_dir() -> Path:
    """時刻付きの成果物ディレクトリを作成する。"""
    return _create_output_dir()


def write_outputs(result: TrajectoryResult, output_dir: Path) -> pd.DataFrame:
    """PDR/PFの共通CSVと方式固有の診断CSVを書き出す。"""
    print_trajectory_summary(result)
    prepared = result.prepared

    dataframe = _build_trajectory_dataframe(result.trajectory, result.t_at_steps)
    _write_csv(dataframe, output_dir / "trajectory.csv", "Trajectory")
    _write_csv(
        pd.DataFrame(
            {
                "step": range(1, len(result.step_lengths) + 1),
                "step_length_m": result.step_lengths,
            }
        ),
        output_dir / "step_lengths.csv",
        "Step lengths",
    )
    _write_csv(
        _build_step_vectors_dataframe(result.trajectory),
        output_dir / "step_vectors.csv",
        "Step vectors",
    )
    _write_csv(
        _build_step_headings_dataframe(result.step_headings),
        output_dir / "step_headings.csv",
        "Step headings",
    )
    _write_csv(
        _build_gyro_bias_dataframe(prepared.df_gyro),
        output_dir / "gyro_bias.csv",
        "Gyro bias",
    )
    _write_csv(
        _build_step_length_observations_dataframe(prepared.length_observations),
        output_dir / "step_length_observations.csv",
        "Step length observations",
    )
    if prepared.motion_posteriors:
        _write_csv(
            _build_motion_posteriors_dataframe(prepared.motion_posteriors),
            output_dir / "motion_posteriors.csv",
            "Motion posteriors",
        )
    if prepared.direction_posteriors:
        _write_csv(
            _build_direction_posteriors_dataframe(prepared.direction_posteriors),
            output_dir / "direction_posteriors.csv",
            "Direction posteriors",
        )
    if prepared.step_detection.method == "paper_vertical_threshold":
        _write_csv(
            _build_step_segments_dataframe(
                prepared.df_acc,
                prepared.step_detection.segments,
            ),
            output_dir / "step_segments.csv",
            "Step segments",
        )
    if result.landmark is not None:
        _write_csv(
            _build_landmark_corrections_dataframe(result.landmark),
            output_dir / "landmark_corrections.csv",
            "Landmark corrections",
        )
    if result.particle is not None:
        columns = [field.name for field in fields(ParticleFilterStepDiagnostics)]
        _write_csv(
            pd.DataFrame(
                [asdict(item) for item in result.particle.diagnostics],
                columns=columns,
            ),
            output_dir / "particle_diagnostics.csv",
            "Particle diagnostics",
        )
    return dataframe


def _write_csv(dataframe: pd.DataFrame, path: Path, label: str) -> None:
    """CSV保存と従来形式の完了表示を共通化する。"""
    dataframe.to_csv(path, index=False)
    print(f"{label} saved to {path}")


def render(
    result: TrajectoryResult,
    pdr_settings: PdrSettings,
    particle_settings: ParticleSettings,
    output_settings: OutputSettings,
    output_dir: Path,
) -> None:
    """要求された軌跡図、センサー図、PF診断画像を描画する。"""
    prepared = result.prepared
    if output_settings.plot:
        _render_trajectory(result, particle_settings, output_dir)
        t_acc = (
            prepared.df_acc["t"].to_numpy()
            if "t" in prepared.df_acc.columns
            else np.arange(len(prepared.df_acc)) / SAMPLING_RATE
        )
        step_signal, signal_label, signal_threshold = _step_plot_signal(
            prepared.df_acc,
            prepared.step_detection,
        )
        plot_step_lengths(
            result.step_lengths,
            output_dir,
            t_at_steps=result.t_at_steps,
            t_acc=t_acc,
            step_signal=step_signal,
            step_signal_label=signal_label,
            step_signal_threshold=signal_threshold,
        )
        plot_step_vectors(
            result.trajectory,
            output_dir,
            df_acc=prepared.df_acc,
            df_gyro=prepared.df_gyro,
            peaks=prepared.step_detection.peaks,
            step_headings=result.step_headings,
            initial_direction=pdr_settings.heading.initial_direction,
        )

    if result.particle is None:
        return
    _render_particle_artifacts(
        result,
        particle_settings,
        output_settings,
        output_dir,
    )


def _render_trajectory(
    result: TrajectoryResult,
    settings: ParticleSettings,
    output_dir: Path,
) -> None:
    """通常PDRとPFで共通の引数を使って軌跡図を描画する。"""
    if result.particle is not None:
        plot_particle_filter_trajectory(
            result.trajectory,
            gx_mean=result.prepared.gx_mean,
            gz_mean=result.prepared.gz_mean,
            floormap_path=settings.floormap_path,
            origin_px=settings.origin_px,
            scale=settings.scale,
            output_dir=output_dir,
            step_headings=result.step_headings,
            landmark=result.landmark,
        )
        return
    plot_trajectory(
        result.trajectory,
        gx_mean=result.prepared.gx_mean,
        gz_mean=result.prepared.gz_mean,
        floormap_path=settings.floormap_path,
        origin_px=settings.origin_px,
        scale=settings.scale,
        output_dir=output_dir,
        step_headings=result.step_headings,
        landmark=result.landmark,
    )


def _render_particle_artifacts(
    result: TrajectoryResult,
    settings: ParticleSettings,
    output: OutputSettings,
    output_dir: Path,
) -> None:
    """PFの段階画像、経路比較、animationを保存する。"""
    assert result.particle is not None
    prepared = result.prepared
    visualization_paths: list[Path] = []
    if output.save_step_frames:
        visualization_paths.extend(
            save_particle_step_frames(
                list(result.particle.stages),
                list(result.particle.diagnostics),
                result.trajectory,
                gx_mean=prepared.gx_mean,
                gz_mean=prepared.gz_mean,
                floormap_path=settings.floormap_path,
                origin_px=settings.origin_px,
                scale=settings.scale,
                output_dir=output_dir,
                step_range=output.step_frames_range,
                arrows=output.step_frames_arrows,
                dpi=output.step_frames_dpi,
            )
        )
    if output.save_path_comparison:
        if len(result.particle.path_comparisons) != 1:
            raise RuntimeError("内部エラー: 代表軌跡候補が収集されませんでした。")
        visualization_paths.append(
            save_particle_path_comparison(
                result.particle.path_comparisons[0],
                gx_mean=prepared.gx_mean,
                gz_mean=prepared.gz_mean,
                floormap_path=settings.floormap_path,
                origin_px=settings.origin_px,
                scale=settings.scale,
                output_path=output_dir / "particle_paths_comparison.png",
                dpi=output.step_frames_dpi,
            )
        )
    if visualization_paths:
        size_mb = generated_files_size(visualization_paths) / (1024 * 1024)
        print(
            "Particle visualization saved: "
            f"{len(visualization_paths)} files, {size_mb:.2f} MiB"
        )
    if output.save_animation:
        save_particle_animation(
            np.asarray(result.particle.all_particles),
            result.trajectory,
            gx_mean=prepared.gx_mean,
            gz_mean=prepared.gz_mean,
            floormap_path=settings.floormap_path,
            origin_px=settings.origin_px,
            scale=settings.scale,
            output_path=output_dir / "particle_filter.mp4",
            landmark=result.landmark,
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
