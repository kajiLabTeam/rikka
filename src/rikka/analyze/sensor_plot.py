"""センサーデータ可視化モジュール"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..config import DATA_DIR
from .pdr import detect_steps, load_sensor_data, process_sensor_data


def plot_sensor_data(data_dir: str | Path = DATA_DIR) -> None:
    """加速度・ジャイロデータをグラフ化して入力フォルダに保存する。

    Args:
        data_dir: Accelerometer.csv と Gyroscope.csv が格納されたディレクトリ
    """
    data_path = Path(data_dir)
    df_acc, df_gyro = load_sensor_data(data_path)
    df_acc, df_gyro = process_sensor_data(df_acc, df_gyro)
    peaks = detect_steps(df_acc)

    t_acc = df_acc["t"].to_numpy() if "t" in df_acc.columns else np.arange(len(df_acc))
    t_gyro = (
        df_gyro["t"].to_numpy() if "t" in df_gyro.columns else np.arange(len(df_gyro))
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=False)
    fig.suptitle(f"Sensor Data — {data_path.name}", fontsize=13)

    # --- 1. 生加速度 ---
    ax = axes[0]
    ax.plot(t_acc, df_acc["x"], label="x", linewidth=0.8)
    ax.plot(t_acc, df_acc["y"], label="y", linewidth=0.8)
    ax.plot(t_acc, df_acc["z"], label="z", linewidth=0.8)
    ax.set_title("Raw Accelerometer [m/s²]")
    ax.set_ylabel("m/s²")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4)

    # --- 2. 線形加速度ノルム + ステップピーク ---
    ax = axes[1]
    ax.plot(
        t_acc, df_acc["lin_norm"], color="lightblue", linewidth=0.8, label="lin_norm"
    )
    ax.plot(
        t_acc,
        df_acc["low_lin_norm"],
        color="steelblue",
        linewidth=1.5,
        label="low_lin_norm",
    )
    if len(peaks) > 0:
        ax.scatter(
            t_acc[peaks],
            df_acc["low_lin_norm"].to_numpy()[peaks],
            marker="^",
            color="red",
            s=40,
            zorder=3,
            label=f"steps ({len(peaks)})",
        )
    ax.set_title("Linear Acceleration Norm [m/s²]")
    ax.set_ylabel("m/s²")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4)

    # --- 3. 生ジャイロ ---
    ax = axes[2]
    ax.plot(t_gyro, df_gyro["x"], label="x", linewidth=0.8)
    ax.plot(t_gyro, df_gyro["y"], label="y", linewidth=0.8)
    ax.plot(t_gyro, df_gyro["z"], label="z", linewidth=0.8)
    ax.set_title("Raw Gyroscope [rad/s]")
    ax.set_ylabel("rad/s")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4)

    # --- 4. 積算角度 ---
    ax = axes[3]
    ax.plot(
        t_gyro,
        df_gyro["angle"],
        color="lightcoral",
        linewidth=0.8,
        label="angle",
    )
    ax.plot(
        t_gyro,
        df_gyro["low_angle"],
        color="crimson",
        linewidth=1.5,
        label="low_angle",
    )
    ax.set_title("Heading Angle [rad]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("rad")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4)

    plt.tight_layout()
    output_path = data_path / "sensor_plot.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


def plot_step_lengths(
    step_lengths: list[float],
    output_dir: Path | None = None,
    *,
    t_at_steps: list[float] | None = None,
    t_acc: np.ndarray | None = None,
    low_lin_norm: np.ndarray | None = None,
) -> None:
    """歩幅の折れ線グラフを描画し、平均・標準偏差をコンソールと図に出力する。

    Args:
        step_lengths: ステップごとの歩幅 [m] のリスト
        output_dir: PNG 保存先ディレクトリ（None なら保存しない）
        t_at_steps: 各ステップのピーク時刻 [s]（時系列モード用）
        t_acc: 加速度データの時刻配列 [s]（時系列モード用）
        low_lin_norm: 平滑化線形加速度ノルム配列（時系列モード用）
    """
    arr = np.array(step_lengths)
    n = len(arr)
    if n == 0:
        print("ステップが検出されませんでした")
        return

    mean = float(arr.mean())
    std = float(arr.std())

    print(f"歩幅  mean={mean:.3f} m  std={std:.3f} m  n={n}")

    stat_text = f"mean={mean:.3f} m\nstd={std:.3f} m\nn={n}"

    if t_at_steps is not None and t_acc is not None and low_lin_norm is not None:
        # --- 時系列モード: x軸を時間、低周波加速度ノルムを背景に描画 ---
        t_steps = np.asarray(t_at_steps)
        if len(t_steps) != n:
            raise ValueError(
                "t_at_steps and step_lengths must have the same length in "
                f"time-series mode: len(t_at_steps)={len(t_steps)}, "
                f"len(step_lengths)={n}"
            )
        fig, ax = plt.subplots(figsize=(12, 4))

        # 左y軸: low_lin_norm の連続時系列
        ax.plot(
            t_acc,
            low_lin_norm,
            color="steelblue",
            linewidth=1.2,
            label="low_lin_norm",
            alpha=0.85,
        )
        # 各ステップのピーク位置に▲マーカー
        ln_at_steps = np.interp(t_steps, t_acc, low_lin_norm)
        ax.scatter(
            t_steps,
            ln_at_steps,
            marker="^",
            color="red",
            s=45,
            zorder=3,
            label=f"steps ({n})",
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("low_lin_norm [m/s²]")
        ax.set_title("Step Length with Linear Acceleration Norm")
        ax.grid(True, linewidth=0.4)

        # 右y軸: 歩幅を各ステップ時刻に点でプロット
        ax2 = ax.twinx()
        ax2.scatter(
            t_steps,
            arr,
            color="darkorange",
            s=35,
            zorder=4,
            marker="o",
            label="step length",
        )
        ax2.axhline(
            mean,
            color="crimson",
            linewidth=1.0,
            linestyle="--",
            label=f"mean={mean:.3f} m",
        )
        ax2.set_ylabel("Step Length [m]")

        ax2.text(
            0.98,
            0.97,
            stat_text,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )

        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="upper left",
            fontsize=9,
        )
    else:
        # --- ステップ番号モード（フォールバック）---
        steps = np.arange(1, n + 1)
        fig, ax = plt.subplots(figsize=(max(6, n * 0.35 + 2), 4))

        ax.plot(
            steps,
            arr,
            color="steelblue",
            linewidth=1.5,
            marker="o",
            markersize=5,
            label="step length",
        )
        ax.axhline(mean, color="crimson", linewidth=1.5, label=f"mean={mean:.3f} m")
        ax.axhspan(
            mean - std,
            mean + std,
            color="crimson",
            alpha=0.15,
            label=f"±1σ ({std:.3f} m)",
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Step Length [m]")
        ax.set_title("Step Length per Step")
        ax.set_xticks(steps)
        ax.grid(True, axis="y", linewidth=0.4)
        ax.text(
            0.98,
            0.97,
            stat_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    if output_dir is not None:
        save_path = Path(output_dir) / "step_lengths.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def _get_step_acceleration_points(
    df_acc: pd.DataFrame | None,
    peaks: np.ndarray | None,
    step_no: int,
    window: int = 50,
) -> NDArray[np.float64] | None:
    """ステップに対応する水平加速度点群を返す。"""
    if df_acc is None or peaks is None or len(peaks) == 0:
        return None

    accel_columns = (
        ("h_y", "h_z") if {"h_y", "h_z"} <= set(df_acc.columns) else ("lin_y", "lin_z")
    )
    if not set(accel_columns) <= set(df_acc.columns):
        return None

    step_index = step_no - 1
    if step_index >= len(peaks):
        return None

    peak = int(peaks[step_index])
    if step_index + 1 < len(peaks):
        start = peak
        end = int(peaks[step_index + 1])
    else:
        start = max(0, peak - window)
        end = min(len(df_acc), peak + window + 1)

    if end <= start:
        return None

    points = np.asarray(
        df_acc.iloc[start:end][list(accel_columns)].to_numpy(dtype=float),
        dtype=np.float64,
    )
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return None
    return points


def _project_acceleration_to_step_axes(
    points: NDArray[np.float64],
    dx: float,
    dy: float,
) -> NDArray[np.float64] | None:
    """水平加速度をステップの進行方向・横方向へ射影する。"""
    length = float(np.hypot(dx, dy))
    if length <= 1e-12:
        return None

    forward_axis = np.array([dx / length, dy / length], dtype=np.float64)
    lateral_axis = np.array([-forward_axis[1], forward_axis[0]], dtype=np.float64)
    forward_acc = points @ forward_axis
    lateral_acc = points @ lateral_axis
    return np.column_stack([forward_acc, lateral_acc])


def _plot_no_acceleration_data(ax: Axes) -> None:
    """加速度データがない場合のプレースホルダーを描画する。"""
    ax.text(
        0.5,
        0.5,
        "No acceleration data",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="0.45",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _set_symmetric_accel_limits(ax: Axes, projected_accel: NDArray[np.float64]) -> None:
    """加速度分布を読みやすくするため、外れ値に強い対称軸範囲を設定する。"""
    limit = float(np.percentile(np.abs(projected_accel), 98))
    if limit <= 1e-12:
        limit = 1.0
    limit = max(1.0, np.ceil(limit * 2) / 2)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)


def plot_step_vectors(
    trajectory: list[list[float]],
    output_dir: Path | None = None,
    *,
    df_acc: pd.DataFrame | None = None,
    peaks: np.ndarray | None = None,
) -> None:
    """各ステップの変位ベクトルを個別画像として保存する。

    ``df_acc`` と ``peaks`` が渡された場合は、同じステップ区間の水平加速度を
    進行方向・横方向へ射影し、時系列と2D分布を併せて描画する。

    Args:
        trajectory: 各ステップの [x, y] 座標リスト（原点を含む）
        output_dir: PNG 保存先ディレクトリ（None なら保存しない）
        df_acc: 処理済み加速度DataFrame
        peaks: ステップピークのインデックス配列
    """
    points = np.asarray(trajectory, dtype=float)
    if len(points) < 2:
        return

    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 0
    if not valid.any():
        return

    original_step_numbers = np.nonzero(valid)[0] + 1
    vectors = vectors[valid]
    lengths = lengths[valid]
    n = len(vectors)
    max_len = float(lengths.max())
    limit = max(1.0, np.ceil(max_len * 10) / 10) * 1.15
    major_ticks = np.arange(-np.ceil(limit), np.ceil(limit) + 0.5, 0.5)
    minor_ticks = np.arange(-np.ceil(limit), np.ceil(limit) + 0.25, 0.25)

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.08, 0.95, n))

    save_dir: Path | None = None
    if output_dir is not None:
        save_dir = Path(output_dir) / "step_vectors"
        save_dir.mkdir(parents=True, exist_ok=True)

    for step_no, (dx, dy), length, color in zip(
        original_step_numbers, vectors, lengths, colors, strict=True
    ):
        heading = float(np.degrees(np.arctan2(dy, dx)))
        accel_points = _get_step_acceleration_points(df_acc, peaks, int(step_no))
        projected_accel = (
            _project_acceleration_to_step_axes(accel_points, float(dx), float(dy))
            if accel_points is not None
            else None
        )

        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0])
        ax_vector = fig.add_subplot(gs[:, 0])
        ax_timeseries = fig.add_subplot(gs[0, 1])
        ax_distribution = fig.add_subplot(gs[1, 1])

        ax_vector.set_aspect("equal", adjustable="box")
        ax_vector.set_xlim(-limit, limit)
        ax_vector.set_ylim(-limit, limit)
        ax_vector.set_xticks(major_ticks)
        ax_vector.set_yticks(major_ticks)
        ax_vector.set_xticks(minor_ticks, minor=True)
        ax_vector.set_yticks(minor_ticks, minor=True)
        ax_vector.grid(which="major", color="0.82", linestyle="--", linewidth=1.0)
        ax_vector.grid(which="minor", color="0.9", linestyle="--", linewidth=0.8)
        ax_vector.axhline(0, color="0.35", linewidth=1.0)
        ax_vector.axvline(0, color="0.35", linewidth=1.0)
        ax_vector.plot([0.0, dx], [0.0, dy], color=color, linewidth=3.0, alpha=0.9)
        ax_vector.scatter(0.0, 0.0, color="0.2", s=28, zorder=3, label="start")
        ax_vector.scatter(dx, dy, color=color, s=40, zorder=3, label="end")
        ax_vector.annotate(
            "",
            xy=(dx, dy),
            xytext=(dx * 0.82, dy * 0.82),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "linewidth": 2.0,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
        ax_vector.text(
            0.98,
            0.97,
            f"dx={dx:.3f} m\n"
            f"dy={dy:.3f} m\n"
            f"length={length:.3f} m\n"
            f"heading={heading:.1f}°",
            transform=ax_vector.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
        )
        ax_vector.set_xlabel("dx [m]")
        ax_vector.set_ylabel("dy [m]")
        ax_vector.set_title(f"Step {step_no:03d} Displacement")
        ax_vector.tick_params(labelsize=8)
        ax_vector.legend(loc="upper left", fontsize=9)

        if projected_accel is None:
            _plot_no_acceleration_data(ax_timeseries)
            _plot_no_acceleration_data(ax_distribution)
        else:
            phase = np.linspace(0.0, 100.0, len(projected_accel))
            forward_acc = projected_accel[:, 0]
            lateral_acc = projected_accel[:, 1]

            ax_timeseries.plot(
                phase,
                forward_acc,
                color="steelblue",
                linewidth=1.4,
                label="forward",
            )
            ax_timeseries.plot(
                phase,
                lateral_acc,
                color="darkorange",
                linewidth=1.4,
                label="lateral",
            )
            ax_timeseries.axhline(0.0, color="0.35", linewidth=0.8)
            ax_timeseries.set_title("Acceleration over Step")
            ax_timeseries.set_xlabel("step phase [%]")
            ax_timeseries.set_ylabel("acceleration [m/s²]")
            ax_timeseries.grid(True, linewidth=0.4)
            ax_timeseries.legend(loc="upper right", fontsize=8)

            ax_distribution.plot(
                forward_acc,
                lateral_acc,
                color="0.55",
                linewidth=0.9,
                alpha=0.45,
                zorder=1,
            )
            ax_distribution.scatter(
                forward_acc,
                lateral_acc,
                c=phase,
                cmap="viridis",
                s=18,
                alpha=0.9,
                zorder=2,
            )
            ax_distribution.scatter(
                forward_acc[0],
                lateral_acc[0],
                color="0.2",
                s=28,
                zorder=3,
                label="start",
            )
            ax_distribution.scatter(
                forward_acc[-1],
                lateral_acc[-1],
                color="crimson",
                s=32,
                zorder=3,
                label="end",
            )
            ax_distribution.axhline(0.0, color="0.35", linewidth=0.8)
            ax_distribution.axvline(0.0, color="0.35", linewidth=0.8)
            ax_distribution.set_aspect("equal", adjustable="box")
            _set_symmetric_accel_limits(ax_distribution, projected_accel)
            ax_distribution.set_title("Acceleration Distribution (color=phase)")
            ax_distribution.set_xlabel("forward acceleration [m/s²]")
            ax_distribution.set_ylabel("lateral acceleration [m/s²]")
            ax_distribution.grid(True, linewidth=0.4)
            ax_distribution.legend(loc="upper right", fontsize=8)

        if save_dir is not None:
            save_path = save_dir / f"step_{step_no:03d}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    if save_dir is not None:
        print(f"Saved step vector images to {save_dir}")
