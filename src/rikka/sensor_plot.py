"""センサーデータ可視化モジュール"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import DATA_DIR
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
) -> None:
    """歩幅の折れ線グラフを描画し、平均・標準偏差をコンソールと図に出力する。

    Args:
        step_lengths: ステップごとの歩幅 [m] のリスト
        output_dir: PNG 保存先ディレクトリ（None なら保存しない）
    """
    arr = np.array(step_lengths)
    mean = float(arr.mean())
    std = float(arr.std())
    n = len(arr)

    print(f"歩幅  mean={mean:.3f} m  std={std:.3f} m  n={n}")

    steps = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.35 + 2), 4))

    ax.plot(
        steps, arr, color="steelblue", linewidth=1.5,
        marker="o", markersize=5, label="step length",
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
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", linewidth=0.4)

    ax.text(
        0.98,
        0.97,
        f"mean={mean:.3f} m\nstd={std:.3f} m\nn={n}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
    )

    plt.tight_layout()
    if output_dir is not None:
        save_path = Path(output_dir) / "step_lengths.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
