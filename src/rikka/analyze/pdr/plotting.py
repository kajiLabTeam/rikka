"""通常 PDR 軌跡の描画処理。"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from ...config import FLOORMAP_ORIGIN_PX, FLOORMAP_PATH, FLOORMAP_SCALE
from .models import StepHeading


def _compute_pixel_coords(
    xs: np.ndarray,
    ys: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """メートル座標をフロアマップのピクセル座標に変換する。"""
    if abs(gx_mean) > abs(gz_mean):
        y_sign = -1 if gx_mean > 0 else 1
    else:
        y_sign = -1 if gz_mean < 0 else 1
    px = origin_px[0] + xs / scale
    py = origin_px[1] + y_sign * ys / scale
    return px, py


def _pixel_vector_from_heading(
    heading: float,
    length_m: float,
    gx_mean: float,
    gz_mean: float,
    scale: float,
) -> tuple[float, float]:
    """メートル座標の方位ベクトルをピクセル座標の差分に変換する。"""
    y_sign = (
        -1
        if (
            (abs(gx_mean) > abs(gz_mean) and gx_mean > 0)
            or (abs(gz_mean) >= abs(gx_mean) and gz_mean < 0)
        )
        else 1
    )
    return (
        length_m * float(np.cos(heading)) / scale,
        y_sign * length_m * float(np.sin(heading)) / scale,
    )


def _is_plot_sidestep_movement(movement_type: str | None) -> bool:
    """描画上で横歩きとして扱う移動タイプかどうかを返す。"""
    return movement_type in {
        "sidestep_left",
        "sidestep_right",
        "turning_sidestep_left",
        "turning_sidestep_right",
    }


def _is_plot_sidestep_suspect_movement(movement_type: str | None) -> bool:
    """描画上で横歩き疑いとして扱う移動タイプかどうかを返す。"""
    return movement_type in {"sidestep_suspect_left", "sidestep_suspect_right"}


def _plot_heading_overlay(
    ax: Axes,
    trajectory: list[list[float]],
    step_headings: list[StepHeading] | None,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> None:
    """軌跡上に移動方向・体の向き・横歩き判定を重ねて描画する。"""
    if step_headings is None or len(step_headings) == 0 or len(trajectory) < 2:
        return

    points = np.asarray(trajectory, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        return

    px, py = _compute_pixel_coords(
        points[:, 0],
        points[:, 1],
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    step_count = min(len(step_headings), len(points) - 1)
    if step_count <= 0:
        return

    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    positive_lengths = lengths[lengths > 1e-12]
    arrow_length_m = (
        float(np.median(positive_lengths)) * 0.45
        if len(positive_lengths) > 0
        else max(scale * 30.0, 0.3)
    )
    arrow_length_m = max(arrow_length_m, scale * 24.0)

    body_label_added = False
    motion_label_added = False
    sidestep_points: list[tuple[float, float]] = []
    sidestep_suspect_points: list[tuple[float, float]] = []
    for i in range(step_count):
        heading = step_headings[i]
        trajectory_movement_type = (
            heading.trajectory_movement_type
            if heading.trajectory_movement_type is not None
            else heading.movement_type
        )
        start_x = float(px[i])
        start_y = float(py[i])
        if heading.selected_heading is not None:
            dx, dy = _pixel_vector_from_heading(
                heading.selected_heading,
                arrow_length_m,
                gx_mean,
                gz_mean,
                scale,
            )
            ax.arrow(
                start_x,
                start_y,
                dx,
                dy,
                width=1.4,
                head_width=10.0,
                head_length=12.0,
                length_includes_head=True,
                color="dodgerblue",
                alpha=0.85,
                zorder=5,
                label="Move heading" if not motion_label_added else None,
            )
            motion_label_added = True
        if heading.body_heading is not None:
            dx, dy = _pixel_vector_from_heading(
                heading.body_heading,
                arrow_length_m * 1.05,
                gx_mean,
                gz_mean,
                scale,
            )
            ax.arrow(
                start_x,
                start_y,
                dx,
                dy,
                width=1.8,
                head_width=12.0,
                head_length=15.0,
                length_includes_head=True,
                color="orangered",
                alpha=0.95,
                zorder=6,
                label="Body heading" if not body_label_added else None,
            )
            body_label_added = True
        if _is_plot_sidestep_suspect_movement(trajectory_movement_type):
            sidestep_suspect_points.append((float(px[i + 1]), float(py[i + 1])))
        elif _is_plot_sidestep_movement(trajectory_movement_type):
            sidestep_points.append((float(px[i + 1]), float(py[i + 1])))

    if sidestep_points:
        sidestep_arr = np.asarray(sidestep_points, dtype=float)
        ax.scatter(
            sidestep_arr[:, 0],
            sidestep_arr[:, 1],
            marker="s",
            s=52,
            facecolors="none",
            edgecolors="lime",
            linewidths=1.8,
            zorder=7,
            label="Sidestep",
        )
    if sidestep_suspect_points:
        sidestep_suspect_arr = np.asarray(sidestep_suspect_points, dtype=float)
        ax.scatter(
            sidestep_suspect_arr[:, 0],
            sidestep_suspect_arr[:, 1],
            marker="D",
            s=46,
            facecolors="none",
            edgecolors="gold",
            linewidths=1.8,
            zorder=7,
            label="Sidestep suspect",
        )


def plot_trajectory(
    trajectory: list[list[float]],
    gx_mean: float = 0.0,
    gz_mean: float = 0.0,
    floormap_path: str | Path = FLOORMAP_PATH,
    origin_px: tuple[int, int] = FLOORMAP_ORIGIN_PX,
    scale: float = FLOORMAP_SCALE,
    output_dir: Path | None = None,
    step_headings: list[StepHeading] | None = None,
) -> None:
    """推定した2次元歩行軌跡をフロアマップ上にプロットする。"""
    df = pd.DataFrame(trajectory, columns=["x", "y"])

    px, py = _compute_pixel_coords(
        df["x"].to_numpy(), df["y"].to_numpy(), gx_mean, gz_mean, origin_px, scale
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    # フロアマップを背景として表示
    map_img = plt.imread(Path(floormap_path))
    ax.imshow(map_img)

    # 軌跡をグラデーション（開始:青 → 終了:赤）で描画
    n = len(px)
    norm = Normalize(vmin=0, vmax=max(n - 1, 1))
    cmap = cm.get_cmap("plasma")
    # 各ステップ間のセグメントに色を付けて LineCollection で描画
    pts = np.column_stack([px, py]).reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segments.tolist(), cmap=cmap, norm=norm, zorder=2)
    lc.set_array(np.arange(n - 1))
    ax.add_collection(lc)
    # 各ステップ点を同じカラーマップで描画
    sc = ax.scatter(px, py, c=np.arange(n), cmap=cmap, norm=norm, s=20, zorder=3)
    fig.colorbar(sc, ax=ax, label="Step")
    # 起点を強調表示
    ax.plot(px[0], py[0], "go", markersize=10, label="Start", zorder=4)
    _plot_heading_overlay(
        ax,
        trajectory,
        step_headings,
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )

    ax.set_title("Walking Trajectory on Floormap")
    ax.legend()
    plt.tight_layout()
    if output_dir is not None:
        # グラフ画像をoutputフォルダに保存
        fig.savefig(output_dir / "trajectory.png", dpi=150, bbox_inches="tight")
    plt.show()
