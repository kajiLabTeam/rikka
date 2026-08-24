"""通常 PDR 軌跡の描画処理。

役割:
    メートル座標の軌跡をフロアマップのピクセル座標へ変換し、ステップ分類、方位、
    始点・終点、BLE ランドマーク補正を重ねた画像として表示・保存する。
依存元:
    ``config`` から地図の既定値、``models`` から ``StepHeading`` を取得し、
    NumPy、Pandas、Matplotlib を座標変換と描画に利用する。
利用先:
    ``plot.pipeline`` が通常軌跡を描画し、particle filter の描画処理も
    座標変換と方位オーバーレイを再利用する。
処理フロー:
    地図を読み、端末姿勢に合わせて座標とベクトルをピクセルへ変換し、軌跡線と
    診断情報を描画して必要なら PNG を保存する。
"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from ...common.config import FLOORMAP_ORIGIN_PX, FLOORMAP_PATH, FLOORMAP_SCALE
from ...common.lib.floormap import compute_pixel_coords, pixel_vector_from_heading
from ...common.lib.models import (
    LandmarkCorrection,
    LandmarkCorrectionResult,
    StepHeading,
)
from ...matplotlib_config import configure_japanese_font

_compute_pixel_coords = compute_pixel_coords
_pixel_vector_from_heading = pixel_vector_from_heading


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
                label="移動方位" if not motion_label_added else None,
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
                label="身体方位" if not body_label_added else None,
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
            label="横歩き",
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
            label="横歩き疑い",
        )


def _plot_landmark_overlay(
    ax: Axes,
    landmark: LandmarkCorrectionResult | None,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> None:
    """補正前軌跡、ランドマーク位置、補正発生地点を重ねて描画する。"""
    if landmark is None:
        return

    raw = np.asarray(landmark.raw_trajectory, dtype=float)
    if raw.ndim == 2 and raw.shape[1] == 2 and len(raw) >= 2:
        raw_px, raw_py = _compute_pixel_coords(
            raw[:, 0], raw[:, 1], gx_mean, gz_mean, origin_px, scale
        )
        ax.plot(
            raw_px,
            raw_py,
            linestyle="--",
            color="gray",
            linewidth=1.4,
            alpha=0.6,
            zorder=1,
            label="補正前軌跡",
        )

    applied = [item for item in landmark.corrections if item.applied]
    _plot_landmark_positions(ax, landmark, applied, gx_mean, gz_mean, origin_px, scale)
    if not applied:
        return

    unique = list(
        {
            (item.beacon_id, item.landmark_x, item.landmark_y): item for item in applied
        }.values()
    )
    _plot_anchor_heading_arrows(ax, unique, gx_mean, gz_mean, origin_px, scale)

    landmark_x = np.array([item.landmark_x for item in applied], dtype=float)
    landmark_y = np.array([item.landmark_y for item in applied], dtype=float)
    lx, ly = _compute_pixel_coords(
        landmark_x, landmark_y, gx_mean, gz_mean, origin_px, scale
    )

    before_x = np.array([item.before_x for item in applied], dtype=float)
    before_y = np.array([item.before_y for item in applied], dtype=float)
    bx, by = _compute_pixel_coords(
        before_x,
        before_y,
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    ax.scatter(
        bx,
        by,
        marker="X",
        s=110,
        color="red",
        edgecolors="black",
        linewidths=0.8,
        zorder=9,
        label="ランドマーク補正",
    )
    for index in range(len(applied)):
        ax.plot(
            [bx[index], lx[index]],
            [by[index], ly[index]],
            color="red",
            linewidth=1.0,
            alpha=0.7,
            zorder=8,
        )


def _plot_landmark_positions(
    ax: Axes,
    landmark: LandmarkCorrectionResult,
    applied: list[LandmarkCorrection],
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> None:
    """登録済みBLE位置を検出の有無にかかわらず地図へ描画する。"""
    applied_by_id = {item.beacon_id: item for item in applied}
    configured_ids = {item.beacon_id for item in landmark.landmarks}
    groups: tuple[tuple[list[tuple[float, float]], str, int, str, str], ...] = (
        (
            [],
            "*",
            260,
            "magenta",
            "ランドマーク",
        ),
        (
            [],
            "D",
            130,
            "cyan",
            "確定ランドマーク（位置）",
        ),
        (
            [],
            "P",
            170,
            "orange",
            "確定ランドマーク（位置・方位）",
        ),
    )
    for definition in landmark.landmarks:
        correction = applied_by_id.get(definition.beacon_id)
        position_sigma = (
            definition.position_sigma_m
            if correction is None
            else correction.anchor_position_sigma_m
        )
        heading = (
            definition.heading_deg
            if correction is None
            else correction.anchor_heading_deg
        )
        group_index = 0 if position_sigma is None else 1 if heading is None else 2
        groups[group_index][0].append(
            (float(definition.pixel_x), float(definition.pixel_y))
        )

    fallback = [item for item in applied if item.beacon_id not in configured_ids]
    if fallback:
        fallback_x, fallback_y = _compute_pixel_coords(
            np.asarray([item.landmark_x for item in fallback], dtype=float),
            np.asarray([item.landmark_y for item in fallback], dtype=float),
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        for item, pixel_x, pixel_y in zip(
            fallback, fallback_x, fallback_y, strict=True
        ):
            group_index = (
                0
                if item.anchor_position_sigma_m is None
                else 1
                if item.anchor_heading_deg is None
                else 2
            )
            groups[group_index][0].append((float(pixel_x), float(pixel_y)))

    for points, marker, size, color, label in groups:
        if not points:
            continue
        coordinates = np.asarray(points, dtype=float)
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            marker=marker,
            s=size,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=8,
            label=label,
        )


def _plot_anchor_heading_arrows(
    ax: Axes,
    corrections: list[LandmarkCorrection],
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> None:
    """方位確定ランドマークへ設定方位を示す矢印を描く。"""
    heading_items = [
        item for item in corrections if item.anchor_heading_deg is not None
    ]
    if not heading_items:
        return
    xs, ys = _compute_pixel_coords(
        np.asarray([item.landmark_x for item in heading_items], dtype=float),
        np.asarray([item.landmark_y for item in heading_items], dtype=float),
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    vectors = np.asarray(
        [
            _pixel_vector_from_heading(
                np.radians(item.anchor_heading_deg),
                1.0,
                gx_mean,
                gz_mean,
                scale,
            )
            for item in heading_items
            if item.anchor_heading_deg is not None
        ],
        dtype=float,
    )
    ax.quiver(
        xs,
        ys,
        vectors[:, 0],
        vectors[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="darkorange",
        width=0.006,
        zorder=9,
        label="確定方位",
    )
    bidirectional = np.asarray(
        [item.anchor_heading_bidirectional for item in heading_items], dtype=bool
    )
    if np.any(bidirectional):
        ax.quiver(
            xs[bidirectional],
            ys[bidirectional],
            -vectors[bidirectional, 0],
            -vectors[bidirectional, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="darkorange",
            width=0.006,
            zorder=9,
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
    landmark: LandmarkCorrectionResult | None = None,
) -> None:
    """推定した2次元歩行軌跡をフロアマップ上にプロットする。"""
    configure_japanese_font()
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
    if landmark is not None:
        lc.set_label("補正後軌跡")
    lc.set_array(np.arange(n - 1))
    ax.add_collection(lc)
    # 各ステップ点を同じカラーマップで描画
    sc = ax.scatter(px, py, c=np.arange(n), cmap=cmap, norm=norm, s=20, zorder=3)
    fig.colorbar(sc, ax=ax, label="歩番号")
    # 起点を強調表示
    ax.plot(px[0], py[0], "go", markersize=10, label="開始点", zorder=4)
    _plot_heading_overlay(
        ax,
        trajectory,
        step_headings,
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    _plot_landmark_overlay(
        ax,
        landmark,
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )

    ax.set_title("フロアマップ上の歩行軌跡")
    ax.legend()
    plt.tight_layout()
    if output_dir is not None:
        # グラフ画像をoutputフォルダに保存
        fig.savefig(output_dir / "trajectory.png", dpi=150, bbox_inches="tight")
    plt.show()
