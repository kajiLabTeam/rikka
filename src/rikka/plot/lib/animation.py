"""粒子フィルタ軌跡の静止画・アニメーション描画。

役割:
    地図上へ代表軌跡と全粒子分布を描き、静止画表示と動画保存を行う。
依存元:
    ``config`` の地図既定値、``plot.lib`` の座標変換と方位重ね描き、
    Matplotlib、NumPy、Pandasを利用する。
利用先:
    PDR pipelineがparticle filter実行後の軌跡表示と任意のanimation保存に使用する。
処理フロー:
    メートル座標を画素座標へ変換して地図へ重ね、動画はMP4保存を試み、利用できない
    場合は同名のGIFへフォールバックする。
"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from ...common.config import FLOORMAP_ORIGIN_PX, FLOORMAP_PATH, FLOORMAP_SCALE
from ...common.lib.floormap import compute_pixel_coords
from ...common.lib.models import LandmarkCorrectionResult, StepHeading
from ...matplotlib_config import configure_japanese_font
from .trajectory import (
    _plot_anchor_heading_arrows as plot_anchor_heading_arrows,
)
from .trajectory import (
    _plot_heading_overlay as plot_heading_overlay,
)
from .trajectory import (
    _plot_landmark_overlay as plot_landmark_overlay,
)


def plot_particle_filter_trajectory(
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
    """PF の平均優先・壁際祖先フォールバック軌跡を描画する。

    Args:
        trajectory: 各ステップの [x, y] 座標リスト（メートル）
        gx_mean: X軸重力成分の平均値
        gz_mean: Z軸重力成分の平均値
        floormap_path: フロアマップ画像のパス
        origin_px: 軌跡起点のピクセル座標
        scale: 1ピクセルあたりのメートル数
        output_dir: 出力ディレクトリ（指定時に PNG 保存）
    """
    configure_japanese_font()
    df = pd.DataFrame(trajectory, columns=["x", "y"])
    px, py = compute_pixel_coords(
        df["x"].to_numpy(), df["y"].to_numpy(), gx_mean, gz_mean, origin_px, scale
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    map_img = plt.imread(Path(floormap_path))
    ax.imshow(map_img)

    n = len(px)
    norm = Normalize(vmin=0, vmax=max(n - 1, 1))
    cmap = cm.get_cmap("plasma")
    pts = np.column_stack([px, py]).reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segments.tolist(), cmap=cmap, norm=norm, zorder=2)
    if landmark is not None:
        lc.set_label("ランドマーク反映後軌跡")
    lc.set_array(np.arange(n - 1))
    ax.add_collection(lc)
    sc = ax.scatter(px, py, c=np.arange(n), cmap=cmap, norm=norm, s=20, zorder=3)
    fig.colorbar(sc, ax=ax, label="歩番号")
    ax.plot(px[0], py[0], "go", markersize=10, label="開始点", zorder=4)
    plot_heading_overlay(
        ax,
        trajectory,
        step_headings,
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    plot_landmark_overlay(
        ax,
        landmark,
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )

    ax.set_title("フロアマップ上のパーティクルフィルタ軌跡")
    ax.legend()
    plt.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir / "pf_trajectory.png", dpi=150, bbox_inches="tight")
    plt.show()


def _draw_animation_landmarks(
    ax: Axes,
    landmark: LandmarkCorrectionResult,
    frame: int,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> None:
    """1フレームへランドマーク位置と、その時点までに起きた補正を描く。

    ``all_particles`` は先頭が初期粒子なので、歩 index ``step_index`` の補正は
    フレーム ``step_index + 1`` に対応する。
    """
    applied = [item for item in landmark.corrections if item.applied]
    if not applied:
        return

    unique = list(
        {
            (item.beacon_id, item.landmark_x, item.landmark_y): item for item in applied
        }.values()
    )
    groups = (
        (
            [item for item in unique if item.anchor_position_sigma_m is None],
            "*",
            260,
            "magenta",
            "ランドマーク",
        ),
        (
            [
                item
                for item in unique
                if item.anchor_position_sigma_m is not None
                and item.anchor_heading_deg is None
            ],
            "D",
            130,
            "cyan",
            "確定ランドマーク（位置）",
        ),
        (
            [item for item in unique if item.anchor_heading_deg is not None],
            "P",
            170,
            "orange",
            "確定ランドマーク（位置・方位）",
        ),
    )
    for items, marker, size, color, label in groups:
        if not items:
            continue
        target = np.asarray(
            [(item.landmark_x, item.landmark_y) for item in items], dtype=float
        )
        target_px, target_py = compute_pixel_coords(
            target[:, 0], target[:, 1], gx_mean, gz_mean, origin_px, scale
        )
        ax.scatter(
            target_px,
            target_py,
            marker=marker,
            s=size,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
            label=label,
        )
    plot_anchor_heading_arrows(ax, unique, gx_mean, gz_mean, origin_px, scale)

    done = [item for item in applied if item.step_index + 1 <= frame]
    if not done:
        return
    before = np.asarray([(item.before_x, item.before_y) for item in done], dtype=float)
    before_px, before_py = compute_pixel_coords(
        before[:, 0], before[:, 1], gx_mean, gz_mean, origin_px, scale
    )
    after = np.asarray(
        [(item.landmark_x, item.landmark_y) for item in done], dtype=float
    )
    after_px, after_py = compute_pixel_coords(
        after[:, 0], after[:, 1], gx_mean, gz_mean, origin_px, scale
    )
    ax.scatter(
        before_px,
        before_py,
        marker="X",
        s=110,
        color="red",
        edgecolors="black",
        linewidths=0.8,
        zorder=7,
        label="ランドマーク補正",
    )
    for index in range(len(done)):
        ax.plot(
            [before_px[index], after_px[index]],
            [before_py[index], after_py[index]],
            color="red",
            linewidth=1.0,
            alpha=0.7,
            zorder=6,
        )


def save_particle_animation(
    all_particles: np.ndarray,
    mean_trajectory: list[list[float]],
    gx_mean: float,
    gz_mean: float,
    floormap_path: str | Path = FLOORMAP_PATH,
    origin_px: tuple[int, int] = FLOORMAP_ORIGIN_PX,
    scale: float = FLOORMAP_SCALE,
    output_path: Path | str = Path("output/particle_filter.mp4"),
    fps: int = 10,
    landmark: LandmarkCorrectionResult | None = None,
) -> None:
    """PF の各ステップのパーティクル分布をフロアマップ上に描画し MP4 として保存する。

    Args:
        all_particles: 全ステップのパーティクル位置 shape=(T, N, 2)
        mean_trajectory: 平均優先・壁際祖先フォールバック軌跡
            （互換性のため既存引数名を維持）
        gx_mean: X軸重力成分の平均値
        gz_mean: Z軸重力成分の平均値
        floormap_path: フロアマップ画像のパス
        origin_px: 軌跡起点のピクセル座標
        scale: 1ピクセルあたりのメートル数
        output_path: 出力ファイルパス（.mp4）
        fps: フレームレート
        landmark: BLE ランドマーク補正結果。粒子が寄る先を確認できるよう、
            ランドマーク位置を全フレームへ、補正が起きた歩を該当フレーム以降へ描く
    """
    from matplotlib.animation import FFMpegWriter, PillowWriter  # noqa: PLC0415

    configure_japanese_font()
    map_img = plt.imread(Path(floormap_path))
    representative_arr = np.array(mean_trajectory)  # shape: (T, 2)

    fig, ax = plt.subplots(figsize=(7, 7))

    def update(frame: int) -> list[Artist]:
        ax.cla()
        ax.imshow(map_img)

        # 全パーティクルを半透明グレーで描画
        px_p, py_p = compute_pixel_coords(
            all_particles[frame, :, 0],
            all_particles[frame, :, 1],
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        ax.scatter(px_p, py_p, s=30, c="cyan", alpha=0.5, zorder=2)

        # ステップ 0 〜 現在の選択軌跡を青線で描画
        if frame > 0:
            px_m, py_m = compute_pixel_coords(
                representative_arr[: frame + 1, 0],
                representative_arr[: frame + 1, 1],
                gx_mean,
                gz_mean,
                origin_px,
                scale,
            )
            ax.plot(px_m, py_m, "b-", linewidth=1.5, zorder=3)

        # 現ステップの選択位置を赤点で描画
        px_c, py_c = compute_pixel_coords(
            representative_arr[frame : frame + 1, 0],
            representative_arr[frame : frame + 1, 1],
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        ax.scatter(px_c, py_c, s=60, c="red", zorder=4)
        if landmark is not None:
            _draw_animation_landmarks(
                ax,
                landmark,
                frame,
                gx_mean,
                gz_mean,
                origin_px,
                scale,
            )
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"ステップ {frame}")
        return []

    anim = FuncAnimation(fig, update, frames=len(all_particles), interval=1000 // fps)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(str(output_path), writer=FFMpegWriter(fps=fps))
        print(f"Animation saved to {output_path}")
    except Exception:
        gif_path = output_path.with_suffix(".gif")
        anim.save(str(gif_path), writer=PillowWriter(fps=fps))
        print(f"ffmpeg が見つかりません。GIF として保存しました: {gif_path}")
    plt.close(fig)
