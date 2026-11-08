"""一様初期化PFの探索・収束・追従を説明する動画。

役割:
    粒子分布とクラスタ指標、伸びる代表軌跡を同時に示し、収束の経緯を伝える。
依存元:
    common の共有結果型・座標変換・地図正規化、Matplotlib、NumPy と
    plot.lib.animation の共通保存関数を使用する。
利用先:
    plot.pipeline が uniform モードのアニメーション保存時に呼び出す。
処理フロー:
    診断と粒子履歴を対応付け、先頭・収束歩・末尾に静止区間を挿入する。
    地図と指標グラフ、フェーズ文を更新し、MP4優先で保存する。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.patches import Circle

from ...common.config import (
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    PF_CONVERGENCE_CELL_M,
)
from ...common.lib.floormap import compute_pixel_coords, normalize_floormap_gray
from ...common.lib.models import ParticleFilterResult
from ...matplotlib_config import configure_japanese_font
from .animation import save_animation_with_fallback


def save_convergence_animation(
    particle: ParticleFilterResult,
    gx_mean: float,
    gz_mean: float,
    floormap_path: str | Path = FLOORMAP_PATH,
    origin_px: tuple[int, int] = FLOORMAP_ORIGIN_PX,
    scale: float = FLOORMAP_SCALE,
    output_path: Path | str = Path("output/convergence_story.mp4"),
    fps: int = 6,
    initial_hold_frames: int | None = None,
    convergence_hold_frames: int | None = None,
    final_hold_frames: int | None = None,
    cell_m: float = PF_CONVERGENCE_CELL_M,
) -> Path:
    """診断ごとに1フレーム進め、静止区間付きの説明動画を保存する。

    hold 引数は追加するフレーム数で、None は順に fps の2・3・3倍とする。
    粒子履歴・代表軌跡は先頭に初期配置を含み、以後は診断列の順に対応する。
    元の歩番号は診断から取得するため、欠けた歩番号も詰めて表示しない。
    最大クラスタの色分けは重心からの距離による近似である。
    """
    holds = (
        fps * 2 if initial_hold_frames is None else initial_hold_frames,
        fps * 3 if convergence_hold_frames is None else convergence_hold_frames,
        fps * 3 if final_hold_frames is None else final_hold_frames,
    )
    diagnostics = particle.convergence_diagnostics
    count = len(diagnostics)
    if fps <= 0 or min(holds) < 0 or scale <= 0 or cell_m <= 0:
        raise ValueError("fps・縮尺・セル幅は正、hold は非負にしてください")
    if not count:
        raise ValueError("収束診断が必要です")
    if (
        len(particle.all_particles) != count + 1
        or len(particle.trajectory) != count + 1
    ):
        raise ValueError(
            "粒子履歴・代表軌跡は診断列に初期配置1要素を加えた長さが必要です"
        )
    convergence = particle.convergence
    convergence_index = next(
        (
            index
            for index, diagnostic in enumerate(diagnostics)
            if convergence is not None
            and diagnostic.step_number == convergence.step_number
        ),
        None,
    )
    if convergence is not None and (
        convergence_index is None
        or convergence.dominant_centroid_x_m is None
        or convergence.dominant_centroid_y_m is None
    ):
        raise ValueError("収束診断には対応する歩と重心が必要です")

    # 初期配置ではなく、要求どおり1歩目の状態を先頭の説明中にも表示する。
    frames = [(0, "initial", 0)] * holds[0]
    for index in range(count):
        frames.append((index, "step", 0))
        if index == convergence_index:
            frames.extend((index, "convergence", tick) for tick in range(holds[1]))
    frames.extend([(count - 1, "final", 0)] * holds[2])

    configure_japanese_font()
    map_gray = normalize_floormap_gray(plt.imread(Path(floormap_path)))
    trajectory = np.asarray(particle.trajectory)

    def pixels(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return compute_pixel_coords(
            points[:, 0], points[:, 1], gx_mean, gz_mean, origin_px, scale
        )

    fig = plt.figure(figsize=(13, 8), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=(1.7, 1))
    map_ax = fig.add_subplot(grid[:, 0])
    cluster_ax = fig.add_subplot(grid[0, 1])
    spread_ax = fig.add_subplot(grid[1, 1])
    text_ax = fig.add_subplot(grid[2, 1])
    text_ax.set_axis_off()
    phase_text = text_ax.text(0, 0.9, "", va="top", fontsize=12, linespacing=1.7)
    steps = [item.step_number for item in diagnostics]
    cursors = []
    for ax, values, label in (
        (cluster_ax, [item.cluster_count for item in diagnostics], "候補クラスタ数"),
        (
            spread_ax,
            [item.dominant_spread_rms_m for item in diagnostics],
            "最大クラスタ広がり [m]",
        ),
    ):
        ax.plot(steps, values)
        ax.set_ylabel(label)
        ax.set_xlabel("歩番号")
        ax.grid(alpha=0.3)
        cursors.append(ax.axvline(steps[0], color="black", label="現在歩"))
        if convergence is not None:
            ax.axvline(
                convergence.step_number, color="red", linestyle="--", label="収束 K"
            )
        ax.legend(loc="best", fontsize=8)

    def update(frame: tuple[int, str, int]) -> list[Artist]:
        index, phase, tick = frame
        diagnostic = diagnostics[index]
        step = diagnostic.step_number
        points = np.asarray(particle.all_particles[index + 1])
        dominant = np.zeros(len(points), dtype=bool)
        center = None
        if (
            diagnostic.dominant_centroid_x_m is not None
            and diagnostic.dominant_centroid_y_m is not None
        ):
            center = np.array(
                [[diagnostic.dominant_centroid_x_m, diagnostic.dominant_centroid_y_m]]
            )
            radius = max(2 * diagnostic.dominant_spread_rms_m, cell_m)
            dominant = np.linalg.norm(points - center, axis=1) <= radius
        map_ax.clear()
        map_ax.imshow(map_gray > 128, cmap="gray", vmin=0, vmax=1)
        for mask, color, label in (
            (~dominant, "gray", "その他の粒子"),
            (dominant, "darkorange", "最大クラスタ（近似）"),
        ):
            map_ax.scatter(*pixels(points[mask]), s=10, c=color, alpha=0.5, label=label)
        map_ax.plot(
            *pixels(trajectory[: index + 2]), color="royalblue", label="代表軌跡"
        )
        if center is not None:
            map_ax.scatter(
                *pixels(center), marker="*", s=180, c="red", label="最大重心"
            )

        text = (
            f"探索中\n候補クラスタ {diagnostic.cluster_count} 個\n"
            f"最大クラスタ広がり {diagnostic.dominant_spread_rms_m:.1f} m\n"
            f"生存粒子 {diagnostic.surviving_particle_count}"
        )
        if convergence is not None and step >= convergence.step_number:
            location = np.array(
                [
                    [
                        convergence.dominant_centroid_x_m,
                        convergence.dominant_centroid_y_m,
                    ]
                ],
                dtype=float,
            )
            cx, cy = pixels(location)
            pulse = 1 + 0.4 * np.sin(2 * np.pi * tick / fps)
            map_ax.add_patch(
                Circle(
                    (float(cx[0]), float(cy[0])),
                    radius=cell_m / scale * (pulse if phase == "convergence" else 1),
                    fill=False,
                    color="red",
                    linewidth=2,
                    label="収束位置",
                )
            )
            if step == convergence.step_number:
                text = (
                    f"収束 K={step}\n"
                    f"位置 ({location[0, 0]:.2f}, {location[0, 1]:.2f}) m\n"
                    f"重み比 {convergence.dominant_weight_ratio:.2f}"
                )
            else:
                distance = float(np.linalg.norm(trajectory[index + 1] - location[0]))
                text = (
                    f"追従中\n収束位置から {distance:.1f} m 移動\n"
                    f"候補クラスタ {diagnostic.cluster_count} 個"
                )
        if phase == "initial":
            text = (
                "初期状態: 初期位置・初期方位を与えず、\n地図の歩行可能領域へ一様配置"
            )
        elif phase == "final":
            text = f"最終軌跡: 全{steps[-1]}歩"
        if convergence is None and index == count - 1 and phase != "initial":
            text += "\n未収束のまま終了"
        phase_text.set_text(text)
        for cursor in cursors:
            cursor.set_xdata([step, step])
        map_ax.legend(loc="upper right", fontsize=8)
        fig.suptitle(f"初期位置・初期方位なしPF: 収束の経緯 — 歩番号 {step}")
        return []

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    return save_animation_with_fallback(anim, fig, output_path, fps)
