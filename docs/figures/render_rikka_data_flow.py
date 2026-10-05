"""前年度論文の図解表現を参考に、Rikka の概要フローを描画する。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Arc,
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Rectangle,
)
from matplotlib.path import Path as MplPath

OUTPUT_DIR = Path(__file__).parent
STEM = "rikka_data_flow"
INK = "#111111"
MUTED = "#5B6470"
LIGHT = "#F3F4F6"


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
    connection: str = "arc3",
    width: float = 2.1,
) -> None:
    """参照図に合わせた太い方向矢印を描く。"""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=width,
            linestyle="--" if dashed else "-",
            color=MUTED if dashed else INK,
            connectionstyle=connection,
            shrinkA=1,
            shrinkB=1,
            zorder=1,
        )
    )


def draw_phone(ax: plt.Axes, x: float, y: float) -> None:
    """加速度・角速度を取得するスマートフォンを描く。"""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            10,
            20,
            boxstyle="round,pad=0.2,rounding_size=1.4",
            facecolor="white",
            edgecolor=INK,
            linewidth=2.3,
        )
    )
    ax.add_patch(Rectangle((x + 1.2, y + 3.1), 7.6, 13.7, fill=False, ec=INK, lw=1.5))
    ax.plot([x + 4.1, x + 5.9], [y + 18.1, y + 18.1], color=INK, lw=1.5)
    ax.add_patch(Circle((x + 5, y + 1.5), 0.55, fill=False, ec=INK, lw=1.4))
    t = np.linspace(0, 1, 120)
    ax.plot(
        x + 2 + 6 * t,
        y + 11.8 + 1.1 * np.sin(3.2 * np.pi * t),
        color=INK,
        lw=1.4,
    )
    ax.plot(
        x + 2 + 6 * t,
        y + 7.2 + 1.2 * np.cos(4.0 * np.pi * t),
        color=INK,
        lw=1.4,
    )


def draw_database(ax: plt.Axes, x: float, y: float) -> None:
    """CSVとして保存されたセンサーデータを描く。"""
    ax.add_patch(Rectangle((x, y + 2), 10, 12, facecolor=LIGHT, edgecolor=INK, lw=2.2))
    ax.add_patch(
        Ellipse((x + 5, y + 14), 10, 4.2, facecolor="white", edgecolor=INK, lw=2.2)
    )
    ax.add_patch(
        Ellipse((x + 5, y + 2), 10, 4.2, facecolor=LIGHT, edgecolor=INK, lw=2.2)
    )
    ax.add_patch(Arc((x + 5, y + 8), 10, 4.2, theta1=180, theta2=360, ec=INK, lw=1.7))
    ax.add_patch(Arc((x + 5, y + 12), 10, 4.2, theta1=180, theta2=360, ec=INK, lw=1.7))


def draw_pdr(ax: plt.Axes, x: float, y: float) -> None:
    """歩行検出と移動量推定を足跡で表す。"""
    for dx, dy, angle in ((1.5, 2.0, 24), (5.3, 6.0, -18), (2.9, 10.3, 22)):
        ax.add_patch(Ellipse((x + dx, y + dy), 2.5, 4.8, angle=angle, fc=INK, ec=INK))
        ax.add_patch(Circle((x + dx + 0.9, y + dy + 2.1), 0.52, fc=INK, ec=INK))
    ax.add_patch(
        FancyArrowPatch(
            (x + 6.7, y + 1),
            (x + 9.8, y + 12.8),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=2,
            color=INK,
        )
    )


def draw_floor_map(ax: plt.Axes, x: float, y: float) -> None:
    """歩行可能領域を与える簡略フロアマップを描く。"""
    ax.add_patch(Rectangle((x, y), 11, 10, fill=False, ec=INK, lw=2))
    ax.plot(
        [x + 3, x + 3, x + 7.5, x + 7.5],
        [y, y + 6.5, y + 6.5, y + 10],
        color=INK,
        lw=2,
    )
    ax.plot([x + 3, x + 6], [y + 3, y + 3], color=INK, lw=2)
    ax.plot([x + 7.5, x + 11], [y + 3.8, y + 3.8], color=INK, lw=2)
    ax.add_patch(Rectangle((x + 2.65, y + 4.8), 0.7, 1.5, fc="white", ec="white"))
    ax.add_patch(Rectangle((x + 7.15, y + 7.1), 0.7, 1.5, fc="white", ec="white"))


def draw_particle_filter(ax: plt.Axes, x: float, y: float) -> None:
    """地図上の候補群と代表軌跡でパーティクルフィルタを表す。"""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            13,
            17,
            boxstyle="round,pad=0.2,rounding_size=1.2",
            facecolor=LIGHT,
            edgecolor=INK,
            linewidth=2.2,
        )
    )
    ax.plot(
        [x + 2, x + 2, x + 7, x + 7, x + 11],
        [y + 2, y + 13, y + 13, y + 5, y + 5],
        color=MUTED,
        lw=1.4,
    )
    particles = (
        (3.0, 4.0),
        (4.1, 5.2),
        (3.4, 7.0),
        (5.0, 8.2),
        (5.7, 10.1),
        (7.6, 10.4),
        (8.8, 8.5),
        (9.7, 7.2),
        (10.4, 6.1),
    )
    for px, py in particles:
        ax.add_patch(Circle((x + px, y + py), 0.38, fc=INK, ec=INK))
    path = MplPath(
        [
            (x + 2.8, y + 3.2),
            (x + 4.0, y + 6.5),
            (x + 6.0, y + 10),
            (x + 10.5, y + 6.5),
        ],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(PathPatch(path, fill=False, ec=INK, lw=2.2))


def draw_trajectory(ax: plt.Axes, x: float, y: float) -> None:
    """推定軌跡と歩行者を描く。"""
    path = MplPath(
        [(x, y + 2), (x + 4, y + 9), (x + 8, y - 1), (x + 13, y + 6)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(PathPatch(path, fill=False, ec=INK, lw=2.4))
    hx, hy = x + 15, y + 11
    ax.add_patch(Circle((hx, hy + 3.0), 1.0, fc=INK, ec=INK))
    ax.plot(
        [hx, hx - 0.5],
        [hy + 2, hy - 2],
        color=INK,
        lw=3.0,
        solid_capstyle="round",
    )
    ax.plot(
        [hx - 0.2, hx - 2.4],
        [hy + 0.3, hy - 1.2],
        color=INK,
        lw=2.7,
        solid_capstyle="round",
    )
    ax.plot(
        [hx - 0.1, hx + 2.1],
        [hy + 0.5, hy - 0.5],
        color=INK,
        lw=2.7,
        solid_capstyle="round",
    )
    ax.plot(
        [hx - 0.5, hx - 2.1],
        [hy - 2, hy - 5],
        color=INK,
        lw=2.9,
        solid_capstyle="round",
    )
    ax.plot(
        [hx - 0.5, hx + 1.5],
        [hy - 2, hy - 4.6],
        color=INK,
        lw=2.9,
        solid_capstyle="round",
    )


def build_figure() -> plt.Figure:
    """視覚的なアイコンを中心にRikkaの主要なデータフローを構成する。"""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Hiragino Sans",
                "Yu Gothic",
                "Noto Sans CJK JP",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure = plt.figure(figsize=(7.087, 3.55), facecolor="white")
    ax = figure.add_axes((0.015, 0.035, 0.97, 0.94))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    draw_phone(ax, 3, 33)
    ax.text(8, 28.1, "スマートフォン", ha="center", va="center", fontsize=12, color=INK)
    ax.text(
        8,
        23.8,
        "加速度・角速度",
        ha="center",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )

    draw_database(ax, 23, 36)
    ax.text(
        28, 28.1, "センサーデータ", ha="center", va="center", fontsize=12, color=INK
    )

    draw_pdr(ax, 46, 39)
    ax.text(51, 34.0, "PDR", ha="center", va="center", fontsize=12.5, color=INK)
    ax.text(
        51,
        29.8,
        "歩行検出・歩幅・移動方向",
        ha="center",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )

    draw_particle_filter(ax, 46, 8)
    ax.text(
        52.5,
        3.7,
        "パーティクルフィルタ",
        ha="center",
        va="center",
        fontsize=11.3,
        color=INK,
    )

    draw_floor_map(ax, 27, 12)
    ax.text(
        32.5, 7.5, "フロアマップ", ha="center", va="center", fontsize=11.3, color=INK
    )

    draw_trajectory(ax, 81, 35)
    ax.text(89, 26.8, "推定軌跡", ha="center", va="center", fontsize=12.5, color=INK)

    arrow(ax, (13.5, 43), (22.5, 43))
    arrow(ax, (33.5, 43), (45.5, 46))

    branch_points = [(33.5, 43), (41, 43), (41, 21.5), (45.5, 21.5)]
    branch_path = MplPath(
        branch_points,
        [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO],
    )
    ax.add_patch(
        FancyArrowPatch(
            path=branch_path,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.1,
            color=INK,
            zorder=1,
        )
    )
    arrow(ax, (38.5, 15), (45.5, 15), width=1.9)

    arrow(ax, (56.5, 46), (72, 46), width=2)
    arrow(ax, (59.5, 18.5), (72, 18.5), width=2)
    ax.plot([72, 72], [18.5, 46], color=INK, lw=2.1)
    arrow(ax, (72, 32.2), (81.2, 41.5), width=2.2)
    return figure


def main() -> None:
    """SVG・PDF・300dpi PNGを同じ図版から生成する。"""
    figure = build_figure()
    figure.savefig(OUTPUT_DIR / f"{STEM}.svg", facecolor="white")
    figure.savefig(OUTPUT_DIR / f"{STEM}.pdf", facecolor="white")
    figure.savefig(
        OUTPUT_DIR / f"{STEM}.png",
        dpi=300,
        facecolor="white",
        pil_kwargs={"compress_level": 6},
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
