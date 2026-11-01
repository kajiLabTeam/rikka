"""ランドマークのみの経路の描画。

役割:
    ランドマークを指定順に直線で結んだ経路をフロアマップ上に描き、各地点の
    beacon_id と区間距離を重ねた画像として表示・保存する。
依存元:
    ``common.config`` の地図既定値、``common.lib.models.LandmarkPath``、
    ``matplotlib_config`` の日本語フォント設定、Matplotlib を利用する。
利用先:
    ``plot.pipeline.render_landmark_path()`` が CLI のランドマーク経路出力で
    使用する。
処理フロー:
    フロアマップを背景に描き、経路線と地点、始終点、区間距離の注記を重ねて、
    出力先が指定されていれば PNG を保存する。
"""

from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt

from ...common.config import FLOORMAP_PATH
from ...common.lib.models import LandmarkPath
from ...matplotlib_config import configure_japanese_font


def plot_landmark_path(
    path: LandmarkPath,
    floormap_path: str | Path = FLOORMAP_PATH,
    output_dir: Path | None = None,
) -> None:
    """ランドマーク経路をフロアマップ上に直線で結んで描画する。"""
    configure_japanese_font()
    pixel_xs = [point.pixel_x for point in path.points]
    pixel_ys = [point.pixel_y for point in path.points]

    fig, ax = plt.subplots(figsize=(7, 7))
    map_img = plt.imread(Path(floormap_path))
    ax.imshow(map_img)

    ax.plot(
        pixel_xs,
        pixel_ys,
        color="tab:blue",
        linewidth=2.0,
        zorder=2,
        label="ランドマーク経路",
    )
    ax.scatter(
        pixel_xs,
        pixel_ys,
        s=60,
        color="tab:blue",
        edgecolors="white",
        linewidths=1.0,
        zorder=3,
    )
    ax.plot(
        pixel_xs[0],
        pixel_ys[0],
        "go",
        markersize=10,
        zorder=4,
        label="開始ランドマーク",
    )
    ax.plot(
        pixel_xs[-1],
        pixel_ys[-1],
        "rs",
        markersize=10,
        zorder=4,
        label="終了ランドマーク",
    )

    for point in path.points:
        # UUID 形式の beacon_id は図では長すぎるため、先頭だけを表示する。
        name = (
            point.beacon_id if len(point.beacon_id) <= 16 else f"{point.beacon_id[:8]}…"
        )
        label = ax.annotate(
            f"{point.order + 1}: {name}",
            (point.pixel_x, point.pixel_y),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            color="black",
            zorder=5,
        )
        label.set_path_effects(
            [path_effects.withStroke(linewidth=2.0, foreground="white")]
        )

    # 区間距離は経路の各線分の中点へ置き、経路長の読み取りに使う。
    for previous, current in zip(path.points[:-1], path.points[1:], strict=True):
        distance = ax.annotate(
            f"{current.segment_length_m:.1f} m",
            (
                (previous.pixel_x + current.pixel_x) / 2.0,
                (previous.pixel_y + current.pixel_y) / 2.0,
            ),
            textcoords="offset points",
            xytext=(0, -12),
            fontsize=8,
            color="tab:blue",
            ha="center",
            zorder=5,
        )
        distance.set_path_effects(
            [path_effects.withStroke(linewidth=2.0, foreground="white")]
        )

    ax.set_title(f"ランドマークのみの経路（全長 {path.total_length_m:.2f} m）")
    ax.legend()
    plt.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir / "landmark_path.png", dpi=150, bbox_inches="tight")
    plt.show()
