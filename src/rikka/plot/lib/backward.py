"""独立した前後PFの比較成果物を保存する。

役割:
    既存成果物を変更せず、後方軌跡・歩別距離・地図上の比較図を追加する。
依存元:
    commonの共有結果型と地図座標変換、共通CSV整形、Matplotlibを使用する。
利用先:
    plot.pipeline が後方PF結果のあるときだけ呼び出す。
処理フロー:
    元の歩時刻でCSVを保存し、前後経路・端点・大きい不一致を描画して閉じる。
"""

from dataclasses import asdict, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...common.lib.floormap import compute_pixel_coords
from ...common.lib.models import BidirectionalAgreement, FloorMap, TrajectoryResult
from ...matplotlib_config import configure_japanese_font
from .outputs import _build_trajectory_dataframe


def save_backward_csv(result: TrajectoryResult, output_dir: Path) -> None:
    """前向きの実行歩時刻と同じ列形式で後方軌跡を保存する。"""
    assert result.particle is not None and result.particle.backward is not None
    backward = result.particle.backward
    _build_trajectory_dataframe(backward.trajectory, result.t_at_steps).to_csv(
        output_dir / "backward_trajectory.csv", index=False
    )
    pd.DataFrame(
        [asdict(item) for item in backward.agreement],
        columns=[field.name for field in fields(BidirectionalAgreement)],
    ).to_csv(output_dir / "bidirectional_agreement.csv", index=False)


def save_backward_comparison(
    result: TrajectoryResult, floormap: FloorMap, output_path: Path
) -> None:
    """地図上で前向きを青、後方を緑、不一致の大きいノードを赤で示す。"""
    assert result.particle is not None and result.particle.backward is not None
    backward = result.particle.backward
    prepared = result.prepared
    configure_japanese_font()
    fig, ax = plt.subplots(figsize=(9, 9))
    try:
        ax.imshow(plt.imread(floormap.path))
        outliers = np.asarray(
            [
                item.distance_m > 3 * backward.median_distance_m
                for item in backward.agreement
            ]
        )
        for points, label, color in (
            (result.trajectory, "前向きPF", "blue"),
            (backward.trajectory, "後方PF", "green"),
        ):
            xy = np.asarray(points)
            x, y = compute_pixel_coords(
                xy[:, 0],
                xy[:, 1],
                prepared.gx_mean,
                prepared.gz_mean,
                floormap.origin_px,
                floormap.scale,
            )
            ax.plot(x, y, color=color, label=label)
            ax.scatter(x[0], y[0], marker="o", color=color, label=f"{label} 開始")
            ax.scatter(x[-1], y[-1], marker="X", color=color, label=f"{label} 終了")
            ax.scatter(
                np.asarray(x)[outliers], np.asarray(y)[outliers], color="red", zorder=5
            )
        ax.set_title(
            f"開始点距離 {backward.start_distance_m:.2f} m / "
            f"中央値 {backward.median_distance_m:.2f} m / "
            f"最大値 {backward.max_distance_m:.2f} m\n"
            "赤: 距離が中央値の3倍を超える歩"
        )
        ax.legend()
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
    finally:
        plt.close(fig)
