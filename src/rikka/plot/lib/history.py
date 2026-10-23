"""PFの元軌跡と独立した過去修正候補を比較して保存する。

役割:
    試作の候補を通常のtrajectory.csvと分離し、修正量・支持・再配置を記録する。
依存元:
    commonの共有結果型・地図座標変換、Pandas、NumPy、Matplotlibを利用する。
利用先:
    plot.pipelineが履歴候補のある実行だけでCSVと比較PNGの保存に使用する。
処理フロー:
    対応表・集計・再配置CSVを保存し、元と候補の地図上の経路と歩間変位を描く。
    図はファイルへ保存して閉じ、対話ウィンドウは開かない。
"""

from dataclasses import asdict, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...common.lib.floormap import compute_pixel_coords
from ...common.lib.models import (
    FloorMap,
    ParticleHistoryCandidate,
    ParticleRelocationEvent,
)


def save_history_csv(candidate: ParticleHistoryCandidate, output_dir: Path) -> None:
    """原点を含む全時点の元座標・候補座標・センサー時刻を別CSVへ保存する。"""
    records = pd.DataFrame([asdict(step) for step in candidate.steps])
    records.to_csv(output_dir / "particle_history_candidate.csv", index=False)
    pd.DataFrame(
        [asdict(event) for event in candidate.relocations],
        columns=[field.name for field in fields(ParticleRelocationEvent)],
    ).to_csv(output_dir / "particle_relocations.csv", index=False)
    summary = {
        "status": candidate.status,
        "search_radius_m": candidate.search_radius_m,
        "max_candidates": candidate.max_candidates,
        "objective_original": candidate.objective_original,
        "objective_candidate": candidate.objective_candidate,
        "changed_points": int((records.correction_m > 1e-8).sum()),
        "max_correction_m": float(records.correction_m.max()),
        "hypothesis_switches": int((records.connection == "hypothesis_switch").sum()),
        "particle_supported_points": int(records.source_particle.notna().sum()),
        "normal_relocation_events": len(candidate.relocations),
    }
    for kind in ("original", "candidate"):
        displacement = records[f"{kind}_displacement_m"].iloc[1:]
        summary[f"{kind}_length_m"] = float(displacement.sum())
        summary[f"{kind}_max_displacement_m"] = (
            float(displacement.max()) if len(displacement) else 0.0
        )
        summary[f"{kind}_stationary_steps"] = int((displacement <= 1e-8).sum())
        summary[f"{kind}_steps_below_0_1m"] = int((displacement < 0.1).sum())
        summary[f"{kind}_min_displacement_m"] = (
            float(displacement.min()) if len(displacement) else 0.0
        )
    pd.DataFrame([summary]).to_csv(
        output_dir / "particle_history_summary.csv", index=False
    )


def save_history_comparison(
    candidate: ParticleHistoryCandidate,
    floormap: FloorMap,
    gx_mean: float,
    gz_mean: float,
    output_path: Path,
) -> None:
    """地図上の候補と元軌跡、および歩間変位を明示して保存する。"""
    original = np.array(
        [[step.original_x_m, step.original_y_m] for step in candidate.steps]
    )
    corrected = np.asarray(candidate.trajectory)
    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    try:
        axes[0].imshow(plt.imread(floormap.path))
        for points, label, style, color in (
            (original, "Forward PF", "--", "gray"),
            (corrected, "History candidate", "-", "tab:orange"),
        ):
            x, y = compute_pixel_coords(
                points[:, 0],
                points[:, 1],
                gx_mean,
                gz_mean,
                floormap.origin_px,
                floormap.scale,
            )
            axes[0].plot(x, y, style, color=color, linewidth=1.5, label=label)
        axes[0].legend(fontsize=8)
        axes[0].set_title("Experimental: particle-supported path")
        axes[0].axis("off")
        axes[1].plot(
            [s.step for s in candidate.steps[1:]],
            [s.original_displacement_m for s in candidate.steps[1:]],
            "--",
            color="gray",
            label="Forward PF",
        )
        axes[1].plot(
            [s.step for s in candidate.steps[1:]],
            [s.candidate_displacement_m for s in candidate.steps[1:]],
            color="tab:orange",
            label="History candidate",
        )
        axes[1].set_xlabel("Source step")
        axes[1].set_ylabel("Displacement per step [m]")
        axes[1].legend()
        axes[1].grid(alpha=0.2)
        axes[1].set_title(candidate.status)
        fig.suptitle("Same forward particles; candidate is not verified ground truth")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
    finally:
        plt.close(fig)
