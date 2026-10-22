"""PF の段階別診断 CSV から再検証用の静止画を生成する。

役割:
    製品アルゴリズムや乱数状態を変更せず、診断結果の粒子分布と軌跡を比較する。
入力:
    専用実行ディレクトリの trajectory_comparison.csv、step_diagnostics.csv、
    particle_stages.csv.gz、および地図座標変換を含む metadata 辞書。
出力:
    同ディレクトリの figures/ に PNG と主要図の SVG を保存する。
処理フロー:
    CSV を読み込み、地図をメートル座標へ配置し、全体推移と自動抽出した
    異常歩の前後における段階別の粒子位置・重みを可視化する。
"""

from pathlib import Path

from rikka.matplotlib_config import (
    configure_japanese_font,
    configure_matplotlib_cache,
)

configure_matplotlib_cache()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _save(fig, directory, name, svg=True):
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=170)
    if svg:
        fig.savefig(directory / f"{name}.svg")
    plt.close(fig)


def _map(ax, metadata):
    path = Path(metadata["floormap_path"])
    image = mpimg.imread(path)
    height, width = image.shape[:2]
    ox, oy = metadata["origin_px"]
    scale = metadata["scale_m_per_px"]
    sign = metadata["y_sign"]
    extent = (
        -ox * scale,
        (width - ox) * scale,
        sign * (height - oy) * scale,
        -sign * oy * scale,
    )
    ax.imshow(image, extent=extent, origin="upper", alpha=0.55, zorder=0)
    ax.set_ylim(sorted(extent[2:]))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]（PF 座標）")
    ax.set_ylabel("y [m]（PF 座標）")
    ax.grid(alpha=0.2)


def _series(ax, df, names):
    for column, label in names:
        if column in df:
            ax.plot(df["step"], df[column], label=label, linewidth=1)
    ax.set_xlabel("歩番号")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def plot_run(run_dir: Path, metadata: dict):
    """記録済み CSV のみを使い、要求された8種類の診断図を保存する。"""
    configure_japanese_font()
    plt.rcParams["font.size"] = 9
    run_dir = Path(run_dir)
    destination = run_dir / "figures"
    destination.mkdir(exist_ok=True)
    trajectory = pd.read_csv(run_dir / "trajectory_comparison.csv")
    steps = pd.read_csv(run_dir / "step_diagnostics.csv")
    particle_path = run_dir / "particle_stages.csv.gz"
    if not particle_path.exists():
        particle_path = run_dir / "particle_stages.csv"
    header = pd.read_csv(particle_path, nrows=0)
    desired = {
        "step",
        "particle_id",
        "parent_particle_id",
        "before_x",
        "before_y",
        "predicted_x",
        "predicted_y",
        "relocated_x",
        "relocated_y",
        "after_x",
        "after_y",
        "collision",
        "relocation_success",
        "before_weight",
        "posterior_weight",
        "after_weight",
        "pdr_angle_diff_deg",
        "normal_pdr_angle_diff_deg",
        "internal_angle_diff_deg",
        "realized_distance_m",
        "distance_ratio",
        "pdr_step_m",
    }
    particles = pd.read_csv(
        particle_path, usecols=sorted(desired & set(header.columns))
    )
    relocated = particles.loc[
        particles["relocation_success"].fillna(False).astype(bool)
    ]

    fig, ax = plt.subplots(figsize=(10, 9))
    _map(ax, metadata)
    ax.autoscale(enable=True, axis="both")
    colors = {
        "pdr": "#888888",
        "current": "#0072B2",
        "sequence": "#E69F00",
        "final": "#D55E00",
    }
    for mode, color in colors.items():
        ax.plot(
            trajectory[f"{mode}_x"],
            trajectory[f"{mode}_y"],
            label=mode,
            color=color,
            alpha=0.85,
            linewidth=1.5,
        )
    stride = max(1, len(trajectory) // 15)
    for _, row in trajectory.iloc[::stride].iterrows():
        ax.annotate(
            str(int(row["step"])),
            (row["final_x"], row["final_y"]),
            fontsize=7,
            color=colors["final"],
        )
    ax.set_title("PDR・current・sequence・最終軌跡（数値は歩番号）")
    ax.legend()
    _save(fig, destination, "01_trajectory_overlay")

    fig, ax = plt.subplots(figsize=(13, 4))
    _series(
        ax,
        trajectory,
        [
            ("pdr_step_m", "PDR 歩幅"),
            ("current_step_m", "current 実変位"),
            ("sequence_step_m", "sequence 実変位"),
            ("final_step_m", "最終実変位"),
        ],
    )
    ax.set_ylabel("1歩の移動距離 [m]")
    ax.set_title("入力歩幅と座標差分による移動距離")
    _save(fig, destination, "02_step_distance")

    fig, ax = plt.subplots(figsize=(13, 4))
    _series(
        ax,
        steps,
        [("collision_ratio", "壁衝突率"), ("relocation_ratio", "円内再配置率")],
    )
    ax.set_ylabel("粒子割合")
    ax.set_ylim(0, 1.02)
    ax.set_title("壁衝突・円内再配置の発生率")
    _save(fig, destination, "03_collision_relocation")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if not relocated.empty and "realized_distance_m" in relocated:
        sample = relocated.iloc[:: max(1, len(relocated) // 40000)]
        pdr = (
            sample["pdr_step_m"]
            if "pdr_step_m" in sample
            else sample["step"].map(steps.set_index("step")["pdr_step_m"])
        )
        axes[0].scatter(pdr, sample["realized_distance_m"], s=2, alpha=0.15)
        upper = max(float(pdr.max()), 1)
        axes[0].plot([0, upper], [0, upper], "k--", label="実変位 = PDR 歩幅")
        axes[0].legend()
        if "distance_ratio" in relocated:
            finite = (
                relocated["distance_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
            )
            axes[1].hist(finite, bins=60)
            axes[1].set_xlabel("実変位 / PDR 歩幅（欠損除外）")
    axes[0].set_xlabel("PDR 歩幅 [m]")
    axes[0].set_ylabel("円内再配置後の実変位 [m]")
    axes[0].set_title("再配置成功粒子（散布図は規則間引き）")
    axes[1].set_ylabel("粒子数")
    fig.suptitle("円内再配置後の移動距離")
    _save(fig, destination, "04_relocation_distance")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, column, label in zip(
        axes,
        ["normal_pdr_angle_diff_deg", "pdr_angle_diff_deg", "internal_angle_diff_deg"],
        ["通常 PDR 方位との差", "PF 入力方位との差", "内部 theta 方位との差"],
        strict=True,
    ):
        if column in relocated:
            ax.hist(relocated[column].dropna(), bins=np.linspace(0, 180, 37))
        else:
            ax.text(0.5, 0.5, "未記録", transform=ax.transAxes, ha="center")
        for boundary in [90, 135]:
            ax.axvline(boundary, color="red", alpha=0.4, linewidth=0.8)
        ax.set_xlabel(f"{label} [deg]")
        ax.set_ylabel("再配置成功粒子数")
    fig.suptitle("実移動方位の最小絶対角差（90°・135°を表示）")
    _save(fig, destination, "05_relocation_heading")

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    _series(
        axes[0],
        steps,
        [
            ("ess_before_observation", "観測前 ESS"),
            ("ess_after_observation", "観測後 ESS"),
            ("ess_after_resampling", "再標本化後 ESS"),
        ],
    )
    _series(
        axes[1],
        steps,
        [
            ("unique_parent_count", "直接親数"),
            ("ancestors_5", "5歩前祖先数"),
            ("ancestors_10", "10歩前祖先数"),
            ("unique_position_count", "一意位置数"),
        ],
    )
    _series(axes[2], steps, [("position_spread_rms_m", "位置分散 RMS [m]")])
    _series(
        axes[3],
        steps,
        [
            ("full_resampling", "全体再標本化"),
            ("partial_replacement_count", "死亡枠補充数"),
        ],
    )
    axes[0].set_ylabel("ESS")
    axes[1].set_ylabel("粒子・祖先数")
    axes[2].set_ylabel("距離 [m]")
    axes[3].set_ylabel("有無 / 数")
    fig.suptitle("ESS・祖先・位置に基づく多様性診断")
    _save(fig, destination, "06_diversity")

    candidates = trajectory.loc[trajectory["step"] > 0]
    centers = []
    if not candidates.empty:
        centers.append(int(candidates.loc[candidates["final_step_m"].idxmax(), "step"]))
        centers.append(
            int(candidates.loc[candidates["current_step_m"].idxmax(), "step"])
        )
        if "final_pdr_angle_diff_deg" in candidates:
            index = candidates["final_pdr_angle_diff_deg"].abs().idxmax()
            if pd.notna(index):
                centers.append(int(candidates.loc[index, "step"]))
        smallest = candidates["final_step_m"] / candidates["pdr_step_m"].replace(
            0, np.nan
        )
        if smallest.notna().any():
            centers.append(int(candidates.loc[smallest.idxmin(), "step"]))
    for center in list(dict.fromkeys(centers))[:4]:
        _particle_panels(particles, trajectory, center, metadata, destination)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    _series(
        axes[0],
        trajectory,
        [
            ("current_step_m", "current"),
            ("sequence_step_m", "sequence"),
            ("final_step_m", "最終"),
        ],
    )
    _series(
        axes[1],
        trajectory,
        [
            ("current_mode_switch", "current モード切替"),
            ("sequence_relocated", "sequence 経路上の再配置"),
        ],
    )
    if "current_mode" in trajectory:
        labels = trajectory["current_mode"].fillna("unknown").astype(str)
        unique = list(dict.fromkeys(labels))
        axes[2].step(
            trajectory["step"],
            labels.map(dict(zip(unique, range(len(unique)), strict=True))),
            where="mid",
        )
        axes[2].set_yticks(range(len(unique)), unique)
    axes[0].set_ylabel("実変位 [m]")
    axes[1].set_ylabel("有無")
    axes[2].set_ylabel("current 選択モード")
    axes[2].set_xlabel("歩番号")
    fig.suptitle("経路選択モードと移動量の対応")
    _save(fig, destination, "08_selection_and_jump")
    return sorted(str(path.relative_to(run_dir)) for path in destination.glob("*"))


def _particle_panels(particles, trajectory, center, metadata, destination):
    window = trajectory.loc[trajectory["step"].between(center - 2, center + 2)]
    subset = particles.loc[particles["step"].isin(window["step"])]
    if window.empty or subset.empty:
        return
    coordinates = []
    for stage in ["before", "predicted", "relocated", "after"]:
        coordinates.append(subset[[f"{stage}_x", f"{stage}_y"]].to_numpy())
    xy = np.concatenate(coordinates)
    low, high = np.nanpercentile(xy, [1, 99], axis=0)
    final_xy = window[["final_x", "final_y"]].to_numpy()
    low = np.minimum(low, np.nanmin(final_xy, axis=0))
    high = np.maximum(high, np.nanmax(final_xy, axis=0))
    margin = max(1.0, float(np.max(high - low)) * 0.12)
    fig, axes = plt.subplots(
        len(window), 4, figsize=(17, 3.5 * len(window)), squeeze=False
    )
    max_weight = max(
        float(subset["after_weight"].max()), float(subset["before_weight"].max()), 1e-12
    )
    last_scatter = None
    for row_number, (_, row) in enumerate(window.iterrows()):
        step = int(row["step"])
        group = subset.loc[subset["step"] == step]
        stages = [
            ("before", "移動前", "before_weight"),
            ("predicted", "PDR 提案", "before_weight"),
            ("relocated", "壁処理後", "posterior_weight"),
            ("after", "再標本化後", "after_weight"),
        ]
        for column, (stage, label, weight) in enumerate(stages):
            ax = axes[row_number, column]
            _map(ax, metadata)
            last_scatter = ax.scatter(
                group[f"{stage}_x"],
                group[f"{stage}_y"],
                c=group[weight],
                cmap="viridis",
                s=9,
                vmin=0,
                vmax=max_weight,
                alpha=0.7,
            )
            affected = group["collision"].fillna(False).astype(bool)
            if stage in ["predicted", "relocated"]:
                ax.scatter(
                    group.loc[affected, f"{stage}_x"],
                    group.loc[affected, f"{stage}_y"],
                    facecolors="none",
                    edgecolors="red",
                    s=18,
                    linewidths=0.4,
                    label="壁衝突 input slot",
                )
            ax.plot(window["final_x"], window["final_y"], "r.-", linewidth=1)
            ax.scatter(
                [row["final_x"]],
                [row["final_y"]],
                marker="x",
                c="red",
                s=45,
                label="最終推定位置",
            )
            ax.set_xlim(low[0] - margin, high[0] + margin)
            ax.set_ylim(low[1] - margin, high[1] + margin)
            ax.set_title(f"歩 {step} / {label}")
            if row_number == 0:
                ax.legend(fontsize=6)
    fig.suptitle(
        f"異常候補 歩 {center} 前後：粒子段階と重み\n"
        "再標本化後は output slot、他段階は input slot（対応は CSV の親 ID）"
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    if last_scatter is not None:
        colorbar_ax = fig.add_axes((0.3, 0.012, 0.4, 0.01))
        fig.colorbar(
            last_scatter,
            cax=colorbar_ax,
            orientation="horizontal",
            label="記録された粒子重み（段階ごとに正規化条件が異なる）",
        )
    fig.savefig(destination / f"07_particle_stages_step_{center:04d}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    _map(ax, metadata)
    center_particles = subset.loc[subset["step"] == center]
    scatter = ax.scatter(
        center_particles["after_x"],
        center_particles["after_y"],
        c=center_particles["after_weight"],
        cmap="viridis",
        s=18,
        alpha=0.55,
        label=f"歩 {center} 再標本化後の粒子",
    )
    for mode, color in [
        ("pdr", "gray"),
        ("current", "#0072B2"),
        ("sequence", "#E69F00"),
        ("final", "#D55E00"),
    ]:
        ax.plot(
            window[f"{mode}_x"],
            window[f"{mode}_y"],
            ".-",
            color=color,
            label=mode,
            linewidth=1.5,
        )
    for _, row in window.iterrows():
        ax.annotate(
            str(int(row["step"])),
            (row["final_x"], row["final_y"]),
            fontsize=10,
            color="#D55E00",
            xytext=(4, 4),
            textcoords="offset points",
        )
        ax.annotate(
            f"C{int(row['step'])}",
            (row["current_x"], row["current_y"]),
            fontsize=9,
            color="#0072B2",
            xytext=(4, -12),
            textcoords="offset points",
        )
    zoom_xy = np.concatenate([final_xy, window[["current_x", "current_y"]].to_numpy()])
    ax.set_xlim(np.nanmin(zoom_xy[:, 0]) - 2.5, np.nanmax(zoom_xy[:, 0]) + 2.5)
    ax.set_ylim(np.nanmin(zoom_xy[:, 1]) - 2.5, np.nanmax(zoom_xy[:, 1]) + 2.5)
    ax.legend(fontsize=8)
    ax.set_title(f"異常候補 歩 {center} の局所拡大（current・最終の前後±2歩）")
    fig.colorbar(scatter, ax=ax, label="再標本化後の重み", shrink=0.65)
    _save(fig, destination, f"07_event_zoom_step_{center:04d}")


def plot_summary(output: Path):
    """複数データ・seed の集約 CSV から比較図を3種類生成する。"""
    configure_japanese_font()
    output = Path(output)
    destination = output / "figures"
    destination.mkdir(exist_ok=True)
    summary = pd.read_csv(output / "summary.csv")
    labels = summary["dataset"].astype(str) + " / seed " + summary["seed"].astype(str)
    x = np.arange(len(summary))
    fig, axes = plt.subplots(
        2, 1, figsize=(max(12, len(summary) * 0.9), 9), sharex=True
    )
    for i, (mode, color) in enumerate(
        [
            ("pdr", "gray"),
            ("current", "#0072B2"),
            ("sequence", "#E69F00"),
            ("final", "#D55E00"),
        ]
    ):
        axes[0].bar(
            x + (i - 1.5) * 0.2,
            summary[f"{mode}_length_m"],
            width=0.2,
            color=color,
            label=mode,
        )
        column = f"{mode}_max_step_m"
        if column in summary:
            axes[1].bar(
                x + (i - 1.5) * 0.2, summary[column], width=0.2, color=color, label=mode
            )
    axes[0].set_ylabel("軌跡長 / 歩幅合計 [m]")
    axes[1].set_ylabel("最大1歩移動量 [m]")
    axes[0].legend(ncol=4)
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[0].set_title("データ・seed 別の軌跡長と最大位置ジャンプ")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    _save(fig, destination, "summary_01_paths")

    fig, axes = plt.subplots(
        2, 1, figsize=(max(12, len(summary) * 0.9), 9), sharex=True
    )
    bottom = np.zeros(len(summary))
    for label, text, color in [
        ("lt0_5", "0.5倍未満", "#56B4E9"),
        ("0_5to1_5", "0.5～1.5倍", "#009E73"),
        ("1_5to3", "1.5～3倍", "#E69F00"),
        ("gt3", "3倍超", "#D55E00"),
    ]:
        values = summary[f"relocated_ratio_{label}_fraction"]
        axes[0].bar(x, values, bottom=bottom, color=color, label=text)
        bottom += values.to_numpy()
    for i, (column, label) in enumerate(
        [
            ("relocated_pdr_gt90_fraction", "PF入力差 >90°"),
            ("relocated_pdr_gt135_fraction", "PF入力差 >135°"),
            ("relocated_internal_gt90_fraction", "内部theta差 >90°"),
            ("relocated_internal_gt135_fraction", "内部theta差 >135°"),
        ]
    ):
        axes[1].bar(x + (i - 1.5) * 0.2, summary[column], width=0.2, label=label)
    axes[0].set_title("再配置成功粒子の実変位 / PDR 歩幅と方位差")
    axes[0].set_ylabel("距離比区分の割合")
    axes[1].set_ylabel("方位差閾値超過割合")
    for ax in axes:
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=4, fontsize=8)
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    _save(fig, destination, "summary_02_relocation")

    fig, axes = plt.subplots(
        2, 1, figsize=(max(12, len(summary) * 0.9), 9), sharex=True
    )
    for column, label in [
        ("ess_after_min", "最小 ESS"),
        ("ess_after_median", "中央値 ESS"),
        ("ancestors5_min", "最小5歩前祖先数"),
        ("ancestors10_min", "最小10歩前祖先数"),
    ]:
        axes[0].plot(x, summary[column], ".-", label=label)
    axes[0].set_ylabel("ESS / 祖先数")
    axes[0].legend(ncol=2)
    axes[0].set_title("ESS と祖先数・再標本化の比較（各指標は実行全体の集約）")
    axes[1].bar(
        x - 0.15, summary["full_resampling_steps"], width=0.3, label="全体再標本化歩数"
    )
    axes[1].bar(
        x + 0.15,
        summary["partial_replacement_total"],
        width=0.3,
        label="死亡枠補充総数",
    )
    axes[1].set_ylabel("歩数 / 粒子枠数")
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    _save(fig, destination, "summary_03_diversity")
    return sorted(
        str(path.relative_to(output)) for path in destination.glob("summary_*")
    )
