"""パーティクルフィルタの段階別状態と代表軌跡候補を画像化する。

役割:
    1歩の開始から確定までの粒子位置・方位・重み・壁判定と、代表軌跡の
    選択候補を診断画像として保存する。
依存元:
    ``models`` の可視化用スナップショット、``pdr.particle_api`` の座標変換、
    Matplotlib、NumPy、フロアマップ画像を利用する。
利用先:
    ``pdr.pipeline`` が明示的な保存オプションを受けた場合だけ呼び出す。
処理フロー:
    収集済み状態を歩単位で共通表示範囲へ変換して6パネル画像を保存し、
    必要に応じて平均・単一祖先・個別粒子の経路比較図も保存する。
"""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba_array
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ...analyze.particle.models import (
    ParticleFilterStepDiagnostics,
    ParticlePathComparison,
    ParticleStepStages,
)
from ...matplotlib_config import configure_japanese_font
from .floormap import compute_pixel_coords, pixel_vector_from_heading

_MOTION_COLORS = np.asarray(["#2563eb", "#f59e0b", "#16a34a", "#9333ea"])
_DISPLAY_NAMES = {
    "none": "なし",
    "forward": "前進",
    "sidestep_left": "左横歩き",
    "sidestep_right": "右横歩き",
    "turning": "旋回",
    "turning_sidestep_left": "旋回付き左横歩き",
    "turning_sidestep_right": "旋回付き右横歩き",
    "sidestep_suspect_left": "左横歩き疑い",
    "sidestep_suspect_right": "右横歩き疑い",
    "local_grid": "局所格子復旧",
    "turn_grid": "旋回格子復旧",
    "checkpoint_replay": "チェックポイント再生",
    "checkpoint_replayed": "チェックポイント再生済み",
    "failed_hold": "復旧失敗・位置保持",
    "weighted_mean": "加重平均",
    "particle_fallback": "粒子祖先フォールバック",
    "current": "加重平均候補",
    "sequence": "単一祖先候補",
}


def _display_name(value: str) -> str:
    """内部識別子を画像表示用の日本語へ変換する。"""
    if value.startswith("fallback_"):
        return f"フォールバック：{_display_name(value.removeprefix('fallback_'))}"
    return _DISPLAY_NAMES.get(value, value)


def _motion_colors(states: np.ndarray) -> np.ndarray:
    """運動状態の整数配列を固定色へ変換する。"""
    return np.asarray(
        _MOTION_COLORS[np.clip(states.astype(int), 0, len(_MOTION_COLORS) - 1)],
        dtype=str,
    )


def _normalized_alpha(weights: np.ndarray) -> np.ndarray:
    """小さい重みも位置を確認できる描画alphaへ正規化する。"""
    maximum = float(np.max(weights)) if len(weights) else 0.0
    if maximum <= 0.0:
        return np.full(len(weights), 0.25)
    return 0.15 + 0.75 * np.clip(weights / maximum, 0.0, 1.0)


def _pixel_positions(
    positions: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> np.ndarray:
    """メートル座標配列をピクセル座標配列へ変換する。"""
    px, py = compute_pixel_coords(
        positions[:, 0],
        positions[:, 1],
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    return np.column_stack((px, py))


def _common_limits(
    stage: ParticleStepStages,
    map_shape: tuple[int, ...],
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """6パネルで共有する自動フィット表示範囲を返す。"""
    positions = np.vstack(
        (stage.before_positions, stage.proposed_positions, stage.after_positions)
    )
    pixels = _pixel_positions(
        positions,
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    center = np.mean(pixels, axis=0)
    span = np.ptp(pixels, axis=0)
    minimum_span = 6.0 / scale
    span = np.maximum(span * 1.2, minimum_span)
    map_height, map_width = map_shape[:2]
    x_limits = (
        max(0.0, center[0] - span[0] / 2.0),
        min(float(map_width), center[0] + span[0] / 2.0),
    )
    y_limits = (
        max(0.0, center[1] - span[1] / 2.0),
        min(float(map_height), center[1] + span[1] / 2.0),
    )
    return x_limits, y_limits


def _prepare_map_axis(
    axis: Axes,
    map_image: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    title: str,
) -> None:
    """地図背景、共通範囲、全体位置インセットを設定する。"""
    axis.imshow(map_image, cmap="gray", origin="upper")
    axis.set_xlim(*x_limits)
    axis.set_ylim(y_limits[1], y_limits[0])
    axis.set_title(title, fontsize=10)
    axis.set_aspect("equal")
    axis.tick_params(labelsize=7)
    overview = inset_axes(axis, width="24%", height="24%", loc="upper right")
    overview.imshow(map_image, cmap="gray", origin="upper")
    overview.set_xticks([])
    overview.set_yticks([])
    overview.plot(
        [x_limits[0], x_limits[1], x_limits[1], x_limits[0], x_limits[0]],
        [y_limits[0], y_limits[0], y_limits[1], y_limits[1], y_limits[0]],
        color="crimson",
        linewidth=0.8,
    )


def _scatter_particles(
    axis: Axes,
    positions_px: np.ndarray,
    weights: np.ndarray,
    colors: np.ndarray,
    marker: str = "o",
) -> None:
    """全粒子を重みalpha付きで散布する。"""
    rgba_colors = to_rgba_array(colors)
    rgba_colors[:, 3] = _normalized_alpha(weights)
    axis.scatter(
        positions_px[:, 0],
        positions_px[:, 1],
        s=8,
        c=rgba_colors,
        marker=marker,
        linewidths=0.25,
    )


def _draw_top_headings(
    axis: Axes,
    positions_px: np.ndarray,
    headings: np.ndarray,
    lengths: np.ndarray,
    weights: np.ndarray,
    colors: np.ndarray,
    arrows: int,
    gx_mean: float,
    gz_mean: float,
    scale: float,
) -> None:
    """重み上位粒子の方位と歩幅を矢印で描く。"""
    count = min(max(arrows, 0), len(weights))
    if count == 0:
        return
    selected = np.argsort(weights)[-count:]
    for index in selected:
        dx, dy = pixel_vector_from_heading(
            float(headings[index]),
            float(lengths[index]),
            gx_mean,
            gz_mean,
            scale,
        )
        axis.arrow(
            positions_px[index, 0],
            positions_px[index, 1],
            dx,
            dy,
            color=str(colors[index]),
            alpha=0.75,
            width=0.35,
            head_width=3.5,
            length_includes_head=True,
        )


def _plot_recent_trajectory(
    axis: Axes,
    trajectory: np.ndarray,
    step: int,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> None:
    """代表軌跡の直近10歩を地図パネルへ重ねる。"""
    start = max(0, step - 10)
    recent = trajectory[start : step + 1]
    if len(recent) < 2:
        return
    pixels = _pixel_positions(recent, gx_mean, gz_mean, origin_px, scale)
    axis.plot(pixels[:, 0], pixels[:, 1], color="black", linewidth=1.0, alpha=0.7)


def _save_step_frame(
    stage: ParticleStepStages,
    trajectory: np.ndarray,
    map_image: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
    arrows: int,
    dpi: int,
    output_path: Path,
) -> None:
    """1歩分の6段階パネルを上3枚・下3枚で保存する。"""
    figure = plt.figure(figsize=(15, 10))
    grid = figure.add_gridspec(2, 3)
    map_axes = [
        figure.add_subplot(grid[row, column]) for row in range(2) for column in range(3)
    ]
    x_limits, y_limits = _common_limits(
        stage,
        map_image.shape,
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    titles = (
        "① 開始",
        "② 提案（絶対方位）",
        "③ 壁判定",
        "④ 観測重み",
        "⑤ 選択・復旧",
        "⑥ 確定",
    )
    for panel_index, (axis, title) in enumerate(zip(map_axes, titles, strict=True)):
        _prepare_map_axis(axis, map_image, x_limits, y_limits, title)
        if panel_index > 0:
            _plot_recent_trajectory(
                axis,
                trajectory,
                stage.step,
                gx_mean,
                gz_mean,
                origin_px,
                scale,
            )

    before_px = _pixel_positions(
        stage.before_positions, gx_mean, gz_mean, origin_px, scale
    )
    proposed_px = _pixel_positions(
        stage.proposed_positions, gx_mean, gz_mean, origin_px, scale
    )
    after_px = _pixel_positions(
        stage.after_positions, gx_mean, gz_mean, origin_px, scale
    )
    before_colors = _motion_colors(stage.before_motion_state)
    proposed_colors = _motion_colors(stage.proposed_motion_state)
    after_colors = _motion_colors(stage.after_motion_state)
    _scatter_particles(map_axes[0], before_px, stage.before_weights, before_colors)
    _scatter_particles(map_axes[1], proposed_px, stage.before_weights, proposed_colors)
    _draw_top_headings(
        map_axes[1],
        before_px,
        stage.proposed_headings,
        stage.proposed_step_lengths,
        stage.before_weights,
        proposed_colors,
        arrows,
        gx_mean,
        gz_mean,
        scale,
    )
    if stage.sensor_heading is not None:
        sensor_dx, sensor_dy = pixel_vector_from_heading(
            stage.sensor_heading,
            stage.deterministic_step_length_m,
            gx_mean,
            gz_mean,
            scale,
        )
        center = np.mean(before_px, axis=0)
        map_axes[1].plot(
            [center[0], center[0] + sensor_dx],
            [center[1], center[1] + sensor_dy],
            color="black",
            linestyle="--",
            linewidth=2.0,
        )

    valid_colors = np.where(stage.valid_transition, "#2563eb", "#dc2626")
    _scatter_particles(map_axes[2], proposed_px, stage.before_weights, valid_colors)
    rejected = np.flatnonzero(~stage.valid_transition)
    for index in rejected:
        map_axes[2].plot(
            [before_px[index, 0], proposed_px[index, 0]],
            [before_px[index, 1], proposed_px[index, 1]],
            color="#dc2626",
            linewidth=0.45,
            alpha=0.45,
        )
    if len(rejected):
        map_axes[2].scatter(
            proposed_px[rejected, 0],
            proposed_px[rejected, 1],
            marker="x",
            color="#dc2626",
            s=15,
        )

    positive = stage.posterior_weights > 0.0
    weight_colors = np.tile(
        np.asarray([0.6118, 0.6392, 0.6863, 1.0]),
        (len(stage.posterior_weights), 1),
    )
    if np.any(positive):
        normalized = stage.posterior_weights[positive]
        normalized = normalized / max(float(np.max(normalized)), 1e-12)
        weight_colors[positive] = colormaps["viridis"](normalized)
    _scatter_particles(
        map_axes[3],
        proposed_px,
        np.maximum(stage.posterior_weights, 1e-12),
        weight_colors,
    )
    map_axes[3].text(
        0.02,
        0.02,
        f"ESS {stage.ess_before_observation:.1f} → {stage.ess_after_observation:.1f}",
        transform=map_axes[3].transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.7},
    )

    _scatter_particles(map_axes[4], after_px, stage.after_weights, after_colors)
    parent_before = before_px[stage.parent_indices]
    line_count = min(arrows, len(after_px))
    selected_lines = np.argsort(stage.after_weights)[-line_count:]
    for index in selected_lines:
        map_axes[4].plot(
            [parent_before[index, 0], after_px[index, 0]],
            [parent_before[index, 1], after_px[index, 1]],
            color="#111827",
            linewidth=0.4,
            alpha=0.35,
        )
        if stage.recovery_mode != "none":
            displacement = after_px[index] - parent_before[index]
            map_axes[4].arrow(
                parent_before[index, 0],
                parent_before[index, 1],
                displacement[0],
                displacement[1],
                color="#7e22ce",
                alpha=0.65,
                width=0.35,
                head_width=3.5,
                length_includes_head=True,
            )
    _scatter_particles(map_axes[5], after_px, stage.after_weights, after_colors)
    for index in selected_lines:
        map_axes[5].plot(
            [before_px[index, 0], after_px[index, 0]],
            [before_px[index, 1], after_px[index, 1]],
            color=str(after_colors[index]),
            linewidth=0.6,
            alpha=0.55,
        )

    figure.suptitle(
        f"パーティクルフィルタ 第{stage.step:03d}歩",
        fontsize=14,
    )
    figure.subplots_adjust(
        left=0.03,
        right=0.98,
        bottom=0.05,
        top=0.93,
        wspace=0.18,
        hspace=0.18,
    )
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def save_particle_step_frames(
    stages: list[ParticleStepStages],
    diagnostics: list[ParticleFilterStepDiagnostics],
    trajectory: list[list[float]] | np.ndarray,
    *,
    gx_mean: float,
    gz_mean: float,
    floormap_path: str | Path,
    origin_px: tuple[int, int],
    scale: float,
    output_dir: str | Path,
    step_range: tuple[int, int] | None = None,
    arrows: int = 40,
    dpi: int = 100,
) -> list[Path]:
    """指定範囲の段階別画像を保存し、生成パスを返す。"""
    configure_japanese_font()
    if arrows < 0:
        raise ValueError("arrows は0以上を指定してください")
    if dpi <= 0:
        raise ValueError("dpi は正の整数を指定してください")
    if step_range is None:
        first, last = 1, len(stages)
    else:
        first, last = step_range
    if first < 1 or first > last:
        raise ValueError("step_range は 1 <= A <= B を満たす必要があります")
    map_image = mpimg.imread(Path(floormap_path))
    frames_dir = Path(output_dir) / "particle_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    trajectory_array = np.asarray(trajectory, dtype=float)
    output_paths: list[Path] = []
    for stage in stages:
        if stage.step < first or stage.step > last:
            continue
        output_path = frames_dir / f"step_{stage.step:03d}.png"
        _save_step_frame(
            stage,
            trajectory_array,
            map_image,
            gx_mean,
            gz_mean,
            origin_px,
            scale,
            arrows,
            dpi,
            output_path,
        )
        output_paths.append(output_path)
    return output_paths


def _plot_path_on_map(
    axis: Axes,
    path: np.ndarray,
    *,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
    label: str,
    selected: bool,
    color: str,
) -> None:
    """候補経路を選択状態に応じた線種で描く。"""
    pixels = _pixel_positions(path, gx_mean, gz_mean, origin_px, scale)
    axis.plot(
        pixels[:, 0],
        pixels[:, 1],
        color=color,
        linewidth=3.0 if selected else 1.4,
        linestyle="-" if selected else "--",
        label=label,
        zorder=4 if selected else 3,
    )


def save_particle_path_comparison(
    comparison: ParticlePathComparison,
    *,
    gx_mean: float,
    gz_mean: float,
    floormap_path: str | Path,
    origin_px: tuple[int, int],
    scale: float,
    output_path: str | Path,
    truth_path: np.ndarray | None = None,
    dpi: int = 100,
) -> Path:
    """代表軌跡候補と個別粒子経路の比較図を保存する。"""
    configure_japanese_font()
    map_image = mpimg.imread(Path(floormap_path))
    figure, (map_axis, heading_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 12),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    map_axis.imshow(map_image, cmap="gray", origin="upper")
    path_count = comparison.particle_paths.shape[0]
    sample_count = min(25, path_count)
    sample_indices = np.linspace(0, path_count - 1, sample_count, dtype=int)
    for index in sample_indices:
        pixels = _pixel_positions(
            comparison.particle_paths[index],
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        map_axis.plot(
            pixels[:, 0],
            pixels[:, 1],
            color="#6b7280",
            linewidth=0.5,
            alpha=0.2,
        )
    _plot_path_on_map(
        map_axis,
        comparison.current_path,
        gx_mean=gx_mean,
        gz_mean=gz_mean,
        origin_px=origin_px,
        scale=scale,
        label=f"加重平均（反転数={comparison.current_reversals}）",
        selected=comparison.selected_mode == "current",
        color="#dc2626",
    )
    _plot_path_on_map(
        map_axis,
        comparison.sequence_path,
        gx_mean=gx_mean,
        gz_mean=gz_mean,
        origin_px=origin_px,
        scale=scale,
        label=f"単一祖先（反転数={comparison.sequence_reversals}）",
        selected=comparison.selected_mode == "sequence",
        color="#2563eb",
    )
    if truth_path is not None:
        _plot_path_on_map(
            map_axis,
            truth_path,
            gx_mean=gx_mean,
            gz_mean=gz_mean,
            origin_px=origin_px,
            scale=scale,
            label="正解軌跡",
            selected=False,
            color="#16a34a",
        )
    map_axis.legend()
    map_axis.set_title(
        f"代表軌跡候補の比較（採用：{_display_name(comparison.selected_mode)}）"
    )
    map_axis.set_aspect("equal")

    for path, label, color in (
        (comparison.current_path, "加重平均", "#dc2626"),
        (comparison.sequence_path, "単一祖先", "#2563eb"),
    ):
        deltas = np.diff(path, axis=0)
        headings = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))
        changes = np.degrees(np.abs(np.diff(headings, prepend=headings[:1])))
        steps = np.arange(1, len(changes) + 1)
        heading_axis.plot(steps, changes, label=label, color=color)
        excessive = changes > 25.0
        heading_axis.scatter(
            steps[excessive],
            changes[excessive],
            color=color,
            marker="x",
        )
    heading_axis.axhline(25.0, color="black", linestyle="--", linewidth=1.0)
    heading_axis.set_xlabel("歩番号")
    heading_axis.set_ylabel("|方位変化| [度]")
    heading_axis.legend()
    figure.tight_layout()
    result_path = Path(output_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(result_path, dpi=dpi)
    plt.close(figure)
    return result_path


def generated_files_size(paths: list[Path]) -> int:
    """存在する生成ファイルの合計バイト数を返す。"""
    return sum(path.stat().st_size for path in paths if path.exists())
