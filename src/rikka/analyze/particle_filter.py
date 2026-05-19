"""パーティクルフィルタによる確率的歩行軌跡推定モジュール"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.ndimage import distance_transform_edt

from ..config import (
    FLOORMAP_ORIGIN_PX,
    FLOORMAP_PATH,
    FLOORMAP_SCALE,
    INITIAL_DIRECTION,
    PF_NUM_PARTICLES,
    PF_SIGMA_HEADING,
    PF_SIGMA_INIT_HEADING,
    PF_SIGMA_STEP_LENGTH_RATIO,
    STEP_LENGTH_METHOD,
    WEINBERG_K,
)
from .pdr import (
    _compute_pixel_coords,
    _estimate_initial_forward_angle,
    _sample_gyro_angle,
    _step_mid_index,
    _step_mid_time,
    estimate_step_length,
    estimate_step_length_forward,
)


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """系統リサンプリングでインデックス配列を返す。O(N)・分散最小。"""
    n = len(weights)
    positions = (np.arange(n) + rng.uniform(0, 1)) / n
    cumsum = np.cumsum(weights)
    return np.searchsorted(cumsum, positions)


def _normalize_floormap_gray(map_raw: np.ndarray) -> np.ndarray:
    """フロアマップ画像を 0..255 のグレースケール配列に正規化する。"""
    map_arr: np.ndarray = np.asarray(map_raw, dtype=float)
    if map_arr.ndim == 3:
        map_arr = np.mean(map_arr[:, :, :3], axis=2)
    if map_arr.size == 0:
        return map_arr
    if float(np.nanmax(map_arr)) <= 1.0:
        map_arr = map_arr * 255.0
    return np.asarray(np.clip(map_arr, 0.0, 255.0), dtype=float)


def _pixel_y_sign(gx_mean: float, gz_mean: float) -> int:
    """メートル座標とピクセル座標のY軸向きを返す。"""
    if abs(gx_mean) > abs(gz_mean):
        return -1 if gx_mean > 0 else 1
    return -1 if gz_mean < 0 else 1


def _compute_meter_coords(
    px: np.ndarray,
    py: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """フロアマップのピクセル座標をメートル座標へ戻す。"""
    y_sign = _pixel_y_sign(gx_mean, gz_mean)
    xs = (px - origin_px[0]) * scale
    ys = (py - origin_px[1]) * scale / y_sign
    return xs, ys


def _snap_trajectory_to_walkable_pixels(
    trajectory: list[list[float]],
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> list[list[float]]:
    """壁上・範囲外の軌跡点を最近傍の歩行可能画素へ寄せる。"""
    if len(trajectory) == 0:
        return trajectory

    walkable = map_gray > 128
    if not walkable.any():
        return trajectory

    map_h, map_w = walkable.shape
    points = np.asarray(trajectory, dtype=float)
    px_f, py_f = _compute_pixel_coords(
        points[:, 0], points[:, 1], gx_mean, gz_mean, origin_px, scale
    )
    pxi = np.round(px_f).astype(int)
    pyi = np.round(py_f).astype(int)

    in_bounds = (0 <= pxi) & (pxi < map_w) & (0 <= pyi) & (pyi < map_h)
    needs_snap = ~in_bounds.copy()
    if in_bounds.any():
        needs_snap[in_bounds] = ~walkable[pyi[in_bounds], pxi[in_bounds]]

    if not needs_snap.any():
        return trajectory

    _, nearest_indices = distance_transform_edt(~walkable, return_indices=True)
    nearest_y = nearest_indices[0]
    nearest_x = nearest_indices[1]

    query_x = pxi.clip(0, map_w - 1)
    query_y = pyi.clip(0, map_h - 1)
    snap_x = query_x.copy()
    snap_y = query_y.copy()
    snap_x[needs_snap] = nearest_x[query_y[needs_snap], query_x[needs_snap]]
    snap_y[needs_snap] = nearest_y[query_y[needs_snap], query_x[needs_snap]]

    xs, ys = _compute_meter_coords(
        snap_x.astype(float),
        snap_y.astype(float),
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    snapped_points = points.copy()
    snapped_points[needs_snap, 0] = xs[needs_snap]
    snapped_points[needs_snap, 1] = ys[needs_snap]
    return [[float(x), float(y)] for x, y in snapped_points]


def _reconstruct_resampled_paths(
    position_history: list[np.ndarray],
    resample_history: list[np.ndarray],
) -> np.ndarray:
    """リサンプリング祖先をたどって最終粒子群の経路を復元する。"""
    if not position_history:
        return np.empty((0, 0, 2), dtype=float)

    n_particles = position_history[0].shape[0]
    n_steps = len(position_history) - 1
    if len(resample_history) != n_steps:
        raise ValueError("position_history と resample_history の長さが一致しません")

    paths = np.empty((n_particles, n_steps + 1, 2), dtype=float)
    if n_steps == 0:
        paths[:, 0, :] = position_history[0]
        return paths

    lineage = resample_history[-1].astype(int, copy=True)
    paths[:, n_steps, :] = position_history[n_steps][lineage]

    for step in range(n_steps - 1, 0, -1):
        lineage = resample_history[step - 1][lineage]
        paths[:, step, :] = position_history[step][lineage]

    paths[:, 0, :] = position_history[0][lineage]
    return paths


def run_particle_filter(
    peaks: np.ndarray,
    df_gyro: pd.DataFrame,
    df_acc: pd.DataFrame,
    gx_mean: float,
    gz_mean: float,
    floormap_path: str | Path = FLOORMAP_PATH,
    origin_px: tuple[int, int] = FLOORMAP_ORIGIN_PX,
    scale: float = FLOORMAP_SCALE,
    initial_direction: float = INITIAL_DIRECTION,
    n_particles: int = PF_NUM_PARTICLES,
    sigma_init_heading: float = PF_SIGMA_INIT_HEADING,
    sigma_heading: float = PF_SIGMA_HEADING,
    sigma_sl_ratio: float = PF_SIGMA_STEP_LENGTH_RATIO,
    weinberg_k: float = WEINBERG_K,
) -> tuple[list[list[float]], list[float], np.ndarray]:
    """パーティクルフィルタでマップマッチング付き歩行軌跡を推定する。

    Args:
        peaks: ステップピークのインデックス配列
        df_gyro: ``low_angle`` 列を含むジャイロスコープDataFrame
        df_acc: 加速度DataFrame
        gx_mean: X軸重力成分の平均値（Y軸反転判定に使用）
        gz_mean: Z軸重力成分の平均値（Y軸反転判定に使用）
        floormap_path: フロアマップ画像のパス
        origin_px: 軌跡起点のピクセル座標
        scale: 1ピクセルあたりのメートル数
        initial_direction: 歩行開始方向のオフセット [度]
        n_particles: パーティクル数
        sigma_init_heading: 粒子ごとの初期方位ばらつき [rad]
        sigma_heading: ステップごとの方位角ノイズ [rad]
        sigma_sl_ratio: ステップ長ノイズの比率
        weinberg_k: Weinbergモデルのスケール係数

    Returns:
        tuple: (加重平均軌跡の座標リスト, 各ステップの決定論的歩幅リスト,
            全ステップのパーティクル位置 shape=(T, N, 2))
    """
    rng = np.random.default_rng()

    # フロアマップをグレースケールで読み込み
    map_gray = _normalize_floormap_gray(plt.imread(Path(floormap_path)))
    map_h, map_w = map_gray.shape

    # 全パーティクルを原点で初期化（[x, y] の2次元状態）
    particles = np.zeros((n_particles, 2))
    heading_bias = rng.normal(0, sigma_init_heading, n_particles)
    weights = np.ones(n_particles) / n_particles

    step_lengths: list[float] = []
    position_history: list[np.ndarray] = [particles.copy()]
    resample_history: list[np.ndarray] = []
    all_particles_list: list[np.ndarray] = [particles.copy()]  # ステップ0（原点）

    direction_offset = float(np.deg2rad(initial_direction))

    phi_0 = (
        _estimate_initial_forward_angle(df_acc, df_gyro, peaks)
        if STEP_LENGTH_METHOD == "forward"
        else 0.0
    )

    for i, p in enumerate(peaks):
        if p >= len(df_acc):
            continue
        if STEP_LENGTH_METHOD == "forward" and i + 1 >= len(peaks):
            continue

        # estimate_trajectory と同一のサンプリングインデックス計算
        mid_idx = _step_mid_index(peaks, i)
        angle_at_mid = _sample_gyro_angle(
            df_gyro,
            sample_index=mid_idx,
            sample_time=_step_mid_time(df_acc, peaks, i),
        )
        if angle_at_mid is None:
            continue
        angle_det = angle_at_mid + direction_offset

        if STEP_LENGTH_METHOD == "forward":
            sl_det = estimate_step_length_forward(df_acc, df_gyro, peaks, i, phi_0)
        else:
            sl_det = estimate_step_length(df_acc, int(p), k=weinberg_k)

        # 予測前の状態を保存（全壁レスキュー用）
        particles_before = particles.copy()

        # 予測ステップ：粒子ごとの方位バイアスをランダムウォークとして累積
        heading_bias += rng.normal(0, sigma_heading, n_particles)
        theta = angle_det + heading_bias
        sl = np.clip(sl_det * (1 + rng.normal(0, sigma_sl_ratio, n_particles)), 0, None)
        particles[:, 0] += sl * np.cos(theta)
        particles[:, 1] += sl * np.sin(theta)

        # 観測モデル：二値閾値でパーティクルの通路/壁を判定
        def _eval_corridor(
            pts: np.ndarray,
            prev_pts: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
            """パーティクル位置の輝度と通路マスクを返す。

            prev_pts を渡すと前後の線分を5点サンプリングし、
            途中で壁を横断したパーティクルも除外する。
            """

            def _brightness(arr: np.ndarray) -> np.ndarray:
                px_f, py_f = _compute_pixel_coords(
                    arr[:, 0], arr[:, 1], gx_mean, gz_mean, origin_px, scale
                )
                pxi = np.round(px_f).astype(int)
                pyi = np.round(py_f).astype(int)
                ib = (0 <= pxi) & (pxi < map_w) & (0 <= pyi) & (pyi < map_h)
                py_s = pyi.clip(0, map_h - 1)
                px_s = pxi.clip(0, map_w - 1)
                return np.where(ib, map_gray[py_s, px_s], 0.0)

            brt = _brightness(pts)
            in_corr = brt > 128

            if prev_pts is not None:
                # 線分上の中間点（t=0.25, 0.5, 0.75）も通路であることを確認
                for t_val in (0.25, 0.5, 0.75):
                    mid = prev_pts + t_val * (pts - prev_pts)
                    in_corr &= _brightness(mid) > 128

            return brt, in_corr

        _, in_corridor = _eval_corridor(particles, prev_pts=particles_before)
        weights *= in_corridor.astype(float)

        total = weights.sum()
        if total < 1e-300:
            # レスキュー：予測前の位置から3倍のノイズで再試行
            particles = particles_before.copy()
            theta_r = (
                angle_det + heading_bias + rng.normal(0, sigma_heading * 3, n_particles)
            )
            sl_r = np.clip(
                sl_det * (1 + rng.normal(0, sigma_sl_ratio * 3, n_particles)), 0, None
            )
            particles[:, 0] += sl_r * np.cos(theta_r)
            particles[:, 1] += sl_r * np.sin(theta_r)
            _, in_corridor = _eval_corridor(particles, prev_pts=particles_before)
            weights = in_corridor.astype(float)
            total = weights.sum()
            if total < 1e-300:
                # レスキューも失敗：予測前の位置に戻して均一化（壁上粒子を残さない）
                particles = particles_before.copy()
                weights[:] = 1.0 / n_particles
            else:
                weights /= total
        else:
            weights /= total

        position_history.append(particles.copy())
        step_lengths.append(sl_det)

        # 系統リサンプリング
        indices = _systematic_resample(weights, rng)
        particles = particles[indices]
        heading_bias = heading_bias[indices]
        weights[:] = 1.0 / n_particles
        resample_history.append(indices)
        all_particles_list.append(particles.copy())

    all_particles = np.stack(all_particles_list)  # shape: (T+1, N, 2)
    particle_paths = _reconstruct_resampled_paths(position_history, resample_history)
    mean_trajectory = particle_paths.mean(axis=0).tolist()
    mean_trajectory = _snap_trajectory_to_walkable_pixels(
        mean_trajectory,
        map_gray,
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    return mean_trajectory, step_lengths, all_particles


def plot_particle_filter_trajectory(
    trajectory: list[list[float]],
    gx_mean: float = 0.0,
    gz_mean: float = 0.0,
    floormap_path: str | Path = FLOORMAP_PATH,
    origin_px: tuple[int, int] = FLOORMAP_ORIGIN_PX,
    scale: float = FLOORMAP_SCALE,
    output_dir: Path | None = None,
) -> None:
    """PF の加重平均軌跡をフロアマップ上にプロットする。

    Args:
        trajectory: 各ステップの [x, y] 座標リスト（メートル）
        gx_mean: X軸重力成分の平均値
        gz_mean: Z軸重力成分の平均値
        floormap_path: フロアマップ画像のパス
        origin_px: 軌跡起点のピクセル座標
        scale: 1ピクセルあたりのメートル数
        output_dir: 出力ディレクトリ（指定時に PNG 保存）
    """
    df = pd.DataFrame(trajectory, columns=["x", "y"])
    px, py = _compute_pixel_coords(
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
    lc.set_array(np.arange(n - 1))
    ax.add_collection(lc)
    sc = ax.scatter(px, py, c=np.arange(n), cmap=cmap, norm=norm, s=20, zorder=3)
    fig.colorbar(sc, ax=ax, label="Step")
    ax.plot(px[0], py[0], "go", markersize=10, label="Start", zorder=4)

    ax.set_title("Particle Filter Trajectory on Floormap")
    ax.legend()
    plt.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir / "pf_trajectory.png", dpi=150, bbox_inches="tight")
    plt.show()


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
) -> None:
    """PF の各ステップのパーティクル分布をフロアマップ上に描画し MP4 として保存する。

    Args:
        all_particles: 全ステップのパーティクル位置 shape=(T, N, 2)
        mean_trajectory: 各ステップの加重平均位置リスト
        gx_mean: X軸重力成分の平均値
        gz_mean: Z軸重力成分の平均値
        floormap_path: フロアマップ画像のパス
        origin_px: 軌跡起点のピクセル座標
        scale: 1ピクセルあたりのメートル数
        output_path: 出力ファイルパス（.mp4）
        fps: フレームレート
    """
    from matplotlib.animation import FFMpegWriter, PillowWriter  # noqa: PLC0415

    map_img = plt.imread(Path(floormap_path))
    mean_arr = np.array(mean_trajectory)  # shape: (T, 2)

    fig, ax = plt.subplots(figsize=(7, 7))

    def update(frame: int) -> list[Artist]:
        ax.cla()
        ax.imshow(map_img)

        # 全パーティクルを半透明グレーで描画
        px_p, py_p = _compute_pixel_coords(
            all_particles[frame, :, 0],
            all_particles[frame, :, 1],
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        ax.scatter(px_p, py_p, s=30, c="cyan", alpha=0.5, zorder=2)

        # ステップ 0 〜 現在の平均軌跡を青線で描画
        if frame > 0:
            px_m, py_m = _compute_pixel_coords(
                mean_arr[: frame + 1, 0],
                mean_arr[: frame + 1, 1],
                gx_mean,
                gz_mean,
                origin_px,
                scale,
            )
            ax.plot(px_m, py_m, "b-", linewidth=1.5, zorder=3)

        # 現ステップの平均位置を赤点で描画
        px_c, py_c = _compute_pixel_coords(
            mean_arr[frame : frame + 1, 0],
            mean_arr[frame : frame + 1, 1],
            gx_mean,
            gz_mean,
            origin_px,
            scale,
        )
        ax.scatter(px_c, py_c, s=60, c="red", zorder=4)
        ax.set_title(f"Step {frame}")
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
