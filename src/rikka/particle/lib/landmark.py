"""ランドマーク観測による粒子重み更新と再配置。

役割:
    検出したランドマークの既知座標から粒子ごとの観測尤度を作り、reset方式では
    ランドマーク周辺の歩行可能位置へ粒子を再配置する。
依存元:
    NumPy、``common.lib.floormap`` の座標変換・歩行可能判定を利用する。
利用先:
    ``particle.lib.propose`` が観測尤度を、``particle.lib.evaluate_map`` が
    reset方式の粒子再配置を使用する。
処理フロー:
    粒子とランドマークの距離から下限付きガウス尤度を作る。再配置時は正規分布の
    候補を生成し、地図上の歩行可能候補だけを採用する。
"""

from collections.abc import Callable

import numpy as np

from ...common.lib.floormap import compute_pixel_coords


def landmark_likelihood(
    particles: np.ndarray,
    landmark_xy: tuple[float, float],
    sigma_m: float,
    floor: float,
) -> np.ndarray:
    """ランドマーク距離に基づく粒子ごとの観測尤度を返す。"""
    delta = particles - np.asarray(landmark_xy, dtype=float)
    squared = np.sum(delta * delta, axis=1)
    gaussian = np.exp(-squared / (2.0 * sigma_m * sigma_m))
    return np.asarray(floor + (1.0 - floor) * gaussian, dtype=float)


def resolve_anchor_heading(
    base_headings: np.ndarray,
    current_headings: np.ndarray,
    heading_deg: float,
    heading_sigma_deg: float,
    bidirectional: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """確定絶対方位を保つ補正項と、一時的な方位ばらつきを返す。"""

    def normalize(values: np.ndarray) -> np.ndarray:
        return np.asarray(np.arctan2(np.sin(values), np.cos(values)), dtype=float)

    target = np.full(len(base_headings), np.radians(heading_deg), dtype=float)
    if bidirectional:
        opposite = normalize(target + np.pi)
        use_opposite = np.abs(normalize(target - current_headings)) > np.pi / 2.0
        target = np.where(use_opposite, opposite, target)
    correction = normalize(target - base_headings)
    drift = normalize(
        rng.normal(0.0, np.radians(heading_sigma_deg), len(base_headings))
    )
    return correction, drift


def meter_walkable_mask(
    points: np.ndarray,
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> np.ndarray:
    """メートル座標点ごとの地図内・歩行可能判定を返す。"""
    pixel_xs, pixel_ys = compute_pixel_coords(
        points[:, 0],
        points[:, 1],
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    xs = np.floor(pixel_xs + 0.5).astype(int)
    ys = np.floor(pixel_ys + 0.5).astype(int)
    map_h, map_w = map_gray.shape
    in_bounds = (xs >= 0) & (xs < map_w) & (ys >= 0) & (ys < map_h)
    walkable = np.zeros(len(points), dtype=bool)
    walkable[in_bounds] = map_gray[ys[in_bounds], xs[in_bounds]] > 128
    return walkable


def reset_particles_to_landmark(
    particle_count: int,
    landmark_xy: tuple[float, float],
    sigma_m: float,
    rng: np.random.Generator,
    is_walkable: Callable[[np.ndarray], np.ndarray],
    max_attempts: int = 20,
) -> np.ndarray:
    """ランドマーク周辺の歩行可能位置へ粒子を再配置する。"""
    center = np.asarray(landmark_xy, dtype=float)
    particles = np.repeat(center[None, :], particle_count, axis=0)
    pending = np.ones(particle_count, dtype=bool)
    for _ in range(max_attempts):
        count = int(np.count_nonzero(pending))
        if count == 0:
            break
        candidates = center + rng.normal(0.0, sigma_m, size=(count, 2))
        accepted = is_walkable(candidates)
        pending_indices = np.flatnonzero(pending)
        particles[pending_indices[accepted]] = candidates[accepted]
        pending[pending_indices[accepted]] = False
    return particles
