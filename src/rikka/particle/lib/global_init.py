"""地図全体からの一様な粒子初期化。

役割:
    歩行可能領域の位置と初期方位を一様にサンプリングし、面積を算出する。
依存元:
    NumPy と common.lib.floormap の compute_meter_coords を利用する。
利用先:
    particle.lib.initialize が初期位置・初期方位を与えないPFで使用する。
処理フロー:
    歩行可能画素を復元抽出し、画素中心の周囲に一様オフセットを加えて
    メートル座標へ変換する。方位は独立に [-π, π) から抽出する。
"""

import numpy as np

from ...common.lib.floormap import compute_meter_coords


def sample_uniform_walkable_particles(
    map_gray: np.ndarray,
    n_particles: int,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """歩行可能画素の面積に比例した一様位置を返す。"""
    walkable = np.flatnonzero(map_gray > 128)
    if not walkable.size:
        raise ValueError("地図に歩行可能画素がありません。")
    selected = rng.choice(walkable, size=n_particles, replace=True)
    ys, xs = np.unravel_index(selected, map_gray.shape)
    # 既存の地図判定では整数が画素中心で、境界は中心 ±0.5 にある。
    offsets = rng.uniform(-0.5, 0.5, size=(n_particles, 2))
    mx, my = compute_meter_coords(
        xs + offsets[:, 0],
        ys + offsets[:, 1],
        gx_mean,
        gz_mean,
        origin_px,
        scale,
    )
    return np.column_stack((mx, my))


def sample_uniform_headings(n_particles: int, rng: np.random.Generator) -> np.ndarray:
    """初期方位を [-π, π) の一様分布から抽出する。"""
    return rng.uniform(-np.pi, np.pi, size=n_particles)


def walkable_area_m2(map_gray: np.ndarray, scale: float) -> float:
    """歩行可能画素の合計面積を平方メートルで返す。"""
    return float(np.count_nonzero(map_gray > 128) * scale**2)
