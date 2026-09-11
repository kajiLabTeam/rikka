"""ランドマーク補正用の相似変換を提供する。

役割:
    アンカー固定の回転・等方スケールを解き、点列へ適用して地図整合を評価する。
依存元:
    ``common.lib.floormap`` の座標変換と線分判定、NumPyを利用する。
利用先:
    通常PDRの過去軌跡補正とPFの代表軌跡補正から使用される。
処理フロー:
    始終点ベクトルから変換を解き、必要なら指数補間で減衰し、指定区間へ適用する。
"""

from typing import NamedTuple

import numpy as np

from ...common.lib.floormap import (
    compute_pixel_coords,
    segment_crosses_only_walkable_cells,
)


class SimilarityTransform(NamedTuple):
    """固定点を中心とする2次元相似変換。"""

    pivot: tuple[float, float]
    rotation_rad: float
    scale: float


def solve_anchor_similarity(
    pivot: tuple[float, float],
    raw_endpoint: tuple[float, float],
    target: tuple[float, float],
    *,
    eps: float = 1e-12,
) -> SimilarityTransform | None:
    """固定点からの2ベクトルを一致させる相似変換を返す。"""
    pivot_array = np.asarray(pivot, dtype=float)
    raw_vector = np.asarray(raw_endpoint, dtype=float) - pivot_array
    target_vector = np.asarray(target, dtype=float) - pivot_array
    raw_norm = float(np.linalg.norm(raw_vector))
    target_norm = float(np.linalg.norm(target_vector))
    if not np.isfinite([*pivot, *raw_endpoint, *target]).all():
        return None
    if raw_norm < eps or target_norm < eps:
        return None
    cross = raw_vector[0] * target_vector[1] - raw_vector[1] * target_vector[0]
    dot = float(np.dot(raw_vector, target_vector))
    return SimilarityTransform(
        pivot=(float(pivot[0]), float(pivot[1])),
        rotation_rad=float(np.arctan2(cross, dot)),
        scale=target_norm / raw_norm,
    )


def damp_transform(
    transform: SimilarityTransform,
    factor: float,
) -> SimilarityTransform:
    """回転を線形、倍率を対数空間で減衰した変換を返す。"""
    if not np.isfinite(factor) or not 0.0 <= factor <= 1.0:
        raise ValueError("factor は 0 以上 1 以下の有限値にしてください。")
    if not np.isfinite(transform.scale) or transform.scale <= 0.0:
        raise ValueError("transform.scale は有限な正の値にしてください。")
    return SimilarityTransform(
        pivot=transform.pivot,
        rotation_rad=transform.rotation_rad * factor,
        scale=transform.scale**factor,
    )


def apply_transform(
    points: list[list[float]],
    transform: SimilarityTransform,
    start_index: int,
    end_index: int | None,
) -> list[list[float]]:
    """点列を複製し、両端を含む指定範囲だけへ変換を適用する。"""
    if start_index < 0:
        raise ValueError("start_index は 0 以上にしてください。")
    resolved_end = len(points) - 1 if end_index is None else end_index
    if resolved_end >= len(points) or start_index > resolved_end + 1:
        raise ValueError("変換範囲が points の範囲外です。")
    transformed = [list(point) for point in points]
    if start_index > resolved_end:
        return transformed
    pivot = np.asarray(transform.pivot, dtype=float)
    cosine = float(np.cos(transform.rotation_rad))
    sine = float(np.sin(transform.rotation_rad))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=float)
    for index in range(start_index, resolved_end + 1):
        vector = np.asarray(points[index], dtype=float) - pivot
        result = pivot + transform.scale * (rotation @ vector)
        transformed[index] = [float(result[0]), float(result[1])]
    return transformed


def count_walkability_violations(
    points: list[list[float]],
    *,
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> int:
    """メートル座標点列のうち歩行不可画素を横切る辺数を返す。"""
    if len(points) < 2:
        return 0
    values = np.asarray(points, dtype=float)
    pixel_xs, pixel_ys = compute_pixel_coords(
        values[:, 0], values[:, 1], gx_mean, gz_mean, origin_px, scale
    )
    return sum(
        not segment_crosses_only_walkable_cells(x0, y0, x1, y1, map_gray)
        for x0, y0, x1, y1 in zip(
            pixel_xs[:-1],
            pixel_ys[:-1],
            pixel_xs[1:],
            pixel_ys[1:],
            strict=True,
        )
    )


def evaluate_transform_walkability(
    points: list[list[float]],
    transform: SimilarityTransform,
    start_index: int,
    end_index: int | None,
    *,
    map_gray: np.ndarray,
    gx_mean: float,
    gz_mean: float,
    origin_px: tuple[int, int],
    scale: float,
) -> int:
    """変換後点列の歩行不可辺数を返す。"""
    transformed = apply_transform(points, transform, start_index, end_index)
    return count_walkability_violations(
        transformed,
        map_gray=map_gray,
        gx_mean=gx_mean,
        gz_mean=gz_mean,
        origin_px=origin_px,
        scale=scale,
    )
