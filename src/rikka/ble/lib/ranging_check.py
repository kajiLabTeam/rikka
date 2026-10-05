"""実測BLEのRSSI距離整合を診断する。

役割:
    PDR軌跡上のビーコン距離とRSSIの相関、パスロス係数、残差を算出する。
依存元:
    ``common.lib.models`` のBLE観測・診断型とNumPyを使用する。
利用先:
    PDR/PF pipeline の補正前preflightと、agent診断ツールから使用される。
処理フロー:
    歩端点を観測時刻へ補間し、ビーコン別に対数距離回帰してPASS条件を判定する。
"""

import warnings

import numpy as np

from ...common.lib.models import BleObservation, FloorMap, RangingConsistency
from ...common.settings import BleLandmarkSettings
from ...landmark.lib.coordinates import build_landmark_meter_map
from .loader import load_ble_observations


def interpolate_trajectory_positions(
    trajectory: list[list[float]],
    step_times: list[float],
    observation_times: np.ndarray,
) -> np.ndarray:
    """歩端点をBLE観測時刻へ線形補間する。"""
    times = np.asarray([0.0, *step_times], dtype=float)
    points = np.asarray(trajectory, dtype=float)
    return np.column_stack(
        (
            np.interp(observation_times, times, points[:, 0]),
            np.interp(observation_times, times, points[:, 1]),
        )
    )


def evaluate_ranging_consistency(
    observations: tuple[BleObservation, ...],
    landmark_meters: dict[str, tuple[float, float]],
    trajectory: list[list[float]],
    t_at_steps: list[float],
) -> tuple[RangingConsistency, ...]:
    """ビーコン別のRSSI距離整合を返す。"""
    results: list[RangingConsistency] = []
    last_time = t_at_steps[-1] if t_at_steps else 0.0
    for beacon_id, landmark_xy in landmark_meters.items():
        rows = [
            item
            for item in observations
            if item.beacon_id == beacon_id and item.timestamp_s <= last_time
        ]
        if len(rows) < 2:
            results.append(RangingConsistency(beacon_id, None, None, None, False))
            continue
        timestamps = np.asarray([item.timestamp_s for item in rows], dtype=float)
        rssi = np.asarray([item.rssi_dbm for item in rows], dtype=float)
        positions = interpolate_trajectory_positions(trajectory, t_at_steps, timestamps)
        distances = np.maximum(
            np.linalg.norm(positions - np.asarray(landmark_xy), axis=1), 0.3
        )
        log_distances = np.log10(distances)
        correlation = float(np.corrcoef(rssi, log_distances)[0, 1])
        design = np.column_stack((np.ones(len(distances)), -10.0 * log_distances))
        coefficients, _, _, _ = np.linalg.lstsq(design, rssi, rcond=None)
        predicted = design @ coefficients
        path_loss_n = float(coefficients[1])
        sigma = float(np.sqrt(np.mean(np.square(rssi - predicted))))
        finite = all(np.isfinite((correlation, path_loss_n, sigma)))
        passed = finite and correlation <= -0.5 and 1.5 <= path_loss_n <= 4.0
        results.append(
            RangingConsistency(
                beacon_id,
                correlation if finite else None,
                path_loss_n if finite else None,
                sigma if finite else None,
                passed,
            )
        )
    return tuple(results)


def aggregate_ranging_correlation(
    observations: tuple[BleObservation, ...],
    landmark_meters: dict[str, tuple[float, float]],
    trajectory: list[list[float]],
    t_at_steps: list[float],
) -> float:
    """全登録ビーコンを連結したRSSIと対数距離の相関を返す。"""
    rssi_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    last_time = t_at_steps[-1] if t_at_steps else 0.0
    for beacon_id, landmark_xy in landmark_meters.items():
        rows = [
            item
            for item in observations
            if item.beacon_id == beacon_id and item.timestamp_s <= last_time
        ]
        if len(rows) < 2:
            continue
        timestamps = np.asarray([item.timestamp_s for item in rows], dtype=float)
        positions = interpolate_trajectory_positions(trajectory, t_at_steps, timestamps)
        distances = np.maximum(
            np.linalg.norm(positions - np.asarray(landmark_xy), axis=1), 0.3
        )
        rssi_parts.append(np.asarray([item.rssi_dbm for item in rows], dtype=float))
        distance_parts.append(np.log10(distances))
    if not rssi_parts:
        return float("nan")
    return float(
        np.corrcoef(np.concatenate(rssi_parts), np.concatenate(distance_parts))[0, 1]
    )


def run_ranging_preflight(
    settings: BleLandmarkSettings,
    floormap: FloorMap,
    trajectory: list[list[float]],
    t_at_steps: list[float],
    gx_mean: float,
    gz_mean: float,
) -> tuple[RangingConsistency, ...]:
    """設定に従って診断し、FAILを警告または例外として扱う。"""
    if not settings.enabled or settings.preflight_mode == "off":
        return ()
    observations = load_ble_observations(settings.data_path)
    landmark_meters = build_landmark_meter_map(
        settings.landmarks, gx_mean, gz_mean, floormap
    )
    results = evaluate_ranging_consistency(
        observations, landmark_meters, trajectory, t_at_steps
    )
    failures = [item for item in results if not item.passed]
    if failures:
        details = ", ".join(
            f"{item.beacon_id}(corr={item.correlation}, n={item.path_loss_n})"
            for item in failures
        )
        message = f"BLE preflight FAIL: {details}"
        if settings.preflight_mode == "error":
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)
    else:
        print("BLE preflight PASS")
    return results
