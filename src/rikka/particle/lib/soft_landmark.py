"""任意の候補位置に対するソフトランドマークの評価。

役割:
    通常候補と復旧候補に同じBLE観測因子を適用し、適用回数と残差を診断する。
依存元:
    commonのパスロス型、landmarkの既存尤度関数、ParticleRuntimeを利用する。
利用先:
    proposeが全候補の因子を構築し、evaluate_mapが復旧後の集合を再重み付けする。
処理フロー:
    現在歩の観測を解決し、位置ごとの尤度を返す。明示的な診断呼び出し時だけ
    実際に残った粒子集合で残差・floor率・適用回数を更新する。
"""

import numpy as np

from ...common.config import (
    BLE_PATH_LOSS_N,
    BLE_PATH_LOSS_TX_POWER_DBM,
    BLE_RSSI_SIGMA_DB,
)
from ...common.lib.models import PathLossModel
from .landmark import expected_rssi_dbm, landmark_likelihood, landmark_range_likelihood
from .state import ParticleRuntime


def soft_landmark_likelihood(
    ctx: ParticleRuntime, positions: np.ndarray
) -> np.ndarray | None:
    """現在歩のソフト因子を副作用なしで評価し、観測なしならNoneを返す。"""
    if ctx.landmark_mode == "ranging":
        observations = ctx.landmark_observations_by_step.get(ctx.step_number, ())
        if not observations or ctx.landmark_range_weight_power == 0.0:
            return None
        log_likelihood = np.zeros(len(positions))
        for observation in observations:
            model = ctx.landmark_definitions[observation.beacon_id].path_loss_model
            model = model or PathLossModel(
                BLE_PATH_LOSS_TX_POWER_DBM, BLE_PATH_LOSS_N, BLE_RSSI_SIGMA_DB
            )
            likelihood = landmark_range_likelihood(
                positions,
                ctx.landmark_meters[observation.beacon_id],
                observation.smoothed_rssi_dbm,
                model,
                ctx.landmark_likelihood_floor,
            )
            with np.errstate(divide="ignore"):
                log_likelihood += np.log(likelihood)
        return np.exp(ctx.landmark_range_weight_power * log_likelihood)
    if ctx.landmark_mode == "observation":
        detection = ctx.landmark_by_step.get(ctx.step_number)
        if detection is not None:
            return landmark_likelihood(
                positions,
                ctx.landmark_meters[detection.beacon_id],
                ctx.landmark_sigma_m,
                ctx.landmark_likelihood_floor,
            )
    return None


def record_soft_landmark_diagnostics(
    ctx: ParticleRuntime, positions: np.ndarray, weights: np.ndarray
) -> None:
    """最終的に採用する集合の観測診断を計算する。広告数を独立適用数にしない。"""
    likelihood = soft_landmark_likelihood(ctx, positions)
    ctx.landmark_update_count = int(likelihood is not None)
    ctx.landmark_observation_count = 0
    ctx.landmark_floor_fraction = None
    ctx.landmark_rssi_residual_mean_db = None
    ctx.landmark_applied = False
    if likelihood is None:
        return
    ctx.landmark_applied = True
    normalized = weights / weights.sum()
    ctx.landmark_likelihood_mean = float(np.sum(normalized * likelihood))
    if ctx.landmark_mode == "observation":
        ctx.landmark_observation_count = 1
        ctx.landmark_floor_fraction = float(
            np.sum(normalized[likelihood <= ctx.landmark_likelihood_floor + 1e-12])
        )
        return
    observations = ctx.landmark_observations_by_step.get(ctx.step_number, ())
    ctx.landmark_observation_count = len(observations)
    residuals = []
    floors = []
    for observation in observations:
        model = ctx.landmark_definitions[observation.beacon_id].path_loss_model
        model = model or PathLossModel(
            BLE_PATH_LOSS_TX_POWER_DBM, BLE_PATH_LOSS_N, BLE_RSSI_SIGMA_DB
        )
        distance = np.linalg.norm(
            positions - ctx.landmark_meters[observation.beacon_id], axis=1
        )
        residual = observation.smoothed_rssi_dbm - expected_rssi_dbm(distance, model)
        residuals.append(float(np.sum(normalized * np.abs(residual))))
        factor = landmark_range_likelihood(
            positions,
            ctx.landmark_meters[observation.beacon_id],
            observation.smoothed_rssi_dbm,
            model,
            ctx.landmark_likelihood_floor,
        )
        floors.append(
            float(np.sum(normalized[factor <= ctx.landmark_likelihood_floor + 1e-12]))
        )
    ctx.landmark_rssi_residual_mean_db = float(np.mean(residuals))
    ctx.landmark_floor_fraction = float(np.mean(floors))
