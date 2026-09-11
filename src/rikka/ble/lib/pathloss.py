"""BLE RSSI と距離を対数距離パスロスモデルで相互変換する。

役割:
    RSSI・距離変換、RSSIばらつきの距離換算、既知距離データからの係数推定を行う。
依存元:
    ``common.lib.models.PathLossModel`` と NumPy を使用する。
利用先:
    BLE検出pipeline、実測診断、particle filterの測距尤度から使用される。
処理フロー:
    距離を0.3m以上へ制限して対数変換し、変換または最小二乗回帰を実行する。
"""

import numpy as np

from ...common.lib.models import PathLossModel

MIN_PATH_LOSS_DISTANCE_M = 0.3


def expected_rssi_dbm(
    distance_m: float | np.ndarray,
    model: PathLossModel,
) -> float | np.ndarray:
    """距離から期待 RSSI [dBm] を返す。"""
    distances = np.maximum(
        np.asarray(distance_m, dtype=float), MIN_PATH_LOSS_DISTANCE_M
    )
    values = model.tx_power_dbm - 10.0 * model.path_loss_n * np.log10(distances)
    return float(values) if values.ndim == 0 else values


def rssi_to_distance_m(rssi_dbm: float, model: PathLossModel) -> float:
    """RSSI [dBm] を推定距離 [m] へ変換する。"""
    return float(
        max(
            MIN_PATH_LOSS_DISTANCE_M,
            10.0 ** ((model.tx_power_dbm - rssi_dbm) / (10.0 * model.path_loss_n)),
        )
    )


def rssi_sigma_to_distance_sigma_m(distance_m: float, model: PathLossModel) -> float:
    """RSSI標準偏差を距離の局所標準偏差へ変換する。"""
    return float(
        distance_m * np.log(10.0) * model.rssi_sigma_db / (10.0 * model.path_loss_n)
    )


def estimate_path_loss(
    distances_m: np.ndarray | list[float],
    rssi_values: np.ndarray | list[float],
) -> PathLossModel:
    """既知距離とRSSIから A、n、残差標準偏差を最小二乗推定する。"""
    distances = np.asarray(distances_m, dtype=float)
    rssi = np.asarray(rssi_values, dtype=float)
    if distances.ndim != 1 or rssi.ndim != 1 or len(distances) != len(rssi):
        raise ValueError(
            "distances_m と rssi_values は同じ長さの1次元列にしてください。"
        )
    if (
        len(distances) < 2
        or not np.isfinite(distances).all()
        or not np.isfinite(rssi).all()
    ):
        raise ValueError("パスロス推定には有限な観測を2件以上指定してください。")
    if np.any(distances <= 0):
        raise ValueError("distances_m は正の値にしてください。")
    design = np.column_stack((np.ones(len(distances)), -10.0 * np.log10(distances)))
    coefficients, _, _, _ = np.linalg.lstsq(design, rssi, rcond=None)
    predicted = design @ coefficients
    sigma = max(
        float(np.sqrt(np.mean(np.square(rssi - predicted)))),
        float(np.finfo(float).eps),
    )
    path_loss_n = float(coefficients[1])
    if path_loss_n <= 0:
        raise ValueError(
            "推定された path_loss_n が正ではありません。起点や入力を確認してください。"
        )
    return PathLossModel(float(coefficients[0]), path_loss_n, sigma)
