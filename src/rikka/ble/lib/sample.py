"""サンプル BLE RSSI データの生成。

役割:
    実測 BLE データが無い段階で、既存の歩行データと同じ時間軸を持つサンプル
    RSSI 列を作り、CSV として書き出す。
依存元:
    ``common.settings.BleSampleSettings`` から生成条件、``common.lib.models`` から
    ``BleObservation`` を取得し、歩軌跡・ビーコン座標と NumPy / Pandas を使う。
利用先:
    CLI の ``ble-sample`` コマンドだけが使用する。本番のランドマーク測位処理
    （loader / detection / correction）はこのモジュールを import しない。
処理フロー:
    センサー時刻範囲からアドバタイズ時刻を作り、歩軌跡を時刻補間する。既定では
    歩行者とビーコンの距離、互換方式では固定ピーク時刻からガウス形状の RSSI を作り、
    ノイズと下限を適用して時刻昇順の観測列と CSV を生成する。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ...common.lib.models import BleObservation
from ...common.settings import BleSampleSettings


def build_sample_times(
    df_acc: pd.DataFrame,
    settings: BleSampleSettings,
) -> np.ndarray:
    """センサー時刻範囲から BLE アドバタイズ時刻の列を作る。"""
    if "t" not in df_acc.columns:
        raise ValueError("サンプル BLE の時間同期には加速度データの t 列が必要です。")
    t_start = max(0.0, float(df_acc["t"].min()))
    t_end = float(df_acc["t"].max())
    if not np.isfinite(t_start) or not np.isfinite(t_end):
        raise ValueError("加速度データの t 列には有限な時刻が必要です。")
    times = np.arange(t_start, t_end + 1e-9, settings.interval_s)
    if len(times) == 0:
        raise ValueError("サンプル BLE の生成対象時刻がありません。")
    return times


def generate_sample_observations(
    times: np.ndarray,
    settings: BleSampleSettings,
    *,
    walker_positions: np.ndarray | None = None,
    landmark_positions: dict[str, tuple[float, float]] | None = None,
) -> tuple[BleObservation, ...]:
    """時刻列とビーコン定義からサンプル RSSI 観測列を作る。"""
    rng = np.random.default_rng(settings.seed)
    span = settings.peak_rssi_dbm - settings.base_rssi_dbm
    per_beacon: list[tuple[str, np.ndarray]] = []
    if settings.mode == "distance":
        if walker_positions is None or landmark_positions is None:
            raise ValueError(
                "distance 方式には walker_positions と landmark_positions が必要です。"
            )
        if walker_positions.shape != (len(times), 2):
            raise ValueError("walker_positions は (時刻数, 2) の配列が必要です。")
        if not np.isfinite(walker_positions).all():
            raise ValueError("walker_positions には有限な座標が必要です。")
        if not landmark_positions:
            raise ValueError("landmark_positions は1件以上必要です。")
        for beacon_id, landmark_position in landmark_positions.items():
            landmark_xy = np.asarray(landmark_position, dtype=float)
            if landmark_xy.shape != (2,) or not np.isfinite(landmark_xy).all():
                raise ValueError(
                    f"ランドマーク {beacon_id} には有限な2次元座標が必要です。"
                )
            distances = np.linalg.norm(walker_positions - landmark_xy, axis=1)
            shape = np.exp(-(distances**2) / (2.0 * settings.sigma_m**2))
            noise = rng.normal(0.0, settings.noise_sigma_db, size=len(times))
            rssi = np.maximum(
                settings.base_rssi_dbm + span * shape + noise,
                settings.min_rssi_dbm,
            )
            per_beacon.append((beacon_id, rssi))
    else:
        for beacon_id, peak_time in settings.peak_times_s:
            shape = np.exp(-((times - peak_time) ** 2) / (2.0 * settings.sigma_s**2))
            noise = rng.normal(0.0, settings.noise_sigma_db, size=len(times))
            rssi = np.maximum(
                settings.base_rssi_dbm + span * shape + noise,
                settings.min_rssi_dbm,
            )
            per_beacon.append((beacon_id, rssi))

    return tuple(
        BleObservation(float(timestamp), beacon_id, round(float(rssi[index]), 2))
        for index, timestamp in enumerate(times)
        for beacon_id, rssi in per_beacon
    )


def resolve_walker_positions(
    sample_times: np.ndarray,
    trajectory: list[list[float]] | np.ndarray,
    t_at_steps: list[float] | np.ndarray,
) -> np.ndarray:
    """BLE サンプル時刻ごとの歩行者位置を歩軌跡から線形補間する。"""
    positions = np.asarray(trajectory, dtype=float)
    step_times = np.asarray(t_at_steps, dtype=float)
    if positions.shape != (len(step_times) + 1, 2):
        raise ValueError("trajectory は t_at_steps より1点多い2次元座標列が必要です。")
    if len(sample_times) == 0:
        raise ValueError("sample_times は1件以上必要です。")
    if len(step_times) > 1 and np.any(np.diff(step_times) <= 0.0):
        raise ValueError("t_at_steps は狭義単調増加である必要があります。")
    trajectory_times = np.concatenate(([float(sample_times[0])], step_times))
    return np.asarray(
        np.column_stack(
            (
                np.interp(sample_times, trajectory_times, positions[:, 0]),
                np.interp(sample_times, trajectory_times, positions[:, 1]),
            )
        ),
        dtype=float,
    )


def _normalized_arclength(points: np.ndarray) -> np.ndarray:
    """座標列の累積弧長を 0..1 へ正規化する。"""
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("軌跡は1点以上の2次元座標列が必要です。")
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    if cumulative[-1] <= 0.0:
        return np.linspace(0.0, 1.0, len(points))
    return np.asarray(cumulative / cumulative[-1], dtype=float)


def map_truth_to_step_times(
    truth_xy: np.ndarray,
    t_at_steps: list[float] | np.ndarray,
    reference_trajectory: list[list[float]] | np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    """正解軌跡を正規化弧長で歩時刻へ対応付ける。"""
    truth = np.asarray(truth_xy, dtype=float)
    reference = np.asarray(reference_trajectory, dtype=float)
    step_times = np.asarray(t_at_steps, dtype=float)
    if reference.shape != (len(step_times) + 1, 2):
        raise ValueError("reference_trajectory と t_at_steps の長さが一致しません。")
    truth_fraction = _normalized_arclength(truth)
    reference_fraction = _normalized_arclength(reference)
    mapped = np.column_stack(
        (
            np.interp(reference_fraction, truth_fraction, truth[:, 0]),
            np.interp(reference_fraction, truth_fraction, truth[:, 1]),
        )
    )
    initial_time = 0.0 if len(step_times) == 0 else min(0.0, float(step_times[0]))
    return mapped, [initial_time, *step_times.tolist()]


def load_truth_trajectory(path: str | Path) -> np.ndarray:
    """正解軌跡CSVを読み、開始点基準のメートル座標へ変換する。"""
    truth_path = Path(path)
    try:
        dataframe = pd.read_csv(truth_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"正解軌跡CSVを読み込めません: {truth_path}") from exc
    if not {"x", "y"}.issubset(dataframe.columns) or dataframe.empty:
        raise ValueError("正解軌跡CSVには x, y 列と1件以上の座標が必要です。")
    truth = dataframe[["x", "y"]].to_numpy(dtype=float)
    if not np.isfinite(truth).all():
        raise ValueError("正解軌跡の座標には有限値が必要です。")
    return np.asarray(truth - truth[0], dtype=float)


def write_sample_csv(
    observations: tuple[BleObservation, ...],
    output_path: str | Path,
) -> Path:
    """サンプル観測列を BLE CSV として保存し、保存先を返す。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(
        (
            {
                "timestamp_s": round(item.timestamp_s, 3),
                "beacon_id": item.beacon_id,
                "rssi_dbm": item.rssi_dbm,
            }
            for item in observations
        ),
        columns=["timestamp_s", "beacon_id", "rssi_dbm"],
    )
    dataframe.to_csv(path, index=False)
    return path


def generate_sample_csv(
    df_acc: pd.DataFrame,
    output_path: str | Path,
    settings: BleSampleSettings | None = None,
    *,
    trajectory: list[list[float]] | np.ndarray | None = None,
    t_at_steps: list[float] | np.ndarray | None = None,
    landmark_positions: dict[str, tuple[float, float]] | None = None,
) -> tuple[Path, int]:
    """センサーデータからサンプル BLE CSV を生成し、保存先と行数を返す。"""
    resolved_settings = BleSampleSettings() if settings is None else settings
    times = build_sample_times(df_acc, resolved_settings)
    walker_positions = None
    if resolved_settings.mode == "distance":
        if trajectory is None or t_at_steps is None:
            raise ValueError("distance 方式には trajectory と t_at_steps が必要です。")
        walker_positions = resolve_walker_positions(times, trajectory, t_at_steps)
    observations = generate_sample_observations(
        times,
        resolved_settings,
        walker_positions=walker_positions,
        landmark_positions=landmark_positions,
    )
    return write_sample_csv(observations, output_path), len(observations)
