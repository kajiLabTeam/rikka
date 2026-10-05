"""非有限のセンサー入力を前処理で拒否する回帰テスト。"""

import numpy as np
import pandas as pd
import pytest

from rikka.common.lib.sensors import process_sensor_data


def _sensor_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    acc = pd.DataFrame(
        {"t": [0.0, 0.1, 0.2], "x": [0.0] * 3, "y": [0.0] * 3, "z": [9.8] * 3}
    )
    gyro = pd.DataFrame(
        {"t": [0.0, 0.1, 0.2], "x": [1.0] * 3, "y": [0.0] * 3, "z": [0.0] * 3}
    )
    return acc, gyro


@pytest.mark.parametrize("sensor_name", ["加速度", "ジャイロ"])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_process_sensor_data_rejects_nonfinite_axes(
    sensor_name: str, axis: str, invalid_value: float
) -> None:
    acc, gyro = _sensor_frames()
    frame = acc if sensor_name == "加速度" else gyro
    frame.loc[1, axis] = invalid_value
    original_acc, original_gyro = acc.copy(), gyro.copy()

    with pytest.raises(ValueError, match=rf"{sensor_name}の {axis} 列.*行位置 1"):
        process_sensor_data(acc, gyro, gyro_bias_method="zero")

    pd.testing.assert_frame_equal(acc, original_acc)
    pd.testing.assert_frame_equal(gyro, original_gyro)


def test_process_sensor_data_preserves_gyro_x_only_input() -> None:
    acc, gyro = _sensor_frames()
    gyro = gyro[["t", "x"]]

    processed_acc, processed_gyro = process_sensor_data(
        acc, gyro, gyro_bias_method="zero"
    )

    np.testing.assert_allclose(processed_gyro["angle"], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(processed_acc["lin_norm"], 0.0)
