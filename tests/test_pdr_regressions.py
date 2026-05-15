from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rikka.analyze import pdr
from rikka.analyze.particle_filter import _normalize_floormap_gray


def test_process_sensor_data_resets_index_and_uses_time_delta_for_gyro() -> None:
    df_acc = pd.DataFrame(
        {
            "t": [0.0, 0.1, 0.3],
            "x": [0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0],
            "z": [9.8, 9.8, 9.8],
        },
        index=[10, 11, 12],
    )
    df_gyro = pd.DataFrame(
        {
            "t": [0.0, 0.1, 0.3],
            "x": [0.0, 1.0, 1.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
        },
        index=[100, 101, 102],
    )

    processed_acc, processed_gyro = pdr.process_sensor_data(df_acc, df_gyro)

    assert processed_acc.index.tolist() == [0, 1, 2]
    assert processed_gyro.index.tolist() == [0, 1, 2]
    np.testing.assert_allclose(
        processed_gyro["angle"].to_numpy(),
        np.array([0.0, 1.0 / 30.0, 0.1]),
    )


def test_sample_gyro_angle_interpolates_by_time() -> None:
    df_gyro = pd.DataFrame({"t": [10.0, 20.0], "low_angle": [0.0, 10.0]})

    assert pdr._sample_gyro_angle(df_gyro, sample_index=0, sample_time=15.0) == 5.0


def test_create_output_dir_avoids_timestamp_collisions(tmp_path) -> None:
    now = datetime(2026, 5, 15, 12, 0, 0, 123456)

    first = pdr._create_output_dir(tmp_path, now)
    second = pdr._create_output_dir(tmp_path, now)

    assert first != second
    assert first.exists()
    assert second.exists()


def test_normalize_floormap_gray_handles_float_and_uint8_images() -> None:
    image_uint8 = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    image_float = image_uint8.astype(float) / 255.0

    np.testing.assert_allclose(
        _normalize_floormap_gray(image_float),
        _normalize_floormap_gray(image_uint8),
    )


def test_particle_animation_respects_save_animation_flag(tmp_path, monkeypatch) -> None:
    map_path = tmp_path / "map.png"
    plt.imsave(map_path, np.ones((8, 8)), cmap="gray")

    df_acc = pd.DataFrame(
        {
            "t": np.arange(5, dtype=float) * 0.01,
            "x": np.zeros(5),
            "y": np.zeros(5),
            "z": np.full(5, 9.8),
        }
    )
    df_gyro = pd.DataFrame(
        {
            "t": np.arange(5, dtype=float) * 0.01,
            "x": np.zeros(5),
            "y": np.zeros(5),
            "z": np.zeros(5),
        }
    )

    output_dirs = iter([tmp_path / "first", tmp_path / "second"])

    def fake_create_output_dir() -> Path:
        output_dir = next(output_dirs)
        output_dir.mkdir()
        return output_dir

    monkeypatch.setattr(pdr, "_create_output_dir", fake_create_output_dir)
    monkeypatch.setattr(pdr, "detect_steps", lambda _df_acc: np.array([], dtype=int))

    import rikka.analyze.particle_filter as particle_filter

    calls = 0

    def fake_save_particle_animation(*_args, **_kwargs) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        particle_filter,
        "save_particle_animation",
        fake_save_particle_animation,
    )

    pdr.run(
        df_acc=df_acc,
        df_gyro=df_gyro,
        plot=False,
        use_particle_filter=True,
        floormap_path=map_path,
        origin_px=(0, 0),
    )
    assert calls == 0

    pdr.run(
        df_acc=df_acc,
        df_gyro=df_gyro,
        plot=False,
        use_particle_filter=True,
        save_animation=True,
        floormap_path=map_path,
        origin_px=(0, 0),
    )
    assert calls == 1
