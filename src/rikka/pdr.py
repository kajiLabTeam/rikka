from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .config import DATA_DIR

# Constants
WINDOW_ACC = 80
WINDOW_GYRO = 40
STEP_LENGTH = 0.3  # meters
ANGLE_SCALE = 1.2
PEAK_DISTANCE = 50
PEAK_HEIGHT = 10
SAMPLING_RATE = 100

# Column name mappings from phyphox CSV format
ACC_COLUMNS = {
    "Time (s)": "t",
    "Acceleration x (m/s^2)": "x",
    "Acceleration y (m/s^2)": "y",
    "Acceleration z (m/s^2)": "z",
}
GYRO_COLUMNS = {
    "Time (s)": "t",
    "Gyroscope x (rad/s)": "x",
    "Gyroscope y (rad/s)": "y",
    "Gyroscope z (rad/s)": "z",
}


def load_sensor_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load accelerometer and gyroscope data from CSV files."""
    data_path = Path(DATA_DIR)
    df_acc = pd.read_csv(data_path / "Accelerometer.csv").rename(columns=ACC_COLUMNS)
    df_gyro = pd.read_csv(data_path / "Gyroscope.csv").rename(columns=GYRO_COLUMNS)
    return df_acc, df_gyro


def process_sensor_data(
    df_acc: pd.DataFrame, df_gyro: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute norms, rolling averages, and angle from raw sensor data."""
    df_acc = df_acc.copy()
    df_gyro = df_gyro.copy()

    df_acc["norm"] = np.sqrt(df_acc["x"] ** 2 + df_acc["y"] ** 2 + df_acc["z"] ** 2)
    df_acc["low_norm"] = df_acc["norm"].rolling(window=WINDOW_ACC).mean()

    df_gyro["norm"] = np.sqrt(df_gyro["x"] ** 2 + df_gyro["y"] ** 2 + df_gyro["z"] ** 2)
    df_gyro["angle"] = np.cumsum(df_gyro["x"]) / SAMPLING_RATE
    df_gyro["low_angle"] = (
        df_gyro["angle"].rolling(window=WINDOW_GYRO, center=True).mean()
    )

    return df_acc, df_gyro


def detect_steps(df_acc: pd.DataFrame) -> np.ndarray:
    """Detect step peaks in the smoothed acceleration norm."""
    peaks, _ = find_peaks(
        df_acc["low_norm"].dropna(),
        distance=PEAK_DISTANCE,
        height=PEAK_HEIGHT,
    )
    return peaks


def estimate_trajectory(peaks: np.ndarray, df_gyro: pd.DataFrame) -> list[list[float]]:
    """Estimate 2D trajectory from step peaks and gyroscope angle."""
    points: list[list[float]] = [[0.0, 0.0]]
    low_angle = df_gyro["low_angle"]

    for p in peaks:
        if p >= len(low_angle):
            continue
        angle = low_angle.iloc[p] * ANGLE_SCALE
        x = points[-1][0] + STEP_LENGTH * float(np.cos(angle))
        y = points[-1][1] + STEP_LENGTH * float(np.sin(angle))
        points.append([x, y])

    return points


def plot_trajectory(trajectory: list[list[float]]) -> None:
    """Plot the estimated 2D walking trajectory."""
    df = pd.DataFrame(trajectory, columns=["x", "y"])

    plt.figure(figsize=(8, 8))
    plt.plot(df["x"], df["y"], ".-", label="Estimated trajectory", zorder=1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Walking Trajectory")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run() -> None:
    """Main PDR pipeline: load -> process -> detect steps -> estimate trajectory."""
    df_acc, df_gyro = load_sensor_data()
    df_acc, df_gyro = process_sensor_data(df_acc, df_gyro)
    peaks = detect_steps(df_acc)
    trajectory = estimate_trajectory(peaks, df_gyro)

    print(f"Steps detected: {len(peaks)}")
    for i, (x, y) in enumerate(trajectory):
        print(f"step {i}: ({x:.3f}, {y:.3f})")

    plot_trajectory(trajectory)
