
from pathlib import Path

from .config import DATA_DIR


def get_data_path() -> Path:
    """Get data directory path from config file."""
    return Path(DATA_DIR)


def load_sensor_data() -> None:
    """Load accelerometer and gyroscope data from CSV files."""
    data_path = get_data_path()

    accelerometer_path = data_path / "Accelerometer.csv"
    gyroscope_path = data_path / "Gyroscope.csv"

    print(f"Data directory: {data_path}")
    print(f"Accelerometer file: {accelerometer_path}")
    print(f"Gyroscope file: {gyroscope_path}")

    # Check if files exist
    if accelerometer_path.exists():
        print("✓ Accelerometer.csv found")
    else:
        print("✗ Accelerometer.csv not found")

    if gyroscope_path.exists():
        print("✓ Gyroscope.csv found")
    else:
        print("✗ Gyroscope.csv not found")


def run() -> None:
    """Main logic for pdr.py"""
    print("pdr.py is running!")
    load_sensor_data()
