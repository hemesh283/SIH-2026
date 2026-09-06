import pandas as pd
from pathlib import Path


IMU_COLUMNS = [
    "time",
    "attitude_roll",
    "attitude_pitch",
    "attitude_yaw",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "gravity_x",
    "gravity_y",
    "gravity_z",
    "acc_x",
    "acc_y",
    "acc_z",
    "mag_x",
    "mag_y",
    "mag_z",
]

VI_COLUMNS = [
    "time",
    "frame",
    "pos_x",
    "pos_y",
    "pos_z",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
]


def load_sequence(sequence_path, imu_file="imu1.csv", vi_file="vi1.csv"):
    """Load one synchronized OxIOD sequence."""
    sequence_path = Path(sequence_path)

    imu_path = sequence_path / imu_file
    vi_path = sequence_path / vi_file

    imu = pd.read_csv(
        imu_path,
        header=None,
        names=IMU_COLUMNS
    )

    vi = pd.read_csv(
        vi_path,
        header=None,
        names=VI_COLUMNS
    )

    if len(imu) != len(vi):
        raise ValueError(
            f"IMU and VI lengths do not match: "
            f"{len(imu)} vs {len(vi)}"
        )

    return imu, vi


if __name__ == "__main__":
    path = Path(
        "data/raw/"
        "Oxford Inertial Odometry Dataset_2.0/"
        "Oxford Inertial Odometry Dataset/"
        "handheld/data1/syn"
    )

    imu, vi = load_sequence(path)

    print("IMU shape:", imu.shape)
    print("VI shape:", vi.shape)

    print("\nIMU:")
    print(imu.head())

    print("\nGround Truth:")
    print(vi.head())