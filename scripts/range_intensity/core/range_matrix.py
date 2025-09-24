#!/usr/bin/env python3
import numpy as np
from range_intensity import ConfigManager


def update_range_matrix(
        range_matrix: np.ndarray,
        r: np.ndarray,
        hor_idx: np.ndarray,
        ver_idx: np.ndarray,
        frame_num: int
) -> None:
    for j in range(len(r)):
        if r[j] == 0:
            continue
        current = range_matrix[ver_idx[j], hor_idx[j], frame_num]
        if current == 0 or current > r[j]:
            range_matrix[ver_idx[j], hor_idx[j], frame_num] = r[j]


def build_range_matrix(
        dataset,
        cfg: ConfigManager,
) -> np.ndarray:
    sensor_meta = dataset.meta

    range_matrix = np.zeros(
        (sensor_meta.vert_total_grids, sensor_meta.horiz_total_grids, cfg.total_frames),
        dtype=np.float32
    )

    range_matrix = dataset.get_range_matrix(range_matrix, cfg.total_frames)

    print(f"Finished building range matrix; total frames used: {cfg.total_frames}")
    return range_matrix
