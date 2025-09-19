#!/usr/bin/env python3
"""
Module: range_matrix.py

Provides functionality to build a 3D range matrix from arbitrary LiDAR datasets,
using a generic SensorMeta helper and user‐supplied callbacks to extract per‐frame point data.
"""

import numpy as np
from range_intensity import process_frame, ConfigManager


def update_range_matrix(
        range_matrix: np.ndarray,
        r: np.ndarray,
        hor_idx: np.ndarray,
        ver_idx: np.ndarray,
        frame_num: int
) -> None:
    """
    Fill the 3D range matrix (vert × horiz × frame) with minimal distances.

    Args:
        range_matrix: 3D numpy array of shape (vert_total_grids, horiz_total_grids, total_frames).
        r: 1D array of ranges for each point in current frame.
        hor_idx: 1D array of horizontal indices (ints).
        ver_idx: 1D array of vertical indices (ints).
        frame_num: Index of the current frame [0, total_frames).
    """
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
    """
    Given a generic dataset, build a 3D range matrix of shape (vert, horiz, total_frames).

    Args:
        dataset: Iterable of records, each record is itself iterable of frames.

    Returns:
        range_matrix: 3D array with shape (vert_total_grids, horiz_total_grids, total_frames).
    """
    sensor_meta = dataset.meta

    range_matrix = np.zeros(
        (sensor_meta.vert_total_grids, sensor_meta.horiz_total_grids, cfg.total_frames),
        dtype=np.float32
    )

    for frame_idx in range(0, cfg.total_frames):
        r, hor_idx, ver_idx = process_frame(dataset, frame_idx)
        update_range_matrix(range_matrix, r, hor_idx, ver_idx, frame_idx)

    print(f"Finished building range matrix; total frames used: {cfg.total_frames}")
    return range_matrix
