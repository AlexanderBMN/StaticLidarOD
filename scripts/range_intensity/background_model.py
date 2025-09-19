#!/usr/bin/env python3
"""
Module: background_model.py

Contains functions to train a background model (threshold matrix)
and perform background/foreground separation on new frames.
"""

import numpy as np
import os
from thresholding import compute_threshold
import open3d as o3d
from range_intensity import get_horizontal_idx, get_vertical_idx, SensorMeta
from typing import Optional


class BackgroundModel:
    def __init__(self, meta: Optional[SensorMeta] = None, range_matrix: Optional[np.ndarray] = None,
                 threshold_matrix: Optional[np.ndarray] = None) -> None:
        self.meta: Optional[SensorMeta] = meta
        self.range_matrix: Optional[np.ndarray] = range_matrix
        self.threshold_matrix: Optional[np.ndarray] = threshold_matrix

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            meta=self.meta.to_dict() if self.meta else {},
            range_matrix=self.range_matrix,
            threshold_matrix=self.threshold_matrix,
        )

    def load(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        self.meta = SensorMeta.from_dict(data["meta"].item())
        self.range_matrix = data["range_matrix"]
        self.threshold_matrix = data["threshold_matrix"]


def train_background_model(range_matrix, sensor_meta):
    """
    Learn a static background threshold for each beam direction.

    Args:
        range_matrix (np.ndarray): 3D array (vert × horiz × frames) of distances.

    Returns:
        np.ndarray: 2D array (vert × horiz) of computed thresholds.
    """
    range_thrld_matrix = np.ones((sensor_meta.vert_total_grids, sensor_meta.horiz_total_grids), dtype=np.float32) * 200
    for i in range(sensor_meta.vert_total_grids):
        for j in range(sensor_meta.horiz_total_grids):
            distances = range_matrix[i, j, :]
            range_thrld_matrix[i, j] = compute_threshold(distances)
        if i % 10 == 0:
            print(f"Background training: processed row {i}/{sensor_meta.vert_total_grids}.")
    print("Background training completed.")
    return range_thrld_matrix


def background_subtraction(test_frame, bg_model: BackgroundModel):
    """
    Separate foreground and background points in a test frame using thresholds.

    Args:
        test_frame: Single LIDAR frame.
        range_thrld_matrix (np.ndarray): 2D array (vert × horiz) of thresholds.
        FOV_vert (float): Vertical field of view in degrees.
        FOV_horiz (float): Horizontal field of view in degrees.
        azimuth_resltn (float): Azimuth resolution per grid cell.
        elevation_resltn (float): Elevation resolution per grid cell.
        horiz_total_grids (int): Number of horizontal grid cells.
        vert_total_grids (int): Number of vertical grid cells.
        beam_altitude_angles (list or None): Predefined beam altitude angles.

    Returns:
        tuple:
            foreground_points (np.ndarray): Points above threshold.
            background_points (np.ndarray): Points below threshold.
            fg_rings (np.ndarray or None): Ring indices for foreground (if available).
            bg_rings (np.ndarray or None): Ring indices for background (if available).
    """
    pcl = test_frame.tower.lidars.UPPER_PLATFORM
    if hasattr(pcl, "dtype") and pcl.dtype.names is not None and "ring" in pcl.dtype.names:
        points = np.stack((pcl["x"], pcl["y"], pcl["z"], pcl["intensity"], pcl["ring"]), axis=-1)
        rings = points[:, 4].astype(int)
    else:
        points = np.stack((pcl["x"], pcl["y"], pcl["z"], pcl["intensity"]), axis=-1)
        rings = None

    x_test, y_test, z_test = points[:, 0], points[:, 1], points[:, 2]
    r_test = np.sqrt(x_test ** 2 + y_test ** 2 + z_test ** 2)

    horizontal_idx = get_horizontal_idx(x_test, y_test, FOV_horiz, azimuth_resltn, horiz_total_grids)
    if rings is None:
        vertical_idx = get_vertical_idx(z_test, r_test, FOV_vert,
                                        elevation_resltn, vert_total_grids,
                                        beam_altitude_angles)
    else:
        vertical_idx = rings

    background_mask = np.array([
        r_test[k] >= range_thrld_matrix[vertical_idx[k], horizontal_idx[k]]
        for k in range(len(r_test))
    ])
    foreground_mask = ~background_mask

    foreground_points = np.stack((x_test, y_test, z_test), axis=-1)[foreground_mask]
    background_points = np.stack((x_test, y_test, z_test), axis=-1)[background_mask]

    if rings is not None:
        fg_rings = rings[foreground_mask]
        bg_rings = rings[background_mask]
    else:
        fg_rings, bg_rings = None, None

    return foreground_points, background_points, fg_rings, bg_rings


def apply_outlier_removal(foreground_points, fg_rings, background_points, bg_rings,
                          nb_points: int = 15, radius: float = 0.8):
    """
    Apply radius-based outlier removal on foreground points and move outliers to background.

    Args:
        foreground_points (np.ndarray): Nx3 array of foreground points.
        fg_rings (np.ndarray or None): Ring indices for foreground points.
        background_points (np.ndarray): Mx3 array of background points.
        bg_rings (np.ndarray or None): Ring indices for background points.
        nb_points (int): Minimum number of neighbors in the given radius.
        radius (float): Radius within which to search for neighbor points.

    Returns:
        tuple:
            new_foreground_points (np.ndarray): Filtered foreground points.
            new_background_points (np.ndarray): Updated background points including outliers.
            new_fg_rings (np.ndarray or None): Ring indices for filtered foreground points.
            new_bg_rings (np.ndarray or None): Ring indices for updated background points.
    """
    fg_pcd = o3d.geometry.PointCloud()
    fg_pcd.points = o3d.utility.Vector3dVector(foreground_points)

    _, ind = fg_pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    inlier_mask = np.zeros(len(foreground_points), dtype=bool)
    inlier_mask[ind] = True

    new_foreground_points = foreground_points[inlier_mask]
    new_background_points = np.vstack((background_points, foreground_points[~inlier_mask]))

    if fg_rings is not None and bg_rings is not None:
        new_fg_rings = fg_rings[inlier_mask]
        new_bg_rings = np.concatenate((bg_rings, fg_rings[~inlier_mask]))
    else:
        new_fg_rings, new_bg_rings = None, None

    return new_foreground_points, new_background_points, new_fg_rings, new_bg_rings
