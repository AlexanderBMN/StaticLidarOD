from typing import Optional

import numpy as np
from range_intensity.datasets import SensorMeta
import open3d as o3d


def get_horizontal_idx(
        x: np.ndarray,
        y: np.ndarray,
        meta: SensorMeta
) -> np.ndarray:
    """
    Compute horizontal index for each point based on (x, y) in LiDAR frame.

    Args:
        x: 1D array of x-coordinates (meters).
        y: 1D array of y-coordinates (meters).
        meta: SensorMeta containing fov_horiz, azimuth_resltn, horiz_total_grids.

    Returns:
        1D array of horizontal indices (int) in [0, horiz_total_grids).
    """
    measured_azimuth = np.degrees(np.arctan2(y, x))
    measured_azimuth[measured_azimuth < 0] += meta.fov_horiz

    return np.floor(measured_azimuth / meta.azimuth_resltn).astype(int) % meta.horiz_total_grids


def get_vertical_idx(
        z: np.ndarray,
        r: np.ndarray,
        meta: SensorMeta
) -> np.ndarray:
    """
    Compute vertical index for each point based on z and r in LiDAR frame.

    Args:
        z: 1D array of z-coordinates (meters).
        r: 1D array of ranges = sqrt(x^2 + y^2 + z^2).
        meta: SensorMeta with fov_vert, elevation_resltn, vert_total_grids,
              and optional beam_altitude_angles.

    Returns:
        1D array of vertical indices (int) in [0, vert_total_grids).
    """
    measured_elevation = np.degrees(np.arcsin(z / r))
    measured_elevation[measured_elevation < 0] += meta.fov_vert

    if meta.beam_altitude_angles is not None:
        diffs = np.abs(meta.beam_altitude_angles[:, None] - measured_elevation[None, :])
        return np.argmin(diffs, axis=0)
    else:
        return (
                np.floor(measured_elevation / meta.elevation_resltn)
                .astype(int)
                % meta.vert_total_grids
        )


def points_to_indices(meta: SensorMeta, x: np.ndarray, y: np.ndarray, z: np.ndarray, ring: Optional[np.ndarray] = None):
    """Compute r, hor_idx, ver_idx given raw point data and sensor meta."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    hor_idx = get_horizontal_idx(x, y, meta)

    if ring is not None:
        ver_idx = ring.astype(int)
    else:
        ver_idx = get_vertical_idx(z, r, meta)

    return r, hor_idx, ver_idx


def background_subtraction(test_frame, bg_model: "BackgroundModel"):
    x, y, z, ring = bg_model.dataset.get_points(test_frame)
    r, hor_idx, ver_idx = points_to_indices(bg_model.dataset.meta, x, y, z, ring)

    background_mask = np.array([
        r[k] >= bg_model.threshold_matrix[ver_idx[k], hor_idx[k]]
        for k in range(len(r))
    ])
    foreground_mask = ~background_mask

    foreground_points = np.stack((x, y, z), axis=-1)[foreground_mask]
    background_points = np.stack((x, y, z), axis=-1)[background_mask]

    if ring is not None:
        fg_rings = ring[foreground_mask]
        bg_rings = ring[background_mask]
    else:
        fg_rings, bg_rings = None, None

    return foreground_points, background_points, fg_rings, bg_rings


def apply_outlier_removal(foreground_points, fg_rings, background_points, bg_rings, config):
    fg_pcd = o3d.geometry.PointCloud()
    fg_pcd.points = o3d.utility.Vector3dVector(foreground_points)

    _, ind = fg_pcd.remove_radius_outlier(nb_points=config["neighbor_points"], radius=config["radius"])
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
