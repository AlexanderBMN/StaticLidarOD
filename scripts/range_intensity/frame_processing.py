import numpy as np
from typing import Tuple
from range_intensity import SensorMeta


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


def process_frame(
        dataset, frame_idx
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = dataset[frame_idx]
    x, y, z, ring = dataset.get_points(frame)

    # Compute range
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Horizontal index
    hor_idx = get_horizontal_idx(x, y, dataset.meta)

    # Vertical index: if 'ring' is provided, use it; else compute from elevation
    if ring is not None:
        ver_idx = ring.astype(int)
    else:
        ver_idx = get_vertical_idx(z, r, dataset.meta)

    return r, hor_idx, ver_idx
