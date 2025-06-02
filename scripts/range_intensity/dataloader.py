#!/usr/bin/env python3
"""
Module: dataloader.py

Provides functionality to load LIDAR frames, extract point coordinates,
compute per-frame ranges and build the 3D range matrix.
"""

import numpy as np


def load_parameters(dataset):
    """
    Extract sensor parameters from the first frame of the dataset.
    """
    meta_frame = dataset[0][0]
    FOV_vert = 22.5
    FOV_horiz = 360
    vert_total_grids = int(meta_frame.tower.lidars.UPPER_PLATFORM.info.vertical_scanlines)
    horiz_total_grids = int(meta_frame.tower.lidars.UPPER_PLATFORM.info.horizontal_scanlines)
    azimuth_resltn = FOV_horiz / horiz_total_grids
    elevation_resltn = FOV_vert / vert_total_grids
    beam_altitude_angles = meta_frame.tower.lidars.UPPER_PLATFORM.info.beam_altitude_angles
    return (FOV_vert, FOV_horiz, vert_total_grids,
            horiz_total_grids, azimuth_resltn, elevation_resltn,
            beam_altitude_angles)


def get_horizontal_idx(x, y, FOV_horiz, azimuth_resltn, horiz_total_grids):
    """
    Compute horizontal index for each point based on (x, y).
    """
    measured_azimuth = np.degrees(np.arctan2(y, x))
    measured_azimuth[measured_azimuth < 0] += FOV_horiz
    return np.floor(measured_azimuth / azimuth_resltn).astype(int) % horiz_total_grids


def get_vertical_idx(z, r, FOV_vert, elevation_resltn, vert_total_grids, beam_altitude_angles=None):
    """
    Compute vertical index for each point based on z and r.
    """
    measured_elevation = np.degrees(np.arcsin(z / r))
    measured_elevation[measured_elevation < 0] += FOV_vert
    if beam_altitude_angles is not None:
        beam_angles = np.array(beam_altitude_angles)
        diffs = np.abs(beam_angles[:, None] - measured_elevation[None, :])
        return np.argmin(diffs, axis=0)
    else:
        return np.floor(measured_elevation / elevation_resltn).astype(int) % vert_total_grids


def process_frame(frame, FOV_vert, FOV_horiz, azimuth_resltn, elevation_resltn,
                  horiz_total_grids, vert_total_grids, beam_altitude_angles=None):
    """
    Extract x, y, z from the frame and compute r, horizontal_idx, vertical_idx.
    """
    pcl = frame.tower.lidars.UPPER_PLATFORM
    if hasattr(pcl, "dtype") and pcl.dtype.names is not None and "ring" in pcl.dtype.names:
        points = np.stack((pcl["x"], pcl["y"], pcl["z"], pcl["intensity"], pcl["ring"]), axis=-1)
    else:
        points = np.stack((pcl["x"], pcl["y"], pcl["z"], pcl["intensity"]), axis=-1)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    hor_idx = get_horizontal_idx(x, y, FOV_horiz, azimuth_resltn, horiz_total_grids)

    if hasattr(pcl, "dtype") and pcl.dtype.names is not None and "ring" in pcl.dtype.names:
        ver_idx = points[:, 4].astype(int)
    else:
        ver_idx = get_vertical_idx(z, r, FOV_vert, elevation_resltn, vert_total_grids, beam_altitude_angles)

    return r, hor_idx, ver_idx


def update_range_matrix(range_matrix, r, hor_idx, ver_idx, frame_num):
    """
    Fill the 3D range matrix (vert × horiz × frame) with minimal distances.
    """
    for j in range(len(r)):
        if r[j] == 0:
            continue
        current = range_matrix[ver_idx[j], hor_idx[j], frame_num]
        if current == 0 or current > r[j]:
            range_matrix[ver_idx[j], hor_idx[j], frame_num] = r[j]


def build_range_matrix(dataset, total_frames):
    """
    Given a Dataloader, build a 3D range matrix of shape (vert, horiz, total_frames).
    """
    (FOV_vert, FOV_horiz, vert_total_grids,
     horiz_total_grids, azimuth_resltn,
     elevation_resltn, beam_altitude_angles) = load_parameters(dataset)

    range_matrix = np.zeros((vert_total_grids, horiz_total_grids, total_frames), dtype=np.float32)
    frame_num = 0
    for record in dataset:
        for frame in record:
            if frame_num >= total_frames:
                break
            r, hor_idx, ver_idx = process_frame(
                frame,
                FOV_vert, FOV_horiz,
                azimuth_resltn, elevation_resltn,
                horiz_total_grids, vert_total_grids,
                beam_altitude_angles
            )
            update_range_matrix(range_matrix, r, hor_idx, ver_idx, frame_num)
            frame_num += 1
            if frame_num % 100 == 0:
                print(f"Processed {frame_num} frames.")
        if frame_num >= total_frames:
            break

    print(f"Finished building range matrix; total frames used: {frame_num}")
    return range_matrix, (FOV_vert, FOV_horiz, vert_total_grids,
                          horiz_total_grids, azimuth_resltn,
                          elevation_resltn, beam_altitude_angles)
