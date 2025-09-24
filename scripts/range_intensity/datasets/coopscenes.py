import numpy as np
import importlib
from typing import Optional, Tuple

coopscenes = importlib.import_module("coopscenes")
Dataloader, Frame = coopscenes.Dataloader, coopscenes.Frame

from .sensor_meta import SensorMeta
from range_intensity.core import points_to_indices, update_range_matrix


class CoopScenes:
    """Wrapper around CoopScenes Dataloader with convenient frame access and metadata."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = Dataloader(dataset_path)
        self.meta = self._build_metadata()

    def _build_metadata(self) -> SensorMeta:
        """Build SensorMeta from the first LiDAR frame in the dataset."""
        frame = self.get_test_frame()
        info = frame.tower.lidars.UPPER_PLATFORM.info

        # FOV depending on sensor type
        if "OS-2" in info.model_name:
            fov_vert, fov_horiz = 22.5, 360.0
        elif "OS-1" in info.model_name:
            fov_vert, fov_horiz = 45.0, 360.0
        elif "OS-0" in info.model_name:
            fov_vert, fov_horiz = 90.0, 360.0
        else:
            fov_vert = float(info.vertical_fov)
            fov_horiz = float(info.horizontal_fov)

        # Horizontal resolution
        if hasattr(info, "horizontal_scanlines"):
            horiz_grids = int(info.horizontal_scanlines)
        elif hasattr(info, "horizontal_angle_spacing"):
            horiz_grids = int(fov_horiz / info.horizontal_angle_spacing)
        else:
            raise ValueError("Sensor info must define horizontal_scanlines or horizontal_angle_spacing.")

        # Beam angles (optional)
        beam_angles = None
        if getattr(info, "beam_altitude_angles", None) is not None:
            beam_angles = np.array(info.beam_altitude_angles, dtype=float)

        return SensorMeta(
            fov_vert=fov_vert,
            fov_horiz=fov_horiz,
            vert_total_grids=int(info.vertical_scanlines),
            horiz_total_grids=horiz_grids,
            beam_altitude_angles=beam_angles,
        )

    def get_range_matrix(self, range_matrix, total_frames):
        frame_num = 0
        for record in self.dataset:
            for frame in record:
                if frame_num >= total_frames:
                    break
                x, y, z, ring = self.get_points(frame)
                r, hor_idx, ver_idx = points_to_indices(self.meta, x, y, z, ring)
                update_range_matrix(range_matrix, r, hor_idx, ver_idx, frame_num)

                frame_num += 1
                if frame_num % 100 == 0:
                    print(f"Processed {frame_num} frames.")
            if frame_num >= total_frames:
                break
        return range_matrix

    @staticmethod
    def get_points(frame: Frame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Extract x,y,z,(ring) and ranges from a frame."""
        pcl = frame.tower.lidars.UPPER_PLATFORM.points

        if pcl.dtype.names is not None and "ring" in pcl.dtype.names:
            x, y, z, ring = pcl["x"], pcl["y"], pcl["z"], pcl["ring"]
        else:
            x, y, z = pcl["x"], pcl["y"], pcl["z"]
            ring = None

        return x, y, z, ring

    def get_test_frame(self):
        return self.dataset[-1][-1]
