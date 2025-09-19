import bisect
import numpy as np
import importlib
from typing import Optional, Tuple

coopscenes = importlib.import_module("coopscenes")
Dataloader, Frame = coopscenes.Dataloader, coopscenes.Frame

from range_intensity import SensorMeta


class CoopScenes:
    """Wrapper around CoopScenes Dataloader with convenient frame access and metadata."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = Dataloader(dataset_path)
        self.meta = self._build_metadata()
        self._num_frames_cumulative = self._compute_cumulative_frames()

    def _compute_cumulative_frames(self) -> list[int]:
        """Compute cumulative frame counts across all records."""
        frame_counts = [record.num_frames for record in self.dataset]
        cumulative_list = np.cumsum(frame_counts).tolist()

        return cumulative_list

    def _build_metadata(self) -> SensorMeta:
        """Build SensorMeta from the first LiDAR frame in the dataset."""
        frame = self.dataset[0][0]
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

    def __getitem__(self, frame_idx: int) -> Frame:
        """Enable list-like access: cs[200] -> Frame with global index 200."""
        return self.get_frame(frame_idx)

    def __len__(self) -> int:
        """Total number of frames across all records."""
        return self._num_frames_cumulative[-1]

    def get_frame(self, frame_idx: int) -> Frame:
        """Return the global frame by index across all records."""
        record_idx = bisect.bisect_right(self._num_frames_cumulative, frame_idx)
        local_idx = frame_idx if record_idx == 0 else frame_idx - self._num_frames_cumulative[record_idx - 1]
        return self.dataset[record_idx][local_idx]

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
