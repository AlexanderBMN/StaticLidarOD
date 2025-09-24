import numpy as np
from typing import Optional, Dict, Any


class SensorMeta:
    def __init__(
            self,
            fov_vert: float,
            fov_horiz: float,
            vert_total_grids: int,
            horiz_total_grids: int,
            beam_altitude_angles: Optional[np.ndarray] = None,
    ):

        self.fov_vert = float(fov_vert)
        self.fov_horiz = float(fov_horiz)
        self.vert_total_grids = int(vert_total_grids)
        self.horiz_total_grids = int(horiz_total_grids)

        if beam_altitude_angles is not None:
            arr = np.array(beam_altitude_angles, dtype=float)
            if arr.shape[0] != self.vert_total_grids:
                raise ValueError(
                    f"beam_altitude_angles length ({arr.shape[0]}) "
                    f"must equal vert_total_grids ({self.vert_total_grids})."
                )
            self.beam_altitude_angles = arr
        else:
            self.beam_altitude_angles = None

        # Compute per‐step angular resolution:
        self.azimuth_resltn = self.fov_horiz / float(self.horiz_total_grids)
        self.elevation_resltn = self.fov_vert / float(self.vert_total_grids)
