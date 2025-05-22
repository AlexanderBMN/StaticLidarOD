import numpy as np
import scipy.io
from coopscenes import Dataloader

dataset = Dataloader('/mnt/hot_data/dataset/coopscenes/seq_1')
frame = dataset[0][0]

pcl = frame.tower.lidars.UPPER_PLATFORM
xyz_points = np.stack((pcl['x'], pcl['y'], pcl['z'], pcl['intensity']), axis=-1)

# Ursprüngliche Daten in Grad
angles_deg = frame.tower.lidars.UPPER_PLATFORM.info.beam_altitude_angles

# Umrechnung von Grad in Radiant
angles_rad = np.deg2rad(angles_deg)

# Umformen zu einem Spaltenvektor (128x1)
angles_rad = angles_rad.reshape(-1, 1)

# Speichern in einer .mat Datei mit dem Variablennamen 'beams_elevation'
scipy.io.savemat('beams_elevation.mat', {'beams_elevation': angles_rad})