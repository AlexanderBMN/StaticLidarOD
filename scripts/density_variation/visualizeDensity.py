import pickle
import numpy as np
import matplotlib.pyplot as plt

# Pfad zum gespeicherten Zwischenmodell (passen Sie den Pfad ggf. an)
intermediate_model_path = "/intermediate_voxel_model_vs_0.5_fs_1.pkl"

# Lade das Zwischenmodell
with open(intermediate_model_path, "rb") as f:
    intermediate_model = pickle.load(f)

voxel_densities = intermediate_model["voxel_densities"]

# Wähle den Voxel mit dem höchsten Dichte-Maximum über alle Zeitreihen
voxel_key = max(voxel_densities, key=lambda voxel: np.max(voxel_densities[voxel]))
density_series = voxel_densities[voxel_key]

# Erstelle die x-Achse als Zeit in Sekunden (bei 1 Punkt pro Sekunde)
x = np.arange(len(density_series))

# Berechne den globalen Maximalwert über alle Voxels
global_max = max(np.max(arr) for arr in voxel_densities.values())

plt.figure(figsize=(10, 5))
plt.plot(x, density_series, color="blue", label="Dichte")
plt.title("Dichte-Zeitreihe Voxel")
plt.xlabel("Zeit (s)")
plt.ylabel("Punkte")
plt.xlim(0, len(density_series) - 1)  # X-Achse beginnt bei 0
plt.ylim(0, global_max * 1.05)         # Y-Achse bis globaler Maximalwert + 5% Puffer
plt.legend()
plt.grid(True)
plt.show()
