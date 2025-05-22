import osmnx as ox
import geopandas as gpd
from shapely.ops import transform as shapely_transform
import pyproj
import numpy as np
import matplotlib.pyplot as plt
from coopscenes import Dataloader

# -------------------------------
# 1. Sensorinformationen abrufen
# -------------------------------
dataset_path = "/mnt/hot_data/dataset/coopscenes/seq_1"
dataset = Dataloader(dataset_path)
meta_frame = dataset[0][0]

sensor_lat = float(meta_frame.tower.GNSS.position[0].latitude)
sensor_lon = float(meta_frame.tower.GNSS.position[0].longitude)
sensor_alt = meta_frame.tower.GNSS.position[0].altitude
extrinsic = meta_frame.tower.GNSS.info.extrinsic  # 4x4 Transformation (numpy array)

# -------------------------------
# 2. OSM-Daten abrufen und ROI definieren
# -------------------------------
radius = 200  # Suchradius in Metern
buffer_distance = 5  # Buffer um die Straßenlinie, falls keine genaue Straßenbreite vorliegt

# OSMnx-Abfrage: Alle Objekte mit dem Tag "highway" innerhalb des Radius abrufen
tags = {'highway': True}
gdf = ox.features_from_point((sensor_lat, sensor_lon), tags=tags, dist=radius)

# Filtere nur Liniengeometrien (Straßenverläufe)
roads = gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])]

# Berechne den lokalen UTM CRS basierend auf den Sensor-Koordinaten:
zone_number = int((sensor_lon + 180) / 6) + 1
if sensor_lat >= 0:
    utm_epsg = 32600 + zone_number
else:
    utm_epsg = 32700 + zone_number
utm_crs = f"EPSG:{utm_epsg}"
print("UTM CRS:", utm_crs)

# Projiziere die Straßen in den UTM-Raum
roads_utm = roads.to_crs(utm_crs)

# Wende einen festen Buffer an, um die tatsächliche Straßenbreite abzubilden
roads_utm['geometry'] = roads_utm.geometry.buffer(buffer_distance)

# Fasse alle gepufferten Straßen zu einem einzigen ROI-Polygon zusammen
roi_polygon = roads_utm.unary_union

# Visualisierung des ROI im UTM-Raum
fig, ax = plt.subplots(figsize=(8, 8))
gpd.GeoSeries(roi_polygon, crs=utm_crs).plot(ax=ax, color='lightblue', edgecolor='blue', alpha=0.5)
roads_utm.plot(ax=ax, color='black', linewidth=1)
ax.set_title('ROI Polygon basierend auf OSM-Straßen und Buffer')
plt.show()

# -------------------------------
# 3. Transformation in das LiDAR-Koordinatensystem
# -------------------------------
# Bestimme die Sensorposition in UTM-Koordinaten
transformer = pyproj.Transformer.from_crs("epsg:4326", utm_crs, always_xy=True)
sensor_x, sensor_y = transformer.transform(sensor_lon, sensor_lat)


# Funktion zur Transformation eines Punktes mittels der extrinsischen Matrix
def transform_point_to_lidar(x, y, z, extrinsic):
    point_h = np.array([x, y, z, 1.0])  # Homogene Koordinaten
    point_lidar_h = extrinsic @ point_h
    return point_lidar_h[:3]


# Funktion, die ein gesamtes Polygon in das LiDAR-Koordinatensystem transformiert
def apply_extrinsic(geom, extrinsic, sensor_origin):
    # sensor_origin: (sensor_x, sensor_y, sensor_alt) in UTM
    def transform_fn(x, y, z=None):
        if z is None:
            z = sensor_origin[2]
        pt = transform_point_to_lidar(x, y, z, extrinsic)
        return pt[0], pt[1]

    return shapely_transform(transform_fn, geom)


# Wende die Transformation auf das ROI-Polygon an
sensor_origin = (sensor_x, sensor_y, sensor_alt)
roi_lidar = apply_extrinsic(roi_polygon, extrinsic, sensor_origin)

# Visualisierung des transformierten ROI im LiDAR-Koordinatensystem
fig, ax = plt.subplots(figsize=(8, 8))
gpd.GeoSeries(roi_lidar).plot(ax=ax, color='lightgreen', edgecolor='green', alpha=0.5)
ax.set_title('ROI Polygon im LiDAR-Koordinatensystem')
plt.show()
