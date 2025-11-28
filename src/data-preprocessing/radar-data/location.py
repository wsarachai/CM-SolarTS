
import numpy as np

def find_nearest_point(latitude, longitude, lat_center, lon_center):
  # Find nearest point first
  lat_idx = np.abs(latitude - lat_center).argmin()
  lon_idx = np.abs(longitude - lon_center).argmin()
      
  # Get window around nearest point
  lat_start = max(0, lat_idx - 1)
  lat_end = min(len(latitude), lat_idx + 2)
  lon_start = max(0, lon_idx - 1)
  lon_end = min(len(longitude), lon_idx + 2)

  # Create mask for points within window
  lat_candidates_idx = [lat_idx for lat_idx in range(lat_start, lat_end)]
  lon_candidates_idx = [lon_idx for lon_idx in range(lon_start, lon_end)]
  lat_candidates = [latitude[lat_idx] for lat_idx in range(lat_start, lat_end)]
  lon_candidates = [longitude[lon_idx] for lon_idx in range(lon_start, lon_end)]

 # Create coordinate meshes for area-weighted averaging
  lat_mesh, lon_mesh = np.meshgrid(lat_candidates, lon_candidates, indexing='ij')

  lat_mesh_rad = np.radians(lat_mesh)
  lat_distances = np.abs(lat_mesh - lat_center) * 111.0  # km
  lon_distances = np.abs(lon_mesh - lon_center) * 111.0 * np.cos(lat_mesh_rad)  # km
  distances = np.sqrt(lat_distances**2 + lon_distances**2)
  
  return lat_candidates_idx, lon_candidates_idx, distances