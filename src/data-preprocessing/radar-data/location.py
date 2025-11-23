
import numpy as np


def find_nearest_point(latitude, longitude, lat_center, lon_center):
  """
  Find all grid points within a circle of specified diameter centered at the target coordinates.
  
  Args:
      latitude (numpy.ndarray): Array of latitude values
      longitude (numpy.ndarray): Array of longitude values
      lat_center (float): Target center latitude
      lon_center (float): Target center longitude
      diameter_km (float): Diameter of the circle in kilometers (default: 1.0 km)
      
  Returns:
      tuple: (lat_indices, lon_indices, distances) - arrays of indices and distances for points within circle
  """
  # Get grid resolution
  lat_res = np.abs(np.median(np.diff(latitude)))
  lon_res = np.abs(np.median(np.diff(longitude)))
  
  # Convert 1km to degrees at this latitude (approximate)
  km_to_lat = 1.0 / 111.0  # 1 degree ≈ 111 km
  km_to_lon = 1.0 / (111.0 * np.cos(np.radians(lat_center)))
  
  # Calculate search radius in grid points
  lat_radius = int(np.ceil(km_to_lat / lat_res))
  lon_radius = int(np.ceil(km_to_lon / lon_res))
  
  # Calculate grid resolution in km
  lat_res_km = lat_res * 111.0
  lon_res_km = lon_res * 111.0 * np.cos(np.radians(lat_center))
  
  print(f"  Grid resolution: {lat_res:.6f}° lat, {lon_res:.6f}° lon")
  print(f"  Grid resolution: {lat_res_km:.3f} km × {lon_res_km:.3f} km")
  print(f"  Search radius: {lat_radius} grid points (lat), {lon_radius} grid points (lon)")

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