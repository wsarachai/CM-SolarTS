
import numpy as np


def find_nearest_point(latitude, longitude, lat_center, lon_center, radius_km=1.0):
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
  
  # Calculate grid resolution in km
  lat_res_km = lat_res * 111.0
  lon_res_km = lon_res * 111.0 * np.cos(np.radians(lat_center))
  
  print(f"  Grid resolution: {lat_res:.6f}° lat, {lon_res:.6f}° lon")
  print(f"  Grid resolution: {lat_res_km:.3f} km × {lon_res_km:.3f} km")

  max_res_km = max(lat_res_km, lon_res_km)
  if radius_km < max_res_km / 2.0:
    print(f"  WARNING: Specified radius {radius_km:.3f} km is smaller than half the grid resolution {max_res_km/2.0:.3f} km")
    print(f"  Adjusting radius to {max_res_km/2.0:.3f} km")
    radius_km = max_res_km / 2.0

  # Convert radius to degrees for initial search window
  radius_lat_deg = radius_km / 111.0
  radius_lon_deg = radius_km / (111.0 * np.cos(np.radians(lat_center)))
  
  # Find approximate search bounds
  lat_min_search = lat_center - radius_lat_deg
  lat_max_search = lat_center + radius_lat_deg
  lon_min_search = lon_center - radius_lon_deg
  lon_max_search = lon_center + radius_lon_deg
  
  # Find grid indices within search bounds
  lat_mask = (latitude >= lat_min_search) & (latitude <= lat_max_search)
  lon_mask = (longitude >= lon_min_search) & (longitude <= lon_max_search)
  
  lat_candidates_idx = np.where(lat_mask)[0]
  lon_candidates_idx = np.where(lon_mask)[0]
  
  # Find all points within the circle
  points_in_circle = []
  
  for lat_idx in lat_candidates_idx:
    for lon_idx in lon_candidates_idx:
      lat = latitude[lat_idx]
      lon = longitude[lon_idx]
      
      # Calculate distance from center in km
      lat_diff_km = (lat - lat_center) * 111.0
      lon_diff_km = (lon - lon_center) * 111.0 * np.cos(np.radians(lat_center))
      distance_km = np.sqrt(lat_diff_km**2 + lon_diff_km**2)
      
      # Check if point is within the circle
      if distance_km <= radius_km:
        points_in_circle.append({
            'lat_idx': lat_idx,
            'lon_idx': lon_idx,
            'lat': lat,
            'lon': lon,
            'distance_km': distance_km
        })

  if not points_in_circle:
    print(f"  WARNING: No points found within {radius_km} km radius circle")
    print(f"  Falling back to nearest single point")
    
    # Fall back to nearest point
    center_lat_idx = np.abs(latitude - lat_center).argmin()
    center_lon_idx = np.abs(longitude - lon_center).argmin()
    
    lat = latitude[center_lat_idx]
    lon = longitude[center_lon_idx]
    lat_diff_km = (lat - lat_center) * 111.0
    lon_diff_km = (lon - lon_center) * 111.0 * np.cos(np.radians(lat_center))
    distance_km = np.sqrt(lat_diff_km**2 + lon_diff_km**2)
    
    return np.array([center_lat_idx]), np.array([center_lon_idx]), np.array([distance_km])
  
  # Sort by distance
  points_in_circle.sort(key=lambda p: p['distance_km'])
  
  # Extract arrays
  lat_indices = np.array([p['lat_idx'] for p in points_in_circle])
  lon_indices = np.array([p['lon_idx'] for p in points_in_circle])
  distances = np.array([p['distance_km'] for p in points_in_circle])
  
  # Calculate bounding box for found points
  lat_min_idx = lat_indices.min()
  lat_max_idx = lat_indices.max()
  lon_min_idx = lon_indices.min()
  lon_max_idx = lon_indices.max()
  
  actual_lat_min = latitude[lat_min_idx]
  actual_lat_max = latitude[lat_max_idx]
  actual_lon_min = longitude[lon_min_idx]
  actual_lon_max = longitude[lon_max_idx]
  
  print(f"  Circle center: ({lat_center:.6f}, {lon_center:.6f})")
  print(f"  Circle radius: {radius_km:.3f} km")
  print(f"  Points found: {len(points_in_circle)}")
  print(f"  Bounding box (degrees): lat[{actual_lat_min:.6f}, {actual_lat_max:.6f}], lon[{actual_lon_min:.6f}, {actual_lon_max:.6f}]")
  print(f"  Distance range: {distances.min():.3f} km to {distances.max():.3f} km")
  
  return lat_indices, lon_indices, distances
