import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta

# Open the NetCDF file
file_path = "z_cams_l_uor_201806_v2_rf_co2_srf_lw.nc"
try:
    print(f"Opening {file_path}...")
    ds = nc.Dataset(file_path, 'r')
    
    # Extract data
    time_var = ds.variables['time'][:]
    lon_var = ds.variables['longitude'][:]
    lat_var = ds.variables['latitude'][:]
    rf_var = ds.variables['rf_co2_srf_lw'][0, :, :]  # Get the first (and only) time step
    
    # Convert time to readable format
    time_units = ds.variables['time'].units
    time_start = datetime.strptime(time_units.split('since ')[1], '%Y-%m-%d %H:%M:%S.%f')
    actual_time = time_start + timedelta(hours=float(time_var[0]))
    
    print(f"Data time: {actual_time}")
    print(f"Longitude range: {lon_var.min():.1f} degrees to {lon_var.max():.1f} degrees")
    print(f"Latitude range: {lat_var.min():.1f} degrees to {lat_var.max():.1f} degrees")
    print(f"CO2 radiative forcing range: {rf_var.min():.4f} to {rf_var.max():.4f} W/m^2")
    
    # Create a map plot
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    
    # Create a meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon_var, lat_var)
    
    # Plot the data
    cs = ax.pcolormesh(lon_grid, lat_grid, rf_var, 
                       cmap='viridis', 
                       transform=ccrs.PlateCarree(),
                       shading='auto')
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax, orientation='vertical', pad=0.05, aspect=20)
    cbar.set_label('CO2 Surface Longwave Radiative Forcing (W/m^2)', fontsize=10)
    
    # Set title and labels
    plt.title(f'CAMS CO₂ Surface Longwave Radiative Forcing\n{actual_time.strftime("%Y-%m-%d")}', 
              fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add gridlines
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('co2_radiative_forcing_map.png', dpi=300, bbox_inches='tight')
    print("Map saved as 'co2_radiative_forcing_map.png'")
    
    # Create a simple line plot showing latitudinal variation
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Calculate mean values for each latitude
    lat_means = np.mean(rf_var, axis=1)
    
    ax2.plot(lat_means, lat_var, 'b-', linewidth=2)
    ax2.set_xlabel('Mean CO₂ Radiative Forcing (W/m²)', fontsize=12)
    ax2.set_ylabel('Latitude (°N)', fontsize=12)
    ax2.set_title(f'Latitudinal Variation of CO₂ Surface Longwave Radiative Forcing\n{actual_time.strftime("%Y-%m-%d")}', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add horizontal lines for reference
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axhline(y=23.5, color='r', linestyle='--', alpha=0.5, label='Tropic of Cancer')
    ax2.axhline(y=-23.5, color='r', linestyle='--', alpha=0.5, label='Tropic of Capricorn')
    ax2.axhline(y=66.5, color='b', linestyle='--', alpha=0.5, label='Arctic Circle')
    ax2.axhline(y=-66.5, color='b', linestyle='--', alpha=0.5, label='Antarctic Circle')
    
    ax2.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('co2_radiative_forcing_lat_profile.png', dpi=300, bbox_inches='tight')
    print("Latitudinal profile saved as 'co2_radiative_forcing_lat_profile.png'")
    
    # Close the dataset
    ds.close()
    print("\nVisualization completed successfully!")
    
except Exception as e:
    print(f"Error visualizing NetCDF file: {e}")
    import traceback
    traceback.print_exc()