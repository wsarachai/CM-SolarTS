import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_netcdf_data(nc_path):
    """
    Visualize the data in a NetCDF file.
    
    Args:
        nc_path (str): Path to the NetCDF file
    """
    print(f"Visualizing NetCDF file: {nc_path}")
    
    try:
        # Open the NetCDF file
        with nc.Dataset(nc_path, 'r') as ds:
            # Extract the data
            time = ds.variables['time'][:]
            longitude = ds.variables['longitude'][:]
            latitude = ds.variables['latitude'][:]
            rf_co2 = ds.variables['rf_co2_srf_lw'][0, :, :]  # Get the first (and only) time step
            
            # Create a meshgrid for plotting
            lon_grid, lat_grid = np.meshgrid(longitude, latitude)
            
            # Create a figure with two subplots
            fig = plt.figure(figsize=(15, 10))
            
            # Plot 1: Global map of CO2 radiative forcing
            ax1 = fig.add_subplot(2, 1, 1)
            contour = ax1.contourf(lon_grid, lat_grid, rf_co2, cmap='jet', levels=20)
            plt.colorbar(contour, ax=ax1, label='CO2 Radiative Forcing (W m-2)')
            ax1.set_title('Carbon Dioxide Instantaneous Longwave Radiative Forcing at Surface')
            ax1.set_xlabel('Longitude (degrees_east)')
            ax1.set_ylabel('Latitude (degrees_north)')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Zonal mean (average over longitudes)
            ax2 = fig.add_subplot(2, 1, 2)
            zonal_mean = np.mean(rf_co2, axis=1)
            ax2.plot(latitude, zonal_mean, 'b-', linewidth=2)
            ax2.set_title('Zonal Mean CO2 Radiative Forcing')
            ax2.set_xlabel('Latitude (degrees_north)')
            ax2.set_ylabel('CO2 Radiative Forcing (W m-2)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('data/co2_radiative_forcing_visualization.png', dpi=300, bbox_inches='tight')
            print("Visualization saved to data/co2_radiative_forcing_visualization.png")
            
            # Print some statistics
            print("\n=== Data Statistics ===")
            print(f"Min value: {np.min(rf_co2):.6f} W m-2")
            print(f"Max value: {np.max(rf_co2):.6f} W m-2")
            print(f"Mean value: {np.mean(rf_co2):.6f} W m-2")
            print(f"Standard deviation: {np.std(rf_co2):.6f} W m-2")
            
            # Show the plot
            plt.show()
    
    except Exception as e:
        print(f"Error visualizing NetCDF file: {e}")

if __name__ == "__main__":
    # Path to the NetCDF file
    netcdf_file_path = "data/z_cams_l_uor_201806_v2_rf_co2_srf_lw.nc"
    
    # Check if the file exists
    if os.path.exists(netcdf_file_path):
        visualize_netcdf_data(netcdf_file_path)
    else:
        print(f"Error: NetCDF file not found at {netcdf_file_path}")