#!/usr/bin/python3
#-*- coding: utf-8 -*-
#
# Test CAMS MSG+HIMAWARI NetCDF volume.
#
# Copyright (c) MINES ParisTech/Vaisala 2021-2025
# @authors: Y-M Saint-Drenan, L. Saboret
#

import sys, os, string
import netCDF4 as nc
import numpy as np
import datetime
import matplotlib
import matplotlib.pyplot as plt
from mpldatacursor import datacursor

# Workaround incompatibility between matplotlib and netcdf4.num2date() in netcdf4 >= 1.4
try:
    import cftime
    matplotlib.units.registry[cftime.real_datetime] = matplotlib.units.registry[datetime.datetime]
except ImportError:
    pass
    
#
# USAGE
#

print("Test CAMS MSG+HIMAWARI NetCDF volume.")
if len(sys.argv)-1 != 2:
    print("Usage: %s ncFilename variable" % (sys.argv[0]))
    print("    ncFilename     NetCDF file to read")
    print("    variable       variable to process among ghi bhi dhi bni gc bc dc bnc")
    sys.exit(0)

# uncomment the next line to break into debugger 
#import pdb; pdb.set_trace()

#
# CHECK PARAMETERS
#

# get NetCDF file to read
ncFilename=sys.argv[1]
print("read %s" % ncFilename)

#Â get variable to test
var=sys.argv[2]

# Get more descriptive long name
if var == "ghi":
    dataset="global_horizontal_irradiation"
elif var == "bhi":
    dataset="beam_horizontal_irradiation"
elif var == "dhi":
    dataset="diffuse_horizontal_irradiation"
elif var == "bni":
    dataset="beam_normal_irradiation"
elif var == "gc":
    dataset="global_horizontal_clear_sky_irradiation"
elif var == "bc":
    dataset="beam_horizontal_clear_sky_irradiation"
elif var == "dc":
    dataset="diffuse_horizontal_clear_sky_irradiation"
elif var == "bnc":
    dataset="beam_normal_clear_sky_irradiation"
else:
    print("ERROR: invalid variable parameter")
    exit(1)

# CAMS-Worldwide grid is: bounding box = (top=90.0Â°, bottom=-90.0Â°, left=-180.0Â°, right=180.0Â°) / resolution=0.1*0.1Â° / 3600*1800 pixels
#
# grid resolution
lat_res=-0.1
lon_res= 0.1
#
# grid bounding box (corner pixel centers)
lat_start=  90.0 + lat_res/2.0
lat_end=   -90.0 - lat_res/2.0  
lon_start=-180.0 + lon_res/2.0
lon_end=   180.0 - lon_res/2.0
#
# grid size (in pixels)
height = int(round((lat_start - lat_end) / abs(lat_res))) + 1 # 1800
width  = int(round((lon_end - lon_start) / lon_res)) + 1      # 3600

#
# TEST 1: EXTRACT TIME SERIES
#

print("Plot time series...")

#lat,lon = 15.5030,32.5604; # test location = Kharthoum
lat,lon = 44.054,5.048; # test location = Carpentras
#lat,lon = 63.431,10.395; # test location = Trondheim
#lat,lon = 1.346,103.822; # test location = Singapore
#lat,lon = -31.9514,115.8617 # test location = Perth

# get pixel in volume
x = round((lon - lon_start)/lon_res + 0.5); 
y = round((lat - lat_start)/lat_res + 0.5);

# read time series
# Note: NetCDF library takes care of converting 255 to NaN and to multiply data by 1.2 scale factor
TestData=nc.Dataset(ncFilename)
try: 
    # read time steps as DateTime objects with netcdf4 >= 1.4
    time=nc.num2date(TestData["time"][:], units=TestData["time"].units, 
                     only_use_cftime_datetimes=False, only_use_python_datetimes=True)
except TypeError:
    # read time steps as DateTime objects with netcdf4 < 1.4
    time=nc.num2date(TestData["time"][:], units=TestData["time"].units)
lat=TestData["latitude"][y]
lon=TestData["longitude"][x]
val=TestData[dataset][:,y,x] # read irradiation time series

# plot irradiation as a time series
N=len(time)
plt.plot(time[:N],val[:N]) 
plt.xlabel('Time')
plt.ylabel(dataset)
plt.title('%s at (%f, %f)' % (dataset,lat,lon))
print("(Close window to continue)")
plt.show() # block until user closes window

#
# TEST 2: READ MAP
#

print("Read map...")

# iTime=96 + 14*4 -1 # test slot = yyyy-mm-02 at 14:00 UT
iTime=96 + 5*4 -1 # test slot = yyyy-mm-02 at 05:00 UT

# read irradiation map
# Note: NetCDF library takes care of converting 255 to NaN and to multiply data by 1.2 scale factor
TestData=nc.Dataset(ncFilename)
try: 
    # read time step as DateTime objects with netcdf4 >= 1.4
    time=nc.num2date(TestData["time"][iTime],units=TestData["time"].units, 
                     only_use_cftime_datetimes=False, only_use_python_datetimes=True)
except TypeError:
    # read time step as DateTime objects with netcdf4 < 1.4
    time=nc.num2date(TestData["time"][iTime],units=TestData["time"].units)
lat=TestData["latitude"][:]
lon=TestData["longitude"][:]
val=TestData[dataset][iTime,:,:] # read irradiation map

# plot irradiation as a map (range = 0-300 Wh/m2)
plt.imshow(val, vmin=0, vmax=300, extent=[min(lon),max(lon),min(lat),max(lat)], cmap='jet')
plt.colorbar() 
datacursor() # show value picker (optional)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('%s on %s' % (dataset, time))
print("(Close window to continue)")
plt.show() # block until user closes window