import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import cProfile

access_dir = r'/scratch/e14/cb2411/access-om2/archive/1deg_jra55_iaf/'
Cp = 3992.1032232964 # J kg-1 degC-1

ocean_heat = xr.open_dataset(os.path.join(access_dir, "output012", "ocean", "ocean_heat.nc"), decode_times=False)
ocean = xr.open_dataset(os.path.join(access_dir, "output012", "ocean", "ocean.nc"), decode_times=False)
ocean_test = xr.open_dataset(os.path.join(access_dir, "output012", "ocean", "ocean_test.nc"), decode_times=False)
ocean_risavg = xr.open_dataset(os.path.join(access_dir, "output012", "ocean", "ocean_risavg.nc"), decode_times=False)
ocean_snap = xr.open_dataset(os.path.join(access_dir, "output011", "ocean", "ocean_snap.nc"), decode_times=False)

inot_time = ocean_snap.time.values[-1]

dt = ocean_heat.average_DT
Dt = np.sum(dt)
ocean_risavg.temp_tendency.sum("st_ocean")*dt + ocean_heat.temp_tendency.sum("st_ocean")*dt
