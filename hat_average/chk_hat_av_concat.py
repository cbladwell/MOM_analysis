import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import sys


data_dir = "/scratch/e14/cb2411/access-om2/archive/1deg_jra55_iaf/output012/ocean"

dt = 5400

otest = xr.open_dataset(os.path.join(data_dir, "ocean_test.nc"), decode_times = False,
                        chunks={"st_ocean":25,"yt_ocean":100,"xt_ocean":120, "time":12})
oheat = xr.open_dataset(os.path.join(data_dir, "ocean_heat.nc"), decode_times = False,
                        chunks={"st_ocean":25,"yt_ocean":100,"xt_ocean":120, "time":12})
ocean = xr.open_dataset(os.path.join(data_dir, "ocean.nc"), decode_times = False,
                        chunks={"st_ocean":25,"yt_ocean":100,"xt_ocean":120, "time":12})
ora = xr.open_dataset(os.path.join(data_dir, "ocean_risavg.nc"), decode_times = False,
                        chunks={"st_ocean":25,"yt_ocean":100,"xt_ocean":120, "time":12})

print(otest.temp_rhodzt_std[0,0,100,100].values)

# print(oheat.average_DT.values*(60*60*24))

print(np.sum(ocean.temp_rhodzt[:,0,100,100].values*(oheat.average_DT.values/365)))
# time in seconds but the scaling cancels


# compute the annual average
# construct the annual time steps --> need to be careful with this
# add in the average multiplied by the initial time for each monthly average
# multiply by monthly time diff/annual time diff
# sum together and subtract the annual average.
