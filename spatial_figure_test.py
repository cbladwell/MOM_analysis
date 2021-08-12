import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import sys




run = sys.argv[1]
access_dir = r'/scratch/e14/cb2411/access-om2/archive/1deg_jra55_iaf/'
ora = xr.open_dataset(os.path.join(access_dir, run, 'ocean/ocean_risavg.nc'), decode_times=False, chunks={"xt_ocean":100, "st_ocean":25})
ogrid = xr.open_dataset(os.path.join(access_dir, run, 'ocean/ocean_grid.nc'), decode_times=False, chunks={"xt_ocean":100, "st_ocean":25})
ocean = xr.open_dataset(os.path.join(access_dir, run, 'ocean/ocean.nc'), decode_times=False, chunks={"xt_ocean":100, "st_ocean":25})
oheat = xr.open_dataset(os.path.join(access_dir, run, 'ocean/ocean_heat.nc'), decode_times=False)
osnap = xr.open_dataset(os.path.join(access_dir, run, 'ocean/ocean_snap.nc'), decode_times=False)

def std_av(ds, diag_name, t0,t1):
  return (ds[diag_name][t0:t1+1]).sum(dim=["time", "st_ocean"])



def hat_av(ds_av, ds_ora):

  av = (ds[diag_name][t0:t1,:,:,:]).sum(dim=["st_ocean"])
  ra = (ds_ora[diag_name][t1,:,:,:]).sum(dim=["st_ocean"]) - (ds_ora[diag_name][t0,:,:,:]).sum(dim=["st_ocean"])
  return av + ra
