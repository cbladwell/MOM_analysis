import xarray as xr
import numpy as np
import sys
import os


def check_cell(temp_rho_dzt, tend_ra, tend_mean, oheat):
    print("  ti  |    ris av    |    mnth av    |   err")
    for t in range(len(oheat.time.values)-1):
        dt = oheat.average_DT[t].values*24*60*60
        av_heat_ra = tend_mean.values[t]*dt+tend_ra.values[t+1]-tend_ra.values[t]
        av_heat_rhodz = (temp_rho_dzt.values[t+1]-temp_rho_dzt.values[t])*Cp
        print(" {0:3d} | {1:16.6f} | {2:12.2f} | {3:15.12f}".format(t, av_heat_ra,
                                        av_heat_rhodz, (av_heat_ra-av_heat_rhodz)/av_heat_ra))


def get_cell_in_time(arr, x,y,z):
    return arr[:,z,y,x]

def extract_time_steps(in_file, ts):
    with open() as f:
        for line in f:
            if line.startswith("! 1e time step:"):
                lspl = line.split(",")
                i = lspl.index("! 1e time step:")
                if float(lspl[i]) != ts:
                    print(line)

if __name__ == "__main__":
    Cp = 3992.1032232964 # J kg-1 deg-1
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = ""
    oheat = xr.open_dataset(os.path.join(path, "ocean_heat.nc"), decode_times=False)
    ora = xr.open_dataset(os.path.join(path, "ocean_risavg.nc"), decode_times=False)
    ocean = xr.open_dataset(os.path.join(path, "ocean.nc"), decode_times=False)
    ogrid = xr.open_dataset(os.path.join(path, "ocean_grid.nc"), decode_times=False)
    osnap = xr.open_dataset(os.path.join(path, "ocean_snap.nc"), decode_times=False)


    check_cell(get_cell_in_time(ocean.temp_rhodzt, 100, 100, 0),
                get_cell_in_time(ora.temp_tendency, 100, 100, 0),
                get_cell_in_time(oheat.temp_tendency, 100, 100, 0),
                oheat)
