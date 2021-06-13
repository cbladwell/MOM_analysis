import xarray as xr
import numpy as np
import sys



def check_cell(temp_rho_dzt, tend_ra, tend_mean, oheat):
    print("  ti  |    ris av    |    mnth av    |   err")
    for t in range(len(DT)-1):
        dt = oheat.average_DT[t].values*24*60*60
        av_heat_ra = tend_mean.values[t]*dt+tend_ra.values[t+1]-tend_ra.values[t]
        av_heat_rhodz = (temp_rho_dzt.values[t+1]-temp_rho_dzt.values[t])*Cp
        print(" {0:3d} | {1:16.6f} | {2:12.2f} | {3:15.12f}".format(t, av_heat_ra,
                                        av_heat_rhodz, (av_heat_ra-av_heat_rhodz)/av_heat_ra))


def get_cell_in_time(arr, x,y,z):
    return arr[:,z,y,x]

if __name__ == "__main__":
    Cp = 3992.1032232964 # J kg-1 deg-1
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = ""
    oheat = xr.open_dataset(os.path.join(path, "ocean_heat.nc"))
    ora = xr.open_dataset(os.path.join(path, "ocean_risavg.nc"))
    ocean = xr.open_dataset(os.path.join(path, "ocean.nc"))
    ogrid = xr.open_dataset(os.path.join(path, "ocean_grid.nc"))
    osnap = xr.open_dataset(os.path.join(path, "ocean_snap.nc"))


    check_cell(get_cell_in_time(ocean.temp_rhodzt), get_cell_in_time(ora.temp_tendency),
                                                get_cell_in_time(oheat.temp_rhodzt), oheat)
