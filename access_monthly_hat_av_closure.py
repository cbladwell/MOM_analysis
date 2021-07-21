import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import sys




run = sys.argv[1]




def check_cell_closure(oheat, ora, ocean, z,y,x):
    rho0 = 1035 # kg/m^3
    Cp = 3992.1032232964 # J kg-1 degC-1
    # check that averaged ocean_heat tendency and ocean temp are equal
    mnth_tend_c1 = (oheat.temp_tendency)[:,z,y,x]
    ra_temp_c1 = (ora.temp_tendency)[:,z,y,x]
    mnth_temp_c1 = ocean.temp_rhodzt[:,z,y,x]*Cp

    print('  i  |      ris av      |    mnth av   |   err')
    for t0 in range(11):
    #     dt = oheat.average_DT.values[t0].astype('timedelta64[s]')/ np.timedelta64(1, 's') # decode_times=True
        dt = oheat.average_DT[t0].values*24*60*60
        mnthly_ra_temp_c1 = mnth_tend_c1.values[t0]*dt+ra_temp_c1.values[t0+1]-ra_temp_c1.values[t0]
        mnthly_av_temp_c1 = mnth_temp_c1.values[t0+1]-mnth_temp_c1.values[t0]
        print(' {0:2d}  | {1:16.6f} | {2:12.2f} | {3:12.8f}'.format(t0,mnthly_ra_temp_c1,
            mnthly_av_temp_c1, (mnthly_ra_temp_c1 - mnthly_av_temp_c1)/mnthly_ra_temp_c1))



def check_cell_closure_for_epoch(oheat, ora, ocean, z,y,x, t0, t1):
    rho0 = 1035 # kg/m^3
    Cp = 3992.1032232964 # J kg-1 degC-1
    # check that averaged ocean_heat tendency and ocean temp are equal
    mnth_tend_c1 = (oheat.temp_tendency)[:,z,y,x]
    ra_temp_c1 = (ora.temp_tendency)[:,z,y,x]
    mnth_temp_c1 = ocean.temp_rhodzt[:,z,y,x]*Cp

    print('      ris av     |      mnth av     |   err')
    #     dt = oheat.average_DT.values[t0].astype('timedelta64[s]')/ np.timedelta64(1, 's') # decode_times=True
    dt0 = oheat.average_DT[t0].values*24*60*60
    dt1 = oheat.average_DT[t0].values*24*60*60
    dts = oheat.average_DT[t0:t1].values*24*60*60
    mnthly_ra_temp_c1 = np.sum(mnth_tend_c1.values[t0:t1]*dts)+ra_temp_c1.values[t1]-ra_temp_c1.values[t0]
    mnthly_av_temp_c1 = mnth_temp_c1.values[t1]-mnth_temp_c1.values[t0]
    print('{1:16.6f} | {2:12.6f} | {3:12.10f}'.format(t0,mnthly_ra_temp_c1,
        mnthly_av_temp_c1, (mnthly_ra_temp_c1 - mnthly_av_temp_c1)/mnthly_ra_temp_c1))

def check_mean_closure_for_epoch(oheat, ora, ocean, t0, t1):
    rho0 = 1035 # kg/m^3
    Cp = 3992.1032232964 # J kg-1 degC-1
    # check that averaged ocean_heat tendency and ocean temp are equal
    mnth_tend = (oheat.temp_tendency).mean(dim=["xt_ocean","yt_ocean","st_ocean"])
    ra_temp = (ora.temp_tendency).mean(dim=["xt_ocean","yt_ocean","st_ocean"])
    mnth_temp = ocean.temp_rhodzt.mean(dim=["xt_ocean","yt_ocean","st_ocean"])*Cp

    print('      ris av     |      mnth av     |   err')
    #     dt = oheat.average_DT.values[t0].astype('timedelta64[s]')/ np.timedelta64(1, 's') # decode_times=True
    dt0 = oheat.average_DT[t0].values*24*60*60
    dt1 = oheat.average_DT[t0].values*24*60*60
    dts = oheat.average_DT[t0:t1].values*24*60*60
    mnthly_ra_temp_c1 = np.sum(mnth_tend.values[t0:t1]*dts)+ra_temp.values[t1]-ra_temp.values[t0]
    mnthly_av_temp_c1 = mnth_temp.values[t1]-mnth_temp.values[t0]
    print('{1:16.6f} | {2:12.6f} | {3:12.10f}'.format(t0,mnthly_ra_temp_c1,
        mnthly_av_temp_c1, (mnthly_ra_temp_c1 - mnthly_av_temp_c1)/mnthly_ra_temp_c1))

def check_global_av_closure(oheat, ora, ocean):
    rho0 = 1035 # kg/m^3
    Cp = 3992.1032232964 # J kg-1 degC-1
    # check that averaged ocean_heat tendency and ocean temp are equal
    mnth_tend = (oheat.temp_tendency).mean(dim=["xt_ocean","yt_ocean","st_ocean"])
    ra_temp = (ora.temp_tendency).mean(dim=["xt_ocean","yt_ocean","st_ocean"])
    mnth_temp = ocean.temp_rhodzt.mean(dim=["xt_ocean","yt_ocean","st_ocean"])*Cp
    print('  i  |     ris av       |   mnth av    |      err     | total error')
    for t0 in range(11):
        dt = oheat.average_DT[t0].values*24*60*60
        mnthly_ra_temp = mnth_tend.values[t0]*dt+ra_temp.values[t0+1]-ra_temp.values[t0]
        mnthly_av_temp = mnth_temp.values[t0+1]-mnth_temp.values[t0]
        print(' {0:2d}  | {1:16.6f} | {2:12.2f} | {3:12.8f} | {4:12.8f}'.format(t0, mnthly_ra_temp, mnthly_av_temp,
                                                                    (mnthly_ra_temp - mnthly_av_temp)/mnthly_ra_temp,
                                                                    mnthly_ra_temp - mnthly_av_temp))


if __name__ == '__main__':
    access_dir = r'/scratch/e14/cb2411/access-om2/archive/1deg_jra55_iaf/'
    ora = xr.open_dataset(os.path.join(access_dir, run, 'ocean/ocean_risavg.nc'), decode_times=False)
    ogrid = xr.open_dataset(os.path.join(access_dir, run, 'ocean/ocean_grid.nc'), decode_times=False)
    ocean = xr.open_dataset(os.path.join(access_dir, run, 'ocean/ocean.nc'), decode_times=False)
    oheat = xr.open_dataset(os.path.join(access_dir, run, 'ocean/ocean_heat.nc'), decode_times=False)
    osnap = xr.open_dataset(os.path.join(access_dir, run, 'ocean/ocean_snap.nc'), decode_times=False)


    check_cell_closure(oheat, ora, ocean, 0,100,100)
    print('\n=================================================\n')
    check_cell_closure_for_epoch(oheat, ora, ocean, 0,100,100, 0, 8)
