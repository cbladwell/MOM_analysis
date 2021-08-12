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

    print('      ris av     |      mnth av     |    abs err   |   rel error')
    #     dt = oheat.average_DT.values[t0].astype('timedelta64[s]')/ np.timedelta64(1, 's') # decode_times=True
    dt0 = oheat.average_DT[t0].values*24*60*60
    dt1 = oheat.average_DT[t0].values*24*60*60
    dts = oheat.average_DT[t0:t1].values*24*60*60
    mnthly_ra_temp_c1 = np.sum(mnth_tend_c1.values[t0:t1]*dts)+ra_temp_c1.values[t1]-ra_temp_c1.values[t0]
    mnthly_av_temp_c1 = mnth_temp_c1.values[t1]-mnth_temp_c1.values[t0]
    print('{0:16.6f} | {1:12.6f} | {2:12.10f} | {3:12.10f}'.format(mnthly_ra_temp_c1,
        mnthly_av_temp_c1, (mnthly_ra_temp_c1 - mnthly_av_temp_c1), (mnthly_ra_temp_c1 - mnthly_av_temp_c1)/mnthly_ra_temp_c1))

def check_global_closure_for_epoch(oheat, ora, ocean, t0, t1):
    rho0 = 1035 # kg/m^3
    Cp = 3992.1032232964 # J kg-1 degC-1
    # check that averaged ocean_heat tendency and ocean temp are equal
    mnth_tend = (oheat.temp_tendency).mean(dim=["xt_ocean","yt_ocean","st_ocean"])
    ra_temp = (ora.temp_tendency).mean(dim=["xt_ocean","yt_ocean","st_ocean"])
    mnth_temp = ocean.temp_rhodzt.mean(dim=["xt_ocean","yt_ocean","st_ocean"])*Cp

    print('      ris av     |      mnth av     |    abs err   |   rel error')
    #     dt = oheat.average_DT.values[t0].astype('timedelta64[s]')/ np.timedelta64(1, 's') # decode_times=True
    dt0 = oheat.average_DT[t0].values*24*60*60
    dt1 = oheat.average_DT[t0].values*24*60*60
    dts = oheat.average_DT[t0:t1].values*24*60*60
    mnthly_ra_temp = np.sum(mnth_tend.values[t0:t1]*dts)+ra_temp.values[t1]-ra_temp.values[t0]
    mnthly_av_temp = mnth_temp.values[t1]-mnth_temp.values[t0]
    print('{0:16.6f} | {1:12.6f} | {2:12.10f} | {3:12.10f}'.format(mnthly_ra_temp,
        mnthly_av_temp, (mnthly_ra_temp - mnthly_av_temp),
        (mnthly_ra_temp - mnthly_av_temp)/mnthly_ra_temp))



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


def check_cell_budget_closure(oheat,osnap,x,y,z,t):
    Cp = 3992.1032232964 # J kg-1 degC-1
    dts = oheat.average_DT.values*24*60*60
    if t < 1:
        print("t must be > 1")
    tend = oheat.temp_tendency[t,z,y,x]
    Hsnap = Cp*(osnap.temp_rhodzt[t,z,y,x]-osnap.temp_rhodzt[t-1,z,y,x])/dts[t]
    sfc_flux = oheat.sw_heat[t,z,y,x] + oheat.temp_vdiffuse_sbc[t,z,y,x] + oheat.sfc_hflux_pme[t,y,x] \
        + oheat.frazil_3d[t,z,y,x] + oheat.temp_eta_smooth[t,y,x] + oheat.temp_rivermix[t,z,y,x]
    transport = oheat.temp_advection[t,z,y,x] + oheat.temp_submeso[t,z,y,x] + oheat.neutral_diffusion_temp[t,z,y,x] \
        + oheat.neutral_gm_temp[t,z,y,x] + oheat.mixdownslope_temp[t,z,y,x] + oheat.temp_sigma_diff[t,z,y,x]
    vertical = oheat.temp_vdiffuse_diff_cbt[t,z,y,x] + oheat.temp_nonlocal_KPP[t,z,y,x] + oheat.temp_vdiffuse_k33[t,z,y,x]
    print("budget closure for ({0},{1},{2},{3}) = {4}".format(x,y,z,t, tend.values - (sfc_flux+transport+vertical).values))


def plot_epoch_diff_single_run(oheat, ora, ocean, t0, t1):
    rho0 = 1035 # kg/m^3
    Cp = 3992.1032232964 # J kg-1 degC-1
    # check that averaged ocean_heat tendency and ocean temp are equal
    mnth_tend = (oheat.temp_tendency).sum(dim=["st_ocean"])
    ra_temp = (ora.temp_tendency).sum(dim=["st_ocean"])
    mnth_temp = ocean.temp_rhodzt.sum(dim=["st_ocean"])*Cp

    dt0 = oheat.average_DT[t0].values*24*60*60
    dt1 = oheat.average_DT[t0].values*24*60*60
    dts = oheat.average_DT[t0:t1].values*24*60*60
    dts = np.tile(dts, (360, 300,1))
    dts = np.swapaxes(dts, 0,2)
    mnthly_ra_temp = np.sum(mnth_tend.values[t0:t1]*dts, axis=0)+ra_temp.values[t1]-ra_temp.values[t0]
    mnthly_av_temp = mnth_temp.values[t1]-mnth_temp.values[t0]
    create_figure(mnthly_av_temp, "epoch_diff")
    create_figure(mnthly_ra_temp, "tend_ra")
    create_figure((mnthly_ra_temp - mnthly_av_temp), "err")


def create_figure(vint_field, name):
    plt.clf()
    plt.pcolormesh(vint_field, cmap="bwr")
    plt.colorbar()
    plt.savefig("/home/561/cb2411/tmp/hat_av_budget_check_"  + name + ".png")

# def plot_vertical_budget(osnap, oheat):
#     tend = oheat.temp_tendency[].sum(dim=["xt_ocean","yt_ocean"])
#     Hsnap = Cp*(osnap.temp_rhodzt[t,z,y,x]-osnap.temp_rhodzt[t-1,z,y,x])/dts[t]
#     sfc_flux = oheat.sw_heat[t,z,y,x] + oheat.temp_vdiffuse_sbc[t,z,y,x] + oheat.sfc_hflux_pme[t,y,x] \
#         + oheat.frazil_3d[t,z,y,x] + oheat.temp_eta_smooth[t,y,x] + oheat.temp_rivermix[t,z,y,x]
#     transport = oheat.temp_advection[t,z,y,x] + oheat.temp_submeso[t,z,y,x] + oheat.neutral_diffusion_temp[t,z,y,x] \
#         + oheat.neutral_gm_temp[t,z,y,x] + oheat.mixdownslope_temp[t,z,y,x] + oheat.temp_sigma_diff[t,z,y,x]
#     vertical = oheat.temp_vdiffuse_diff_cbt[t,z,y,x] + oheat.temp_nonlocal_KPP[t,z,y,x] + oheat.temp_vdiffuse_k33[t,z,y,x]

def get_hat_av(oheat, ora, diag_list, t0, t1):
    dts = oheat.average_DT[t0:t1]*24*60*60
    oheat_diags = oheat[diag_list[0]].sum(dim=["xt_ocean","yt_ocean"])
    ora_diags = ora[diag_list[0]][t1,:,:,:].sum(dim=["xt_ocean","yt_ocean"]) - ora[diag_list[0]][t0,:,:,:].sum(dim=["xt_ocean","yt_ocean"])
    for diag in diag_list[1:]:
        oheat_diags = oheat_diags + oheat[diag].sum(dim=["xt_ocean","yt_ocean"])
        ora_diags = ora_diags + ora[diag][t1,:,:,:].sum(dim=["xt_ocean","yt_ocean"]) - ora[diag][t0,:,:,:].sum(dim=["xt_ocean","yt_ocean"])
    process_hat_av = np.sum(oheat_diags[t0:t1]*dts) + ora_diags
    return process_hat_av

def get_hat_av_sfc(oheat, ora, diag_name, t0, t1):
    dts = oheat.average_DT[t0:t1]*24*60*60
    oheat_diag = oheat[diag_name]
    ora_diag = ora[diag_name][t1,:,:,:] - ora[diag_name][t0,:,:,:]
    process_hat_av = np.sum(oheat_diag[t0:t1]*dts) + ora_diag
    return process_hat_av



heat_diags = ["sw_heat", "temp_vdiffuse_sbc","frazil_3d","temp_rivermix", "temp_advection", "temp_submeso",
        "neutral_diffusion_temp","neutral_gm_temp",
        "mixdownslope_temp",'temp_sigma_diff', "temp_vdiffuse_diff_cbt", "temp_nonlocal_KPP",
        "temp_vdiffuse_k33"]

def plot_standard_z_budget(oheat, osnap, t0, t1, output=None):
    Cp = 3992.1032232964 # J kg-1 degC-1
    dts = oheat.average_DT.values*24*60*60
    tend = Cp*(osnap.temp_rhodzt.mean(dim=["xt_ocean","yt_ocean"])[t1,:]-osnap.temp_rhodzt.mean(dim=["xt_ocean","yt_ocean"])[t0,:])/(dts[t1]-dts[t0])
    heat_diags = ["sw_heat", "temp_vdiffuse_sbc","frazil_3d","temp_rivermix", "temp_advection", "temp_submeso",
        "neutral_diffusion_temp","neutral_gm_temp",
        "mixdownslope_temp",'temp_sigma_diff', "temp_vdiffuse_diff_cbt", "temp_nonlocal_KPP",
        "temp_vdiffuse_k33"]
    plt.clf()
    plt.plot(tend[:25], oheat.st_ocean.values[:25], label="epoch difference")
    for diag in heat_diags:
        ohds = oheat[diag].sum(dim=["xt_ocean","yt_ocean"]).mean(dim="time")
        plt.plot(ohds[:25], oheat.st_ocean.values[:25], label=diag)
    plt.legend()
    plt.gca().invert_yaxis()
    if output:
        plt.savefig("/home/561/cb2411/tmp/z_budget_all_diags_t={0}_{1}.png".format(t0, output))
    else:
        plt.savefig("/home/561/cb2411/tmp/z_budget_all_diags_t={0}-t={1}.png".format(t0, t1))



def plot_all_z_terms(ocean, oheat, ora, t0, t1, output=None):
    Cp = 3992.1032232964 # J kg-1 degC-1
    dts = oheat.average_DT.values*24*60*60
    dts = np.tile(dts, (50,1))
    dts = np.swapaxes(dts, 0,1)
    tend = oheat.temp_tendency.sum(dim=["xt_ocean","yt_ocean"])*dts
    ra_temp = ora["temp_tendency"][t1,:,:,:].sum(dim=["xt_ocean","yt_ocean"]) - ora["temp_tendency"][t0,:,:,:].sum(dim=["xt_ocean","yt_ocean"])
    epoch = (ocean.temp_rhodzt.mean(dim=["xt_ocean","yt_ocean"])[t1,:]-ocean.temp_rhodzt.mean(dim=["xt_ocean","yt_ocean"])[t0,:])*Cp
    heat_diags = ["sw_heat", "temp_vdiffuse_sbc","frazil_3d","temp_rivermix", "temp_advection", "temp_submeso",
        "neutral_diffusion_temp","neutral_gm_temp",
        "mixdownslope_temp",'temp_sigma_diff', "temp_vdiffuse_diff_cbt", "temp_nonlocal_KPP",
        "temp_vdiffuse_k33"]
    plt.clf()
    hat_av = np.sum(tend[t0:t1,:], axis=0) + ra_temp
    plt.plot(hat_av[:25], oheat.st_ocean.values[:25], label="tendency")
    plt.plot(epoch[:25], oheat.st_ocean.values[:25], label="epoch difference")
    for diag in heat_diags:
        ohds = oheat[diag].sum(dim=["xt_ocean","yt_ocean"])*dts
        orads = ora[diag][t1,:,:,:].sum(dim=["xt_ocean","yt_ocean"]) - ora[diag][t0,:,:,:].sum(dim=["xt_ocean","yt_ocean"])
        hat_av = np.sum(ohds[0:-1,:], axis=0) + orads
        plt.plot(hat_av[:25], oheat.st_ocean.values[:25], label=diag)
    plt.legend()
    plt.gca().invert_yaxis()

    # mnthly_ra_temp = np.sum((tend[t0:t1]*dts).values) + np.sum(ra_temp.values[t1])-np.sum(ra_temp.values[t0])
    # mnthly_av_temp = np.sum(epoch.values[t1]-epoch.values[t0])
    # print(mnthly_ra_temp)

    # print('{0:16.6f} | {1:12.6f} | {2:12.10f} | {3:12.10f}'.format(mnthly_ra_temp,
    #     mnthly_av_temp, (mnthly_ra_temp - mnthly_av_temp),
    #     (mnthly_ra_temp - mnthly_av_temp)/mnthly_ra_temp))


    if output:
        plt.savefig("/home/561/cb2411/tmp/ra_z_all_diags_t={0}_{1}.png".format(t0, output))
    else:
        plt.savefig("/home/561/cb2411/tmp/ra_z_all_diags_t={0}-t={1}.png".format(t0, t1))

# for each month for each diagnostic output the depth field of both hat average and standard average with run number
def monthly_depth_figure(ocean, oheat, ora, osnap, run):
    for t0 in range(11):
        t1 = t0+1
        print("t={0}".format(t0))
        plot_standard_z_budget(oheat, osnap, t0, t1, output=run)
        plot_all_z_terms(ocean, oheat, ora, t0, t1, output=run)

# for each month output the vertically integrated map of both hat average and

def get_vint_hat_av(oheat, ora, diag_list, t0, t1):
    dts = oheat.average_DT[t0:t1]*24*60*60
    oheat_diags = oheat[diag_list[0]].sum(dim=["st_ocean"])
    ora_diags = ora[diag_list[0]][t1,:,:,:].sum(dim=["st_ocean"]) - ora[diag_list[0]][t0,:,:,:].sum(dim=["st_ocean"])
    for diag in diag_list[1:]:
        oheat_diags = oheat_diags + oheat[diag].sum(dim=["st_ocean"])
        ora_diags = ora_diags + ora[diag][t1,:,:,:].sum(dim=["st_ocean"]) - ora[diag][t0,:,:,:].sum(dim=["st_ocean"])
    process_hat_av = np.sum(oheat_diags[t0:t1]*dts) + ora_diags
    return process_hat_av

def plot_vert_int_budget_ra(ocean, oheat, ora, t0, t1):
    Cp = 3992.1032232964 # J kg-1 degC-1
    # tend = oheat.temp_tendency.sum(dim=["st_ocean"])
    # ra_temp = (ora.temp_tendency).sum(dim=["st_ocean"])
    # epoch = ocean.temp_rhodzt.mean(dim=["st_ocean"])*Cp
    sfc_flux = get_vint_hat_av(oheat, ora, ["sw_heat", "temp_vdiffuse_sbc","frazil_3d","temp_rivermix"], t0, t1)
    eta_pme_hflux = get_hat_av_sfc(oheat, ora, "sfc_hflux_pme", t0, t1) + get_hat_av_sfc(oheat, ora, "temp_eta_smooth", t0, t1)
    sfc_heat_flux = sfc_flux + eta_pme_hflux
    # transport = get_vint_hat_av(oheat, ora, ["temp_advection", "temp_submeso",
    #                                     "neutral_diffusion_temp","neutral_gm_temp",
    #                                     "mixdownslope_temp",'temp_sigma_diff'], t0, t1)
    # vertical = get_vint_hat_av(oheat, ora, ["temp_vdiffuse_diff_cbt", "temp_nonlocal_KPP",
    #                                     "temp_vdiffuse_k33"], t0, t1)


    print('      ris av     |      mnth av     |    abs err   |   rel error')
    #     dt = oheat.average_DT.values[t0].astype('timedelta64[s]')/ np.timedelta64(1, 's') # decode_times=True
    dt0 = oheat.average_DT[t0].values*24*60*60
    dt1 = oheat.average_DT[t0].values*24*60*60
    dts = oheat.average_DT[t0:t1].values*24*60*60
    dts = np.tile(dts, (50,1))
    dts = np.swapaxes(dts, 0,1)

    # plt.clf()
    # plt.pcolormesh(np.sum(tend[t0:t1]*dts, axis=0)+ ra_temp.values[t1]-ra_temp.values[t0], tend.st_ocean.values)
    # plt.colorbar()
    # plt.savefig("vert_int_tendency.png")
    # plt.clf()
    # plt.pcolormesh(epoch.values[t1,:]-epoch.values[t0,:], tend.st_ocean.values)
    # plt.colorbar()
    # plt.savefig("vert_int_epoch.png")
    plt.clf()
    plt.pcolormesh(sfc_flux)
    plt.colorbar()
    plt.savefig("vert_int_sfc_flux.png")
    # plt.clf()
    # plt.pcolormesh(transport)
    # plt.colorbar()
    # plt.savefig("vert_int_transport.png")
    # plt.clf()
    # plt.pcolormesh(vertical)
    # plt.colorbar()
    # plt.savefig("vert_int_vertical_mix.png")

# plot hat average advection for a single period

def plot_ra_diag_single_run(oheat, ora, diag_name, t0, t1):
    rho0 = 1035 # kg/m^3
    Cp = 3992.1032232964 # J kg-1 degC-1
    # check that averaged ocean_heat tendency and ocean temp are equal
    mnth_tend = (oheat[diag_name]).sum(dim=["st_ocean"])
    ra_temp = (ora[diag_name]).sum(dim=["st_ocean"])

    dt0 = oheat.average_DT[t0].values*24*60*60
    dt1 = oheat.average_DT[t0].values*24*60*60
    dts = oheat.average_DT[t0:t1].values*24*60*60
    dts = np.tile(dts, (360, 300,1))
    dts = np.swapaxes(dts, 0,2)
    mnthly_ra_temp = np.sum(mnth_tend.values[t0:t1]*dts, axis=0)+ra_temp.values[t1]-ra_temp.values[t0]
    create_figure(mnthly_ra_temp, diag_name + "_tend_ra_t={0}".format(t0))
    create_figure(mnth_tend, diag_name + "_standard_av_t={0}".format(t0))

def plot_vert_int_ra(oheat, ora):
    heat_diags = ["sw_heat", "temp_vdiffuse_sbc","frazil_3d","temp_rivermix", "temp_advection", "temp_submeso",
        "neutral_diffusion_temp","neutral_gm_temp",
        "mixdownslope_temp",'temp_sigma_diff', "temp_vdiffuse_diff_cbt", "temp_nonlocal_KPP",
        "temp_vdiffuse_k33"]
    for t0 in range(11):
        t1 = t0+1
        print("t={0}".format(t0))
        for diag in heat_diags:
            plot_ra_diag_single_run(oheat, ora, ocean, diag, t0, t1)


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

    print('\n=================================================\n')
    check_cell_budget_closure(oheat, osnap, 100,100,0,1)

    print('\n=================================================\n')
    check_global_closure_for_epoch(oheat, ora, ocean, 0, 8)


    plot_epoch_diff_single_run(oheat, ora, ocean, 0, 8)


    # plot_vertical_budget_ra(ocean, oheat, ora, 0, 11)

    # plot_all_z_terms(ocean, oheat, ora, 0, 11)

    # plot_standard_z_budget(oheat, osnap, 0, 11)

    # monthly_depth_figure(ocean, oheat, ora, osnap, run)

    sfc_flux = get_vint_hat_av(oheat, ora, ["sw_heat", "temp_vdiffuse_sbc","frazil_3d","temp_rivermix"], 0, 11)
