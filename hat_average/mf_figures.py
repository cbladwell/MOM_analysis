

import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import cProfile



def get_outputs(data_dir, exclude=[]):
  """Get the outputs from a simulation in a run directory, excluding any provided to
  as a list in the exclude parameter"""
  if len(exclude) == 0:
    return sorted([_ for _ in os.listdir(data_dir) if _.startswith("output")])
  else:
    return sorted([_ for _ in os.listdir(data_dir) if _.startswith("output") and _ not in exclude])

def get_ds(data_dir, runs, nc_file):
  """Concatenate and return multiple datasets"""
  paths = [os.path.join(data_dir, run, "ocean", nc_file) for run in runs]
  return xr.open_mfdataset(paths, decode_times = False, chunks={"st_ocean":25,"yt_ocean":100,"xt_ocean":120, "time":12}, concat_dim="time")

def compute_hat_average(std_av, ra_av, av_dt, init_t, final_t):
  std_int = std_av.isel(time=slice(init_t,final_t))*av_dt.isel(time=slice(init_t,final_t))

  hat_av_total = std_int.sum(dim="time") + ra_av.isel(time=final_t).drop("time")# - ra_av.isel(time=init_t).drop("time")
  # hat_av_total.name = std_int.name + "_hat_av"
  # return hat_av_total

def compute_epoch_diff(std_av, init_t, final_t):
  """Compute the epoch difference where epoch is single time step"""
  return std_av[final_t,:,:]- std_av[init_t,:,:]

def plot_spatial(field, mask, output, vmin=None, vmax=None, extend=None):
  plt.clf()
  cm = plt.pcolormesh(field*mask, cmap="bwr", vmin=vmin, vmax=vmax)
  cbar = plt.colorbar(cm, extend=extend)
  cbar.set_label("Jm$^{-2}$")
  plt.savefig(output)


def get_land_mask(basin_mask_nc):
    """Get mask of the continental regions"""
    bsn_msk = xr.open_dataset(basin_mask_nc)
    return np.where(np.isnan(bsn_msk.BASIN_MASK.values[0,:,:]), np.nan, 1)

def compute_std_av(std_av_da, average_output_lengths):
  return std_av_da*average_output_lengths/average_output_lengths.sum(dim="time").values()


def combined_hat_average(ra_av_da, std_av_da, init_t, dt, combine_periods, average_output_lengths):
  """"Combine multiple rising average periods.
  test for computing the simple case of a single year"""
  quotient, remainder = divmod(len(ra_av_da["time"].values), combine_periods)
  if remainder != 0:
    print("Error: number of number of averages no multiple of combine periods")
    return -1
  std_av_total_da = compute_std_av(std_av_da, average_output_lengths)
  tn0 = init_t + average_output_lengths*(24*60*60)
  ra_time_wght_total_av_da = (ra_av_da + tn0*std_av_da)/combine_periods - std_av_total_da

  return ra_time_wght_total_av_da


if __name__ == "__main__":

#%%
  access_dir = r'/scratch/e14/cb2411/access-om2/archive/1deg_jra55_iaf/'
  Cp = 3992.1032232964 # J kg-1 degC-1

  outputs = get_outputs(access_dir, exclude=["output000"])
  ocean = get_ds(access_dir, outputs, "ocean.nc")
  ocean_heat = get_ds(access_dir, outputs, "ocean_heat.nc")
  ocean_risavg = get_ds(access_dir, outputs, "ocean_risavg.nc")
  ocean_snap = get_ds(access_dir, outputs, "ocean_snap.nc")

#%%
  # Get the land mask
  land_mask = get_land_mask("/home/561/cb2411/data/basin_mask.nc")

#%%
  temp = ocean.temp_rhodzt.sum("st_ocean")*Cp
  init = 0
  ntstep = len(ocean.time.values)
  final = ntstep - 1
  temp_epoch = compute_epoch_diff(temp, init, final)
  temp_tend = ocean_heat.temp_tendency.sum("st_ocean")
  temp_tend_ra = ocean_risavg.temp_tendency.sum("st_ocean")
  # tend_hat = compute_hat_average(temp_tend, temp_tend_ra, ocean_heat.average_DT, init, final)
  dts = ocean_heat.average_DT.values*24*60*60
  dts = np.tile(dts, (360, 300,1))
  dts = np.swapaxes(dts, 0,2)

  tend_hat_av = (temp_tend[init:final,:,:]*ocean_heat.average_DT.isel(time=slice(init,final))*24*60*60).sum(dim="time") + \
                temp_tend_ra.isel(time=final).drop("time") - temp_tend_ra.isel(time=init).drop("time")

#%%
  """load values --> don't load if not necessary"""
#   temp_epoch = temp_epoch.values
#   tend_hat_av = tend_hat_av.values

# #%%
#   ## test closure of the vertically integrated
#   print("temp epoch | temp tend hat |   err")
#   print("{0:1.6f} | {1:1.6f} | {2:1.6f} ".format(temp_epoch[100,100], tend_hat_av[100,100], temp_epoch[100,100]-tend_hat_av[100,100]))


#%%
  """Spatial plot"""
  # plot_spatial(temp_epoch, land_mask, "/home/561/cb2411/tmp/epoch_diff_full.png")
  # plot_spatial(tend_hat_av, land_mask, "/home/561/cb2411/tmp/tend_hat_av_full.png")
  # plot_spatial((temp_epoch-tend_hat_av), land_mask, "/home/561/cb2411/tmp/err_hat_av_full.png")

#%%
  """Compute snapshots at the initial and final state"""
  # ocean_snap0 = xr.open_dataset(os.path.join(access_dir, "output000", "ocean", "ocean_snap.nc"),
  #                             decode_times=False, chunks={"st_ocean":25,"yt_ocean":100,"xt_ocean":120, "time":12})
  # init_heat = ocean_snap.temp_rhodzt.sum("st_ocean")[0,:,:]*Cp # fix this
  # final_heat = ocean_snap.temp_rhodzt.sum("st_ocean")[-1,:,:]*Cp
  # init_heat = init_heat.load()
  # final_heat = final_heat.load()
  # plot_spatial(final_heat-init_heat, land_mask, "/home/561/cb2411/tmp/heat_change.png")


#%%
  """Compute sfc hat averaged diagnostics"""
  # sw_heat = ocean_heat.sw_heat.sum("st_ocean")
  # sw_heat_ra = ocean_risavg.sw_heat.sum("st_ocean")
  # sw_heat_hat_av = (sw_heat[init:final,:,:]*ocean_heat.average_DT.isel(time=slice(init,final))*24*60*60).sum(dim="time") + \
  #               sw_heat_ra.isel(time=final).drop("time") - sw_heat_ra.isel(time=init).drop("time")
  # temp_vdiffuse_sbc = ocean_heat.temp_vdiffuse_sbc.sum("st_ocean")
  # temp_vdiffuse_sbc_ra = ocean_heat.temp_vdiffuse_sbc.sum("st_ocean")
  # temp_vdiffuse_sbc_hat_av = (temp_vdiffuse_sbc[init:final,:,:]*ocean_heat.average_DT.isel(time=slice(init,final))*24*60*60).sum(dim="time") + \
  #               temp_vdiffuse_sbc_ra.isel(time=final).drop("time") - temp_vdiffuse_sbc_ra.isel(time=init).drop("time")
  # frazil_3d = ocean_heat.frazil_3d.sum("st_ocean")
  # frazil_3d_ra = ocean_heat.frazil_3d.sum("st_ocean")
  # frazil_3d_hat_av = (frazil_3d[init:final,:,:]*ocean_heat.average_DT.isel(time=slice(init,final))*24*60*60).sum(dim="time") + \
  #               frazil_3d_ra.isel(time=final).drop("time") - frazil_3d_ra.isel(time=init).drop("time")

  # temp_rivermix = ocean_heat.temp_rivermix.sum("st_ocean")
  # temp_rivermix_ra = ocean_heat.temp_rivermix.sum("st_ocean")
  # temp_rivermix_hat_av = (temp_rivermix[init:final,:,:]*ocean_heat.average_DT.isel(time=slice(init,final))*24*60*60).sum(dim="time") + \
  #               temp_rivermix_ra.isel(time=final).drop("time") - temp_rivermix_ra.isel(time=init).drop("time")

  # sfc_hflux_pme = ocean_heat.sfc_hflux_pme
  # sfc_hflux_pme_ra = ocean_heat.sfc_hflux_pme
  # sfc_hflux_pme_hat_av = (sfc_hflux_pme[init:final,:,:]*ocean_heat.average_DT.isel(time=slice(init,final))*24*60*60).sum(dim="time") + \
  #               sfc_hflux_pme_ra.isel(time=final).drop("time") - sfc_hflux_pme_ra.isel(time=init).drop("time")
  # temp_eta_smooth = ocean_heat.temp_eta_smooth
  # temp_eta_smooth_ra = ocean_heat.temp_eta_smooth
  # temp_eta_smooth_hat_av = (temp_eta_smooth[init:final,:,:]*ocean_heat.average_DT.isel(time=slice(init,final))*24*60*60).sum(dim="time") + \
  #               temp_eta_smooth_ra.isel(time=final).drop("time") - temp_eta_smooth_ra.isel(time=init).drop("time")



  # sw_heat_hat_av = sw_heat_hat_av.load()
  # temp_vdiffuse_sbc_hat_av = temp_vdiffuse_sbc_hat_av.load()
  # frazil_3d_hat_av = frazil_3d_hat_av.load()
  # temp_rivermix_hat_av = temp_rivermix_hat_av.load()
  # sfc_hflux_pme_hat_av = sfc_hflux_pme_hat_av.load()
  # temp_eta_smooth_hat_av = temp_eta_smooth_hat_av.load()

  # sfc_flux = sw_heat_hat_av + temp_vdiffuse_sbc_hat_av + frazil_3d_hat_av + temp_rivermix_hat_av + sfc_hflux_pme_hat_av + temp_eta_smooth_hat_av

  # # print(sfc_flux.shape)
  # plot_spatial(sfc_flux, land_mask, "/home/561/cb2411/tmp/sfc_flux_full.png")

#%%
  """ Compute transport hat averaged diagnostics"""
  # temp_advection = ocean_heat.temp_advection.sum("st_ocean")
  # temp_advection_ra = ocean_risavg.temp_advection.sum("st_ocean")
  # temp_advection_hat_av = (temp_advection[init:final,:,:]*ocean_heat.average_DT.isel(time=slice(init,final))*24*60*60).sum(dim="time") + \
  #               temp_advection_ra.isel(time=final).drop("time") - temp_advection_ra.isel(time=init).drop("time")
  # temp_submeso = ocean_heat.temp_submeso.sum("st_ocean")
  # temp_submeso_ra = ocean_heat.temp_submeso.sum("st_ocean")
  # temp_submeso_hat_av = (temp_submeso[init:final,:,:]*ocean_heat.average_DT.isel(time=slice(init,final))*24*60*60).sum(dim="time") + \
  #               temp_submeso_ra.isel(time=final).drop("time") - temp_submeso_ra.isel(time=init).drop("time")
  # neutral_diffusion_temp = ocean_heat.neutral_diffusion_temp.sum("st_ocean")
  # neutral_diffusion_temp_ra = ocean_heat.neutral_diffusion_temp.sum("st_ocean")
  # neutral_diffusion_temp_hat_av = (neutral_diffusion_temp[init:final,:,:]*ocean_heat.average_DT.isel(time=slice(init,final))*24*60*60).sum(dim="time") + \
  #               neutral_diffusion_temp_ra.isel(time=final).drop("time") - neutral_diffusion_temp_ra.isel(time=init).drop("time")

  # neutral_gm_temp = ocean_heat.neutral_gm_temp.sum("st_ocean")
  # neutral_gm_temp_ra = ocean_heat.neutral_gm_temp.sum("st_ocean")
  # neutral_gm_temp_hat_av = (neutral_gm_temp[init:final,:,:]*ocean_heat.average_DT.isel(time=slice(init,final))*24*60*60).sum(dim="time") + \
  #               neutral_gm_temp_ra.isel(time=final).drop("time") - neutral_gm_temp_ra.isel(time=init).drop("time")

  # mixdownslope_temp = ocean_heat.mixdownslope_temp.sum("st_ocean")
  # mixdownslope_temp_ra = ocean_heat.mixdownslope_temp.sum("st_ocean")
  # mixdownslope_temp_hat_av = (mixdownslope_temp[init:final,:,:]*ocean_heat.average_DT.isel(time=slice(init,final))*24*60*60).sum(dim="time") + \
  #               mixdownslope_temp_ra.isel(time=final).drop("time") - mixdownslope_temp_ra.isel(time=init).drop("time")
  # temp_sigma_diff = ocean_heat.temp_sigma_diff.sum("st_ocean")
  # temp_sigma_diff_ra = ocean_heat.temp_sigma_diff.sum("st_ocean")
  # temp_sigma_diff_hat_av = (temp_sigma_diff[init:final,:,:]*ocean_heat.average_DT.isel(time=slice(init,final))*24*60*60).sum(dim="time") + \
  #               temp_sigma_diff_ra.isel(time=final).drop("time") - temp_sigma_diff_ra.isel(time=init).drop("time")


  # temp_advection_hat_av = temp_advection_hat_av.load()
  # temp_submeso_hat_av = temp_submeso_hat_av.load()
  # neutral_diffusion_temp_hat_av = neutral_diffusion_temp_hat_av.load()
  # neutral_gm_temp_hat_av = neutral_gm_temp_hat_av.load()
  # mixdownslope_temp_hat_av = mixdownslope_temp_hat_av.load()
  # temp_sigma_diff_hat_av = temp_sigma_diff_hat_av.load()

  # transport_hat_av = temp_advection_hat_av + temp_submeso_hat_av + neutral_diffusion_temp_hat_av + neutral_gm_temp_hat_av + mixdownslope_temp_hat_av + temp_sigma_diff_hat_av
  # print(transport_hat_av.shape)
  # plot_spatial(transport_hat_av, land_mask, "/home/561/cb2411/tmp/ocean_heat_transport_full.png")

  # plot_spatial(temp_advection_hat_av, land_mask, "/home/561/cb2411/tmp/advection.png", vmin=-1e11, vmax=1e11, extend="both")
  # plot_spatial(neutral_diffusion_temp_hat_av, land_mask, "/home/561/cb2411/tmp/neutral_diffusion_hat.png", vmin=-0.7e11, vmax=0.7e11, extend="both")

#%%
  """Standard surface diagnostics"""
  # sw_heat_std = ocean_heat.sw_heat.sum("st_ocean").mean("time")
  # temp_vdiffuse_sbc_std = ocean_heat.temp_vdiffuse_sbc.sum("st_ocean").mean("time")
  # frazil_3d_std = ocean_heat.frazil_3d.sum("st_ocean").mean("time")
  # temp_rivermix_std = ocean_heat.temp_rivermix.sum("st_ocean").mean("time")
  # sfc_hflux_pme_std = ocean_heat.sfc_hflux_pme.mean("time")
  # temp_eta_smooth_std = ocean_heat.temp_eta_smooth.mean("time")

  # sw_heat_std = sw_heat_std.load()
  # temp_vdiffuse_sbc_std = temp_vdiffuse_sbc_std.load()
  # frazil_3d_std = frazil_3d_std.load()
  # temp_rivermix_std = temp_rivermix_std.load()
  # sfc_hflux_pme_std = sfc_hflux_pme_std.load()
  # temp_eta_smooth_std = temp_eta_smooth_std.load()

  # print("fluxes loaded")

  # sfc_flux_std = sw_heat_std + temp_vdiffuse_sbc_std + frazil_3d_std+ temp_rivermix_std + sfc_hflux_pme_std + temp_eta_smooth_std
  # Dt = (ocean_heat.average_DT*24*60*60).sum(dim="time")

  # plot_spatial(sfc_flux_std*Dt, land_mask, "/home/561/cb2411/tmp/sfc_flux_standard.png", vmin=-0.7e11, vmax=0.7e11, extend="both")

  # plot_spatial(sw_heat_std*Dt, land_mask, "/home/561/cb2411/tmp/sw_heat_flux_standard.png")
  # plot_spatial(temp_vdiffuse_sbc_std*Dt, land_mask, "/home/561/cb2411/tmp/temp_vdiffuse_sbc_std_standard.png")
  # plot_spatial(frazil_3d_std*Dt, land_mask, "/home/561/cb2411/tmp/frazil_3d_std_standard.png")
  # plot_spatial(sfc_hflux_pme_std*Dt, land_mask, "/home/561/cb2411/tmp/sfc_hflux_pme_std_standard.png")
  # plot_spatial(temp_eta_smooth_std*Dt, land_mask, "/home/561/cb2411/tmp/temp_eta_smooth_std_standard.png")

#%%
  """## Compute standard transport diagnostics"""

  temp_advection_std = ocean_heat.temp_advection.sum("st_ocean").mean("time")
  # temp_submeso_std = ocean_heat.temp_submeso.sum("st_ocean").mean("time")
  neutral_diffusion_temp_std = ocean_heat.neutral_diffusion_temp.sum("st_ocean").mean("time")
  # neutral_gm_temp_std = ocean_heat.neutral_gm_temp.sum("st_ocean").mean("time")
  # mixdownslope_temp_std = ocean_heat.mixdownslope_temp.sum("st_ocean").mean("time")
  # temp_sigma_diff_std = ocean_heat.temp_sigma_diff.sum("st_ocean").mean("time")

  # temp_advection_std = temp_advection_std.load()
  # temp_submeso_std = temp_submeso_std.load()
  # neutral_diffusion_temp_std = neutral_diffusion_temp_std.load()
  # neutral_gm_temp_std = neutral_gm_temp_std.load()
  # mixdownslope_temp_std = mixdownslope_temp_std.load()
  # temp_sigma_diff_std = temp_sigma_diff_std.load()
  # print("transport loaded")

  Dt = (ocean_heat.average_DT*24*60*60).sum(dim="time")
  # transport_std = temp_advection_std + temp_submeso_std + neutral_diffusion_temp_std + neutral_gm_temp_std + mixdownslope_temp_std + temp_sigma_diff_std
  # plot_spatial(transport_std*Dt, land_mask, "/home/561/cb2411/tmp/transport_standard.png", vmin=-0.7e11, vmax=0.7e11, extend="both")

  plot_spatial(temp_advection_std*Dt, land_mask, "/home/561/cb2411/tmp/temp_advection_standard.png", vmin=-1e11,vmax=1e11,extend="both")
  # plot_spatial(temp_submeso_std*Dt, land_mask, "/home/561/cb2411/tmp/temp_submeso_standard.png")
  plot_spatial(neutral_diffusion_temp_std*Dt, land_mask, "/home/561/cb2411/tmp/neutral_diffusion_temp_standard.png", vmin=-1e11,vmax=1e11,extend="both")
  # plot_spatial(neutral_gm_temp_std*Dt, land_mask, "/home/561/cb2411/tmp/neutral_gm_temp_standard.png")
  # plot_spatial(mixdownslope_temp_std*Dt, land_mask, "/home/561/cb2411/tmp/mixdownslope_temp_standard.png")
  # plot_spatial(temp_sigma_diff_std*Dt, land_mask, "/home/561/cb2411/tmp/temp_sigma_diff_standard.png")

#%%
  """plot transport + sfc flux. Should equal tendency"""
  # plot_spatial(transport_std*Dt + sfc_flux_std*Dt, land_mask, "/home/561/cb2411/tmp/transport+sfc_flux.png")
