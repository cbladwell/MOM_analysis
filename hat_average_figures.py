import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re


def list_outputs(data_dir):
    return sorted([_ for _ in os.listdir(data_dir) if _.startswith("output")])

def global_average(data_dir, nc_name, diag_name):
    """plot the vertically integrated diag"""
    outputs = list_outputs(data_dir)
    ds = xr.open_dataset(os.path.join(data_dir, outputs[1], "ocean", nc_name), decode_times=False, chunks={"st_ocean":5})
    glb_mean = ds[diag_name].sum(dim=["st_ocean", "xt_ocean", "yt_ocean"])
    for _ in outputs[2:]:
        ds = xr.open_dataset(os.path.join(data_dir, _, "ocean", nc_name), decode_times=False, chunks={"st_ocean":5})
        glb_mean = xr.concat([glb_mean, ds[diag_name].sum(dim=["st_ocean", "xt_ocean", "yt_ocean"])], "time")
    return glb_mean


def concat_vert_int(data_dir, nc_name, diag_name):
    outputs = list_outputs(data_dir)
    ds = xr.open_dataset(os.path.join(data_dir, outputs[1], "ocean", nc_name), decode_times=False, chunks={"st_ocean":5})
    v_int = ds[diag_name].sum(dim="st_ocean")
    for _ in outputs[2:]:
        ds = xr.open_dataset(os.path.join(data_dir, _, "ocean", nc_name), decode_times=False, chunks={"st_ocean":5})
        v_int = xr.concat([v_int, ds[diag_name].sum(dim="st_ocean")], "time")
    return v_int

def concat_av_DT(data_dir, nc_name):
    outputs = list_outputs(data_dir)
    ds = xr.open_dataset(os.path.join(data_dir, outputs[1], "ocean", nc_name), decode_times=False, chunks={"st_ocean":5})
    avdt = ds["average_DT"]
    for _ in outputs[2:]:
        ds = xr.open_dataset(os.path.join(data_dir, _, "ocean", nc_name), decode_times=False, chunks={"st_ocean":5})
        avdt = xr.concat([avdt, ds["average_DT"]], "time")
    return avdt.values


def plot_global_av(diags):
    """Plot the global average of multiple diags"""
    for diag in diags:
        diag.plot()
    plt.xlabel("time")
    # plt.show()
    print("save to: ", "/home/561/cb2411/temp_" + diag.name + "_gbl_av.png")
    plt.savefig("/home/561/cb2411/tmp/" + diag.name + "_gbl_av.png")

# def combine_hat_av(nc_ra, nc_name, diag_name)

def get_land_mask(basin_mask_nc):
    """Get mask of the continental regions"""
    bsn_msk = xr.open_dataset(basin_mask_nc)
    return np.where(np.isnan(bsn_msk.BASIN_MASK.values[0,:,:]), np.nan, 1)



def plot_spatial_vert_int_map(vert_int_diag, basin_mask, name=None, id=""):
    """Spatial plot of a """
    X,Y = np.meshgrid(vert_int_diag.xt_ocean.values, vert_int_diag.yt_ocean.values)
    plt.clf()
    plt.pcolormesh(vert_int_diag*basin_mask, cmap="bwr")
    plt.colorbar()
    if not name:
        plt.savefig("/home/561/cb2411/tmp/" + vert_int_diag.name + "_spatial_map" + id + ".png")
    else:
        plt.savefig("/home/561/cb2411/tmp/" + name + "_spatial_map" + id + ".png")

def concat_hat_av_vert_int(data_dir, nc_name, ra_name, diag_name, init_run_number):
    """init time  and final time """
    vert_int = concat_vert_int(data_dir, nc_name, diag_name)
    init_output = "output" + str(init_run_number).zfill(3)
    final_output = find_output_from_times(vert_int.time.values, init_run_number)
    init_ra = xr.open_dataset(os.path.join(data_dir, init_output, "ocean", ra_name), decode_times=False, chunks={"st_ocean":5})
    final_ra = xr.open_dataset(os.path.join(data_dir, final_output, "ocean", ra_name), decode_times=False, chunks={"st_ocean":5})
    dt0 = init_ra.average_DT[0].values*24*60*60
    dt1 = final_ra.average_DT[-1].values*24*60*60
    dts = concat_av_DT(data_dir, nc_name)*24*60*60
    dts = np.tile(dts, (360, 300,1))
    dts = np.swapaxes(dts, 0,2)
    print(dts.shape)
    ra = final_ra[diag_name].sum(dim="st_ocean")[-1,:,:] + \
        (vert_int[1:-1,:,:]*dts[1:-1,:,:]).sum(dim="time") - \
        init_ra[diag_name].sum(dim="st_ocean")[0,:,:]
    return ra

def epoch_diff(data_dir, nc_name, diag_name, init_run_number, final_run_number):
    """return the epoch difference for a diagnosic"""
    init_output = "output" + str(init_run_number).zfill(3)
    final_output = "output" + str(final_run_number).zfill(3)
    init_run = xr.open_dataset(os.path.join(data_dir, init_output, "ocean", nc_name), decode_times=False, chunks={"st_ocean":5})
    final_run = xr.open_dataset(os.path.join(data_dir, final_output, "ocean", nc_name), decode_times=False, chunks={"st_ocean":5})
    dt0 = init_run.average_DT[0].values*24*60*60
    dt1 = final_run.average_DT[-1].values*24*60*60
    epoch_diff = final_run[diag_name].sum(dim="st_ocean")[-1,:,:] - \
                init_run[diag_name].sum(dim="st_ocean")[0,:,:]
    return epoch_diff


def find_output_from_times(time_array, init_run_number):
    n = str(init_run_number + (len(time_array)-1)//12)
    return "output" + n.zfill(3)




# Note: temp_tendency is the heat content change but temp_rhodzt is the temp
#       multiplied by the density and temperature: i.e multiply by temp_rhodzt*Cp


# ./gadi_jupyter -q express -n 1 -P e14


if __name__ == "__main__":
    data_dir = r"/scratch/e14/cb2411/access-om2/archive/1deg_jra55_iaf"
    print(list_outputs(data_dir))
    temp_gbl_av = global_average(data_dir, "ocean.nc", "temp_rhodzt")
    # print(temp_gbl_av)
    # plot_global_av([temp_gbl_av])

    Cp = 3992.1032232964

    temp_vert_int = concat_vert_int(data_dir, "ocean.nc", "temp_rhodzt")

    temp_epoch_diff = epoch_diff(data_dir, "ocean.nc", "temp_rhodzt", 1, 7)*Cp

    temp_tend_ra = concat_hat_av_vert_int(data_dir, "ocean_heat.nc", "ocean_risavg.nc", "temp_tendency", 1)

    bsn_msk = get_land_mask("/home/561/cb2411/data/basin_mask.nc")

    plot_spatial_vert_int_map(temp_vert_int.mean(dim="time"), bsn_msk)

    plot_spatial_vert_int_map(temp_epoch_diff, bsn_msk, id="_epoch")

    plot_spatial_vert_int_map(temp_tend_ra, bsn_msk, id="_ra")

    plot_spatial_vert_int_map(temp_epoch_diff- temp_tend_ra, bsn_msk, name="temp_epoch_diff_hat_av")
