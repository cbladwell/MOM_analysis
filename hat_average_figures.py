import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def global_average(data_dir, nc_name, diag_name):
    """plot the vertically integrated diag"""
    outputs = list_outputs(data_dir)
    ds = xr.open_dataset(os.path.join(data_dir, outputs[0], nc_name), decode_times=False, chunks={"st_ocean":5})
    glb_mean = ds["diag_name"].sum(dim=["st_ocean", "xt_ocean", "yt_ocean"])
    for _ in outputs[1:]:
        ds = xr.open_dataset(os.path.join(data_dir, _, ds), decode_times=False, chunks={"st_ocean":5})
        glb_mean = xarray.concat([v_int, ds["diag_name"].sum(dim=["st_ocean", "xt_ocean", "yt_ocean")], "time")
    return glb_mean

def list_outputs(data_dir):
    return [_ for _ in os.listdir(data_dir) if _.startswith("output")]


def concat_vert_int(data_dir, nc_name, diag_name):
    outputs = list_outputs(data_dir)
    ds = xr.open_dataset(os.path.join(data_dir, outputs[0], nc_name), decode_times=False, chunks={"st_ocean":5})
    v_int = ds["diag_name"].sum(dim="st_ocean")
    for _ in outputs[1:]:
        ds = xr.open_dataset(os.path.join(data_dir, _, ds), decode_times=False, chunks={"st_ocean":5})
        v_int = xarray.concat([v_int, ds["diag_name"].sum(dim="st_ocean")], "time")
    return v_int


def plot_global_av(diags):
    for _ in diags:
        _.plot()
    plt.xlabal("time")
    print("dir: ~/temp" + vert_int_diag.name + "map")

# def combine_hat_av(nc_ra, nc_name, diag_name):


def plot_epoch_diff_map(vert_int_diag):
    X,Y = np.meshgrid(vert_int_diag.xt_ocean, vert_int_diag.yt_ocean)
    plt.pcolormesh(X,Y, vert_int_diag)
    plt.savefig("~/temp" + vert_int_diag.name + "map")



# ./gadi_jupyter -q express -n 1 -P e14


if __name__ == "__main__":
    data_dir = r"/scratch/e14/cb2411/access-om2/archive/1deg_jra55_iaf"
    temp_gbl_av = global_average(data_dir, "ocean.nc", "temp_rhodzt")
    plot_global_av(temp_gbl_av)
