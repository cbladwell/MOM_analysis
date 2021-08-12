using NCDatasets

datadir = "/scratch/e14/cb2411/access-om2/archive/1deg_jra55_iaf/"
run = "output004"
oheat = NCDataset(joinpath(datadir, run, "ocean_heat.nc")
ora = NCDataset(joinpath(datadir, run, "ocean_heat.nc")
ocean = NCDataset(joinpath(datadir, run, "ocean_heat.nc")


sfc_flux = oheat["sw_heat"]
prinln(sfc_flux)
