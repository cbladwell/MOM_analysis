import pysftp
import os
import sys
import getpass
import xarray as xr
import requests
import paramiko

hostname = "gadi.nci.org.au"
username = "cb2411"
password = getpass.getpass('Password:')

def get_remote_content(hostname, username, password):
    with pysftp.Connection(host=hostname, username=username, password=password) as sftp:
        print("Connection succesfully stablished ...")

        # Switch to a remote directory
        # sftp.cwd('/scratch/e14/cb2411/access-om2/archive/1deg_jra55_iaf/')
        sftp.cwd('/home/561/cb2411/tmp')

        # Obtain structure of the remote directory
        directory_structure = sftp.listdir_attr()

        # Print data
        for attr in directory_structure:
            print(attr.filename)


def get_file(hostname, username, password, file_path):
    port = 22
    transport = paramiko.Transport((hostname, port))
    transport.connect(username=username, password=password)

    sftp = paramiko.SFTPClient.from_transport(transport)
    print(file_path)

    local_path = "/home/toff/Documents/output/hat_average/" + file_path.split("/")[-1]
    sftp.get(file_path, local_path)

    sftp.close()
    transport.close()




ds_url = 'gadi.nci.org.au'

# get_remote_content(hostname, username, password)

file_list = ["/home/561/cb2411/tmp/temp_rhodzt_spatial_map.png",
            "/home/561/cb2411/tmp/temp_rhodzt_spatial_map_epoch.png",
            "/home/561/cb2411/tmp/temp_tendency_spatial_map_ra.png",
            "/home/561/cb2411/tmp/temp_epoch_diff_hat_av_spatial_map.png",
            "/home/561/cb2411/tmp/temp_tendency_depth_ra.png",
            "/home/561/cb2411/tmp/temp_depth_epoch_diff.png"
            ]
budget_figures = ["/home/561/cb2411/tmp/hat_av_budget_check_err.png",
                "/home/561/cb2411/tmp/hat_av_budget_check_tend_ra.png",
                "/home/561/cb2411/tmp/hat_av_budget_check_epoch_diff.png",
                "/home/561/cb2411/tmp/epoch_budget_processes_depth.png",
                "/home/561/cb2411/tmp/ra_z_all_diags.png",
                "/home/561/cb2411/tmp/z_budget_all_diags.png"
                ]

depth_figures = ["ra_z_all_diags_t=0_output004.png",
                "ra_z_all_diags_t=0-t=11.png", "ra_z_all_diags_t=10_output004.png",
                "ra_z_all_diags_t=1_output004.png","ra_z_all_diags_t=2_output004.png",
                "ra_z_all_diags_t=3_output004.png","ra_z_all_diags_t=4_output004.png",
                "ra_z_all_diags_t=5_output004.png","ra_z_all_diags_t=6_output004.png",
                "ra_z_all_diags_t=7_output004.png","ra_z_all_diags_t=8_output004.png",
                "ra_z_all_diags_t=9_output004.png", "z_budget_all_diags_t=0_output004.png",
                "z_budget_all_diags_t=0-t=11.png", "z_budget_all_diags_t=10_output004.png",
                "z_budget_all_diags_t=1_output004.png", "z_budget_all_diags_t=2_output004.png",
                "z_budget_all_diags_t=3_output004.png", "z_budget_all_diags_t=4_output004.png",
                "z_budget_all_diags_t=5_output004.png", "z_budget_all_diags_t=6_output004.png",
                "z_budget_all_diags_t=7_output004.png", "z_budget_all_diags_t=8_output004.png",
                "z_budget_all_diags_t=9_output004.png"]
for path in depth_figures:
    try:
        path = "/home/561/cb2411/tmp/{0}".format(path)
        get_file(hostname, username, password, path)

    except:
        print(path + " not present")
