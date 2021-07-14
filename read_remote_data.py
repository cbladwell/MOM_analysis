import pysftp
import os
import sys
import getpass
import xarray as xr
import requests
from pydata.cas.urs import setup_session

hostname = "gadi.nci.org.au"
username = "cb2411"
password = getpass.getpass('Password:')

def get_remote_content(hostmame, username, password):
    with pysftp.Connection(host=hostname, username=username, password=password) as sftp:
        print("Connection succesfully stablished ...")

        # Switch to a remote directory
        sftp.cwd('/scratch/e14/cb2411/access-om2/archive/1deg_jra55_iaf/')

        # Obtain structure of the remote directory
        directory_structure = sftp.listdir_attr()

        # Print data
        for attr in directory_structure:
            if attr.filename.startswith("output"):
                ogrid = xr.open_dataset(os.path.join(sftp.pwd, attr.filename, "ocean/ocean_grid.nc"))
                print(ogrid)



ds_url = 'gadi.nci.org.au'

session = setup_session(username, password, check_url=ds_url)
store = xr.backends.PydapDataStore.open(ds_url, session=session)

ds = xr.open_dataset(store)
