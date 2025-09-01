#!/usr/bin/env python3

# Creating a land sea mask based on where sea surface temperature is defined.
moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

sys.path.insert(0, moduledir)
from Functions import get_config, prepare_map, ERA5date_to_datetime, str_to_datetime

# --------------------------------------------------------------------------------------------
config = get_config()

var = 'mask'
fvar = 'sst'

lon1, lon2 = config['region']['lon']
lat1, lat2 = config['region']['lat']

sstfile = config['dirs']['orig'] + config['data_original']['era5'].replace('[var]', fvar)

outdir = config['dirs']['data'] + config['dirs']['ext']
maskfile = outdir + config['variables']['mask'] + '.nc'

date0 = 20000101  # random date to use (they should all be the same in the mask)
# ---------------------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(outdir):
	os.makedirs(outdir)


# load the sst data, select a date and convert to DataArray
x = xr.open_dataset(sstfile)
x = x['sst'].loc[{'date' : date0}]

# create the mask
x = xr.where(np.isnan(x), 0, 1)
x = x.rename('mask')

# save the mask
x.to_netcdf(maskfile)
