#!/usr/bin/env python3

# Calculating global mean sea surface temperature from ERA5 data.
moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, moduledir)
from Functions import get_config, ERA5date_to_datetime, select_subset_ERA5

# -------------------------------------------------------------------------------------------------
config = get_config()
var = 'glosst'
vname = config['variables'][var]
fvar = 'sst'

y1, y2 = config['time']['start'], config['time']['end']

filein = config['dirs']['orig'] + config['data_original']['era5'].replace('[var]', fvar)

outdir = config['dirs']['data'] + config['dirs']['ext']
fileout = outdir + vname + '.csv'

figdir = config['dirs']['figs'] + config['dirs']['ext']
figname = figdir + vname + '_timeseries.png'
# -------------------------------------------------------------------------------------------------


# --- Creating output directories if they do not already exist ---
if not os.path.exists(outdir):
	os.makedirs(outdir)
if not os.path.exists(figdir):
	os.makedirs(figdir)



# --- Get data ---
# load file
x = xr.open_dataset(filein.replace('[var]', fvar), engine = 'netcdf4')

# select timespan
x = select_subset_ERA5(x, ylim = [y1, y2])
t = x.coords['date'].values
td = ERA5date_to_datetime(t)


# --- Calculate global mean temperature ---
x = x[fvar].mean(dim = ['longitude', 'latitude'], skipna = True).values


# --- Save the data ---
data = pd.DataFrame(data = x, index = td, columns = [vname])
data.to_csv(fileout)


# --- Plotting ---
fig, ax = plt.subplots(1, 1, figsize = [14, 4])
ax.plot(td, x)
ax.set_xlim([td[0], td[-1]])
ax.tick_params(axis = 'both', labelsize = 'x-large')
ax.set_ylabel('Global mean SST (K)', fontsize = 'xx-large')
	
fig.savefig(figname)
plt.close(fig)
