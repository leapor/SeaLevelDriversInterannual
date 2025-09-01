#!/usr/bin/env python3

# Getting the Greenland ice discharge and runoff and creating one time series from it.
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
from Functions import get_config, npdatetime64_to_datetime

# -------------------------------------------------------------------------------------------------
config = get_config()
var = 'green'
vname = config['variables'][var]
y1, y2 = config['time']['start'], config['time']['end']

filein = config['dirs']['orig'] + config['data_original'][var]

outdir = config['dirs']['data'] + config['dirs']['ext']
fileout = outdir + vname + '.csv'

figdir = config['dirs']['figs'] + config['dirs']['ext']
figname = figdir + vname + '_Timeseries.png'

fvars = ['runoff_ice', 'solid_ice']
vmask = 'LSMGr'
col = config['colors'][1]
# -------------------------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(outdir):
	os.makedirs(outdir)
if not os.path.exists(figdir):
	os.makedirs(figdir)


# --- Get data ---
data = xr.open_dataset(filein, engine = 'netcdf4')

# coordinates
lon = data['lon'].values
lat = data['lat'].values
t = data.coords['TIME'].values
td = npdatetime64_to_datetime(t)

# Greenland mask (1 = Greenland; 0 = not Greenland)
mask = data['LSMGr'].values

# select timespan from data
ind = (td >= datetime.date(y1, 1, 1)) & (td <= datetime.date(y2, 12, 1))
t = t[ind]
td = td[ind]
data = data.loc[{'TIME' : t}]


# --- Calculate totals ---
x = np.zeros((len(t), len(fvars)))
for i in range(len(fvars)):
	temp = data[fvars[i]].values
	temp = temp * mask
	x[:, i] = np.nansum(data[fvars[i]].values, axis = (1, 2))
	
	
# --- Combine all runoff and discharge ---
xtot = np.nansum(x, axis = 1)


# --- Save the data ---
data = pd.DataFrame(data = xtot, index = td, columns = [vname])
data.to_csv(fileout)

# --- Plotting ---
fig, ax = plt.subplots(1, 1, figsize = [14, 4])
for i in range(len(fvars)):
	ax.plot(td, x[:, i], label = fvars[i], fillstyle = 'none', color = col[i])
ax.plot(td, xtot, label = 'total', fillstyle = 'none', color = col[len(fvars)])

ax.set_xlim([td[0], td[-1]])
ax.tick_params(axis = 'both', labelsize = 'x-large')
ax.set_ylabel('Greenland runoff and\ndischarge (km3)', fontsize = 'xx-large')
ax.legend(fontsize = 'x-large', ncols = len(var)+1, loc = 'upper left')

fig.savefig(figname)
plt.close(fig)
