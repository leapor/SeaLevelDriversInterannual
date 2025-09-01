#!/usr/bin/env python3

# Getting the Antarctica dataset and creating one time series from it.
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
from Functions import get_config

# -------------------------------------------------------------------------------------------------
config = get_config()
var = 'anta'
vname = config['variables'][var]
y1, y2 = config['time']['start'], config['time']['end']

filein = config['dirs']['orig'] + config['data_original'][var]

outdir = config['dirs']['data'] + config['dirs']['ext']
fileout = outdir + vname+'.csv'

figdir = config['dirs']['figs'] + config['dirs']['ext']
figname = figdir + vname+'_Timeseries.png'

fvar = 'mrros'
# -------------------------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(outdir):
	os.makedirs(outdir)
if not os.path.exists(figdir):
	os.makedirs(figdir)


# --- Create a time series ---
t = [datetime.date(y, m, 1) for y in range(y1, y2+1) for m in range(1, 13)]


# --- Get data ---
ny = y2-y1+1
x = np.zeros((ny, 12))
for y, i in zip(range(y1, y2+1), range(ny)):
	data = xr.open_dataset(filein.replace('[yyyy]', str(y)))
	temp = np.squeeze(data[fvar].values)
	x[i, :] = np.nansum(temp, axis = (1,2))

# --- Reorganize and save the data ---
x = np.reshape(x, -1)
data = pd.DataFrame(data = x, index = t, columns = [vname])
data.to_csv(fileout)


# --- Plotting ---
fig, ax = plt.subplots(1, 1, figsize = [14, 4])
ax.plot(t, x)
ax.set_xlim([t[0], t[-1]])
ax.tick_params(axis = 'both', labelsize = 'x-large')
ax.set_ylabel('Antarctica runoff (mm weq)', fontsize = 'xx-large')
	
fig.savefig(figname)
plt.close(fig)
