#!/usr/bin/env python3

# NAO index - selecting the required years and saving data in a csv format.
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
var = 'nao'
vname = config['variables'][var]
y1, y2 = config['time']['start'], config['time']['end']

filein = config['dirs']['orig'] + config['data_original'][var]

outdir = config['dirs']['data'] + config['dirs']['ext']
fileout = outdir + vname +'.csv'

figdir = config['dirs']['figs'] + config['dirs']['ext']
figname = figdir + vname + '.png'
# -------------------------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(outdir):
	os.makedirs(outdir)
if not os.path.exists(figdir):
	os.makedirs(figdir)


# --- Read data ---
data = pd.read_csv(filein, sep=r'\s+', names = ['year', 'month', vname])


# --- Prepare data ---
# create timestamps
t = np.empty(len(data), dtype = datetime.date)
for i in range(len(data)):
	t[i] = datetime.date(data.loc[i, 'year'], data.loc[i, 'month'], 1)

# replace index with timespan
data.set_index(t, inplace = True)

# select time span
ind = (data.loc[:, 'year'] >= y1) & (data.loc[:, 'year'] <= y2)
data = data.loc[ind, :]
t = t[ind]

# remove unnecessary columns (year and month)
data.drop(labels = ['year', 'month'], axis = 1, inplace = True)

# save the data
data.to_csv(fileout)


# --- Plotting ---
fig, ax = plt.subplots(1, 1, figsize = [14, 4])
ax.plot(t, data.loc[:, vname].values)
ax.set_xlim([t[0], t[-1]])
ax.tick_params(axis = 'both', labelsize = 'x-large')
ax.set_ylabel('NAO index', fontsize = 'xx-large')
	
fig.savefig(figname)
plt.close(fig)
