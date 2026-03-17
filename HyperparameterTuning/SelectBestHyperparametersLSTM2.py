#!/usr/bin/env python3

# Selecting best hyperparameters from the results of the hyperparameter tuning script
# for each sequence length separately.
# LSTM2 experiment
# No plotting needed because it was all plotted with the SelectBestHyperparametersLSTM script.

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import norm

sys.path.insert(0, moduledir)
from Functions import get_config, change_boxplot_colors

# --------------------------------------------------------------------------------------
config = get_config()

dim = config['dim']
met = 'ExpVar'

tar = 'sl'
slfile = config['dirs']['data'] + config['dirs']['ext'] + config['variables'][tar] + '.csv'

datadir = config['dirs']['data'] + config['dirs']['tune']
infile = 'LSTM_tuning.nc'
infile2 = 'LSTM_hyperparameters.csv' # selected hyperparams in 1st version of the experiment
outfile = 'LSTM_hyperparameters_all.csv' # hyperparams for all stations and sequence lengths
outfile2 = 'LSTM2_hyperparameters.csv' # hyperparams for seq len that was not used in 1st version

figdir = config['dirs']['figs'] + config['dirs']['tune']
figname = 'LSTM_hyperparameters_{stat:n}.png'


dx = 0.1

col0 = ['darkgrey', 'black', 'darkgoldenrod']
col1 = ['rosybrown', 'maroon', 'darkgoldenrod']
col2 = 'teal'
# ------------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(figdir):
	os.makedirs(figdir)



# --- Load hyperparameters ---
# all hyperparameters from tuning script
hyp = xr.open_dataset(datadir + infile, engine = 'netcdf4')
stat = hyp.coords[dim['s']].values

activation = hyp.coords[dim['a']].values
seq_len = hyp.coords[dim['n']].values
units = hyp.coords[dim['u']].values
model = hyp.coords[dim['m']].values


# previously selected hyperparameters (sequence length + station names and locations)
hyper = pd.read_csv(datadir+infile2, index_col = 0)
statname = {int(s): hyper.loc[s, 'name'] for s in stat}
lon = {int(s): hyper.loc[s, 'lon'] for s in stat}
lat = {int(s): hyper.loc[s, 'lat'] for s in stat}
#seq_len2 = {int(s): int(seq_len[0]) if hyper.loc[s,dim['n']]==seq_len[1] else int(seq_len[1]) \
#	for s in stat}
seq_len1 = {int(s): hyper.loc[s,dim['n']] for s in stat}



# --- Find the best hyperparams combination for each station and sequence length ---
median = hyp[met].median(dim = dim['m'])
best = median.argmax(dim = (dim['a'], dim['u']))
best = median.isel(best)
best = best.to_dataframe()

# add station names and locations
best.insert(loc = 0, column = 'name', value = statname)
best.insert(loc = 1, column = 'lon', value = lon)
best.insert(loc = 2, column = 'lat', value = lat)

for n in seq_len:
	for s in stat:
		best.loc[(s,n),'name'] = statname[s]
		best.loc[(s,n),'lon'] = lon[s]
		best.loc[(s,n),'lat'] = lat[s]

# save the hyperparameters for both checked seq len
best.to_csv(datadir+outfile)




# --- Separate the hyperparameters for the seq len that was not in 1st experiment ---
best2 = best.copy(deep = True)
for s in stat:
	best2 = best2.drop(labels = (s,seq_len1[s]), axis = 0)
best2 = best2.reset_index(level = 1)

# save the hyperparameters for the second batch of experiments
best2.to_csv(datadir+outfile2)
