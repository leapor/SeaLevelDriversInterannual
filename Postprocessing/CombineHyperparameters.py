#!/usr/bin/env python3

# needs environmentSLD-vis

# Combine hyperparameters from all neural network models.
# It drops the worse models for stations and sequence lengths for which two models exist
# (transition zone, sequence length 1&2).
# ANN models have 3 tuned hyperparameters (number of layers, activation function and
# number of units per layer); they are stored under seq_len = 0.
# LSTM models have 2 tuned hyperparameters (activation function and number of units per
# layer); They are stored under seq_len = [1,2,3,4,5,6,12]

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, moduledir)
from Functions import get_config

# ---------------------------------------------------------------------------------------
config = get_config()

dim = config['dim']
mod = ['ANN', 'LSTM']

hyperdir = config['dirs']['data'] + config['dirs']['tune']
resdir = config['dirs']['data'] + config['dirs']['ana']

hyperfile = '{mod:s}_hyperparameters.csv'
TZfile = 'TransitionZoneBetterModel.nc'
outfile = 'Hyperparameters.nc'
# ---------------------------------------------------------------------------------------

# --- Load the transizion zone better model info ---
best = xr.open_dataset(resdir+TZfile, engine='netcdf4')['best']



# --- Load all hyperparameter files ---
# ANN
ann = pd.read_csv(hyperdir+hyperfile.format(mod='ANN'), index_col=0)

# LSTM & LSTM2
lstm = pd.read_csv(hyperdir+hyperfile.format(mod='LSTM'), index_col=0)
lstm2 = pd.read_csv(hyperdir+hyperfile.format(mod='LSTM2'), index_col=0)

# LSTMtzd
lstmtzd = pd.read_csv(hyperdir+hyperfile.format(mod='LSTMtzd'), index_col=(0,1))

# LSTMd
lstmd = pd.read_csv(hyperdir+hyperfile.format(mod='LSTMd'), index_col=(0,1))




# --- Get coordinates and station names and locations ---
stat = ann.index.values
statTZ = lstmtzd.index.levels[0].values
statD = lstmd.index.levels[0].values

seq_lenM = np.array([1,2])
seq_lenD = lstmd.index.levels[1].values-1
seq_len = np.concat([np.array([0]), seq_lenM, seq_lenD])

statname = ann.loc[:,'name'].values
lon = ann.loc[:,'lon'].values
lat = ann.loc[:,'lat'].values


# --- Combine hyperparameters into one dataset ---
# create the dataset
coords = {dim['s']:stat, dim['n']:seq_len}
data_vars = {\
	'name' : ((dim['s']), statname), \
	'lon' : ((dim['s']), lon), \
	'lat' : ((dim['s']), lat), \
	dim['l'] : ((dim['s']), np.full((len(stat)), 0, dtype=int)), \
	dim['a'] : ((dim['s'],dim['n']), np.full((len(stat),len(seq_len)),'',dtype=object)),\
	dim['u'] : ((dim['s'], dim['n']), np.full((len(stat), len(seq_len)), 0, dtype=int))}
hyper = xr.Dataset(coords = coords, data_vars = data_vars)

# store ANN hyperparameters
hyper[dim['l']].loc[{dim['s']:stat}] = ann.loc[:,dim['l']].values
hyper[dim['a']].loc[{dim['s']:stat, dim['n']:0}] = ann.loc[:,dim['a']].values
hyper[dim['u']].loc[{dim['s']:stat, dim['n']:0}] = ann.loc[:,dim['u']].values

# store LSTM and LSTM2 hyperparameters
for s in stat:
	# LSTM
	hyper[dim['a']].loc[{dim['s']:s, dim['n']:lstm.loc[s,dim['n']]-1}] = lstm.loc[s,dim['a']]
	hyper[dim['u']].loc[{dim['s']:s, dim['n']:lstm.loc[s,dim['n']]-1}] = lstm.loc[s,dim['u']]
	
	# LSTM2
	hyper[dim['a']].loc[{dim['s']:s, dim['n']:lstm2.loc[s,dim['n']]-1}] = lstm2.loc[s,dim['a']]
	hyper[dim['u']].loc[{dim['s']:s, dim['n']:lstm2.loc[s,dim['n']]-1}] = lstm2.loc[s,dim['u']]

# store LSTMd hyperparameters
for s in statD:
	for n in seq_lenD:
		hyper[dim['a']].loc[{dim['s']:s, dim['n']:n}] = lstmd.loc[(s,n+1), dim['a']]
		hyper[dim['u']].loc[{dim['s']:s, dim['n']:n}] = lstmd.loc[(s,n+1), dim['u']]
		
# store LSTMtzd high seq len hyperparameters (without 1&2)
for s in statTZ:
	for n in seq_lenD:
		hyper[dim['a']].loc[{dim['s']:s, dim['n']:n}] = lstmtzd.loc[(s,n+1), dim['a']]
		hyper[dim['u']].loc[{dim['s']:s, dim['n']:n}] = lstmtzd.loc[(s,n+1), dim['u']]
		
# replace hyperparameters for seq len 1&2 in the transition zone with LSTMtz if they are better
for s in statTZ:
	for n in seq_lenM:
		if best.loc[{dim['s']:s, dim['n']:n}].values:
			hyper[dim['a']].loc[{dim['s']:s, dim['n']:n}] = lstmtzd.loc[(s,n+1),dim['a']]
			hyper[dim['u']].loc[{dim['s']:s, dim['n']:n}] = lstmtzd.loc[(s,n+1),dim['u']]



# --- Save the combined hyperparameters into file ---
hyper.to_netcdf(resdir+outfile)
