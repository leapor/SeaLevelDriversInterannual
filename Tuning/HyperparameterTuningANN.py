#!/usr/bin/env python3
# Finding the best hyperparameters for the feed-forward neural networks.
# Tests for number of layers, number of units per layer (same in each layer) and
# activation function (same in each layer).

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
import datetime
import netCDF4

sys.path.insert(0, moduledir)
from Functions import get_config, split_data, ensembleANN

# --------------------------------------------------------------------------------------
config = get_config()

tar = 'sl'
slfile = config['dirs']['data'] + config['dirs']['ext'] + config['variables'][tar] + '.csv'

indir = config['dirs']['data'] + config['dirs']['pro']
infile = 'data_station_{stat:n}_train.csv'

outdir = config['dirs']['data'] + config['dirs']['tune']
outfile = 'ANN_tuning.nc'

activation = ['relu', 'sigmoid', 'tanh']
layers = [1, 2, 3, 4, 5, 6]
units = [5, 10, 20, 30, 50, 100]

learning_rate = 0.001
loss_fun = 'mean_squared_error'
N = 10       # number of ensemble members
epochs = 20     # maximum number of epochs
val_len = 6  # length of the val set (in years)

# short names of the dimensions
dim = {'s': 'stations', 'a':'activation', 'l':'layers', 'u': 'units', 'm': 'model_run'}

# subset of stations
sind = 43
# --------------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(outdir):
	os.makedirs(outdir)
	
# --- Get the list of stations ---
stat = pd.read_csv(slfile, index_col = 0)
ntot = stat.shape[0]

# select a subset of stations
stat = stat.iloc[sind-1:, :]
n = len(stat)
ndone = ntot-n

# get station name and id
statname = stat.loc[:, 'name'].values
stat = list(stat.index)


# --- Fitting parameters ---
# (always the same)
fitting_params = {'valy' : val_len, 'epochs' : epochs, 'batch_size' : 1}


# --- Prepare the output dataframe ---
if os.path.exists(outdir+outfile):
	res = xr.open_dataset(outdir+outfile, engine = 'netcdf4')
	os.remove(outdir+outfile)      # delete the old file to avoid writing errors!
	res.to_netcdf(outdir+outfile, mode='a')
	print('--- output file exists - %2i stations already tuned' % ndone)
else:
	coords = {dim['s'] : stat, dim['a'] : activation, dim['l'] : layers, dim['u'] : units, \
		dim['m'] : range(1, N+1)}
	data_vars = {\
		'RMSE' : ((dim['s'], dim['a'], dim['l'], dim['u'], dim['m']), \
		np.empty((len(stat), len(activation), len(layers), len(units), N))), \
		'ExpVar' : ((dim['s'], dim['a'], dim['l'], dim['u'], dim['m']), \
		np.empty((len(stat), len(activation), len(layers), len(units), N))), \
		'ExTime' : ((dim['s'], dim['a'], dim['l'], dim['u']), \
		np.empty((len(stat), len(activation), len(layers), len(units))))}
	res = xr.Dataset(data_vars = data_vars, coords = coords)
	res.to_netcdf(outdir+outfile)
	print('--- output file created')



# --- Find best hyperparameters for each station ---
for i, s, sn in zip(range(len(stat)), stat, statname):
	print('Station %2i/%2i: %s (%i)' % (ndone+i+1, ntot, sn, s))
	station_start = time.time()

	# load data
	data = pd.read_csv(indir + infile.format(stat=s), index_col = 0, parse_dates = True)
	
	for a in activation:
		for l in layers:
			for u in units:			
				# hyperparameters
				hyperparams = {'layers' : l, 'units' : u, 'activation' : a, \
					'learning_rate' : learning_rate, 'loss_fun' : loss_fun}
			
				# train ensemble
				start_time = time.time()
				metrics = ensembleANN(data, N, hyperparams, fitting_params, 	
					verbose = False)
					
				# get execution time
				res['ExTime'].loc[{dim['s']:s, dim['a']:a, dim['l']:l, dim['u']:u}] =\
					time.time() - start_time
				
				# save metrics
				res['RMSE'].loc[{dim['s']:s, dim['a']:a, dim['l']:l, dim['u']:u}] =\
					metrics.loc[:, 'RMSE']
				res['ExpVar'].loc[{dim['s']:s, dim['a']:a, dim['l']:l, dim['u']:u}] =\
					metrics.loc[:, 'ExpVar']
				
				# screen output
				print('%s station %2i/%2i | %7s %1i %3i | %5.1f s  %6.2f %6.2f'  % \
					(time.strftime("%H:%M:%S", time.localtime()), \
					ndone+i+1, ntot, a, l, u, (time.time() - start_time), \
					np.nanmedian(metrics.loc[:, 'RMSE']), \
					np.nanmedian(metrics.loc[:, 'ExpVar'])))
				
				
		# intermittently save results (in case of crashes)
		res.to_netcdf(outdir+outfile, mode = 'a')
		print('------------ saved')
	print('========== tuning time for station: %6.1f min' % ((time.time()-station_start)/60))
