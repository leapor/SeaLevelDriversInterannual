#!/usr/bin/env python3
# Finding the best hyperparameters for LSTM networks for stations and sequence lengths not included in LSTM, LSTM2 and LSTMtz experiments.
# Tests for number of units per layer (same in each layer) and activation function
# (same in each layer) for different sequence lengths.

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
import datetime

sys.path.insert(0, moduledir)
from Functions import get_config, split_data, ensembleLSTM

# ------------------------------------------------------------------------------------
config = get_config()

dim = config['dim']
mod = 'LSTMd'

statfile = config['dirs']['data'] + config['dirs']['ext'] + config['variables']['sl']+'.csv'
statfileTZ = config['dirs']['data'] + config['dirs']['ana'] + 'TransitionZoneStationsSelected.csv'

indir = config['dirs']['data'] + config['dirs']['pro']
infile = 'data_station_{stat:n}_train.csv'

outdir = config['dirs']['data'] + config['dirs']['tune']
outfile = '{:s}_tuning.nc'.format(mod)

sequence_len = [6,7,13]  #[4, 5, 6, 7, 13]
activation = ['relu', 'sigmoid', 'tanh']
units = [10, 20, 30, 50, 100, 200, 300]

learning_rate = 0.001
loss_fun = 'mean_squared_error'
N = 10       # number of ensemble members
epochs = 10     # maximum number of epochs
val_len = 6  # length of the val set (in years)


# subset of stations (start from and end at)
sind = 1
send = 99
# ------------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(outdir):
	os.makedirs(outdir)
	

	
# --- Get the list of stations ---
stat = pd.read_csv(statfile, index_col = 0).loc[:, ['name']]
statTZ = pd.read_csv(statfileTZ, index_col=0).index.values
stat = stat.drop(labels=statTZ, axis=0)
ntot = len(stat.index)

# select a subset of stations (if execution was interrupted before)
stat = stat.iloc[sind-1:, :]
n = len(stat)
ndone = ntot-n

# further select a subset if I want to stop before the end of list
if (send == sind):
	stat = stat.iloc[0:1, :]
	n = 1
	ndone = -99

# get station id and name
statname = stat.loc[:,'name'].values
stat = list(stat.index)




# --- Prepare the output dataframe ---
if os.path.exists(outdir+outfile):
	res = xr.open_dataset(outdir+outfile, engine = 'netcdf4')
	os.remove(outdir+outfile)      # delete the old file to avoid writing errors!
	res.to_netcdf(outdir+outfile)  # re-create the file
	print('--- output file exists - %2i stations already tuned' % ndone)
else:
	coords = {dim['s'] : stat, dim['n'] : sequence_len, dim['a'] : activation, \
	dim['u'] : units, dim['m'] : range(1, N+1)}
	data_vars = {\
		'RRMSE' : ((dim['s'], dim['n'], dim['a'], dim['u'], dim['m']), \
		np.full((len(stat), len(sequence_len), len(activation), len(units), N), np.nan)), \
		'ExpVar' : ((dim['s'], dim['n'], dim['a'], dim['u'], dim['m']), \
		np.full((len(stat), len(sequence_len), len(activation), len(units), N), np.nan)), \
		'ExTime' : ((dim['s'], dim['n'], dim['a'], dim['u']), \
		np.full((len(stat), len(sequence_len), len(activation), len(units)), np.nan))}
	res = xr.Dataset(data_vars = data_vars, coords = coords)
	res.to_netcdf(outdir+outfile)
	print('--- output file created')




# --- Find best hyperparameters for each station ---
for i, s, sn in zip(range(len(stat)), stat, statname):
	print('Station %2i/%2i: %s (%i)' % (ndone+i+1, ntot, sn, s))
	station_start = time.time()

	# load data
	data = pd.read_csv(indir + infile.format(stat=s), index_col = 0, parse_dates = True)
	
	for sl in sequence_len:
		for a in activation:
			for u in units:			
				# hyperparameters
				hyperparams = {'units' : u, 'activation' : a, \
					'learning_rate' : learning_rate, 'loss_fun' : loss_fun}
					
				# fitting parameters
				fitting_params = {'valy' : val_len, 'epochs' : epochs, \
					'batch_size' : 1, 'sequence_len' : sl}
			
				# train ensemble
				start_time = time.time()
				metrics = ensembleLSTM(data, N, hyperparams, fitting_params, 	
					verbose = False)
					
				# get execution time
				res['ExTime'].loc[{dim['s']:s, dim['n']:sl, dim['a']:a, dim['u']:u}] \
					= time.time() - start_time
				
				# save metrics
				res['RRMSE'].loc[{dim['s']:s, dim['n']:sl, dim['a']:a, dim['u']:u}] =\
					metrics.loc[:, 'RRMSE']
				res['ExpVar'].loc[{dim['s']:s, dim['n']:sl, dim['a']:a, dim['u']:u}]\
					= metrics.loc[:, 'ExpVar']
				
				# screen output
				print('%s station %2i/%2i | %1i %7s %3i | %5.1f s  %6.2f %6.2f'  % \
					(time.strftime("%H:%M:%S", time.localtime()), \
					ndone+i+1, ntot, sl, a, u, (time.time() - start_time), \
					np.nanmedian(metrics.loc[:, 'RRMSE']), \
					np.nanmedian(metrics.loc[:, 'ExpVar'])))
				
				
			# intermittently save results (in case of crashes)
			res.to_netcdf(outdir+outfile, mode = 'a')
			print('------------ saved')
	print('========== tuning time for station: %6.1f min' % ((time.time()-station_start)/60))
