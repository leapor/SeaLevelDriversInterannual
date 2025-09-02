#!/usr/bin/env python3

# needs environment SLD-ML

# Predict test set with an ensemble of trained LSTMs and do permutation feature 
# importance.
# For LSTMs in all stations outside of the LSTMtzd experiment.
# (39 stations, sequence length 4,5,6,7,13)

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import time
import datetime
#import keras.backend as K

sys.path.insert(0, moduledir)
from Functions import get_config, split_data, create_sequence, relative_explained_variance, corr2, \
	RRMSE

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# -----------------------------------------------------------------------------------
config = get_config()

mod = 'LSTMd'
met = ['ExpVar', 'Corr2', 'RRMSE']

dim = config['dim']
N = config['hyper']['N']       # number of ensemble members

datadir = config['dirs']['data'] + config['dirs']['pro']
hyperdir = config['dirs']['data'] + config['dirs']['tune']
moddir = config['dirs']['mod'] + mod + '/'
outdir = config['dirs']['data'] + config['dirs']['fi']

datafile = 'data_station_{stat:n}_test.csv'
hyperfile = mod + '_hyperparameters.csv'
modfile = mod + '_trained_model_{stat:n}_{seq:n}_{i:02n}.pkl'
outfile1 = mod + '_feature_importance_timeseries.nc'
outfile2 = mod + '_feature_importance_metrics.nc'

s0 = 9     # first station in the whole LSTMd list
s1 = 682 # station from which to start
s2 = 2106 # station with which to end
# -----------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(outdir):
	os.makedirs(outdir)
	
	
# --- Get hyperparameters ---
hyper = pd.read_csv(hyperdir+hyperfile) #, index_col = (0,1))
stat0 = list(dict.fromkeys(hyper.loc[:,dim['s']].values))

hyper = hyper.loc[(hyper[dim['s']]>=s1) & (hyper[dim['s']]<=s2), :]

stat = list(dict.fromkeys(hyper.loc[:, dim['s']]))
seq_len = list(dict.fromkeys(hyper.loc[:, dim['n']]))
nstat = len(stat)

hyper.set_index(keys = [dim['s'],dim['n']], inplace=True)
statname = hyper.loc[(stat, seq_len[0]), 'name'].values




# --- Everything per station ---
for s, si in zip(stat, range(nstat)):
	start_time = time.time()

	# load data
	data = pd.read_csv(datadir+datafile.format(stat=s), index_col = 0, parse_dates = True)
	t0 = data.index.to_numpy()
	
	# datasets for storing results
	if (si == 0) & (s0==s1):
		# if it's the first run of the script create new dataframes for storing results
		# extract coordinate info from the data file
		features = list(data.columns)
		features[0] = 'all'
		
		# create the time series dataset
		coords = {dim['s']: stat0, dim['n']: seq_len, dim['t']: t0, dim['m']: range(N), \
			dim['f']: features}
		data_vars = {'true': ((dim['s'],dim['t']), np.full((len(stat0), len(t0)), np.nan)), \
			'pred': ((dim['s'], dim['n'], dim['t'], dim['m'], dim['f']), \
			np.full((len(stat0), len(seq_len), len(t0), N, len(features)), np.nan))}
		res = xr.Dataset(coords = coords, data_vars = data_vars)
		res.to_netcdf(outdir+outfile1)
		
		# create the metrics dataset
		coords = {dim['s']: stat0, dim['n']: seq_len, dim['m']: range(N), dim['f']: features}
		data_vars = {met[0]: ((dim['s'], dim['n'], dim['m'], dim['f']), \
			np.full((len(stat0), len(seq_len), N, len(features)), np.nan)), \
			met[1]: ((dim['s'], dim['n'], dim['m'], dim['f']), \
			np.full((len(stat0), len(seq_len), N, len(features)), np.nan)), \
			met[2]: ((dim['s'], dim['n'], dim['m'], dim['f']), \
			np.full((len(stat0), len(seq_len), N, len(features)), np.nan))}
		metrics = xr.Dataset(coords = coords, data_vars = data_vars)
		metrics.to_netcdf(outdir+outfile2)
		
		print('New files created!')
	elif (si==0) & (s1>s0):
		# if some results are already stored, save new results into the existing files
		res = xr.open_dataset(outdir+outfile1, engine='netcdf4')
		metrics = xr.open_dataset(outdir+outfile2, engine='netcdf4')
		res.close()
		metrics.close()
		
		features = list(data.columns)
		features[0] = 'all'
		
		print('Files exist, append new results!')
	
	# save the true values
	ytrue = data.iloc[:, 0].values
	res['true'].loc[{dim['s']:s, dim['t']: t0}] = ytrue
	
	
	# going through all sequence lengths
	for n in seq_len:
		# prepare data
		X = data.iloc[:, 1:].values
		y = data.iloc[:, 0].values
		t = data.index.date
	
		X, y, t = create_sequence(X, y, n, t = t)
	
		ind = ~np.isnan(y)
		X = X[ind]
		t = t[ind]
	
		# predict test set
		for i in range(N):		
			# load model
			model = joblib.load(moddir+modfile.format(stat=s, seq=n, i=i))
			#print('\nLoaded model:')
			#print(model)
		
			# predict baseline + calculate baseline metrics
			ypred = np.squeeze(model.predict(X, verbose = 0))
			#print('Model after predicting baseline:')
			#print(model)
			res['pred'].loc[{dim['s']:s, dim['n']:n, dim['t']:t, dim['m']:i, \
				dim['f']:features[0]}] = ypred
			ypred = res['pred'].loc[{dim['s']:s, dim['n']:n, dim['m']:i, \
				dim['f']:features[0]}]
			metrics['ExpVar'].loc[{dim['s']:s, dim['n']:n, dim['m']:i, \
				dim['f']:features[0]}] = relative_explained_variance(ytrue, ypred)
			metrics['Corr2'].loc[{dim['s']:s, dim['n']:n, dim['m']:i, \
				dim['f']:features[0]}] = corr2(ytrue, ypred)
			metrics['RRMSE'].loc[{dim['s']:s, dim['n']:n, dim['m']:i, \
				dim['f']:features[0]}] = RRMSE(ytrue, ypred)
		
			# permutation feature importance
			for f, fi in zip(features[1:], range(len(features)-1)):
				# permute feature
				Xper = np.copy(X)
				Xper[:,:,fi] = np.random.permutation(Xper[:,:,fi])
				
				# predict
				ypred = np.squeeze(model.predict(Xper, verbose = 0))
				#print('Model after predicting permuted feature:')
				#print(model)
				res['pred'].loc[{dim['s']:s, dim['n']:n, dim['t']:t, dim['m']:i, \
					dim['f']:f}] = ypred
				ypred = res['pred'].loc[{dim['s']:s, dim['n']:n, dim['m']:i, \
					dim['f']:f}]
					
				# calculate metrics
				metrics['ExpVar'].loc[{dim['s']:s, dim['n']:n, dim['m']:i, \
					dim['f']:f}] = relative_explained_variance(ytrue, ypred)
				metrics['Corr2'].loc[{dim['s']:s, dim['n']:n, dim['m']:i, \
					dim['f']:f}] = corr2(ytrue, ypred)
				metrics['RRMSE'].loc[{dim['s']:s, dim['n']:n, dim['m']:i, \
					dim['f']:f}] = RRMSE(ytrue, ypred)
					
				del Xper
				del ypred
				#K.clear_session()
			#del model
					
		print('%s %2i/%2i  %4i %2i' % ((time.strftime("%H:%M:%S", time.localtime())), \
		si+1, nstat, s, n))
		
	# save the results
	res.to_netcdf(outdir+outfile1, mode = 'a')
	metrics.to_netcdf(outdir+outfile2, mode = 'a')
	
	print('-- completed station %2i/%2i %s (%i) -> time needed: %6.1f min' \
		% (si+1, nstat, statname[si], s, (time.time()-start_time)/60))		
