#!/usr/bin/env python3

# needs environment SLD-ML

# Predict test set with an ensemble of trained ANNs and do permutation feature
# importance.

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

sys.path.insert(0, moduledir)
from Functions import get_config, split_data, RRMSE, relative_explained_variance

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# -------------------------------------------------------------------------------------
config = get_config()

mod = 'ANN'
met = ['ExpVar', 'RRMSE']

N = config['hyper']['N']       # number of ensemble members

datadir = config['dirs']['data'] + config['dirs']['pro']
hyperdir = config['dirs']['data'] + config['dirs']['tune']
moddir = config['dirs']['mod'] + mod + '/'
outdir = config['dirs']['data'] + config['dirs']['fi']

datafile = 'data_station_{stat:n}_test.csv'
hyperfile = mod + '_hyperparameters.csv'
modfile = mod + '_trained_model_{stat:n}_{i:02n}.pkl'
outfile1 = mod + '_feature_importance_timeseries.nc'
outfile2 = mod + '_feature_importance_metrics.nc'

dim = config['dim']
# -------------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(outdir):
	os.makedirs(outdir)
	
	
# --- Get hyperparameters ---
hyper = pd.read_csv(hyperdir+hyperfile, index_col = 0)
stat = list(hyper.index)
nstat = len(stat)


# --- Everything per station ---
for s, si in zip(stat, range(nstat)):
	start_time = time.time()

	# load data
	data = pd.read_csv(datadir+datafile.format(stat=s), index_col = 0, parse_dates = True)
	
	
	# create xarray datasets for storing results (4D)
	if (si == 0):
		# extract coordinate info from the data file
		features = list(data.columns)
		features[0] = 'all'
		t = data.index.to_numpy()
		
		# create the time series dataset
		coords = {dim['s']: stat, dim['t']: t, dim['m']: range(N), dim['f']: features}
		data_vars = {'true' : ((dim['s'], dim['t']), np.full((len(stat), len(t)), np.nan)), \
			'pred' : ((dim['s'], dim['t'], dim['m'], dim['f']), \
			np.full((len(stat), len(t), N, len(features)), np.nan))}
		res = xr.Dataset(coords = coords, data_vars = data_vars)
		res.to_netcdf(outdir+outfile1)
		
		# create the metrics dataset
		coords = {dim['s']: stat, dim['m']: range(N), dim['f']: features}
		data_vars = {met[0]: ((dim['s'],dim['m'],dim['f']), \
			np.empty((len(stat),N,len(features)))), \
			met[1]: ((dim['s'],dim['m'],dim['f']), \
			np.empty((len(stat),N,len(features))))}
		metrics = xr.Dataset(coords = coords, data_vars = data_vars)
		metrics.to_netcdf(outdir+outfile2)
	
	# prepare data
	data.dropna(axis = 0, inplace = True)
	
	X = data.iloc[:,1:].values
	t = data.index.to_numpy()
	ytrue = data.iloc[:,0].values
	res['true'].loc[{dim['s']:s, dim['t']: t}] = ytrue   # save the true value
	
	# predict test set
	for i in range(N):		
		# load model
		model = joblib.load(moddir+modfile.format(stat=s, i=i))
		
		# predict
		ypred = np.squeeze(model.predict(X, verbose = 0))
		res['pred'].loc[{dim['s']:s, dim['t']:t, dim['m']:i, dim['f']:features[0]}] = ypred
		
		# calculate metrics for the baseline prediction
		metrics[met[0]].loc[{dim['s']:s, dim['m']:i, dim['f']:'all'}] = \
			relative_explained_variance(ytrue, ypred)
		metrics[met[1]].loc[{dim['s']:s, dim['m']:i, dim['f']:'all'}] = \
			RRMSE(ytrue, ypred)
		
		# permutation feature importance
		for f, fi in zip(features[1:], range(len(features)-1)):
			# permute feature
			Xper = np.copy(X)
			Xper[:,fi] = np.random.permutation(Xper[:,fi])
			
			# predict
			ypred = np.squeeze(model.predict(Xper, verbose = 0))
			res['pred'].loc[{dim['s']:s, dim['t']:t, dim['m']:i, dim['f']:f}] = ypred
			
			# calculate metrics for the feature importance predictions
			metrics[met[0]].loc[{dim['s']:s, dim['m']:i, dim['f']:f}] = \
				relative_explained_variance(ytrue, ypred)
			metrics[met[1]].loc[{dim['s']:s, dim['m']:i, dim['f']:f}] = \
				RRMSE(ytrue, ypred)
			
		# intermittently save the predictions
		res.to_netcdf(outdir+outfile1, mode = 'a')
		metrics.to_netcdf(outdir+outfile2, mode = 'a')
		print('%s station %2i/%2i model %3i/%3i | %6.2f%% %6.2f%%'  % \
			(time.strftime("%H:%M:%S", time.localtime()), si+1, nstat, i+1, N, \
			metrics[met[0]].loc[{dim['s']:s, dim['m']:i, dim['f']:'all'}], \
			metrics[met[1]].loc[{dim['s']:s, dim['m']:i, dim['f']:'all'}]))
	
	print('-- completed station %2i/%2i %s (%i) -> time needed: %6.1f min' % \
		(si+1, nstat, hyper.loc[s,'name'], s, (time.time()-start_time)/60))
