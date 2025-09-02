#!/usr/bin/env python3

# needs environmentSLD-ML

# Predicting training set timeseries and metrics for LSTM2 models.
# NOTE: There is a conflict between tensorflow/keras and netcdf, which causes a crash if 
# the dataset is saved after loading models with joblib, but it works if the (empty) 
# metrics dataset is saved first and then the actual results are appended into it.

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import datetime

sys.path.insert(0, moduledir)
from Functions import get_config, split_data, create_sequence, relative_explained_variance, corr2

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ------------------------------------------------------------------------------------
config = get_config()

dim = config['dim']
mod = 'LSTM2'

datadir = config['dirs']['data'] + config['dirs']['pro']
hyperdir = config['dirs']['data'] + config['dirs']['tune']
resdir = config['dirs']['data'] + config['dirs']['exp']
moddir = config['dirs']['mod'] + mod + '/'
valdir = config['dirs']['data'] + config['dirs']['val']

datafile = 'data_station_{stat:n}_train.csv'
hyperfile = mod + '_hyperparameters.csv'
resfile = mod + '_validation_results_{stat:n}.csv'
modfile = mod + '_trained_model_{stat:n}_{i:02n}.pkl'
fileout1 = mod + '_train_timeseries_{stat:n}.csv'
fileout2 = mod + '_train_metrics.nc'
# -------------------------------------------------------------------------------------

	
# --- Get hyperparameters ---
hyper = pd.read_csv(hyperdir+hyperfile, index_col = 0)
stat = hyper.index.values
statname = hyper.loc[:, 'name'].values
lon = hyper.loc[:,'lon'].values
lat = hyper.loc[:,'lat'].values
nstat = len(stat)




# --- Predict validation set ---
for s, si in zip(stat, range(nstat)):
	# load data
	data = pd.read_csv(datadir+datafile.format(stat=s), index_col = 0, parse_dates = True)
	
	# load validation metrics to get the validation set time span
	valmet = pd.read_csv(resdir+resfile.format(stat=s), index_col = 0)
	ens = list(valmet.index)
	
	# create a dataframe for storing metrics
	if si==0:
		coords = {dim['s']:stat, dim['m']:ens}
		data_vars = {\
			'name' : ((dim['s']), statname), \
			'lon' : ((dim['s']), lon), \
			'lat' : ((dim['s']), lat), \
			'ExpVar' : ((dim['s'], dim['m']), np.full((nstat, len(ens)), np.nan)), \
			'Corr2' : ((dim['s'], dim['m']), np.full((nstat, len(ens)), np.nan)), \
			'ValStart' : ((dim['s'],dim['m']), np.full((nstat,len(ens)), 0, dtype=int)),\
			'ValEnd' : ((dim['s'],dim['m']), np.full((nstat,len(ens)), 0, dtype=int))}
		metrics = xr.Dataset(coords=coords, data_vars=data_vars)
		metrics.to_netcdf(valdir+fileout2)
	
	# create a dataframe for storing results
	ts = pd.DataFrame(index = data.index, columns = ['true'] + ens)
	ts.loc[:, 'true'] = data.iloc[:,0].values
	
	# prepare data
	X = data.iloc[:, 1:].values
	y = data.iloc[:, 0].values
	t = data.index.date
	
	X, y, t = create_sequence(X, y, hyper.loc[s,dim['n']], t = t)
	
	ind = ~np.isnan(y)
	X = X[ind]
	y = y[ind]
	t = t[ind]
	
	# predict training set
	for i in ens:
		# separate validation set
		subset = [valmet.loc[i,'ValStart'], valmet.loc[i,'ValEnd']]
		[Xtr, ytr, ttr] , _ = split_data([X, y], subset = subset, t = t)
		
		# load model
		model = joblib.load(moddir+modfile.format(stat=s, i=i))
		
		# predict
		ypred = np.squeeze(model.predict(Xtr, verbose = 0))
		ts.loc[ttr, i] = ypred
		
		# calculate metrics
		metrics['ExpVar'].loc[{dim['s']:s, dim['m']:i}] = \
			relative_explained_variance(ytr, ypred)
		metrics['Corr2'].loc[{dim['s']:s, dim['m']:i}] = \
			corr2(ytr, ypred)
		
		# save validation set start and end
		metrics['ValStart'].loc[{dim['s']:s, dim['m']:i}] = valmet.loc[i,'ValStart']
		metrics['ValEnd'].loc[{dim['s']:s, dim['m']:i}] = valmet.loc[i,'ValEnd']
		
	# save the predictions
	ts.to_csv(valdir+fileout1.format(stat=s))
	print('-- completed station %2i/%2i %s (%i)' % (si+1, nstat, hyper.loc[s,'name'], s))
	
	
# --- Save the metrics ---
metrics.to_netcdf(valdir+fileout2, mode='a')
