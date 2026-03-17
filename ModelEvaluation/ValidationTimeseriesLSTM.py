#!/usr/bin/env python3

# needs evnvironment SLD-ML

# Save the validation set predictions for all stations, one file per station.
# Save a list of best models, one file for all stations.

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
from Functions import get_config, split_data, create_sequence

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ------------------------------------------------------------------------------------
config = get_config()

mod = 'LSTM'
met = 'ExpVar'

N = config['hyper']['N']       # number of ensemble members
Nd = config['hyper']['Ndrop']  # how many worst ensemble members to drop

datadir = config['dirs']['data'] + config['dirs']['pro']
hyperdir = config['dirs']['data'] + config['dirs']['tune']
resdir = config['dirs']['data'] + config['dirs']['exp']
moddir = config['dirs']['mod'] + mod + '/'
valdir = config['dirs']['data'] + config['dirs']['val']

datafile = 'data_station_{stat:n}_train.csv'
hyperfile = mod + '_hyperparameters.csv'
resfile = mod + '_validation_results_{stat:n}.csv'
modfile = mod + '_trained_model_{stat:n}_{i:02n}.pkl'
fileout1 = mod + '_best_'+str(N-Nd)+'_models.csv'
fileout2 = mod + '_val_timeseries_{stat:n}.csv'

dim = config['dim']
# ------------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(valdir):
	os.makedirs(valdir)
	
	
# --- Get list of stations and sequence length from the hyperparameters file ---
hyper = pd.read_csv(hyperdir+hyperfile, index_col = 0)
stat = list(hyper.index)
nstat = len(stat)


# --- Create a variable for storing best models for all stations ---
best = pd.DataFrame(index = stat, columns = range(N-Nd))


# --- Predict validation set ---
for s, si in zip(stat, range(nstat)):
	# load data
	data = pd.read_csv(datadir+datafile.format(stat=s), index_col = 0, parse_dates = True)
	
	# load metrics (includes the validation set time span)
	metrics = pd.read_csv(resdir+resfile.format(stat=s), index_col = 0)
	ens = list(metrics.index)
	
	# drop the worst models and save the list of best
	metrics_best = metrics.sort_values(by = met, axis = 0)
	metrics_best = metrics_best.iloc[Nd:, :]
	best.loc[s,:] = list(metrics_best.index)
	
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
	
	# predict validation set
	for i in ens:
		# separate validation set
		subset = [metrics.loc[i,'ValStart'], metrics.loc[i,'ValEnd']]
		_ , [Xval, yval, tval] = split_data([X, y], subset = subset, t = t)
		
		# load model
		model = joblib.load(moddir+modfile.format(stat=s, i=i))
		
		# predict
		ypred = np.squeeze(model.predict(Xval, verbose = 0))
		ts.loc[tval, i] = ypred
		
	# save the predictions
	ts.to_csv(valdir+fileout2.format(stat=s))
	print('-- completed station %2i/%2i %s (%i)' % (si+1, nstat, hyper.loc[s,'name'], s))
	
	
# --- Save the best models ---
best.to_csv(valdir+fileout1)
