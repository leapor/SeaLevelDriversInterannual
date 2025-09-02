#!/usr/bin/env python3

# needs environment SLD-ML

# Save the validation set predictions for all stations, one file per station.

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
from Functions import get_config, create_sequence, split_data

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ------------------------------------------------------------------------------------
config = get_config()

mod = 'LSTMtzd'
met = 'ExpVar'

hyperdir = config['dirs']['data'] + config['dirs']['tune']
datadir = config['dirs']['data'] + config['dirs']['pro']
resdir = config['dirs']['data'] + config['dirs']['exp']
moddir = config['dirs']['mod'] + mod + '/'
valdir = config['dirs']['data'] + config['dirs']['val']

statfile = config['dirs']['data'] + config['dirs']['ext'] + 'sea_level.csv'
hyperfile = '{mod:s}_hyperparameters.csv'
datafile = 'data_station_{stat:n}_train.csv'
resfile = mod + '_validation_results_{stat:n}_{seq:n}.csv'
modfile = mod + '_trained_model_{stat:n}_{seq:n}_{i:02n}.pkl'
fileout = mod + '_{seq:n}_val_timeseries_{stat:n}.csv'
fileout2 = mod + '_validation_metrics.nc'

dim = config['dim']
# ------------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(valdir):
	os.makedirs(valdir)

# --- Get stations and sequence length from hyperparameters file ---
hyper = pd.read_csv(hyperdir+hyperfile.format(mod=mod), index_col = (0,1))
stat = list(dict.fromkeys(hyper.index.get_level_values(0)))
seq_len = list(dict.fromkeys(hyper.index.get_level_values(1)))
statname = list(dict.fromkeys(hyper.loc[:, 'name'].values))

# --- Predict validation set ---
for s, si in zip(stat, range(len(stat))):
	# load data
	data = pd.read_csv(datadir+datafile.format(stat=s), index_col = 0, parse_dates = True)
	ttot = data.index
	
	for n in seq_len:
		# load validation set start and end year
		res = pd.read_csv(resdir+resfile.format(stat=s, seq=n), index_col = 0)
		ens = list(res.index.values)
	
		# create a dataframe for storing results
		ts = pd.DataFrame(index = data.index, columns = ['true'] + ens)
		ts.loc[:, 'true'] = data.iloc[:,0].values
	
		# --- Prepare data
		# prepare data
		X = data.iloc[:, 1:].values
		y = data.iloc[:, 0].values
		t = data.index.date
		
		X, y, t = create_sequence(X, y, n, t = t)
		
		ind = ~np.isnan(y)
		X = X[ind]
		y = y[ind]
		t = t[ind]
		
		# predict validation set
		for i in ens:
			# separate validation set
			subset = [res.loc[i,'ValStart'], res.loc[i,'ValEnd']]
			_ , [Xval, yval, tval] = split_data([X, y], subset = subset, t = t)
			
			# load model
			model = joblib.load(moddir+modfile.format(stat=s, seq=n, i=i))
			
			# predict
			ypred = model.predict(Xval, verbose = 0)
			ts.loc[tval, i] = np.squeeze(ypred)
		
		# save the predictions
		ts.to_csv(valdir+fileout.format(seq = n, stat = s))
		print('    %4i %2i' % (s, n))
	
	print('-- completed station %2i/%2i %s (%i)' % ((si+1), len(stat), statname[si], s))
	
	
