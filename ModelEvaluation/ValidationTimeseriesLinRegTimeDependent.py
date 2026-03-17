#!/usr/bin/env python3

# needs environmentSLD-vis

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
from Functions import get_config, split_data

# -------------------------------------------------------------------------------------
config = get_config()

mod = 'LinRegTimeDependentEns'
met = 'ExpVar'

datadir = config['dirs']['data'] + config['dirs']['pro']
resdir = config['dirs']['data'] + config['dirs']['exp']
moddir = config['dirs']['mod'] + mod + '/'
valdir = config['dirs']['data'] + config['dirs']['val']

statfile = config['dirs']['data'] + config['dirs']['ext'] + 'sea_level.csv'
datafile = 'data_station_{stat:n}_train.csv'
resfile = mod + '_training_results.nc'
modfile = mod + '_trained_{stat:n}_{ns:n}_{i:n}.pkl'
fileout = mod + '_{ns:n}_val_timeseries_{stat:n}.csv'

dim = config['dim']
N = config['hyper']['N']       # number of ensemble members
nmax = 12 # maximum sequence length
# ------------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(valdir):
	os.makedirs(valdir)


# --- Get the training results ---
res = xr.open_dataset(resdir+resfile, engine = 'netcdf4')
stat = res.coords[dim['s']].values
statname = {s : str(res['name'].loc[{dim['s'] : s}].values) for s in stat}
ens = list(res.coords[dim['m']].values)
features = res.coords[dim['f']].values
nstat = len(stat)


# --- Predict validation set ---
for s, si in zip(stat, range(nstat)):
	# load data
	data = pd.read_csv(datadir+datafile.format(stat=s), index_col = 0, parse_dates = True)
	ttot = data.index
	
	for ns in range(nmax+1):	
		# create a dataframe for storing results
		ts = pd.DataFrame(index = data.index, columns = ['true'] + ens)
		ts.loc[:, 'true'] = data.iloc[:,0].values
	
		# --- Prepare data
		# add previous time steps
		colname = [data.columns[0]] + \
			['{:s}-{:n}'.format(f, n) for n in range(ns+1) for f in features]
		data_ext = pd.DataFrame(columns = colname, index = ttot[ns:])
		data_ext.iloc[:, 0] = data.iloc[ns:, 0].values
		cn = ['{:s}-{:n}'.format(f, 0) for f in features]
		data_ext.loc[:, cn] = data.iloc[ns:, 1:].values
		
		for i in range(1, ns+1):
			cn = ['{:s}-{:n}'.format(f, i) for f in features]
			data_ext.loc[:, cn] = data.iloc[ns-i:-i, 1:].values
			
		# remove NaNs and separate features, target and time
		data_ext.dropna(axis = 0, inplace = True)
	
		X = data_ext.iloc[:, 1:].values
		y = data_ext.iloc[:, 0].values
		t = data_ext.index.date   # (this is time without NaNs)
		
		# predict validation set
		for i in ens:
			# separate validation set
			subset = [res['ValStart'].loc[{dim['s']:s,dim['n']:ns,dim['m']:i}].values, \
				res['ValEnd'].loc[{dim['s']:s,dim['n']:ns,dim['m']:i}].values]
			_ , [Xval, yval, tval] = split_data([X, y], subset = subset, t = t)
			
			# load model
			model = joblib.load(moddir+modfile.format(stat=s, ns=ns, i=i))
			
			# predict
			ypred = model.predict(Xval)
			ts.loc[tval, i] = ypred
		
		# save the predictions
		ts.to_csv(valdir+fileout.format(ns = ns, stat=s))
	print('-- completed station %2i/%2i %s (%i)' % (si+1, nstat, statname[s], s))
