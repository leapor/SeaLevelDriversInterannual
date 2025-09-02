#!/usr/bin/env python3

# needs environment SLD-vis

# Predicting training set timeseries and metrics for linear regression models.

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
from Functions import get_config, split_data, relative_explained_variance, corr2

# -------------------------------------------------------------------------------------
config = get_config()

dim = config['dim']
mod = 'LinRegTimeDependentEns'

datadir = config['dirs']['data'] + config['dirs']['pro']
resdir = config['dirs']['data'] + config['dirs']['exp']
moddir = config['dirs']['mod'] + mod + '/'
valdir = config['dirs']['data'] + config['dirs']['val']

statfile = config['dirs']['data'] + config['dirs']['ext'] + 'sea_level.csv'
datafile = 'data_station_{stat:n}_train.csv'
resfile = mod + '_training_results.nc'
modfile = mod + '_trained_{stat:n}_{seq:n}_{i:n}.pkl'
fileout1 = mod + '_{seq:n}_train_timeseries_{stat:n}.csv'
fileout2 = mod + '_train_metrics.nc'
# -----------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(valdir):
	os.makedirs(valdir)


# --- Get stations, sequence length and coefficients from training results file ---
res = xr.open_dataset(resdir+resfile, engine = 'netcdf4')

stat = res.coords[dim['s']].values
statname = res['name'].values
seq_len = res.coords[dim['n']].values
ens = list(res.coords[dim['m']].values)
features = res.coords[dim['f']].values
nstat = len(stat)



# --- Create a dataframe for storing metrics ---
coords = {dim['s']:stat, dim['n']: seq_len, dim['m']:ens}
data_vars = {\
	'name' : ((dim['s']), statname), \
	'ExpVar' : ((dim['s'],dim['n'],dim['m']), np.full((nstat,len(seq_len),len(ens)), np.nan)), \
	'Corr2' : ((dim['s'],dim['n'],dim['m']), np.full((nstat,len(seq_len),len(ens)), np.nan)), \
	'ValStart' : ((dim['s'],dim['n'],dim['m']), \
	np.full((nstat,len(seq_len),len(ens)), 0, dtype=int)), \
	'ValEnd' : ((dim['s'],dim['n'],dim['m']), \
	np.full((nstat,len(seq_len),len(ens)), 0, dtype=int))}
metrics = xr.Dataset(coords=coords, data_vars=data_vars)



# --- Predict validation set ---
for s, si in zip(stat, range(nstat)):
	# load data
	data = pd.read_csv(datadir+datafile.format(stat=s), index_col = 0, parse_dates = True)
	ttot = data.index
	
	for n in seq_len:
		# create a dataframe for storing timeseries
		ts = pd.DataFrame(index = data.index, columns = ['true'] + ens)
		ts.loc[:, 'true'] = data.iloc[:,0].values
	
		# --- Prepare data
		# add previous time steps
		colname = [data.columns[0]] + \
			['{:s}-{:n}'.format(f, n) for n in range(n+1) for f in features]
		data_ext = pd.DataFrame(columns = colname, index = ttot[n:])
		data_ext.iloc[:, 0] = data.iloc[n:, 0].values
		cn = ['{:s}-{:n}'.format(f, 0) for f in features]
		data_ext.loc[:, cn] = data.iloc[n:, 1:].values
		
		for i in range(1, n+1):
			cn = ['{:s}-{:n}'.format(f, i) for f in features]
			data_ext.loc[:, cn] = data.iloc[n-i:-i, 1:].values
			
		# remove NaNs and separate features, target and time
		data_ext.dropna(axis = 0, inplace = True)
	
		X = data_ext.iloc[:, 1:].values.astype(float)
		y = data_ext.iloc[:, 0].values.astype(float)
		t = data_ext.index.date   # (this is time without NaNs)
		
		# predict validation set
		for i in ens:
			# separate validation set
			subset = [res['ValStart'].loc[{dim['s']:s,dim['n']:n,dim['m']:i}].values, \
				res['ValEnd'].loc[{dim['s']:s,dim['n']:n,dim['m']:i}].values]
			[Xtr,ytr,ttr] , _ = split_data([X, y], subset = subset, t = t)
			
			# load model
			model = joblib.load(moddir+modfile.format(stat=s, seq=n, i=i))
			
			# predict
			ypred = model.predict(Xtr)
			ts.loc[ttr, i] = ypred
			
			# calculate metrics
			metrics['ExpVar'].loc[{dim['s']:s, dim['n']:n , dim['m']:i}] = \
				relative_explained_variance(ytr, ypred)
			metrics['Corr2'].loc[{dim['s']:s, dim['n']:n, dim['m']:i}] = \
				corr2(ytr, ypred)
				
			# save validation set start and end
			metrics['ValStart'].loc[{dim['s']:s, dim['n']:n, dim['m']:i}] = \
				res['ValStart'].loc[{dim['s']:s,dim['n']:n,dim['m']:i}].values
			metrics['ValEnd'].loc[{dim['s']:s, dim['n']:n, dim['m']:i}] = \
				res['ValEnd'].loc[{dim['s']:s,dim['n']:n,dim['m']:i}].values
		
		# save the predictions
		ts.to_csv(valdir+fileout1.format(seq=n, stat=s))
		
	# -- Save the metrics (after each station)
	metrics.to_netcdf(valdir+fileout2, mode='a')
	print('-- completed station %2i/%2i %s (%i)' % (si+1, nstat, statname[si], s))
