#!/usr/bin/env python3

# needs environment SLD-ML

# Experiments: ANN, LSTM, LSTM2, LinReg

# Predict the test set for all trained models.
# Calculate relative explained variance and relative
# root mean squared error for all stations and models.

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
from Functions import get_config, split_data, create_sequence, relative_explained_variance, RRMSE

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --------------------------------------------------------------------------------------
config = get_config()

mod = ['ANN', 'LSTM', 'LinRegTimeDependentEns']
met = ['ExpVar', 'RRMSE']

hyperdir = config['dirs']['data'] + config['dirs']['tune']
datadir = config['dirs']['data'] + config['dirs']['pro']
resdir = config['dirs']['data'] + config['dirs']['exp']
moddir = config['dirs']['mod'] + '{mod:s}/'
outdir = config['dirs']['data'] + config['dirs']['ts']

metfile = '{mod:s}_training_results.nc'
hyperfile = '{mod:s}_hyperparameters.csv'
datafile = 'data_station_{stat:n}_test.csv'
modfileNN = '{mod:s}_trained_model_{stat:n}_{i:02n}.pkl'
modfileLR = '{mod:s}_trained_{stat:n}_{ns:n}_{i:n}.pkl'
outfile = 'test_set_results.nc'

dim = config['dim']
N = config['hyper']['N']       # number of ensemble members
nmax = 12    # maximum sequence length
# -----------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(outdir):
	os.makedirs(outdir)


# --- Get the list of stations ---
# (from the linear regression results file)
temp = xr.open_dataset(resdir+metfile.format(mod=mod[-1]), engine = 'netcdf4')
stat = list(temp.coords[dim['s']].values)
statname = [str(temp['name'].loc[{dim['s'] : s}].values) for s in stat]
nstat = len(stat)
ens = temp.coords[dim['m']].values
models = mod[:2] + ['LR'+str(i) for i in temp.coords[dim['n']].values]


# --- Get the LSTM hyperparameters (sequence length) ---
hyper = pd.read_csv(hyperdir+hyperfile.format(mod='LSTM'), index_col = 0)



# --- Evaluate all models on the test set ---
for s,si in zip(stat, range(nstat)):
	print('Station %2i/%2i: %s (%i)' % (si+1, nstat, statname[si], s))	
	
	# -- Load data
	data = pd.read_csv(datadir+datafile.format(stat=s), index_col = 0, parse_dates = True)
	features = list(data.columns)[1:]
	t0 = data.index.values
	nt = len(t0)
	ytrue = data.iloc[:,0].values
	
	# -- Create a dataset for storing results
	if (s==stat[0]):
		coords = {dim['s']: stat, 'mod': models, dim['m']: ens, dim['t']: t0}
		data_vars = {'name' : ((dim['s']), statname), \
			dim['n'] : ((dim['s']), np.ones((nstat), dtype = np.int8)), \
			'true' : ((dim['s'], dim['t']), np.full((nstat, nt), np.nan)), \
			'pred' : ((dim['s'], 'mod', dim['m'], dim['t']), \
			np.full((nstat, len(models), N, nt), np.nan)), \
			'ExpVar' : ((dim['s'], 'mod', dim['m']), \
			np.full((nstat, len(models), N), np.nan)), \
			'RRMSE' : ((dim['s'], 'mod', dim['m']), \
			np.full((nstat, len(models), N), np.nan))}
		res = xr.Dataset(coords = coords, data_vars = data_vars)
	
	# -- Save true values
	res['true'].loc[{dim['s'] : s}] = ytrue
	
	
	# -- Prepare data
	# ANN (1)
	data1 = data.dropna(axis = 0)
	
	X1 = data1.iloc[:, 1:].values
	t1 = data1.index.to_numpy()
	
	# LSTM (2)
	seq_len = hyper.loc[s, dim['n']]
	res[dim['n']].loc[{dim['s']:s}] = seq_len-1 # to make it comparable to LinReg
	
	X2 = data.iloc[:,1:].values
	y2 = data.iloc[:,0].values
	t2 = data.index.values
	
	X2, y2, t2 = create_sequence(X2, y2, seq_len, t = t2)
	
	ind = ~np.isnan(y2)
	X2 = X2[ind]
	t2 = t2[ind]
	
	# Linear Regression (3) (has another dimension for different sequence lengths)
	X3 = [None] * (nmax+1)
	t3 = [None] * (nmax+1)
	
	for ns in range(nmax+1):
		colname = [data.columns[0]] + \
			['{:s}-{:n}'.format(f, n) for n in range(ns+1) for f in features]
		data3 = pd.DataFrame(columns = colname, index = t0[ns:])
		data3.iloc[:,0] = data.iloc[ns:,0].values
		cn = ['{:s}-{:n}'.format(f, 0) for f in features]  # 0 added time steps
		data3.loc[:, cn] = data.iloc[ns:, 1:].values # because -0 as index not allowed
		for i in range(1, ns+1):
			cn = ['{:s}-{:n}'.format(f, i) for f in features]
			data3.loc[:,cn] = data.iloc[ns-i:-i,1:].values
		data3.dropna(axis = 0, inplace = True)
		X3[ns] = data3.iloc[:,1:].values
		t3[ns] = data3.index.values
		
		
	# -- Load and evaluate models
	# ANN (1)
	m = 'ANN'
	for i in ens:
		model = joblib.load(moddir.format(mod=m)+modfileNN.format(mod=m, stat=s, i=i-1))
		ypred = np.squeeze(model.predict(X1, verbose = 0))
		res['pred'].loc[{dim['s']:s, 'mod':m, dim['m']:i, dim['t']:t1}] = ypred
	print('-- ' + m)
	
	# LSTM (2)
	m = 'LSTM'
	for i in ens:
		model = joblib.load(moddir.format(mod=m)+modfileNN.format(mod=m, stat=s, i=i-1))
		ypred = np.squeeze(model.predict(X2, verbose = 0))
		res['pred'].loc[{dim['s']:s, 'mod':m, dim['m']:i, dim['t']:t2}] = ypred
	print('-- ' + m)
	
	# Linear Regression (3)
	m0 = mod[-1]
	m = 'LR'
	for ns in range(nmax+1):
		for i in ens:
			model = joblib.load(moddir.format(mod=m0)+ \
				modfileLR.format(mod=m0, stat=s, ns=ns, i=i))
			ypred = model.predict(X3[ns])
			res['pred'].loc[{dim['s']:s,'mod':m+str(ns),dim['m']:i,dim['t']:t3[ns]}] = \
				ypred
	print('-- Linear Regression')
		
		
	# --- Calculate metrics
	for m in models:
		for i in ens:
			res['ExpVar'].loc[{dim['s']:s, 'mod':m, dim['m']:i}] = \
				relative_explained_variance(res['true'].loc[{dim['s']:s}].values, \
				res['pred'].loc[{dim['s']:s, 'mod':m, dim['m']:i}].values)
			res['RRMSE'].loc[{dim['s']:s, 'mod':m, dim['m']:i}] = \
				RRMSE(res['true'].loc[{dim['s']:s}].values, \
				res['pred'].loc[{dim['s']:s, 'mod':m, dim['m']:i}].values)
	print('----- metrics')


# --- Save results ---
res.to_netcdf(outdir+outfile)
print('--------------> Test set evaluation completed!')
