#!/usr/bin/env python3

# needs environment SLD-vis

# Feature importance for linear regression.
# Load the test set results created by the TestSet/TestSetEvaluation script.
# Perform permutation feature importance for all sequence lengths.

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
from Functions import get_config, create_sequence, RRMSE, relative_explained_variance

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# -----------------------------------------------------------------------------------
config = get_config()

mod = 'LinRegTimeDependentEns'
met = ['ExpVar', 'RRMSE']

datadir = config['dirs']['data'] + config['dirs']['pro']
resdir = config['dirs']['data'] + config['dirs']['ts']
moddir = config['dirs']['mod'] + mod + '/'
outdir = config['dirs']['data'] + config['dirs']['fi']

datafile = 'data_station_{stat:n}_test.csv'
resfile = 'test_set_results.nc'  # contains all timeseries evaluated on the test set
modfile = mod + '_trained_{stat:n}_{ns:n}_{i:n}.pkl'
outfile1 = mod + '_feature_importance_timeseries.nc'
outfile2 = mod + '_feature_importance_metrics.nc'

dim = config['dim']
N = config['hyper']['N']       # number of ensemble members
# -----------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(outdir):
	os.makedirs(outdir)


# --- Get the test set predicted time series ---
test = xr.open_dataset(resdir+resfile, engine = 'netcdf4')
stat = test.coords[dim['s']].values
ens = test.coords[dim['m']].values
models = test.coords['mod'].values
statname = test['name'].values
nstat = len(stat)

models = [m for m in models if 'LR' in m]  # select lin reg models (drops ANN and LSTM)
test = test.loc[{'mod' : models}]


# --- Everything per station ---
for s, si in zip(stat, range(nstat)):
	start_time = time.time()

	# load data
	data0 = pd.read_csv(datadir+datafile.format(stat=s), index_col = 0, parse_dates = True)
	t0 = data0.index.values
	
	# create an xarray dataset for storing results (4D)
	if (si == 0):
		# extract coordinate info from the data file
		features = list(data0.columns)
		features[0] = 'all'
		t = data0.index.to_numpy()
		
		# create the time series dataset
		coords = {dim['s']: stat, 'mod': models, dim['t']: t, dim['m']: ens, \
			dim['f']: features}
		data_vars = {'name' : ((dim['s']), statname), \
			'true' : ((dim['s'], dim['t']), np.full((len(stat), len(t)), np.nan)), \
			'pred' : ((dim['s'], 'mod', dim['t'], dim['m'], dim['f']), \
			np.full((len(stat), len(models), len(t), N, len(features)), np.nan))}
		res = xr.Dataset(coords = coords, data_vars = data_vars)
		res.to_netcdf(outdir+outfile1)
		
		# create the metrics dataset
		coords = {dim['s']: stat, 'mod': models, dim['m']: ens, dim['f']: features}
		data_vars = {'name' : ((dim['s']), statname), \
			met[0]: ((dim['s'], 'mod', dim['m'], dim['f']), \
			np.empty((len(stat), len(models), N, len(features)))), \
			met[1]: ((dim['s'], 'mod', dim['m'], dim['f']), \
			np.empty((len(stat), len(models), N, len(features))))}
		metrics = xr.Dataset(coords = coords, data_vars = data_vars)
		metrics.to_netcdf(outdir+outfile2)
		
	# save the true values (extracted from the test results file)
	res['true'].loc[{dim['s']:s}] = test['true'].loc[{dim['s']:s}].values
	
	# prediction of the time series (extracting baseline + predicting feature importance)
	for m in models:
		ns = int(m[2:])  # sequence length
		
		# prepare data
		colname = [data0.columns[0]] + \
			['{:s}-{:n}'.format(f, n) for n in range(ns+1) for f in features[1:]]
		data = pd.DataFrame(columns = colname, index = t0[ns:])
		data.iloc[:,0] = data0.iloc[ns:,0].values
		cn = ['{:s}-{:n}'.format(f, 0) for f in features[1:]]  # 0 added time steps
		data.loc[:, cn] = data0.iloc[ns:, 1:].values # because -0 as index not allowed
		for i in range(1, ns+1):
			cn = ['{:s}-{:n}'.format(f, i) for f in features[1:]]
			data.loc[:,cn] = data0.iloc[ns-i:-i,1:].values
		data.dropna(axis = 0, inplace = True)
		#X = data.iloc[:,1:].values
		t = data.index.values
		
		for i in ens:
			# extract and store the baseline results
			res['pred'].loc[{dim['s']:s, 'mod':m, dim['t']:t, dim['m']:i, \
				dim['f']:features[0]}] = \
				test['pred'].loc[{dim['s']:s, 'mod':m, \
				dim['m']:i, dim['t']:t}].values
				
			# load model
			model = joblib.load(moddir+modfile.format(stat=s, ns = ns, i=i))
			
			# permutation feature importance
			for f in features[1:]:
				# permute feature
				feat = [fc for fc in data.columns if f in fc] # all steps of feature
				
				dataper = data.copy(deep = True)
				for ft in feat:
					dataper.loc[:,ft] = \
						np.random.permutation(dataper.loc[:,ft].values)
				Xper = dataper.iloc[:,1:].values
				
				# predict
				ypred = model.predict(Xper)
				res['pred'].loc[{dim['s']:s, 'mod':m, dim['t']:t, dim['m']:i, \
					dim['f']: f}] = ypred
				
		print('Prediction %2i/%2i %5i %20s | %4s' % (si+1, nstat, s, statname[si], m))
					
	# save the predicted timeseries (after each station)
	res.to_netcdf(outdir+outfile1, mode = 'a')
	
	# calculate metrics
	for m in models:
		for i in ens:
			for f in features:
				metrics[met[0]].loc[{dim['s']:s,'mod':m,dim['m']:i,dim['f']:f}] = \
					relative_explained_variance(\
					res['true'].loc[{dim['s']:s}].values, \
					res['pred'].loc[{dim['s']:s, 'mod':m, dim['m']:i, \
					dim['f']:f}].values)
				metrics[met[1]].loc[{dim['s']:s,'mod':m,dim['m']:i,dim['f']:f}] = \
					RRMSE(\
					res['true'].loc[{dim['s']:s}].values, \
					res['pred'].loc[{dim['s']:s, 'mod':m, dim['m']:i, \
					dim['f']:f}].values)
	
	# save the metrics (after each station)
	metrics.to_netcdf(outdir+outfile2, mode = 'a')
	
	print('--- completed station %2i/%2i %s (%i) -> time needed: %6.1f min' % \
		(si+1, nstat, statname[si], s, (time.time()-start_time)/60))
	
print(res)
print(metrics)
