#!/usr/bin/env python3

# needs SLD-vis environment

# Fitting the multiple linear regression model with all sequence lengths.


moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
import datetime
import joblib
from sklearn.linear_model import LinearRegression

sys.path.insert(0, moduledir)
from Functions import get_config, create_sequence, RRMSE, relative_explained_variance, \
	ensembleLinReg

# ------------------------------------------------------------------------------------
config = get_config()

mod = 'LinRegTimeDependentEns'
met = ['RRMSE', 'ExpVar']

datadir = config['dirs']['data'] + config['dirs']['pro']
resdir = config['dirs']['data'] + config['dirs']['exp']     
moddir = config['dirs']['mod'] + mod + '/'

statfile = config['dirs']['data'] + config['dirs']['ext'] + 'sea_level.csv'
datafile = 'data_station_{stat:n}_train.csv'
resfile = mod + '_training_results.nc'
modfile = mod + '_trained_{stat:n}_{ns:n}_{{}}.pkl'

# short names of the dimensions
dim = config['dim']

valy = config['hyper']['val_len']
nmax = 12 # maximum sequence length
N = config['hyper']['N']
# ------------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(resdir):
	os.makedirs(resdir)
if not os.path.exists(moddir):
	os.makedirs(moddir)
	
	
# --- Get the list of stations ---
stat = pd.read_csv(statfile, index_col = 0)
lon = stat.loc[:, 'lon'].values
lat = stat.loc[:, 'lat'].values
statname = stat.loc[:, 'name'].values
stat = stat.index.values



# --- Linear regression for all stations ---
for s in stat:
	# load data
	data = pd.read_csv(datadir+datafile.format(stat=s), index_col = 0, parse_dates = True)
	ttot = data.index
	
	# prepare the results variables
	if (s == stat[0]):
		features = list(data.columns)[1:]
		coords = {dim['s']: stat, dim['n']: range(nmax+1), dim['f']: features, \
			'timestep': range(nmax+1), dim['m'] : range(1, N+1)}
		data_vars = {'name' : ((dim['s']), statname), \
			met[0] : ((dim['s'], dim['n'], dim['m']), \
			np.full((len(stat), nmax+1, N), np.nan)), \
			met[1] : ((dim['s'], dim['n'], dim['m']), \
			np.full((len(stat), nmax+1, N), np.nan)), \
			'coefficients' : ((dim['s'], dim['n'], dim['m'], dim['f'], 'timestep'), \
			np.full((len(stat), nmax+1, N, len(features), nmax+1), np.nan)), \
			'intercept' : ((dim['s'], dim['n'], dim['m']), \
			np.full((len(stat), nmax+1, N), np.nan)), \
			'ValStart' : ((dim['s'], dim['n'], dim['m']), \
			np.full((len(stat), nmax+1, N), 0)), \
			'ValEnd' : ((dim['s'], dim['n'], dim['m']), \
			np.full((len(stat), nmax+1, N), 0))}
		res = xr.Dataset(coords = coords, data_vars = data_vars)
	
	# loop through sequence length
	for ns in range(nmax+1):
		# -- Prepare data
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
			
		# remove NaNs
		data_ext = data_ext.dropna(axis = 0)
		
		
		# --- Create and fit ensemble
		ens = ensembleLinReg(data_ext, N, valy, \
			store_models = moddir+modfile.format(stat=s, ns=ns), \
			verbose = False)
		
		
		# --- Store results
		# validation set start and end
		for i in range(1, N+1):
			res['ValStart'].loc[{dim['s']:s, dim['n']:ns, dim['m']:i}] = \
				ens.loc[i,'ValStart']
			res['ValEnd'].loc[{dim['s']:s, dim['n']:ns, dim['m']:i}] = \
				ens.loc[i, 'ValEnd']
		
		# coefficients
		for i in range(ns+1):
			cn = ['{:s}-{:n}'.format(f, i) for f in features]
			res['coefficients'].loc[{dim['s']: s, dim['n']: ns, 'timestep': i}] = \
				ens[cn]
		res['intercept'].loc[{dim['s']: s, dim['n']: ns}] = ens['intercept']
		
		# metrics
		res['RRMSE'].loc[{dim['s']:s, dim['n']: ns}] = ens['RRMSE']
		res['ExpVar'].loc[{dim['s']:s, dim['n']:ns}] = ens['ExpVar']
				
		# --- Print results
		for i in range(1, N+1):
			print('%4i %25s %2i %3i | %4i %4i | %6.2f  %6.2f' % (s, \
				res['name'].loc[{dim['s']:s}].values, ns, i, \
				res['ValStart'].loc[{dim['s']:s, dim['n']:ns, dim['m']:i}], \
				res['ValEnd'].loc[{dim['s']:s, dim['n']:ns, dim['m']:i}], \
				res['RRMSE'].loc[{dim['s']:s, dim['n']: ns, dim['m']:i}], \
				res['ExpVar'].loc[{dim['s']:s, dim['n']:ns, dim['m']:i}]))
		print('-------------------------------------------------------')
	print('================================================================')
	

# --- Save results ---
print(res)
res.to_netcdf(resdir+resfile)
