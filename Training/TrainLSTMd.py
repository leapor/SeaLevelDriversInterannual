#!/usr/bin/env python3

# needs SLD-ML environment

# LSTMd experiment
# Train LSTM models that are not included into LSTM, LSTM2 and LSTMtzd experiments.
# Train models for all sequence lengths. Select hyperparameters based on the best models.
# Save all models.
# Save the training history for all models.
# Save results into one file per station: explained variance, relative RMSE, start and
# end year of the validation set.

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
import datetime

sys.path.insert(0, moduledir)
from Functions import get_config, ensembleLSTM


# ------------------------------------------------------------------------------------
config = get_config()

mod = 'LSTMd'
dim = config['dim']

datadir = config['dirs']['data'] + config['dirs']['pro']
hyperdir = config['dirs']['data'] + config['dirs']['tune']
resdir = config['dirs']['data'] + config['dirs']['exp']     
moddir = config['dirs']['mod'] + mod + '/'

datafile = 'data_station_{stat:n}_train.csv'
hyperfile = mod + '_hyperparameters.csv'
resfile = mod + '_validation_results_{stat:n}_{seq:n}.csv'
histfile = mod + '_training_history_{stat:n}_{seq:n}.csv'
modfile = mod + '_trained_model_{stat:n}_{seq:n}_{{:02n}}.pkl'

learning_rate = config['hyper']['learning_rate']
loss_fun = config['hyper']['loss_fun']
epochs = config['hyper']['epochs']     # maximum number of epochs
val_len = config['hyper']['val_len']  # length of the val set (in years)
N = config['hyper']['N']       # number of ensemble members

sstart = 2106
nstart = 4
# ------------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(resdir):
	os.makedirs(resdir)
if not os.path.exists(moddir):
	os.makedirs(moddir)
	
	
# --- Get hyperparameters ---
hyper = pd.read_csv(hyperdir+hyperfile, index_col = (0,1))
stat = list(dict.fromkeys(hyper.index.get_level_values(0)))
nstat = len(stat)
seq_len = list(dict.fromkeys(hyper.index.get_level_values(1)))



# --- Training models ---
for s, si in zip(stat, range(len(stat))):
	start_time_tot = time.time()
	print('%s station %2i/%2i %s (%i)' % (time.strftime("%H:%M:%S", time.localtime()), \
		si+1, nstat, hyper.loc[(s,seq_len[0]),'name'], s))
		
	# skip if sstart is higher
	if sstart > s: continue

	# load data
	data = pd.read_csv(datadir+datafile.format(stat=s), index_col = 0, parse_dates = True)
	
	# loop through sequence lengths
	for n in seq_len:
		start_time = time.time()
		print(' - start training station %2i/%2i (%i) - sequence=%2i - %s' % (\
			si+1, nstat, s, n, (time.strftime("%H:%M:%S", time.localtime()))))
		
		# skip if s==sstart but nstart > n
		if (s==sstart) and (nstart>n): continue
	
		# input parameters
		hyperparams = {dim['u'] : hyper.loc[(s,n), dim['u']], \
			dim['a'] : hyper.loc[(s,n), dim['a']], \
			'learning_rate' : learning_rate, 'loss_fun' : loss_fun}
		fitting_params = {'valy' : val_len, 'epochs' : epochs, 'batch_size' : 1, \
			'sequence_len' : n}
		
		# train ensemble
		metrics = ensembleLSTM(data, N, hyperparams, fitting_params, \
			store_models = moddir+modfile.format(stat=s, seq=n), \
			save_hist = resdir+histfile.format(stat=s, seq=n))
	
		# save results
		metrics.to_csv(resdir+resfile.format(stat=s, seq=n))
	
		print('  --> training time for station %2i/%2i (%i) - sequence=%2i: %6.1f m' % (\
			si+1, nstat, s, n, (time.time()-start_time)/60))
	print('--> total training time for station %2i/%2i (%i): %6.1f m' % (\
		si+1, nstat, s, (time.time()-start_time_tot)/60))
