#!/usr/bin/env python3

# needs environment SLD-vis

# Combine results (predictions and metrics) for training, validation and test set for:
# 1. main experiments for the whole region (ANN, LSTM, LSTM2, LinReg)
# 2. LSTMtzd detailed transition zone experiments
# 3. LSTMd detailed experiments outside the transition zone
# Calculates RMSE for all predictions in the end. (It was not previously saved)
# Calculates metrics for training and validation sets
# Saves 2 versions:
#    1. all models
#    2. seq len that exists for both LR and NN (0,1,2,3,4,5,6,12), with one less ens mem
#       and with the indices of the median ensemble member for each model

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, moduledir)
from Functions import get_config, relative_explained_variance, corr2, RMSE

# ---------------------------------------------------------------------------------------
config = get_config()

dim = config['dim']
metmain = 'ExpVar'
modLR = 'LinRegTimeDependentEns'
#modNN = ['ANN', 'LSTM', 'LSTM2', 'LSTMtzd', 'LSTMd']
met = ['ExpVar', 'Corr2']
var = met + ['pred']
var2 = ['ValStart', 'ValEnd']

hyperdir = config['dirs']['data'] + config['dirs']['tune']# stat location + LSTM seq len
valdir = config['dirs']['data'] + config['dirs']['val']
resdir = config['dirs']['data'] + config['dirs']['fi']    # feature importance results

hyperfile = '{mod:s}_hyperparameters.csv'
valfile1 = '{mod:s}_val_timeseries_{stat:n}.csv'
valfile2 = '{mod:s}_{seq:n}_val_timeseries_{stat:n}.csv'
trainfile1 = '{mod:s}_train_timeseries_{stat:n}.csv'
trainfile2 = '{mod:s}_{seq:n}_train_timeseries_{stat:n}.csv'
trainfilemet = '{mod:s}_train_metrics.nc'
testfile = '{mod:s}_feature_importance_timeseries.nc'
resfile = '{mod:s}_feature_importance_metrics.nc'
TZfile = 'TransitionZoneBetterModel.nc'

outdir = config['dirs']['data'] + config['dirs']['ana']
outfiletest1 = 'ResultsAllTestSet.nc'
outfileval1 = 'ResultsAllValSet.nc'
outfiletrain1 = 'ResultsAllTrainSet.nc'

outfiletest2 = 'ResultsTestSet.nc'
outfileval2 = 'ResultsValSet.nc'
outfiletrain2 = 'ResultsTrainSet.nc'
# ---------------------------------------------------------------------------------------


# --- Creating output directories if they do not already exist ---
if not os.path.exists(outdir):
	os.makedirs(outdir)



# --- Sea level stations locations  and LSTM sequence length ---
hyper = pd.read_csv(hyperdir+hyperfile.format(mod='LSTM'), index_col = 0)
hyper2 = pd.read_csv(hyperdir+hyperfile.format(mod='LSTM2'), index_col = 0)
stat = hyper.index.values.astype(int)

lon = hyper.loc[:,'lon'].values
lat = hyper.loc[:,'lat'].values
statname = hyper.loc[:,'name'].values
seq_LSTM = hyper.loc[:, dim['n']].values.astype(int)-1
seq_LSTM2 = hyper2.loc[:,dim['n']].values.astype(int)-1


# --- Better model for stations and seq len for which two exist ---
best = xr.open_dataset(outdir+TZfile, engine = 'netcdf4')['best']



# -------------------------------------------------------------------
# --- TEST SET ---
# -------------------------------------------------------------------
# -- Linear regression models
temp1 = xr.open_dataset(resdir+resfile.format(mod=modLR), engine = 'netcdf4')
temp2 = xr.open_dataset(resdir+testfile.format(mod=modLR), engine = 'netcdf4')
LR = xr.merge([temp1[met], temp2[['true', 'pred']]])

LR.coords[dim['m']] = LR.coords[dim['m']].values-1
LR.coords['mod'] = [int(m[2:]) for m in LR.coords['mod'].values]
LR = LR.rename({'mod' : dim['n']})

for v in var:
	LR[v] = LR[v].expand_dims(dim = {'mod' : ['LR']})



# -- Neural network models
# ANN: add sequence len dim and set to zero
temp1 = xr.open_dataset(resdir+resfile.format(mod='ANN'), engine = 'netcdf4')
temp2 = xr.open_dataset(resdir+testfile.format(mod='ANN'), engine = 'netcdf4')
ANN = xr.merge([temp1[met], temp2[['true', 'pred']]])
ANN[var] = ANN[var].expand_dims(dim = {dim['n'] : [0]})


# LSTM & LSTM2 (all stations with seq len 1 and 2) - combine into LSTM
temp1 = xr.open_dataset(resdir+resfile.format(mod='LSTM'), engine='netcdf4')
temp2 = xr.open_dataset(resdir+testfile.format(mod='LSTM'), engine='netcdf4')
LSTM = xr.merge([temp1[met], temp2[['pred']]])

temp1 = xr.open_dataset(resdir+resfile.format(mod='LSTM2'), engine='netcdf4')
temp2 = xr.open_dataset(resdir+testfile.format(mod='LSTM2'), engine='netcdf4')
LSTM2 = xr.merge([temp1[met], temp2[['pred']]])

for v in var:
	LSTM[v] = LSTM[v].expand_dims(dim = {dim['n']:[1,2]}).copy()
		
	for si in range(len(stat)):	
		LSTM[v].loc[{dim['s']:stat[si], dim['n']:seq_LSTM2[si]}] = \
			LSTM2[v].loc[{dim['s']:stat[si]}]


# LSTMtzd & LSTMd (seq len 3,4,5,6 and 12, ignore 1&2 from LSTMtzd) - combine into LSTMd
temp1 = xr.open_dataset(resdir+resfile.format(mod='LSTMtzd'), engine='netcdf4')
temp2 = xr.open_dataset(resdir+testfile.format(mod='LSTMtzd'), engine='netcdf4')
LSTMtzd = xr.merge([temp1[met], temp2[['pred']]])
statTZ = LSTMtzd.coords[dim['s']].values

temp1 = xr.open_dataset(resdir+resfile.format(mod='LSTMd'), engine='netcdf4')
temp2 = xr.open_dataset(resdir+testfile.format(mod='LSTMd'), engine='netcdf4')
LSTMd = xr.merge([temp1[met], temp2[['pred']]])
statD = LSTMd.coords[dim['s']].values

LSTMd = xr.concat([LSTMtzd.loc[{dim['n']:LSTMd.coords[dim['n']].values}], LSTMd], dim = dim['s'])
LSTMd.coords[dim['n']] = LSTMd.coords[dim['n']]-1

seq_len_d = LSTMd.coords[dim['n']].values


# combine all neural network experiments
NN = xr.concat([ANN, LSTM, LSTMd], dim = dim['n'])

for v in var:
	NN[v] = NN[v].expand_dims(dim = {'mod': ['NN']}).copy()



# -- Combine linear regression and neural network models --
res = xr.concat([LR, NN], dim = 'mod', data_vars='minimal', coords='minimal', compat='override')

# get coordinates
models = res.coords['mod'].values
seq_len = res.coords[dim['n']].values
stat = res.coords[dim['s']].values
ens = res.coords[dim['m']].values
features = res.coords[dim['f']].values
t = res.coords[dim['t']].values


# -- Calculate RMSE (was not previously calculated) --
res['RMSE'] = res['Corr2'].copy()
res['RMSE'][:] = np.nan

for s in stat:
    true = res['true'].loc[{dim['s']:s}].values
    
    for m in models:
        for n in seq_len:
            for i in ens:
                for f in features:
                    pred = res['pred'].loc[{dim['s']:s, 'mod':m, dim['n']:n, \
                        dim['m']:i, dim['f']:f}].values
                    
                    if np.isnan(pred).all(): continue
                    
                    res['RMSE'].loc[{dim['s']:s, 'mod':m, dim['n']:n, dim['m']:i, \
                        dim['f']:f}] = RMSE(true, pred)

# update list of variables to include RMSE
var = var + ['RMSE']



# -------------------------------------------
# --- TRAINING AND VALIDATION SETS ---
# -------------------------------------------
# -- create the training and the validation datasets
train = res.copy(deep=True)
train = train.loc[{dim['t']:t[0], dim['f']:'all'}].drop_vars([dim['t'], dim['f']])

val = res.copy(deep=True)
val = val.loc[{dim['t']:t[0], dim['f']:'all'}].drop_vars([dim['t'], dim['f']])



# -- Time series
# ANN (including the dataset to store validation results and saving true values)
for s, si in zip(stat, range(len(stat))):
	tempt = pd.read_csv(valdir+trainfile1.format(mod='ANN', stat=s), index_col=0, \
		parse_dates=True)
	tempv = pd.read_csv(valdir+valfile1.format(mod='ANN', stat=s), index_col=0, \
		parse_dates=True)
	
	# expand the time dimension in the train and validation datasets
	if si==0:
		tv = tempv.index.values
		for v in ['true', 'pred']:
			train[v] = train[v].expand_dims(dim={dim['t']: len(tv)}).copy()
			val[v] = val[v].expand_dims(dim={dim['t']: len(tv)}).copy()
		train.coords[dim['t']] = tv
		val.coords[dim['t']] = tv
	
	# save true values
	train['true'].loc[{dim['s']:s}] = tempt.loc[:,'true'].values
	val['true'].loc[{dim['s']:s}] = tempv.loc[:,'true'].values
	
	# save predictions
	train['pred'].loc[{'mod':'NN', dim['s']:s, dim['n']:0}] = \
		tempt.loc[:,[str(i) for i in ens]].values
	val['pred'].loc[{'mod':'NN', dim['s']:s, dim['n']:0}] = \
		tempv.loc[:,[str(i) for i in ens]].values

# LSTM (1 & 2)
for s, si in zip(stat, range(len(stat))):
	# LSTM
	tempt = pd.read_csv(valdir+trainfile1.format(mod='LSTM', stat=s), index_col=0, \
		parse_dates=True)
	train['pred'].loc[{'mod': 'NN', dim['s']:s, dim['n']:seq_LSTM[si]}] = \
		tempt.loc[:,[str(i) for i in ens]].values
	
	tempv = pd.read_csv(valdir+valfile1.format(mod='LSTM', stat=s), index_col=0, \
		parse_dates=True)
	val['pred'].loc[{'mod': 'NN', dim['s']:s, dim['n']:seq_LSTM[si]}] = \
		tempv.loc[:,[str(i) for i in ens]].values
		
	# LSTM2
	tempt = pd.read_csv(valdir+trainfile1.format(mod='LSTM2', stat=s), index_col=0, \
		parse_dates=True)
	train['pred'].loc[{'mod': 'NN', dim['s']:s, dim['n']:seq_LSTM2[si]}] = \
		tempt.loc[:,[str(i) for i in ens]].values
	
	tempv = pd.read_csv(valdir+valfile1.format(mod='LSTM2', stat=s), index_col=0, \
		parse_dates=True)
	val['pred'].loc[{'mod': 'NN', dim['s']:s, dim['n']:seq_LSTM2[si]}] = \
		tempv.loc[:,[str(i) for i in ens]].values

# LSTMtzd
for s, si in zip(statTZ, range(len(statTZ))):
	for n in seq_len_d:
		tempt = pd.read_csv(valdir+trainfile2.format(mod='LSTMtzd', stat=s, seq=(n+1)), \
			index_col=0, parse_dates=True)
		train['pred'].loc[{'mod':'NN', dim['s']:s, dim['n']:n}] = \
			tempt.loc[:,[str(i) for i in ens]].values
	
		tempv = pd.read_csv(valdir+valfile2.format(mod='LSTMtzd', stat=s, seq=(n+1)), \
			index_col=0, parse_dates=True)
		val['pred'].loc[{'mod':'NN', dim['s']:s, dim['n']:n}] = \
			tempv.loc[:,[str(i) for i in ens]].values
			
# LSTMd
for s, si in zip(statD, range(len(statD))):
	for n in seq_len_d:
		tempt = pd.read_csv(valdir+trainfile2.format(mod='LSTMd', stat=s, seq=(n+1)), \
			index_col=0, parse_dates=True)
		train['pred'].loc[{'mod':'NN', dim['s']:s, dim['n']:n}] = \
			tempt.loc[:, [str(i) for i in ens]].values
	
		tempv = pd.read_csv(valdir+valfile2.format(mod='LSTMd', stat=s, seq=(n+1)), \
			index_col=0, parse_dates=True)
		val['pred'].loc[{'mod':'NN', dim['s']:s, dim['n']:n}] = \
			tempv.loc[:, [str(i) for i in ens]].values


# linear regression
for n in seq_len:
	for s in stat:
		tempt = pd.read_csv(valdir+trainfile2.format(mod=modLR, seq=n, stat=s), \
		    index_col=0, parse_dates=True)
		train['pred'].loc[{'mod':'LR', dim['s']:s, dim['n']:n}] = \
			tempt.loc[:,[str(i+1) for i in ens]].values
	
		tempv = pd.read_csv(valdir+valfile2.format(mod=modLR, seq=n, stat=s), \
		    index_col=0, parse_dates=True)
		val['pred'].loc[{'mod':'LR', dim['s']:s, dim['n']:n}] = \
			tempv.loc[:,[str(i+1) for i in ens]].values



# -- Calculate metrics
for s in stat:
	true = val['true'].loc[{dim['s']:s}].values
	
	for i in ens:
		for m in models:
			for n in seq_len:
				predt = train['pred'].loc[{dim['s']:s, dim['m']:i, 'mod':m, \
					dim['n']:n}].values
				predv = val['pred'].loc[{dim['s']:s, dim['m']:i, 'mod':m, \
					dim['n']:n}].values
				
				# skip for missing predictions	
				if np.isnan(predv).all(): continue
				
				# calculate and save metric
				train['ExpVar'].loc[{dim['s']:s, dim['m']:i, 'mod':m, dim['n']:n}] = \
					relative_explained_variance(true, predt)
				train['Corr2'].loc[{dim['s']:s, dim['m']:i, 'mod':m, dim['n']:n}] = \
					corr2(true, predt)
				train['RMSE'].loc[{dim['s']:s, dim['m']:i, 'mod':m, dim['n']:n}] = \
				    RMSE(true, predt)
				
				val['ExpVar'].loc[{dim['s']:s, dim['m']:i, 'mod':m, dim['n']:n}] = \
					relative_explained_variance(true, predv)
				val['Corr2'].loc[{dim['s']:s, dim['m']:i, 'mod':m, dim['n']:n}] = \
					corr2(true, predv)
				val['RMSE'].loc[{dim['s']:s, dim['m']:i, 'mod':m, dim['n']:n}] = \
				    RMSE(true, predv)



# --------------------------------------------------------
# --- Add validation set start and end year to all datasets ---
# --------------------------------------------------------
# -- create a dataset to store start and end year
coords = {'mod': models, dim['s']:stat, dim['n']:seq_len, dim['m']:ens}
data_vars = {\
	'ValStart' : (('mod', dim['s'], dim['n'], dim['m']), \
	np.full((2, len(stat), len(seq_len), len(ens)), 0, dtype=int)), \
	'ValEnd' : (('mod', dim['s'], dim['n'], dim['m']), \
	np.full((2, len(stat), len(seq_len), len(ens)), 0, dtype=int))}
valyears = xr.Dataset(coords = coords, data_vars = data_vars)


# -- read data for each model from training set metrics files
# ANN
temp = xr.open_dataset(valdir+trainfilemet.format(mod='ANN'), engine='netcdf4')
valyears[var2].loc[{'mod':'NN', dim['n']:0}] = temp[var2]

# LSTM & LSTM2
temp1 = xr.open_dataset(valdir+trainfilemet.format(mod='LSTM'), engine='netcdf4')
temp2 = xr.open_dataset(valdir+trainfilemet.format(mod='LSTM2'), engine='netcdf4')
for s,si in zip(stat, range(len(stat))):
	valyears[var2].loc[{'mod':'NN', dim['n']:seq_LSTM[si], dim['s']:s}] = \
		temp1[var2].loc[{dim['s']:s}]
		
	valyears[var2].loc[{'mod':'NN', dim['n']:seq_LSTM2[si], dim['s']:s}] = \
		temp2[var2].loc[{dim['s']:s}]
	
# LSTMtzd (without seq len 1 & 2)
valyearsTZ = xr.open_dataset(valdir+trainfilemet.format(mod='LSTMtzd'), engine='netcdf4')
valyearsTZ.coords[dim['n']] = valyearsTZ.coords[dim['n']].values - 1
valyears[var2].loc[{'mod':'NN', dim['n']: seq_len_d, dim['s']:statTZ}] = \
	valyearsTZ[var2].loc[{dim['n']:seq_len_d, dim['s']:statTZ}]

# LSTMd
temp = xr.open_dataset(valdir+trainfilemet.format(mod='LSTMd'), engine='netcdf4')
temp.coords[dim['n']] = temp.coords[dim['n']].values - 1
valyears[var2].loc[{'mod':'NN', dim['n']:seq_len_d, dim['s']: statD}] = \
	temp[var2]
	
# linear regression
temp = xr.open_dataset(valdir+trainfilemet.format(mod=modLR), engine='netcdf4')
temp.coords[dim['m']] = ens
valyears[var2].loc[{'mod':'LR'}] = temp[var2]

# -- add the information to all datasets
res = xr.merge([res, valyears])
val = xr.merge([val, valyears])
train = xr.merge([train, valyears])

# -- update var to include ValStart and ValEnd
var = var + var2



# ----------------------------------------------------
# --- Extra experiments in TZD for seq len 1&2 ---
# -----------------------------------------------------
# -- create separate files for the TZ seq1&2 experiments
# test set (existing data)
LSTMtzd.coords[dim['n']] = LSTMtzd.coords[dim['n']]-1
resTZ = LSTMtzd.loc[{dim['n']:[1,2]}]

# test set RMSE
resTZ['RMSE'] = resTZ['Corr2'].copy()
resTZ['RMSE'][:] = np.nan
for s in statTZ:
    true = res['true'].loc[{dim['s']:s}].values
    
    for n in [1,2]:
        for i in ens:
            for f in features:
                pred = resTZ['pred'].loc[{dim['s']:s, dim['n']:n, dim['m']:i, \
                    dim['f']:f}].values
                
                resTZ['RMSE'].loc[{dim['s']:s, dim['n']:n, dim['m']:i, dim['f']:f}] = \
                    RMSE(true, pred)

# training set
trainTZ = resTZ.copy(deep=True).loc[{dim['f']:'all',dim['t']:t[0]}].drop_vars([dim['t'], dim['f']])
trainTZ['pred'] = trainTZ['pred'].expand_dims(dim={dim['t']: len(tv)}).copy()
trainTZ.coords[dim['t']] = tv

# validation set
valTZ = resTZ.copy(deep=True).loc[{dim['f']:'all',dim['t']:t[0]}].drop_vars([dim['t'], dim['f']])
valTZ['pred'] = valTZ['pred'].expand_dims(dim={dim['t']: len(tv)}).copy()
valTZ.coords[dim['t']] = tv

# populate those files with data
for s in statTZ:
	true = val['true'].loc[{dim['s']:s}].values

	for n in [1,2]:
		# get time series
		tempt = pd.read_csv(valdir+trainfile2.format(mod='LSTMtzd', stat=s, seq=(n+1)), \
			index_col=0, parse_dates=True)
		trainTZ['pred'].loc[{dim['s']:s, dim['n']:n}] = \
			tempt.loc[:, [str(i) for i in ens]].values
		
		tempv = pd.read_csv(valdir+valfile2.format(mod='LSTMtzd', stat=s, seq=(n+1)), \
			index_col=0, parse_dates=True)
		valTZ['pred'].loc[{dim['s']:s, dim['n']:n}] = \
			tempv.loc[:,[str(i) for i in ens]].values
			
		
		# calculate metrics
		for i in ens:
			predt = trainTZ['pred'].loc[{dim['s']:s, dim['n']:n, dim['m']:i}].values
			trainTZ['ExpVar'].loc[{dim['s']:s, dim['n']:n, dim['m']:i}] = \
				relative_explained_variance(true, predt)
			trainTZ['Corr2'].loc[{dim['s']:s, dim['n']:n, dim['m']:i}] = \
				corr2(true, predt)
			trainTZ['RMSE'].loc[{dim['s']:s, dim['n']:n, dim['m']:i}] = \
			    RMSE(true, predt)
		
			predv = valTZ['pred'].loc[{dim['s']:s, dim['n']:n, dim['m']:i}].values
			valTZ['ExpVar'].loc[{dim['s']:s, dim['n']:n, dim['m']:i}] = \
				relative_explained_variance(true, predv)
			valTZ['Corr2'].loc[{dim['s']:s, dim['n']:n, dim['m']:i}] = \
				corr2(true, predv)
			valTZ['RMSE'].loc[{dim['s']:s, dim['n']:n, dim['m']:i}] = \
			    RMSE(true, predv)

# add validation set start and end year to all datasets
valyearsTZ = valyearsTZ[['ValStart', 'ValEnd']].loc[{dim['n']: [1,2]}]
resTZ = xr.merge([resTZ, valyearsTZ])
valTZ = xr.merge([valTZ, valyearsTZ])
trainTZ = xr.merge([trainTZ, valyearsTZ])


# -- replace the main experiments with TZ experiments if they are better
for s in statTZ:
	for n in [1,2]:
		if best.loc[{dim['s']:s, dim['n']:n}].values:
			res[var].loc[{'mod':'NN', dim['s']:s, dim['n']:n}] = \
				resTZ[var].loc[{dim['s']:s, dim['n']:n}]
			train[var].loc[{'mod':'NN', dim['s']:s, dim['n']:n}] = \
				trainTZ[var].loc[{dim['s']:s, dim['n']:n}]
			val[var].loc[{'mod':'NN', dim['s']:s, dim['n']:n}] = \
				valTZ[var].loc[{dim['s']:s, dim['n']:n}]



# --------------------------------------------
# --- Add station name and location to all datasets ---
# ----------------------------------------------
coords = {dim['s']:stat}
data_vars = {\
	'name' : ((dim['s']), statname), \
	'lon' : ((dim['s']), lon), \
	'lat' : ((dim['s']), lat)}
meta = xr.Dataset(coords = coords, data_vars = data_vars)

res = xr.merge([res, meta])
val = xr.merge([val, meta])
train = xr.merge([train, meta])


# --------------------------------------------------------
# --- Save all complete datasets ---
# --------------------------------------------------------
res.to_netcdf(outdir+outfiletest1)
val.to_netcdf(outdir+outfileval1)
train.to_netcdf(outdir+outfiletrain1)


# --------------------------------------------------------
# --- Limit results datasets to seq len common to both models ---
# --------------------------------------------------------
# --- Drop extra models ---
seq_len = train['pred'].notnull().any(dim=[dim['s'], dim['m'], dim['t']]).all(dim='mod')
seq_len = seq_len.where(seq_len, drop=True).coords[dim['n']].values

res = res.loc[{dim['n']:seq_len}]
val = val.loc[{dim['n']:seq_len}]
train = train.loc[{dim['n']:seq_len}]


# --- Find indices of median models (according to test set main metric) ---
# drop the last ensemble member if the size of the ensemble is even
if (len(ens) % 2 == 0):
    ens = ens[:-1]
    res = res.loc[{dim['m']:ens}]
    val = val.loc[{dim['m']:ens}]
    train = train.loc[{dim['m']:ens}]

# find the median ensemble member index
medind = int(len(ens)/2)    # index of the median member
base = res[metmain].loc[{dim['f']:'all'}]   # baseline metric
sortord = np.argsort(-base, axis=base.get_axis_num(dim['m'])) # sort order
medmem = sortord.loc[{dim['m']: medind}].reset_coords(drop=True) # extracted median ind

# add median ensemble member as variable to all datasets
res['median'] = medmem
val['median'] = medmem
train['median'] = medmem



# --- Save all datasets --- 
res.to_netcdf(outdir+outfiletest2)
val.to_netcdf(outdir+outfileval2)
train.to_netcdf(outdir+outfiletrain2)
