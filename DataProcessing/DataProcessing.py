#!/usr/bin/env python3

# Preparing all variables to be used as input for machine learning models.
# Creates one dataset per sea level station, with both sea level and drivers inside.

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import datetime
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, moduledir)
from Functions import get_config, detrend, deseason, split_data, plot_2_timeseries

# -------------------------------------------------------------------------------------------------
config = get_config()

tstest = config['time']['test']  # test set start and end year

varloc = config['drivers']['local']
varglo = config['drivers']['remote']
nameloc = [config['variables'][v] for v in varloc]
nameglo = [config['variables'][v] for v in varglo]
tar = 'sl'
tname = config['variables'][tar]

indir = config['dirs']['data'] + config['dirs']['ext']
outdir = config['dirs']['data'] + config['dirs']['pro']
figdir = config['dirs']['figs'] + config['dirs']['pro']

filename0 = 'data_station_{stat:n}.csv'
filename1 = 'data_station_{stat:n}_train.csv'
filename2 = 'data_station_{stat:n}_test.csv'

figname1 = 'deseason_{stat:n}.png'
figname1a = 'seasonal_cycle_{stat:n}.png'
figname2 = 'detrend_{stat:n}.png'
figname2a = 'trend_{stat:n}.png'

ic = 'id'  # index column
# -------------------------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(outdir):
	os.makedirs(outdir)
if not os.path.exists(figdir):
	os.makedirs(figdir)


# --- Load all data ---
# and convert to the same format

# sea level
y = pd.read_csv(indir+tname+'.csv', index_col = ic)
stat = list(y.index)
statname = y.loc[:, 'name'].values
t = list(y.columns)
t = [ti for ti in t if ti[0].isdigit()]
td = pd.DatetimeIndex(t)
y = y[t].transpose()

# local variables (separate file for each variable)
loc = dict.fromkeys(varloc)
for i in range(len(varloc)):
	loc[varloc[i]] = pd.read_csv(indir+nameloc[i]+'.csv', index_col = ic)
	loc[varloc[i]] = loc[varloc[i]][t].transpose()

# global variables (all in one file)
glo = [None] * len(varglo)
for i in range(len(varglo)):
	glo[i] = pd.read_csv(indir+nameglo[i]+'.csv', index_col = 0)
glo = pd.concat(glo, axis = 1)


# --- Combine data ---
# reorganize the local data to be all variables for one station together
locstat = dict.fromkeys(stat)
for s in stat:
	locstat[s] = pd.DataFrame(index = t, columns = nameloc, dtype = np.float64)
	for i in range(len(varloc)):
		locstat[s].loc[:, nameloc[i]] = loc[varloc[i]].loc[:, s].values
	
# combine target, local and global variables into one dataframe per station
data = dict.fromkeys(stat)
for s in stat:
	data[s] = pd.concat([y.loc[:, s], locstat[s], glo], axis = 1)
	data[s].rename({s : tname}, axis = 1, inplace = True)
	data[s].index = td

# --- Process data ---
scaler = StandardScaler()
train = dict.fromkeys(stat)
test = dict.fromkeys(stat)

for s, name in zip(stat, statname):
	# remove seasonal cycle
	temp = deseason(data[s], excl = tstest, \
		plot_cycle = figdir+figname1a.format(stat = s))
	plot_2_timeseries(data[s], temp, \
		figname = figdir+figname1.format(stat = s), \
		ylab1 = name, \
		ylab2 = 'Variables', \
		title='Removing seasonal cycle for '+name+' ('+ str(s)+')')
	data[s] = temp
	
	# detrend
	temp = data[s].copy(deep = True)
	temp.loc[:,:] = detrend(data[s], plot_trend = figdir+figname2a.format(stat=s))
	plot_2_timeseries(data[s], temp, \
		figname = figdir+figname2.format(stat = s), \
		ylab1 = name, \
		ylab2 = 'Variables', \
		title='Detrending '+name+' ('+ str(s)+')', \
		sharey = False)
	data[s] = temp
	
	# train-test split
	train[s], test[s] = split_data(data[s], subset = tstest)
	
	# standardisation
	train[s].loc[:, nameloc+nameglo] = scaler.fit_transform(train[s].loc[:, nameloc+nameglo])
	test[s].loc[:, nameloc+nameglo] = scaler.transform(test[s].loc[:, nameloc+nameglo])
	
	
# --- Save data ---
for s in stat:
	data[s].to_csv(outdir+filename0.format(stat = s))
	train[s].to_csv(outdir+filename1.format(stat = s))
	test[s].to_csv(outdir+filename2.format(stat = s))
