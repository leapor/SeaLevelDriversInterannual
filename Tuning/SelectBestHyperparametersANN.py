#!/usr/bin/env python3

# Selecting best hyperparameters from the results of the hyperparameter tuning script.
# ANN experiment

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import norm

sys.path.insert(0, moduledir)
from Functions import get_config, change_boxplot_colors

# -----------------------------------------------------------------------------------
config = get_config()

tar = 'sl'
slfile = config['dirs']['data'] + config['dirs']['ext'] + config['variables'][tar] + '.csv'

datadir = config['dirs']['data'] + config['dirs']['tune']
infile = 'ANN_tuning.nc'
outfile = 'ANN_hyperparameters.csv'

figdir = config['dirs']['figs'] + config['dirs']['tune']
figname = 'ANN_hyperparameters_{stat:n}.png'

# short names of the dimensions
dim = {'s': 'stations', 'a':'activation', 'l':'layers', 'u': 'units', 'm': 'model_run'}

dx = 0.1

col0 = ['darkgrey', 'black', 'darkgoldenrod']
col1 = ['rosybrown', 'maroon', 'darkgoldenrod']
col2 = 'teal'
# ----------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(figdir):
	os.makedirs(figdir)



# --- Load hyperparameters ---
hyp = xr.open_dataset(datadir + infile, engine = 'netcdf4')
stat = hyp.coords[dim['s']].values
#stat = stat[:43]

# select only stations with data (7-32)
hyp = hyp.loc[{dim['s'] : stat}]

# coordinates
activation = hyp.coords[dim['a']].values
layers = hyp.coords[dim['l']].values
units = hyp.coords[dim['u']].values
model = hyp.coords[dim['m']].values



# --- Get station names ---
stations = pd.read_csv(slfile, index_col = 0)
statname = {s: stations.loc[s, 'name'] for s in stat}
lon = {s: stations.loc[s, 'lon'] for s in stat}
lat = {s: stations.loc[s, 'lat'] for s in stat}


# --- Find the best hyperparams combination ---
# mean
mean = hyp['ExpVar'].mean(dim = dim['m'])
best_mean = mean.argmax(dim = (dim['a'], dim['l'], dim['u']))
best_mean = mean.isel(best_mean)
best_mean = best_mean.to_dataframe()
for s in stat:
	temp = hyp['ExpVar'].loc[{dim['s'] : s, \
		dim['a'] : best_mean.loc[s,dim['a']], \
		dim['l'] : best_mean.loc[s,dim['l']], \
		dim['u'] : best_mean.loc[s,dim['u']]}].values
	best_mean.loc[s,'ExpVarMin'] = np.min(temp)
	best_mean.loc[s,'ExpVarMax'] = np.max(temp)

# median
med = hyp['ExpVar'].median(dim = dim['m'])
best_med = med.argmax(dim = (dim['a'], dim['l'], dim['u']))
best_med = med.isel(best_med)
best_med = best_med.to_dataframe()
for s in stat:
	temp = hyp['ExpVar'].loc[{dim['s'] : s, \
		dim['a'] : best_med.loc[s,dim['a']], \
		dim['l'] : best_med.loc[s,dim['l']], \
		dim['u'] : best_med.loc[s,dim['u']]}].values
	best_med.loc[s,'ExpVarMin'] = np.min(temp)
	best_med.loc[s,'ExpVarMax'] = np.max(temp)
	
# save the results
best = best_med.iloc[:, :-2]
best.insert(loc = 0, column = 'name', value = statname)
best.insert(loc = 1, column = 'lon', value = lon)
best.insert(loc = 2, column = 'lat', value = lat)
best.to_csv(datadir+outfile)
print(best)


# --- Plot ---
# boxplot properties
p1 = {'linewidth' : 3}
p2 = {'linewidth' : 4}

# plotting
for s in stat:
	# create figure for each station
	fig, ax = plt.subplots(len(activation), len(layers), figsize = [18, 12], sharex = True, \
		sharey = True)
	plt.subplots_adjust(left=0.06, bottom=0.07, right=0.95, top=0.92, hspace=0.0, wspace=0.0)
	
	# check for outliers
	# (if one value is significantly different than others crop the figure to fit the others)
	ylim = hyp['ExpVar'].loc[{dim['s']: s}].quantile([0.01, 1]).values
	ylim = [ylim[0]-5, ylim[-1]+5]
	
	for ai, a in zip(range(len(activation)), activation):
		for li, l in zip(range(len(layers)), layers):
			# plot data
			x = hyp['ExpVar'].loc[{dim['s']:s, dim['a']:a, dim['l']:l}].values
			x = np.transpose(x)
				
			bp = ax[ai,li].boxplot(x, tick_labels = units, patch_artist = True, \
				showmeans = True, meanline = True, \
				whiskerprops = p1, capprops = p1, \
				medianprops = p2, meanprops = p1 | {'linestyle': 'solid'}, \
				zorder = 99)
			change_boxplot_colors(bp, None, *col0)
			
			# mark the best model based on median
			if (a == best_med.loc[s,dim['a']]) and (l == best_med.loc[s,dim['l']]):
				ind = np.where((units == best_med.loc[s,dim['u']]))[0][0]
				change_boxplot_colors(bp, ind, *col1)
			
			# based best model based on mean
			if (a == best_mean.loc[s,dim['a']]) and (l == best_mean.loc[s,dim['l']]):
				ind = np.where((units == best_mean.loc[s,dim['u']]))[0][0]
				ax[ai,li].scatter(ind+1, best_mean.loc[s,'ExpVar'], \
					s=100, c=col2, marker='X', zorder = 100)
				#ax[ai,li].axhline(best_mean.loc[s,'ExpVar'], color = col1[1])
				
			# add lines showing the median-based best model to all the subplots
			xlim = ax[ai,li].get_xlim()
			ax[ai,li].fill_between(xlim, \
				*best_med.loc[s,['ExpVarMin','ExpVarMax']].values, \
				hatch = '//', edgecolor = col1[0], facecolor = 'none', zorder = 1)
			ax[ai,li].axhline(best_med.loc[s,'ExpVar'], color = col1[1], \
				linewidth = 2, zorder = 3)
				
			# add lines showing the mean-based best model to all subplots
			ax[ai,li].fill_between(xlim, \
				*best_mean.loc[s,['ExpVarMin', 'ExpVarMax']].values, \
				hatch = '\\', edgecolor = col2, facecolor = 'none', zorder = 2)
			ax[ai,li].axhline(best_mean.loc[s,'ExpVar'], color = col2, \
				linewidth = 2, zorder = 4)
				
			# formatting	
			ax[ai,li].tick_params(axis='both', labelsize=16)
			if (ai==0): ax[ai,li].set_title(str(l)+' '+dim['l'], fontsize = 20)
			if (li==len(layers)-1): ax[ai,li].text(1.02, 0.5, a, fontsize = 20, \
				ha = 'left', va = 'center', rotation = 'vertical', \
				transform = ax[ai,li].transAxes)
			ax[ai,li].grid()
			ax[ai,li].set_ylim(ylim)
			
	fig.text(0.01, 0.5, 'Relative explained variance (%)', fontsize = 24, \
		ha = 'left', va = 'center', rotation = 'vertical')
	fig.text(0.5, 0.01, dim['u'], fontsize = 24, ha = 'center', va = 'bottom')
	fig.text(0.5, 0.99, 'Tuning ANN for '+statname[s]+' ('+str(s)+')', fontsize = 24, \
		ha = 'center', va = 'top')
	fig.savefig(figdir+figname.format(stat=s))
	plt.close(fig)
