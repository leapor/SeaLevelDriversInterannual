#!/usr/bin/env python3

# Selecting best hyperparameters from the results of the hyperparameter tuning script.
# LSTMd experiment
# Detailed analysis of the best hyperparameters for each sequence length and then 
# comparsion of the results for different sequence lengths.
# Discards the worst model and compares the results with the whole ensemble results.

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

dim = config['dim']
mod = 'LSTMd'
met = 'ExpVar'

tar = 'sl'
slfile = config['dirs']['data'] + config['dirs']['ext'] + config['variables'][tar] + '.csv'

datadir = config['dirs']['data'] + config['dirs']['tune']
infile = '{:s}_tuning.nc'.format(mod)
outfile = '{:s}{{:s}}_hyperparameters.csv'.format(mod)

figdir = config['dirs']['figs'] + config['dirs']['tune']
figname1 = '{:s}_hyperparameters_{{:n}}.png'.format(mod)
figname2 = '{:s}_hyperparameters.png'.format(mod)

discard = 3  # how many worst models to remove
best_type = ['_whole_ens', '']

dx = 0.1
w = 0.6

col0 = ['darkgrey', 'black', 'darkgrey']
col = [['gainsboro', 'dimgrey'], ['rosybrown', 'maroon']]
col1 = ['rosybrown', 'maroon', 'darkgoldenrod']
col2 = ['lightblue', 'teal']
p1 = {'linewidth' : 3}
p2 = {'linewidth' : 4}
p3 = {'linewidth' : 0.1}
# -----------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(figdir):
	os.makedirs(figdir)



# --- Load hyperparameters ---
hyp = xr.open_dataset(datadir + infile, engine = 'netcdf4')
stat = hyp.coords[dim['s']].values

# coordinates
activation = hyp.coords[dim['a']].values
sequence_len = hyp.coords[dim['n']].values
units = hyp.coords[dim['u']].values
ens = hyp.coords[dim['m']].values


# --- Get station names and locations ---
stations = pd.read_csv(slfile, index_col = 0)
stations = stations.loc[stat, :]
statname = {(s, n): stations.loc[s, 'name'] for s in stat for n in sequence_len}
lon = {(s, n): stations.loc[s, 'lon'] for s in stat for n in sequence_len}
lat = {(s, n): stations.loc[s, 'lat'] for s in stat for n in sequence_len}



# --- Find the best hyperparams combination for each sequence length ---
# discard the worst models
hyp2 = hyp[met].loc[{dim['m']: ens[:(len(ens)-discard)]}]
for s in stat:
	for n in sequence_len:
		for a in activation:
			for u in units:
				x = hyp[met].loc[{dim['s']:s, dim['n']:n, dim['a']:a, \
					dim['u']:u}].values
				x = np.sort(x)
				x = x[discard:]
				x = x[::-1]
				hyp2.loc[{dim['s']:s, dim['n']:n, dim['a']:a, \
					dim['u']:u}] = x

hyp = [hyp[met], hyp2]

# --- For all models and best models separately ---
best = [None] * len(hyp)
res = [None] * len(hyp)

for i in range(len(hyp)):
	# find the best models
	med = hyp[i].median(dim = dim['m'])
	best[i] = med.argmax(dim = (dim['a'], dim['u']))
	best[i] = med.isel(best[i])
	best[i] = best[i].to_dataframe()
	
	for s in stat:
		for n in sequence_len:
			temp = hyp[i].loc[{dim['s'] : s, dim['n'] : n, \
				dim['a'] : best[i].loc[(s, n), dim['a']], \
				dim['u'] : best[i].loc[(s, n), dim['u']]}].values
			best[i].loc[(s, n),'ExpVarMin'] = np.min(temp)
			best[i].loc[(s, n),'ExpVarMax'] = np.max(temp)

	# reorganize, add station info and save the results
	best[i] = best[i].iloc[:, [-5, -4, -2, -3, -1]]
	best[i].insert(loc = 0, column = 'name', value = statname)
	best[i].insert(loc = 1, column = 'lon', value = lon)
	best[i].insert(loc = 2, column = 'lat', value = lat)
	best[i].to_csv(datadir+outfile.format(best_type[i]))
	
	# extract the data for best models
	res[i] = [None] * len(stat)
	for s, si in zip(stat, range(len(stat))):
		temp = [None] * len(sequence_len)
		for n, ni in zip(sequence_len, range(len(sequence_len))):
			temp[ni] = hyp[i].loc[{dim['s']:s, dim['n']:n, \
				dim['a'] : best[i].loc[(s,n), dim['a']], \
				dim['u'] : best[i].loc[(s,n), dim['u']]}]
			temp[ni] = temp[ni].reset_coords([dim['a'], dim['u']])
		res[i][si] = xr.concat(temp, dim = dim['n'])
	res[i] = xr.concat(res[i], dim = dim['s'])




# --- Plot all hyperparameter combinations ---
for s,si in zip(stat, range(len(stat))):
	# create figure
	fig, ax = plt.subplots(len(activation), len(sequence_len), figsize = [24, 10], \
		sharex=True, sharey=True)
	plt.subplots_adjust(left=0.04, bottom=0.08, right=0.98, top=0.92, hspace=0.0, wspace=0.05)
	
	# check for outliers
	# (if one value is significantly different than others crop the figure to fit the others)
	ylim = hyp[1].loc[{dim['s']: s}].quantile([0.01, 1]).values
	ylim = [ylim[0]-5, ylim[-1]+5]
	
	# for each sequence length and activation function
	for ni, n in zip(range(len(sequence_len)), sequence_len):
		for ai, a in zip(range(len(activation)), activation):
			# plot data
			x = hyp[1].loc[{dim['s']:s, dim['a']:a, dim['n']:n}].values
			x = np.transpose(x)
			
			bp = ax[ai,ni].boxplot(x, tick_labels = units, patch_artist = True, \
				showmeans = True, meanline = True, \
				whiskerprops = p1, capprops = p1, \
				medianprops = p2, meanprops = p3 | {'linestyle': 'solid'}, \
				zorder = 99)
			change_boxplot_colors(bp, None, *col0)
			
			# mark the best model for each sequence length
			if (a == best[1].loc[(s,n),dim['a']]):
				ind = np.where((units == best[1].loc[(s,n),dim['u']]))[0][0]
				change_boxplot_colors(bp, ind, *col1)
				
			# add lines showing the best model
			xlim = ax[ai,ni].get_xlim()
			ax[ai,ni].fill_between(xlim, \
				*best[1].loc[(s,n),['ExpVarMin','ExpVarMax']].values, \
				hatch = '//', edgecolor = col1[0], facecolor = 'none', zorder = 1)
			ax[ai,ni].axhline(best[1].loc[(s,n),'ExpVar'], color = col1[1], \
				linewidth = 2, zorder = 3)
				
			# formatting
			ax[ai,ni].tick_params(axis='x', labelsize = 12, labelrotation=45)
			ax[ai,ni].tick_params(axis='y', labelsize=16)
			if (ai==0): ax[ai,ni].set_title('{:s}: {:n}'.format(dim['n'],n), \
				fontsize = 20)
			if (ni==len(sequence_len)-1): ax[ai,ni].text(1.02, 0.5, a, fontsize = 20, \
				ha = 'left', va = 'center', rotation = 'vertical', \
				transform = ax[ai,ni].transAxes)
			ax[ai,ni].grid()
			ax[ai,ni].set_ylim(ylim)
	
	
	# title + labels + save figure
	fig.text(0.01, 0.5, 'Relative explained variance (%)', fontsize = 24, \
		ha = 'left', va = 'center', rotation = 'vertical')
	fig.text(0.5, 0.01, dim['u'], fontsize = 24, ha = 'center', va = 'bottom')
	fig.text(0.5, 0.99, 'Tuning {:s} for {:s} ({:n}) after dropping worst {:n} ensemble members'.format(mod,statname[(s,sequence_len[0])], s, discard), \
		fontsize = 24, ha = 'center', va = 'top')
	fig.savefig(figdir+figname1.format(s))
	plt.close(fig)



# --- Plot best hyperparameters for each station and sequence length ---
fig, ax = plt.subplots(8, 5, figsize = [24, 16], sharey = True)
plt.subplots_adjust(left=0.06, bottom=0.06, right=0.98, top=0.95, hspace=0.2, wspace=0.02)
ax = np.reshape(ax, -1)

for s, si in zip(stat, range(len(stat))):
	bp = [None] * 2
	for i in range(2):
		x = res[i]['ExpVar'].loc[{dim['s']:s}].values
		x = np.transpose(x)
		bp[i] = ax[si].boxplot(x, patch_artist = True, whiskerprops = p1, capprops = p1, \
			medianprops = p2, tick_labels = sequence_len-1, widths = w)
			
		for j in range(len(sequence_len)):
			bp[i]['boxes'][j].set_color(col[i][0])
			bp[i]['whiskers'][j*2].set_color(col[i][0])
			bp[i]['whiskers'][j*2+1].set_color(col[i][0])
			bp[i]['caps'][j*2].set_color(col[i][0])
			bp[i]['caps'][j*2+1].set_color(col[i][0])
			bp[i]['fliers'][j].set_markerfacecolor(col[i][0])
			bp[i]['fliers'][j].set_markeredgecolor(col[i][0])
			bp[i]['medians'][j].set_color(col[i][1])
	
	# add best hyperparameters for each sequence length
	for n, ni in zip(sequence_len, range(1, len(sequence_len)+1)):
		ax[si].text(ni, 30, best[1].loc[(s,n), dim['a']], fontsize = 10, ha = 'center', \
			va = 'top')
		ax[si].text(ni, 30, best[1].loc[(s,n), dim['u']], fontsize = 10, ha = 'center', \
			va = 'bottom')
		ax[si].text(ni, 10, best[0].loc[(s,n), dim['a']], fontsize = 10, ha = 'center', \
			va = 'top')
		ax[si].text(ni, 10, best[0].loc[(s,n), dim['u']], fontsize = 10, ha = 'center', \
			va = 'bottom')
	
	# mark the best sequence length for each station
	for i in range(2):
		best_seq = res[i]['ExpVar'].loc[{dim['s']:s}].median(dim = dim['m'])
		ind = best_seq.argmax(dim = dim['n']).values+1
		med = best_seq.max(dim = dim['n']).values
	
		ax[si].axvspan(ind-w/2, ind+w/2, color = col2[i])
		ax[si].axhline(med, color = col2[i], lw = 4)
		
	# formatting
	ax[si].tick_params(axis = 'both', labelsize = 10)
	ax[si].set_title('{:s} ({:n})'.format(statname[(s, sequence_len[0])],s), fontsize = 12)
	ax[si].grid(axis = 'y')
		
fig.text(0.01, 0.5, 'Relative explained variance (%)', fontsize = 24, \
	ha = 'left', va = 'center', rotation = 'vertical')
fig.text(0.5, 0.02, '{:s} (months before current)'.format(dim['n']), \
	fontsize = 24, ha = 'center', va = 'bottom')
fig.text(0.5, 0.99, 'Tuning LSTM for stations outside the TZD experiment and with n>2', \
	fontsize = 24, ha = 'center', va = 'top')
fig.savefig(figdir+figname2)
plt.close(fig)
