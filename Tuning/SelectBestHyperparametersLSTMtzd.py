#!/usr/bin/env python3

# needs SLD-vis environment

# Selecting best hyperparameters from the results of the hyperparameter tuning script.
# LSTMtzd experiment
# Detailed analysis of the best hyperparameters for each sequence length and then 
# comparsion of the results for different sequence lengths.
# Discards the worst model and compares the results with the whole ensemble results.
# Focuses only on median, without average explained variance.

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

# ----------------------------------------------------------------------------------
config = get_config()

tar = 'sl'
slfile = config['dirs']['data'] + config['dirs']['ext'] + config['variables'][tar] + '.csv'

datadir = config['dirs']['data'] + config['dirs']['tune']
infile = 'LSTM_tuning_TZ.nc'
outfile = 'LSTMtz{:s}_hyperparameters.csv'

figdir = config['dirs']['figs'] + config['dirs']['tune']
figname = 'LSTMtz_hyperparameters.png'

dim = config['dim']

discard = 3  # how many worst models to remove
best_type = ['d_all_models', 'd']

dx = 0.1
w = 0.6

col = [['gainsboro', 'dimgrey'], ['rosybrown', 'maroon']]
col2 = ['lightblue', 'teal']
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
hyp = hyp['ExpVar']
hyp2 = hyp.loc[{dim['m']: ens[:(len(ens)-discard)]}]
for s in stat:
	for n in sequence_len:
		for a in activation:
			for u in units:
				x = hyp.loc[{dim['s']:s, dim['n']:n, dim['a']:a, \
					dim['u']:u}].values
				x = np.sort(x)
				x = x[discard:]
				x = x[::-1]
				hyp2.loc[{dim['s']:s, dim['n']:n, dim['a']:a, \
					dim['u']:u}] = x
hyp = [hyp, hyp2]


# find best models
best = [None] * 2
res = [None] * 2

for i in range(2):
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



# --- Plot ---
# boxplot properties
p1 = {'linewidth' : 3}
p2 = {'linewidth' : 4}

# plotting
fig, ax = plt.subplots(3, 2, figsize = [18, 18], sharey = True)
plt.subplots_adjust(left=0.06, bottom=0.06, right=0.98, top=0.95, hspace=0.14, wspace=0.02)
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
		ax[si].text(ni, 24, best[1].loc[(s,n), dim['a']], fontsize = 16, ha = 'center', \
			va = 'top')
		ax[si].text(ni, 24, best[1].loc[(s,n), dim['u']], fontsize = 16, ha = 'center', \
			va = 'bottom')
		ax[si].text(ni, 15, best[0].loc[(s,n), dim['a']], fontsize = 16, ha = 'center', \
			va = 'top')
		ax[si].text(ni, 15, best[0].loc[(s,n), dim['u']], fontsize = 16, ha = 'center', \
			va = 'bottom')
	
	# mark the best sequence length for each station
	for i in range(2):
		best_seq = res[i]['ExpVar'].loc[{dim['s']:s}].median(dim = dim['m'])
		ind = best_seq.argmax(dim = dim['n']).values+1
		med = best_seq.max(dim = dim['n']).values
	
		ax[si].axvspan(ind-w/2, ind+w/2, color = col2[i])
		ax[si].axhline(med, color = col2[i], lw = 4)
		
	# formatting
	ax[si].tick_params(axis = 'both', labelsize = 16)
	ax[si].set_title('{:s} ({:n})'.format(statname[(s, sequence_len[0])],s), fontsize = 20)
	ax[si].grid(axis = 'y')
		
fig.text(0.01, 0.5, 'Relative explained variance (%)', fontsize = 24, \
	ha = 'left', va = 'center', rotation = 'vertical')
fig.text(0.5, 0.02, '{:s} (months before current)'.format(dim['n']), \
	fontsize = 24, ha = 'center', va = 'bottom')
fig.text(0.5, 0.99, 'Tuning LSTM for stations in the Baltic-North Sea transition zone', \
	fontsize = 24, ha = 'center', va = 'top')
fig.savefig(figdir+figname)
plt.close(fig)
