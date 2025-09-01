#!/usr/bin/env python3

# Checking the correlations between the features on a full dataset (which is not standardized).

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import numpy as np
import pandas as pd
import datetime
from scipy.stats import pearsonr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, moduledir)
from Functions import get_config

# -------------------------------------------------------------------------------------------------
config = get_config()

var = config['drivers']['local'] + config['drivers']['remote']
name = [config['variables'][v] for v in var]

indir = config['dirs']['data'] + config['dirs']['pro']
outdir = config['dirs']['data'] + config['dirs']['corr']
figdir = config['dirs']['figs'] + config['dirs']['corr']

statfile = config['dirs']['data'] + config['dirs']['ext'] + config['variables']['sl'] + '.csv'

filename0 = 'data_station_[id].csv'
filename1 = 'correlation_[id].csv'
filename2 = 'pvalue_[id].csv'
figname = 'correlation_[id].png'

ic = 'id'  # index column
significance = 0.05
maxcorr = 0.5
# -------------------------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(outdir):
	os.makedirs(outdir)
if not os.path.exists(figdir):
	os.makedirs(figdir)
	
	
# --- Get station index from the sea level file ---
stat = pd.read_csv(statfile)
statname = stat.loc[:, 'name'].values
stat = stat.loc[:, ic].values



# --- Plotting preparation ---
ticks = [a+0.5 for a in range(len(var))]
tickscb = np.arange(-1.0, 1.2, 0.2)
labelscb = [str(round(a, 1)) for a in tickscb]
cmap = plt.get_cmap('coolwarm', len(tickscb)-1)
#norm = mcolors.BoundaryNorm(boundaries = [0.05, 100], ncolors = 2)



	
# --- Calculate correlations for all stations ---
for s, n in zip(stat, statname):
	# load data, remove sea level (target), and remove rows with NaNs
	data = pd.read_csv(indir + filename0.replace('[id]', str(s)), index_col = 0)
	data = data[name]
	data.dropna(axis = 0, inplace = True)
	
	# calculate correlation
	corr = pd.DataFrame(columns = name, index = name, dtype = float)
	pval = pd.DataFrame(columns = name, index = name, dtype = float)
	
	for v1 in name:
		for v2 in name:
			corr.loc[v1, v2], pval.loc[v1, v2] = pearsonr(data.loc[:, v1].values, \
				data.loc[:, v2].values)
	
	# save results
	corr.to_csv(outdir + filename1.replace('[id]', str(s)))
	pval.to_csv(outdir + filename2.replace('[id]', str(s)))
	
	# mask significant (to hatch only insignificant)
	pvalm = np.ma.masked_less(pval.values, significance)
	
	
	# --- Plot correlations ---
	# prepare figure
	fig, ax = plt.subplots(1, 1, figsize = [12, 12.5])
	plt.subplots_adjust(left=0.15, right = 0.95, top = 0.85, bottom = 0.1)
	
	# plot correlations and p-values
	h = plt.pcolormesh(corr.values, vmin = -1, vmax = 1, cmap = cmap)
	plt.pcolor(pvalm, hatch = 'x', alpha = 0.0)
	
	# add correlations as text
	for i in range(len(var)):
		for j in range(len(var)):
			if not pvalm[i,j]:
				plt.text(ticks[i], ticks[j], str(round(corr.iloc[i,j], 2)), \
					ha = 'center', va = 'center', fontsize = 14)
	
	# mark correlations higher than 0.5 (in absolute sense)
	corr = np.ma.masked_outside(corr.values, -maxcorr, maxcorr)
	for i in range(len(var)):
		for j in range(len(var)):
			if not corr[i,j] and (i != j):
				ax.add_patch(mpl.patches.Rectangle((i, j), 1, 1, \
					lw = 3, fill = False, edgecolor = 'k'))
	
	# add tick labels, colorbar and title
	plt.xticks(ticks = ticks, labels = var, fontsize = 16)
	plt.yticks(ticks = ticks, labels = var, fontsize = 16)
	cb = plt.colorbar(h, ax = ax, location = 'bottom', shrink = 0.7, fraction = 0.08, pad = 0.08)
	cb.ax.set_xticks(ticks = tickscb, labels = labelscb, fontsize = 16)
	plt.suptitle('Correlations between all sea level drivers for\n'+n+' ('+str(s)+')', \
		fontsize = 24, y = 0.95)
	
	# save figure
	fig.savefig(figdir + figname.replace('[id]', str(s)))
	plt.close(fig)
	# ------------ End plot --------------------------------
