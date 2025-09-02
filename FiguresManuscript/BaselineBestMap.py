#!/usr/bin/env python3

# needs environmentSLD-vis

# Plotting the baseline predictions and which model is best.
# Top left: Baseline of the best model overall - median.
# Top right: Baseline of the best model overall - interquantile range.
# Bottom left: Model type and sequence length of the best model overall.
# Bottom right: Difference between the best NN and best LR model.

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap

sys.path.insert(0, moduledir)
from Functions import get_config

# --------------------------------------------------------------------------------------
config = get_config()

dim = config['dim']
met = 'ExpVar'
models = {'LR': 'linear regression', 'NN': 'neural network'}

resdir = config['dirs']['data'] + config['dirs']['ana']
figdir = config['dirs']['figs'] + config['dirs']['fin']

resfile = 'ResultsTestSet2012-2016.nc'
figname = 'BaselineBestMap.png'

# plotting parameters
figsize = [20, 14]
nr, nc = 2, 2
subplots_adjust=dict(bottom=0.03,top=0.97,left=0.03,right=0.98,wspace=0.05,hspace=0.03)
lonlim, latlim = config['region']['lon'], config['region']['lat']
proj = ccrs.PlateCarree()
colland = 'lightgrey'
fs0, fs1, fs2 = 18, 16, 14
tx, ty, bbox = 0.02, 0.98, {'facecolor':'white', 'edgecolor':'none', 'alpha':0.3}
txtpar = dict(ha='left', va='top', size=fs0, bbox=bbox)

m1, m2 = '^', {'LR': 'o', 'NN': 'd'}
params = dict(s=300, edgecolor='k', lw=0.6)
cmap1, cmap2 = 'jet', 'seismic'
vmax, step = 10, 2 # max diff and step in the diff plot

titles = ['a) Best model median', \
    'b) Best model interquantile range', \
    'c) Best model type and sequence length', \
    'd) Difference between neural network and linear regression']
labels = ['Explained variance {:s} (%)'.format(v) for v in ['median', 'IQR']] + \
    ['Sequence length (months)', 'Difference in explained variance median (NN-LR) (%)']
# --------------------------------------------------------------------------------------

# --- Create directory for saving figures ---
if not os.path.exists(figdir):
	os.makedirs(figdir)
	

# --- Load and prepare data ---
res = xr.open_dataset(resdir+resfile, engine='netcdf4')

# coordinates
lon = res['lon'].values
lat = res['lat'].values
stat = res.coords[dim['s']].values
mod = res.coords['mod'].values
seq_len = res.coords[dim['n']].values

# extract baseline metric and calculate median
res = res[met].loc[{dim['f']:'all'}].reset_coords(drop=True)
med = res.median(dim=dim['m'])

# extract best model of each type and calculate difference (for d)
diff = med.max(dim=dim['n'])
diff = (diff.loc[{'mod':'NN'}] - diff.loc[{'mod':'LR'}]).values

# find best model overall (type and seq len) (for c)
bestmod = med.max(dim=dim['n']).idxmax(dim='mod').values
bestseq = np.zeros(np.shape(bestmod), dtype=int)
for s, i in zip(stat, range(len(stat))):
    bestseq[i] = med.loc[{dim['s']:s, 'mod':bestmod[i]}].idxmax(dim=dim['n'])

# extract best model overall and calculate median and explained variance (for a&b)
best = [None] * len(stat)
for s, i in zip(stat, range(len(stat))):
    best[i] = res.loc[{dim['s']:s, 'mod':bestmod[i], dim['n']:bestseq[i]}]
best = xr.concat(best, dim=dim['s'])
med = best.median(dim=dim['m']).values
iqr = (best.quantile(0.75, dim=dim['m']) - best.quantile(0.25, dim=dim['m'])).values




# --- Plot ---
# prepare figure
fig, ax = plt.subplots(nr, nc, figsize=figsize, subplot_kw={'projection':proj})
plt.subplots_adjust(**subplots_adjust)

# prepare handles
hc = [None] * nr*nc
cb = [None] * len(hc)
hl = [None] * len(mod)

# prepare maps
for a in ax.flat:
    a.set_extent(lonlim + latlim, crs = proj)
    a.add_feature(cfeature.LAND.with_scale('10m'), edgecolor=colland, facecolor=colland)
    gl = a.gridlines(draw_labels = True, x_inline = False, y_inline = False)
    gl.right_labels = False
    gl.top_labels = False
    gl.xlabel_style = {'size': fs2}
    gl.ylabel_style = {'size': fs2, 'rotation': 'vertical'}
    
# -- plot
# upper left: best model median explained variance
cmap = plt.get_cmap(cmap1)
hc[0] = ax[0,0].scatter(lon, lat, c=med, marker=m1, cmap=cmap, **params)

# upper right: best model IQR of explained variance
cmap = plt.get_cmap(cmap1+'_r')
hc[1] = ax[0,1].scatter(lon, lat, c=iqr, marker=m1, cmap=cmap, **params)

# lower left: best model overall (type and sequence length)
bounds = list(seq_len)+[max(seq_len)+1]
norm = BoundaryNorm(boundaries=bounds, ncolors=len(seq_len))
ticks = [(bounds[i]+bounds[i+1])/2 for i in range(len(bounds)-1)]
cmap = plt.get_cmap(cmap1, len(seq_len))
for m,i in zip(mod, range(len(mod))):
    ind = (bestmod==m)
    hc[2] = ax[1,0].scatter(lon[ind], lat[ind], c=bestseq[ind], marker=m2[m], \
        cmap=cmap, norm=norm, **params)
    hl[i] = plt.scatter([], [], color=cmap(1), marker=m2[m], **params, label=models[m])
        
# lower right: difference between NN and LR (medians)
bounds = np.arange(-vmax, vmax+1, step)
norm = BoundaryNorm(boundaries=bounds, ncolors=len(bounds)+2, extend='both')
ind = np.concatenate([np.linspace(0, 0.45, int(len(bounds)/2)+1), \
    np.linspace(0.55, 1, int(len(bounds)/2)+1)])
cmap = ListedColormap(plt.get_cmap(cmap2)(ind))
hc[3] = ax[1,1].scatter(lon, lat, c=diff, marker=m1, cmap=cmap, norm=norm, **params)

# colorbars
for a,i in zip(ax.flat, range(len(hc))):
    cb[i] = plt.colorbar(hc[i], ax=a, location='bottom', shrink=0.9, aspect=30, \
        fraction=0.15, pad=0.05)
    cb[i].ax.tick_params(labelsize=fs1)
    cb[i].set_label(labels[i], fontsize=fs0)
cb[2].set_ticks(ticks, labels=seq_len)
cb[3].set_ticks(bounds, labels=bounds)
    
# legend
ax[1,0].legend(handles=hl, loc='lower right', fontsize=fs0)

# titles
for a, i in zip(ax.flat, range(len(titles))):
    a.text(tx, ty, titles[i], **txtpar, transform=a.transAxes)

# save figure
fig.savefig(figdir+figname)
plt.close(fig)
