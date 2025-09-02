#!/usr/bin/env python3

# needs environmentSLD-vis

# Plotting permutation feature importance on a map for the best model at each station.
# Median of the difference between baseline and exp var with permuted driver.
# All drivers are shown despite some not contributing almost anything.
# Discrete colormap with unevenly spaced boundaries (more bins at lower values).
# Main figure does not outline the dominant drivers, but make an analysis figure that
# does.

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import string
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, moduledir)
from Functions import get_config, convert_to_marker_size

# ------------------------------------------------------------------------
config = get_config()

dim = config['dim']
var = config['variables']    # features
varname = config['varnames'] # features display name
featord = config['drivers']['order']  # display order of features
met = 'ExpVar'

resdir = config['dirs']['data'] + config['dirs']['ana']
figdir = config['dirs']['figs'] + config['dirs']['fin']

resfile = 'ResultsTestSet2012-2016.nc'
figname = 'FeatureImportanceMap.png'

mincont = 2 # minimal contribution

# plotting parameters
figsize = [26, 20]
nr, nc = 4, 3
subplots_adjust=dict(bottom=0.02,top=0.98,left=0.02,right=0.98,wspace=0.08,hspace=0.01)
lonlim, latlim = config['region']['lon'], config['region']['lat']
proj = ccrs.PlateCarree()

marker, msize = '^', [100,400]
lc = 'lightgrey'   # land color
cmap = 'gist_stern_r'
lw, lw0 = 1.5, 0.6

fs0, fs1, fs2 = 20, 16, 14
tx, ty = 0.02, 0.98
bbox = {'facecolor':'white', 'edgecolor':'none', 'alpha':0.3}
txtpar = dict(ha='left', va='top', fontsize=fs0, bbox=bbox)

lab = 'Difference in explained variance'
labels = [lab +' median (%)', lab +' interquantile range (%)']
# ------------------------------------------------------------------------


# ---------------------------------------
# --- Load and prepare data ---
# ---------------------------------------
res = xr.open_dataset(resdir+resfile, engine='netcdf4')

# coordinates
lon = res['lon'].values
lat = res['lat'].values
features = res.coords[dim['f']].values[1:]
stat = res.coords[dim['s']].values
seq_len = res.coords[dim['n']].values
mod = res.coords['mod'].values

# extract metric
res = res[met]

# -- Find and extract best model based on baseline metric median
med = res.loc[{dim['f']:'all'}].median(dim['m']).reset_coords(drop=True)
best = med.stack(model=('mod',dim['n'])).idxmax(dim='model').values

temp = [None] * len(stat)
for s, si in zip(stat, range(len(stat))):
    m, n = best[si]
    temp[si] = res.loc[{dim['s']:s, 'mod':m, dim['n']:n}]
res = xr.concat(temp, dim=dim['s']).reset_coords(drop=True)


# -- Calculate diff between baseline and feature importance predictions
base = res.loc[{dim['f']:'all'}]
res = base - res.loc[{dim['f']:features}]

# -- Reorder features and replace their name with display names
features = [var[f] for f in featord]
res = res.loc[{dim['f']:features}]

temp = {var[k] : varname[k] for k in varname}
for i in range(len(features)):
    features[i] = temp[features[i]]
res.coords[dim['f']] = features


# -- Calculate medians and IQR
med = res.median(dim['m'])
iqr = res.quantile(0.75, dim['m']) - res.quantile(0.25, dim['m'])


# -- Find dominant driver at each station
domfeat = med.loc[{dim['f']:features}].idxmax(dim=dim['f']).values


# -- Transform IQR for plotting as size  
iqrs, legend = convert_to_marker_size(iqr, lims=msize)




# --------------------------------------------
# --- Plot ---
# --------------------------------------------
# prepare colormaps
bounds = [mincont, 5, 10, 20, 30, 40, 50, 60, 80, 100]#list(range(mincont, 101, 10))
norm = BoundaryNorm(boundaries=bounds, ncolors=len(bounds)+1, extend='both')
cmap = ListedColormap(plt.get_cmap(cmap)(np.linspace(0, 0.95, len(bounds)+1)))

# subplot letters
letters = string.ascii_lowercase

# prepare figure
fig, ax = plt.subplots(nr, nc, figsize=figsize, subplot_kw = {'projection': proj})
plt.subplots_adjust(**subplots_adjust)
    
for f, a, l in zip(features, ax.flat, letters):
    # prepare map
    a.set_extent(lonlim + latlim, crs = proj)
    a.add_feature(cfeature.LAND.with_scale('10m'), edgecolor=lc, facecolor=lc)
    gl = a.gridlines(draw_labels = True, x_inline = False, y_inline = False)
    
    gl.ylocator = mticker.MultipleLocator(5)
    gl.right_labels = False
    gl.top_labels = False
    gl.xlabel_style = {'size': fs2}
    gl.ylabel_style = {'size': fs2, 'rotation': 'vertical'}
        
    # extract data
    c = med.loc[{dim['f']:f}].values
    s = iqrs.loc[{dim['f']:f}].values
    mask = (c >= mincont)
        
    # plot
    #h = a.scatter(lon[mask], lat[mask], c=c[mask], s=s[mask], \
    #    marker=marker, cmap=cmap, norm=norm, edgecolor='k', lw=lw0)
        
    h = a.scatter(lon, lat, c=c, s=s, marker=marker, cmap=cmap, norm=norm, \
        edgecolor='k', lw=lw0)
    
    # backup plot only: outline dominant driver and print the number of stations it is
    # dominant at
    #dom = (domfeat==f)
    #a.scatter(lon[mask&dom], lat[mask&dom], c=c[mask&dom], s=s[mask&dom], \
    #    marker=marker, cmap=cmap, norm=norm, edgecolor='k', lw=lw)
    #if np.sum(dom)>0:
    #    a.text(0.97, 0.02, np.sum(dom), ha='right', va='bottom', fontsize=fs0, \
    #        transform=a.transAxes)
        
    # title
    a.text(tx, ty, '{:s}) {:s}'.format(l, f), **txtpar, transform=a.transAxes)
        
# -- Colorbar for medians and legend for IQR (in the unused subplot)
cax = ax.flat[-1]
cax.set_axis_off()
    
# colorbar
cb = plt.colorbar(h, ax=cax, location='top', shrink=0.9, aspect = 20, fraction=0.25)
cb.ax.tick_params(labelsize=fs1)
cb.ax.xaxis.set_ticks_position('bottom')
cb.ax.xaxis.set_label_position('bottom')
cb.set_label(labels[0], fontsize=fs0)
cb.set_ticks(ticks=bounds, labels=bounds, fontsize=fs1)
    
# legend (markers + labels)
pos = np.linspace(0.1, 0.9, len(legend[1]))
for i in range(len(legend[1])):
    cax.scatter(pos[i], 0.55, color='none', s=legend[1][i], marker=marker, \
        edgecolor='k', lw=0.8, transform=cax.transAxes, clip_on=False)
    cax.text(pos[i], 0.45, legend[0][i], fontsize=fs1, ha='center', va='top', \
        transform=cax.transAxes)
cax.text(0.5, 0.35, labels[1], fontsize=fs0, ha='center', va='top', \
    transform=cax.transAxes)

# save figure
fig.savefig(figdir+figname)
