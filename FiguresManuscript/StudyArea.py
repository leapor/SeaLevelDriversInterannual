#!/usr/bin/env python3

# needs environmentSLD-vis

# Plot the study area.
#  - bathymetry+elevation (GEBCO)
#  - locations of sea level stations, with PSMSL numbers
#  - ERA5 grid (re-gridded netcdf, not model resolution) - only on sea
#  - distinction between ERA5 sea and land; i.e., which grid points have SST
#  - size of the integrated area for some drivers at one station

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from adjustText import adjust_text

sys.path.insert(0, moduledir)
from Functions import get_config

# -------------------------------------------------------------------
config = get_config()

dim = config['dim']

origdir = config['dirs']['orig']
resdir = config['dirs']['data'] + config['dirs']['ana']
figdir = config['dirs']['figs'] + config['dirs']['fin']

bathyfile = 'GEBCO_27_Jun_2025_coarser.nc'
maskfile = 'ERA5_downloaded2024-11-06/ERA5_monthly_sst.nc' # sst file; land where NaN
resfile = 'ResultsTestSet.nc'  # needed only for stations list and locations

figname = 'StudyArea.png'

box_eg = 302 # which station to show box size on
box_size = 2 # how many degrees on each side of the station to integrate

# plotting parameters
figsize = [20, 14]
proj = ccrs.PlateCarree()
cmap = plt.get_cmap('BrBG_r')
lonlim, latlim = config['region']['lon'], config['region']['lat']
vmax = 1600  # max elevation and bathymetry
col = 'grey' # grid points color
# -------------------------------------------------------------------


# --- Create directory for saving figures ---
if not os.path.exists(figdir):
	os.makedirs(figdir)


# --- Load and prepare data ---
# bathymetry
bathy = xr.open_dataset(resdir+bathyfile, engine='netcdf4')
bathy = bathy['elevation']

# ERA5 grid and land/ocean mask
mask = xr.open_dataset(origdir+maskfile, engine='netcdf4')
mask = mask.rename({'longitude' : 'lon', 'latitude' : 'lat'})
mask = mask['sst'].sel(lon=slice(*lonlim), lat=slice(*latlim[::-1]))
mask = mask.isnull().all(dim='date')

# stations location
loc = xr.open_dataset(resdir+resfile, engine='netcdf4')
loc = loc[['lon', 'lat', 'name']]
stat = loc.coords[dim['s']].values



# --- Plot ---
# prepare map
fig, ax = plt.subplots(1, 1, figsize = figsize, subplot_kw = {'projection':proj})
plt.subplots_adjust(bottom=0.02, top=0.97, left=0.05, right=0.99)
ax.set_extent(lonlim + latlim, crs = proj)
gl = ax.gridlines(draw_labels = True, x_inline = False, y_inline = False)
ax.add_feature(cfeature.LAND.with_scale('10m'), edgecolor='grey', facecolor='none', \
    lw=1, zorder=100)

gl.right_labels = False
gl.top_labels = False
gl.xlabel_style = {'size': 16}
gl.ylabel_style = {'size': 16, 'rotation': 'vertical'}

# plot elevation (incl. colorbar)
pcm = ax.pcolormesh(bathy['lon'], bathy['lat'], bathy.values , \
    cmap=cmap, shading='auto', vmin=-vmax, vmax=vmax)

cbar = plt.colorbar(pcm, ax=ax, location = 'bottom', extend='both', \
    shrink=0.7, fraction=0.1, pad=0.05, aspect=30)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Elevation (m)', fontsize=20)

# plot grid points / land-ocean mask
lon2d, lat2d = np.meshgrid(mask['lon'].values, mask['lat'].values)
ax.scatter(lon2d[~mask.values], lat2d[~mask.values], marker='o', color=col, s=5)

# plot stations' locations
h = ax.scatter(loc['lon'].values, loc['lat'].values, marker='^', s=100, color='red')
txt = [None] * len(stat)
for i in range(len(stat)):
    s = stat[i]
    txt[i] = ax.text(loc['lon'].loc[{dim['s']:s}].values, \
        loc['lat'].loc[{dim['s']:s}].values, \
        s, \
        ha='center', va='center', fontsize=16, color='k', fontweight='bold')
adjust_text(txt, objects=h, ax=ax)

# plot size of integrated area (on sample station)
lon0 = loc['lon'].loc[{dim['s']:box_eg}].values
lat0 = loc['lat'].loc[{dim['s']:box_eg}].values

rect = patches.Rectangle((lon0-box_size, lat0-box_size), box_size*2, box_size*2, \
    linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect)

fig.savefig(figdir+figname)
