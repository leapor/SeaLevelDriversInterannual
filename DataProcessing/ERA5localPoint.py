#!/usr/bin/env python3

# Extracting ERA5 data at sea level stations' locations for variables that use pointwise data.
moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import NearestNDInterpolator

sys.path.insert(0, moduledir)
from Functions import get_config, prepare_map, select_subset_ERA5, extract_ERA5_point, \
	str_to_datetime, ERA5date_to_datetime, plot_timeseries

# -------------------------------------------------------------------------------------------------
config = get_config()
y1, y2 = config['time']['start'], config['time']['end']
lon1, lon2 = config['region']['lon']
lat1, lat2 = config['region']['lat']

var = config['drivers']['point']
vname = [config['variables'][v] for v in var]

slfile = config['dirs']['data'] + config['dirs']['ext'] + config['variables']['sl'] + '.csv'
maskfile = config['dirs']['data'] + config['dirs']['ext'] + config['variables']['mask'] + '.nc'
infile = config['dirs']['orig'] + config['data_original']['era5']

outdir = config['dirs']['data'] + config['dirs']['ext']

figdir = config['dirs']['figs'] + config['dirs']['ext']
figname1 = '[var]_Map.png'
figname2 = '[var]_Timeseries.png'

timesteps = ['1960-01-01', '1980-04-01', '2000-07-01', '2016-10-01']

proj = ccrs.PlateCarree()
# -------------------------------------------------------------------------------------------------


# --- Creating output directories if they do not already exist ---
if not os.path.exists(outdir):
	os.makedirs(outdir)
if not os.path.exists(figdir):
	os.makedirs(figdir)


# --- Get the locations of sea level stations ---
loc = pd.read_csv(slfile, index_col = 0)
coords = loc.loc[:, ['lon', 'lat']]

# --- Load the ERA5 data ---
for i in range(len(var)):
	x = extract_ERA5_point(infile.replace('[var]', var[i]), var[i], coords, maskfile, [y1, y2])
	x.to_csv(outdir+vname[i]+'.csv')
	print('extracted %s' % vname[i])

	
# ------------------------------------------------------------------------------------------------
# --------- Plotting --------- (same as in ERA5localinteg)
# ------------------------------------------------------------------------------------------------

# --- Plotting maps ---
for i in range(len(var)):
	# load data
	data1 = xr.open_dataset(infile.replace('[var]', var[i]), engine = 'netcdf4')
	data2 = pd.read_csv(outdir+vname[i]+'.csv', index_col = 'id')
	
	# prepare original
	data1 = select_subset_ERA5(data1, [lon1, lon2], [lat1, lat2])
	lon = data1.coords['longitude'].values
	lat = data1.coords['latitude'].values
	t = data1.coords['date'].values
	td = ERA5date_to_datetime(t)
	
	ind = np.isin(td, str_to_datetime(timesteps))
	td = td[ind]
	t = t[ind]
	data1 = data1.loc[{'date' : t}]
	
	# prepare extracted
	data2 = data2.loc[:, ['lon', 'lat'] + timesteps]
	
	# plot
	fig, ax = plt.subplots(2, 2, figsize = [18, 14], subplot_kw={'projection': proj}, \
		layout = 'constrained')
	ax = np.reshape(ax, -1)
	
	for ti in range(len(timesteps)):
		x1 = data1[var[i]].loc[{'date' : t[ti]}].values
		vmin, vmax = np.nanmin(x1), np.nanmax(x1)
	
		prepare_map(ax[ti], lonlim = [lon1, lon2], latlim = [lat1, lat2])
		h = ax[ti].pcolormesh(lon, lat, x1, \
			shading = 'nearest', vmin = vmin, vmax = vmax)
		ax[ti].scatter(data2.lon, data2.lat, c=data2.loc[:, timesteps[ti]].values, \
			marker = '^', norm = 'linear', vmin = vmin, vmax = vmax, \
			edgecolors = 'k', s = 200)
		ax[ti].add_feature(cfeature.LAND, edgecolor = 'gray', facecolor = 'none', \
			zorder = 100)
		
		plt.colorbar(h, ax = ax[ti], location = 'bottom')
		ax[i].set_title(timesteps[ti], fontsize = 16)
	
	fig.suptitle('Extracting '+vname[i], fontsize = 20)	
	fig.savefig(figdir+figname1.replace('[var]', vname[i]))
	plt.close(fig)
	
	
# --- Plotting time series ---
for i in range(len(var)):
	# load data
	data = pd.read_csv(outdir+vname[i]+'.csv', index_col = 'id')
	
	plot_timeseries(data, \
		figname = figdir+figname2.replace('[var]', vname[i]), \
		ylab1 = vname[i], \
		ylab2 = 'Location IDs ('+str(len(data.index))+' locations)', \
		title='Extracted '+vname[i]+' from ERA5')
