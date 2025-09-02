#!/usr/bin/env python3

# Functiones needed for the Sea level drivers experiment.

import logging
import os
import sys
import yaml
import random
import numpy as np
import pandas as pd
import xarray as xr
import math
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -----------------------------------------------------------------------------------
def get_config():
# get the name of the config file from command line
	# give warning and exit if there is no config file given
	if (len(sys.argv) != 2):
		print('ERROR: Need a configuration file!')
		quit()

	configfile = sys.argv[1]
	
	# give warning and exit if configuration file does not exist
	if not os.path.isfile(configfile):
		print('ERROR: Configuration file does not exist!')
		quit()
	
	# get config info or quit with error if config file not valid
	with open(configfile, 'r') as file:
		try:
			config = yaml.safe_load(file)
		except:
			print('ERROR: Not a valid yaml file!')
			quit()
	return config
	
	
def get_psmsl_filelist(datadir):
	metafile = 'filelist.txt'
	sep = ';'
	colnames = ['id', 'lat', 'lon', 'name', 'coast', 'code', 'flag']
	
	# load the metadata file
	meta = pd.read_csv(datadir+metafile, sep = ';', names = colnames, index_col = 'id')
	colnames = colnames[1:]
	
	# strip all values because many have whitespaces
	for c in colnames:
		if isinstance(meta[c][meta.index[0]], str):
			meta[c] = [f.strip() for f in meta[c]]
	
	# convert flag from Y/N to 1/0
	meta['flag'] = meta['flag'].map({'Y': 1, 'N': 0})
	
	return meta
	
def get_psmsl_missing_months(datadir, timespan, stations = 'all'):
	# In addition to the information from filelist.txt, it finds the number of missing
	# months from the dataset and the number of months with missing days in the timespan given
	# and returns it the extended metadata file.
	
	# dataset information
	datasub = 'data/'
	suffix = '.rlrdata'
	sep = ';'
	colnames = ['time', 'rlr', 'miss', 'flag']
	colextra = ['miss_mon', 'miss_day']  # number of missing months; number of months with days missing
	
	# get the list of stations
	meta = get_psmsl_filelist(datadir)
	
	# add the extra columns
	meta[colextra] = None
	
	# load all stations rlr data
	data = get_psmsl_data(datadir, meta.index, timespan)
	
	for i in meta.index:
		# check if the station data exists (some files are empty)
		if i not in data:
			continue
			
		# get the number of missing months in the timespan
		miss = (pd.isnull(data[i].loc[:, 'rlr']) | \
			(data[i].loc[:, 'flag'] > 0))
		days = ~pd.isnull(data[i].loc[:, 'rlr']) & \
			(data[i].loc[:, 'miss'] > 0)
		meta.loc[i, colextra] = sum(miss), sum(days)

	return meta
	
	
def get_psmsl_data(datadir, statlist = 'all', timespan = [0, 9999]):
	# loads psmsl stations contained in statlist or all of them if statlist=='all'
	# returns a dictionary, where station number is key and the whole station file is value
	# converts missing values to NaN
	# crops all time series to timespan if timespan is given, where timespan = [y1, y2]
	# pads with NaNs if the series is shorter than timespan
	
	datasub = 'data/'
	suffix = '.rlrdata'
	sep = ';'
	miss = -99999
	colnames = ['time', 'rlr', 'miss', 'flag']
	
	# time array in both the psmsl and datetime format for the desired time span
	t = np.array([round(y + (m-0.5)/12.0, 4) \
		for y in range(timespan[0], timespan[-1]+1) for m in range(1, 13)])
	year = np.floor(t).astype(int)
	month = np.round(((t-year)*12 + 0.5)).astype(int)
	td = [datetime.date(y, m, 1) for y, m in zip(year, month)]
	
	# get the list of stations
	meta = get_psmsl_filelist(datadir)
	
	# select the stations
	if isinstance(statlist, str) and (statlist == 'all'):
		statlist = meta.index
	
	# select stations from the list
	meta = meta.loc[statlist, :]
	
	# load all stations rlr data
	data = {}
	for i in statlist:
		# load station data
		x = pd.read_csv(datadir+datasub+str(i)+suffix, sep = sep, names = colnames, \
			index_col = 'time')
			
		# skip if file is empty
		if x.empty:
			continue
		
		# extract the data for the timespan if series is longer
		x = x.loc[(x.index >= t[0]) & (x.index <= t[-1]), :]
		
		# replace missing values with NaN
		x['rlr'] = x['rlr'].replace({miss : np.nan})
		
		# extend all time series to the full timespan if the series is shorter
		x = x.reindex(index = t)
		
		# replace index with datetime index
		x.index = td
		
		# add to dictionary
		data[i] = x
		
	return data


def plot_missing_data(ax, meta, x1, x2, b1 = 3, b2 = 6, labels1 = None, labels2 = None, \
	write_stat_id = True):
	# Adds values x1 as triangle up sorted into bins with b1 borders, and values x2 as triangle
	# down sorted into bins with b2 borders to ax (which is a map).
	# 1 is supposed to be missing months in a dataset, and 2 months with some data missing.
	# Dataset is assumed to have more months where some data is missing (2) than fully missing
	# months.
	# Values for lon, lat and station number are in the meta dataframe.
	# If b1 and b2 are given then labels need to be given too!!!
	
	from adjustText import adjust_text
	
	# calculating bounds if necessary
	# ny is the number of years if given or None if bounds are given
	if isinstance(b1, int):
		ny1 = b1
		b1 = [0] + [12*y+1 for y in range(b1+1)]
	else:
		ny1 = None
	if isinstance(b2, int):
		ny2 = b2 
		b2 = [0] + [12*y+1 for y in range(b2+1)]
	else:
		ny2 = None
	
	# preparing to plot
	cm1 = plt.get_cmap('jet', len(b1))
	cm2 = plt.get_cmap('jet', len(b2))
	
	norm1 = mcolors.BoundaryNorm(boundaries=b1, ncolors=len(b1), extend = 'max')
	norm2 = mcolors.BoundaryNorm(boundaries=b2, ncolors = len(b2), extend = 'max')
	
	ticks1 = [(b1[i]+b1[i+1])/2 for i in range(len(b1)-1)] + [b1[-1]]
	ticks2 = [(b2[0]+b2[1])/2] + [b for b in b2[2:]]
	
	# calculate labels if they are not given
	# if bounds are default labels are in years
	# if bounds are provided and labels are not given labels are same as bounds and colorbar
	# label is given in months, not years
	if (labels1 == None):
		if isinstance(ny1, int):
			labels1 = ['0', '< 1 yr'] + \
				[str(y)+'-'+str(y+1)+' yr' for y in range(1, ny1)] + \
				['> '+str(ny1)+' yr']
		else:
			labels1 = b1
	
	if (labels2 == None):
		if isinstance(ny2, int):
			labels2 = ['0'] + ['< '+str(y)+' yr' for y in range(1, ny2+1)]
		else:
			labels2 = b2
	
	
	# plot invisible diamonds to be used for aligning station labels
	h0 = ax.scatter(meta.lon, meta.lat, s=150, c = 'white', marker = 'd')
	
	# plot data
	h1 = ax.scatter(meta.lon, meta.lat, s=100, c = x1, norm = norm1, marker = 10, \
		cmap = cm1, label = 'missing months')
	h2 = ax.scatter(meta.lon, meta.lat, s=100, c = x2, norm = norm2, marker = 11, \
		cmap = cm2, label = 'months with missing days')
		
	# add station IDs
	if write_stat_id:
		texts = [None] * len(meta.index)
		for i, ii in zip(meta.index, range(len(meta.index))):
			texts[ii] = ax.text(meta.loc[i, 'lon'], meta.loc[i, 'lat'], i, \
				ha = 'center', va = 'center', fontsize = 'small', \
				color = 'darkgreen', \
				bbox=dict(facecolor='white', alpha=0.3, edgecolor='white'))
		adjust_text(texts, objects = h0, ax = ax)
		return h1, h2, ticks1, ticks2, labels1, labels2, texts
	else:
		return h1, h2, ticks1, ticks2, labels1, labels2
	
	
def prepare_map(ax, lonlim = [0,359], latlim = [-90,90]):
	import cartopy.crs as ccrs
	import cartopy.feature as cfeature

	if isinstance(lonlim, np.ndarray): lonlim = list(lonlim)
	if isinstance(latlim, np.ndarray): latlim = list(latlim)
	
	ax.set_extent(lonlim + latlim, crs = ccrs.PlateCarree())
	gl = ax.gridlines(draw_labels = True, \
		x_inline = False, y_inline = False)
	ax.add_feature(cfeature.LAND, edgecolor = None, facecolor = 'grey', \
		alpha = 0.4)
	gl.right_labels = False
	gl.top_labels = False
	
	return
	
def draw_regions(reg, ax, colors = None):
	# Drawing region borders as rectangles on a map.
	regions = reg.index.values
	
	if colors == None:
		colors = list(mcolors.TABLEAU_COLORS.keys())
	elif isinstance(colors, str):
		colors = [colors]*len(regions)
	
	for r, c in zip(regions, colors):	
		ax.add_patch(mpl.patches.Rectangle((reg.loc[r, 'lon1'], reg.loc[r, 'lat1']), \
			reg.loc[r, 'lon2']-reg.loc[r, 'lon1'], \
			reg.loc[r, 'lat2']-reg.loc[r, 'lat1'], \
			lw = 3, fill = False, edgecolor = c))
	
	return
	

def extract_ERA5_point(file, var, coords, mask, timespan = None):
	# Extract the values from an ERA5 file for variable var at coordinates coord, using
	# nearest neighbor interpolation and only sea grid points.
	# coords is an n x 2 pd dataframe, with station id as index, and lon and lat as columns.
	# Data is extracted for n points at locations identified with id.
	# Returns a pandas dataframe with locations as rows and time steps as columns.
	# File mask contains the land ocean mask needed to extract only from sea points.
	
	from scipy.interpolate import NearestNDInterpolator
	
	# load the mask
	mask = xr.open_dataset(mask, engine = 'netcdf4')
	
	# load the dataset
	data = xr.open_dataset(file, engine = 'netcdf4')
		
	# reduce the area for faster interpolation and select timespan if given
	lon1 = np.amin(coords.loc[:, 'lon'])-1
	lon2 = np.amax(coords.loc[:, 'lon'])+1
	lat1 = np.amin(coords.loc[:, 'lat'])-1
	lat2 = np.amax(coords.loc[:, 'lat'])+1
	
	data = select_subset_ERA5(data, [lon1, lon2], [lat1, lat2], timespan)
	mask = select_subset_ERA5(mask, [lon1, lon2], [lat1, lat2])
	
	# get coordinates
	lon = data.coords['longitude'].values
	lat = data.coords['latitude'].values
	t = data.coords['date'].values
	td = ERA5date_to_datetime(t)
	
	# remove land points
	mask = xr.where(mask, mask, np.nan)
	data = data * mask.rename({'mask' : var})
	
	# create an output pandas dataframe
	xi = pd.DataFrame(columns = td, index = coords.index)
	xi = pd.concat([coords, xi], axis = 1)
	
	# extract the data
	for i in range(len(t)):
		# get data for one time step
		x = data[var].loc[{'date' : t[i]}].values
		
		# convert data to 1D
		x = np.reshape(x, -1)
		
		# convert lon and lat to 1D + find the location of sea points
		if (i == 0):
			# reshape lon and lat
			lon1, lat1 = np.meshgrid(lon, lat)
			lon1 = np.reshape(lon1, -1)
			lat1 = np.reshape(lat1, -1)
		
			# find sea grid points
			sea = ~np.isnan(x)
			
			# remove land grid points from lon and lat
			lon1 = lon1[sea]
			lat1 = lat1[sea]
			coords1 = np.stack((lon1, lat1), axis = 1)
		
		# remove missing grid points from the time step
		x = x[sea]
		
		# interpolation
		interp = NearestNDInterpolator(coords1, x)
		xi.loc[:, td[i]] = interp(coords.values)
	
	return xi


def extract_ERA5_integ(file, var, coords, box, timespan = None):
	# Extracts the area integrated values from an ERA5 file for variable var at 
	# coordinates coord. Integrates over a box with box degrees in each direction.
	# coords is an n x 2 pd dataframe, with station id as index, and lon and lat as columns.
	# Data is extracted for n points at locations identified with id.
	# Returns a pandas dataframe with locations as rows and time steps as columns.
	
	# load the dataset
	data = xr.open_dataset(file, engine = 'netcdf4')
	
	# reduce the area for faster interpolation and select timespan if given
	lon1 = np.amin(coords.loc[:, 'lon'])-box-1
	lon2 = np.amax(coords.loc[:, 'lon'])+box+1
	lat1 = np.amin(coords.loc[:, 'lat'])-box-1
	lat2 = np.amax(coords.loc[:, 'lat'])+box+1
	
	data = select_subset_ERA5(data, [lon1, lon2], [lat1, lat2], timespan)
	
	# get coordinates
	lon = data.coords['longitude'].values
	lat = data.coords['latitude'].values
	t = data.coords['date'].values
	td = ERA5date_to_datetime(t)
	
	# create an output pandas dataframe
	xi = pd.DataFrame(columns = td, index = coords.index)
	xi = pd.concat([coords, xi], axis = 1)
	
	# extract the data
	for s in coords.index:
		# get the data from a box around the station
		lon1 = coords.loc[s, 'lon'] - box
		lon2 = coords.loc[s, 'lon'] + box
		lat1 = coords.loc[s, 'lat'] - box
		lat2 = coords.loc[s, 'lat'] + box
		x = select_subset_ERA5(data, [lon1, lon2], [lat1, lat2])
		
		
		# calculate the sum
		xi.loc[s, td] = x[var].sum(dim = ['longitude', 'latitude'], skipna = True).values
		
	return xi

def ERA5date_to_datetime(t):
	# Converts the date from an ERA5 netcdf file (integers yyyymmdd) to datetime.date.
	td = np.empty(len(t), dtype = datetime.date)
	for i in range(len(t)):
		year = np.floor(t[i]/10000).astype(int)
		month = np.floor((t[i]-year*10000)/100).astype(int)
		day = np.floor(t[i]-year*10000-month*100).astype(int)
		td[i] = datetime.date(year, month, day)
		
	return td
	
def str_to_datetime(t):
	# Converts an array of dates in string format to an array of datetime dates.
	td = [None]*len(t)
	for i in range(len(t)):
		td[i] = datetime.datetime.strptime(t[i], '%Y-%m-%d').date()
		
	return td
	
def npdatetime64_to_datetime(t):
	# Converts np.datetime64 dates to datetime.date.
	years = t.astype('datetime64[Y]').astype(int)+1970
	months = t.astype('datetime64[M]').astype(int) % 12 + 1
	days = (t - t.astype('datetime64[M]')).astype(int) + 1
	
	td = np.empty(len(t), dtype = datetime.date)
	for i in range(len(t)):
		td[i] = datetime.date(years[i], months[i], days[i])
		
	return td
	
def plot_timeseries(data, figname, figsize = None, ylab1 = '', ylab2 = '', title = ''):
	# Plotting time series contained in pandas dataframe data, where there are some columns of
	# metadata at the beginning and then each column represents one time step. Each row is one
	# time series.
	# Optionally plots a second dataset on the same figure using a secondary axis.	
	
	# extract the time series if there is extra metadata
	t = list(data.columns)
	t = [t0 for t0 in t if t0[0].isdigit()]
	x = data[t]
	t = str_to_datetime(t)
	n = len(x.index)   # number of series
	
	# prepare the figure
	if figsize == None:
		fh = 2*n
		if (n > 12): fh = 24
		if (n <= 2): fh = 5
		figsize = [14, fh]
	fig, ax = plt.subplots(n, 1, figsize = figsize)
	plt.subplots_adjust(left=0.07, right=0.9, top=0.95, bottom=0.05, hspace = 0.0)
	
	# plot timeseries
	for i in range(n):
		ax[i].plot(t, x.iloc[i, :])
		ax[i].set_xlim([t[0], t[-1]])
		if (i != n-1):
			ax[i].xaxis.set_tick_params(labelbottom = False)
		else:
			ax[i].xaxis.set_tick_params(labelsize = 18)
		ax[i].yaxis.set_label_position('right')
		ax[i].set_ylabel(data.index[i], rotation = 0, fontsize = 18, \
			ha = 'left', va = 'center')
	
	# formatting
	fig.text(0.01, 0.5, ylab1, ha = 'left', va = 'center', fontsize = 24, rotation = 'vertical')
	fig.text(0.99, 0.5, ylab2, ha = 'right', va = 'center', fontsize = 24, rotation = 'vertical')
	
	# save figure
	fig.suptitle(title, fontsize = 24)
	fig.savefig(figname)
	plt.close(fig)
	return
	
def plot_2_timeseries(data1, data2, figname, figsize = None, ylab1 = '', ylab2 = '', title = '', \
	sharey = False):
	# Plotting the time series contained in data1 and data2, which need to have the same format
	# (same number of columns, though the length can differ).
	# Time steps are in rows.
	
	col = ['maroon', 'teal']
	
	# timestamps and column names
	t1 = data1.index
	t2 = data2.index
	name = data1.columns
	n = len(name)
	
	# prepare the figure
	if figsize == None:
		fh = 2*n
		if (n > 12): fh = 24
		if (n < 2) : fh = 5
		figsize = [15, fh]
	fig, ax = plt.subplots(n, 1, figsize = figsize)
	plt.subplots_adjust(left=0.07, right = 0.85, top = 0.95, bottom = 0.05, hspace = 0.0)
	
	# plot timeseries
	for i in range(n):
		ax2 = ax[i].twinx()
	
		ax[i].plot(t1, data1.loc[:, name[i]], color = col[0])
		ax2.plot(t2, data2.loc[:, name[i]], color = col[1], alpha = 0.8)
		
		ax[i].set_xlim([min(t1[0], t2[0]), max(t1[-1], t2[-1])])
		if (i != n-1):
			ax[i].xaxis.set_tick_params(labelbottom = False)
		else:
			ax[i].xaxis.set_tick_params(labelsize = 18)
		ax2.set_ylabel(name[i].replace('_', '\n'), \
			rotation = 0, fontsize = 18, ha = 'left', va = 'center')
			
		# set yrange if sharey = True
		if sharey:
			ax2.set_ylim(ax[i].get_ylim())
			
	# formatting
	fig.text(0.01, 0.5, ylab1, ha = 'left', va = 'center', fontsize = 24, rotation = 'vertical')
	fig.text(0.87, 0.96, ylab2, ha = 'left', va = 'top', fontsize = 24)
	
	# save figure
	fig.suptitle(title, fontsize = 24)
	fig.savefig(figname)
	plt.close(fig)
	return
	
	
def select_subset_ERA5(x, lonlim = None, latlim = None, ylim = None):
	# Selecting a subset of a larger ERA5 file based on lon, lat and time limits.
	# ylim = (y1, y2)
	lon = x.coords['longitude'].values
	lat = x.coords['latitude'].values
	try:  # in case the variable does not have time coordinate
		t = x.coords['date'].values
		td = ERA5date_to_datetime(t)
	except:
		t = None
	
	if (lonlim is not None):
		ind = (x.coords['longitude'] >= lonlim[0]) & (x.coords['longitude'] <= lonlim[-1])
		lon = lon[ind]
	if (latlim is not None):
		ind = (x.coords['latitude'] >= latlim[0]) & (x.coords['latitude'] <= latlim[-1])
		lat = lat[ind]
	if (ylim is not None) and (t is not None):
		ind = (td >= datetime.date(ylim[0], 1, 1)) & (td <= datetime.date(ylim[-1], 12, 1))
		t = t[ind]
		td = td[ind]
	
	if (t is not None):	
		x = x.loc[{'longitude' : lon, 'latitude' : lat, 'date' : t}]
	else:
		x = x.loc[{'longitude' : lon, 'latitude' : lat}]
	return x
	
def detrend(data, plot_trend = False):
	# detrends data along axis = 0; data may contain NaNs
	temp = data.copy(deep = True)
	x0 = np.arange(temp.shape[0], dtype = float)
	trend = data.copy(deep = True)
	
	for i in range(temp.shape[1]):
		# extract time series
		y = data.iloc[:,i].values
		x = np.copy(x0)
		x[np.isnan(y)] = np.nan
		n = sum(~np.isnan(y))
		
		# calculate trend
		a = (np.nansum(y)*np.nansum(x*x) - np.nansum(x)*np.nansum(x*y)) / \
			(n*np.nansum(x*x) - np.nansum(x)*np.nansum(x))
		b = (n*np.nansum(x*y) - np.nansum(x)*np.nansum(y)) / \
			(n*np.nansum(x*x) - np.nansum(x)*np.nansum(x))
		trend.iloc[:,i] = a + b*x
		
		# remove trend
		temp.iloc[:,i] = y - trend.iloc[:,i].values
		
	# plot trend if needed
	if type(plot_trend) is str:
		plot_2_timeseries(data, trend, figname = plot_trend, sharey = True)	
	
	return temp
	
	
	
def deseason(data, excl = None, plot_cycle = None):
	# deseason data, where rows in data are time steps and use a datetime index
	x = data.copy(deep = True)
	
	# if timespan is given(as [y1, y2]) skip those years when calculating the seasonal params
	if excl is not None:
		xs = x.loc[~((x.index.year >= excl[0]) & (x.index.year <= excl[-1])), :]
	else:
		xs = x.copy(deep = True)
	
	smean = np.empty((data.shape[1], 12))
	sstd = np.empty((data.shape[1], 12))
	for m in range(1, 13):
		ind = (x.index.month == m)
		inds = (xs.index.month == m)
		smean[m-1, :] = xs.loc[inds, :].mean()
		sstd[m-1, :] = xs.loc[inds, :].std()
		
		x.loc[ind, :] = x.loc[ind, :].sub(smean[m-1, :], axis = 1)
		x.loc[ind, :] = x.loc[ind, :].div(sstd[m-1, :], axis = 1)
	
	# plot the seasonal cycle if plot_cycle is given (the figs are saved to plot_cycle dir)	
	if type(plot_cycle) is str:
		months = list(pd.date_range(start='2024-01-01', periods=12, \
			freq='ME').strftime('%b'))
	
		n = x.shape[1]
		name = x.columns
		fig, ax = plt.subplots(n, figsize = [10, 18], sharex = True)
		plt.subplots_adjust(left=0.07, right = 0.75, top = 0.95, bottom = 0.05, hspace = 0.0)
		for i in range(n):
			ax2 = ax[i].twinx()
			
			h1, = ax[i].plot(range(1, 13), smean[:, i], color = 'maroon', label = 'mean')
			h2, = ax2.plot(range(1, 13), sstd[:, i], color = 'teal', label = 'std')
			
			ax2.set_ylabel(name[i].replace('_', '\n'), \
				rotation = 0, fontsize = 18, ha = 'left', va = 'center')
			ax[i].set_xlim([1, 12])
			ax[i].set_xticks(ticks = range(1, 13), labels = months, fontsize = 18)
			ax[i].grid()
		ax[0].legend(handles = [h1, h2], loc = 'upper center', fontsize = 18, ncols = 2)
		fig.savefig(plot_cycle)
		plt.close(fig)
	return x
	
	
def split_data(data, subset = None, t = None):
	# Split the data by selecting a subset of it.
	# Returns x1 (main part of the dataset without the removed subset) and x2 (extracted subset).
	# Subset can be given as:
	#  - a list or tupple of integers with length 2 - start and end year of the subset
	#  - an integer - length of the subset in years; exact time span to be selected randomly
	#  - None/nothing - length of the subset is 10% of the dataset length (rounded to full
	#    years), exact time span is selected randomly
	# data can be either a dataset or a numpy array, in which case a time array needs to also
	# be given and the first dimension (zero-th) needs to be time
	
	# if time is not given it is taken from the index of the dataframe
	if t is None:
		t = data.index.date
	
	## start and end of the whole dataset
	wy1, wy2 = t[0].year, t[-1].year
	
	# finding the start and the end year of the subset if necessary
	if (type(subset) is list) or (type(subset) is tuple):
		y1, y2 = subset
	else:
		if type(subset) is int:
			ny = subset
		else:
			ny = int(round(0.1 * len(data.index)/12))
		
		y1 = random.randint(wy1, wy2-ny+1)
		y2 = y1 + ny - 1
		subset = [2011, 2016]
	
	# finding the indices of the subset
	ind = (t >= datetime.date(y1, 1, 1)) & (t <= datetime.date(y2, 12, 31))
	
	# splitting the data
	if isinstance(data, pd.DataFrame):
		# pandas dataframe
		train = data.loc[~ind, :].copy(deep = True)
		test = data.loc[ind, :].copy(deep = True)
	else:
		# list with 2 numpy arrays (for X and y)
		X, y = data
		train = [X[~ind], y[~ind], t[~ind]]
		test = [X[ind], y[ind], t[ind]]
	return train, test
	

def create_sequence(X, y, ns, t = None):
	# convert data from shape (n x nf) to shape (ns x n X nf)
	# ns = number of time steps in a sequence
	# n = total number of time steps (samples)
	# nf = number of features
	n, nf = np.shape(X)
	
	x = np.empty((n-ns+1, ns, nf))
	for i in range(ns):
		x[:, i, :] = X[i : n-ns+i+1, :]
	y = y[ns-1:]
	
	if t is not None:
		t = t[ns-1:]
		return x, y, t
	else:
		return x, y
	
def model_builder_ANN(layers, units, activation, learning_rate, loss_fun):
	# builds and compiles a feed-forward neural network model
	
	logging.disable(logging.WARNING)
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense
	from tensorflow.keras.optimizers import Adam
	
	model = Sequential()
	for i in range(layers):
		model.add(Dense(units = units, activation = activation))
	model.add(Dense(units = 1, activation = None))

	model.compile(optimizer = Adam(learning_rate = learning_rate), loss = loss_fun)
	return model
	
def model_builder_LSTM(units, activation, learning_rate, loss_fun):
	# builds and compiles an LSTM neural network model with 2 layers
	# ns = length of sequence
	
	logging.disable(logging.WARNING)
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense, LSTM
	from tensorflow.keras.optimizers import Adam
	
	model = Sequential()
	model.add(LSTM(units = units, activation = activation, return_sequences = True, \
		go_backwards = False))
	model.add(LSTM(units = units, activation = activation, return_sequences = False, \
		go_backwards = False))
	model.add(Dense(units = 1, activation = None))
	
	model.compile(optimizer = Adam(learning_rate = learning_rate), loss = loss_fun)
	return model

def RMSE(y_true, y_pred):
    # root mean squared error
    loss = np.sqrt(np.nanmean(np.square(y_true - y_pred)))
    return loss
	
def RRMSE(y_true, y_pred):
	# root mean squared error (relative)
	loss = np.sqrt(np.nanmean(np.square(y_true - y_pred))/np.nansum(y_pred*y_pred))*100
	return loss

	
def relative_explained_variance(y_true, y_pred):
	x = np.nanvar(y_true) - np.nanvar(y_true - y_pred)
	x = x/np.nanvar(y_true)*100
	return x
	
def corr2(y_true, y_pred):
	# relative explained variance defined as correlation squared multiplied by 100
	nan = (np.isnan(y_true) | np.isnan(y_pred))
	yt = y_true[~nan]
	yp = y_pred[~nan]
	
	x = np.corrcoef(yt, yp)
	x = x[0,1]
	x = x**2 * 100
	return x
	
def sort_ensemble(key, *var, reverse=True):
    # sorting one or multiple DataArrays given in *var based on the DataArray key
    # along the last coordinate
    # All DataArrays need to have model run coordinate as last!!!
    # By default it sorts the values from lowest to highest!
    
    idx = np.argsort(key, axis=-1)
    if reverse:
        idx = np.flip(idx, axis=-1)
    var_sorted = [np.take_along_axis(v, idx, axis=-1) for v in var]
    return (idx, *var_sorted)
    
def extract_xr_by_coord_index(data, ind0, dim):
    # Given xarray of indices (obtained by np.argwhere or similar) extract those
    # from a DataArray or Dataset.
    #
    # data = dataset or dataarray
    # ind0 = dataarray with indices of the ensemble median members
    # dim = dimension for which to extract by index
    
    # temporarily convert to Dataset if data is DataArray
    if isinstance(data, xr.DataArray):
        data = data.to_dataset()
        isarray = True
    else:
        isarray = False
    var = list(data.data_vars)
    
    # loop through all variables in the dataset
    for v in var:
        dims = list(data[v].dims)
        
        # skip if variable does not have ensemble dimension
        if dim not in dims: continue
        
        # transpose data var to have dim at the end
        dims = [d for d in dims if d!=dim]  # data dims without ens dim
        data[v] = data[v].transpose(*dims, dim)
        
        # transpose index to have the same dims as data var (use ind as temp variable)
        ind = ind0.expand_dims(dim=[d for d in dims+[dim] if d not in ind0.dims])
        ind = ind.transpose(*dims, dim)
        
        # select the indices along all axes (has to be on numpy)
        temp = np.take_along_axis(data[v].values, ind.values, axis=-1)
        
        # convert back to xarray
        temp = xr.DataArray(data=temp, coords=data[v].isel({dim:[0]}).coords) 
        data[v] = temp.squeeze(drop=True)
    
    # if data was DataArray convert back to DataArray
    if isarray:
        data=data[var[0]]
    
    return data


def convert_to_marker_size(data, lims=[1,300], legend_vals=[], reverse=False):
    import matplotlib.ticker as mticker

    # remove outliers (rare extreme values) from the dataset
    lim = data.quantile(0.98).values
    data = xr.where(data <= lim, data, lim)
    
    # find min and max of the dataset
    vmin = data.min().values
    vmax = data.max().values
    
    # create legend labels (if not given)
    if len(legend_vals)==0:
        locator = mticker.MaxNLocator(integer=False, prune='both')
        vals = locator.tick_values(vmin, vmax)
    else:
        vals = legend_vals
    
    # convert dataset and legend sizes
    dataout = (data - vmin) / (vmax - vmin) # scales to 0-1
    sizes = (vals - vmin) / (vmax - vmin)
    
    if reverse: # reverses the order if needed
        dataout = 1 - dataout
        sizes = 1 - sizes
    
    dataout = dataout * (lims[1] - lims[0]) + lims[0] # scales to lims
    sizes = sizes * (lims[1] - lims[0]) + lims[0]
    
    return dataout, [vals, sizes]
	
def ensembleLinReg(data, N, valy, store_models = False, plot_res = False, verbose = True):
	# Fits an ensemble of N linear regression models using a subset of data (excluding a
	# random segment of data with length valy years).
	
	from sklearn.linear_model import LinearRegression
	import joblib
		
	# prepare data
	data = data.dropna(axis = 0, inplace = False)
	features = list(data.columns)[1:]
	
	X = data.iloc[:, 1:].values
	y = data.iloc[:, 0].values
	t = data.index.date
	
	# output file(s)
	res = pd.DataFrame(index = range(1, N+1), \
		columns = features + ['intercept', 'RMSE', 'ExpVar','ValStart','ValEnd'])
		
	tval = np.empty((valy*12, N), dtype = datetime.date)
	ytrue = np.full((valy*12, N), np.nan)
	ypred = np.full((valy*12, N), np.nan)
		
	# train ensemble
	for i in range(N):
		# split the data into train and val
		[Xtr, ytr, ttr], [Xva, yva, tva] = split_data([X, y], subset = valy, t = t)
		nval = len(tva)
		res.loc[i+1, 'ValStart'] = tva[0].year
		res.loc[i+1, 'ValEnd'] = tva[-1].year
		
		# build model
		model = LinearRegression()
		
		# fit model
		model.fit(Xtr, ytr)
		res.loc[i+1, features] = model.coef_
		res.loc[i+1, 'intercept'] = model.intercept_
		
		# predict val set
		tval[:nval,i] = tva
		ytrue[:nval,i] = yva
		ypred[:nval,i] = model.predict(Xva)
		
		# calculate metrics
		res.loc[i+1, 'RRMSE'] = RRMSE(ytrue[:,i], ypred[:,i])
		res.loc[i+1, 'ExpVar'] = relative_explained_variance(ytrue[:,i], ypred[:,i])
		
		# save model if needed
		if isinstance(store_models, str):
			joblib.dump(model, store_models.format(i+1))
			
		# print
		if verbose:
			print('  -- model %2i/%2i trained' % (i+1, N))
			
			
	# --- Plot figures if needed ---
	if isinstance(plot_res, str):
		fig, ax = plt.subplots(min(N, 12), 1, figsize = [12, 18], sharey = True)
		plt.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05, hspace=0.22)
			
		for i in range(min(N, 12)):
			h1, = ax[i].plot(tval[:,i], ytrue[:,i], color = 'teal', label = 'true')
			h2, = ax[i].plot(tval[:,i], ypred[:,i], color = 'maroon', label = 'pred')
			ax[i].set_xlim([tval[0, i], tval[-1, i]])
			ax[i].text(0.01, 0.95, str(i+1), ha = 'left', va = 'top', fontsize = 16, \
				transform = ax[i].transAxes)
				
		ax[0].legend(handles = [h1, h2], loc = 'upper center', fontsize = 16, ncol = 2)
		fig.suptitle('Prediction of the validation set', fontsize = 24, y = 0.98)
		fig.savefig(plot_res)
		plt.close(fig)
		
	return res
		
	
	
def ensembleANN(data, N, hyperparams, fitting_params, store_models = False, \
	save_hist = False, plot_res = False, plot_hist = False, verbose = True):
	
	# Runs an ensemble of N feed-forward ANNs with hyperparameters given in dictionary
	# hyperparams and fitting parameters given in fitting_params. The named variables
	# should contain the location for storing data and figures, if storage and plotting
	# are desired. Files that exist for every ensemble member should include placeholders
	# for ensemble member number ({:02n}). Prepares the data for the ensemble as well.
	# Returns validation RMSE and explained variances in a dataframe.
	#
	# data = pandas dataframe with data, where 1st column is target and index is time
	# N = number of ensemble members
	# hyperparams (needed for the model_builder function):
	#   - layers = number of Dense layers
	#   - units = number of units in each layer (same for every layer)
	#   - activation = activation function (same for every layer; str)
	#   - learning_rate = starting learning rate for the Adam optimizer
	#   - loss_fun = loss function to be used for training (str or function)
	# fitting_params (needed for fitting):
	#   - valy = number of years to be used for the validation set.
	#   - epochs = number of epochs (maximum, early stopping will be used)
	#   - batch_size (batch size for each model run)
	# named variables:
	#   - store_models = path to store the trained models (.pkl, incl. member placeholder)
	#   - save_hist = path to file for storing training and val loss during training (.csv)
	#   - plot_res = path to validation prediction figure (.png)
	#   - plot_hist = path to training loss history figure (.png)
	#   - verbose = whether to print which model is being run
	#
	# Note: It cannot plot more than 12 ensemble members. If there are more, both plots are
	# done for the first 12.
	
	logging.disable(logging.WARNING)
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	from keras.callbacks import EarlyStopping
	import joblib
	
	# prepare data
	data = data.dropna(axis = 0, inplace = False)
	valy = fitting_params['valy']
	
	# separate features and target
	X = data.iloc[:, 1:].values
	y = data.iloc[:, 0].values
	t = data.index.date
	
	# prepare for training
	es = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 0, \
		restore_best_weights = True)
		
	columns = [str(i+1)+'_loss' for i in range(N)] + [str(i+1)+'_val_loss' for i in range(N)]
	hist = pd.DataFrame(index = range(1,fitting_params['epochs']+1), \
		columns = columns)
	tval = np.empty((valy*12, N), dtype = datetime.date)
	ytrue = np.empty((valy*12, N)); ytrue[:] = np.nan
	ypred = np.empty((valy*12, N)); ypred[:] = np.nan
	
	metrics = pd.DataFrame(index = range(N), columns = ['RMSE', 'ExpVar','ValStart','ValEnd'])
	
	# --- Train ensemble ---
	for i in range(N):
		# train - val split
		[Xtr, ytr, ttr], [Xva, yva, tva] = split_data([X, y], subset = valy, t = t)
		nval = len(tva)
		metrics.loc[i, 'ValStart'] = tva[0].year
		metrics.loc[i, 'ValEnd'] = tva[-1].year
	
		# build model
		model = model_builder_ANN(**hyperparams)
	
		# fit model
		history = model.fit(Xtr, ytr, epochs = fitting_params['epochs'], \
			batch_size = fitting_params['batch_size'], validation_data = (Xva, yva), \
			verbose = 0, callbacks = [es])
		
		# store history
		lt = len(history.history['loss'])
		hist.loc[1:lt, str(i+1)+'_loss'] = history.history['loss']
		hist.loc[1:lt, str(i+1)+'_val_loss'] = history.history['val_loss']
		
		# predict val set
		tval[:nval,i] = tva
		ytrue[:nval,i] = yva
		ypred[:nval,i] = np.squeeze(model.predict(Xva, verbose = 0))
		
		# calculate metrics
		metrics.loc[i, 'RMSE'] = RRMSE(ytrue[:,i], ypred[:,i])
		metrics.loc[i, 'ExpVar'] = relative_explained_variance(ytrue[:,i], ypred[:,i])
		
		# save model if needed
		if isinstance(store_models, str):
			joblib.dump(model, store_models.format(i))
			
		# print
		if verbose:
			print('  -- model %2i/%2i trained' % (i+1, N))
				
	
	# --- Save history if needed ---
	if isinstance(save_hist, str):
		hist.to_csv(save_hist)
		
		
	# --- Plot figures if needed ---
	# validation set results
	if isinstance(plot_res, str):
		fig, ax = plt.subplots(min(N, 12), 1, figsize = [12, 18], sharey = True)
		plt.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05, hspace=0.22)
			
		for i in range(min(N, 12)):
			h1, = ax[i].plot(tval[:,i], ytrue[:,i], color = 'teal', label = 'true')
			h2, = ax[i].plot(tval[:,i], ypred[:,i], color = 'maroon', label = 'pred')
			ax[i].set_xlim([tval[0, i], tval[-1, i]])
			ax[i].text(0.01, 0.95, str(i+1), ha = 'left', va = 'top', fontsize = 16, \
				transform = ax[i].transAxes)
				
		ax[0].legend(handles = [h1, h2], loc = 'upper center', fontsize = 16, ncol = 2)
		fig.suptitle('Prediction of the validation set', fontsize = 24, y = 0.98)
		fig.savefig(plot_res)
		plt.close(fig)
		
	# training history
	if isinstance(plot_hist, str):
		fig, ax = plt.subplots(4, 3, figsize = [14, 18], sharex = True, sharey = True)
		plt.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05, hspace=0.0, \
			wspace = 0.0)
		ax = np.reshape(ax, -1)
		
		for i in range(min(N, 12)):
			#lenhist = sum(~np.isnan(hist.loc[:,str(i+1)+'_loss'].values))
			ax[i].plot(hist.index, hist.loc[:,str(i+1)+'_loss'].values, \
				color = 'teal', label = 'loss')
			ax[i].plot(hist.index, hist.loc[:,str(i+1)+'_val_loss'].values, \
				color = 'maroon', label = 'val_loss')
			ax[i].text(0.02, 0.98, str(i+1), ha = 'left', va = 'top', fontsize = 16, \
				transform = ax[i].transAxes)
				
		fig.text(0.5, 0.01, 'Epoch', fontsize = 20, ha = 'center', va = 'bottom')
		fig.text(0.01, 0.5, 'Loss', fontsize = 20, ha = 'left', va = 'center', \
			rotation = 'vertical')
		ax[1].legend(loc = 'upper center', fontsize = 16)
		fig.suptitle('Loss ('+hyperparams['loss_fun']+') during training', \
			fontsize = 24, y = 0.98)
		fig.savefig(plot_hist)
		plt.close(fig)
	
	return metrics
	
	
def ensembleLSTM(data, N, hyperparams, fitting_params, store_models = False, \
	save_hist = False, plot_res = False, plot_hist = False, verbose = True):
	
	# Runs an ensemble of N LSTM NNs with hyperparameters given in dictionary
	# hyperparams and fitting parameters given in fitting_params. The named variables
	# should contain the location for storing data and figures, if storage and plotting
	# are desired. Files that exist for every ensemble member should include placeholders
	# for ensemble member number ({:02n}). Prepares the data for the ensemble as well.
	# Returns validation RMSE and explained variances in a dataframe.
	#
	# data = pandas dataframe with data, where 1st column is target and index is time
	# N = number of ensemble members
	# hyperparams (needed for the model_builder function):
	#   - units = number of units in each layer (same for every layer)
	#   - activation = activation function (same for every layer; str)
	#   - learning_rate = starting learning rate for the Adam optimizer
	#   - loss_fun = loss function to be used for training (str or function)
	# fitting_params (needed for fitting):
	#   - valy = number of years to be used for the validation set.
	#   - epochs = number of epochs (maximum, early stopping will be used)
	#   - batch_size = batch size for each model run
	#   - sequence_len = sequence length, including current time step
	# named variables:
	#   - store_models = path to store the trained models (.pkl, incl. member placeholder)
	#   - save_hist = path to file for storing training and val loss during training (.csv)
	#   - plot_res = path to validation prediction figure (.png)
	#   - plot_hist = path to training loss history figure (.png)
	#   - verbose = whether to print which model is being run
	#
	# Note: It cannot plot more than 12 ensemble members. If there are more, both plots are
	# done for the first 12.
	
	logging.disable(logging.WARNING)
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	from keras.callbacks import EarlyStopping
	import joblib
	
	# extract some parameters
	valy = fitting_params['valy']
	ns = fitting_params['sequence_len']
	
	# separate features and target
	X = data.iloc[:, 1:].values
	y = data.iloc[:, 0].values
	t = data.index.date
		
	# reshape data for LSTM
	X, y, t = create_sequence(X, y, ns, t = t)
	
	# remove missing data
	ind = ~np.isnan(y)
	X = X[ind]
	y = y[ind]
	t = t[ind]
	
	# prepare for training
	es = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 0, \
		restore_best_weights = True)
		
	columns = [str(i+1)+'_loss' for i in range(N)] + [str(i+1)+'_val_loss' for i in range(N)]
	hist = pd.DataFrame(index = range(1,fitting_params['epochs']+1), \
		columns = columns)
	tval = np.empty((valy*12, N), dtype = datetime.date)
	ytrue = np.empty((valy*12, N)); ytrue[:] = np.nan
	ypred = np.empty((valy*12, N)); ypred[:] = np.nan
	
	metrics = pd.DataFrame(index = range(N), columns = ['RRMSE','ExpVar','ValStart','ValEnd'])
	
	# --- Train ensemble ---
	for i in range(N):
		# train - val split
		[Xtr, ytr, ttr], [Xva, yva, tva] = split_data([X, y], subset = valy, t = t)
		nval = len(tva)
		metrics.loc[i, 'ValStart'] = tva[0].year
		metrics.loc[i, 'ValEnd'] = tva[-1].year
	
		# build model
		model = model_builder_LSTM(**hyperparams)
	
		# fit model
		history = model.fit(Xtr, ytr, epochs = fitting_params['epochs'], \
			batch_size = fitting_params['batch_size'], validation_data = (Xva, yva), \
			verbose = 0, callbacks = [es])
		
		# store history
		lt = len(history.history['loss'])
		hist.loc[1:lt, str(i+1)+'_loss'] = history.history['loss']
		hist.loc[1:lt, str(i+1)+'_val_loss'] = history.history['val_loss']
		
		# predict val set
		tval[:nval,i] = tva
		ytrue[:nval,i] = yva
		ypred[:nval,i] = np.squeeze(model.predict(Xva, verbose = 0))
		
		# calculate metrics
		metrics.loc[i, 'RRMSE'] = RRMSE(ytrue[:,i], ypred[:,i])
		metrics.loc[i, 'ExpVar'] = relative_explained_variance(ytrue[:,i], ypred[:,i])
		
		# save model if needed
		if isinstance(store_models, str):
			joblib.dump(model, store_models.format(i))
			
		# print
		if verbose:
			print('  -- model %2i/%2i trained' % (i+1, N))
				
	
	# --- Save history if needed ---
	if isinstance(save_hist, str):
		hist.to_csv(save_hist)
		
		
	# --- Plot figures if needed ---
	# validation set results
	if isinstance(plot_res, str):
		fig, ax = plt.subplots(min(N, 12), 1, figsize = [12, 18], sharey = True)
		plt.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05, hspace=0.22)
			
		for i in range(min(N, 12)):
			h1, = ax[i].plot(tval[:,i], ytrue[:,i], color = 'teal', label = 'true')
			h2, = ax[i].plot(tval[:,i], ypred[:,i], color = 'maroon', label = 'pred')
			ax[i].set_xlim([tval[0, i], tval[-1, i]])
			ax[i].text(0.01, 0.95, str(i+1), ha = 'left', va = 'top', fontsize = 16, \
				transform = ax[i].transAxes)
				
		ax[0].legend(handles = [h1, h2], loc = 'upper center', fontsize = 16, ncol = 2)
		fig.suptitle('Prediction of the validation set', fontsize = 24, y = 0.98)
		fig.savefig(plot_res)
		plt.close(fig)
		
	# training history
	if isinstance(plot_hist, str):
		fig, ax = plt.subplots(4, 3, figsize = [14, 18], sharex = True, sharey = True)
		plt.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05, hspace=0.0, \
			wspace = 0.0)
		ax = np.reshape(ax, -1)
		
		for i in range(min(N, 12)):
			#lenhist = sum(~np.isnan(hist.loc[:,str(i+1)+'_loss'].values))
			ax[i].plot(hist.index, hist.loc[:,str(i+1)+'_loss'].values, \
				color = 'teal', label = 'loss')
			ax[i].plot(hist.index, hist.loc[:,str(i+1)+'_val_loss'].values, \
				color = 'maroon', label = 'val_loss')
			ax[i].text(0.02, 0.98, str(i+1), ha = 'left', va = 'top', fontsize = 16, \
				transform = ax[i].transAxes)
				
		fig.text(0.5, 0.01, 'Epoch', fontsize = 20, ha = 'center', va = 'bottom')
		fig.text(0.01, 0.5, 'Loss', fontsize = 20, ha = 'left', va = 'center', \
			rotation = 'vertical')
		ax[1].legend(loc = 'upper center', fontsize = 16)
		fig.suptitle('Loss ('+hyperparams['loss_fun']+') during training', \
			fontsize = 24, y = 0.98)
		fig.savefig(plot_hist)
		plt.close(fig)
	
	return metrics
	
	
def change_boxplot_colors(boxplot, ind = None, col1 = 'silver', col2 = 'black', col3 = None):
	# Changes the colors of the boxplot at index ind. ind can be integer (one box), list
	# (multiple boxes) or None (all boxes).
	# col1 - box (edges and fill), whiskers, caps and fliers
	# col2 - median
	# col3 - mean
	
	if ind == None:
		ind = range(len(boxplot['boxes']))
	elif not hasattr(ind, '__iter__'):
		ind = [ind]

	for i in ind:
		boxplot['whiskers'][i*2].set_color(col1)
		boxplot['whiskers'][i*2+1].set_color(col1)
		boxplot['caps'][i*2].set_color(col1)
		boxplot['caps'][i*2+1].set_color(col1)
		boxplot['boxes'][i].set_facecolor(col1)
		boxplot['boxes'][i].set_edgecolor(col1)
		boxplot['fliers'][i].set_markerfacecolor(col1)
		boxplot['fliers'][i].set_markeredgecolor(col1)
		boxplot['medians'][i].set_color(col2)
		boxplot['means'][i].set_color(col3)
	return
