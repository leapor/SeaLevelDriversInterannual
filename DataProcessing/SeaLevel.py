#!/usr/bin/env python3

# Selecting the sea level stations based on their location (within the area of interest
# defined as a longitude-latitude box), time span (covering the time span needed for this
# study), amount of missing data within that time span, and station and data flags.
moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import numpy as np
import pandas as pd
from datetime import date
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

sys.path.insert(0, moduledir)
from Functions import get_config, get_psmsl_filelist, get_psmsl_missing_months, get_psmsl_data, \
	plot_missing_data, prepare_map, draw_regions
	
	
# ---------------------------------------------------------------------------------------
config = get_config()
var = 'sl'
vname = config['variables'][var]
lon1, lon2 = config['region']['lon']
lat1, lat2 = config['region']['lat']
y1, y2 = config['time']['start'], config['time']['end']
mongap = config['time']['mongap']
daygap = 7  # selected daygap

datadir = config['dirs']['orig'] + config['data_original'][var]
regfile = config['dirs']['orig'] + config['data_original']['reg']

outdir = config['dirs']['data'] + config['dirs']['ext']
fileout = outdir + config['variables'][var] + '.csv'

figdir = config['dirs']['figs'] + config['dirs']['ext']
figname1 = vname + '_Map.png'
figname2 = vname + '_Timeseries.png'

proj = ccrs.PlateCarree()
col = config['colors'][1]
# -------------------------------------------------------------------------------------------------

# --- Creating output directories if they do not already exist ---
if not os.path.exists(outdir):
	os.makedirs(outdir)
if not os.path.exists(figdir):
	os.makedirs(figdir)



# ----------------------------------------------------------------------
# --- Getting data and checking for missing values ---
# ----------------------------------------------------------------------
# --- Get and prepare the data ---
# get the regions
reg = pd.read_csv(regfile, index_col = 0)
regions = reg.index.values

# load the metadata file
meta = get_psmsl_missing_months(datadir, [y1, y2])

# drop the stations from outside the region of interest and timespan
meta = meta.loc[((meta.lon > lon1) & (meta.lon < lon2) & \
	(meta.lat > lat1) & (meta.lat < lat2) & \
	(meta.flag == 0) & \
	(meta.miss_mon <= mongap)), :]

# get the data
data = get_psmsl_data(datadir, meta.index, [y1, y2])
t = np.array(data[meta.index[0]].index)


# --- Calculate number of missing months and months with gaps ---
num_miss = np.zeros((len(meta.index)), dtype = np.int16)
num_gap = np.zeros((len(meta.index)), dtype = np.int16)

for i, ii in zip(meta.index, range(len(meta.index))):
	num_miss[ii] = sum(np.isnan(data[i].loc[:, 'rlr'].values) | \
		(data[i].loc[:, 'miss'] > daygap))
	num_gap[ii] = sum((data[i].loc[:, 'miss'] > 0) & \
		(data[i].loc[:, 'miss'] <= daygap))


# --- Extracting and saving useful data ---
# replace missing months and months with missing days in meta with information regarding the
# selected maximum allowed number of days missing per month
meta.loc[:, 'miss_mon'] = num_miss
meta.loc[:, 'miss_day'] = num_gap

# find region information for each station
for r in regions:
	# calculate the number of stations per region (and add it to reg)
	ind = (meta.lon >= reg.loc[r, 'lon1']) & (meta.lon <= reg.loc[r, 'lon2']) & \
		(meta.lat >= reg.loc[r, 'lat1']) & (meta.lat <= reg.loc[r, 'lat2'])
	reg.loc[r, 'stations'] = sum(ind)
	
	# add the region information to the meta file
	meta.loc[ind, 'reg'] = r
	meta.loc[ind, 'reg_name'] = reg.loc[r, 'region']
	
# reorganize columns in meta and remove unnecessary ones (coast, code and flag)
meta = meta.loc[:, ['lon', 'lat', 'name', 'reg', 'reg_name', 'miss_mon', 'miss_day']]
	
# set to NaN months with too many days missing
for i in meta.index:
	data[i].loc[data[i].loc[:, 'miss'] > daygap, :] = np.nan

# converting data from dictionary to pandas dataframe
for i in meta.index:
	if i == meta.index[0]:
		temp = data[i].loc[:, ['rlr']]
	else:
		temp.loc[:, i] = data[i].loc[:, 'rlr']
	temp.rename(columns = {'rlr' : i}, inplace = True)

# select only usable stations considering selected daygap
meta = meta.loc[meta.loc[:, 'miss_mon'] <= mongap, :]
data = temp.loc[:, meta.index]

# combine station information and time series into one dataframe
data = data.transpose()
data = meta.join(data)

# save to file
data.to_csv(fileout)



# ------------------------------------------------------------------------
# --- Plotting ---
# ------------------------------------------------------------------------
# --- Plot maps ---
fig, ax = plt.subplots(1, 1, figsize = [14, 11], subplot_kw={'projection': proj})
plt.subplots_adjust(bottom = 0.0, top = 1.0, left = 0.05, right = 0.99)
prepare_map(ax, [lon1, lon2], [lat1, lat2])

# plotting the number of missing/partially missing months
h1, h2, ticks1, ticks2, labels1, labels2, texts = plot_missing_data(ax, data, \
	data.loc[:, 'miss_mon'].values, data.loc[:, 'miss_day'].values, b1 = 3, b2 = 6)
		
# draw regions
draw_regions(reg, ax)
		
# adding a legend
ax.legend(loc = 'lower right', fontsize = 'xx-large')

# adding colorbars
cbar1 = plt.colorbar(h1, ax = ax, shrink = 0.8, fraction = 0.12, pad = -0.4, \
	location = 'top')
cbar2 = plt.colorbar(h2, ax = ax, shrink = 0.8, fraction = 0.1, pad = 0.025, \
	location = 'bottom')
cbar1.set_ticks(ticks = ticks1, labels = labels1, size = 'x-large')
cbar2.set_ticks(ticks = ticks2, labels = labels2, size = 'x-large')
cbar1.set_label('Missing data (out of '+str(y2-y1+1)+' years)', size = 'xx-large')
cbar2.set_label('Months with some days missing (out of '+str(y2-y1+1)+' years)', \
	size = 'xx-large')
		
# write the number of usable stations in the area of interest
ax.text(0.02, 0.98, str(data.shape[0])+ ' stations', ha = 'left', va = 'top', \
	fontsize = 'xx-large', transform = ax.transAxes, snap = True)
	
fig.savefig(figdir + figname1)
plt.close(fig)

	
# --- Plot time series ---
# extract the time series
x = data[t]
xm = (np.isnan(x))
n = len(x.index)

# prepare the figure
fig, ax = plt.subplots(n, 1, figsize = [14, 24])
plt.subplots_adjust(left=0.07, right=0.9, top=0.95, bottom=0.05, hspace = 0.0)

# plot time series
for i in range(n):
	ax[i].plot(t, x.iloc[i, :])
	for ti in range(len(t)):
		if xm.iloc[i, ti]: ax[i].axvline(t[ti], color = 'grey')
	
	ax[i].set_xlim([t[0], t[-1]])
	if (i != n-1):
		ax[i].xaxis.set_tick_params(labelbottom = False)
	else:
		ax[i].xaxis.set_tick_params(labelsize = 18)
	ax[i].yaxis.set_label_position('right')
	ax[i].set_ylabel(data.index[i], rotation = 0, fontsize = 18, \
		ha = 'left', va = 'center')

# formatting
fig.text(0.01, 0.5, vname + ' (mm)', ha = 'left', va = 'center', fontsize = 24, \
	rotation = 'vertical')
fig.text(0.99, 0.5, 'Location IDs ('+str(n)+' locations)', \
	ha = 'right', va = 'center', fontsize = 24, rotation = 'vertical')
fig.suptitle('Sea level observations', fontsize = 24)

# save figure
fig.savefig(figdir + figname2)
plt.close(fig)
