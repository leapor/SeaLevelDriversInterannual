#!/usr/bin/env python3

# needs environment SLD-vis

# Re-calculating metrics for all models to take into account shorter length of the 
# data set when sequence length is longer.
# Only for test set.
# The time series is shortened for all sequence lengths to only contain months for which
# all models provide prediction and ExpVar is calculated.
# The best model overall is also found and saved.
# Data is saved as ResultsTestSet2012-2016.nc.
# Also switches the order of models, so that NN comes first.

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, moduledir)
from Functions import get_config, relative_explained_variance

# ----------------------------------------------------------------------------------
config = get_config()

dim = config['dim']
met = 'ExpVar'
var = ['name', 'lon', 'lat', 'ValStart', 'ValEnd', met, 'true', 'pred', 'median']
resdir = config['dirs']['data'] + config['dirs']['ana']
figdir = config['dirs']['figs'] + config['dirs']['ana'] + 'Timeseries/'

resfile = 'ResultsTestSet.nc'
outfile = 'ResultsTestSet2012-2016.nc'
# -----------------------------------------------------------------------------------

# --- Load data ---
res = xr.open_dataset(resdir+resfile, engine='netcdf4')

# get coordinates
stat = res.coords[dim['s']].values
seq_len = res.coords[dim['n']].values
mod = res.coords['mod'].values
ens = res.coords[dim['m']].values
time = res.coords[dim['t']].values
features = res.coords[dim['f']].values


# --- Switch the order of models ---
mod = ['NN', 'LR']
res = res.loc[{'mod': mod}]



# --- Crop the time series ---
ind = res['pred'].isnull().any(dim=['mod',dim['n'],dim['m'],dim['f']]).all(dim=dim['s'])
out = res[var].loc[{dim['t']:time[~ind]}].copy(deep=True)



# --- Recalculate metric ---
for s in stat:
    true = out['true'].loc[{dim['s']:s}].values
    
    for m in mod:
        for n in seq_len:
            for i in ens:
                for f in features:
                    pred = out['pred'].loc[{dim['s']:s, 'mod':m, dim['n']:n, dim['m']:i, \
                        dim['f']:f}].values
                    out[met].loc[{dim['s']:s,'mod':m,dim['n']:n,dim['m']:i,dim['f']:f}] =\
                        relative_explained_variance(true, pred)


# --- Find the new median ensemble member ---
medind = int(len(ens)/2)    # index of the median member
base = out[met].loc[{dim['f']:'all'}]   # baseline metric
sortord = np.argsort(-base, axis=base.get_axis_num(dim['m'])) # sort order
medmem = sortord.loc[{dim['m']: medind}].reset_coords(drop=True) # extracted median ind
out['median'] = medmem


# --- Save the new dataset ---
out.to_netcdf(resdir+outfile)
