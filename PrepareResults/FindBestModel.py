#!/usr/bin/env python3

# needs environmentSLD-vis

# Finding and saving the best seq len per model type and best model over into a csv file.

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, moduledir)
from Functions import get_config

# ---------------------------------------------------------------------------------------------
config = get_config()

dim = config['dim']
met = 'ExpVar'

resdir = config['dirs']['data'] + config['dirs']['ana']
resfile = 'ResultsTestSet2012-2016.nc'
outfile = 'BestModelList.csv'

cols = ['name', 'lon', 'lat']
cols2 = ['best', 'best_type']
# ---------------------------------------------------------------------------------------------

# --- Load data ---
res = xr.load_dataset(resdir+resfile, engine='netcdf4')

stat = res.coords[dim['s']].values
mod = res.coords['mod'].values
seq = res.coords[dim['n']].values



# --- Find best model at each location ---
res = res[[met, *cols]].loc[{dim['f']:'all'}].drop_vars(dim['f'])
res = res.median(dim['m'])

# best sequence length for each model type
best = res[met].idxmax(dim=dim['n']).astype(int).to_dataset(name='best_seq')
best[met] = res[met].max(dim=dim['n'])

# best model overall
best['best_type'] = best[met].idxmax(dim='mod')

best['best'] = best['best_type'].copy(deep=True)
for s in stat:
    best['best'].loc[{dim['s']:s}] = best['best_type'].loc[{dim['s']:s}].values + \
        str(best['best_seq'].loc[{dim['s']:s, 'mod':best['best_type'].loc[{dim['s']:s}].values}].values)


# --- Output ---
# convert to pandas
out = pd.DataFrame(index=pd.Index(stat, name='ID'), columns=cols + cols2 + ['best_seq_{:s}'.format(m) for m in mod])

for c in cols:
    out[c] = res[c]

for c in cols2:
    out[c] = best[c]

for m in mod:
    out['best_seq_{:s}'.format(m)] = best['best_seq'].loc[{'mod':m}]

# save to csv
out.to_csv(resdir+outfile)

print(out)
