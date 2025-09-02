#!/usr/bin/env python3

# needs environmentSLD-vis

# Plot baseline test set predictions (ExpVar) as boxplots.

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerTuple

sys.path.insert(0, moduledir)
from Functions import get_config

# ------------------------------------------------------------------------------
config = get_config()

dim = config['dim']
met = 'ExpVar'
models = {'LR': 'Linear regression', 'NN': 'Neural network'}
labels = {'x': 'Sequence length (months)', 'y': 'Explained variance (%)'}

resdir = config['dirs']['data'] + config['dirs']['ana']
figdir = config['dirs']['figs'] + config['dirs']['fin']

resfile = 'ResultsTestSet2012-2016.nc'
figname = 'BaselineAllBox.png'

# plotting parameters
nr, nc = 12, 4  # 9, 5
figsize = [26, 32]
subplots_adjust=dict(bottom=0.04,top=0.98,left=0.06,right=0.99,wspace=0.13,hspace=0.04)
w = 0.4
alpha = 0.3
lw1, lw2 = 1, 2
col = {'NN': 'maroon', 'LR': 'teal', 'best': 'lightgrey'}

fs0, fs1, fs2 = 26, 20, 16 # font size for title, subtitles, tick labels
tx, ty = 0.015, 0.03
txtpar = dict(ha='left', va='bottom', fontsize=fs1, bbox={'facecolor':'white', \
    'edgecolor':'none', 'alpha':0.3})
# ------------------------------------------------------------------------------

# --- Load and prepare data ---
res = xr.open_dataset(resdir+resfile, engine='netcdf4')
name = res['name'].values
res = res[met].loc[{dim['f']:'all'}].reset_coords(drop=True)

stat = res.coords[dim['s']].values
seq_len = res.coords[dim['n']].values
mod = res.coords['mod'].values


# --- Find best sequence length and best model ---
# best seq len for each model type
bestseq=res.median(dim=dim['m']).idxmax(dim=dim['n']).astype(int)

# best model overall
best=res.median(dim=dim['m']).max(dim=dim['n']).idxmax(dim='mod').values




# --- Plot ---
# box positions and ticks
pos = np.zeros((len(mod), len(seq_len)))
for i in range(len(mod)):
    pos[i,:] = np.arange(len(seq_len)) + w*(i-0.5)
xticks = np.mean(pos, axis=0)
gridx = (pos[0,1:] + pos[-1,:-1])/2

# prepare figure
fig, ax = plt.subplots(nr, nc, figsize=figsize, sharex=True)
plt.subplots_adjust(**subplots_adjust)
#ax = np.reshape(ax, -1)

# separate ax into plotting ax and colorbar ax
cblen = nr*nc-len(stat)  # number of axes colorbar uses
cbax = ax[:cblen, -1]
ax = np.concatenate((ax[:,:nc-1].flat, ax[cblen:, -1]), axis=0)

# prepare legend handles
han = [None] * (len(mod)*2+1)  

# plot
for s, si in zip(stat, range(len(stat))):
    for m, mi in zip(mod, range(len(mod))):
        # extract data
        x = res.loc[{dim['s']:s, 'mod':m}].values.T
        
        # plot ensemble
        bp = ax[si].boxplot(x, positions=pos[mi], widths=w, patch_artist=True, \
            boxprops={'facecolor':'none', 'edgecolor':col[m], 'lw':lw1}, \
                capprops={'color':col[m], 'lw':lw1}, \
                whiskerprops={'color':col[m], 'lw':lw1}, \
                #flierprops={'markeredgecolor': col[m]}, \
                showfliers=False, \
                medianprops={'lw': lw1, 'color': 'k'})
    
        # mark the best seq len for each model type
        bs = np.where(seq_len == bestseq.loc[{'mod':m, dim['s']:s}].values)[0][0]
        bsv = res.loc[{dim['s']:s, 'mod':m, dim['n']:seq_len[bs]}].\
            median(dim=dim['m']).values
        
        bp['boxes'][bs].set_facecolor(to_rgba(col[m], alpha=alpha))
        bp['medians'][bs].set_color(col[m])
        bp['medians'][bs].set_linewidth(lw2)
        
        hl = ax[si].axhline(bsv, lw=lw2, color=col[m])
            
        # get boxplot handles
        han[mi] = (bp['boxes'][0] if bs!=0 else bp['boxes'][1], bp['medians'][0])
        han[2+mi] = (bp['boxes'][bs], hl)
        
        # mark the best model overall
        if best[si]==m:
            han[-1] = ax[si].axvspan(pos[mi,bs]-w/2, pos[mi,bs]+w/2, \
                color=col['best'], zorder=0)
                
            bestmod = '{:2s}{:d}'.format(best[si],seq_len[bs])
        
    # station name and number (shift the ylim to accomodate it)
    txt = '{:s} ({:d}): {:s}'.format(name[si], s, bestmod)
    ax[si].text(tx, ty, txt, **txtpar, transform=ax[si].transAxes)
    ymin, ymax = ax[si].get_ylim()
    ax[si].set_ylim([ymin-0.15*(ymax-ymin), ymax])
    
    # subplot formatting
    ax[si].tick_params(axis='x', labelsize=fs1, length=0)
    ax[si].tick_params(axis='y', labelsize=fs2)
    ax[si].set_xticks(xticks, labels = seq_len)
    ax[si].grid(axis='y')
    ax[si].set_xticks(gridx, minor=True)
    ax[si].grid(axis='x', which='minor')

# common legend
for a in cbax:
    a.axis('off')
leglab = ['{:s}'.format(models[m]) for m in mod] + \
    ['Best {:s} model'.format(m) for m in mod] + ['Best model overall']
cbax[1].legend(handles=han, labels = leglab, loc='center', fontsize=fs0, ncol=1, \
    handler_map={tuple: HandlerTuple(ndivide=1)})

#bbox_to_anchor=(0.5, 0.985)

# figure formatting + save figure
fig.supxlabel(labels['x'], fontsize=fs0)
fig.supylabel(labels['y'], fontsize=fs0)

fig.savefig(figdir+figname)
