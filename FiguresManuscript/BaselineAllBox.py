#!/usr/bin/env python3

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
#from matplotlib.patches import Rectangle

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
figname = 'BaselineAllBox.pdf'

# plotting parameters
nr, nc = 12, 4  # 9, 5
figsize = [26, 32]
subplots_adjust=dict(bottom=0.04,top=0.98,left=0.06,right=0.99,wspace=0.13,hspace=0.04)
w = 0.4
alpha = 0.3
lw1, lw2, lwf = 1, 2, 2
col = {'NN': 'maroon', 'LR': 'teal', 'best': 'lightgrey'}

fs0, fs1, fs2 = 26, 20, 16 # font size for title, subtitles, tick labels
tx, ty = 0.015, 0.03
txtpar = dict(ha='left', va='bottom', fontsize=fs1, bbox={'facecolor':'white', \
    'edgecolor':'none', 'alpha':0.3})
    
# order of the stations on the grid
statord = np.array([ \
    [  -1, 313, 203,  79], \
    [  -1, 682,  88, 194], \
    [  -1, 509, 285, 315], \
    [ 413, 302, 172,  14], \
    [  20, 179, 376, 239], \
    [  22,   9,  78,2105], \
    [  23,  32,2106,  69], \
    [ 236,  80,  70, 118], \
    [  25,1036, 119, 397], \
    [1037,  81, 113, 330], \
    [  24, 789,  98, 120], \
    [   7,  13,   8,  11]])

# list of regions
reg = {\
    'Norwegian Sea': [509, 682, 313], \
    'North Sea': [413, 20, 9, 22, 32, 23, 25, 236, 24, 1037, 7, 1036, 80], \
    'Baltic Sea': [118, 315, 14, 239, 376, 172, 285, 194, 79, 203, 88, 78, 2105, 69, 2106, 70], \
    'Danish Straits': [119, 397, 81, 113, 330, 789, 98, 120, 13, 8, 11], \
    'Skagerrak': [302, 179]}
colorder = [3, 0, 2, 1, 4]
colreg = {r: plt.rcParams['axes.prop_cycle'].by_key()['color'][o] for r, o in zip(reg.keys(), colorder)}
# ------------------------------------------------------------------------------

# --- Load and prepare data ---
res = xr.open_dataset(resdir+resfile, engine='netcdf4')
name = res['name']
res = res[met].loc[{dim['f']:'all'}].reset_coords(drop=True)

stat = res.coords[dim['s']].values
seq_len = res.coords[dim['n']].values
mod = res.coords['mod'].values
name = {int(s): str(name.loc[{dim['s']:s}].values) for s in stat}


# --- Prepare colors for stations ---
colstat = {c: reg[r] for r, c in colreg.items()}
colstat = {s: c for c,r in colstat.items() for s in r}



# --- Find best sequence length and best model ---
# best seq len for each model type
bestseq=res.median(dim=dim['m']).idxmax(dim=dim['n']).astype(int)

# best model overall
best=res.median(dim=dim['m']).max(dim=dim['n']).idxmax(dim='mod') #.values
best = {int(s): str(best.loc[{dim['s']:s}].values) for s in stat}



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

# prepare legend handles
han = [None] * (len(mod)*2+1)  

# plot
for s, a in zip(statord.flat, ax.flat):
    # skip legend axes
    if s==-1: continue
    
    # otherwise plot   
    for m, mi in zip(mod, range(len(mod))):
        # extract data
        x = res.loc[{dim['s']:s, 'mod':m}].values.T
        
        # plot ensemble
        bp = a.boxplot(x, positions=pos[mi], widths=w, patch_artist=True, \
            boxprops={'facecolor':'none', 'edgecolor':col[m], 'lw':lw1}, \
                capprops={'color':col[m], 'lw':lw1}, \
                whiskerprops={'color':col[m], 'lw':lw1}, \
                showfliers=False, \
                medianprops={'lw': lw1, 'color': 'k'})
    
        # mark the best seq len for each model type
        bs = np.where(seq_len == bestseq.loc[{'mod':m, dim['s']:s}].values)[0][0]
        bsv = res.loc[{dim['s']:s, 'mod':m, dim['n']:seq_len[bs]}].\
            median(dim=dim['m']).values
        
        bp['boxes'][bs].set_facecolor(to_rgba(col[m], alpha=alpha))
        bp['medians'][bs].set_color(col[m])
        bp['medians'][bs].set_linewidth(lw2)
        
        hl = a.axhline(bsv, lw=lw2, color=col[m])
            
        # get boxplot handles
        han[mi] = (bp['boxes'][0] if bs!=0 else bp['boxes'][1], bp['medians'][0])
        han[2+mi] = (bp['boxes'][bs], hl)
        
        # mark the best model overall
        if best[s]==m:
            han[-1] = a.axvspan(pos[mi,bs]-w/2, pos[mi,bs]+w/2, color=col['best'], zorder=0)
            bestmod = '{:2s}{:d}'.format(best[s],seq_len[bs])
        
    # station name and number (shift the ylim to accomodate it)
    txt = '{:s} ({:d}): {:s}'.format(name[s], s, bestmod)
    a.text(tx, ty, txt, **txtpar, transform=a.transAxes)
    ymin, ymax = a.get_ylim()
    a.set_ylim([ymin-0.15*(ymax-ymin), ymax])
    
    # subplot formatting
    a.tick_params(axis='x', labelsize=fs1, length=0)
    a.tick_params(axis='y', labelsize=fs2)
    a.set_xticks(xticks, labels = seq_len)
    a.grid(axis='y')
    a.set_xticks(gridx, minor=True)
    a.grid(axis='x', which='minor')
    
    # change ax frame color based on region
    for spine in ['bottom', 'top', 'right', 'left']:
        a.spines[spine].set_color(colstat[s])
        a.spines[spine].set_linewidth(lwf)

# ax for common legends
cbax = ax[statord==-1]
for a in cbax:
    a.axis('off')
legprops = dict(fontsize=fs0, mode='expand', alignment='center', \
    title_fontproperties={'size':fs0, 'weight':'bold'})

# common legend for models
leglab = ['{:s}'.format(models[m]) for m in mod] + \
    ['Best {:s} model'.format(m) for m in mod] + ['Best model overall']
cbax[0].legend(handles=han, labels=leglab, title = 'Models', loc='upper center', \
    handler_map={tuple: HandlerTuple(ndivide=1)}, **legprops)

# common legend for regions
x, y = 2.5, 1
rect = [(-x,-y), (x, -y), (x, y), (-x, y), (-x, -y)]
dummy = [None] * len(reg)
for i, r in zip(range(len(reg)), reg.keys()):
    dummy[i] = plt.scatter([],[], s=4000, marker=rect, facecolor='none', edgecolor=colreg[r], linewidths=lwf, label=r)
cbax[2].legend(handles=dummy, title='Regions', loc='lower center', handlelength=2.5, **legprops)

# figure formatting + save figure
fig.supxlabel(labels['x'], fontsize=fs0)
fig.supylabel(labels['y'], fontsize=fs0)

fig.savefig(figdir+figname)
