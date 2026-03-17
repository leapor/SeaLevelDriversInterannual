#!/usr/bin/env python3

# Plot time series of sea level and its potential drivers for a sample station.
# Drivers are re-ordered for easier presentation: GloSST right after other ERA5 drivers.

moduledir = '../Scripts/'  # directory with the Functions file

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, moduledir)
from Functions import get_config

# -----------------------------------------------------------------------------
config = get_config()

# coordinates, sample station ID, sample seq len
dim = config['dim']
sample_stat = 119
seq_len = 6 # sample sequence
sample_t = '2015-01-01'
sl = 'sea_level'

# directory and file names
indir = config['dirs']['data'] + config['dirs']['pro']
resdir = config['dirs']['data'] + config['dirs']['ana']
figdir = config['dirs']['figs'] + config['dirs']['fin']

infile = 'data_station_{:d}_test.csv' # station ID, dataset (train/test)
resfile = 'ResultsTestSet2012-2016.nc'
figname = 'Schematic.pdf'

# variable names and order
var = config['variables']    # short name -> long name
varname = config['varnames'] # short name -> display name
featord = config['drivers']['order']  # display order of features

# figure parameters
width, height = 18, 13
subplots_adjust = dict(bottom=0.05, top=0.96, left=0.2, right=0.97, wspace=0, hspace=0)
fs1, fs2 = 24, 30

# plotting parameters
col1, col2, col3 = 'k', 'red', 'darkgrey'
alpha = 0.5
ms = 10  # marker size for sequence plot
ylim = np.array([-1,1]) * 2
xlim = pd.to_datetime(['2014-01-01', '2015-12-01'])
# -----------------------------------------------------------------------------


# --------------------------------------------------------------
# --- Helper functions for plotting ---
# --------------------------------------------------------------
# --- Function that makes all given axes invisible ---
def make_invisible(ax):
    for a in ax.flatten():
        a.axis('off')
        
# --- Function that keeps only the outside frame ---
# (keeps the background)
def frame_subset(fig, ax, rows, cols, col='k', lw=3):
    # ax = full grid
    # rows, cols = rows and cols of the subset to frame
    # frame is for all subplots between smallest and largest indices of subset
    
    # hide all spines
    nr, nc = ax.shape
    for i in range(nr):
        for j in range(nc):
            for spine in ax[i,j].spines.values():
                spine.set_visible(False)
                            
    # add back frame
    bottom_left = ax[rows[-1],cols[0]].get_position()
    top_right = ax[rows[0],cols[-1]].get_position()
    width = top_right.x1 - bottom_left.x0
    height = top_right.y1 - bottom_left.y0
    rect = Rectangle((bottom_left.x0, bottom_left.y0), width, height, clip_on=False, \
        facecolor='none', edgecolor=col, linewidth=lw, transform=fig.transFigure)
    fig.patches.append(rect)
    
    return


# --------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

# --- Create directory for saving figures ---
if not os.path.exists(figdir):
	os.makedirs(figdir)



# --- Load and prepare data ---
# inputs
data = pd.read_csv(indir+infile.format(sample_stat), index_col=0, parse_dates=True)

t = data.index
y = data.loc[:,sl]
X = data.loc[:,[var[f] for f in featord]]
dr = X.columns

# results
res = xr.load_dataset(resdir+resfile, engine='netcdf4')
res = res.loc[{dim['s']:sample_stat, dim['n']:seq_len, dim['f']:'all', 'mod':'NN'}]
pred = res['pred'].loc[{dim['m']:res['median'].values}].to_pandas()



# --- Plot ---
plt.rcParams['xtick.labelsize'] = fs1
height_ratios = [1] * len(dr) + [2] + [2]
sample = t[t<=sample_t][-1-seq_len:]

# create figure
fig, ax = plt.subplots(len(dr)+2,1, figsize=[width, height], \
    sharex=True, squeeze=False, height_ratios=height_ratios)
plt.subplots_adjust(**subplots_adjust)

# plot drivers
for d,i in zip(dr, range(len(dr))):
    ax[i,0].plot(t, X.loc[:,d], color=col1, clip_on=False)
    ax[i,0].plot(sample, X.loc[sample,d], 'o', color=col1, ms=ms, clip_on=False)
    
# plot sea level
ax[-1,0].plot(t, y, color=col1)
ax[-1,0].plot(sample[-1], pred.loc[sample[-1]], 'd', color=col2, ms=ms)

# format subplots
for i in range(len(ax)):
    ax[i,0].set_xlim([t[0], t[-1]])
    ax[i,0].set_yticks([])
    ax[i,0].set_facecolor('none')
    ax[i,0].set_ylim(ylim)
    ax[i,0].tick_params(length=0)
     
ax[0,0].set_ylim([ylim[0], max(X.iloc[:,0])*1.1])
ax[len(dr)-1,0].set_ylim([min(X.iloc[:,-1])*1.1, ylim[1]])
ax[-1,0].set_ylim(auto=True)

# variable names
var = {name: n for n, name in var.items()}
varname = [varname[f] for f in featord] + ['', varname['sl']]

for i in range(len(varname)):
    ax[i,0].set_ylabel(varname[i], fontsize=fs1, rotation='horizontal', ha='right', va='center', labelpad=12)


# add arrows for input/output
for s in sample:
    ax[0,0].vlines(s, ylim[0], X.loc[s, dr[0]], color=col1, alpha=alpha, zorder=1)
    for i in range(1, len(dr)):
        ax[i,0].axvline(s, color=col1, alpha=alpha, zorder=1)
    ax[-2,0].axvline(s, color=col1, alpha=alpha, ymin=0.5, zorder=1)
    ax[-2,0].scatter(s, ylim[1]*0.5, s=100, c=col1, alpha=alpha, marker='v')
    
ax[-2,0].axvline(sample[-1], color=col2, alpha=alpha, ymax=0.5, zorder=1)
ax[-1,0].vlines(sample[-1], pred.loc[sample[-1]], ylim[1], color=col2, alpha=alpha, zorder=1)
ax[-1,0].scatter(sample[-1], pred.loc[sample[-1]]+0.6, s=100, c=col2, alpha=alpha, marker='v')

# add rectangle for model
width = pd.DateOffset(months=seq_len+4)
height_rat = 0.4
model = Rectangle((sample[0]-pd.DateOffset(months=2), ylim[0]*height_rat), width, ylim[1]*height_rat*2, \
    facecolor='white', edgecolor='k', lw=3)
ax[-2,0].add_patch(model)
ax[-2,0].text(sample[int(seq_len/2)], 0, 'MODEL', ha='center', va='center', fontsize=fs2)


# frames
frame_subset(fig, ax, range(len(dr)), [0,0])
frame_subset(fig, ax, range(len(dr)+1, len(dr)+2), [0,0])

fig.savefig(figdir+figname)
plt.close(fig)



quit()














# plot sea level
i=0; d=sl
fig, ax = plt.subplots(1,len(cols), figsize=[width, height], \
    sharex='col', sharey=True, width_ratios=cols, squeeze=False)
plt.subplots_adjust(**subplots_adjust['sl'])
for s,j in zip(ds, c):
    ax[i,j].plot(data2[s].loc[:,d], color=col['sl'])
    ax[i,j].set_xlim(xlim[s])
    ax[i,j].set_xticks(xticks[s], labels=xlab[s])
    ax[i,j].set_ylim(ylim)
    ax[i,j].set_yticks([])
frame_col(fig, ax, 0)
frame_col(fig, ax, 1)
make_invisible(ax[:,c[-1]+1:])
fig.savefig(figdir+figname.format(varname['sl'], name))
plt.close(fig)


# --- Training set (test is lighter) ---
# fig name; column widths; which columns to plot in
name = figs['tr']
cols = [tlen['tr']/tlen['all'], tlen['te']/tlen['all'], (w-1)*tlen['te']/tlen['all']]
c = [0,1]

# plot drivers
fig, ax = plt.subplots(len(dr),len(cols), figsize=[width, height*ratio], \
    sharex='col', sharey=True, width_ratios=cols, squeeze=False)
plt.subplots_adjust(**subplots_adjust['d'])

for s,j in zip(ds, c):
    a=alpha['+'] if s=='tr' else alpha['-']
    for d,i in zip(dr, range(len(dr))):
        ax[i,j].plot(data2[s].loc[:,d], color=col['d'], alpha=a)
        ax[i,j].set_xlim(xlim[s])
        ax[i,j].set_xticks(xticks[s], labels=xlab[s])
        ax[i,j].set_ylim(ylim)
        ax[i,j].set_yticks([]) 
frame_col(fig, ax, 0)
frame_col(fig, ax, 1)     
make_invisible(ax[:,c[-1]+1:])
fig.savefig(figdir+figname.format(varname['d'], name))
plt.close(fig)

# plot sea level
i=0; d=sl
fig, ax = plt.subplots(1,len(cols), figsize=[width, height], \
    sharex='col', sharey=True, width_ratios=cols, squeeze=False)
plt.subplots_adjust(**subplots_adjust['sl'])
for s,j in zip(ds, c):
    a=alpha['+'] if s=='tr' else alpha['-']
    ax[i,j].plot(data2[s].loc[:,d], color=col['sl'], alpha=a)
    ax[i,j].set_xlim(xlim[s])
    ax[i,j].set_xticks(xticks[s], labels=xlab[s])
    ax[i,j].set_ylim(ylim)
    ax[i,j].set_yticks([])
frame_col(fig, ax, 0)
frame_col(fig, ax, 1)
make_invisible(ax[:,c[-1]+1:])
fig.savefig(figdir+figname.format(varname['sl'], name))
plt.close(fig)


# --- Sequence on training set (test is lighter) ---
# fig name; column widths; which columns to plot in; time for sample sequence
name = figs['seq']
cols = [tlen['tr']/tlen['all'], tlen['te']/tlen['all'], (w-1)*tlen['te']/tlen['all']]
c = [0,1]
tsam = time['tr'][ns-seq_len:ns+1]

# plot drivers
fig, ax = plt.subplots(len(dr),len(cols), figsize=[width, height*ratio], \
    sharex='col', sharey=True, width_ratios=cols, squeeze=False)
plt.subplots_adjust(**subplots_adjust['d'])

for s,j in zip(ds, c):
    a=alpha['+'] if s=='tr' else alpha['-']
    for d,i in zip(dr, range(len(dr))):
        ax[i,j].plot(data2[s].loc[:,d], color=col['d'], alpha=a)
        if s=='tr':
            ax[i,j].axvline(tsam[-1], color=col['sl'], lw=lw['ens'])
            ax[i,j].plot(data2[s].loc[tsam,d], 'o-', color=col['d'], ms=ms, \
                zorder=100, lw=lw['main'])
        ax[i,j].set_xlim(xlim[s])
        ax[i,j].set_xticks(xticks[s], labels=xlab[s])
        ax[i,j].set_ylim(ylim)
        ax[i,j].set_yticks([])
frame_col(fig, ax, 0)
frame_col(fig, ax, 1)        
make_invisible(ax[:,c[-1]+1:])
fig.savefig(figdir+figname.format(varname['d'], name))
plt.close(fig)

# plot sea level
i=0; d=sl
fig, ax = plt.subplots(1,len(cols), figsize=[width, height], \
    sharex='col', sharey=True, width_ratios=cols, squeeze=False)
plt.subplots_adjust(**subplots_adjust['sl'])
for s,j in zip(ds, c):
    a=alpha['+'] if s=='tr' else alpha['-']
    ax[i,j].plot(data2[s].loc[:,d], color=col['sl'], alpha=a)
    if s=='tr':
        ax[i,j].axvline(tsam[-1], color=col['sl'], lw=lw['ens'])
        ax[i,j].plot(data2[s].loc[tsam[-1:],d], 'o-', color=col['sl'], ms=ms, \
            lw=lw['main'], zorder=100, markeredgecolor='k')
    ax[i,j].set_xlim(xlim[s])
    ax[i,j].set_xticks(xticks[s], labels=xlab[s])
    ax[i,j].set_ylim(ylim)
    ax[i,j].set_yticks([])
frame_col(fig, ax, 0)
frame_col(fig, ax, 1)
make_invisible(ax[:,c[-1]+1:])
fig.savefig(figdir+figname.format(varname['sl'], name))
plt.close(fig)


# --- Test set (train is lighter) ---
# fig name; column widths; which columns to plot in; xticks&labs
name = figs['te']
cols = [tlen['tr']/tlen['all'], w*tlen['te']/tlen['all']]
c = [0,1]
xticks2 = {'tr': xticks['tr'], 'te': xticks['tew']}
xlab2 = {'tr': xlab['tr'], 'te': xlab['tew']}

# plot drivers
fig, ax = plt.subplots(len(dr),len(cols), figsize=[width, height*ratio], \
    sharex='col', sharey=True, width_ratios=cols, squeeze=False)
plt.subplots_adjust(**subplots_adjust['d'])

for s,j in zip(ds, c):
    if s=='te':
        a=alpha['+']
        linestyle='-o'
    else:
        a=alpha['-']
        linestyle='-'
    for d,i in zip(dr, range(len(dr))):
        ax[i,j].plot(data2[s].loc[:,d], linestyle, color=col['d'], alpha=a, \
            markersize=ms2)
        ax[i,j].set_xlim(xlim[s])
        ax[i,j].set_xticks(xticks2[s], labels=xlab2[s])
        ax[i,j].set_ylim(ylim)
        ax[i,j].set_yticks([])
frame_col(fig, ax, 0)
frame_col(fig, ax, 1)        
make_invisible(ax[:,c[-1]+1:])
fig.savefig(figdir+figname.format(varname['d'], name))
plt.close(fig)

# plot sea level
i=0; d=sl
fig, ax = plt.subplots(1,len(cols), figsize=[width, height], \
    sharex='col', sharey=True, width_ratios=cols, squeeze=False)
plt.subplots_adjust(**subplots_adjust['sl'])
for s,j in zip(ds, c):
    a=alpha['+'] if s=='te' else alpha['-']
    if s=='te':
        ax[i,j].plot(t, pred0, color=col['pred'], lw=lw['ens'])
    ax[i,j].plot(data2[s].loc[:,d], color=col['sl'], alpha=a, lw=lw['main'])
    ax[i,j].set_xlim(xlim[s])
    ax[i,j].set_xticks(xticks2[s], labels=xlab2[s])
    ax[i,j].set_ylim(ylim)
    ax[i,j].set_yticks([])
frame_col(fig, ax, 0)
frame_col(fig, ax, 1)
make_invisible(ax[:,c[-1]+1:])
fig.savefig(figdir+figname.format(varname['sl'], name))
plt.close(fig)


# --- Feature importance (no train set) ---
# fig name; column widths; which columns to plot in; xticks&labs
name = figs['fi']
cols = [tlen['tr']/tlen['all'], w*tlen['te']/tlen['all']]
c = 1; s = 'te'
xticks2 = {'te': xticks['tew']}
xlab2 = {'te': xlab['tew']}

# plot drivers
fig, ax = plt.subplots(len(dr),len(cols), figsize=[width, height*ratio], \
    sharex='col', sharey=True, width_ratios=cols, squeeze=False)
plt.subplots_adjust(**subplots_adjust['d'])

for d,i in zip(dr, range(len(dr))):
    if d==feat:
        x=np.random.permutation(data2[s].loc[:,d].values)
        linestyle='o'
    else:
        x=data2[s].loc[:,d].values
        linestyle='-o'
    ax[i,c].plot(time['te'], x, linestyle, color=col['d'], markersize=ms2)
    ax[i,c].set_xlim(xlim[s])
    ax[i,c].set_xticks(xticks2[s], labels=xlab2[s])
    ax[i,c].set_ylim(ylim)
    ax[i,j].set_yticks([])
frame_col(fig, ax, 1)     
make_invisible(ax[:,:c])
fig.savefig(figdir+figname.format(varname['d'], name))
plt.close(fig)

# plot sea level
i=0; d=sl
fig, ax = plt.subplots(1,len(cols), figsize=[width, height], \
    sharex='col', sharey=True, width_ratios=cols, squeeze=False)
plt.subplots_adjust(**subplots_adjust['sl'])
ax[i,c].plot(t, pred, color=col['pred'], lw=lw['ens'])
ax[i,c].plot(data2[s].loc[:,d], color=col['sl'], alpha=a, lw=lw['main'])
ax[i,c].set_xlim(xlim[s])
ax[i,c].set_xticks(xticks2[s], labels=xlab2[s])
ax[i,c].set_ylim(ylim)
ax[i,j].set_yticks([])
frame_col(fig, ax, 1)
make_invisible(ax[:,:c])
fig.savefig(figdir+figname.format(varname['sl'], name))
plt.close(fig)


# --- Sequences sketch ---
# drivers and sea level on the same figure
# 3 drivers, 2 years of data (training), seq=0 and seq=3
# fig name; column widths; which columns to plot in; time for sample sequence
name = figs['sk']
s = 'tr'
varsub = [dr[1], dr[3], dr[8], ' ', sl]; print(varsub)
seq = [0, 3]
y = [2001, 2002]
tlim = pd.to_datetime({'year':y, 'month':[1,12], 'day':[1,1]})
ind = np.where(time['tr']==\
    pd.to_datetime({'year':[y[-1]], 'month':[7], 'day':[1]})[0])[0][0]

for n in seq:
    tsam = time['tr'][ind-n-1:ind]
    
    fig, ax = plt.subplots(len(varsub),1, figsize=[width/4, height*1.5], \
        sharex='col', sharey=True, squeeze=False)
    #plt.subplots_adjust(bottom=0, top=1, left=0,right=1, wspace=0, hspace=0)
    plt.subplots_adjust(**subplots_adjust['d'])

    for v,i in zip(varsub, range(len(varsub))):
        if v==' ':
            ax[i,0].axvline(tsam[-1], color='k')
            continue
        elif v==sl:
            ax[i,0].plot(data2[s].loc[tsam[-1:],v], 'o', color=col['d'], \
                ms=ms, zorder=100)
        else:
            ax[i,0].plot(data2[s].loc[tsam,v], 'o', color=col['d'], \
                ms=ms, zorder=100)
        
        ax[i,0].plot(data2[s].loc[:,v], color=col['d'])
        ax[i,0].axvline(tsam[-1], color='k')
        ax[i,0].set_xlim(tlim)
        ax[i,0].set_ylim(ylim)
        ax[i,0].set_yticks([])
    frame_col(fig, ax, 0)
    make_invisible(ax)
    fig.savefig(figdir+figname.format('',name.format(n)))
    plt.close(fig)
