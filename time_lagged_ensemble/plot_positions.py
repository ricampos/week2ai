#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 plot_positions.py
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import yaml
from wcpval import *
import gc
import warnings; warnings.filterwarnings("ignore")
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature
from matplotlib import ticker
from matplotlib.patches import Circle
from pylab import *
import cartopy
import cartopy.crs as ccrs
import warnings; warnings.filterwarnings("ignore")

sl=14
matplotlib.rcParams.update({'font.size': sl}); plt.rc('font', size=sl) 
matplotlib.rc('xtick', labelsize=sl); matplotlib.rc('ytick', labelsize=sl); matplotlib.rcParams.update({'font.size': sl})

if __name__ == "__main__":

    wvar = 'hs'
    # ==== Plot point positions ====
    plt.close('all')
    plt.figure(figsize=(7, 5))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=-90))
    ax.set_extent([179, 355, 15, 72.], crs=ccrs.PlateCarree())

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'size': 9, 'color': 'k', 'rotation': 0}
    gl.ylabel_style = {'size': 9, 'color': 'k', 'rotation': 0}

    # --- SET LONGITUDE TICKS EVERY 20 DEGREES ---
    gl.xlocator = mticker.MultipleLocator(20)
    # Optional: hide top/right labels for a cleaner map layout
    gl.top_labels = False
    gl.right_labels = False
    # --------------------------------------------

    ax.add_feature(cartopy.feature.OCEAN, facecolor=("white"))
    ax.add_feature(cartopy.feature.LAND, facecolor=("lightgrey"), edgecolor='grey', linewidth=0.5, zorder=2)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='grey', linestyle='-', linewidth=0.5, alpha=1, zorder=3)
    ax.coastlines(resolution='50m', color='dimgrey', linewidth=0.5, linestyle='-', alpha=1, zorder=4)

    fobsname = "/work/noaa/marine/ricardo.campos/work/analysis/3assessments/fuzzy_verification/data/Pacific/Altimeter.Buoy.PointExtract_20201001to20260101.nc"
    obs = read_obs(fobsname, wvar, "max")
    bid = np.array([1, 3, 4, 5, 8, 10, 13]).astype('int')  # Pacific
    for i in range(0, len(bid)):
        blat = float(obs['lat'][bid][i]); blon = float(obs['lon'][bid][i])
        ax.scatter(blon, blat, s=9, c='navy', marker='o', transform=ccrs.PlateCarree(), zorder=3)

    fobsname = "/work/noaa/marine/ricardo.campos/work/analysis/3assessments/fuzzy_verification/data/Atlantic/Altimeter.Buoy.PointExtract_20201001to20260101.nc"
    obs = read_obs(fobsname, wvar, "max")
    bid = np.array([12,15,16,19,20,21,22,24,25,31]).astype('int')  # Atlantic na_center_north
    for i in range(0, len(bid)):
        blat = float(obs['lat'][bid][i]); blon = float(obs['lon'][bid][i])
        ax.scatter(blon, blat, s=9, c='navy', marker='o', transform=ccrs.PlateCarree(), zorder=3)

    plt.savefig('Point_Positions.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format='png', transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.close()

