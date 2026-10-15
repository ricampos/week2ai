#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
probmaps_gefs_valdt.py

VERSION AND LAST UPDATE:
 v1.0  11/12/2025
 v2.0  02/18/2026
 v3.0  09/08/2026

PURPOSE:
 Automatic spatial validation of the week-2 GEFS probabilistic forecast
 (see probmaps.py / probmaps_gefs.py, operational).
 A forecast cycle (fcycle) made 7-14 days ago is compared against a
  pseudo ground-truth field built entirely from archived GEFS ensemble
  data (no independent obs/analysis needed).
 Ground truth is built by stitching together the first hours of lead time 
 from every archived GEFS cycle available
 Archive directory/file convention expected (same as the operational
  read_gefs() in probmaps.py):
   <gefspath>/GEFSv12Waves_<YYYYMMDD><HH>/gefs.wave.<YYYYMMDD>.<EE>.global.0p25.f<LLL>.grib2
  e.g. .../GEFSv12Waves_2026090600/gefs.wave.20260906.00.global.0p25.f006.grib2

USAGE:
 Same 5 input arguments as probmaps_gefs.py / probmaps_gefs_valdt.py v2.0:
  1) .yaml configuration file (same probmaps_gefs*.yaml used operationally,
     plus the new keys below);
  2) forecast cycle (YYYYMMDDHH) being validated;
  3) initial day of the validation window (e.g. 7);
  4) final day of the validation window (e.g. 14);
  5) variable to process: WS10 or Hs (run once per variable).

 Example:
  python3 probmaps_gefs_valdt.py probmaps_gefs_internal.yaml 2026082000 7 14 Hs

OUTPUT:
 png figures (spatial validation probability maps) saved in outpath,

DEPENDENCIES:
 See the imports below.

AUTHOR and DATE:
 11/12/2025: Ricardo M. Campos, first version.
 02/18/2026: Ricardo M. Campos, improvement in the spatial validation.
 09/08/2026: Ricardo M. Campos, ground-truth construction rewritten 

PERSON OF CONTACT:
 Ricardo M Campos: ricardo.campos@noaa.gov

"""

# Pay attention to the pre-requisites and libraries
import matplotlib
import os
matplotlib.use('Agg')
import xarray as xr
import matplotlib.pyplot as plt
import yaml
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
import numpy as np
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import shutil
import sys
import warnings; warnings.filterwarnings("ignore")
# --------------------------------------------------------------------------
sl = 13  # plot style configuration
matplotlib.rcParams.update({'font.size': sl}); plt.rc('font', size=sl)
matplotlib.rc('xtick', labelsize=sl); matplotlib.rc('ytick', labelsize=sl); matplotlib.rcParams.update({'font.size': sl})

def _read_member(fname, xrvar):
    '''Read one member file; returns (field flipped to ascending lat, lat, lon).'''
    ds = xr.open_dataset(fname, engine='cfgrib', backend_kwargs={'indexpath': ''})
    la = np.sort(np.array(ds.latitude.values))
    lo = np.array(ds.longitude.values)
    aux = np.array(np.flip(ds[xrvar].values[:], axis=0)).astype('float')
    ds.close(); del ds
    return aux, la, lo

def build_gefs_truth(gefspath, fcycle, ltime1, ltime2, tres, nenm, fvarname,
                     cycle_base=6, max_lead=48, min_members=None):
    '''
    Pseudo ground truth on the forecast time axis (every `tres` h) for
    [fcycle + (ltime1-1) days, fcycle + ltime2 days].

    For each valid time AND each member, the most recent archived cycle that
    has that member's file is used (6h grid, up to max_lead h before the
    valid time). A valid time is kept only if at least `min_members` members
    were found (default: half of nenm); members with nothing found stay NaN.

    Returns tfmod (time, member, lat, lon), lat, lon, twtime.
    '''

    fcycle_dt = pd.to_datetime(fcycle, format='%Y%m%d%H')
    win_start = fcycle_dt + pd.Timedelta(hours=(ltime1 - 1) * 24)
    win_end = fcycle_dt + pd.Timedelta(hours=ltime2 * 24)
    valid_times = pd.date_range(win_start, win_end, freq=str(int(tres)) + 'h')

    if fvarname.upper() == "WS10" or fvarname.upper() == "WND" or fvarname.upper() == "U10":
        xrvar = 'ws'
    else:
        xrvar = 'swh'

    if min_members is None:
        min_members = max(1, nenm // 2)

    root = gefspath.rstrip('/')
    nback = int(max_lead // cycle_base)

    lat = None; lon = None
    tfmod_list = []; twtime_list = []
    missing_times = []; member_fallbacks = 0; member_gaps = 0; max_lead_used = 0

    for vt in valid_times:
        init0 = vt.floor(str(int(cycle_base)) + 'h')   # latest cycle at/before vt
        slice_fmod = None
        nfound = 0

        for enm in range(0, nenm):
            for k in range(0, nback + 1):
                init = init0 - pd.Timedelta(hours=k * cycle_base)
                lead = int(round((vt - init).total_seconds() / 3600.))
                if lead > max_lead:
                    break

                cdate = init.strftime('%Y%m%d'); chour = init.strftime('%H')
                fname = (root + "/GEFSv12Waves_" + cdate + chour + "/gefs.wave." + cdate + "." +
                         str(int(enm)).zfill(2) + ".global.0p25.f" + str(int(lead)).zfill(3) + ".grib2")
                if not os.path.isfile(fname):
                    continue
                try:
                    aux, la, lo = _read_member(fname, xrvar)
                except Exception:
                    continue   # unreadable/corrupt -> older cycle for this member

                if lat is None:
                    lat = la; lon = lo
                if slice_fmod is None:
                    slice_fmod = np.full((nenm, lat.shape[0], lon.shape[0]), np.nan, 'f')
                slice_fmod[enm, :, :] = aux
                nfound += 1
                if k > 0:
                    member_fallbacks += 1
                max_lead_used = max(max_lead_used, lead)
                break
            else:
                member_gaps += 1   # no break: nothing usable within max_lead
                continue
            # (break path lands here; nothing else to do)

        if slice_fmod is None or nfound < min_members:
            missing_times.append(vt)
            continue

        tfmod_list.append(slice_fmod)
        twtime_list.append(np.datetime64(vt))

    if len(tfmod_list) == 0:
        sys.exit(" build_gefs_truth: no usable archived cycles for the validation window "
                 "(max lead " + str(max_lead) + "h).")

    print(" build_gefs_truth: " + str(len(tfmod_list)) + "/" + str(valid_times.shape[0]) +
          " valid times filled; max lead used " + str(max_lead_used) + "h; " +
          str(member_fallbacks) + " member reads used an older cycle; " +
          str(member_gaps) + " member slots left empty.")
    if len(missing_times) > 0:
        print(" build_gefs_truth: WARNING, " + str(len(missing_times)) + " valid times dropped "
              "(fewer than " + str(min_members) + " members available).")

    tfmod = np.array(tfmod_list); twtime = np.array(twtime_list)
    del tfmod_list, twtime_list

    return tfmod, lat, lon, twtime

if __name__ == "__main__":

    # Input Arguments -----
    fconfig = str(sys.argv[1])
    fcycle = str(sys.argv[2])
    fcdate = str(fcycle[0:8]); fchour = str(fcycle[8:10])
    ltime1 = int(sys.argv[3])
    ltime2 = int(sys.argv[4])
    fvarname = str(sys.argv[5])

    # Fixed configuration variables, read yaml file -----------
    print(" "); print(" Reading yaml configuration file ...")
    with open(fconfig, 'r') as file:
        wconfig = yaml.safe_load(file)

    ftag = str(wconfig['ftag'])
    mode = str(wconfig['mode'])
    nenm = wconfig['nenm']
    tres = wconfig['tres']
    nmax = wconfig['nmax']
    spws = wconfig['spws']
    slonmin = wconfig['lonmin'] - spws; slonmax = wconfig['lonmax'] + spws
    slatmin = wconfig['latmin'] - spws; slatmax = wconfig['latmax'] + spws
    spctl = wconfig['spctl']
    gft = wconfig['gft']
    plevels = np.array(wconfig['plevels']).astype('float')
    hplevels = np.array(wconfig['hplevels']).astype('float')
    pcolors = np.array(wconfig['pcolors']).astype('str')
    hpcolors = np.array(wconfig['hpcolors']).astype('str')

    # Forecast archive path (files created by download_GEFSwaves.sh)
    gefspath = wconfig['mpath'] if 'mpath' in wconfig else wconfig['gefspath']
    if gefspath[-1] != '/':
        gefspath = gefspath + "/"

    outpath = str(wconfig['outpath'])
    if outpath[-1] != '/':
        outpath = outpath + "/"

    umf = 1.  # unit conversion, when necessary

    if fvarname.upper() == "WS10" or fvarname.upper() == "WND" or fvarname.upper() == "U10":
        qqvmax = wconfig['qqvmax_wnd']
        qlev = np.array(wconfig['qlev_wnd']).astype('float')
        funits = str('knots')
        umf = 1.94  # m/s to knots
    elif fvarname.upper() == "HS":
        qqvmax = wconfig['qqvmax_hs']
        funits = str('m')
        qlev = np.array(wconfig['qlev_hs']).astype('float')
    else:
        sys.exit(" Input variable " + fvarname + " not included in the list. Please select only one: WS10, Hs.")

    # bias correction, configurable
    bc_slope = 1.05
    bc_intercept = -0.20

    print(" Reading yaml configuration file, OK."); print(" ")

    # Time range forecast intervall string, for the plots
    if (ltime2 - ltime1) < 0:
        aux = np.copy(ltime1); ltime1 = np.copy(ltime2)
        ltime2 = np.copy(aux); del aux

    if ltime1 == 1 and (ltime2 - ltime1) == 7:
        trfi = str("Week 1 - ")
    elif ltime1 == 7 and (ltime2 - ltime1) == 7:
        trfi = str("Week 2 - ")
    elif ltime1 == 14 and (ltime2 - ltime1) == 7:
        trfi = str("Week 3 - ")
    elif ltime1 == 21 and (ltime2 - ltime1) == 7:
        trfi = str("Week 4 - ")
    elif ltime1 == 28 and (ltime2 - ltime1) == 7:
        trfi = str("Week 5 - ")
    elif (ltime2 - ltime1) > 0:
        trfi = str("Days " + str(ltime1) + "-" + str(ltime2) + " , ")
    elif ltime2 == ltime1:
        trfi = str("Day " + str(ltime1) + " , ")
    else:
        trfi = ''

    print(" "); print(" 1. Reading Forecast Data ...")

    auxltime = np.arange((ltime1 - 1) * 24, ((ltime2) * 24) + 1, tres)
    auxltime[auxltime > 384] = 384; auxltime[auxltime < 0] = 0

    xrvar = 'swh' if fvarname.upper() == "HS" else 'ws'

    # READ GEFS Ensemble Forecast files (same convention as read_gefs in probmaps.py)
    c = 0
    for t in range(0, auxltime.shape[0]):
        for enm in range(0, nenm):
            fname = (gefspath + "GEFSv12Waves_" + fcdate + fchour + "/gefs.wave." + fcdate + "." +
                     str(int(enm)).zfill(2) + ".global.0p25.f" + str(int(auxltime[t])).zfill(3) + ".grib2")
            if c == 0:
                ds = xr.open_dataset(fname, engine='cfgrib', backend_kwargs={'indexpath': ''})
                wtime = np.atleast_1d(np.array(ds.time.values))
                lat = np.array(ds.latitude.values); lat = np.sort(lat); lon = np.array(ds.longitude.values)
                fmod = np.zeros((auxltime.shape[0], nenm, lat.shape[0], lon.shape[0]), 'f') * np.nan
                ds.close(); del ds

            ds = xr.open_dataset(fname, engine='cfgrib', backend_kwargs={'indexpath': ''})
            fmod[t, enm, :, :] = np.array(np.flip(ds[xrvar].values[:], axis=0)).astype('float')
            ds.close(); del ds
            c = c + 1
        print(repr(t))

    # Quick simple quality control
    fmod[fmod >= qqvmax] = np.nan; fmod[fmod < 0.] = np.nan
    # Unit conversion
    fmod = fmod * umf
    # Select domain of interest
    indlat = np.where((lat >= (slatmin - spws)) & (lat <= (slatmax + spws)))
    indlon = np.where((lon >= (slonmin - spws)) & (lon <= (slonmax + spws)))
    if np.size(indlon) > 0 and np.size(indlat) > 0:
        lat = np.copy(lat[indlat[0]]); lon = np.copy(lon[indlon[0]])
        fmod = np.copy(fmod[:, :, indlat[0], :][:, :, :, indlon[0]])
    else:
        sys.exit(" Min/Max lat and lon incorrect when applied to the forecast file. Check longitude standards.")

    print(" 1. Forecast Data ... OK"); print(" ")

    print(" 2. Build ground truth from archived GEFS cycles ...")
    tfmod, tlat, tlon, twtime = build_gefs_truth(gefspath, fcycle, ltime1, ltime2, tres, nenm,
                                                 fvarname, cycle_base=6, max_lead=48)

    # Sanity check: truth grid must match the forecast grid
    if tlat.shape[0] != (lat.shape[0] + np.size(np.where((tlat < (slatmin - spws)) | (tlat > (slatmax + spws))))):
        pass  # grids can differ in extent before subsetting; only shape/spacing matters, checked implicitly below
    tindlat = np.where((tlat >= (slatmin - spws)) & (tlat <= (slatmax + spws)))
    tindlon = np.where((tlon >= (slonmin - spws)) & (tlon <= (slonmax + spws)))
    if np.size(tindlon) == 0 or np.size(tindlat) == 0:
        sys.exit(" Min/Max lat and lon incorrect when applied to the truth/reference file. Check longitude standards.")

    tfmod = np.copy(tfmod[:, :, tindlat[0], :][:, :, :, tindlon[0]])
    # Quick simple quality control (same thresholds as the forecast field)
    tfmod[tfmod >= qqvmax] = np.nan; tfmod[tfmod < 0.] = np.nan
    tfmod = tfmod * umf

    # Reduce the stitched truth series to a single representative field per
    # member (mean of the top-nmax instances across the validation window),
    # then apply the linear bias correction.
    hfmod = np.nanmean(np.sort(tfmod, axis=0)[-nmax::, :, :, :], axis=0)
    hfmod = hfmod * bc_slope + bc_intercept

    del tfmod, tindlat, tindlon, indlat, indlon
    print(" 2. Ground truth ... OK"); print(" ")

    print(" "); print(" 3. Space-Time Cells and Probabilities ...")

    # n-max expansion and reshape (forecast field, same as operational probmaps.py)
    fmod = np.sort(fmod, axis=0)
    fmod = np.copy(fmod[-nmax::, :, :, :])
    fmod = np.array(fmod.reshape(nmax * nenm, lat.shape[0], lon.shape[0]))
    fmod[fmod > 200] = np.nan

    gspws = int(np.floor(spws / np.diff(lat).mean()) / 2)

    probecdf = np.zeros((qlev.shape[0], lat.shape[0], lon.shape[0]), 'f')
    for i in range(0, qlev.shape[0]):
        for j in range(0, lat.shape[0]):
            for k in range(0, lon.shape[0]):
                if np.any(fmod[:, j, k] > 0.0):
                    if (j >= gspws) and (j <= lat.shape[0] - gspws) and (k >= gspws) and (k <= lon.shape[0] - gspws):
                        aux = np.array(fmod[:, (j - gspws):(j + gspws + 1), :][:, :, (k - gspws):(k + gspws + 1)])
                        aux = aux.reshape(nmax * nenm, aux.shape[1] * aux.shape[2])
                        aux = np.sort(aux, axis=1)
                        ind = np.where(np.mean(aux, axis=0) >= 0.)
                        if np.size(ind) > 0:
                            aux = np.array(aux[:, ind[0]])
                            aux = np.array(aux[:, int(np.floor(aux.shape[1] * (spctl / 100)))::])
                            aux = aux.reshape(aux.shape[0] * aux.shape[1])
                            probecdf[i, j, k] = np.size(aux[aux > qlev[i]]) / np.size(aux[aux > 0.])
                        del aux

    print(" 3. Space-Time Cells and Probabilities ... OK"); print(" ")

    # PLOTS
    print(" 4. Probability Maps ...")
 
    clabels = []; clevels = []
    for j in range(0, np.size(plevels) - 1):
        clabels = np.append(clabels, ">" + str(int(plevels[j] * 100)).zfill(2) + "%")
        clevels = np.append(clevels, (plevels[j] + plevels[j + 1]) / 2)
 
    cmap = ListedColormap(pcolors)
 
    for i in range(0, qlev.shape[0]):
 
        if i == int(qlev.shape[0] - 1):
            plevels = hplevels
            pcolors = hpcolors
            clabels = []; clevels = []
            for j in range(0, np.size(plevels) - 1):
                clabels = np.append(clabels, ">" + str(int(plevels[j] * 100)) + "%")
                clevels = np.append(clevels, (plevels[j] + plevels[j + 1]) / 2)
            cmap = ListedColormap(pcolors)
 
        plt.figure(figsize=(9, 5.5))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=-90))
        ax.set_extent([slonmin + spws, slonmax - spws, slatmin + spws, slatmax - spws], crs=ccrs.PlateCarree())
        gl = ax.gridlines(crs=ccrs.PlateCarree(), xlocs=range(-180, 180, 20), draw_labels=True,
                           linewidth=0.5, color='grey', alpha=0.5, linestyle='--')
        gl.xlabel_style = {'size': 9, 'color': 'k', 'rotation': 0}; gl.ylabel_style = {'size': 9, 'color': 'k', 'rotation': 0}
        ax.add_feature(cartopy.feature.OCEAN, facecolor=("white"))
        ax.add_feature(cartopy.feature.LAND, facecolor=("lightgrey"), edgecolor='grey', linewidth=0.5, zorder=3)
        ax.add_feature(cartopy.feature.BORDERS, edgecolor='silver', linestyle='-', linewidth=0.5, alpha=1, zorder=3)
        ax.coastlines(resolution='50m', color='silver', linewidth=0.5, linestyle='-', alpha=0.5, zorder=4)
        title = "Prob " + fvarname + ">" + str(qlev[i]).zfill(1) + funits + ", Cycle " + fcycle[0:8] + " " + fcycle[8:10] + "Z \n"
        title += r"$\bf{" + trfi + "Valid: " + pd.to_datetime(wtime[0] + np.timedelta64(ltime1, 'D')).strftime('%B %d, %Y') + " - "
        title += pd.to_datetime(wtime[0] + np.timedelta64(ltime2, 'D')).strftime('%B %d, %Y') + "}$"
        cs = ax.contourf(lon, lat, gaussian_filter(probecdf[i, :, :], gft), levels=plevels, alpha=0.7,
                          cmap=cmap, zorder=1, transform=ccrs.PlateCarree())

        for j in range(0, nenm):
            if np.all(np.isnan(hfmod[j, :, :])):
                continue
            ax.contour(lon, lat, hfmod[j, :, :], levels=[qlev[i]], colors='dimgrey', alpha=0.7,
                       linewidths=1, zorder=2, transform=ccrs.PlateCarree())
        
        ax.contour(lon, lat, np.nanmean(hfmod, axis=0), levels=[qlev[i]], colors='k', linewidths=1.5,
                   zorder=2, transform=ccrs.PlateCarree())

        ax.set_title(title); del title
        plt.tight_layout()
        ax2 = plt.gca(); pos = ax2.get_position(); l, b, w, h = pos.bounds
        cax = plt.axes([l + 0.06, b - 0.07, w - 0.12, 0.03])
        cbar = plt.colorbar(cs, cax=cax, orientation='horizontal', ticks=clevels, format='%g')
        cbar.ax.set_xticklabels(clabels)
        cbar.ax.tick_params(length=0)
        for label in cbar.ax.get_xticklabels():
            label.set_weight('bold')
 
        plt.axes(ax2); plt.tight_layout()
        plt.text(-90., 76., 'Validation', color='k', fontsize=13, fontweight='bold')
 
        figname = (outpath + "ProbMap_SpatialValidation_" + fcycle[0:8] + "_" + fvarname + "_" +
                   str(qlev[i]).zfill(1) + "_fcst" +
                   pd.to_datetime(wtime[0] + np.timedelta64(ltime1, 'D')).strftime('%Y%m%d') + "to" +
                   pd.to_datetime(wtime[0] + np.timedelta64(ltime2, 'D')).strftime('%Y%m%d') + "_" + ftag)
        plt.savefig(figname + ".png", dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
                    format='png', transparent=False, bbox_inches='tight', pad_inches=0.1)
 
        # Also save a stable, date-free filename. `figname`
        figname_latest = (outpath + "ProbMap_SpatialValidation_" + fvarname + "_" +
                           str(qlev[i]).zfill(1) + "_latest_GEFS_" + ftag)
        shutil.copyfile(figname + ".png", figname_latest + ".png")
 
        plt.close('all')
        del ax2, figname, figname_latest
        print("   Plot ... qlev " + repr(qlev[i]))
 
    print(" 4. Probability Plots ... OK"); print(" ")

