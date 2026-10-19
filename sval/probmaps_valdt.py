#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
probmaps_gefs_valdt.py

VERSION AND LAST UPDATE:
 v1.0  11/12/2025
 v1.1  02/18/2026
 v1.2  09/08/2026
 v2.0  10/19/2026

PURPOSE:
 Automatic spatial validation of the week-2 probabilistic forecast
 (see probmaps.py / probmaps_gefs.py, operational).
 A forecast cycle (fcycle) made 7-14 days ago is compared against a
  pseudo ground-truth field built entirely from archived ensemble
  data (no independent obs/analysis needed).
 Ground truth is built by stitching together the first hours of lead time 
 from every archived forecast cycle available
 Archive directory/file convention expected (same as the operational
  read_gefs() in probmaps.py).

USAGE:
 Same 5 input arguments as probmaps_gefs.py / probmaps_gefs_valdt.py v2.0:
  1) .yaml configuration file (same *.yaml used operationally,
     plus the new keys below);
  2) forecast cycle (YYYYMMDDHH) being validated;
  3) initial day of the validation window (e.g. 7);
  4) final day of the validation window (e.g. 14);
  5) variable to process: WS10 or Hs (run once per variable).

 Example:
  python3 probmaps_gefs.py probmaps_gefs.yaml 2026082000 7 14 Hs

OUTPUT:
 png figures (spatial validation probability maps) saved in outpath,

DEPENDENCIES:
 See the imports below.

AUTHOR and DATE:
 11/12/2025: Ricardo M. Campos, first version.
 02/18/2026: Ricardo M. Campos, improvement in the spatial validation.
 09/08/2026: Ricardo M. Campos, ground-truth construction rewritten 
 10/19/2026: Ricardo M. Campos, new version to work with both GEFS and ECMWF ensembles

PERSON OF CONTACT:
 Ricardo M Campos: ricardo.campos@noaa.gov

"""

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
import netCDF4 as nc
import cartopy
import cartopy.crs as ccrs
import shutil
import sys
import warnings; warnings.filterwarnings("ignore")

# Plot style configuration
sl = 13
matplotlib.rcParams.update({'font.size': sl}); plt.rc('font', size=sl)
matplotlib.rc('xtick', labelsize=sl); matplotlib.rc('ytick', labelsize=sl)

# --------------------------------------------------------------------------
# READ FORECAST DATA
# --------------------------------------------------------------------------
def read_gefs(auxltime, fvarname, nenm, mpath, fcdate, fchour):
    '''Read GEFS grib2 ensemble forecast files.'''
    c = 0
    xrvar = 'swh' if fvarname.upper() in ["HS", "HOS"] else 'ws'
    for t in range(0, auxltime.shape[0]):
        for enm in range(0, nenm):
            fname = f"{mpath}GEFSv12Waves_{fcdate}{fchour}/gefs.wave.{fcdate}.{str(int(enm)).zfill(2)}.global.0p25.f{str(int(auxltime[t])).zfill(3)}.grib2"
            if c == 0:
                ds = xr.open_dataset(fname, engine='cfgrib')
                wtime = np.atleast_1d(np.array(ds.time.values))
                lat = np.sort(np.array(ds.latitude.values))
                lon = np.array(ds.longitude.values)
                fmod = np.zeros((auxltime.shape[0], nenm, lat.shape[0], lon.shape[0]), 'f') * np.nan
                ds.close(); del ds

            ds = xr.open_dataset(fname, engine='cfgrib')
            fmod[t, enm, :, :] = np.array(np.flip(ds[xrvar].values[:], axis=0)).astype('float')
            ds.close(); del ds
            c += 1
    return fmod, lat, lon, wtime

def read_mm(auxltime, model, fvarname, mvar, nenm, mpath, fcdate, fchour):
    '''Read ECMWF and EnvCanada netcdf ensemble forecast files.'''
    auxvarn = 'wind' if fvarname.upper() in ["WS10", "WND", "U10"] else 'wave'
    mtag = 'ECMWF_ENS' if model == 'ECMWF' else ('GEWPS' if auxvarn == 'wave' else 'GEPS')
    omvar = mvar
    c = 0
    for enm in range(0, nenm):
        fname = f"{mpath}{auxvarn}/{fchour}/{mtag}_{auxvarn}_{fcdate}{fchour}.{str(enm).zfill(2)}.nc"
        try:
            f = nc.Dataset(fname)
            if omvar.split(' ')[0] in list(f.variables.keys()):
                if c == 0:
                    ftime = np.array(f.variables['time'][:]).astype('int')
                    indt = np.where(np.isin(ftime, auxltime))[0]
                    lat = np.sort(np.array(f.variables['lat'][:]))
                    lon = np.array(f.variables['lon'][:]); lon = lon + 180.; half = len(lon) // 2
                    fmod = np.zeros((len(indt), nenm, len(lat), len(lon)), 'f')
                    wtime = np.atleast_1d(np.datetime64(pd.to_datetime(fcdate + fchour, format='%Y%m%d%H')))

                if auxvarn == 'wind':
                    mvar_arr = np.array(omvar.split(' ')).astype('str')
                    auxu = np.array(f.variables[mvar_arr[0]][indt, :, :])
                    auxv = np.array(f.variables[mvar_arr[1]][indt, :, :])
                    aux = np.sqrt(auxu**2 + auxv**2)[:, 0, :, :]
                    if model == 'ECMWF':
                        aux = np.flip(aux, axis=1)
                else:
                    aux = np.flip(np.array(f.variables[mvar][indt, :, :]), axis=1) if model == 'ECMWF' else np.array(f.variables[mvar][indt, :, :])

                aux = np.roll(aux, shift=half, axis=2)
                aux[aux < -100] = np.nan; aux[aux > 100] = np.nan
                fmod[:, enm, :, :] = aux
                f.close()
            else:
                sys.exit(f" Variable {mvar} not found in {fname}")
        except Exception as e:
            sys.exit(f" Could not read {fname}: {e}")
        c += 1
    return fmod, lat, lon, wtime

# --------------------------------------------------------------------------
# GROUND TRUTH CONSTRUCTION
# --------------------------------------------------------------------------
def _read_member_truth(fname, model, xrvar, mvar, auxvarn):
    '''Helper reader for individual member ground-truth files.'''
    if model == 'GEFS':
        ds = xr.open_dataset(fname, engine='cfgrib', backend_kwargs={'indexpath': ''})
        la = np.sort(np.array(ds.latitude.values))
        lo = np.array(ds.longitude.values)
        aux = np.array(np.flip(ds[xrvar].values[:], axis=0)).astype('float')
        ds.close()
    else:  # ECMWF / NetCDF
        f = nc.Dataset(fname)
        la = np.sort(np.array(f.variables['lat'][:]))
        lo = np.array(f.variables['lon'][:]) + 180.
        half = len(lo) // 2
        if auxvarn == 'wind':
            mvar_arr = np.array(mvar.split(' ')).astype('str')
            auxu = np.array(f.variables[mvar_arr[0]][0, :, :])
            auxv = np.array(f.variables[mvar_arr[1]][0, :, :])
            aux = np.sqrt(auxu**2 + auxv**2)
            if model == 'ECMWF':
                aux = np.flip(aux, axis=0)
        else:
            aux = np.flip(np.array(f.variables[mvar][0, :, :]), axis=0) if model == 'ECMWF' else np.array(f.variables[mvar][0, :, :])
        aux = np.roll(aux, shift=half, axis=1)
        f.close()
    return aux, la, lo

def build_truth(mpath, model, fcycle, ltime1, ltime2, tres, nenm, fvarname, mvar,
                cycle_base=6, max_lead=48, min_members=None):
    '''Builds pseudo ground truth for both GEFS and ECMWF archived runs.'''
    fcycle_dt = pd.to_datetime(fcycle, format='%Y%m%d%H')
    win_start = fcycle_dt + pd.Timedelta(hours=(ltime1 - 1) * 24)
    win_end = fcycle_dt + pd.Timedelta(hours=ltime2 * 24)
    valid_times = pd.date_range(win_start, win_end, freq=f"{int(tres)}h")

    auxvarn = 'wind' if fvarname.upper() in ["WS10", "WND", "U10"] else 'wave'
    xrvar = 'ws' if auxvarn == 'wind' else 'swh'
    mtag = 'ECMWF_ENS' if model == 'ECMWF' else ('GEWPS' if auxvarn == 'wave' else 'GEPS')

    if min_members is None:
        min_members = max(1, nenm // 2)

    root = mpath.rstrip('/')
    nback = int(max_lead // cycle_base)
    lat = None; lon = None
    tfmod_list = []; twtime_list = []

    for vt in valid_times:
        init0 = vt.floor(f"{int(cycle_base)}h")
        slice_fmod = None
        nfound = 0

        for enm in range(0, nenm):
            for k in range(0, nback + 1):
                init = init0 - pd.Timedelta(hours=k * cycle_base)
                lead = int(round((vt - init).total_seconds() / 3600.))
                if lead > max_lead:
                    break

                cdate = init.strftime('%Y%m%d'); chour = init.strftime('%H')
                if model == 'GEFS':
                    fname = f"{root}/GEFSv12Waves_{cdate}{chour}/gefs.wave.{cdate}.{str(int(enm)).zfill(2)}.global.0p25.f{str(int(lead)).zfill(3)}.grib2"
                else:
                    fname = f"{root}/{auxvarn}/{chour}/{mtag}_{auxvarn}_{cdate}{chour}.{str(enm).zfill(2)}.nc"

                if not os.path.isfile(fname):
                    continue
                try:
                    aux, la, lo = _read_member_truth(fname, model, xrvar, mvar, auxvarn)
                except Exception:
                    continue

                if lat is None:
                    lat = la; lon = lo
                if slice_fmod is None:
                    slice_fmod = np.full((nenm, lat.shape[0], lon.shape[0]), np.nan, 'f')
                slice_fmod[enm, :, :] = aux
                nfound += 1
                break

        if slice_fmod is None or nfound < min_members:
            continue

        tfmod_list.append(slice_fmod)
        twtime_list.append(np.datetime64(vt))

    if len(tfmod_list) == 0:
        sys.exit(f" build_truth: no usable archived cycles found for model {model}.")

    return np.array(tfmod_list), lat, lon, np.array(twtime_list)

# --------------------------------------------------------------------------
# MAIN PROCESS
# --------------------------------------------------------------------------
if __name__ == "__main__":

    fconfig = str(sys.argv[1])
    fcycle = str(sys.argv[2])
    fcdate = str(fcycle[0:8]); fchour = str(fcycle[8:10])
    ltime1 = int(sys.argv[3])
    ltime2 = int(sys.argv[4])
    fvarname = str(sys.argv[5])

    print("\n Reading yaml configuration file ...")
    with open(fconfig, 'r') as file:
        wconfig = yaml.safe_load(file)

    ftag = str(wconfig['ftag'])
    model = str(wconfig.get('model', 'GEFS'))
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

    mpath = wconfig['mpath'] if 'mpath' in wconfig else wconfig['gefspath']
    if mpath[-1] != '/': mpath += "/"

    outpath = str(wconfig['outpath'])
    if outpath[-1] != '/': outpath += "/"

    umf = 1.
    if fvarname.upper() in ["WS10", "WND", "U10"]:
        qqvmax = wconfig['qqvmax_wnd']
        mvar = wconfig.get('mvar_wnd', 'ws')
        qlev = np.array(wconfig['qlev_wnd']).astype('float')
        funits = 'knots'
        umf = 1.94
    elif fvarname.upper() == "HS":
        qqvmax = wconfig['qqvmax_hs']
        mvar = wconfig.get('mvar_hs', 'swh')
        funits = 'm'
        qlev = np.array(wconfig['qlev_hs']).astype('float')
    else:
        sys.exit(f" Input variable {fvarname} not supported.")

    bc_slope = 1.05
    bc_intercept = -0.20

    trfi = f"Days {ltime1}-{ltime2} , " if ltime2 != ltime1 else f"Day {ltime1} , "

    print("\n 1. Reading Forecast Data ...")
    auxltime = np.arange((ltime1 - 1) * 24, ((ltime2) * 24) + 1, tres)
    auxltime[auxltime > 384] = 384; auxltime[auxltime < 0] = 0

    if model == 'GEFS':
        fmod, lat, lon, wtime = read_gefs(auxltime, fvarname, nenm, mpath, fcdate, fchour)
    else:
        fmod, lat, lon, wtime = read_mm(auxltime, model, fvarname, mvar, nenm, mpath, fcdate, fchour)

    fmod[fmod >= qqvmax] = np.nan; fmod[fmod < 0.] = np.nan
    fmod = fmod * umf

    indlat = np.where((lat >= (slatmin - spws)) & (lat <= (slatmax + spws)))
    indlon = np.where((lon >= (slonmin - spws)) & (lon <= (slonmax + spws)))
    lat = lat[indlat[0]]; lon = lon[indlon[0]]
    fmod = fmod[:, :, indlat[0], :][:, :, :, indlon[0]]

    print("\n 2. Building ground truth from archived cycles ...")
    tfmod, tlat, tlon, twtime = build_truth(mpath, model, fcycle, ltime1, ltime2, tres, nenm, fvarname, mvar)

    tindlat = np.where((tlat >= (slatmin - spws)) & (tlat <= (slatmax + spws)))
    tindlon = np.where((tlon >= (slonmin - spws)) & (tlon <= (slonmax + spws)))
    tfmod = tfmod[:, :, tindlat[0], :][:, :, :, tindlon[0]]
    tfmod[tfmod >= qqvmax] = np.nan; tfmod[tfmod < 0.] = np.nan
    tfmod = tfmod * umf

    hfmod = np.nanmean(np.sort(tfmod, axis=0)[-nmax::, :, :, :], axis=0)
    hfmod = hfmod * bc_slope + bc_intercept

    print("\n 3. Space-Time Cells and Probabilities ...")
    fmod = np.sort(fmod, axis=0)[-nmax::, :, :, :]
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

    print("\n 4. Plotting Spatial Validation Maps ...")
    for i in range(0, qlev.shape[0]):
        if i == int(qlev.shape[0] - 1):
            plevels = hplevels
            pcolors = hpcolors
        
        clabels = [f">{int(p*100)}%" for p in plevels[:-1]]
        clevels = [(plevels[j] + plevels[j + 1]) / 2 for j in range(len(plevels) - 1)]
        cmap = ListedColormap(pcolors)

        plt.figure(figsize=(9, 5.5))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=-90))
        ax.set_extent([slonmin + spws, slonmax - spws, slatmin + spws, slatmax - spws], crs=ccrs.PlateCarree())
        gl = ax.gridlines(crs=ccrs.PlateCarree(), xlocs=range(-180, 180, 20), draw_labels=True,
                          linewidth=0.5, color='grey', alpha=0.5, linestyle='--')
        gl.xlabel_style = {'size': 9, 'color': 'k', 'rotation': 0}
        gl.ylabel_style = {'size': 9, 'color': 'k', 'rotation': 0}
        
        ax.add_feature(cartopy.feature.OCEAN, facecolor=("white"))
        ax.add_feature(cartopy.feature.LAND, facecolor=("lightgrey"), edgecolor='grey', linewidth=0.5, zorder=3)
        ax.add_feature(cartopy.feature.BORDERS, edgecolor='silver', linestyle='-', linewidth=0.5, alpha=1, zorder=3)
        ax.coastlines(resolution='50m', color='silver', linewidth=0.5, linestyle='-', alpha=0.5, zorder=4)

        title = f"Prob {fvarname}>{qlev[i]}{funits}, Cycle {fcycle[0:8]} {fcycle[8:10]}Z ({model})\n"
        title += r"$\bf{" + trfi + "Valid: " + pd.to_datetime(wtime[0] + np.timedelta64(ltime1, 'D')).strftime('%B %d, %Y') + " - "
        title += pd.to_datetime(wtime[0] + np.timedelta64(ltime2, 'D')).strftime('%B %d, %Y') + "}$"

        cs = ax.contourf(lon, lat, gaussian_filter(probecdf[i, :, :], gft), levels=plevels, alpha=0.7,
                          cmap=cmap, zorder=1, transform=ccrs.PlateCarree())

        for j in range(0, nenm):
            if not np.all(np.isnan(hfmod[j, :, :])):
                ax.contour(lon, lat, hfmod[j, :, :], levels=[qlev[i]], colors='dimgrey', alpha=0.7,
                           linewidths=1, zorder=2, transform=ccrs.PlateCarree())

        ax.contour(lon, lat, np.nanmean(hfmod, axis=0), levels=[qlev[i]], colors='k', linewidths=1.5,
                   zorder=2, transform=ccrs.PlateCarree())

        ax.set_title(title)
        plt.tight_layout()
        
        ax2 = plt.gca(); pos = ax2.get_position(); l, b, w, h = pos.bounds
        cax = plt.axes([l + 0.06, b - 0.07, w - 0.12, 0.03])
        cbar = plt.colorbar(cs, cax=cax, orientation='horizontal', ticks=clevels, format='%g')
        cbar.ax.set_xticklabels(clabels)
        cbar.ax.tick_params(length=0)
        for label in cbar.ax.get_xticklabels():
            label.set_weight('bold')

        plt.axes(ax2); plt.tight_layout()
        plt.text(-90., 76., f'Validation ({model})', color='k', fontsize=13, fontweight='bold')

        figname = f"{outpath}ProbMap_SpatialValidation_{fcycle[0:8]}_{fvarname}_{qlev[i]}_fcst_{model}_{ftag}"
        plt.savefig(f"{figname}.png", dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
                    format='png', transparent=False, bbox_inches='tight', pad_inches=0.1)

        figname_latest = f"{outpath}ProbMap_SpatialValidation_{fvarname}_{qlev[i]}_latest_{model}_{ftag}"
        shutil.copyfile(f"{figname}.png", f"{figname_latest}.png")

        plt.close('all')

    print("\n Done.")

