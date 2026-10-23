#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wcpval.py

VERSION AND LAST UPDATE:
 v1.0  10/30/2023
 v1.1  11/03/2023
 v1.2  05/21/2024
 v1.3  01/10/2025
 v1.4  05/01/2025
 v1.5  07/08/2025
 v2.0  10/08/2026

PURPOSE:
 Validation of long-term probabilistic wave forecasts using 
 fuzzy verification. This can be used to assess and optimize
 the parameters of operational probability maps.
 The probability maps are based on GEFSv12 and have been validated against
  NDBC buoys, GDAS, and GEFS hindcasts by appending 24-hour segments 
  from consecutive cycles.

USAGE:
 These are auxiliar functions used by the main script fuzzy_verification_GEFS.py

DEPENDENCIES:
 See the imports below.

AUTHOR and DATE:
 10/30/2023: Ricardo M. Campos, first version, named wprob.py.
 11/03/2023: Ricardo M. Campos, fuzzy verification including categorical 
  and probabilistic assessments.
 05/21/2024: Ricardo M. Campos, renamed to wcpval.py, new reliability curve plot.
 01/10/2025: Ricardo M. Campos, climatology and persistence added to reliability diagrams and ROC
  plots. New updates added to the reliability diagram plots.
 05/01/2025: Ricardo M. Campos, edit in read_data (dependence on NDBC buoy data has been removed).
 07/08/2025: Ricardo M. Campos, bias correction applied to the hindcast used as ground truth.
 10/08/2026: Ricardo M. Campos, added new functions to improve memory load and efficiency.

PERSON OF CONTACT:
 Ricardo M Campos: ricardo.campos@noaa.gov

"""

import matplotlib
# matplotlib.use('Agg')
from matplotlib.dates import DateFormatter
import netCDF4 as nc
import xarray as xr
import numpy as np
import pandas as pd
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import properscoring as ps
from datetime import datetime
import yaml
import mvalstats
import pvalstats
from pvalstats import ModelObsPlot
from lmoments3 import lmom_ratios
import warnings; warnings.filterwarnings("ignore")

sl=13
matplotlib.rcParams.update({'font.size': sl}); plt.rc('font', size=sl) 
matplotlib.rc('xtick', labelsize=sl); matplotlib.rc('ytick', labelsize=sl); matplotlib.rcParams.update({'font.size': sl})

pcolors = np.array(['navy', 'firebrick', 'dimgrey', 'darkgreen', 'fuchsia', 'gold', 'blue', 'salmon', 'lime', 'darkviolet', 'yellow',
    'cornflowerblue', 'red', 'green', 'violet', 'orange'])

lstl  = np.array(['-','--','-.'])

# READ OBSERVATIONS 
def read_obs(fname,wvar,altselm):
    '''
    wvar: hs u10
    altselm: mean, p80, p90, p95, p95, max
    '''
    ds = xr.open_dataset(fname)
    f = nc.Dataset(fname)

    result = {'bdate': ds.time.values,'btime': f.variables['time'][:], 'bid': f.variables['ID'][:],
        'lat': f.variables['lat'][:], 'lon': f.variables['lon'][:], 
        'mlat': f.variables['mlat'][:], 'mlon': f.variables['mlon'][:], 
        'buoy': f.variables["buoy_"+wvar][:], 'alt': f.variables["alt_"+wvar+"_"+altselm][:]}

    return result

# READ DATA and Bias Correction
def _init_worker(bid, wvar, indt_read, indlat, indlon, latm, lonm, ensm_len,
                  stations_with_obs, obs_by_station, obs_btime, include_buoy):
    """Runs once per worker process, stashes shared read-only context in globals
    so we don't re-pickle/re-send it on every one of the 3800 file tasks."""
    global _G
    _G = dict(bid=bid, wvar=wvar, indt_read=indt_read, indlat=indlat, indlon=indlon,
               latm=latm, lonm=lonm, ensm_len=ensm_len,
               stations_with_obs=stations_with_obs, obs_by_station=obs_by_station,
               obs_btime=obs_btime, include_buoy=include_buoy)

def _read_and_match_one_file(fname):
    """
    Runs in a worker process for ONE file:
    - reads that file's full-grid hindcast (small: ~tens of MB) and center-point forecast
    - immediately mines it for buoy/altimeter matches against obs (same logic as the
      old bias_correction inner loop), station by station
    - returns only small results: center-point series + per-station match rows
    The full-grid array (hc_full) is never returned or kept — it's discarded when
    this function returns, so nothing multi-file-sized ever exists at once.
    """
    global _G
    bid = _G['bid']; wvar = _G['wvar']; indt_read = _G['indt_read']
    indlat = _G['indlat']; indlon = _G['indlon']
    latm = _G['latm']; lonm = _G['lonm']; ensm_len = _G['ensm_len']
    stations_with_obs = _G['stations_with_obs']
    obs_by_station = _G['obs_by_station']; obs_btime = _G['obs_btime']
    include_buoy = _G['include_buoy']

    f = nc.Dataset(fname)
    try:
        ftime_i = np.array(f.variables['time'][indt_read]).astype('double')
        ctime_i = np.double(f.variables['time'][0])
        # (nstations, nlead, nens, nlat, nlon) — one file only, small
        hc_full = np.array(f.variables[wvar + "_gefs_hindcast"][bid, indt_read, :, :, :], dtype='f')
        fc_center = np.array(f.variables[wvar + "_gefs_forecast"][bid, indt_read, :, indlat, indlon], dtype='f')
    finally:
        f.close()

    hc_center = np.array(hc_full[:, :, :, indlat, indlon])  # (nstations, nlead, nens)
    nlead = hc_full.shape[1]

    # per-station match accumulation for this file only
    match_results = {}  # i -> (fmodel_rows, fobs_rows)

    for i in stations_with_obs:
        sobs = obs_by_station[i]
        fmodel_rows = []
        fobs_rows = []
        neighbor_cache = {}  # (indplat, indplon, passed) per altimeter point p — same for every k in this file

        for k in range(nlead):
            indt = np.where(np.abs(ftime_i[k] - obs_btime) < 1800.)
            if np.size(indt) > 0:
                indt = indt[0][0]

                if include_buoy == 'yes' and sobs['buoy'][indt] > 0.:
                    fmodel_rows.append(hc_full[i, k, :, indlat, indlon])
                    fobs_rows.append(np.full(ensm_len, sobs['buoy'][indt]))

                if np.any(np.isnan(sobs['alt'][indt, :]) == False):
                    for p in range(sobs['mlat'].shape[0]):
                        if sobs['alt'][indt, :][p] > 0.:

                            if p not in neighbor_cache:
                                indplat = np.where(
                                    np.abs(latm[i, :] - sobs['mlat'][p]) ==
                                    np.min(np.abs(latm[i, :] - sobs['mlat'][p]))
                                )[0][0]
                                indplon = np.where(
                                    np.abs(lonm[i, :] - sobs['mlon'][p]) ==
                                    np.min(np.abs(lonm[i, :] - sobs['mlon'][p]))
                                )[0][0]

                                stresult = np.round(np.abs(mvalstats.metrics(
                                    hc_full[i, :, 0, indlat, indlon],
                                    hc_full[i, :, 0, indplat, indplon]
                                )[[2, 5, 7]]), 3)
                                L1, L2, t3_p, t4_p = lmom_ratios(hc_full[i, :, 0, indlat, indlon], nmom=4)
                                l1_p = L2 / L1
                                L1, L2, t3_n, t4_n = lmom_ratios(hc_full[i, :, 0, indplat, indplon], nmom=4)
                                l1_n = L2 / L1

                                passed = (stresult[0] < 0.1 and stresult[1] < 0.1 and stresult[2] > 0.8
                                          and np.abs(l1_p - l1_n) < 0.05
                                          and np.abs(t3_p - t3_n) < 0.05
                                          and np.abs(t4_p - t4_n) < 0.05)
                                neighbor_cache[p] = (indplat, indplon, passed)

                            indplat, indplon, passed = neighbor_cache[p]
                            if passed:
                                fmodel_rows.append(hc_full[i, k, :, indplat, indplon])
                                fobs_rows.append(np.full(ensm_len, sobs['alt'][indt, :][p]))

        match_results[i] = (fmodel_rows, fobs_rows)

    del hc_full  # explicit — full grid dies here, never leaves this function
    return ftime_i, ctime_i, hc_center, fc_center, match_results

def read_data_and_bias_correction(wlist, bid, ltime1, ltime2, wvar, obs, spctl, opath,
                                    include_buoy="no", max_workers=None):
    '''
    Fused replacement for read_data() + bias_correction().
    Streams through files one at a time (in parallel workers), extracting only
    center-point series and small obs-matched rows. The full-grid hindcast for
    all files together is NEVER held in memory at once — only one file's grid
    exists transiently inside each worker. This keeps peak memory tiny and
    constant regardless of how many years/stations you add later.

    Returns: gdata (metadata + center-point gefs_forecast), gefs_hindcast_bc
             (center-point, bias-corrected where obs exist).
    '''

    lstw = int(len(wlist))

    f = nc.Dataset(wlist[0])
    stations = np.array([f.variables['ID'][i] for i in bid]).astype('str')
    latm = f.variables['lat'][bid, :]; lonm = f.variables['lon'][bid, :]
    ensm = f.variables['ensemble_member'][:]
    indlat = int(np.floor(len(latm[0, :]) / 2))
    indlon = int(np.floor(len(lonm[0, :]) / 2))
    auxt = np.array(f.variables['time'][:])
    axt = np.array(auxt - auxt.min()) / 3600.
    indt = np.where((axt >= ltime1 * 24) & (axt <= ltime2 * 24))[0]
    f.close(); del f, auxt, axt

    if indt.size == 0:
        raise ValueError("indt is empty — check ltime1/ltime2 against the file's time range.")

    if np.all(np.diff(indt) == 1):
        indt_read = slice(int(indt[0]), int(indt[-1]) + 1)
    else:
        indt_read = indt

    nstations = len(bid); nlead = len(indt); ensm_len = len(ensm)

    if max_workers is None:
        max_workers = len(os.sched_getaffinity(0))

    small_gb = (lstw * nstations * nlead * ensm_len * 4 * 2) / 1e9  # forecast + hindcast center arrays
    print(f" Peak memory now bounded by per-file grids (~tens of MB each) "
          f"+ ~{small_gb:.2f} GB of center-point arrays — no more multi-file full-grid array.")

    # Which stations have matching obs, and pre-sliced per-station obs (keeps
    # worker IPC small — we never send the full obs arrays to every task)
    stations_with_obs = []
    obs_by_station = {}
    for i in range(nstations):
        if stations[i] in obs['bid']:
            inds = np.where(stations[i] == obs['bid'])[0][0]
            stations_with_obs.append(i)
            obs_by_station[i] = {
                'buoy': obs['buoy'][inds, :],
                'alt': obs['alt'][inds, :, :],
                'mlat': obs['mlat'][inds, :],
                'mlon': obs['mlon'][inds, :],
            }
    obs_btime = obs['btime']

    ctime = np.full(lstw, np.nan, dtype='double')
    ftime = np.full((lstw, nlead), np.nan, dtype='double')
    gefs_forecast = np.full((lstw, nstations, nlead, ensm_len), np.nan, dtype='f')
    hc_center_all = np.full((lstw, nstations, nlead, ensm_len), np.nan, dtype='f')

    all_fmodel = {i: [] for i in stations_with_obs}
    all_fobs = {i: [] for i in stations_with_obs}

    done_count = 0
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(bid, wvar, indt_read, indlat, indlon, latm, lonm, ensm_len,
                  stations_with_obs, obs_by_station, obs_btime, include_buoy)
    ) as ex:
        futures = {ex.submit(_read_and_match_one_file, wlist[i]): i for i in range(lstw)}
        for fut in as_completed(futures):
            i = futures[fut]
            ftime_i, ctime_i, hc_center, fc_center, match_results = fut.result()

            ftime[i, :] = ftime_i
            ctime[i] = ctime_i
            gefs_forecast[i, :, :, :] = fc_center
            hc_center_all[i, :, :, :] = hc_center

            for si, (fmodel_rows, fobs_rows) in match_results.items():
                if fmodel_rows:
                    all_fmodel[si].extend(fmodel_rows)
                    all_fobs[si].extend(fobs_rows)

            del ftime_i, ctime_i, hc_center, fc_center, match_results
            done_count += 1
            print(" read+matched file " + repr(done_count) + " of " + repr(lstw) + " (index " + repr(i) + ")")

    cdate = [datetime.utcfromtimestamp(ts) for ts in ctime]
    gdata = {'cdate': cdate, 'ctime': ctime, 'ftime': ftime, 'stations': stations, 'lft': nlead, 'ensm': ensm,
        'latm': latm, 'lonm': lonm, 'indlat': indlat, 'indlon': indlon,
        'indt': indt, 'gefs_forecast': gefs_forecast}

    # ---- Calibration step: same math as the old bias_correction second half,
    #      just applied to accumulated matches instead of a full-grid array ----
    pctlarr = np.arange(spctl, 99.9, 0.1)
    gefs_hindcast_bc = np.copy(hc_center_all)  # uncorrected default for stations without obs

    for i in stations_with_obs:
        fobs_rows = all_fobs[i]
        if len(fobs_rows) == 0:
            continue

        fmodel = np.array(all_fmodel[i])
        fobs = np.array(fobs_rows)

        fmodel_bc = np.copy(fmodel)
        for j in range(ensm_len):
            qmm_slope = 1.; qmm_intercept = 0.
            if np.size(np.where(fobs[:, j] > 0)) > 200.:
                qmm_slope, qmm_intercept = qm_train(model=fmodel[:, j], obs=fobs[:, j])
            elif np.size(np.where(fobs[:, j] > 0)) > 100.:
                qmm_intercept = np.nanmean(
                    np.nanpercentile(fobs[:, j], pctlarr) - np.nanpercentile(fmodel[:, j], pctlarr))

            gefs_hindcast_bc[:, i, :, j] = qmcal(model=hc_center_all[:, i, :, j],
                                                   slope=qmm_slope, intercept=qmm_intercept)
            fmodel_bc[:, j] = qmcal(model=fmodel[:, j], slope=qmm_slope, intercept=qmm_intercept)
            del qmm_slope, qmm_intercept
            print(" QMM bias correction station " + stations[i] + "  member" + repr(j))

        merr = np.zeros((ensm_len, 12), 'f')
        merr_cal = np.zeros((ensm_len, 12), 'f')
        for j in range(ensm_len):
            if np.any(fmodel[:, j] > -999.) == True and np.any(fobs[:, j] > -999.) == True:
                merr[j, :] = mvalstats.metrics(fmodel[:, j], fobs[:, j], pctlerr='yes')[:]
                merr_cal[j, :] = mvalstats.metrics(fmodel_bc[:, j], fobs[:, j], pctlerr='yes')[:]

        fig1 = plt.figure(1, figsize=(5, 4.5)); ax = fig1.add_subplot(111)
        ax.plot(merr[:, 0], 'k'); ax.plot(merr_cal[:, 0], 'r')
        plt.grid(c='grey', ls=':', alpha=0.5, zorder=1)
        ax.set_xlabel("Ens Members"); ax.set_ylabel("Bias")
        plt.tight_layout()
        plt.savefig(opath + "/Eval_" + wvar + "_" + stations[i] + "_Bias.png",
                    dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
                    format='png', transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig1); del fig1, ax

        fig1 = plt.figure(1, figsize=(5, 4.5)); ax = fig1.add_subplot(111)
        ax.plot(merr[:, 1], 'k'); ax.plot(merr_cal[:, 1], 'r')
        plt.grid(c='grey', ls=':', alpha=0.5, zorder=1)
        ax.set_xlabel("Ens Members"); ax.set_ylabel("Bias")
        plt.tight_layout()
        plt.savefig(opath + "/Eval_" + wvar + "_" + stations[i] + "_RMSE.png",
                    dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
                    format='png', transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig1); del fig1, ax

        fmodel_f = fmodel.flatten(); fmodel_bc_f = fmodel_bc.flatten(); fobs_f = fobs.flatten()
        mop = ModelObsPlot(model=[fmodel_f, fmodel_bc_f], obs=fobs_f, mlabels=["GEFSv12", "GEFSv12_BC"],
                            ftag=opath + "/Eval_" + stations[i] + "_", vaxisname=wvar)
        mop.qqplot(); mop.scatterplot(); mop.taylordiagram()

        del fmodel, fobs, fmodel_bc, fmodel_f, fmodel_bc_f, fobs_f, merr, merr_cal

    return gdata, gefs_hindcast_bc

def data_assimilation(gefs_hindcast_bc, gdata, obs, spctl, wvar, opath, include_buoy="yes"):
    '''
    Data assimilation of GEFS hindcast data. Using altimeter and/or buoy data.
    First test version, just replacing model by obs.

    gefs_hindcast_bc: center-point array (lstw, nstations, nlead, nens),
    as returned by the updated bias_correction().
    '''

    gefs_hindcast = np.nanmean(gefs_hindcast_bc, axis=3)
    gefs_hindcast_da = np.copy(gefs_hindcast)

    bda = 0; sda = 0
    for i in range(0, len(gdata['stations'])):
        if gdata['stations'][i] in obs['bid']:
            inds = np.where(gdata['stations'][i] == obs['bid'])[0][0]
            for j in range(0, gdata['ftime'].shape[0]):
                for k in range(0, gdata['ftime'].shape[1]):

                    indt = np.where(np.abs(gdata['ftime'][j, k] - obs['btime']) < 1800.)
                    if np.size(indt) > 0:
                        indt = indt[0][0]

                        if include_buoy == 'yes' and np.nanmean(obs['buoy'][inds, indt]) > 0.1:
                            gefs_hindcast_da[j, i, k] = np.nanmean(obs['buoy'][inds, indt])
                            bda = bda + 1

                        if np.nanmean(obs['alt'][inds, indt, 8]) > 0.1:
                            gefs_hindcast_da[j, i, k] = np.nanmean(obs['alt'][inds, indt, 8])
                            sda = sda + 1

    print(" Total of " + repr(bda) + " buoy records assimilated")
    print(" Total of " + repr(sda) + " altimeter records assimilated")

    return gefs_hindcast_da, bda, sda

# ---- Shared calibration step, used by both the center-point and windowed
#      read+bias-correction functions below, so the QMM logic lives in one
#      place only. ----
def _calibrate_stations(stations, ensm_len, stations_with_obs, all_fmodel, all_fobs,
                         hc_center_all, spctl, wvar, opath):
    """
    Applies quantile-mapping calibration per station/member, using accumulated
    obs-matched rows, and produces the diagnostic plots. Returns the
    bias-corrected center-point hindcast array.
    """
    pctlarr = np.arange(spctl, 99.9, 0.1)
    gefs_hindcast_bc = np.copy(hc_center_all)

    for i in stations_with_obs:
        fobs_rows = all_fobs[i]
        if len(fobs_rows) == 0:
            continue

        fmodel = np.array(all_fmodel[i])
        fobs = np.array(fobs_rows)

        fmodel_bc = np.copy(fmodel)
        for j in range(ensm_len):
            qmm_slope = 1.; qmm_intercept = 0.
            if np.size(np.where(fobs[:, j] > 0)) > 200.:
                qmm_slope, qmm_intercept = qm_train(model=fmodel[:, j], obs=fobs[:, j])
            elif np.size(np.where(fobs[:, j] > 0)) > 100.:
                qmm_intercept = np.nanmean(
                    np.nanpercentile(fobs[:, j], pctlarr) - np.nanpercentile(fmodel[:, j], pctlarr))

            gefs_hindcast_bc[:, i, :, j] = qmcal(model=hc_center_all[:, i, :, j],
                                                   slope=qmm_slope, intercept=qmm_intercept)
            fmodel_bc[:, j] = qmcal(model=fmodel[:, j], slope=qmm_slope, intercept=qmm_intercept)
            del qmm_slope, qmm_intercept
            print(" QMM bias correction station " + stations[i] + "  member" + repr(j))

        merr = np.zeros((ensm_len, 12), 'f')
        merr_cal = np.zeros((ensm_len, 12), 'f')
        for j in range(ensm_len):
            if np.any(fmodel[:, j] > -999.) == True and np.any(fobs[:, j] > -999.) == True:
                merr[j, :] = mvalstats.metrics(fmodel[:, j], fobs[:, j], pctlerr='yes')[:]
                merr_cal[j, :] = mvalstats.metrics(fmodel_bc[:, j], fobs[:, j], pctlerr='yes')[:]

        fig1 = plt.figure(1, figsize=(5, 4.5)); ax = fig1.add_subplot(111)
        ax.plot(merr[:, 0], 'k'); ax.plot(merr_cal[:, 0], 'r')
        plt.grid(c='grey', ls=':', alpha=0.5, zorder=1)
        ax.set_xlabel("Ens Members"); ax.set_ylabel("Bias")
        plt.tight_layout()
        plt.savefig(opath + "/Eval_" + wvar + "_" + stations[i] + "_Bias.png",
                    dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
                    format='png', transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig1); del fig1, ax

        fig1 = plt.figure(1, figsize=(5, 4.5)); ax = fig1.add_subplot(111)
        ax.plot(merr[:, 1], 'k'); ax.plot(merr_cal[:, 1], 'r')
        plt.grid(c='grey', ls=':', alpha=0.5, zorder=1)
        ax.set_xlabel("Ens Members"); ax.set_ylabel("Bias")
        plt.tight_layout()
        plt.savefig(opath + "/Eval_" + wvar + "_" + stations[i] + "_RMSE.png",
                    dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
                    format='png', transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig1); del fig1, ax

        fmodel_f = fmodel.flatten(); fmodel_bc_f = fmodel_bc.flatten(); fobs_f = fobs.flatten()
        mop = ModelObsPlot(model=[fmodel_f, fmodel_bc_f], obs=fobs_f, mlabels=["GEFSv12", "GEFSv12_BC"],
                            ftag=opath + "/Eval_" + stations[i] + "_", vaxisname=wvar)
        mop.qqplot(); mop.scatterplot(); mop.taylordiagram()

        del fmodel, fobs, fmodel_bc, fmodel_f, fmodel_bc_f, fobs_f, merr, merr_cal

    return gefs_hindcast_bc

# ---- WINDOWED variant: reads a small spatial window of forecast around each
#      station (for probforecast's fuzzy/neighborhood approach), instead of
#      just the center point. Used by ensemble_lag_prob.py.
#      Named distinctly from the center-point versions above so
#      ensemble_lag_det.py's functions and call sites are untouched. ----
def _init_worker_window(bid, wvar, indt_read, indlat, indlon, latm, lonm, ensm_len,
                         stations_with_obs, obs_by_station, obs_btime, include_buoy, hw):
    """Runs once per worker process, stashes shared read-only context in globals."""
    global _GW
    _GW = dict(bid=bid, wvar=wvar, indt_read=indt_read, indlat=indlat, indlon=indlon,
               latm=latm, lonm=lonm, ensm_len=ensm_len,
               stations_with_obs=stations_with_obs, obs_by_station=obs_by_station,
               obs_btime=obs_btime, include_buoy=include_buoy, hw=hw)

def _read_and_match_one_file_window(fname):
    """
    Runs in a worker process for ONE file. Same matching logic as
    _read_and_match_one_file, except the forecast is read as a small spatial
    window (hw grid points each side of center) instead of a single point —
    needed by probforecast(), which needs a spatial neighborhood.
    """
    global _GW
    bid = _GW['bid']; wvar = _GW['wvar']; indt_read = _GW['indt_read']
    indlat = _GW['indlat']; indlon = _GW['indlon']
    latm = _GW['latm']; lonm = _GW['lonm']; ensm_len = _GW['ensm_len']
    stations_with_obs = _GW['stations_with_obs']
    obs_by_station = _GW['obs_by_station']; obs_btime = _GW['obs_btime']
    include_buoy = _GW['include_buoy']; hw = _GW['hw']

    f = nc.Dataset(fname)
    try:
        ftime_i = np.array(f.variables['time'][indt_read]).astype('double')
        ctime_i = np.double(f.variables['time'][0])
        # Full grid per file — small (one file), needed for altimeter neighbor search
        hc_full = np.array(f.variables[wvar + "_gefs_hindcast"][bid, indt_read, :, :, :], dtype='f')
        # Spatial window around center — only what probforecast will ever use
        fc_window = np.array(f.variables[wvar + "_gefs_forecast"][
            bid, indt_read, :, indlat - hw:indlat + hw + 1, indlon - hw:indlon + hw + 1], dtype='f')
    finally:
        f.close()

    hc_center = np.array(hc_full[:, :, :, indlat, indlon])  # (nstations, nlead, nens)
    nlead = hc_full.shape[1]

    match_results = {}

    for i in stations_with_obs:
        sobs = obs_by_station[i]
        fmodel_rows = []
        fobs_rows = []
        neighbor_cache = {}

        for k in range(nlead):
            indt = np.where(np.abs(ftime_i[k] - obs_btime) < 1800.)
            if np.size(indt) > 0:
                indt = indt[0][0]

                if include_buoy == 'yes' and sobs['buoy'][indt] > 0.:
                    fmodel_rows.append(hc_full[i, k, :, indlat, indlon])
                    fobs_rows.append(np.full(ensm_len, sobs['buoy'][indt]))

                if np.any(np.isnan(sobs['alt'][indt, :]) == False):
                    for p in range(sobs['mlat'].shape[0]):
                        if sobs['alt'][indt, :][p] > 0.:

                            if p not in neighbor_cache:
                                indplat = np.where(
                                    np.abs(latm[i, :] - sobs['mlat'][p]) ==
                                    np.min(np.abs(latm[i, :] - sobs['mlat'][p]))
                                )[0][0]
                                indplon = np.where(
                                    np.abs(lonm[i, :] - sobs['mlon'][p]) ==
                                    np.min(np.abs(lonm[i, :] - sobs['mlon'][p]))
                                )[0][0]

                                stresult = np.round(np.abs(mvalstats.metrics(
                                    hc_full[i, :, 0, indlat, indlon],
                                    hc_full[i, :, 0, indplat, indplon]
                                )[[2, 5, 7]]), 3)
                                L1, L2, t3_p, t4_p = lmom_ratios(hc_full[i, :, 0, indlat, indlon], nmom=4)
                                l1_p = L2 / L1
                                L1, L2, t3_n, t4_n = lmom_ratios(hc_full[i, :, 0, indplat, indplon], nmom=4)
                                l1_n = L2 / L1

                                passed = (stresult[0] < 0.1 and stresult[1] < 0.1 and stresult[2] > 0.8
                                          and np.abs(l1_p - l1_n) < 0.05
                                          and np.abs(t3_p - t3_n) < 0.05
                                          and np.abs(t4_p - t4_n) < 0.05)
                                neighbor_cache[p] = (indplat, indplon, passed)

                            indplat, indplon, passed = neighbor_cache[p]
                            if passed:
                                fmodel_rows.append(hc_full[i, k, :, indplat, indplon])
                                fobs_rows.append(np.full(ensm_len, sobs['alt'][indt, :][p]))

        match_results[i] = (fmodel_rows, fobs_rows)

    del hc_full
    return ftime_i, ctime_i, hc_center, fc_window, match_results

def read_data_and_bias_correction_window(wlist, bid, ltime1, ltime2, wvar, obs, spctl, opath,
                                           spws, include_buoy="no", max_workers=None):
    '''
    Windowed sibling of read_data_and_bias_correction(). Streams through files
    one at a time (in parallel workers), same as the center-point version, but
    gdata['gefs_forecast'] carries a small spatial window ((2*gspws+1) x
    (2*gspws+1)) around each station's center point, needed by probforecast()
    for probabilistic/fuzzy forecasts. Used by ensemble_lag_prob.py.

    gspws is computed here (once, from first-file metadata) before the main
    read loop, so only the window actually needed is ever read or stored —
    never the full station grid.

    Returns: gdata (metadata + windowed gefs_forecast, plus gdata['gspws']),
             gefs_hindcast_bc (center-point, bias-corrected where obs exist).
    '''

    lstw = int(len(wlist))

    f = nc.Dataset(wlist[0])
    stations = np.array([f.variables['ID'][i] for i in bid]).astype('str')
    latm = f.variables['lat'][bid, :]; lonm = f.variables['lon'][bid, :]
    ensm = f.variables['ensemble_member'][:]
    indlat = int(np.floor(len(latm[0, :]) / 2))
    indlon = int(np.floor(len(lonm[0, :]) / 2))
    auxt = np.array(f.variables['time'][:])
    axt = np.array(auxt - auxt.min()) / 3600.
    indt = np.where((axt >= ltime1 * 24) & (axt <= ltime2 * 24))[0]
    f.close(); del f, auxt, axt

    if indt.size == 0:
        raise ValueError("indt is empty — check ltime1/ltime2 against the file's time range.")

    if np.all(np.diff(indt) == 1):
        indt_read = slice(int(indt[0]), int(indt[-1]) + 1)
    else:
        indt_read = indt

    nstations = len(bid); nlead = len(indt); ensm_len = len(ensm)

    gspws = int(np.floor(spws / np.diff(latm).mean()) / 2)
    max_hw = min(indlat, latm.shape[1] - 1 - indlat, indlon, lonm.shape[1] - 1 - indlon)
    if gspws > max_hw:
        print(f" Requested spatial window (gspws={gspws}) exceeds available grid "
              f"half-width ({max_hw}); clipping to {max_hw}.")
        gspws = max_hw
    hw = gspws
    winlat = 2 * hw + 1

    if max_workers is None:
        max_workers = len(os.sched_getaffinity(0))

    fc_shape = (lstw, nstations, nlead, ensm_len, winlat, winlat)
    fc_gb = (np.prod(fc_shape) * 4) / 1e9
    hc_center_gb = (lstw * nstations * nlead * ensm_len * 4) / 1e9
    print(f" gspws={gspws} (window {winlat}x{winlat}) — "
          f"forecast array will use ~{fc_gb:.2f} GB, hindcast-center array ~{hc_center_gb:.2f} GB")

    stations_with_obs = []
    obs_by_station = {}
    for i in range(nstations):
        if stations[i] in obs['bid']:
            inds = np.where(stations[i] == obs['bid'])[0][0]
            stations_with_obs.append(i)
            obs_by_station[i] = {
                'buoy': obs['buoy'][inds, :],
                'alt': obs['alt'][inds, :, :],
                'mlat': obs['mlat'][inds, :],
                'mlon': obs['mlon'][inds, :],
            }
    obs_btime = obs['btime']

    ctime = np.full(lstw, np.nan, dtype='double')
    ftime = np.full((lstw, nlead), np.nan, dtype='double')
    gefs_forecast = np.full(fc_shape, np.nan, dtype='f')
    hc_center_all = np.full((lstw, nstations, nlead, ensm_len), np.nan, dtype='f')

    all_fmodel = {i: [] for i in stations_with_obs}
    all_fobs = {i: [] for i in stations_with_obs}

    done_count = 0
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker_window,
        initargs=(bid, wvar, indt_read, indlat, indlon, latm, lonm, ensm_len,
                  stations_with_obs, obs_by_station, obs_btime, include_buoy, hw)
    ) as ex:
        futures = {ex.submit(_read_and_match_one_file_window, wlist[i]): i for i in range(lstw)}
        for fut in as_completed(futures):
            i = futures[fut]
            ftime_i, ctime_i, hc_center, fc_window, match_results = fut.result()

            ftime[i, :] = ftime_i
            ctime[i] = ctime_i
            gefs_forecast[i, :, :, :, :, :] = fc_window
            hc_center_all[i, :, :, :] = hc_center

            for si, (fmodel_rows, fobs_rows) in match_results.items():
                if fmodel_rows:
                    all_fmodel[si].extend(fmodel_rows)
                    all_fobs[si].extend(fobs_rows)

            del ftime_i, ctime_i, hc_center, fc_window, match_results
            done_count += 1
            print(" read+matched file " + repr(done_count) + " of " + repr(lstw) + " (index " + repr(i) + ")")

    cdate = [datetime.utcfromtimestamp(ts) for ts in ctime]
    gdata = {'cdate': cdate, 'ctime': ctime, 'ftime': ftime, 'stations': stations, 'lft': nlead, 'ensm': ensm,
        'latm': latm, 'lonm': lonm, 'indlat': indlat, 'indlon': indlon,
        'indt': indt, 'gspws': gspws, 'gefs_forecast': gefs_forecast}

    gefs_hindcast_bc = _calibrate_stations(stations, ensm_len, stations_with_obs,
                                             all_fmodel, all_fobs, hc_center_all, spctl, wvar, opath)

    return gdata, gefs_hindcast_bc

# Quantile Mapping bias-correction
def qm_train(model=None,obs=None,prob=None):
    """ 
    Univariate linear regression calibration using the Quantile Mapping Method
    Fit module
    Input:
    - model data
    - observations ("truth")
    - probability array (optional) to define the percentiles 
    Output:
    - slope
    - intercept
    """

    if (np.size(model)>2)==False or (np.size(obs)>2)==False:
        raise ValueError("Model and Obs arrays must be informed.")

    if np.size(model) != np.size(obs):
        raise ValueError("Model and Obs arrays must have the same sizes.")

    # Probability array
    if prob==None:
        prob=np.append(np.array(np.arange(1.,50.+0.1,1.),'f'),np.array(np.arange(50.5,99.5+0.1,0.5),'f'))
        prob=np.append(prob,np.array([99.7,99.8,99.9,99.95,99.99]))

    slope,intercept = np.polyfit(np.nanpercentile(model,prob),np.nanpercentile(obs,prob),1)
    # model_cal=np.array((model*slope)+intercept)
    # print(" QMM linear regression. Slope: "+repr(slope)+"  Intercept: "+repr(intercept))

    return float(slope),float(intercept)

def qmcal(model=None,slope=1.,intercept=0.,pprint='no'):
    """ 
    Univariate linear regression calibration using the Quantile Mapping Method
    Calibration module based on previously trained qm_train
    Input:
    - model data
    - slope
    - intercept
    Output:
    - calibrated model data
    """

    if (np.size(model)>2)==False:
        raise ValueError("Model array must be informed.")

    model_cal=np.array((model*slope)+intercept)
    if pprint=='yes':
        print(" QMM linear regression. Slope: "+repr(slope)+"  Intercept: "+repr(intercept))

    return np.array(model_cal).astype('float')

# ---- STATISTICAL MODELLING -----

# Function to check np.sort using multidimensional array
def tsort(model):

    fmodel=np.copy(model)
    if len(fmodel.shape) == 4:
        for i in range(0,model.shape[1]):
            for j in range(0,model.shape[2]):
                for k in range(0,model.shape[3]):
                    fmodel[:,i,j,k] = np.sort(model[:,i,j,k])

    elif len(fmodel.shape) == 3:
        for i in range(0,model.shape[1]):
            for j in range(0,model.shape[2]):
                fmodel[:,i,j] = np.sort(model[:,i,j])

    elif len(fmodel.shape) == 2:
        for i in range(0,model.shape[1]):
            fmodel[:,i] = np.sort(model[:,i])

    elif len(fmodel.shape) == 2:
        fmodel = np.sort(model[:])

    return fmodel

def nmaxsel(wdata,nmax):

    lstw = int(wdata.shape[0])
    wdata_tmax = np.mean(wdata,axis=1)
    for i in range(0,lstw):
        ind = np.where(wdata[i,:]>-999.)
        if np.size(ind)>int(np.floor(wdata.shape[1]/2)):
            ind = ind[0]
            wdata_tmax[i] = np.nanmean(np.sort(wdata[i,ind])[-nmax::])
        else:
            print(" nmaxsel(): forecast time series incomplete "+repr(i))
            wdata_tmax[i] = np.nan

        del ind

    return wdata_tmax

def probforecast(nmax,gspws,spctl,mod_forecast,qlev):

    lstw=int(mod_forecast.shape[0])
    nensm=int(mod_forecast.shape[2])
    indlat = int(np.floor(mod_forecast.shape[3]/2)); indlon = int(np.floor(mod_forecast.shape[4]/2))

    prob_forecast = np.zeros((lstw,len(qlev)),'f')*np.nan
    for i in range(0,lstw):
        fmod=np.array(mod_forecast[i,:,:,(indlat-gspws):(indlat+gspws+1),:][:,:,:,(indlon-gspws):(indlon+gspws+1)])
        fmod=tsort(fmod)
        fmod=np.copy(fmod[-nmax::,:,:,:])
        fmod=np.array(fmod.reshape(nmax*nensm,fmod.shape[2],fmod.shape[3]))
        fmod=np.array(fmod.reshape(nmax*nensm,fmod.shape[1]*fmod.shape[2]))
        fmod=np.sort(fmod,axis=1)
        ind=np.where(np.mean(fmod,axis=0)>=0.)
        if np.size(ind)>0:
            fmod=np.array(fmod[:,ind[0]])            
            fmod=np.array(fmod[:,int(np.floor(fmod.shape[1]*(spctl/100)))::])
            fmod=fmod.reshape(fmod.shape[0]*fmod.shape[1])
            for j in range(0,len(qlev)):
                prob_forecast[i,j] = np.size(fmod[fmod>qlev[j]]) / np.size(fmod[fmod>0.])

        if i==0:
            fmod_result = np.array([fmod])
        else:
            fmod_result = np.append(fmod_result,np.array([fmod]),axis=0)

    return prob_forecast, fmod_result

def prob_validation(nmax,spws,gspws,spctl,prob_u10_gefs_forecast,fmod_result_u10,prob_hs_gefs_forecast,fmod_result_hs,u10_obs_tmax,hs_obs_tmax,qlev_wnd,qlev_hs,plevels,ftag):

    # Binary Categorical array
    # U10
    fprob_u10_gefs_forecast = probforecast_binary(prob_u10_gefs_forecast,qlev_wnd,plevels)
    # Hs
    fprob_hs_gefs_forecast = probforecast_binary(prob_hs_gefs_forecast,qlev_hs,plevels)

    print(" Prob Forecast and Binary array, OK")

    lplev=int(len(plevels)-1)

    # ----------- Error metrics: TP, FP, TN, FN, CSI;  Brier; ROC_AUC ---------------
    ceval_gefs_u10 = np.zeros((len(qlev_wnd),lplev,3),'f')*np.nan
    for i in range(0,len(qlev_wnd)):
        for j in range(0,lplev):
            bresult = categorical_bin_eval(fprob_u10_gefs_forecast[:,i,j],u10_obs_tmax,qlev_wnd[i])
            ceval_gefs_u10[i,j,0] = float(bresult['POD'])
            ceval_gefs_u10[i,j,1] = float(bresult['FAR'])
            ceval_gefs_u10[i,j,2] = float(bresult['CSI'])
            del bresult

    ceval_gefs_hs = np.zeros((len(qlev_hs),lplev,3),'f')*np.nan
    for i in range(0,len(qlev_hs)):
        for j in range(0,lplev):
            bresult = categorical_bin_eval(fprob_hs_gefs_forecast[:,i,j],hs_obs_tmax,qlev_hs[i])
            ceval_gefs_hs[i,j,0] = float(bresult['POD'])
            ceval_gefs_hs[i,j,1] = float(bresult['FAR'])
            ceval_gefs_hs[i,j,2] = float(bresult['CSI'])
            del bresult

    # --- ROC Curve ---
    # U10
    froc_u10=[]
    for i in range(0,len(qlev_wnd)):
        fftag=ftag+"ProbEvents_U10_"+repr(np.round(qlev_wnd[i],2))
        true_binary = (u10_obs_tmax > qlev_wnd[i]).astype(int)
        roc = roc_plot(true_binary,prob_u10_gefs_forecast[:,i],fftag)
        froc_u10 = np.append(froc_u10,roc); del true_binary, fftag, roc

    # Hs
    froc_hs=[]
    for i in range(0,len(qlev_hs)):
        fftag=ftag+"ProbEvents_Hs_"+repr(np.round(qlev_hs[i],2))
        true_binary = (hs_obs_tmax > qlev_hs[i]).astype(int)
        roc = roc_plot(true_binary,prob_hs_gefs_forecast[:,i],fftag)
        froc_hs = np.append(froc_hs,roc); del true_binary, fftag, roc

    print(" Validation: Categorical OK")
    # --- Brier Score ---
    # U10
    fbriers_u10=[]
    for i in range(0,len(qlev_wnd)):    
        fftag=ftag+"ProbEvents_U10_"+repr(np.round(qlev_wnd[i],2))
        briers = brier_score(prob_u10_gefs_forecast[:,i],u10_obs_tmax,qlev_wnd[i],fftag)
        fbriers_u10 = np.append(fbriers_u10,briers)
        del briers, fftag

    # Hs
    fbriers_hs=[]
    for i in range(0,len(qlev_hs)):    
        fftag=ftag+"ProbEvents_Hs_"+repr(np.round(qlev_hs[i],2))
        briers = brier_score(prob_hs_gefs_forecast[:,i],hs_obs_tmax,qlev_hs[i],fftag)
        fbriers_hs = np.append(fbriers_hs,briers)
        del briers, fftag

    print(" Validation: Brier Score OK")

    # --- CRPS ---
    # U10 
    fftag=ftag+"ProbEvents_U10"
    crps_u10, mean_crps_u10 = crps(fmod_result_u10, u10_obs_tmax, fftag)
    del fftag
    # Hs
    fftag=ftag+"ProbEvents_hs"
    crps_hs, mean_crps_hs = crps(fmod_result_hs, hs_obs_tmax, fftag)
    del fftag

    plt.close('all')
    print(" Validation: CRPS OK")

    # ================================================

    # Save results
    stat_results = {
    'ceval_gefs_hs': ceval_gefs_hs,
    'ceval_gefs_u10': ceval_gefs_u10,
    'froc_hs': froc_hs,
    'froc_u10': froc_u10,
    'fbriers_u10': fbriers_u10,
    'fbriers_hs': fbriers_hs,
    'crps_u10': crps_u10,
    'mean_crps_u10': mean_crps_u10,
    'crps_hs': crps_hs,
    'mean_crps_hs': mean_crps_hs
    }

    # Save the dictionary to a pickle file
    with open(ftag+'STAT.RESULTS.pkl', 'wb') as f:
        pickle.dump(stat_results, f)

    return stat_results
    print(" Results saved in the pickle file STAT.RESULTS.pkl")

    del u10_obs_tmax, hs_obs_tmax, prob_u10_gefs_forecast, prob_hs_gefs_forecast, fprob_u10_gefs_forecast, fprob_hs_gefs_forecast
    del stat_results, ceval_gefs_u10, ceval_gefs_hs, froc_u10, froc_hs, fbriers_u10, fbriers_hs, crps_u10, mean_crps_u10, crps_hs, mean_crps_hs

    print(" - Done Optmz_nmax"+str(int(nmax))+"_spws"+str(int(spws*100)).zfill(3)+"_spctl"+str(int(spctl)))

# Binary Categorical array
def probforecast_binary(prob_forcast,qlev,plevels):

    lstw = int(prob_forcast.shape[0])
    lplev = int(len(plevels)-1)
    fprob_forecast = np.zeros((lstw,len(qlev),lplev),'f')*np.nan

    for i in range(0,lstw):
        for j in range(0,len(qlev)):
            for k in range(0,lplev):
                if prob_forcast[i,j] >= plevels[k]:
                    fprob_forecast[i,j,k] = int(1)
                else:
                    fprob_forecast[i,j,k] = int(0)

    return fprob_forecast

def categorical_bin_eval(forecasted,obs,threshold):

    ind = np.where((forecasted>-999.)&(obs>-999.))
    if np.size(ind)>1:
        forecasted=forecasted[ind[0]]
        obs=obs[ind[0]]

        SIZE = int(len(obs))
        EVENTS = int(np.size(np.where((obs >= threshold)==True)))

        # Calculate True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN)
        if np.any((obs >= threshold)==True):
            TP = np.sum((forecasted == int(1)) & (obs >= threshold))
            FP = np.sum((forecasted == int(1)) & (obs < threshold))
            FN = np.sum((forecasted == int(0)) & (obs >= threshold))
            TN = np.sum((forecasted == int(0)) & (obs < threshold))

            # POD (Probability of Detection) also name as True Positive Rate (TPR) or Sensitivity
            if (TP + FN) == 0:
                POD = np.nan
            else:
                POD = np.round((TP / (TP + FN)),3)

            # FAR (False Alarm Ratio)
            if (FP + TP)==0:
                FAR = 0
            else:
                FAR = np.round((FP / (FP + TP)),3)

            # Critical Success Index (CSI)
            if (TP + FN + FP)==0:
                CSI = np.nan
            else:
                CSI = np.round((TP / (TP + FN + FP)),3)

            # ETS (Equitable Threat Score)
            ETS = np.round(((TP - (TP + FN) * (TP + FP) / (TP + FN + FP + TN)) / (TP + FN + FP - (TP + FN) * (TP + FP) / (TP + FN + FP + TN))) ,3)
        else:
            TP=np.nan; FP=np.nan; FN=np.nan; TN=np.nan
            POD=np.nan; FAR=np.nan; CSI=np.nan; ETS=np.nan

        result = {'SIZE':SIZE, 'EVENTS':EVENTS,
            'TP':TP, 'FP':FP, 'FN':FN, 'TN':TN,
            'POD':POD, 'FAR':FAR, 'CSI':CSI, 'ETS': ETS}

        return result

def brier_score(prob_forecast,obs,threshold):

    # Brier Score
    true_binary = (obs > threshold).astype(int)
    briers=0
    for i in range(0,len(true_binary)):
        briers = briers+(true_binary[i]-prob_forecast[i])**2

    briers = briers/len(true_binary)

    return float(briers)

def crps(fmod,obs):

    # CRPS
    batch_size = 1000
    if fmod.shape[0]>batch_size:
        crps=[]
        # Process in batches
        for i in range(0, fmod.shape[0], batch_size):
            fmod_batch = fmod[i:i+batch_size]
            obs_batch = obs[i:i+batch_size]
            # Compute CRPS for this batch
            crps_batch = ps.crps_ensemble(obs_batch, fmod_batch)
            crps.append(crps_batch)

        crps = np.array(np.concatenate(crps))
    else:
        crps = ps.crps_ensemble(obs,fmod)

    fcrps_mean = np.nanmean(crps)

    return np.array(crps).astype('float'), float(fcrps_mean)
    del crps, fcrps_mean

def roc_plot(true_binary,predicted_probabilities,ftag):

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(true_binary, predicted_probabilities)
    # Calculate AUC (Area Under the ROC Curve)
    roc_auc = auc(fpr, tpr)

    if len(ftag)>0:
        # Plot the ROC curve
        fig1, ax = plt.subplots(figsize=(5,4.5))
        ax.plot([0, 1], [0, 1], color='dimgray', lw=2, linestyle='--', alpha=0.8, zorder=1)
        ax.plot(fpr, tpr, color='firebrick', lw=1, marker='.', alpha=0.8, zorder=2)
        ax.plot(fpr, tpr, color='firebrick', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})', zorder=3)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01) 
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right',fontsize=sl-3)
        plt.title('ROC Curve')
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
        plt.savefig(ftag+"_ROC.png", dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
            format='png', bbox_inches='tight', pad_inches=0.1)

        plt.close(fig1)

    return float(roc_auc)

def roc_plot2(true_binary,predicted_probabilities,mlabels,plevels,ftag):

    from sklearn.metrics import confusion_matrix, accuracy_score

    predicted_probabilities = np.atleast_2d(predicted_probabilities)
    froc_auc = []

    fig1, ax = plt.subplots(figsize=(5,4.5))
    ax.plot([0, 1], [0, 1], color='dimgray', lw=2, linestyle='--', alpha=0.8, zorder=1)

    for i in range(0,predicted_probabilities.shape[0]):

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(true_binary, predicted_probabilities[i,:])

        ## Calculate confusion matrix
        for j in range(0,len(plevels)):
            y_pred = (predicted_probabilities[i,:] >= plevels[j]).astype(int)
            tn, fp, fn, tp = confusion_matrix(true_binary,y_pred).ravel()
            pod = tp / (tp + fn)
            far = fp / (fp + tp)
            csi = tp / (tp + fp + fn)
            print("CSI prob lev "+str(plevels[j])+" = "+repr(np.round(csi,3)))

        print("--------"+repr(i))

        # Calculate AUC (Area Under the ROC Curve)
        roc_auc = auc(fpr, tpr)
        froc_auc = np.append(froc_auc,roc_auc)

        # Plot the ROC curve
        if i==0:
            ax.plot(fpr, tpr, color=pcolors[i], lw=1, marker='.', alpha=0.8, zorder=2)      
            ax.plot(fpr, tpr, color=pcolors[i], lw=2, linestyle=lstl[i], label=f'{mlabels[i]} (AUC = {roc_auc:.2f})', zorder=3)
        else:
            ax.plot(fpr, tpr, color=pcolors[i], lw=1, linestyle=lstl[i], label=f'{mlabels[i]} (AUC = {roc_auc:.2f})',alpha=0.7 , zorder=3)

    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01) 
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right',fontsize=sl-3)
    # plt.title('ROC Curve')
    plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
    plt.savefig(ftag+"_ROC.png", dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
        format='png', bbox_inches='tight', pad_inches=0.1)

    plt.close(fig1)

    return np.array(froc_auc).astype('float')

def reliability_curve(true_binary,predicted_probabilities,cfreq,mlabels,nbins,pmax,ftag):

    predicted_probabilities = np.atleast_2d(predicted_probabilities)

    xd = yd = [0,1]
    clim_x = clim_y = [cfreq,cfreq]
    sk_line = [cfreq/2,(1-cfreq)/2+cfreq]

    fig1, ax = plt.subplots(figsize=(5.5,5.))
    ax.axhline(cfreq,color='grey',linestyle='-.',linewidth=1, zorder=2)
    ax.plot(clim_x,yd,color='grey',linestyle=':',linewidth=0.5)
    ax.text(0., cfreq, 'Clim', color='dimgrey', fontsize=6)
    ax.plot(xd,sk_line,color='grey',linestyle='--',linewidth=0.5)
    ax.fill_between(xd,xd,sk_line,facecolor='peachpuff',alpha=0.2)
    ax.fill_betweenx(yd,yd,clim_x,facecolor='peachpuff',alpha=0.2)

    for i in range(0,predicted_probabilities.shape[0]):
        ry, rx = calibration_curve(true_binary, predicted_probabilities[i,:], n_bins=nbins)
        # Max prob 
        indp = np.where(rx<pmax)[0]

        if i==0:
            ax.plot(rx[indp], ry[indp],color=pcolors[i],linestyle=lstl[i], label=mlabels[i], linewidth=2., zorder=4)
            ax.plot(rx[indp], ry[indp], marker='s', linewidth=1, color=pcolors[i], zorder=4)
        else:
            ax.plot(rx[indp], ry[indp],color=pcolors[i],linestyle=lstl[i], label=mlabels[i], linewidth=1.,alpha=0.7, zorder=3)
            # ax.plot(rx[indp], ry[indp], marker='.', linewidth=1, color=pcolors[i], zorder=3)

    line = mlines.Line2D([0, 1], [0, 1], color='dimgrey',linestyle='--', zorder=2)
    transform = ax.transAxes; line.set_transform(transform); ax.add_line(line)
    ax.set_xlabel('Forecast probability')
    ax.set_ylabel('Observed relative frequency')
    ax.legend(loc='lower right',fontsize=sl-3)
    plt.grid(c='silver', ls='--', alpha=0.2, zorder=1)
    plt.ylim(ymax = 1.005, ymin = -0.005)
    plt.xlim(xmax = 1.005, xmin = -0.005)
    # ax_inset = inset_axes(ax, width="30%", height="30%", loc="upper left")
    ax_inset = fig1.add_axes([0.16, 0.75, 0.25, 0.19])
    # Plot the histogram in the inset
    ax_inset.hist(predicted_probabilities[0,:], bins=np.arange(0, 1.1, 0.1), color='gray' , edgecolor='black', alpha=0.5, zorder=2)
    ax_inset.set_xticks(np.arange(0., 1.1, 0.1))
    ax_inset.set_xticklabels([f'{x:.1f}' for x in np.arange(0., 1.1, 0.1)])
    ax_inset.tick_params(axis='x', which='major', labelsize=5)
    ax_inset.set_yticks([]); ax_inset.set_yticklabels([])
    ax_inset.set_xlim(0,1)
    plt.tight_layout()
    # plt.show()
    # ax.set_title("Reliability diagram" )
    plt.savefig(ftag+"_ReliabilityDiagram.png", dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
        format='png', bbox_inches='tight', pad_inches=0.1)

    plt.close(fig1)

