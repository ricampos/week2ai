#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Probabilistic model. Hyperparameter optimization.

import matplotlib
# matplotlib.use('Agg')
import pickle
from matplotlib.dates import DateFormatter
from scipy.ndimage.filters import gaussian_filter
import netCDF4 as nc
import numpy as np
import pandas as pd
import sys
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import properscoring as ps
from datetime import datetime
# from pysteps import verification
import yaml
from pvalstats import ModelObsPlot
import dproc
import wprob
import warnings; warnings.filterwarnings("ignore")

sl=13
matplotlib.rcParams.update({'font.size': sl}); plt.rc('font', size=sl) 
matplotlib.rc('xtick', labelsize=sl); matplotlib.rc('ytick', labelsize=sl); matplotlib.rcParams.update({'font.size': sl})


if __name__ == "__main__":

    # select one point
    stations = np.array(['46005','46006','46066']).astype('str')
    # Forecast Lead Time (Day) and intervall
    ltime1=7; ltime2=14
    # opath="/work/noaa/marine/ricardo.campos/work/analysis/4postproc/output/qmm"
    opath="/home/ricardo/cimas/analysis/4postproc/output/qmm"

    # ---- Statistical parameters (exaclty the same as the operational config file) -----
    qlev_hs = np.array([ 4.,  6.,  9.]).astype('float')
    qlev_wnd = np.array([28.0, 34.0, 41.0, 48.0]).astype('float')/1.94
    plevels = np.array([0.15, 0.5, 0.65, 0.8, 1.0])

    # Input Argument. Optimized Parameters
    spws = float(2.5)
    nmax = int(2)

    print(" ")
    print(" === Bias Correction, QMM === ")
    print(" ")
    # ------------------

    # READ DATA - Week 2
    # list of netcdf files generated with buildfuzzydataset.py (GEFS, GDAS, and NDBC buoy)
    # ls -d $PWD/*.nc > list.txt &
    wlist=np.atleast_1d(np.loadtxt('list.txt',dtype=str)) 
    lstw=int(len(wlist)); lplev=int(len(plevels)-1)

    # week 2 array
    for i in range(0,len(stations)):

        # cdate,ctime,ensm,latm,lonm,indlat,indlon,u10_ndbc,hs_ndbc,u10_gefs_hindcast,hs_gefs_hindcast,au10_gefs_forecast,ahs_gefs_forecast,indt = wprob.read_data(wlist,stations[i],ltime1,ltime2)
        ENSDATA = wprob.read_data(wlist,stations[i],ltime1,ltime2)

        if i==0:
            gspws = int(np.floor(spws/np.diff(ENSDATA['latm']).mean())/2)
            cdate = np.array(ENSDATA['cdate'])
            bid=np.zeros((len(ENSDATA['ctime'])),'f')+float(stations[i])
            # ens forecast
            u10_gefs_forecast = np.array(ENSDATA['u10_gefs_forecast'])
            hs_gefs_forecast = np.array(ENSDATA['hs_gefs_forecast'])
            # Ground truth
            u10_obs = np.nanmean([ENSDATA['u10_ndbc'],np.nanmean(ENSDATA['u10_gefs_hindcast'][:,:,:,ENSDATA['indlat'],ENSDATA['indlon']],axis=2)],axis=0)
            hs_obs = np.nanmean([ENSDATA['hs_ndbc'],np.nanmean(ENSDATA['hs_gefs_hindcast'][:,:,:,ENSDATA['indlat'],ENSDATA['indlon']],axis=2)],axis=0)

        else:
            cdate=np.append(cdate,ENSDATA['cdate'])
            bid=np.append(bid,np.zeros((len(ENSDATA['ctime'])),'f')+float(stations[i]))
            # ens forecast
            u10_gefs_forecast = np.append(u10_gefs_forecast,ENSDATA['u10_gefs_forecast'],axis=0)
            hs_gefs_forecast = np.append(hs_gefs_forecast,ENSDATA['hs_gefs_forecast'],axis=0)
            # Ground truth
            au10_obs = np.nanmean([ENSDATA['u10_ndbc'],np.nanmean(ENSDATA['u10_gefs_hindcast'][:,:,:,ENSDATA['indlat'],ENSDATA['indlon']],axis=2)],axis=0)
            ahs_obs = np.nanmean([ENSDATA['hs_ndbc'],np.nanmean(ENSDATA['hs_gefs_hindcast'][:,:,:,ENSDATA['indlat'],ENSDATA['indlon']],axis=2)],axis=0)
            u10_obs = np.append(u10_obs,au10_obs,axis=0)
            hs_obs = np.append(hs_obs,ahs_obs,axis=0)
            del au10_obs, ahs_obs 

        del ENSDATA
        print(' ---- Ok ---- Read Station: '+stations[i])

    print(" READ DATA OK")
    # ------------------------------------------------------------------

    # === Bias-Correction Quantile Mapping Method ===
    # Univariate Linear Bias-Correction:

    # Training and Validation sets
    # indtrain = [i for i, date in enumerate(cdate) if date < datetime(2023, 10, 1, 0, 0)]
    # indval = [i for i, date in enumerate(cdate) if date >= datetime(2023, 10, 1, 0, 0)]

    qmm_u10 = np.zeros((len(stations),u10_gefs_forecast.shape[2],u10_gefs_forecast.shape[3],u10_gefs_forecast.shape[4],2),'f')*np.nan
    qmm_hs = np.zeros((len(stations),hs_gefs_forecast.shape[2],hs_gefs_forecast.shape[3],hs_gefs_forecast.shape[4],2),'f')*np.nan

    # array of probabilities (focused on the extremal tail)
    p = np.arange(50,71,5)[1::]; p=np.append(p,np.arange(73,81,3))
    p=np.append(p,np.arange(82,91,2)); p=np.append(p,np.arange(91,96,1))
    p=np.append(p,[95.5,96.,96.5,97.,97.3,97.6,97.9,98.2,98.5,98.7,98.9,99.1,99.3,99.5,99.6,99.7,99.8,99.9,99.92,99.94,99.96,99.98])

    u10_gefs_forecast_cal = np.copy(u10_gefs_forecast)
    hs_gefs_forecast_cal = np.copy(hs_gefs_forecast)

    for b in range(0,len(stations)):
        for i in range(0,qmm_hs.shape[1]):
            for j in range(0,qmm_hs.shape[2]):
                for k in range(0,qmm_hs.shape[3]):

                    indb = np.where(bid==float(stations[b]))[0]

                    # U10
                    model = u10_gefs_forecast[indb,:,i,j,k].reshape(len(indb)*u10_obs.shape[1])
                    obs = u10_obs[indb,:].reshape(len(indb)*u10_obs.shape[1])
                    slope,intercept = dproc.qm_train(model=model,obs=obs,prob=p,pprint='yes')
                    qmm_u10[b,i,j,k,0] = slope; qmm_u10[b,i,j,k,1] = intercept
                    model_cal = dproc.qmcal(model=u10_gefs_forecast_cal[indb,:,i,j,k],slope=slope,intercept=intercept,pprint='no')
                    u10_gefs_forecast_cal[indb,:,i,j,k] = np.array(model_cal)
                    del model, obs, slope, intercept, model_cal

                    # Hs
                    model = hs_gefs_forecast[indb,:,i,j,k].reshape(len(indb)*hs_obs.shape[1])
                    obs = hs_obs[indb,:].reshape(len(indb)*hs_obs.shape[1])
                    slope,intercept = dproc.qm_train(model=model,obs=obs,prob=p,pprint='yes')
                    qmm_hs[b,i,j,k,0] = slope; qmm_hs[b,i,j,k,1] = intercept
                    model_cal = dproc.qmcal(model=hs_gefs_forecast_cal[indb,:,i,j,k],slope=slope,intercept=intercept,pprint='no')
                    hs_gefs_forecast_cal[indb,:,i,j,k] = np.array(model_cal)
                    del model, obs, slope, intercept, model_cal

                    print(" bid"+stations[b]+" i"+repr(i)+" j"+repr(j)+" k"+repr(k))
                    del indb


    # Plot
    p = np.arange(0,92,2)[1::]; p=np.append(p,np.arange(91,99,1))
    p=np.append(p,[99.,99.1,99.3,99.5,99.7])
    aux=np.linspace(1.5,10.,p.shape[0])
    fig1 = plt.figure(1,figsize=(5,4.5)); ax = fig1.add_subplot(111)
    ax.plot(aux,aux,'k', linestyle='--', linewidth=1.,alpha=0.9,zorder=2)
    i=0
    for j in range(0,qmm_hs.shape[2]):
        for k in range(0,qmm_hs.shape[3]):
            a = hs_gefs_forecast[indb,:,i,j,k].reshape(len(indb)*hs_obs.shape[1])
            b = hs_gefs_forecast_cal[indb,:,i,j,k].reshape(len(indb)*hs_obs.shape[1])

            qma = np.zeros((p.shape[0]),'f')*np.nan
            qmb = np.zeros((p.shape[0]),'f')*np.nan
            qobs = np.zeros((p.shape[0]),'f')*np.nan
            for l in range(0,p.shape[0]):
                qobs[l] = np.nanpercentile(obs,p[l])
                qma[l] = np.nanpercentile(a,p[l])
                qmb[l] = np.nanpercentile(b,p[l])

            ax.plot(gaussian_filter(qobs,1),gaussian_filter(qma,1), color='cornflowerblue', linestyle='-',linewidth=0.5,alpha=0.8,zorder=3)
            ax.plot(gaussian_filter(qobs,1),gaussian_filter(qmb,1), color='firebrick', linestyle='-',linewidth=0.5,alpha=0.8,zorder=3)

    plt.grid(c='grey', ls=':', alpha=0.5,zorder=1)
    plt.ylim(ymax = aux.max(), ymin = aux.min())
    plt.xlim(xmax = aux.max(), xmin = aux.min())
    plt.locator_params(axis='y', nbins=7) ; plt.locator_params(axis='x', nbins=7) 
    ax.set_xlabel('Obs'); ax.set_ylabel('Model')
    plt.tight_layout()
    plt.savefig(opath+"/QQplot_example.png", dpi=200, facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1); del fig1, ax
    # -------------------------------------------



    # Min duration of event (consider 6-hourly resolution)
    u10_obs_tmax = wprob.nmaxsel(u10_obs,1)
    hs_obs_tmax = wprob.nmaxsel(hs_obs,1)

    for spctl in np.array([80,83,85,86,87,88,89,90,91,93,95,98]).astype('float'):

        ftag=opath+"/Original_nmax"+str(int(nmax))+"_spws"+str(int(spws*100)).zfill(3)+"_spctl"+str(int(spctl))+"_"
        prob_u10_gefs_forecast, fmod_result_u10 = wprob.probforecast(nmax,gspws,spctl,u10_gefs_forecast,qlev_wnd)
        prob_hs_gefs_forecast, fmod_result_hs = wprob.probforecast(nmax,gspws,spctl,hs_gefs_forecast,qlev_hs)
        wprob.prob_validation(nmax,gspws,spctl,cdate,prob_u10_gefs_forecast,fmod_result_u10,prob_hs_gefs_forecast,fmod_result_hs,u10_obs_tmax,hs_obs_tmax,qlev_wnd,qlev_hs,plevels,ftag)
        del ftag

        ftag=opath+"/BiasCorrectedQMM_nmax"+str(int(nmax))+"_spws"+str(int(spws*100)).zfill(3)+"_spctl"+str(int(spctl))+"_"
        prob_u10_gefs_forecast_cal, fmod_result_u10_cal = wprob.probforecast(nmax,gspws,spctl,u10_gefs_forecast_cal,qlev_wnd)
        prob_hs_gefs_forecast_cal, fmod_result_hs_cal = wprob.probforecast(nmax,gspws,spctl,hs_gefs_forecast_cal,qlev_hs)
        wprob.prob_validation(nmax,gspws,spctl,cdate,prob_u10_gefs_forecast,fmod_result_u10,prob_hs_gefs_forecast,fmod_result_hs,u10_obs_tmax,hs_obs_tmax,qlev_wnd,qlev_hs,plevels,ftag)
        del ftag


