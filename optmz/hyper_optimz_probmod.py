#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Probabilistic model. Hyperparameter optimization.

import matplotlib
# matplotlib.use('Agg')
import pickle
from matplotlib.dates import DateFormatter
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
# from pvalstats import ModelObsPlot
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
    opath="/work/noaa/marine/ricardo.campos/work/analysis/4postproc/output/optmz"
    # opath="/home/ricardo/cimas/analysis/4postproc/output"

    # Input Argument
    spws = float(sys.argv[1]) # [0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0]

    # ---- Read statistical parameters (exaclty the same as the operational config file) -----
    qlev_hs = np.array([ 4.,  6.,  9.]).astype('float')
    qlev_wnd = np.array([28.0, 34.0, 41.0, 48.0]).astype('float')/1.94
    plevels = np.array([0.15, 0.5, 0.65, 0.8, 1.0])

    print(" ")
    print(" === Simulation Optmz_spws"+str(int(2*spws*100)).zfill(3))
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
            # ens forecast
            u10_gefs_forecast = np.array(ENSDATA['u10_gefs_forecast'])
            hs_gefs_forecast = np.array(ENSDATA['hs_gefs_forecast'])
            # Ground truth
            u10_obs = np.nanmean([ENSDATA['u10_ndbc'],np.nanmean(ENSDATA['u10_gefs_hindcast'][:,:,:,ENSDATA['indlat'],ENSDATA['indlon']],axis=2)],axis=0)
            hs_obs = np.nanmean([ENSDATA['hs_ndbc'],np.nanmean(ENSDATA['hs_gefs_hindcast'][:,:,:,ENSDATA['indlat'],ENSDATA['indlon']],axis=2)],axis=0)

        else:
            cdate=np.append(cdate,ENSDATA['cdate'])
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

    # === Optimization loop ===

    sim=0 # total of 36
    for nmax in np.array([1,2,3,4]).astype('int'):
        for spctl in np.array([80,83,85,87,89,91,93,95,98]).astype('float'):

            print(" ")
            print(" - starting Optmz_nmax"+str(int(nmax))+"_spws"+str(int(spws*100)).zfill(3)+"_spctl"+str(int(spctl)))

            ftag=opath+"/Optmz_nmax"+str(int(nmax))+"_spws"+str(int(spws*100)).zfill(3)+"_spctl"+str(int(spctl))+"_"

            # Min duration of event (consider 6-hourly resolution)
            u10_obs_tmax = wprob.nmaxsel(u10_obs,1)
            hs_obs_tmax = wprob.nmaxsel(hs_obs,1)

            # Probabilistic Forecast array
            # U10 
            prob_u10_gefs_forecast, fmod_result_u10 = wprob.probforecast(nmax,gspws,spctl,u10_gefs_forecast,qlev_wnd)
            # Hs
            prob_hs_gefs_forecast, fmod_result_hs = wprob.probforecast(nmax,gspws,spctl,hs_gefs_forecast,qlev_hs)

            # Binary Categorical array
            # U10
            fprob_u10_gefs_forecast = wprob.probforecast_binary(prob_u10_gefs_forecast,qlev_wnd,plevels)
            # Hs
            fprob_hs_gefs_forecast = wprob.probforecast_binary(prob_hs_gefs_forecast,qlev_hs,plevels)

            print(" Prob Forecast and Binary array, OK")

            # ----------- Error metrics: TP, FP, TN, FN, CSI;  Brier; ROC_AUC ---------------
            ceval_gefs_u10 = np.zeros((len(qlev_wnd),lplev,3),'f')*np.nan
            for i in range(0,len(qlev_wnd)):
                for j in range(0,lplev):
                    bresult = wprob.categorical_bin_eval(fprob_u10_gefs_forecast[:,i,j],u10_obs_tmax,qlev_wnd[i])
                    ceval_gefs_u10[i,j,0] = float(bresult['POD'])
                    ceval_gefs_u10[i,j,1] = float(bresult['FAR'])
                    ceval_gefs_u10[i,j,2] = float(bresult['CSI'])
                    del bresult

            ceval_gefs_hs = np.zeros((len(qlev_hs),lplev,3),'f')*np.nan
            for i in range(0,len(qlev_hs)):
                for j in range(0,lplev):
                    bresult = wprob.categorical_bin_eval(fprob_hs_gefs_forecast[:,i,j],hs_obs_tmax,qlev_hs[i])
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
                roc = wprob.roc_plot(true_binary,prob_u10_gefs_forecast[:,i],fftag)
                froc_u10 = np.append(froc_u10,roc); del true_binary, fftag, roc

            # Hs
            froc_hs=[]
            for i in range(0,len(qlev_hs)):
                fftag=ftag+"ProbEvents_Hs_"+repr(np.round(qlev_hs[i],2))
                true_binary = (hs_obs_tmax > qlev_hs[i]).astype(int)
                roc = wprob.roc_plot(true_binary,prob_hs_gefs_forecast[:,i],fftag)
                froc_hs = np.append(froc_hs,roc); del true_binary, fftag, roc

            print(" Validation: Categorical OK")
            # --- Brier Score ---
            # U10
            fbriers_u10=[]
            for i in range(0,len(qlev_wnd)):    
                fftag=ftag+"ProbEvents_U10_"+repr(np.round(qlev_wnd[i],2))
                briers = wprob.brier_score(prob_u10_gefs_forecast[:,i],u10_obs_tmax,qlev_wnd[i],cdate,fftag)
                fbriers_u10 = np.append(fbriers_u10,briers)
                del briers, fftag

            # Hs
            fbriers_hs=[]
            for i in range(0,len(qlev_hs)):    
                fftag=ftag+"ProbEvents_Hs_"+repr(np.round(qlev_hs[i],2))
                briers = wprob.brier_score(prob_hs_gefs_forecast[:,i],hs_obs_tmax,qlev_hs[i],cdate,fftag)
                fbriers_hs = np.append(fbriers_hs,briers)
                del briers, fftag

            print(" Validation: Brier Score OK")

            # --- CRPS ---
            # U10 
            fftag=ftag+"ProbEvents_U10"
            crps_u10, mean_crps_u10 = wprob.crps(fmod_result_u10, u10_obs_tmax, cdate, fftag)
            del fftag
            # Hs
            fftag=ftag+"ProbEvents_hs"
            crps_hs, mean_crps_hs = wprob.crps(fmod_result_hs, hs_obs_tmax, cdate, fftag)
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

            print(" Results saved in the pickle file STAT.RESULTS.pkl")
            # Load the dictionary from the pickle file
            # with open('Optmz_nmax2_spws400_spctl87_STAT.RESULTS.pkl', 'rb') as f:
                # stat_results = pickle.load(f)

            del u10_obs_tmax, hs_obs_tmax, prob_u10_gefs_forecast, prob_hs_gefs_forecast, fprob_u10_gefs_forecast, fprob_hs_gefs_forecast
            del stat_results, ceval_gefs_u10, ceval_gefs_hs, froc_u10, froc_hs, fbriers_u10, fbriers_hs, crps_u10, mean_crps_u10, crps_hs, mean_crps_hs

            print(" - Done Optmz_nmax"+str(int(nmax))+"_spws"+str(int(spws*100)).zfill(3)+"_spctl"+str(int(spctl)))
            print(" Sim "+repr(sim))
            print(" ")
            sim = sim + 1


