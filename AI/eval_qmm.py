#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
# matplotlib.use('Agg')
import pickle
from matplotlib.dates import DateFormatter
import netCDF4 as nc
import numpy as np
import pandas as pd
import sys
import pandas as pd
import matplotlib.pyplot as plt
import properscoring as ps
from datetime import datetime
# from pvalstats import ModelObsPlot
import warnings; warnings.filterwarnings("ignore")

sl=13
matplotlib.rcParams.update({'font.size': sl}); plt.rc('font', size=sl) 
matplotlib.rc('xtick', labelsize=sl); matplotlib.rc('ytick', labelsize=sl); matplotlib.rcParams.update({'font.size': sl})


if __name__ == "__main__":

    # select one point
    stations = np.array(['46005','46006','46066']).astype('str')
    # Forecast Lead Time (Day) and intervall
    ltime1=7; ltime2=14
    opath="/work/noaa/marine/ricardo.campos/work/analysis/4postproc/output/qmm" # on Orion

    # Input Argument. Optimized Parameters  

    print(" ")
    print(" === Eval Bias Correction, QMM === ")
    print(" ")
    # ------------------

    # ---- Statistical parameters (exaclty the same as the operational config file) -----
    qlev_hs = np.array([ 4.,  6.,  9.]).astype('float')
    qlev_wnd = np.array([28.0, 34.0, 41.0, 48.0]).astype('float')/1.94
    plevels = np.array([0.15, 0.5, 0.65, 0.8, 1.0])

    spctl = np.array([80,83,85,87,89,91,93,95,98]).astype('float')

    c=0
    for i in range(0,len(spctl)):

        if c==0:

            POD_hs = np.zeros((4,len(spctl),len(qlev_hs),len(plevels)-1),'f')*np.nan
            FAR_hs = np.zeros((4,len(spctl),len(qlev_hs),len(plevels)-1),'f')*np.nan
            CSI_hs = np.zeros((4,len(spctl),len(qlev_hs),len(plevels)-1),'f')*np.nan
            POD_u10 = np.zeros((4,len(spctl),len(qlev_wnd),len(plevels)-1),'f')*np.nan
            FAR_u10 = np.zeros((4,len(spctl),len(qlev_wnd),len(plevels)-1),'f')*np.nan
            CSI_u10 = np.zeros((4,len(spctl),len(qlev_wnd),len(plevels)-1),'f')*np.nan
            ROCauc_hs = np.zeros((4,len(spctl),len(qlev_hs)),'f')*np.nan
            ROCauc_u10 = np.zeros((4,len(spctl),len(qlev_wnd)),'f')*np.nan
            BRIER_hs = np.zeros((4,len(spctl),len(qlev_hs)),'f')*np.nan
            BRIER_u10 = np.zeros((4,len(spctl),len(qlev_wnd)),'f')*np.nan
            CRPS_hs = np.zeros((4,len(spctl)),'f')*np.nan
            CRPS_u10 = np.zeros((4,len(spctl)),'f')*np.nan


        fname = "/work/noaa/marine/ricardo.campos/work/analysis/4postproc/output/optmz/pickle/Optmz_nmax2_spws200_spctl"+str(int(spctl[i]))+"_STAT.RESULTS.pkl"
        with open(fname,'rb') as f:
            stat_results = pickle.load(f)

        POD_hs[0,i,:,:] = stat_results['ceval_gefs_hs'][:,:,0]
        FAR_hs[0,i,:,:] = stat_results['ceval_gefs_hs'][:,:,1] 
        CSI_hs[0,i,:,:] = stat_results['ceval_gefs_hs'][:,:,2]
        POD_u10[0,i,:,:] = stat_results['ceval_gefs_u10'][:,:,0]
        FAR_u10[0,i,:,:] = stat_results['ceval_gefs_u10'][:,:,1] 
        CSI_u10[0,i,:,:] = stat_results['ceval_gefs_u10'][:,:,2]
        ROCauc_hs[0,i,:] = stat_results['froc_hs'][:]
        ROCauc_u10[0,i,:] = stat_results['froc_u10'][:]
        BRIER_hs[0,i,:] = stat_results['fbriers_hs'][:]
        BRIER_u10[0,i,:] = stat_results['fbriers_u10'][:]
        CRPS_hs[0,i] = stat_results['mean_crps_hs']
        CRPS_u10[0,i] = stat_results['mean_crps_u10']

        fname = "/work/noaa/marine/ricardo.campos/work/analysis/4postproc/output/optmz/pickle/Optmz_nmax2_spws400_spctl"+str(int(spctl[i]))+"_STAT.RESULTS.pkl"
        with open(fname,'rb') as f:
            stat_results = pickle.load(f)

        POD_hs[1,i,:,:] = stat_results['ceval_gefs_hs'][:,:,0]
        FAR_hs[1,i,:,:] = stat_results['ceval_gefs_hs'][:,:,1] 
        CSI_hs[1,i,:,:] = stat_results['ceval_gefs_hs'][:,:,2]
        POD_u10[1,i,:,:] = stat_results['ceval_gefs_u10'][:,:,0]
        FAR_u10[1,i,:,:] = stat_results['ceval_gefs_u10'][:,:,1] 
        CSI_u10[1,i,:,:] = stat_results['ceval_gefs_u10'][:,:,2]
        ROCauc_hs[1,i,:] = stat_results['froc_hs'][:]
        ROCauc_u10[1,i,:] = stat_results['froc_u10'][:]
        BRIER_hs[1,i,:] = stat_results['fbriers_hs'][:]
        BRIER_u10[1,i,:] = stat_results['fbriers_u10'][:]
        CRPS_hs[1,i] = stat_results['mean_crps_hs']
        CRPS_u10[1,i] = stat_results['mean_crps_u10']

        del stat_results
        fname = "/work/noaa/marine/ricardo.campos/work/analysis/4postproc/output/qmm/spws_2p0/pickle/BiasCorrectedQMM_nmax2_spws200_spctl"+str(int(spctl[i]))+"_STAT.RESULTS.pkl"
        with open(fname,'rb') as f:
            stat_results = pickle.load(f)

        POD_hs[2,i,:,:] = stat_results['ceval_gefs_hs'][:,:,0]
        FAR_hs[2,i,:,:] = stat_results['ceval_gefs_hs'][:,:,1] 
        CSI_hs[2,i,:,:] = stat_results['ceval_gefs_hs'][:,:,2]
        POD_u10[2,i,:,:] = stat_results['ceval_gefs_u10'][:,:,0]
        FAR_u10[2,i,:,:] = stat_results['ceval_gefs_u10'][:,:,1] 
        CSI_u10[2,i,:,:] = stat_results['ceval_gefs_u10'][:,:,2]
        ROCauc_hs[2,i,:] = stat_results['froc_hs'][:]
        ROCauc_u10[2,i,:] = stat_results['froc_u10'][:]
        BRIER_hs[2,i,:] = stat_results['fbriers_hs'][:]
        BRIER_u10[2,i,:] = stat_results['fbriers_u10'][:]
        CRPS_hs[2,i] = stat_results['mean_crps_hs']
        CRPS_u10[2,i] = stat_results['mean_crps_u10']

        del stat_results
        fname = "/work/noaa/marine/ricardo.campos/work/analysis/4postproc/output/qmm/spws_4p0/pickle/BiasCorrectedQMM_nmax2_spws400_spctl"+str(int(spctl[i]))+"_STAT.RESULTS.pkl"
        with open(fname,'rb') as f:
            stat_results = pickle.load(f)

        POD_hs[3,i,:,:] = stat_results['ceval_gefs_hs'][:,:,0]
        FAR_hs[3,i,:,:] = stat_results['ceval_gefs_hs'][:,:,1] 
        CSI_hs[3,i,:,:] = stat_results['ceval_gefs_hs'][:,:,2]
        POD_u10[3,i,:,:] = stat_results['ceval_gefs_u10'][:,:,0]
        FAR_u10[3,i,:,:] = stat_results['ceval_gefs_u10'][:,:,1] 
        CSI_u10[3,i,:,:] = stat_results['ceval_gefs_u10'][:,:,2]
        ROCauc_hs[3,i,:] = stat_results['froc_hs'][:]
        ROCauc_u10[3,i,:] = stat_results['froc_u10'][:]
        BRIER_hs[3,i,:] = stat_results['fbriers_hs'][:]
        BRIER_u10[3,i,:] = stat_results['fbriers_u10'][:]
        CRPS_hs[3,i] = stat_results['mean_crps_hs']
        CRPS_u10[3,i] = stat_results['mean_crps_u10']

        c=c+1
        print(" Read "+fname); print(" "+repr(c)); print(" ")


    # ====== Save validation results ====== 

    hd_crps = "spws200, spws200_QM, spws400, spws400_QM "

    # --- Hs ---
    hd = "spws200, spws200_QM, spws400, spws400_QM "+' '.join(np.array(np.round(qlev_hs,2)).astype('str'))

    # POD
    for i in range(0,len(plevels)-1):
        fname = opath+"/EvalQMM_POD_Hs_"+str(plevels[i])+".txt"
        result=np.round(np.nanmax(POD_hs[:,:,:,i],axis=1),4)
        ifile = open(fname,'w')
        ifile.write(hd+' \n')
        np.savetxt(ifile,result.astype('str'),fmt="%s",delimiter='	') 
        ifile.close(); del ifile, fname, result

    # FAR
    for i in range(0,len(plevels)-1):
        fname = opath+"/EvalQMM_FAR_Hs_"+str(plevels[i])+".txt"
        result=np.round(np.nanmax(FAR_hs[:,:,:,i],axis=1),4)
        ifile = open(fname,'w')
        ifile.write(hd+' \n')
        np.savetxt(ifile,result.astype('str'),fmt="%s",delimiter='	') 
        ifile.close(); del ifile, fname, result

    # CSI
    for i in range(0,len(plevels)-1):
        fname = opath+"/EvalQMM_CSI_Hs_"+str(plevels[i])+".txt"
        result=np.round(np.nanmax(CSI_hs[:,:,:,i],axis=1),4)
        ifile = open(fname,'w')
        ifile.write(hd+' \n')
        np.savetxt(ifile,result.astype('str'),fmt="%s",delimiter='	') 
        ifile.close(); del ifile, fname, result

    # ROC
    fname = opath+'/EvalQMM_ROC_Hs.txt'
    result=np.round(np.nanmax(ROCauc_hs,axis=1),4)
    ifile = open(fname,'w')
    ifile.write(hd+' \n')
    np.savetxt(ifile,result.astype('str'),fmt="%s",delimiter='	') 
    ifile.close(); del ifile, fname, result

    # BRIER
    fname = opath+'/EvalQMM_BRIER_Hs.txt'
    result=np.round(np.nanmax(BRIER_hs,axis=1),4)
    ifile = open(fname,'w')
    ifile.write(hd+' \n')
    np.savetxt(ifile,result.astype('str'),fmt="%s",delimiter='	') 
    ifile.close(); del ifile, fname, result

    # CRPS
    fname = opath+'/EvalQMM_CRPS_Hs.txt'
    result=np.round(np.nanmax(CRPS_hs,axis=1),4)
    ifile = open(fname,'w')
    ifile.write(hd_crps+' \n')
    np.savetxt(ifile,result.astype('str'),fmt="%s",delimiter='	') 
    ifile.close(); del ifile, fname, result

    # --- U10 ---
    hd = "spws200, spws200_QM, spws400, spws400_QM "+' '.join(np.array(np.round(qlev_wnd,2)).astype('str'))

    # POD
    for i in range(0,len(plevels)-1):
        fname = opath+"/EvalQMM_POD_U10_"+str(plevels[i])+".txt"
        result=np.round(np.nanmax(POD_u10[:,:,:,i],axis=1),4)
        ifile = open(fname,'w')
        ifile.write(hd+' \n')
        np.savetxt(ifile,result.astype('str'),fmt="%s",delimiter='	') 
        ifile.close(); del ifile, fname, result

    # FAR
    for i in range(0,len(plevels)-1):
        fname = opath+"/EvalQMM_FAR_U10_"+str(plevels[i])+".txt"
        result=np.round(np.nanmax(FAR_u10[:,:,:,i],axis=1),4)
        ifile = open(fname,'w')
        ifile.write(hd+' \n')
        np.savetxt(ifile,result.astype('str'),fmt="%s",delimiter='	') 
        ifile.close(); del ifile, fname, result

    # CSI
    for i in range(0,len(plevels)-1):
        fname = opath+"/EvalQMM_CSI_U10_"+str(plevels[i])+".txt"
        result=np.round(np.nanmax(CSI_u10[:,:,:,i],axis=1),4)
        ifile = open(fname,'w')
        ifile.write(hd+' \n')
        np.savetxt(ifile,result.astype('str'),fmt="%s",delimiter='	') 
        ifile.close(); del ifile, fname, result

    # ROC
    fname = opath+'/EvalQMM_ROC_U10.txt'
    result=np.round(np.nanmax(ROCauc_u10,axis=1),4)
    ifile = open(fname,'w')
    ifile.write(hd+' \n')
    np.savetxt(ifile,result.astype('str'),fmt="%s",delimiter='	') 
    ifile.close(); del ifile, fname, result

    # BRIER
    fname = opath+'/EvalQMM_BRIER_U10.txt'
    result=np.round(np.nanmax(BRIER_u10,axis=1),4)
    ifile = open(fname,'w')
    ifile.write(hd+' \n')
    np.savetxt(ifile,result.astype('str'),fmt="%s",delimiter='	') 
    ifile.close(); del ifile, fname, result

    # CRPS
    fname = opath+'/EvalQMM_CRPS_U10.txt'
    result=np.round(np.nanmax(CRPS_u10,axis=1),4)
    ifile = open(fname,'w')
    ifile.write(hd_crps+' \n')
    np.savetxt(ifile,result.astype('str'),fmt="%s",delimiter='	') 
    ifile.close(); del ifile, fname, result

