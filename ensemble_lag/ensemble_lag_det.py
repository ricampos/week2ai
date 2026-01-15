#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ensemble_lag_det.py

VERSION AND LAST UPDATE:
 v1.0  09/16/2025
 v2.0  01/15/2026

PURPOSE:
 Experiments with ensemble lag, including validation and comparisons.
 This can be used to assess and optimize the ensemble size, lag, and parameters of the prob forecast.

USAGE:
 Input arguments: see the first block and input parameters below

DEPENDENCIES:
 See the imports below.
 Check https://github.com/NOAA-EMC/gefswaves_reforecast/tree/main/fuzzy_verification for wcpval.py

AUTHOR and DATE:
 09/16/2025: Ricardo M. Campos, first version based on fuzzy_verification_GEFS.py
 01/15/2026: Ricardo M. Campos, complete validation statistics.

PERSON OF CONTACT:
 Ricardo M Campos: ricardo.campos@noaa.gov

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
import warnings; warnings.filterwarnings("ignore")

sl=13
matplotlib.rcParams.update({'font.size': sl}); plt.rc('font', size=sl)
matplotlib.rc('xtick', labelsize=sl); matplotlib.rc('ytick', labelsize=sl); matplotlib.rcParams.update({'font.size': sl})

def rankhist(model,obs,ftag):

        from scipy.stats import rankdata
        # remove NaN
        ind=np.where( (np.mean(model,axis=1)>-999.) & (obs>-999.) )
        if np.size(ind)>0:
            obs=np.copy(obs[ind[0]])
            model=np.copy(model[ind[0],:]).T; del ind
        else:
            raise ValueError(' No quality data available.')

        combined=np.vstack((obs[np.newaxis],model))
        ranks=np.apply_along_axis(lambda x: rankdata(x,method='min'),0,combined)
        ties=np.sum(ranks[0]==ranks[1:], axis=0)
        ranks=ranks[0]
        tie=np.unique(ties)
        for i in range(1,len(tie)):
            index=ranks[ties==tie[i]]
            ranks[ties==tie[i]]=[np.random.randint(index[j],index[j]+tie[i]+1,tie[i])[0] for j in range(len(index))]

        result = np.histogram(ranks, bins=np.linspace(0.5, combined.shape[0]+0.5, combined.shape[0]+1))
        # plot
        plt.close('all')
        fig1 = plt.figure(1,figsize=(6,5)); ax1 = fig1.add_subplot(111)
        fresult=result[0]/np.sum(result[0])
        rxaxis=range(1,len(result[0])+1)
        plt.bar(rxaxis,fresult,color='grey', edgecolor='k')
        plt.xlim(xmax = len(result[0])+0.9, xmin = 0.1)
        ax1.set_xlabel('Rank',size=sl); ax1.set_ylabel('Probability',size=sl)
        plt.tight_layout(); # plt.axis('tight') 
        plt.grid(c='grey', ls='--', alpha=0.3)
        plt.savefig("RankHistogram_"+ftag+".png", dpi=200, facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close('all'); del fig1, ax1


if __name__ == "__main__":

    # point manual selection (index)
    # bid = np.array([24,25,31]).astype('int')  # Atlantic na_center_north
    bid = np.array([2,10,11,13]).astype('int')  # Pacific np_center_north
    lbid = len(bid)
    # variable (u10 or hs)
    wvar='hs'
    # Forecast Lead Time (Day) and intervall
    ltime1=0; ltime2=np.inf
    # output path
    # opath="/work/noaa/marine/ricardo.campos/work/analysis/4postproc/ensemble_lag/Atlantic/det"
    opath="/work/noaa/marine/ricardo.campos/work/analysis/4postproc/ensemble_lag/Pacific/det"
    # file tag for output file names
    ftag=opath+"/ValidationEnsLag_"+wvar+"_"

    # Observations
    # fobsname="/work/noaa/marine/ricardo.campos/work/analysis/3assessments/fuzzy_verification/data/Atlantic/Altimeter.Buoy.PointExtract.Atlantic_20201001to20250101.nc"
    fobsname="/work/noaa/marine/ricardo.campos/work/analysis/3assessments/fuzzy_verification/data/Pacific/Altimeter.Buoy.PointExtract.Pacific_20201001to20250101.nc"

    # obs = read_obs(fobsname,wvar,"mean")
    obs = read_obs(fobsname,wvar,"max")

    # ---- Read config file and statistical parameters (exaclty the same as the operational) -----
    print(" "); print(" Reading yaml configuration file ...")
    with open('probmaps_gefs.yaml', 'r') as file:
        wconfig = yaml.safe_load(file)

    if wvar=="u10":
        fqlev = np.array(wconfig['qlev_wnd']).astype('float')
        qlev = fqlev/1.94 # m/s
    else:
        qlev = np.array(wconfig["qlev_"+wvar]).astype('float')
        fqlev = qlev

    plevels = np.array(wconfig['hplevels']).astype('float'); lplev=int(len(plevels))
    nmax = int(wconfig['nmax']); spws = int(wconfig['spws']); spctl = float(wconfig['spctl'])
    print(" Read yaml configuration file, OK")

    # -------------------------------------
    # READ DATA
    print(" Reading Model Data ...")
    # list of netcdf files generated with buildfuzzydataset.py
    # ls -d $PWD/*.nc > list.txt &
    wlist = np.atleast_1d(np.loadtxt("list.txt",dtype=str)) 
    gdata = read_data(wlist,bid,ltime1,ltime2,wvar)
    indlat = gdata['indlat']; indlon = gdata['indlon']
    print(" Reading Model Data, OK")

    print(" Prepare arrays ...")
    # lengths
    lstw=int(len(wlist))
    lft = gdata['lft']; lct = len(gdata['ctime'])
    llatm = len(gdata['latm'][0,:]); llonm = len(gdata['lonm'][0,:])
    lensm = len(gdata['ensm'])

    gdata['lonm'][gdata['lonm']>180] = gdata['lonm'][gdata['lonm']>180]-360.

    print(" Bias correction and DA ...")
    # Bias correction
    gefs_hindcast_bc = bias_correction(gdata,obs,spctl,wvar,opath,include_buoy="yes")
    # Data Assimilation
    gefs_hindcast_da, bda, sda = data_assimilation(gefs_hindcast_bc,gdata,obs,spctl,wvar,opath,include_buoy="yes")

    print(" Ensemble Lag mount ...")
    # ground truth
    gefs_hindcast = np.copy(gefs_hindcast_da)
    # center point of the spatial window around the target location
    gefs = gdata['gefs_forecast'][:,:,:,:,gdata['indlat'],gdata['indlon']]
    elag = np.arange(0,2*24+1,12) # max of 2 days: goal is week2 and limit is 16 days.
    g_lg = np.zeros((gdata['ftime'].shape[0],len(gdata['stations']),gdata['ftime'].shape[1],len(gdata['ensm'])*len(elag)),'f')*np.nan
    for i in range(len(elag),gdata['ftime'].shape[0]):
        for j in range(0,len(elag)):
            if gdata['ftime'][i,0] == gdata['ftime'][i-j,j*2]:
                q = 31*j; p = j*2
                if p==0:
                    g_lg[i,:,:,:][:,:,q:q+31] = gefs[i-j,:,p::,:]
                else:
                    g_lg[i,:,:,:][:,0:-p,:][:,:,q:q+31] = gefs[i-j,:,p::,:]

    g_lg = g_lg[len(elag)::,:,:,:]
    gefs_hindcast = gefs_hindcast[len(elag)::,:,:]
    del gefs

    ind_elag = np.array([6,11,21,31,62,93,124,155]) # indexes to select each one.

    print(" Ens Mean and Spread ...")
    # control
    g_control = g_lg[:,:,:,0]
    # ensemble mean and standard deviation (31 members, "default")
    indstdrd = 3
    g_l = np.zeros((len(ind_elag),g_lg.shape[0],g_lg.shape[1],g_lg.shape[2]),'f')*np.nan
    s_l = np.zeros((len(ind_elag),g_lg.shape[0],g_lg.shape[1],g_lg.shape[2]),'f')*np.nan
    for i in range(0,len(ind_elag)):
        g_l[i,:,:,:,] = np.nanmean(g_lg[:,:,:,0:ind_elag[i]],axis=3)
        s_l[i,:,:,:,] = np.std(g_lg[:,:,:,0:ind_elag[i]],axis=3)

    # 14 days array
    frtags=np.array(np.arange(0,14,1)+1.).astype('int')

    hours = np.arange(0, 14*24 + 6, 6); nsteps_per_day = 24 // 6; ndays = len(hours) // nsteps_per_day
    frintervals = np.array([[i*nsteps_per_day, (i+1)*nsteps_per_day] for i in range(ndays)])
    frintervals[-1,-1]=55

    # =====================================================
    print(" VALIDATION ...")
    # 8 error metrics
    nerrm=np.array(['bias','RMSE','NBias','NRMSE','SCrmse','SI','HH','CC'])
    hdem="bias, RMSE, NBias, NRMSE, SCrmse, SI, HH, CC"

    errm = np.zeros((len(ind_elag)+1,len(frtags),len(gdata['stations']),8),'f')*np.nan

    for i in range(0,len(frtags)):
        for j in range(0,len(gdata['stations'])): 
            a = gefs_hindcast[:,j,frintervals[i,0]:(frintervals[i,1]+1)].reshape(gefs_hindcast.shape[0]*((frintervals[i,1]+1)-frintervals[i,0]))

            # Control
            m = g_control[:,j,frintervals[i,0]:(frintervals[i,1]+1)].reshape(gefs_hindcast.shape[0]*((frintervals[i,1]+1)-frintervals[i,0]))
            ind = np.where((a>0.1)&(m>0.1))
            if np.size(ind)>10:
                ind=ind[0]
                errm[0,i,j,:]=np.array(mvalstats.metrics(a[ind],m[ind],0.2,20.,19.)[0:-1])

            del m

            for k in range(0,len(ind_elag)):
                m = g_l[k,:,j,frintervals[i,0]:(frintervals[i,1]+1)].reshape(gefs_hindcast.shape[0]*((frintervals[i,1]+1)-frintervals[i,0]))
                ind = np.where((a>0.1)&(m>0.1))
                if np.size(ind)>10:
                    ind=ind[0]
                    errm[k+1,i,j,:]=np.array(mvalstats.metrics(a[ind],m[ind],0.2,20.,19.)[0:-1])

                del m

    # ================= Plots ================= 

    lagcol = ['y','c','lime','k','b','m','darkgreen','r']
    lagmarl = [':',':','-.','--','-.','--','-.','--']

    #
    fig1 = plt.figure(1,figsize=(8,5)); ax = fig1.add_subplot(111)
    ax.plot(frtags,np.nanmean(errm[0,:,:,0],axis=1),'grey',linestyle='--',label='Ctrl', linewidth=2.,zorder=2)
    for i in range(0,len(ind_elag)):
        ax.plot(frtags,np.nanmean(errm[i+1,:,:,0],axis=1),lagcol[i],linestyle=lagmarl[i],label="EM_"+repr(ind_elag[i])+"m", linewidth=2.,zorder=2)

    ax.set_xticks(np.arange(1, 14 + 1, 1))
    ax.set_xlabel('Forecast Lead Time (Days)',size=sl)
    ax.set_ylabel('Bias')
    plt.legend(loc='best',fontsize=sl-3)
    ax.grid(c='dimgrey', ls='--', alpha=0.3)
    plt.tight_layout();plt.axis('tight')
    plt.xlim(xmin = 0.9, xmax = 14.1)
    plt.savefig("ErrXForecastTime_Bias.png", dpi=200, facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1); del fig1, ax
    plt.close('all')

    #
    fig1 = plt.figure(1,figsize=(8,5)); ax = fig1.add_subplot(111)
    ax.plot(frtags,np.nanmean(errm[0,:,:,1],axis=1),'grey',linestyle='--',label='Ctrl', linewidth=2.,zorder=2)
    for i in range(0,len(ind_elag)):
        ax.plot(frtags,np.nanmean(errm[i+1,:,:,1],axis=1),lagcol[i],linestyle=lagmarl[i],label="EM_"+repr(ind_elag[i])+"m", linewidth=2.,zorder=2)

    ax.set_xticks(np.arange(1, 14 + 1, 1))
    ax.set_xlabel('Forecast Lead Time (Days)',size=sl)
    ax.set_ylabel('RMSE')
    plt.legend(loc='best',fontsize=sl-3)
    ax.grid(c='dimgrey', ls='--', alpha=0.3)
    plt.tight_layout();plt.axis('tight')
    plt.xlim(xmin = 0.9, xmax = 14.1)
    plt.savefig("ErrXForecastTime_RMSE.png", dpi=200, facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1); del fig1, ax
    plt.close('all')

    #
    fig1 = plt.figure(1,figsize=(8,5)); ax = fig1.add_subplot(111)
    ax.plot(frtags,np.nanmean(errm[0,:,:,4],axis=1),'grey',linestyle='--',label='Ctrl', linewidth=2.,zorder=2)
    for i in range(0,len(ind_elag)):
        ax.plot(frtags,np.nanmean(errm[i+1,:,:,4],axis=1),lagcol[i],linestyle=lagmarl[i],label="EM_"+repr(ind_elag[i])+"m", linewidth=2.,zorder=2)

    ax.set_xticks(np.arange(1, 14 + 1, 1))
    ax.set_xlabel('Forecast Lead Time (Days)',size=sl)
    ax.set_ylabel('SCrmse')
    plt.legend(loc='best',fontsize=sl-3)
    ax.grid(c='dimgrey', ls='--', alpha=0.3)
    plt.tight_layout();plt.axis('tight')
    plt.xlim(xmin = 0.9, xmax = 14.1)
    plt.savefig("ErrXForecastTime_SCrmse.png", dpi=200, facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1); del fig1, ax
    plt.close('all')

    #
    fig1 = plt.figure(1,figsize=(8,5)); ax = fig1.add_subplot(111)
    ax.plot(frtags,np.nanmean(errm[0,:,:,5],axis=1),'grey',linestyle='--',label='Ctrl', linewidth=2.,zorder=2)
    for i in range(0,len(ind_elag)):
        ax.plot(frtags,np.nanmean(errm[i+1,:,:,5],axis=1),lagcol[i],linestyle=lagmarl[i],label="EM_"+repr(ind_elag[i])+"m", linewidth=2.,zorder=2)

    ax.set_xticks(np.arange(1, 14 + 1, 1))
    ax.set_xlabel('Forecast Lead Time (Days)',size=sl)
    ax.set_ylabel('SI')
    plt.legend(loc='best',fontsize=sl-3)
    ax.grid(c='dimgrey', ls='--', alpha=0.3)
    plt.tight_layout();plt.axis('tight')
    plt.xlim(xmin = 0.9, xmax = 14.1)
    plt.savefig("ErrXForecastTime_SI.png", dpi=200, facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1); del fig1, ax
    plt.close('all')

    #
    fig1 = plt.figure(1,figsize=(8,5)); ax = fig1.add_subplot(111)
    ax.plot(frtags,np.nanmean(errm[0,:,:,7],axis=1),'grey',linestyle='--',label='Ctrl', linewidth=2.,zorder=2)
    for i in range(0,len(ind_elag)):
        ax.plot(frtags,np.nanmean(errm[i+1,:,:,7],axis=1),lagcol[i],linestyle=lagmarl[i],label="EM_"+repr(ind_elag[i])+"m", linewidth=2.,zorder=2)

    ax.set_xticks(np.arange(1, 14 + 1, 1))
    ax.set_xlabel('Forecast Lead Time (Days)',size=sl)
    ax.set_ylabel('CC')
    plt.legend(loc='best',fontsize=sl-3)
    ax.grid(c='dimgrey', ls='--', alpha=0.3)
    plt.tight_layout();plt.axis('tight')
    plt.xlim(xmin = 0.9, xmax = 14.1)
    plt.savefig("ErrXForecastTime_CC.png", dpi=200, facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1); del fig1, ax
    plt.close('all')


    # -------
    fig1 = plt.figure(1,figsize=(8,5)); ax = fig1.add_subplot(111)
    # ax.plot(frtags,(np.nanmean(errm[0,:,:,0],axis=1)-np.nanmean(errm[indstdrd+1,:,:,0],axis=1)),'grey',linestyle='--',label='Ctrl', linewidth=2.,zorder=2)
    for i in range(0,len(ind_elag)):
        if i!=indstdrd:
            ax.plot(frtags,(np.nanmean(errm[i+1,:,:,0],axis=1)-np.nanmean(errm[indstdrd+1,:,:,0],axis=1)),lagcol[i],linestyle=lagmarl[i],label="EM_"+repr(ind_elag[i])+"m", linewidth=2.,zorder=2)

    ax.axhline(y=0, color='k', linewidth=1)
    ax.set_xticks(np.arange(1, 14 + 1, 1))
    ax.set_xlabel('Forecast Lead Time (Days)',size=sl)
    ax.set_ylabel('Bias')
    plt.legend(loc='best',fontsize=sl-4)
    ax.grid(c='dimgrey', ls='--', alpha=0.3)
    plt.tight_layout();plt.axis('tight')
    plt.xlim(xmin = 0.9, xmax = 14.1)
    plt.savefig("ErrXForecastTime_minusEM_Bias.png", dpi=200, facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1); del fig1, ax; plt.close('all')

    fig1 = plt.figure(1,figsize=(8,5)); ax = fig1.add_subplot(111)
    # ax.plot(frtags,(np.nanmean(errm[0,:,:,1],axis=1)-np.nanmean(errm[indstdrd+1,:,:,1],axis=1)),'grey',linestyle='--',label='Ctrl', linewidth=2.,zorder=2)
    saux=[]
    for i in range(0,len(ind_elag)):
        if i!=indstdrd:
            ax.plot(frtags,(np.nanmean(errm[i+1,:,:,1],axis=1)-np.nanmean(errm[indstdrd+1,:,:,1],axis=1)),lagcol[i],linestyle=lagmarl[i],label="EM_"+repr(ind_elag[i])+"m", linewidth=2.,zorder=2)
            saux=np.append(saux,(np.nanmean(errm[i+1,:,:,1],axis=1)-np.nanmean(errm[indstdrd+1,:,:,1],axis=1)))

    ax.axhline(y=0, color='k', linewidth=1)
    ax.set_xticks(np.arange(1, 14 + 1, 1))
    ax.set_xlabel('Forecast Lead Time (Days)',size=sl)
    ax.set_ylabel('RMSE')
    plt.legend(loc='best',fontsize=sl-4)
    ax.grid(c='dimgrey', ls='--', alpha=0.3)
    plt.tight_layout();plt.axis('tight')
    plt.xlim(xmin = 0.9, xmax = 14.1)
    sa2 = saux.max()-saux.min()
    plt.ylim(ymin = saux.min()-0.15*sa2, ymax = saux.max()+0.05*np.abs(saux.max()))
    plt.savefig("ErrXForecastTime_minusEM_RMSE.png", dpi=200, facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1); del fig1, ax; plt.close('all')

    fig1 = plt.figure(1,figsize=(8,5)); ax = fig1.add_subplot(111)
    # ax.plot(frtags,(np.nanmean(errm[0,:,:,4],axis=1)-np.nanmean(errm[indstdrd+1,:,:,4],axis=1)),'grey',linestyle='--',label='Ctrl', linewidth=2.,zorder=2)
    saux=[]
    for i in range(0,len(ind_elag)):
        if i!=indstdrd:
            ax.plot(frtags,(np.nanmean(errm[i+1,:,:,4],axis=1)-np.nanmean(errm[indstdrd+1,:,:,4],axis=1)),lagcol[i],linestyle=lagmarl[i],label="EM_"+repr(ind_elag[i])+"m", linewidth=2.,zorder=2)
            saux=np.append(saux,(np.nanmean(errm[i+1,:,:,4],axis=1)-np.nanmean(errm[indstdrd+1,:,:,4],axis=1)))

    ax.axhline(y=0, color='k', linewidth=1)
    ax.set_xticks(np.arange(1, 14 + 1, 1))
    ax.set_xlabel('Forecast Lead Time (Days)',size=sl)
    ax.set_ylabel('SCrmse')
    plt.legend(loc='best',fontsize=sl-4)
    ax.grid(c='dimgrey', ls='--', alpha=0.3)
    plt.tight_layout();plt.axis('tight')
    plt.xlim(xmin = 0.9, xmax = 14.1)
    plt.ylim(ymin = saux.min()-0.05*np.abs(saux.min()), ymax = saux.max()+0.05*np.abs(saux.max()))
    plt.savefig("ErrXForecastTime_minusEM_SCrmse.png", dpi=200, facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1); del fig1, ax; plt.close('all')

    fig1 = plt.figure(1,figsize=(8,5)); ax = fig1.add_subplot(111)
    # ax.plot(frtags,(np.nanmean(errm[0,:,:,5],axis=1)-np.nanmean(errm[indstdrd+1,:,:,5],axis=1)),'grey',linestyle='--',label='Ctrl', linewidth=2.,zorder=2)
    saux=[]
    for i in range(0,len(ind_elag)):
        if i!=indstdrd:
            ax.plot(frtags,(np.nanmean(errm[i+1,:,:,5],axis=1)-np.nanmean(errm[indstdrd+1,:,:,5],axis=1)),lagcol[i],linestyle=lagmarl[i],label="EM_"+repr(ind_elag[i])+"m", linewidth=2.,zorder=2)
            saux=np.append(saux,(np.nanmean(errm[i+1,:,:,5],axis=1)-np.nanmean(errm[indstdrd+1,:,:,5],axis=1)))

    ax.axhline(y=0, color='k', linewidth=1)
    ax.set_xticks(np.arange(1, 14 + 1, 1))
    ax.set_xlabel('Forecast Lead Time (Days)',size=sl)
    ax.set_ylabel('SI')
    plt.legend(loc='best',fontsize=sl-4)
    ax.grid(c='dimgrey', ls='--', alpha=0.3)
    plt.tight_layout();plt.axis('tight')
    plt.xlim(xmin = 0.9, xmax = 14.1)
    plt.ylim(ymin = saux.min()-0.05*np.abs(saux.min()), ymax = saux.max()+0.05*np.abs(saux.max()))
    plt.savefig("ErrXForecastTime_minusEM_SI.png", dpi=200, facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1); del fig1, ax; plt.close('all')

    fig1 = plt.figure(1,figsize=(8,5)); ax = fig1.add_subplot(111)
    # ax.plot(frtags,(np.nanmean(errm[0,:,:,7],axis=1)-np.nanmean(errm[indstdrd+1,:,:,7],axis=1)),'grey',linestyle='--',label='Ctrl', linewidth=2.,zorder=2)
    saux=[]
    for i in range(0,len(ind_elag)):
        if i!=indstdrd:
            ax.plot(frtags,(np.nanmean(errm[i+1,:,:,7],axis=1)-np.nanmean(errm[indstdrd+1,:,:,7],axis=1)),lagcol[i],linestyle=lagmarl[i],label="EM_"+repr(ind_elag[i])+"m", linewidth=2.,zorder=2)
            saux=np.append(saux,(np.nanmean(errm[i+1,:,:,7],axis=1)-np.nanmean(errm[indstdrd+1,:,:,7],axis=1)))

    ax.axhline(y=0, color='k', linewidth=1)
    ax.set_xticks(np.arange(1, 14 + 1, 1))
    ax.set_xlabel('Forecast Lead Time (Days)',size=sl)
    ax.set_ylabel('CC')
    plt.legend(loc='best',fontsize=sl-4)
    ax.grid(c='dimgrey', ls='--', alpha=0.3)
    plt.tight_layout();plt.axis('tight')
    plt.xlim(xmin = 0.9, xmax = 14.1)
    sa2 = saux.max()-saux.min()
    plt.ylim(ymin = saux.min()-0.05*np.abs(saux.min()), ymax = saux.max()+0.15*sa2)
    plt.savefig("ErrXForecastTime_minusEM_CC.png", dpi=200, facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1); del fig1, ax; plt.close('all')

    # ---------------------------
    print(" Spread ...")

    for i in range(0,frintervals.shape[0]):

        ensprd = np.zeros((len(ind_elag)),'f')*np.nan

        for j in range(0,len(ind_elag)):

            m = g_lg[:,:,frintervals[i,:][0]:frintervals[i,:][1],:][:,:,:,0:ind_elag[j]]; m=m.reshape(m.shape[0]*m.shape[1]*m.shape[2],m.shape[3])
            ensprd[j] = np.nanmean(np.std(m,1)); del m

        fig1, ax = plt.subplots(figsize=(6, 5))
        ax.plot(ind_elag, ensprd, color='k', marker='.', linestyle='', linewidth=2., zorder=2)
        ax.plot(ind_elag, ensprd, color='k', linestyle='-', linewidth=2.,label='Day'+repr(i+1), zorder=2)
        ax.set_xlim(1, ind_elag.max()+1)
        ax.set_xlabel("Ensemble Size")
        ax.set_ylabel("Spread")
        ax.legend(fontsize=sl)
        plt.tight_layout(); # plt.axis('tight') 
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
        plt.savefig("SpredXEnsSize_Day"+str(i+1).zfill(2)+".png", dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
                    format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close('all')

        del ensprd


    print(" Rank Histogram ...")
    for i in [1,4,7,10,14]:
        a = gefs_hindcast[:,:,frintervals[i-1,:][0]:frintervals[i-1,:][1]]; a=a.reshape(a.shape[0]*a.shape[1]*a.shape[2])
        for j in range(0,len(ind_elag)):
            ftag="EL_"+str(ind_elag[j]).zfill(2)+"members_Day"+str(i).zfill(2)
            m = g_lg[:,:,frintervals[i-1,:][0]:frintervals[i-1,:][1],:][:,:,:,0:ind_elag[j]]; m=m.reshape(m.shape[0]*m.shape[1]*m.shape[2],m.shape[3])
            rankhist(m,a,ftag); del m

        del a

    print(" "); print("Analysis Completed"); print(" ")

