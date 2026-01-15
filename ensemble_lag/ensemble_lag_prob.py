#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ensemble_lag_prob.py

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
 01/15/2026: Ricardo M. Campos, complete validation statistics (probabilistic in this code).

PERSON OF CONTACT:
 Ricardo M Campos: ricardo.campos@noaa.gov

"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.ndimage.filters import gaussian_filter
import yaml
from wcpval import *
import warnings; warnings.filterwarnings("ignore")

sl=13
matplotlib.rcParams.update({'font.size': sl}); plt.rc('font', size=sl)
matplotlib.rc('xtick', labelsize=sl); matplotlib.rc('ytick', labelsize=sl); matplotlib.rcParams.update({'font.size': sl})

if __name__ == "__main__":

    # point manual selection (index)
    # bid = np.array([24,25,31]).astype('int')  # Atlantic na_center_north
    bid = np.array([2,10,13]).astype('int')  # Pacific np_center_north
    lbid = len(bid)
    # variable (u10 or hs)
    wvar='hs'
    # Forecast Lead Time (Day) and intervall
    ltime1=7; ltime2=14
    # output path
    # opath="/work/noaa/marine/ricardo.campos/work/analysis/4postproc/ensemble_lag/Atlantic/prob"
    opath="/work/noaa/marine/ricardo.campos/work/analysis/4postproc/ensemble_lag/Pacific/prob"

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

    plevels = np.array(wconfig['plevels']).astype('float'); lplev=int(len(plevels)-1)
    nmax = int(wconfig['nmax']); spws = int(wconfig['spws']); spctl = float(wconfig['spctl'])
    print(" Read yaml configuration file, OK")

    # -------------------------------------
    # READ DATA
    print(" Reading Model Data ...")
    # list of netcdf files generated with buildfuzzydataset.py
    # ls -d $PWD/*.nc > list.txt &
    wlist = np.atleast_1d(np.loadtxt("list.txt",dtype=str)) 
    gdata = read_data(wlist,bid,0,np.inf,wvar)

    print(" Prepare arrays ...")
    indlat = gdata['indlat']; indlon = gdata['indlon']
    lstw=int(len(wlist))
    lft = gdata['lft']; lct = len(gdata['ctime'])
    llatm = len(gdata['latm'][0,:]); llonm = len(gdata['lonm'][0,:])

    gdata['lonm'][gdata['lonm']>180] = gdata['lonm'][gdata['lonm']>180]-360.

    gspws=int(np.floor(spws/np.diff(gdata['latm']).mean())/2)

    print(" Bias correction and DA ...")
    # Bias correction
    gefs_hindcast_bc = bias_correction(gdata,obs,spctl,wvar,opath,include_buoy="yes")
    # Data Assimilation
    gefs_hindcast_da, bda, sda = data_assimilation(gefs_hindcast_bc,gdata,obs,spctl,wvar,opath,include_buoy="yes")
    del gefs_hindcast_bc
    print(" Bias correction and DA Ok")

    # Select Week-2 interval
    auxt = np.array(gdata['ftime'][10,:]-np.min(gdata['ftime'][10,:]))/3600.
    indt = np.where( (auxt>=ltime1*24) & (auxt<=ltime2*24) )[0]
    gefs_hindcast_da = gefs_hindcast_da[:,:,indt]; del auxt
    lft = int(len(indt))

    print(" Ensemble Lag mount ...")
    elag = np.arange(0,2*24+1,12) # max of 2 days: goal is week2 and limit is 16 days.
    g_lg = np.zeros((gdata['ftime'].shape[0],len(gdata['stations']),gdata['ftime'].shape[1],len(gdata['ensm'])*len(elag),llatm,llonm),'f')*np.nan
    indc=[]; indfc=[]
    for i in range(len(elag),gdata['ftime'].shape[0]):
        for j in range(0,len(elag)):
            if gdata['ftime'][i,0] == gdata['ftime'][i-j,j*2]:
                q = 31*j; p = j*2
                # print(repr(q)+" "+repr(p))
                if p==0:
                    g_lg[i,:,:,:,:,:][:,:,q:q+31,:,:] = gdata['gefs_forecast'][i-j,:,p::,:,:,:]
                else:
                    g_lg[i,:,:,:,:,:][:,0:-p,:,:,:][:,:,q:q+31,:,:] = gdata['gefs_forecast'][i-j,:,p::,:,:,:]

                indfc=np.append(indfc,int(i))
            else:
                print("unmatched time at "+repr(i)+" "+repr(j))
                indc=np.append(indc,int(i))

    indc=np.array(indc).astype('int'); indc=np.unique(indc)
    indfc=np.array(indfc).astype('int'); indfc=np.unique(indfc)
    indfc=np.setdiff1d(indfc, indc, assume_unique=False)

    print(" Ensemble Lag mount, loop done.")
    del gdata

    print(" Select clean data and Week-2 interval.")
    gefs_hindcast_da = gefs_hindcast_da[indfc,:,:]
    g_lg = g_lg[:,:,indt,:,:,:]
    g_lg = g_lg[indfc,:,:,:,:,:]
    lct=int(g_lg.shape[0])
    print(" Selection, OK")
    ind_elag = np.array([6,11,21,31,62,93,124,155])
    # -------------------------------

    mcrps = np.zeros((len(ind_elag)),'f')*np.nan
    roc_auc = np.zeros((len(ind_elag),len(qlev)),'f')*np.nan
    roc_fpr = np.linspace(0.0, 1.0, 100)
    roc_tpr = np.zeros((len(ind_elag),len(qlev),100),'f')*np.nan
    briers = np.zeros((len(ind_elag),len(qlev)),'f')*np.nan
    ce_pod = np.zeros((len(ind_elag),len(qlev),lplev),'f')*np.nan
    ce_far = np.zeros((len(ind_elag),len(qlev),lplev),'f')*np.nan
    ce_csi = np.zeros((len(ind_elag),len(qlev),lplev),'f')*np.nan
    print(" Val arrays initialized")

    for l in range(0,len(ind_elag)):

        lensm=int(ind_elag[l])

        # Hindcast and ground thuth
        gtloop=int(lct*lbid)
        gefs_hindcast = np.tile(gefs_hindcast_da[:,:,:,None], lensm).reshape(gtloop,lft,lensm)
        gefs_hindcast_t = np.zeros((gtloop),'f')*np.nan
        for i in range(0,gtloop):
            aux = np.nanmean(gefs_hindcast[i,:,:],axis=1) # average ensemble members (hindcast)
            ind = np.where(aux>-999.)
            if np.size(ind)>int(np.floor(lft/2)) :
                ind = ind[0]
                gefs_hindcast_t[i] = np.nanmean(np.sort(aux[ind])[-nmax::])
            else:
                print(" forecast time series incomplete "+repr(i))
                gefs_hindcast_t[i] = np.nan

            del ind,aux
        # -----------------------

        gefs_forecast = g_lg[:,:,:,0:ind_elag[l],:,:].reshape(gtloop,lft,lensm,llatm,llonm)
        prob_gefs_forecast, fmod_result = probforecast(nmax,gspws,spctl,gefs_forecast,qlev)
        fprob_gefs_forecast = probforecast_binary(prob_gefs_forecast,qlev,plevels)

        # --- CRPS ---
        lixo, mcrps[l]  = crps(fmod_result,gefs_hindcast_t)

        for i in range(0,len(qlev)):
            try:
                # ROC curve
                true_binary = (gefs_hindcast_t > qlev[i]).astype(int)
                fpr, tpr, _ = roc_curve(true_binary, prob_gefs_forecast[:,i])
                roc_auc[l,i] = auc(fpr, tpr)
                tpr_interp = np.interp(roc_fpr, fpr, tpr)
                roc_tpr[l,i,:] = tpr_interp
                del true_binary, fpr, tpr, tpr_interp
                # Brier score
                briers[l,i] = brier_score(prob_gefs_forecast[:,i],gefs_hindcast_t,qlev[i])
                for j in range(0,lplev):
                    ceval_gefs_ghnd = categorical_bin_eval(fprob_gefs_forecast[:,i,j],gefs_hindcast_t,qlev[i])
                    ce_pod[l,i,j] = float(ceval_gefs_ghnd['POD'])
                    ce_far[l,i,j] = float(ceval_gefs_ghnd['FAR'])
                    ce_csi[l,i,j] = float(ceval_gefs_ghnd['CSI'])
                    del ceval_gefs_ghnd
            except:
		        print("  - Did not generate statistics for "+repr(ind_elag[l])+" "+repr(qlev[i]))
            else:
                print("  - Ok "+repr(ind_elag[l])+" "+repr(qlev[i]))

    print(" Statistics have been computed. Starting plots ...")

    # PLOTS --------------

    lagcol = ['y','c','lime','k','b','m','darkgreen','r']
    lagmarl = [':',':','-.','--','-.','--','-.','--']
    lagzo = np.array([3,3,4,5,4,4,4,3]).astype('int')

    # Categorical Val
    for i in range(0,len(qlev)):
        for j in range(0,len(plevels)-1): 
            plt.close('all')
            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax1.plot(ind_elag,ce_csi[:,i,j], color='k', linestyle='--', linewidth=2., label='CSI', zorder=3)
            ax1.plot(ind_elag,ce_csi[:,i,j], color='k', marker='.', linewidth=2., zorder=3)
            ax1.set_ylabel("CSI", color='k')
            ax1.tick_params(axis='y', labelcolor='k')
            ax1.set_xlabel("Ensemble Size")
            ax1.grid(c='grey', ls='--', alpha=0.3, zorder=1)

            ax2 = ax1.twinx()
            ax2.plot(ind_elag,ce_pod[:,i,j], color='royalblue', linestyle='--', linewidth=2., label='POD', zorder=3)
            ax2.set_ylabel("POD", color='royalblue')
            ax2.tick_params(axis='y', labelcolor='royalblue')

            ax3 = ax1.twinx()  # third y-axis
            ax3.spines["right"].set_position(("axes", 1.18))  # move it right of the default right axis
            ax3.plot(ind_elag, ce_far[:, i, j], color='firebrick', linestyle='-.', linewidth=2., label='FAR', zorder=3)
            ax3.set_ylabel("FAR", color='firebrick')
            ax3.tick_params(axis='y', labelcolor='firebrick')

            fig.tight_layout()
            plt.savefig("CSIPODFAR_q"+repr(qlev[i])+"_p"+repr(plevels[j])+".png", dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
                format='png', bbox_inches='tight', pad_inches=0.1)

    print(" Categorical Eval Done")

    # CRPS   
    plt.close('all')
    fig1, ax = plt.subplots(figsize=(6.5,6))
    ax.plot(ind_elag, mcrps, color='k', lw=2, zorder=3) 
    ax.plot(ind_elag, mcrps, 'k.') 
    # ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01) 
    ax.set_xlabel('Ensemble Size')
    ax.set_ylabel('CRPS')
    plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
    plt.savefig("CRPS.png", dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
        format='png', bbox_inches='tight', pad_inches=0.1)

    plt.close(fig1)
    print(" CRPS Done")

    for i in range(0,len(qlev)): 
        # Brier Score
        plt.close('all')
        fig1, ax = plt.subplots(figsize=(6.5,6))
        ax.plot(ind_elag, briers[:,i], color='k', lw=2, zorder=3) 
        ax.plot(ind_elag, briers[:,i], 'k.') 
        # ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01) 
        ax.set_xlabel('Ensemble Size')
        ax.set_ylabel('Brier Score')
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
        plt.savefig("BrierScore_"+repr(qlev[i])+".png", dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
            format='png', bbox_inches='tight', pad_inches=0.1)

        plt.close(fig1)

        # ROC AUC
        plt.close('all')
        fig1, ax = plt.subplots(figsize=(6.5,6))
        ax.plot(ind_elag, roc_auc[:,i], color='k', lw=2, zorder=3) 
        ax.plot(ind_elag, roc_auc[:,i], 'k.') 
        # ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01) 
        ax.set_xlabel('Ensemble Size')
        ax.set_ylabel('ROC AUC')
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
        plt.savefig("ROC_AUC_"+repr(qlev[i])+".png", dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
            format='png', bbox_inches='tight', pad_inches=0.1)

        plt.close(fig1)

        # ROC --------------
        plt.close('all')
        fig1, ax = plt.subplots(figsize=(6.5,6))
        ax.plot([0, 1], [0, 1], color='dimgray', lw=2, linestyle='--', alpha=0.8, zorder=1)
        for l in range(0,len(ind_elag)):
            ax.plot(roc_fpr, roc_tpr[l,i,:], color=lagcol[l], linestyle=lagmarl[l], lw=1, alpha=0.8, zorder=lagzo[l],label=repr(ind_elag[l])+'m') 

        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01) 
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right',fontsize=sl-3)
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
        plt.savefig("ROC_"+repr(qlev[i])+".png", dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
            format='png', bbox_inches='tight', pad_inches=0.1)

        plt.close(fig1)

        try:
            plt.close('all')
            fig1, ax = plt.subplots(figsize=(8,6))
            plt.axhline(y=0.,color='dimgrey',lw=2,zorder=1)
            for l in range(0,len(ind_elag)):
                if l!=3:
                    ax.plot(roc_fpr, gaussian_filter(roc_tpr[l,i,:]-roc_tpr[3,i,:],2), color=lagcol[l], linestyle=lagmarl[l], lw=2, alpha=0.8, zorder=lagzo[l],label=repr(ind_elag[l])+'m') 

            diff = roc_tpr[1::, i, 20:-1] - roc_tpr[3, i, 20:-1]
            mx = np.nanmax(np.abs(diff))

            if not np.isfinite(mx) or mx == 0:
                nm = 1e-6  # or any small fallback value you like
            else:
                nm = 0.95 * mx

            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-nm, nm)

            ax.legend(fontsize=sl-3)
            plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR: NewEnsemble - Default(31)')
            plt.tight_layout()

            plt.savefig("ROC_diff_"+repr(qlev[i])+".png", dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
                        format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close(fig1)
        except:
            print("  - Did not process ROD diff, "+repr(qlev[i]))
        else:
            print("  - Ok ROC diff, "+repr(qlev[i]))

        print(" Brier Score and ROC Done "+repr(qlev[i]))

    # ----------------------------------------

    print(" "); print("Analysis Completed"); print(" ")

