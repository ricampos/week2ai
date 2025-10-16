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
from scipy.stats import skew
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import properscoring as ps
from datetime import datetime
# from pysteps import verification
import yaml

import seaborn as sns
import ppscore as pps

from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from scipy.ndimage.filters import gaussian_filter
from scipy.stats import gaussian_kde, linregress
from scipy.stats import gaussian_kde

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
    opath="/work/noaa/marine/ricardo.campos/work/analysis/4postproc/output/ml" # on Orion

    # ---- Statistical parameters (exaclty the same as the operational config file) -----
    qlev_hs = np.array([ 4.,  6.,  9.]).astype('float')
    qlev_wnd = np.array([28.0, 34.0, 41.0, 48.0]).astype('float')/1.94
    plevels = np.array([0.15, 0.5, 0.65, 0.8, 1.0])

    # Input Argument. Optimized Parameters
    spws = float(2.5)
    nmax = int(2)
    spctl = float(91)

    print(" ")
    print(" === Machine Learning === ")
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


    # --- Ground Truth (Target) ---
    #  min duration of event (consider 6-hourly resolution)
    u10_obs_tmax = wprob.nmaxsel(u10_obs,1)
    hs_obs_tmax = wprob.nmaxsel(hs_obs,1)
    #  binary array (event, not-event)
    true_binary_u10 = np.zeros((len(qlev_wnd),u10_gefs_forecast.shape[0]),'f')*np.nan
    true_binary_hs = np.zeros((len(qlev_hs),hs_gefs_forecast.shape[0]),'f')*np.nan
    for i in range(0,len(qlev_wnd)):
        true_binary_u10[i,:] = (u10_obs_tmax > qlev_wnd[i]).astype(int)

    for i in range(0,len(qlev_hs)):
        true_binary_hs[i,:] = (hs_obs_tmax > qlev_hs[i]).astype(int)

    # ----------------------------

    # --- Probabilities and Array ---s
    # U10
    prob_u10_gefs_forecast, fmod_result_u10 = wprob.probforecast(nmax,gspws,spctl,u10_gefs_forecast,qlev_wnd)
    # Hs
    prob_hs_gefs_forecast, fmod_result_hs = wprob.probforecast(nmax,gspws,spctl,hs_gefs_forecast,qlev_hs)
    # ----------------------------

    # ====== PDF =========
    plt.close('all')
    # Hs 
    px=np.linspace(np.nanmin([np.nanmin(hs_obs_tmax),np.nanmin(fmod_result_hs)]),np.nanmax(hs_obs_tmax)+0.5,100)
    for i in range(0,fmod_result_hs.shape[0],10):

        model = np.sort(fmod_result_hs[i,:])
        obs = float(hs_obs_tmax[i])

        # plot
        plt.close('all')
        fig1 = plt.figure(1,figsize=(5,5)); ax = fig1.add_subplot(111)
        # PDF Model
        dx = gaussian_kde(model)
        ax.fill_between(px, 0., gaussian_filter(dx(px), 1.), color='silver', alpha=0.7,zorder=1)
        ax.plot(px,gaussian_filter(dx(px), 1.),color='silver',linestyle='-',linewidth=0.5,alpha=0.7,zorder=2)
        ax.plot(px,gaussian_filter(dx(px), 1.),color='black',marker='.',linestyle='-',label='Model', linewidth=1.,alpha=0.7,zorder=3)
        # Obs
        ax.axvline(x=obs, color='firebrick', linestyle='--', linewidth=2., label='Obs')
        ax.text(obs, 0, 'Obs', color='firebrick', weight='bold', size=12,zorder=4)
        plt.legend(fontsize=sl-2)
        # qlevs
        for j in range(0,len(qlev_hs)):
            ax.axvline(x=qlev_hs[j], color='green', linestyle='-', linewidth=1.,alpha=0.6, label='Thresholds',zorder=2)
            ax.text(qlev_hs[j], 0, str(np.round(qlev_hs[j],1)).zfill(2), color='darkgreen', size=9, alpha=0.8,zorder=3)

        plt.grid(c='grey', ls=':', alpha=0.5,zorder=1)
        plt.locator_params(axis='y', nbins=7) ; plt.locator_params(axis='x', nbins=7) 
        ax.set_ylabel("Probability Density")
        ax.set_xlabel("Significant Wave Height (m)")

        plt.tight_layout(); plt.axis('tight'); plt.ylim(ymin = -0.005)
        plt.xlim(xmin = px.min(), xmax = px.max())

        plt.savefig(opath+"/PDF_Hs_"+repr(i)+".png", dpi=200, facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig1); del fig1, ax,

    # U10 
    px=np.linspace(np.nanmin([np.nanmin(u10_obs_tmax),np.nanmin(fmod_result_u10)]),np.nanmax(u10_obs_tmax)+2.,100)
    for i in range(0,fmod_result_u10.shape[0],10):

        model = np.sort(fmod_result_u10[i,:])
        obs = float(u10_obs_tmax[i])

        # plot
        plt.close('all')
        fig1 = plt.figure(1,figsize=(5,5)); ax = fig1.add_subplot(111)
        # PDF Model
        dx = gaussian_kde(model)
        ax.fill_between(px, 0., gaussian_filter(dx(px), 1.), color='silver', alpha=0.7,zorder=1)
        ax.plot(px,gaussian_filter(dx(px), 1.),color='silver',linestyle='-',linewidth=0.5,alpha=0.7,zorder=2)
        ax.plot(px,gaussian_filter(dx(px), 1.),color='black',marker='.',linestyle='-',label='Model', linewidth=1.,alpha=0.7,zorder=3)
        # Obs
        ax.axvline(x=obs, color='firebrick', linestyle='--', linewidth=2., label='Obs')
        ax.text(obs, 0, 'Obs', color='firebrick', weight='bold', size=12,zorder=4)
        plt.legend(fontsize=sl-2)
        # qlevs
        for j in range(0,len(qlev_wnd)):
            ax.axvline(x=qlev_wnd[j], color='green', linestyle='-', linewidth=1.,alpha=0.6, label='Thresholds',zorder=2)
            ax.text(qlev_wnd[j], 0, str(np.round(qlev_wnd[j],1)).zfill(2), color='darkgreen', size=9, alpha=0.8,zorder=3)

        plt.grid(c='grey', ls=':', alpha=0.5,zorder=1)
        plt.locator_params(axis='y', nbins=7) ; plt.locator_params(axis='x', nbins=7) 
        ax.set_ylabel("Probability Density")
        ax.set_xlabel("Wind Speed (m/s)")

        plt.tight_layout(); plt.axis('tight'); plt.ylim(ymin = -0.005)
        plt.xlim(xmin = px.min(), xmax = px.max())

        plt.savefig(opath+"/PDF_U10_"+repr(i)+".png", dpi=200, facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig1); del fig1, ax,

    # ================


    # =========== FEATURE SELECTION ===========

    # --- Input Space (fmod) ---

    pctl = np.array([80,90,95,99])

    # U10
    for i in range(0,fmod_result_u10.shape[0]):
        if i==0:
            mip_u10 = np.array([np.mean(fmod_result_u10[i,:]), np.var(fmod_result_u10[i,:]),skew(fmod_result_u10[i,:])])
            mip_u10 = np.append(mip_u10,np.array(np.nanpercentile(fmod_result_u10[i,:],pctl)))
            mip_u10 = np.append(mip_u10,np.array(np.nanmax(fmod_result_u10[i,:])))
            mip_u10 = np.array([mip_u10])

        else:
            aux_mip_u10 = np.array([np.mean(fmod_result_u10[i,:]), np.var(fmod_result_u10[i,:]),skew(fmod_result_u10[i,:])])
            aux_mip_u10 = np.append(aux_mip_u10,np.array(np.nanpercentile(fmod_result_u10[i,:],pctl)))
            aux_mip_u10 = np.append(aux_mip_u10,np.array(np.nanmax(fmod_result_u10[i,:])))
            aux_mip_u10 = np.array([aux_mip_u10])

            mip_u10 = np.append(mip_u10,aux_mip_u10,axis=0)
            del aux_mip_u10

    # Hs
    for i in range(0,fmod_result_hs.shape[0]):
        if i==0:
            mip_hs = np.array([np.mean(fmod_result_hs[i,:]), np.var(fmod_result_hs[i,:]),skew(fmod_result_hs[i,:])])
            mip_hs = np.append(mip_hs,np.array(np.nanpercentile(fmod_result_hs[i,:],pctl)))
            mip_hs = np.append(mip_hs,np.array(np.nanmax(fmod_result_hs[i,:])))
            mip_hs = np.array([mip_hs])

        else:
            aux_mip_hs = np.array([np.mean(fmod_result_hs[i,:]), np.var(fmod_result_hs[i,:]),skew(fmod_result_hs[i,:])])
            aux_mip_hs = np.append(aux_mip_hs,np.array(np.nanpercentile(fmod_result_hs[i,:],pctl)))
            aux_mip_hs = np.append(aux_mip_hs,np.array(np.nanmax(fmod_result_hs[i,:])))
            aux_mip_hs = np.array([aux_mip_hs])

            mip_hs = np.append(mip_hs,aux_mip_hs,axis=0)
            del aux_mip_hs

    # --- Input array ---
    mip = np.c_[mip_u10, mip_hs] 
    # --- Output Space ---
    # Continuous
    # mop = np.c_[u10_obs_tmax, hs_obs_tmax]
    # Binary Categorical
    mop = np.c_[true_binary_u10.T, true_binary_hs .T]


    fnvars = np.array(['mean_u10','var_u10','skew_u10','pctl_80_u10','pctl_90_u10','pctl_95_u10','pctl_99_u10','max_u10',
        'mean_hs','var_hs','skew_hs','pctl_80_hs','pctl_90_hs','pctl_95_hs','pctl_99_hs','max_hs','Obs_max_u10','Obs_max_hs'])

    # pandas dataframe
    df = pd.DataFrame(np.c_[ np.c_[mip,np.array([u10_obs_tmax]).T], np.array([hs_obs_tmax]).T] , columns = fnvars)

    # ------------------------------------

    # --- Cross-correlation (Linear, Pearson) ---

    plt.close('all')
    fig1 = plt.figure(1,figsize=(7,6))
    lincorr = df.corr()
    ax = sns.heatmap(
        lincorr, 
        vmin=-0.5, vmax=1.0, center=0,
        cmap=sns.diverging_palette(220, 20, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );

    plt.tight_layout();plt.axis('tight')
    plt.grid(c='k', ls=':', alpha=0.1)
    plt.savefig(opath+"/CrossCorrelPearson.png", dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
    lincorr.to_csv(opath+"/CrossCorrelPearson.csv", sep='\t')

    bcolors = ['navy','navy']
    for i in range(0,len(fnvars)-2):
        bcolors = np.append(bcolors,'dimgray')

    plt.close('all') 
    i=-2
    fig, ax = plt.subplots(figsize=(9,5)) 
    bars = ax.bar(np.flip(fnvars), np.flip(lincorr.values[i,:]), color=bcolors, edgecolor='black')
    # Rotate and align the x-axis labels diagonally
    plt.grid(c='grey', ls=':', alpha=0.5,zorder=1)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Correlation to '+fnvars[i])
    plt.tight_layout(); plt.axis('tight')
    plt.savefig(opath+"/CorrelPearson_to_"+fnvars[i]+".png", dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)

    i=-1
    fig, ax = plt.subplots(figsize=(9,5)) 
    bars = ax.bar(np.flip(fnvars), np.flip(lincorr.values[i,:]), color=bcolors, edgecolor='black')
    # Rotate and align the x-axis labels diagonally
    plt.grid(c='grey', ls=':', alpha=0.5,zorder=1)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Correlation to '+fnvars[i])
    plt.tight_layout(); plt.axis('tight')
    plt.savefig(opath+"/CorrelPearson_to_"+fnvars[i]+".png", dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
    # ----------------------------------------------------------


    #  --- Correlation Matrix PPS score (nonlinear) --- 
    fig2 = plt.figure(2,figsize=(7,6))
    nlcorr = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
    ax = sns.heatmap(
        nlcorr, 
        vmin=-0.2, vmax=1.0, center=0,
        cmap=sns.diverging_palette(220, 20, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );

    plt.tight_layout();plt.axis('tight')
    plt.grid(c='k', ls=':', alpha=0.1)
    plt.savefig(opath+"/ppscore.png", dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
    nlcorr.to_csv(opath+"/ppscore.csv", sep='\t')

    plt.close('all') 
    i=-2
    fig, ax = plt.subplots(figsize=(9,5)) 
    bars = ax.bar(np.flip(fnvars), np.flip(nlcorr.values[i,:]), color=bcolors, edgecolor='black')
    # Rotate and align the x-axis labels diagonally
    plt.grid(c='grey', ls=':', alpha=0.5,zorder=1)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('ppscore to '+fnvars[i])
    plt.tight_layout(); plt.axis('tight')
    plt.savefig(opath+"/ppscore_to_"+fnvars[i]+".png", dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)

    i=-1
    fig, ax = plt.subplots(figsize=(9,5)) 
    bars = ax.bar(np.flip(fnvars), np.flip(nlcorr.values[i,:]), color=bcolors, edgecolor='black')
    # Rotate and align the x-axis labels diagonally
    plt.grid(c='grey', ls=':', alpha=0.5,zorder=1)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('ppscore to '+fnvars[i])
    plt.tight_layout(); plt.axis('tight')
    plt.savefig(opath+"/ppscore_to_"+fnvars[i]+".png", dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
    # ----------------------------------------------------------


    # ==== Normalization ====
    normet_input = 1 # arrays between 0 and 1.
    # -- Input --
    # npinpmin = np.array(np.min(mip,axis=0)-np.std(mip,axis=0))
    npinpmin = np.array(np.nanmin(mip,axis=0))
    # npinpmax = np.array(np.max(mip,axis=0)+np.std(mip,axis=0))
    npinpmax = np.array(np.nanmax(mip,axis=0))
    x1,mipnp1,mipnp2=dproc.normalization(mip,normet_input,npinpmin,npinpmax)

    # -- Output --
    v1 = np.copy(mop)


    # === Recursive feature elimination (RFECV) ===

    fnvars = np.array(['mean_u10','var_u10','skew_u10','pctl_80_u10','pctl_90_u10','pctl_95_u10','pctl_99_u10','max_u10',
        'mean_hs','var_hs','skew_hs','pctl_80_hs','pctl_90_hs','pctl_95_hs','pctl_99_hs','max_hs'])

    # ----- Random Forest -----
    top_features_RandomForest = np.zeros((len(fnvars)),'f').astype('int')
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    for i in range(1,len(fnvars)):
        rfe = RFE(estimator=model, n_features_to_select=int(i))
        # Fit RFE to the training data
        rfe.fit(x1, v1)
        # Get the selected features
        aux = np.array(np.where(rfe.support_== True)[0]).astype('int')
        top_features_RandomForest[aux] = top_features_RandomForest[aux]+1
        print(" Random Forest RFECV "+repr(i))
        del rfe, aux

    top_features_RandomForest = top_features_RandomForest / np.nanmax(top_features_RandomForest)

    fig, ax = plt.subplots(figsize=(9,6))
    bars = ax.bar(fnvars, top_features_RandomForest, color='dimgray', edgecolor='black')
    # Rotate and align the x-axis labels diagonally
    plt.grid(c='grey', ls=':', alpha=0.5,zorder=1)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Random Forest Classifier RFE, Feature Importance')
    plt.tight_layout(); plt.axis('tight')
    plt.savefig(opath+"/RandomForest_RFE"+".png", dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)


    # ----- XGBoost Classifier -----
    top_features_XGBoost = np.zeros((len(fnvars)),'f').astype('int')
    model = XGBClassifier(random_state=42)
    for i in range(1,len(fnvars)):
        rfe = RFE(estimator=model, n_features_to_select=int(i))
        # Fit RFE to the training data
        rfe.fit(x1, v1)
        # Get the selected features
        aux = np.array(np.where(rfe.support_== True)[0]).astype('int')
        top_features_XGBoost[aux] = top_features_XGBoost[aux]+1
        print(" XGBoost Classifier RFECV "+repr(i))
        del rfe, aux

    top_features_XGBoost = top_features_XGBoost / np.nanmax(top_features_XGBoost)

    fig, ax = plt.subplots(figsize=(9,6))
    bars = ax.bar(fnvars, top_features_XGBoost, color='dimgray', edgecolor='black')
    # Rotate and align the x-axis labels diagonally
    plt.grid(c='grey', ls=':', alpha=0.5,zorder=1)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('XGBoost Classifier RFE, Feature Importance')
    plt.tight_layout(); plt.axis('tight')
    plt.savefig(opath+"/XGBoost_RFE"+".png", dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)


    # ----- MLP-NN Classifier -----
    top_features_MLPNN = np.zeros((len(fnvars)),'f').astype('int')

    rst=np.arange(3,45,3).astype('int') 
    for j in range(0,len(rst)):
        model = MLPClassifier(hidden_layer_sizes=(10,10),  activation='tanh', solver='adam', alpha=0.0001, batch_size='auto', \
            learning_rate='adaptive', learning_rate_init=10e-5, power_t=0.5, max_iter=int(10e10), shuffle=True, \
            random_state=rst[j], tol=10e-10, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, \
            early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Fit the model
        model.fit(x1, v1)
        # Get feature importances from the model
        feature_importances = np.abs(model.coefs_[0])
        # Sum the absolute values of coefficients along the rows
        feature_scores = np.sum(feature_importances, axis=1)

        for i in range(1,len(fnvars)):
            aux = np.array(np.flip(np.argsort(feature_scores))[0:i]).astype('int')
            top_features_MLPNN[aux] = top_features_MLPNN[aux]+1
            print(" MLPNN Classifier RFECV "+repr(i))
            del aux

        print(repr(j))

    top_features_MLPNN = top_features_MLPNN / np.nanmax(top_features_MLPNN)

    fig, ax = plt.subplots(figsize=(9,6))
    bars = ax.bar( fnvars, top_features_MLPNN, color='dimgray', edgecolor='black')
    # Rotate and align the x-axis labels diagonally
    plt.grid(c='grey', ls=':', alpha=0.5,zorder=1)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('MLP-NN Classifier RFE, Feature Importance')
    plt.tight_layout(); plt.axis('tight')
    plt.savefig(opath+"/MLPNN_RFE"+".png", dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)


    # Summary involving all the models
    top_features = top_features_RandomForest + top_features_XGBoost + top_features_MLPNN
    top_features = top_features / np.nanmax(top_features)

    fig, ax = plt.subplots(figsize=(9,6))
    bars = ax.bar( fnvars, top_features_MLPNN, color='dimgray', edgecolor='black')
    # Rotate and align the x-axis labels diagonally
    plt.grid(c='grey', ls=':', alpha=0.5,zorder=1)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('RFE, Feature Importance')
    plt.tight_layout(); plt.axis('tight')
    plt.savefig(opath+"/TOTAL_RFE"+".png", dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)

    plt.close('all')

