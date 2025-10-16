#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Probabilistic model. Hyperparameter optimization.

import matplotlib
# matplotlib.use('Agg')
import numpy as np
import pandas as pd
import sys
import gc  # garbage collector module
from scipy.stats import skew
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_curve, auc
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
    # opath="/work/noaa/marine/ricardo.campos/work/analysis/4postproc/output/ml"
    opath="/home/ricardo/cimas/analysis/4postproc/output/ml"

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

    # Training and Validation sets
    indtrain = [i for i, date in enumerate(cdate) if date < datetime(2023, 10, 1, 0, 0)]
    indval = [i for i, date in enumerate(cdate) if date >= datetime(2023, 10, 1, 0, 0)]
    indtest = [i for i, date in enumerate(cdate) if date >= datetime(2023, 10, 1, 0, 0)]

    # nbid = np.copy(bid); nbid[nbid==46005.] = 1; nbid[nbid==46006.] = 2; nbid[nbid==46066.] = 3

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

    # --- Input Space (fmod) ---
    # U10
    prob_u10_gefs_forecast, fmod_result_u10 = wprob.probforecast(nmax,gspws,spctl,u10_gefs_forecast,qlev_wnd)
    # Hs
    prob_hs_gefs_forecast, fmod_result_hs = wprob.probforecast(nmax,gspws,spctl,hs_gefs_forecast,qlev_hs)

    # Build array with features
    fnvars = np.array(['mean_u10','var_u10','skew_u10','pctl_80_u10','pctl_90_u10','pctl_95_u10','pctl_99_u10',
        'mean_hs','var_hs','skew_hs','pctl_80_hs','pctl_90_hs','pctl_95_hs','pctl_99_hs'])

    pctl = np.array([80,90,95,99])

    # U10
    for i in range(0,fmod_result_u10.shape[0]):
        if i==0:
            mip_u10 = np.array([np.mean(fmod_result_u10[i,:]), np.var(fmod_result_u10[i,:]),skew(fmod_result_u10[i,:])])
            mip_u10 = np.append(mip_u10,np.array(np.nanpercentile(fmod_result_u10[i,:],pctl)))
            mip_u10 = np.array([mip_u10])

        else:
            aux_mip_u10 = np.array([np.mean(fmod_result_u10[i,:]), np.var(fmod_result_u10[i,:]),skew(fmod_result_u10[i,:])])
            aux_mip_u10 = np.append(aux_mip_u10,np.array(np.nanpercentile(fmod_result_u10[i,:],pctl)))
            aux_mip_u10 = np.array([aux_mip_u10])

            mip_u10 = np.append(mip_u10,aux_mip_u10,axis=0)
            del aux_mip_u10

    # Hs
    for i in range(0,fmod_result_hs.shape[0]):
        if i==0:
            mip_hs = np.array([np.mean(fmod_result_hs[i,:]), np.var(fmod_result_hs[i,:]),skew(fmod_result_hs[i,:])])
            mip_hs = np.append(mip_hs,np.array(np.nanpercentile(fmod_result_hs[i,:],pctl)))
            mip_hs = np.array([mip_hs])

        else:
            aux_mip_hs = np.array([np.mean(fmod_result_hs[i,:]), np.var(fmod_result_hs[i,:]),skew(fmod_result_hs[i,:])])
            aux_mip_hs = np.append(aux_mip_hs,np.array(np.nanpercentile(fmod_result_hs[i,:],pctl)))
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

    # --- Normalization ---
    normet_input = 1 # arrays between 0 and 1.
    # -- Input --
    # npinpmin = np.array(np.min(mip,axis=0)-np.std(mip,axis=0))
    npinpmin = np.array(np.nanmin(mip,axis=0))
    # npinpmax = np.array(np.max(mip,axis=0)+np.std(mip,axis=0))
    npinpmax = np.array(np.nanmax(mip,axis=0))
    x1,mipnp1,mipnp2=dproc.normalization(mip,normet_input,npinpmin,npinpmax)

    # -- Output --
    v1 = np.copy(mop) # binary variables, (0,1), don't need normalization

    # --- Train/Test and Independent Validation set ---
    X_train = np.array(x1[indtrain,:])
    X_test = np.array(x1[indval,:])
    y_train = np.array(v1[indtrain])
    y_test = np.array(v1[indval])

    print(" Training records: "+repr(X_train.shape[0]))
    print(" Validation records: "+repr(X_test.shape[0])+"  "+repr(np.round((100*(len(indval)/len(indtrain))),2))+"%")


    #    ---- ML model run ----

    # ========== Random Forest =============
    model = RandomForestClassifier(max_depth=4,min_samples_leaf=60,min_samples_split=4,n_estimators=100,random_state=42, n_jobs=-1)
    model.fit(x1[indtrain,:], v1[indtrain,:])
    # Predict probabilities for the threshold on validation data
    prob_RF = np.array(model.predict_proba(x1[indtest,:]))[:,:,1].T
    # Convert probabilities to binary predictions
    plabels_RF = (prob_RF > 0.5).astype(int)
    # Evaluate the model on the validation set and print classification report
    print(classification_report(v1[indtest,:], plabels_RF))
    print(" ")
    del model


    # ========== XGBoost =============
    model = XGBClassifier(colsample_bytree=0.8, gamma=1.0, learning_rate=0.01, max_depth=4, min_child_weight=5, \
        n_estimators=200, reg_alpha=1.0, reg_lambda=1.0, subsample=0.8, random_state=42, n_jobs=-1)

    model.fit(x1[indtrain,:], v1[indtrain,:])

    # Predict probabilities for the threshold on validation data
    prob_XG = np.array(model.predict_proba(x1[indtest,:]))[:,:]
    # Convert probabilities to binary predictions
    plabels_XG = (prob_XG > 0.5).astype(int)
    # Evaluate the model on the validation set and print classification report
    print(classification_report(v1[indtest,:], plabels_XG))
    print(" ")
    del model


    # ========== MLP-NN =============
    model = MLPClassifier(early_stopping=True, solver='adam', validation_fraction=0.1, n_iter_no_change=100, random_state=42, \
        activation='tanh', alpha=0.1, hidden_layer_sizes=(200, 200), max_iter=100000000, \
        batch_size='auto', learning_rate='adaptive', learning_rate_init=10e-5, power_t=0.5, shuffle=True, \
        tol=10e-10, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, \
        beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.fit(x1[indtrain,:], v1[indtrain,:])

    # Predict probabilities for the threshold on validation data
    prob_NN = np.array(model.predict_proba(x1[indtest,:]))[:,:]
    # Convert probabilities to binary predictions
    plabels_NN = (prob_NN > 0.5).astype(int)
    # Evaluate the model on the validation set and print classification report
    print(classification_report(v1[indtest,:], plabels_NN))
    print(" ")
    del model


    # ================= Validation =================

    # Tables
    # GEFS
    ftag = opath+"/Teste_GEFS_"
    stat_results_gefs = wprob.prob_validation(nmax,spws,gspws,spctl,cdate[indtest],prob_u10_gefs_forecast[indtest,:],fmod_result_u10[indtest,:],prob_hs_gefs_forecast[indtest,:],fmod_result_hs[indtest,:],u10_obs_tmax[indtest],hs_obs_tmax[indtest],qlev_wnd,qlev_hs,plevels,ftag)
    # RF
    ftag = opath+"/Teste_RF_"
    stat_results_RF = wprob.prob_validation(nmax,spws,gspws,spctl,cdate[indtest],prob_RF[:,0:4],fmod_result_u10[indtest,:],prob_RF[:,4::],fmod_result_hs[indtest,:],u10_obs_tmax[indtest],hs_obs_tmax[indtest],qlev_wnd,qlev_hs,plevels,ftag)
    # XG
    ftag = opath+"/Teste_XG_"
    stat_results_XG = wprob.prob_validation(nmax,spws,gspws,spctl,cdate[indtest],prob_XG[:,0:4],fmod_result_u10[indtest,:],prob_XG[:,4::],fmod_result_hs[indtest,:],u10_obs_tmax[indtest],hs_obs_tmax[indtest],qlev_wnd,qlev_hs,plevels,ftag)
    # NN
    ftag = opath+"/Teste_NN_"
    stat_results_NN = wprob.prob_validation(nmax,spws,gspws,spctl,cdate[indtest],prob_NN[:,0:4],fmod_result_u10[indtest,:],prob_NN[:,4::],fmod_result_hs[indtest,:],u10_obs_tmax[indtest],hs_obs_tmax[indtest],qlev_wnd,qlev_hs,plevels,ftag)


    POD_hs_gefs = stat_results_gefs['ceval_gefs_hs'][:,:,0]
    POD_hs_RF = stat_results_RF['ceval_gefs_hs'][:,:,0]
    POD_hs_XG = stat_results_XG['ceval_gefs_hs'][:,:,0]
    POD_hs_NN = stat_results_NN['ceval_gefs_hs'][:,:,0]

    FAR_hs_gefs = stat_results_gefs['ceval_gefs_hs'][:,:,1]
    FAR_hs_RF = stat_results_RF['ceval_gefs_hs'][:,:,1]
    FAR_hs_XG = stat_results_XG['ceval_gefs_hs'][:,:,1]
    FAR_hs_NN = stat_results_NN['ceval_gefs_hs'][:,:,1]

    CSI_hs_gefs = stat_results_gefs['ceval_gefs_hs'][:,:,2]
    CSI_hs_RF = stat_results_RF['ceval_gefs_hs'][:,:,2]
    CSI_hs_XG = stat_results_XG['ceval_gefs_hs'][:,:,2]
    CSI_hs_NN = stat_results_NN['ceval_gefs_hs'][:,:,2]

    # --- Hs ---
    hd = ' '.join(np.array(np.round(qlev_hs,2)).astype('str'))
    # POD
    for i in range(0,len(plevels)-1):
        fname = opath+"/Eval_POD_Hs_"+str(plevels[i])+".txt"
        rPOD = np.array([POD_hs_gefs[:,i],POD_hs_RF[:,i],POD_hs_XG[:,i],POD_hs_NN[:,i]])
        result=np.round(rPOD,4)
        ifile = open(fname,'w')
        ifile.write(hd+' \n')
        np.savetxt(ifile,result.astype('str'),fmt="%s",delimiter='	') 
        ifile.close(); del ifile, fname, result

    # CSI
    for i in range(0,len(plevels)-1):
        fname = opath+"/Eval_CSI_Hs_"+str(plevels[i])+".txt"
        rCSI = np.array([CSI_hs_gefs[:,i],CSI_hs_RF[:,i],CSI_hs_XG[:,i],CSI_hs_NN[:,i]])
        result=np.round(rCSI,4)
        ifile = open(fname,'w')
        ifile.write(hd+' \n')
        np.savetxt(ifile,result.astype('str'),fmt="%s",delimiter='	') 
        ifile.close(); del ifile, fname, result

    # --- ROC Curve ---
    colors = np.array(['firebrick', 'green', 'blue', 'gold'])
    fnames = np.array(['GEFSv12','RandomForest','XGBoost','MLP_NN'])

    # U10
    for i in range(0,len(qlev_wnd)):

        ftag=opath+"/ProbEvents_U10_"+repr(np.round(qlev_wnd[i],2))

        true_binary = (u10_obs_tmax[indtest] > qlev_wnd[i]).astype(int)

        fpr_GEFS, tpr_GEFS, _ = roc_curve(true_binary, prob_u10_gefs_forecast[indtest,i])
        auc_GEFS = str(np.round(auc(fpr_GEFS, tpr_GEFS),2))
        fpr_RF, tpr_RF, _ = roc_curve(true_binary, prob_RF[:,0:4][:,i])
        auc_RF = str(np.round(auc(fpr_RF, tpr_RF),2))
        fpr_XG, tpr_XG, _ = roc_curve(true_binary, prob_XG[:,0:4][:,i])
        auc_XG = str(np.round(auc(fpr_XG, tpr_XG),2))
        fpr_NN, tpr_NN, _ = roc_curve(true_binary, prob_NN[:,0:4][:,i])
        auc_NN = str(np.round(auc(fpr_NN, tpr_NN),2))

        # Plot the ROC curve
        fig1, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], color='dimgray', lw=3, linestyle='--', alpha=0.8, zorder=1)

        ax.plot(fpr_GEFS, tpr_GEFS, color=colors[0], lw=3, marker='.', label=fnames[0]+" ("+auc_GEFS+")", alpha=0.8, zorder=3)
        ax.plot(fpr_RF, tpr_RF, color=colors[1], lw=3, marker='.', label=fnames[1]+" ("+auc_RF+")", alpha=0.8, zorder=3)
        ax.plot(fpr_XG, tpr_XG, color=colors[2], lw=3, marker='.', label=fnames[2]+" ("+auc_XG+")", alpha=0.8, zorder=3)
        ax.plot(fpr_NN, tpr_NN, color=colors[3], lw=3, marker='.', label=fnames[3]+" ("+auc_NN+")", alpha=0.8, zorder=3)

        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01) 
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right',fontsize=sl-2)
        plt.title('ROC Curve')
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)

        plt.savefig(ftag+"_ROC.png", dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
            format='png', bbox_inches='tight', pad_inches=0.1)

        plt.close(fig1)

    # Hs
    for i in range(0,len(qlev_hs)):

        ftag=opath+"/ProbEvents_Hs_"+repr(np.round(qlev_hs[i],2))

        true_binary = (hs_obs_tmax[indtest] > qlev_hs[i]).astype(int)

        fpr_GEFS, tpr_GEFS, _ = roc_curve(true_binary, prob_hs_gefs_forecast[indtest,i])
        auc_GEFS = str(np.round(auc(fpr_GEFS, tpr_GEFS),2))
        fpr_RF, tpr_RF, _ = roc_curve(true_binary, prob_RF[:,4::][:,i])
        auc_RF = str(np.round(auc(fpr_RF, tpr_RF),2))
        fpr_XG, tpr_XG, _ = roc_curve(true_binary, prob_XG[:,4::][:,i])
        auc_XG = str(np.round(auc(fpr_XG, tpr_XG),2))
        fpr_NN, tpr_NN, _ = roc_curve(true_binary, prob_NN[:,4::][:,i])
        auc_NN = str(np.round(auc(fpr_NN, tpr_NN),2))

        # Plot the ROC curve
        fig1, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], color='dimgray', lw=3, linestyle='--', alpha=0.8, zorder=1)

        ax.plot(fpr_GEFS, tpr_GEFS, color=colors[0], lw=3, marker='.', label=fnames[0]+" ("+auc_GEFS+")", alpha=0.8, zorder=3)
        ax.plot(fpr_RF, tpr_RF, color=colors[1], lw=3, marker='.', label=fnames[1]+" ("+auc_RF+")", alpha=0.8, zorder=3)
        ax.plot(fpr_XG, tpr_XG, color=colors[2], lw=3, marker='.', label=fnames[2]+" ("+auc_XG+")", alpha=0.8, zorder=3)
        ax.plot(fpr_NN, tpr_NN, color=colors[3], lw=3, marker='.', label=fnames[3]+" ("+auc_NN+")", alpha=0.8, zorder=3)

        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01) 
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right',fontsize=sl-2)
        plt.title('ROC Curve')
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)

        plt.savefig(ftag+"_ROC.png", dpi=200, facecolor='w', edgecolor='w', orientation='portrait',
            format='png', bbox_inches='tight', pad_inches=0.1)

        plt.close(fig1)


