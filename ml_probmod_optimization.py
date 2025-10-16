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

    sel_model = int(sys.argv[1])

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


    # Random initialization
    rst=np.arange(6,43,6).astype('int')

    # Score metric for hyperparameter optimization
    # scoring_metric = 'neg_log_loss'
    scoring_metrics = np.array(['roc_auc','accuracy','neg_log_loss'])


    # ========== Random Forest =============
    if sel_model == 1:      

        # Parameters space
        max_depth = np.array([4, 5, 6, 8, 10, 20])
        min_samples_leaf = np.array([20, 40, 50, 60, 80, 100]).astype('int')
        min_samples_split  = np.array([4, 5, 6, 8, 10]).astype('int')
        n_estimators = np.array([10, 50, 100, 200, 300, 400]).astype('int')

        # Define the parameter grid to search
        param_grid = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        }

        best_rf = np.zeros((len(scoring_metrics),len(rst),len(param_grid)),'f')*np.nan

        for i in range(0,len(scoring_metrics)):
            for j in range(0,len(rst)):

                # Define the RandomForestClassifier
                model = RandomForestClassifier(random_state=rst[j])
                # Cross Validation
                inner_cv = KFold(n_splits=3, shuffle=True, random_state=rst[j])
                # Create the GridSearchCV object
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring_metrics[i], cv=inner_cv, n_jobs=-1)
                # Fit the grid search to the data
                grid_search.fit(x1[indtrain,:], v1[indtrain,:])

                best_rf[i,j,0] = grid_search.best_params_['max_depth']
                best_rf[i,j,1] = grid_search.best_params_['min_samples_leaf']
                best_rf[i,j,2] = grid_search.best_params_['min_samples_split']
                best_rf[i,j,3] = grid_search.best_params_['n_estimators']

                # Print the best parameters and the corresponding model
                print("RF Best Parameters ("+scoring_metrics[i]+") ramdom Init "+repr(rst[j])+" :", grid_search.best_params_)
                best_model = grid_search.best_estimator_

                # Predict probabilities for the threshold on validation data
                probabilities = np.array(best_model.predict_proba(x1[indval,:]))[:,:,1]
                # Convert probabilities to binary predictions
                predicted_labels = (probabilities > 0.5).astype(int)

                # Evaluate the model on the validation set and print classification report
                print(classification_report(v1[indval,:], predicted_labels.T))
                print(" ")

                del probabilities, predicted_labels, grid_search, best_model, inner_cv, model
                gc.collect()

    # scoring_metric('accuracy') = 'max_depth': 5, 'min_samples_leaf': 40, 'min_samples_split': 4, 'n_estimators': 100
    # scoring_metric('roc_auc') = 'max_depth': 5, 'min_samples_leaf': 100, 'min_samples_split': 2, 'n_estimators': 100
    #  scoring_metric('roc_auc') = 'max_depth': 4, 'min_samples_leaf': 60, 'min_samples_split': 4, 'n_estimators': 200

    # If class balance is a concern and you want a metric that is less affected by imbalanced datasets, ROC AUC may be a more informative metric

    # ========== XGBoost =============
    if sel_model == 2:

        # Parameters space
        learning_rate = np.array([0.01, 0.1]).astype('float')
        n_estimators = np.array([10, 50, 100, 200, 300]).astype('int')
        max_depth = np.array([4, 5, 6, 8, 10])
        min_child_weight = np.array([3,5,10]).astype('int')
        subsample = np.array([0.8, 0.9, 1.0])
        colsample_bytree = np.array([0.8, 0.9, 1.0])
        gamma = np.array([0.1, 0.5, 1.0, 2.0])
        reg_alpha = np.array([0.1, 0.5, 1.0, 2.0])
        reg_lambda = np.array([0.1, 0.5, 1.0, 2.0])

        # Define the parameter grid to search
        param_grid = {
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda
        }

        best_xg = np.zeros((len(scoring_metrics),len(param_grid)),'f')*np.nan

        for i in range(0,len(scoring_metrics)):

            # Define the XGBClassifier
            # model = XGBClassifier(alpha=alpha, reg_lambda=reg_lambda, gamma=gamma, random_state=42)
            model = XGBClassifier(random_state=42)
            # Cross Validation
            inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
            # Create the GridSearchCV object
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring_metrics[i], cv=inner_cv, n_jobs=-1)
            # Fit the grid search to the data
            grid_search.fit(x1[indtrain,:], v1[indtrain,:])

            best_xg[i,0] = grid_search.best_params_['learning_rate']
            best_xg[i,1] = grid_search.best_params_['n_estimators']
            best_xg[i,2] = grid_search.best_params_['max_depth']
            best_xg[i,3] = grid_search.best_params_['min_child_weight']
            best_xg[i,4] = grid_search.best_params_['subsample']
            best_xg[i,5] = grid_search.best_params_['colsample_bytree']
            best_xg[i,6] = grid_search.best_params_['gamma']
            best_xg[i,7] = grid_search.best_params_['reg_alpha']
            best_xg[i,8] = grid_search.best_params_['reg_lambda']

            # Print the best parameters and the corresponding model
            print("XG Best Parameters ("+scoring_metrics[i]+") :", grid_search.best_params_)
            best_model = grid_search.best_estimator_

            # Predict probabilities for the threshold on validation data
            probabilities = np.array(best_model.predict_proba(x1[indval,:]))[:,:]
            # Convert probabilities to binary predictions
            predicted_labels = (probabilities > 0.5).astype(int)

            # Evaluate the model on the validation set and print classification report
            print(classification_report(v1[indval,:], predicted_labels))
            print(" ")

            del probabilities, predicted_labels, grid_search, best_model, inner_cv, model


    # Best Parameters (roc_auc) : {'colsample_bytree': 0.8, 'gamma': 1.0, 'learning_rate': 0.01, 'max_depth': 4, 'min_child_weight': 5, 'n_estimators': 200, 'reg_alpha': 1.0, 'reg_lambda': 1.0, 'subsample': 0.8}


    # ========== MLP-NN =============
    if sel_model == 3: 

        # For binary classification tasks, it uses the binary cross-entropy loss by default.

        # Parameters space
        hidden_layer_sizes = [(2,), (10,), (20,), (50,), (100,), (10, 10), (20, 20), (50, 50), (100, 100), (200, 200), (500, 500)]
        activation = ['tanh', 'relu']
        max_iter = [1000, 100000, 10e10]
        alpha = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0]

        param_grid = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'max_iter': max_iter,
            'alpha': alpha
            }

        for i in range(0,len(scoring_metrics)):
            for j in range(0,len(rst)):

                # model = MLPClassifier(early_stopping=True, solver='adam', validation_fraction=0.1, n_iter_no_change=10, random_state=rst[j])

                model = MLPClassifier(early_stopping=True, solver='adam', validation_fraction=0.1, n_iter_no_change=50, random_state=rst[j], \
                    batch_size='auto', learning_rate='adaptive', learning_rate_init=10e-5, power_t=0.5, shuffle=True, \
		       		tol=10e-10, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, \
		       		beta_1=0.9, beta_2=0.999, epsilon=1e-08)

                # Cross Validation
                inner_cv = KFold(n_splits=3, shuffle=True, random_state=rst[j])
                # Create the GridSearchCV object
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring_metrics[i], cv=inner_cv, n_jobs=-1)
                # Fit the grid search to the data
                grid_search.fit(x1[indtrain,:], v1[indtrain,:])

                # Print the best parameters and the corresponding model
                print("NN Best Parameters ("+scoring_metrics[i]+") ramdom Init "+repr(rst[j])+" :", grid_search.best_params_)
                best_model = grid_search.best_estimator_

                # Predict probabilities for the threshold on validation data
                probabilities = np.array(best_model.predict_proba(x1[indval,:]))[:,:]
                # Convert probabilities to binary predictions
                predicted_labels = (probabilities > 0.5).astype(int)

                # Evaluate the model on the validation set and print classification report
                print(classification_report(v1[indval,:], predicted_labels))
                print(" ")

                del probabilities, predicted_labels, grid_search, best_model, inner_cv, model


    # Best Parameters (roc_auc) : {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (100, 100), 'max_iter': 200}

    # Best Parameters (roc_auc) ramdom Init 12 : {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (200, 200), 'max_iter': 1000}
    # Best Parameters (roc_auc) ramdom Init 36 : {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50,), 'max_iter': 1000}
    # Best Parameters (roc_auc) ramdom Init 42 : {'activation': 'relu', 'alpha': 2.0, 'hidden_layer_sizes': (10, 10), 'max_iter': 1000}
    #   Best Parameters (roc_auc) ramdom Init 9 : {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': (200, 200), 'max_iter': 1000}



