import sys
from datetime import datetime

import joblib
import numpy as np
from scipy.stats import skew

from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import wprob

#------------------------------------------------------------------------
# -- initialize and get random forest
rf = RandomForestClassifier(max_depth=4, min_samples_leaf=60, \
        min_samples_split=4, n_estimators=100, random_state=42, n_jobs=-1)

#rf = joblib.load('random_forest.joblib')

# -- xgboost
xgb = XGBClassifier(colsample_bytree=0.8, gamma=1.0, learning_rate=0.01, \
        max_depth=4, min_child_weight=5, n_estimators=200, reg_alpha=1.0, \
        reg_lambda=1.0, subsample=0.8, random_state=42, n_jobs=-1)
#xgb = joblib.load('xgb.joblib')


# -- MLP classifier
mlp = MLPClassifier(early_stopping=True, solver='adam',  \
        validation_fraction=0.1, n_iter_no_change=100, random_state=42,   \
        activation='tanh', alpha=0.1, hidden_layer_sizes=(200, 200),  \
        max_iter=100000000,  batch_size='auto', learning_rate='adaptive',  \
        learning_rate_init=10e-5, power_t=0.5, shuffle=True,  tol=10e-10,  \
        verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,   \
        beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#mlp = joblib.load('mlp.joblib')


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# Get training and validation data:
if __name__ == "__main__":

    #User-sensitive
    #opath="/home/ricardo/cimas/analysis/4postproc/output/ml"
    opath = "/scratch3/AOML/aoml-phod/Ricardo.Campos/data/week2runs"
    # list of netcdf files generated with buildfuzzydataset.py (GEFS, GDAS, and NDBC buoy)
    # ls -d $PWD/*.nc > list.txt &
    wlist=np.atleast_1d(np.loadtxt('../list.txt',dtype=str))

    # ------------- should be relatively unchanged between users
    # select one point
    stations = np.array(['46005','46006','46066']).astype('str')
    # Forecast Lead Time (Day) and intervall
    ltime1=7
    ltime2=14

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
    lstw=int(len(wlist))
    lplev=int(len(plevels)-1)

    # week 2 array
    for i in range(0,len(stations)):

        # cdate,ctime,ensm,latm,lonm,indlat,indlon,u10_ndbc,hs_ndbc,
        # u10_gefs_hindcast,hs_gefs_hindcast,au10_gefs_forecast,ahs_gefs_forecast,
        # indt = wprob.read_data(wlist,stations[i],ltime1,ltime2)
        #RG: user-sensitive
        ENSDATA = wprob.read_data(wlist,stations[i],ltime1,ltime2)

        if i==0:
            gspws = int(np.floor(spws/np.diff(ENSDATA['latm']).mean())/2)
            cdate = np.array(ENSDATA['cdate'])
            bid=np.zeros((len(ENSDATA['ctime'])),'f')+float(stations[i])
            # ens forecast
            u10_gefs_forecast = np.array(ENSDATA['u10_gefs_forecast'])
            hs_gefs_forecast  = np.array(ENSDATA['hs_gefs_forecast'])
            # Ground truth
            u10_obs = np.nanmean([ENSDATA['u10_ndbc'], \
                    np.nanmean(ENSDATA['u10_gefs_hindcast'][:,:,:,ENSDATA['indlat'], \
                    ENSDATA['indlon']],axis=2)],axis=0)
            hs_obs = np.nanmean([ENSDATA['hs_ndbc'], \
                    np.nanmean(ENSDATA['hs_gefs_hindcast'][:,:,:,ENSDATA['indlat'], \
                    ENSDATA['indlon']],axis=2)],axis=0)
        else:
            cdate=np.append(cdate,ENSDATA['cdate'])
            bid=np.append(bid,np.zeros((len(ENSDATA['ctime'])),'f')+float(stations[i]))
            # ens forecast
            u10_gefs_forecast = np.append(u10_gefs_forecast,ENSDATA['u10_gefs_forecast'],axis=0)
            hs_gefs_forecast = np.append(hs_gefs_forecast,ENSDATA['hs_gefs_forecast'],axis=0)
            # Ground truth
            au10_obs = np.nanmean([ENSDATA['u10_ndbc'], \
                    np.nanmean(ENSDATA['u10_gefs_hindcast'][:,:,:,ENSDATA['indlat'], \
                    ENSDATA['indlon']],axis=2)],axis=0)
            ahs_obs = np.nanmean([ENSDATA['hs_ndbc'], \
                    np.nanmean(ENSDATA['hs_gefs_hindcast'][:,:,:,ENSDATA['indlat'], \
                    ENSDATA['indlon']],axis=2)],axis=0)
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
    fnvars = np.array(['mean_u10','var_u10','skew_u10','pctl_80_u10',
        'pctl_90_u10','pctl_95_u10','pctl_99_u10', 'mean_hs','var_hs',
        'skew_hs','pctl_80_hs','pctl_90_hs','pctl_95_hs','pctl_99_hs'])

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
    X_test  = np.array(x1[indval,:])
    y_train = np.array(v1[indtrain])
    y_test  = np.array(v1[indval])

    print(" Training records: "+repr(X_train.shape[0]))
    print(" Validation records: "+repr(X_test.shape[0])+"  "+repr(np.round((100*(len(indval)/len(indtrain))),2))+"%")


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# Using pre-trained estimators
stacker = StackingClassifier(
  estimators = [
      ('rf', rf),
      ('xgb', xgb),
      ('mlp', mlp)
      ],
  final_estimator=RandomForestClassifier(random_state=43),
  cv = 5
)

stacker.fit(X_train, y_train)
ystack = stacker.predict(X_test)
for i in range(0,len(y_test) ):
    print(i, ystack[i], y_test[i])

# scores for individual classifier, then stacker
for name, clf in stacker.named_estimators_.items():
    print('name = ',name,' score ', clf.score(X_test, y_test) )

# feature importance -------------------------------------------------
# Extract names from your estimators list
base_model_names = [name for name, clf in stacker.estimators]

# Map them to the final estimator's importances
importances = stacker.final_estimator_.feature_importances_

for name, score in zip(base_model_names, importances):
    print(f"Base Model: {name:5} | Importance to Stacker: {score:.4f}")

# For the base model's random forest
base_rf = stacker.named_estimators_['rf']
for i, score in enumerate(base_rf.feature_importances_):
    print(f"RF {i}: {score:.4f}")
