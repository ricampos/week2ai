#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dproc.py

VERSION AND LAST UPDATE:
 v1.0  09/20/2023

PURPOSE:
 Auxiliar functions for data reading and processing, statistics,
 filtering, and data-mining.

USAGE:
 Functions:
   wlevconv
   interp_nan
   read_obs
   read_obs_ft
   read_modelh
   read_model_ft
   ensproc
   qm_train
   qmcal
   qm_seasonal_train
   qmcal_seasonal
   hyear
   movavf
   butter_filter
   selevents
   weighted_blending
"""

import pylab
from pylab import *
from matplotlib.mlab import *
import scipy.stats as stats
from scipy import signal
import statistics
import numpy as np
import pandas as pd
import statistics
import pickle
import xarray as xr
import time
from calendar import timegm
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
import datetime
import pandas as pd
import mvalstats
import pvalstats
from pvalstats import ModelObsPlot
tstart = time.time()


def wlevconv(data_lev1=None,lev1=None,lev2=None,alpha=0.15):
    """
    Wind height conversion DNV-C209 standard/recommendation
    Input:
    - time-series (numpy array) of origin level
    - height (meters) of origin level
    - height (meters) of target level
    - alpha (see DNV-C209 standard/recommendation)
    Output:
    - time-series converted to the target level

    Example1 (10-m wind from model to 59-m anemometer position):
    lev1=10; lev2=59; data_lev1=10 #(m/s)
    converted_wind = wlevconv(data_lev1,lev1,lev2)

    Example2 (80-m height data to 10-m level):
    lev1=80; lev2=10; data_lev1=10 #(m/s)
    converted_wind = wlevconv(data_lev1,lev1,lev2)
    """

    if np.any(lev1)==None or np.any(lev2)==None or np.any(data_lev1)==None:
        raise ValueError("Two levels (in meters) and one data record (wind in m/s) must be informed.")

    mfactor=((lev2/lev1)**(alpha))
    data_lev2 = (mfactor * data_lev1)

    return data_lev2
    print(' Wind conversion ok')

def interp_nan(data,lmt=10**50):
    '''
    Fill NaN values with linear interpolation.
    User can enter one or two inputs:
      1) time-series containing NaN values to fill in
      2) maximum number of consecutive NaN values to interpolate (to
         avoid interpolating long segments)
    '''
    data=np.array(data)
    lmt=int(lmt)

    if data.ndim>1:
        raise ValueError(' Input array with too many dimensions. Only time-series (1 dimension) allowed.')
    else:
        indd=np.where(data>-999.)
        if np.size(indd)>0:
            indd=indd[0][-1] # last valid number
            adata=np.array(data[0:indd])

            # using pandas
            A=pd.Series(adata)
            # B=A.interpolate(method="polynomial",order=2,limit=lmt)
            B=A.interpolate(method="linear",limit=lmt)
            B=np.array(B.values)

            data[0:indd]=np.array(B)
           
            del lmt,A,B

    return data


# Read Met data ==================
def read_obs(fname,ftime,fnan=True,fsg=6):
    """ 
    Read in-situ measurements; convert wind speed/dir to U and V;
      and fill in small NaN gaps.
    Input:
    - file name (including full path)
    - target time array (ftime) using seconds since 1970 as reference.
    - allow (or not) fill NaN
    - maximum time-range to interpolate NaN, fsg, in hours
    Output:
    - numpy array containing the observations (allocated to ftime array)
    - time darray (datetime)
    - name of the stations
    - name of the variables
    - latitude
    - longitude
    """

    # Observations
    with open(fname, 'rb') as file:
        fobs = pickle.load(file)

    lat=np.array(fobs['meta']['LATITUDE']); lon=np.array(fobs['meta']['LONGITUDE'])
    nstations=np.array(list(fobs.keys()))[0:-1]

    # original variables
    vsel=['Anemometer_1_Avg [m/s]','Windvane_1_Avg [°]','Termometer_1_Avg [°C]','Barometer_1_Avg [hPa]','Hygrometer_1_Avg [%]']
    # final variables
    fvsel=['wind_speed','u-wind','v-wind','temp','press','rh']

    auxobs=np.zeros((len(nstations),len(vsel),len(ftime)),'f')*np.nan
    obs=np.zeros((len(nstations),len(vsel)+1,len(ftime)),'f')*np.nan
    for i in range(0,len(nstations)):

        datetime_index = pd.to_datetime(fobs[nstations[i]].index)
        auxt = datetime_index.astype(np.int64) // 10**9
        del datetime_index
        for j in range(0,len(ftime)):
            indt=np.where(np.abs(ftime[j]-auxt)<3600)
            # indt=np.where( ((auxt-ftime[j])<3600) & ((auxt-ftime[j])>=-1) )
            if np.size(indt)>0:
                indt=indt[0]
                for k in range(0,len(vsel)):
                    if vsel[k] in fobs[nstations[i]].keys():
                        auxobs[i,k,j]=float(np.nanmean(fobs[nstations[i]][vsel[k]].values[indt]))

                del indt

    fdatetime = np.array([datetime.datetime.utcfromtimestamp(t) for t in ftime])

    # Create U and V components based on Wind Intensity and Direction
    obs[:,0,:] = auxobs[:,0,:] # wind speed
    # Quality control
    aux=obs[:,0,:]; aux[aux<0]=np.nan; aux[aux>80]=np.nan; obs[:,0,:]=aux; del aux # remove outliers

    obs[:,1,:] = -auxobs[:,0,:] * np.sin(np.deg2rad(auxobs[:,1,:])) # U component
    obs[:,2,:] = -auxobs[:,0,:] * np.cos(np.deg2rad(auxobs[:,1,:])) # V component
    # allocate remaining variables:
    obs[:,3::,:] = auxobs[:,2::,:]

    # Quality Control

    if fnan == True:
        # fill_NaN small gaps
        for i in range(0,len(nstations)):
            for j in range(0,len(vsel)):
                obs[i,j,:]=interp_nan(obs[i,j,:],lmt=int(fsg/(np.diff(ftime).mean()/3600)))

    # data
    fvsel=np.array(fvsel).astype('str')
    return np.array(obs),fdatetime,nstations,fvsel,lat,lon


# Read Met data ==================
def read_obs_ft(fname,ctime,fctime,fsg=6,fnan=True):
    """
    Read in-situ measurements and allocate to forecast model time array
      using ctime and fctime; convert wind speed/dir to U and V;
      and fill in small NaN gaps.
    Input:
    - file name (including full path)
    - cycle time array (ctime) using seconds since 1970 as reference.
    - forecast lead time (fctime) in seconds. For ex. 0, 3600., 7200.
    - allow (or not) fill NaN
    - maximum time-range to interpolate NaN, fsg, in hours
    Output:
    - numpy array containing the observations (allocated to ftime array)
    - time darray (datetime)
    - name of the stations
    - name of the variables
    - latitude
    - longitude
    """

    fdatetime = np.array([datetime.datetime.utcfromtimestamp(t) for t in ctime])

    # Observations
    with open(fname, 'rb') as file:
        fobs = pickle.load(file)

    lat=np.array(fobs['meta']['LATITUDE']); lon=np.array(fobs['meta']['LONGITUDE'])
    nstations=np.array(list(fobs.keys()))[0:-1]

    # original variables
    vsel=['Anemometer_1_Avg [m/s]','Windvane_1_Avg [°]','Termometer_1_Avg [°C]','Barometer_1_Avg [hPa]','Hygrometer_1_Avg [%]']
    # final variables
    fvsel=['wind_speed','u-wind','v-wind','temp','press','rh']

    auxobs=np.zeros((len(nstations),len(vsel),len(ctime),len(fctime)),'f')*np.nan
    obs=np.zeros((len(nstations),len(vsel)+1,len(ctime),len(fctime)),'f')*np.nan
    for s in range(0,len(nstations)):
        datetime_index = pd.to_datetime(fobs[nstations[s]].index)
        auxt = datetime_index.astype(np.int64) // 10**9
        del datetime_index
        for i in range(0,len(ctime)):
            for j in range(0,len(fctime)):
                indt=np.where(np.abs((ctime[i]+fctime[j])-auxt)<3600)
                # indt=np.where( ((auxt-(ctime[i]+fctime[j]))<3600) & ((auxt-(ctime[i]+fctime[j]))>=-1) )
                if np.size(indt)>0:
                    indt=indt[0]
                    for k in range(0,len(vsel)):
                        if vsel[k] in fobs[nstations[s]].keys():
                            auxobs[s,k,i,j]=float(np.nanmean(fobs[nstations[s]][vsel[k]].values[indt]))

                    del indt

    # Create U and V components based on Wind Intensity and Direction
    obs[:,0,:,:] = auxobs[:,0,:,:] # wind speed
    # Quality control
    aux=obs[:,0,:,:]; aux[aux<0]=np.nan; aux[aux>80]=np.nan; obs[:,0,:,:]=aux; del aux # remove outliers

    obs[:,1,:,:] = -auxobs[:,0,:,:] * np.sin(np.deg2rad(auxobs[:,1,:,:])) # U component
    obs[:,2,:,:] = -auxobs[:,0,:,:] * np.cos(np.deg2rad(auxobs[:,1,:,:])) # V component
    # allocate remaining variables:
    obs[:,3::,:,:] = auxobs[:,2::,:,:]

    if fnan == True:
        # fill_NaN small gaps
        for s in range(0,len(nstations)):
            for k in range(0,len(vsel)):
                for i in range(0,len(ctime)):
                    obs[s,k,i,:] = interp_nan(obs[s,k,i,:],lmt=int(fsg/(np.diff(fctime).mean()/3600)))

    # data
    fvsel=np.array(fvsel).astype('str')
    return np.array(obs),fdatetime,nstations,fvsel,lat,lon


def read_modelh(fname,ftime,anh=59.,fsg=6,fnan=True):
    """ 
    Read model data (hindcasted using 24-h slices); convert wind height to anh level;
      and fill in small NaN gaps.
    Input:
    - file name (including full path)
    - target time array (ftime) using seconds since 1970 as reference.
    - target level to convert height
    - max hours for interpolation (fill in NaN gaps)
    Output:
    - numpy array containing the model output
    - time darray (datetime)
    - name of the stations
    - name of the variables
    - latitude of the stations
    - longitude of the stations
    """

    df = xr.open_dataset(fname)
    t = np.array(df['time'].values).astype('double')
    lat=np.array(df.latitude.values); lon=np.array(df.longitude.values)
    nstations=np.array(df.station.values).astype('str')
    nvars=np.array(list(df.data_vars.keys()))[0:-2].astype('str')

    fdatetime = np.array([datetime.datetime.utcfromtimestamp(t) for t in ftime])

    model=np.zeros((len(nstations),len(nvars),len(ftime)),'f')*np.nan
    for i in range(0,len(ftime)):
        indt=np.where(np.abs(ftime[i]-t)<3600)
        if np.size(indt)>0:
            indt=indt[0]
            for j in range(0,len(nvars)):
                model[:,j,i]=np.nanmean(df[nvars[j]].values[:,indt],axis=1)

            del indt

    # fill_NaN small gaps
    if fnan == True:
        for i in range(0,len(nstations)):
            for j in range(0,len(nvars)):
                model[i,j,:]=interp_nan(model[i,j,:],lmt=int(fsg/(np.diff(ftime).mean()/3600)))

    # convert wind speed (only) to anh level
    for i in [10,80,100,200]:
        auxvar="wnd"+str(i)+"m"
        if auxvar in nvars:
            ind=np.where(nvars==auxvar)[0]
            model[:,ind,:]=wlevconv(data_lev1=model[:,ind,:],lev1=i,lev2=anh)
            # Quality Control
            aux=model[:,ind,:]; aux[aux<0]=np.nan; aux[aux>80]=np.nan; model[:,ind,:]=aux; del aux, ind # Exclude outliers

    for i in [10,100]:
        auxvar="si"+str(i)
        if auxvar in nvars:
            ind=np.where(nvars==auxvar)[0]
            model[:,ind,:]=wlevconv(data_lev1=model[:,ind,:],lev1=i,lev2=anh)
            # Quality Control
            aux=model[:,ind,:]; aux[aux<0]=np.nan; aux[aux>80]=np.nan; model[:,ind,:]=aux # Exclude outliers
            # Rename Variable
            nvars[ind]="wnd"+str(i)+"m"
            del aux, ind

    # Replace the substring in each string using a list comprehension
    nvars = [string.replace('millibars','mb') for string in nvars]
    nvars=np.array(nvars).astype('str')
    # Quality control, remove NaN
    model[model==-999.]=np.nan; model[model==999.]=np.nan

    return np.array(model),fdatetime,nstations,nvars,lat,lon


def read_model_ft(fprfx,ctime,fctime,anh=59.,fsg=6,frd=None,fnan=True):
    """ 
    Read forecast data with 2 time dimensions, cycle time ctime (seconds since 1970 UTC, usually 1 cycle per day)
      and forecast time ftime (usually 1 hour, starts with 0, 3600 etc); convert wind height to anh level;
      and fill in small NaN gaps.
    Input:
    - file name prefix (including full path), the last YYYYMMDD_YYYYMMDD.nc will be
      looped based on cycle time (ctime) and forecast time (ftime)
    - cycle time array (ctime) using seconds since 1970 as reference.
    - forecast lead time (fctime) in seconds. For ex. 0, 3600., 7200.
    - target level to convert height
    - max hours for interpolation (fill in NaN gaps)
    Output:
    - numpy array containing the forecast model output
    - cycle time darray (datetime)
    - name of the stations
    - name of the variables
    - latitude of the stations
    - longitude of the stations
    """

    # WIND SPEED (only speed, not components), is converted to 59m. The name is kept as 100m, 80m etc, but
    # this name is kept to know the source only. The real level of the variable, after this functions, is 59m!

    fdatetime = np.array([datetime.datetime.utcfromtimestamp(t) for t in ctime])
    if frd == None:
        frd = (np.nanmax(fctime)-np.nanmin(fctime))/(3600.*24.) # forecast range in days

    c=0
    for i in range(0,len(ctime)):

        dcycle_in = str(time.gmtime(ctime[i])[0])+str(time.gmtime(ctime[i])[1]).zfill(2)+str(time.gmtime(ctime[i])[2]).zfill(2)
        dcycle_fin = str(time.gmtime(ctime[i]+frd*3600*24)[0])+str(time.gmtime(ctime[i]+frd*3600*24)[1]).zfill(2)+str(time.gmtime(ctime[i]+frd*3600*24)[2]).zfill(2)
        fname = fprfx+dcycle_in+"_"+dcycle_fin+".nc"

        try:
            df = xr.open_dataset(fname)
            t = np.array(df['time'].values).astype('double')
        except:
            print(" read_model_ft File not Found: "+fname)
        else:

            if c==0:
                lat=np.array(df.latitude.values); lon=np.array(df.longitude.values)
                nstations=np.array(df.station.values).astype('str')
                nvars=np.array(list(df.data_vars.keys()))[0:-2].astype('str')
                model=np.zeros((len(nstations),len(nvars),len(ctime),len(fctime)),'f')*np.nan

            for j in range(0,len(fctime)):
                indt=np.where(np.abs((ctime[i]+fctime[j])-t)<3600)
                if np.size(indt)>0:
                    indt=indt[0]
                    for k in range(0,len(nvars)):
                        model[:,k,i,j]=np.nanmean(df[nvars[k]].values[:,indt],axis=1)

                    del indt

            df.close(); del df

            if fnan == True:
                # fill_NaN small gaps
                for j in range(0,len(nstations)):
                    for k in range(0,len(nvars)):
                        model[j,k,i,:]=interp_nan(model[j,k,i,:],lmt=int(fsg/(np.diff(fctime).mean()/3600)))

            c=c+1
            # print(" read_modelf OK "+fname)

        del fname, dcycle_in, dcycle_fin

    if c>0:
        # convert wind speed (only) to anh (59m) level
        for j in [10,80,100,200]:
            auxvar="wnd"+str(j)+"m"
            if auxvar in nvars:
                ind=np.where(nvars==auxvar)[0]
                model[:,ind,:,:] = wlevconv(data_lev1=model[:,ind,:,:],lev1=j,lev2=anh)
                aux=model[:,ind,:,:]; aux[aux<0]=np.nan; aux[aux>80]=np.nan; model[:,ind,:,:]=aux
                del aux, ind

        for j in [10,100]:
            auxvar="si"+str(j)
            if auxvar in nvars:
                ind=np.where(nvars==auxvar)[0]
                model[:,ind,:,:]=wlevconv(data_lev1=model[:,ind,:,:],lev1=j,lev2=anh)
                # Quality Control
                aux=model[:,ind,:,:]; aux[aux<0]=np.nan; aux[aux>80]=np.nan; model[:,ind,:,:]=aux # Exclude outliers
                # rename variable
                nvars[ind]="wnd"+str(j)+"m"
                del aux, ind

        # Replace the substring in each string using a list comprehension
        nvars = [string.replace('millibars','mb') for string in nvars]
        nvars=np.array(nvars).astype('str')

        model[model==-999.]=np.nan; model[model==999.]=np.nan

        return np.array(model),fdatetime,nstations,nvars,lat,lon

    else:
        raise ValueError("No file found. No data read and allocated. "+fprfx)


def ensproc(fpath=None,fsfx=None,ftime=None,nmembers=30,fnan=True):
    """
    GEFS ensemble proc: mean, spread, skewness.
    Inputs: 
    - file path and sufix (to generate fname)
    - time array (seconds since 1970)
    - number of members (including the control 00)
    """

    if (fpath is None) | (fsfx is None) | (ftime is None):
        raise ValueError("Path, Sufix, and ftime (seconds since 1970) must be informed")

    for i in range(0,int(nmembers)+1):
        fname=fpath+"gefs_"+str(i).zfill(2)+fsfx # '_20220101_20230731.nc'
        agefs,mfdatetime,mnstations,nvars,mlat,mlon = read_modelh(fname,ftime,fnan=fnan)
        agefs[agefs<=-999.]=np.nan; agefs[agefs==999.]=np.nan
        if i==0:
            gefs=np.zeros((int(nmembers)+1,agefs.shape[0],agefs.shape[1],agefs.shape[2]),'float')*np.nan
            gefs[i,:,:,:]=np.array(agefs)
            fnvars=np.array(np.copy(nvars))
        else:
            for j in range(0,len(fnvars)):
                indv=np.where(nvars==fnvars[j])
                if np.size(indv)>0:
                    gefs[i,:,j,:] = agefs[:,indv[0][0],:]
                    del indv

        del agefs, nvars

    # Quality control
    gefs[gefs==-999.]=np.nan; gefs[gefs==999.]=np.nan

    gefs_em = np.nanmean(gefs,axis=0)
    gefs_std = np.nanstd(gefs,axis=0,ddof=1)
    gefs_skew = stats.skew(gefs,axis=0)

    # data
    return np.array(gefs_em),np.array(gefs_std),np.array(gefs_skew),mfdatetime,mnstations,fnvars,mlat,mlon


# Quantile Mapping bias-correction
def qm_train(model=None,obs=None,prob=None,pprint='yes'):
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

    if len(prob)>0:
        prob=np.array(np.arange(0.5,99.5+0.1,0.5),'f')  # Probability array

    slope,intercept = np.polyfit(np.nanpercentile(model,prob),np.nanpercentile(obs,prob),1)
    # model_cal=np.array((model*slope)+intercept)
    if pprint=='yes':
        print(" QMM linear regression. Slope: "+repr(slope)+"  Intercept: "+repr(intercept))

    return float(slope),float(intercept)

def qmcal(model=None,slope=1.,intercept=0.,pprint='yes'):
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

# Quantile Mapping bias-correction taking into account the seasonal cycle
def qm_seasonal_train(model=None,obs=None,ftime=None,twindow=45,prob=None,minfsize=10):
    """
    Univariate linear regression calibration using the Quantile Mapping Method
    Seasonal cycle is considered. Training module.
    Input:
    - model data
    - observations ("truth")
    - Time array (UTC, seconds since 1970)
    - Time window (in days) to pool in data (-twindow to +twindow) for the quantile mapping
    - probability array (optional) to define the percentiles
    Output:
    - slope (array with len = 365)
    - intercept (array with len = 365)
    - day-of-the-year (array with len = 365, days from 1 to 365)
    """

    if (np.size(model)>2)==False or (np.size(obs)>2)==False or (np.size(ftime)>2)==False:
        raise ValueError("Model, Obs, and Time (seconds since 1970) arrays must be informed.")

    if np.size(model) != np.size(obs) or np.size(obs) != np.size(ftime):
        raise ValueError("Model, Obs, and Time arrays must have the same sizes.")

    if len(model.shape)>1:
        raise ValueError("Only 1D Model array accepted.")

    if prob==None:
        prob=np.array(np.arange(0.5,99.+0.1,0.5),'f')  # Probability array

    # Time reference
    adof = np.arange(1,365+1) # day of the year
    # model days
    mdof = np.array([datetime.datetime.utcfromtimestamp(ts).timetuple().tm_yday for ts in ftime]) # day of the year
    mdof[mdof>365]=365 # Leap year

    # QMM
    model_cal=np.copy(model)
    slope=np.zeros((len(adof)),'f')*np.nan; intercept=np.zeros((len(adof)),'f')*np.nan
    for i in range(0,len(adof)):
        aux=np.array([adof[i]-twindow,adof[i]+twindow]); aux[aux<1]=aux[aux<1]+365; aux[aux>365]=aux[aux>365]-365
        if aux[1]<aux[0]:
            ind=np.where((mdof>=aux[0]) | (mdof<=aux[1]))
        else:
            ind=np.where((mdof>=aux[0]) & (mdof<=aux[1]))

        if np.size(ind)>minfsize:
            ind=ind[0]
            slope[i],intercept[i] = np.polyfit(np.nanpercentile(model[ind],prob),np.nanpercentile(obs[ind],prob),1)

    return np.array(slope).astype('float'),np.array(intercept).astype('float'),np.array(adof).astype('int')

def qmcal_seasonal(model=None,ftime=None,slope=None,intercept=None):
    """
    Univariate linear regression calibration using the Quantile Mapping Method
    Seasonal cycle is considered. Calibration module based on previously trained qm_seasonal_train
    Input:
    - model data
    - Time (UTC, seconds since 1970)
    - Slope
    - Intercept
    Output:
    - calibrated model data
    """

    if (np.size(model)>2)==False or (np.size(ftime)>2)==False or (np.size(slope)>2)==False or (np.size(intercept)>2)==False:
        raise ValueError("Model, Time (seconds since 1970), Slope, and Intercept arrays must be informed.")

    if np.size(model) != np.size(ftime):
        raise ValueError("Model and Time arrays must have the same sizes.")

    if (np.size(slope) != 365) or (np.size(intercept) != 365):
        raise ValueError("slope and intercept arrays must have len=365, generated with qm_seasonal_train.")

    if len(model.shape)>1:
        raise ValueError("Only 1D Model array accepted.")

    # Time reference
    adof = np.arange(1,365+1) # day of the year
    # model days
    mdof = np.array([datetime.datetime.utcfromtimestamp(ts).timetuple().tm_yday for ts in ftime]) # day of the year
    mdof[mdof>365]=365 # Leap year

    # QMM
    model_cal=np.copy(model)
    for i in range(0,len(adof)):
        ind=np.where(mdof==adof[i])
        if np.size(ind)>0:
            ind=ind[0]
            model_cal[ind]=np.array((model[ind]*slope[i])+intercept[i])

    return np.array(model_cal).astype('float')


def hyear(model=None,obs=None,ftime=None,fnan=True,fsg=6):
    """ 
    Rebuild dataset with homogeneous amount of records throughout the year.
    Not suitable for time-series modeling using sequence of forward time steps!
    Input:
    - Model (1D array)
    - Obs (1D array)
    - Time array (UTC, seconds since 1970)
    - allow (or not) fill NaN
    - max hours for interpolation (fill in NaN gaps)
    Output:
    - nmodel
    - nobs
    - ntime
    - pickv
    - dupl
    - fic
    """

    if (np.size(model)>2)==False and (np.size(obs)>2)==False:
        raise ValueError("At least one array, Model and/or Obs, must be informed.")

    if np.size(model) != np.size(ftime):
        raise ValueError("Model and Time arrays must have the same sizes.")

    if (np.size(obs)>2):
        if np.size(obs) != np.size(ftime):
            raise ValueError("Obs and Time arrays must have the same sizes.") 

    # time interval
    dt=float(statistics.mode(np.diff(ftime)))

    if fnan == True:
        # small gap interpolation, fill NaNs
        if fsg>0:
            model=interp_nan(model,lmt=int(fsg/(dt/3600)))
            obs=interp_nan(obs,lmt=int(fsg/(dt/3600)))

    # Number of years
    nby=(ftime[-1]-ftime[0])/(365.*24*3600)
    fnby=int(np.ceil(nby))
    # new time array
    datein=str(time.gmtime(ftime.min())[0])+'010100'
    datefin=str(time.gmtime(ftime.min())[0]+fnby)+'010100'
    ntime=np.array(np.arange(float(timegm( time.strptime(datein, '%Y%m%d%H') )),float(timegm( time.strptime(datefin, '%Y%m%d%H') )),dt)).astype('double')
    y=[];m=[]; ym=[]
    for i in range(0,len(ntime)):
        y=np.append(y,int(time.gmtime(ntime[i])[0]))
        m=np.append(m,int(time.gmtime(ntime[i])[1]))
        ym=np.append(ym,str(int(time.gmtime(ntime[i])[0]))+str(int(time.gmtime(ntime[i])[1])).zfill(2))

    nmodel=np.zeros((ntime.shape[0]),'f')*np.nan
    nobs=np.zeros((ntime.shape[0]),'f')*np.nan
    pickv=np.zeros((ntime.shape[0]),'f')
    dupl=np.zeros((ntime.shape[0]),'f')*np.nan
    fid=np.zeros((ntime.shape[0]),'f')*np.nan

    # first, data allocation
    for i in range(0,len(ntime)):
        indt=np.where(np.abs(ntime[i]-ftime)<3600)
        if np.size(indt)>0:
            indt=indt[0]
            nmodel[i]=np.nanmean(model[indt])
            nobs[i]=np.nanmean(obs[indt])
            del indt

    fnmodel=np.copy(nmodel); fnobs=np.copy(nobs)

    # Data homogenization in Time (year/mont)
    for i in range(0,len(np.unique(ym))):
        ind=np.where(ym==np.unique(ym)[i])
        if np.size(ind)>0:
            ind=np.array(ind[0]).astype('int')
            for j in range(0,np.size(ind)):
                # identify NaN in this block of month year
                if (nmodel[ind[j]]>-999)==False or (nobs[ind[j]]>-999)==False:
                    # search for valid records in this month (independent of the year)
                    indr=np.where((m==float(np.unique(ym)[i][-2::])) & (nmodel>-999.) & (nobs>-999.) )
                    # if there is any valid record in this month, it will be use to fill in the NaN
                    if np.size(indr)>0:
                        indr=np.array(indr[0]).astype('int')
                        # avoid filling in multiple times with the same data record (use pickv)
                        indmr=np.where(pickv[indr]==np.nanmin(pickv[indr]))[0]
                        indr=indr[indmr]
                        ri = np.random.randint(len(indr))
                        # fill in
                        fnmodel[ind[j]]=nmodel[indr[ri]]
                        fnobs[ind[j]]=nobs[indr[ri]]
                        # inform this specific record has been used (inform the source)
                        pickv[indr[ri]]=pickv[indr[ri]]+1
                        # inform in the final time-series this record comes from a fill-in process.
                        dupl[ind[j]]=dupl[ind[j]]+1
                        # and inform the index from where this fill-in record has been taken. 
                        fid[ind[j]]=float(indr[ri])

    return np.array(fnmodel).astype('float'), np.array(fnobs).astype('float'), ntime, np.array(pickv), np.array(dupl), np.array(fid)


#  Moving-average filter ==================
def movavf(data=None,ftime=None,ft=3):
    """ 
    Moving-average filter
    Input:
    - dataset with hourly data.
    - time array of the given time-series, using seconds since 1970 as reference.
    - filtering window (hors).
    Output:
    - filtered data
    https://towardsdatascience.com/moving-averages-in-python-16170e20f6c
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    """

    if data is None or ftime is None:
        raise ValueError("Missing required input arguments.")
        return

    data = np.array(data).astype('float')
    ftime = np.array(ftime).astype('double')
    ft=int(ft)

    if ft % 2 == 0:  # If n is even, add 1 to make it odd
        ft += 1

    if len(data.shape)>1:
        raise ValueError('Data array must have one dimension only (same as the time).')

    if ftime.shape[0] != data.shape[0]:
        raise ValueError('Data array does not match the time array.')

    ind=np.where(data>-999.)
    if np.size(ind)>1:
        ind=ind[0]
        ndata=data[ind]; nftime=ftime[ind]
    else:
        raise ValueError('No valid data in the dataset.')

    dtarr=np.diff(nftime)
    dt = statistics.mode(dtarr)
    ft=int(ft*(3600./dt)); dft=int(np.floor(ft/2))
    inddt=np.where(dtarr>dt)
    if np.size(inddt)>0:
        inddt = inddt[0]
        inddt = np.append(np.append(0,inddt),len(dtarr))
    else:
        inddt = np.array([0,len(dtarr)]).astype('int')

    for i in range(0,len(inddt)-1):
        fndata = np.array(ndata[inddt[i]:inddt[i+1]+1]).astype('float')
        if len(fndata[:])>ft:
            fndata[dft:-dft] = np.array(np.convolve(fndata[:], np.ones((ft,))/ft, mode='valid')).astype('float')
            data[ind[int(inddt[i]):int(inddt[i+1]+1)]] = fndata[:]

        del fndata

    return np.array(data).astype('float')


def butter_filter(data=None, cutoff_freq=None, sampling_rate=None, ftype='lowpass', order=4):
    """ 
    Butterworth Filter that can be used as low-pass or high-pass filter
    ftype='lowpass' or 'highpass'
    https://en.wikipedia.org/wiki/Butterworth_filter
    Input:
    - time-series (1-dimensional array, continuous variable)
    - cutoff frequency
    - sampling rate 
    - filter type
    - order
    Output:
    - filtered time-series
    """
    if data is None or cutoff_freq is None or sampling_rate is None:
        raise ValueError("Missing required input arguments.")
        return

    nyq_freq = float(1./sampling_rate) * 0.5
    normal_cutoff = float(cutoff_freq) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype=ftype)
    y = signal.filtfilt(b, a, data)
    return y


# Select indexes of peaks and troughs ==================
def selevents(tseries=None,ftime=None,twindow=24.,thrc=-999.):
    """ 
    Select events based on the identification of crests and troughs
    Input:
    - time-series of a continuous variable
    - time array of the given time-series, using seconds since 1970 as reference.
    - time window to select your minimum distance between peaks (in hours), 
    and ensure statistical independency (iid events)
    - threshold defining a minimum value the peaks must have to be consider an event.
    Output:
    - index of the crests
    - index of the troughs
    """

    if (np.size(tseries)>2)==False or (np.size(ftime)>2)==False:
        raise ValueError("Data and time array must be informed.")

    if np.size(tseries) != np.size(ftime):
        raise ValueError("Time-series and time array must have same sizes.")

    nrt=int(twindow/(np.diff(ftime).mean()/3600)); nrtd=int(np.ceil(nrt/2))

    # Peak
    indcrest=np.array([0]).astype('int')
    for i in range(nrtd,int(len(ftime)-nrtd)):
        # check if there is valid data within the segment
        if np.size(np.where(tseries[(i-nrtd):(i+nrtd+1)]>-999))>=2:
            # ensure there is a peak, not a single slope
            if np.nanmax(tseries[(i-nrtd):(i+nrtd+1)])>tseries[(i-nrtd):(i+nrtd+1)][0] and np.nanmax(tseries[(i-nrtd):(i+nrtd+1)])>tseries[(i-nrtd):(i+nrtd+1)][-1] and np.nanmax(tseries[(i-nrtd):(i+nrtd+1)])>thrc:
                newind=int((i-nrtd)+np.min(np.where(tseries[(i-nrtd):(i+nrtd+1)]==np.nanmax(tseries[(i-nrtd):(i+nrtd+1)]))))
                if int(indcrest[-1])!=newind:
                    if np.abs(indcrest[-1]-newind)<nrt:
                        if tseries[newind]>tseries[int(indcrest[-1])]:
                            indcrest=indcrest[0:-1]
                            indcrest=np.append(indcrest,newind)                         

                    else:
                            indcrest=np.append(indcrest,newind)

    indcrest=np.array(np.unique(indcrest[1::])).astype('int')

    # Trough
    indtrough=[]
    if np.nanmean(tseries[0:indcrest[0]])>-999:
        indtrough = np.append(indtrough,np.min(np.where(tseries[0:indcrest[0]]==np.nanmin(tseries[0:indcrest[0]]))))

    for i in range(0,len(indcrest)-1):
        indtrough = np.append(indtrough,indcrest[i]+np.min(np.where(tseries[indcrest[i]:indcrest[i+1]]==np.nanmin(tseries[indcrest[i]:indcrest[i+1]]))))

    if np.nanmean(tseries[indcrest[-1]::])>-999:
        indtrough = np.append(indtrough,np.min(np.where(tseries[indcrest[-1]::]==np.nanmin(tseries[indcrest[-1]::]))))

    indtrough=np.array(np.unique(indtrough)).astype('int')

    print(" Total of "+repr(len(indcrest))+" events.")
    return indcrest, indtrough


#  Weighted blending to combine multiple forecast and obtain a single prediction ==================
def weighted_blending(model=None,obs=None,ftime=None,twindow=45):
    """ 
    Weighted blending of multiple forecast models
    Input:
    - dataset of model array [model, time]
    - observations for the exact time [time]
    - time array of the given time-series, using seconds since 1970 as reference.
    - Time window (in days) to pool in data (-twindow to +twindow)
    Output:
    - weights for each model [model,julian-day(365)]
    """

    if model is None or obs is None or ftime is None:
        raise ValueError("Missing required input arguments.")
        return

    model = np.array(np.atleast_2d(model)).astype('float')
    obs = np.array(obs).astype('float')
    ftime = np.array(ftime).astype('double')

    if model.shape[0] > model.shape[1]:
        model = model.T

    if len(obs.shape)>1:
        raise ValueError('Observation arrays must have one dimension only (same measurement for all the models).')

    if model.shape[1] != obs.shape[0]:
        raise ValueError('Model and observation arrays must have the same index/time size.')

    if ftime.shape[0] != model.shape[1]:
        raise ValueError('Forecast time array does not match the model/obs size.')

    # Time reference
    adof = np.arange(1,365+1) # day of the year
    # model days
    mdof = np.array([datetime.datetime.utcfromtimestamp(ts).timetuple().tm_yday for ts in ftime]) # day of the year
    mdof[mdof>365]=365 # Leap year

    m=np.zeros((model.shape[0],len(adof)),'f')*np.nan
    wb=np.zeros((model.shape[0],len(adof)),'f')*np.nan 

    for i in range(0,len(adof)):
        aux=np.array([adof[i]-twindow,adof[i]+twindow]); aux[aux<1]=aux[aux<1]+365; aux[aux>365]=aux[aux>365]-365
        if aux[1]<aux[0]:
            ind=np.where((mdof>=aux[0]) | (mdof<=aux[1]))
        else:
            ind=np.where((mdof>=aux[0]) & (mdof<=aux[1]))

        if np.size(ind)>0:
            ind=ind[0]
            for j in range(0,model.shape[0]):
                # error metric
                m[j,i] = float(1.-mvalstats.metrics(model[j,ind],obs[ind],vmin=-999.)[6]) # HH Hanna and Heinold (1985)

            # weights
            wb[:,i] = m[:,i]/np.sum(m[:,i])

    del m
    return np.array(wb).astype('float'), np.array(adof).astype('int')


def normalization(*args):
    ''' Normalization Run
    input X[time, variables], method, min(x), max(x)
    '''

    if len(args) == 3:
        method = 1
        x=np.copy(args[0]).astype('f')
        xp1=np.copy(args[1]).astype('f')
        xp2=np.copy(args[2]).astype('f')
    elif len(args) == 4:
        x=np.copy(args[0]).astype('f')
        method=int(np.copy(args[1])) 
        xp1=np.copy(args[2]).astype('f')
        xp2=np.copy(args[3]).astype('f')

    elif len(args) < 3:
        sys.exit(' ERROR! More inputs are requested')

    x=np.atleast_2d(x)

    # normalization to create arrays between 0 and 1.
    if method == 1:
        x1 = ( x - xp1 ) / ( xp2 - xp1 )

    # same as 1 with log function applied
    elif method ==2:
        x = np.copy(np.log(x-xp1+1.))
        x1 = x / xp2

    # normalization to create arrays between -1 and 1.
    if method == 3:
        x1 = 2.*(( x - xp1 ) / ( xp2 - xp1 ))-1.

    # same as 3 with log function applied
    if method == 4:
        x = np.copy(np.log(x-xp1+1.))
        x1 = 2.*(x / xp2)-1.

    # subtract mean and divide by 3*standardDeviation
    elif method == 5:
        x1= (x-xp1) / (3.*xp2)

    # non-negative values
    elif method ==6:
        x1 = x - xp1

    return x1,xp1,xp2


def denormalization(*args):
    ''' DeNormalization
    input X[time, variables], method, min(x), max(x)
    '''
    x=np.copy(args[0]).astype('f')
    method=np.copy(args[1]).astype('i')
    xp1=np.copy(args[2]).astype('f')
    xp2=np.copy(args[3]).astype('f')
    if method==1:
        y=(x*(xp2-xp1)+xp1)
    elif method==2:
        y=x*xp2
        y=np.exp(y)-1.+xp1
    elif method==3:
        y=(((x+1.)/2.)*(xp2-xp1)+xp1)
    elif method==4:
        y=((x+1.)/2.)*xp2
        y=np.exp(y)-1.+xp1
    elif method==5:
        y=(x*(3*xp2)+xp1)
    elif method==6:
        y=(x+xp1)
    return y


