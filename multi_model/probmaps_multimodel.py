#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
probmaps_multimodel.py

VERSION AND LAST UPDATE:
 v1.0  10/19/2026

PURPOSE:
 Multi-model version of probmaps.py. The members of NOAA and ECMWF ensembles
  (e.g. 31 from NOAA/GEFS + 51 from ECMWF = 82 members) are appended into
  one single ensemble, and the probability maps (and percentile maps) are
  computed from that combined ensemble, exactly as probmaps.py does for a
  single ensemble.
 The global hazards outlook (probability map) shows the probability of
  a variable (wind speed or significant wave height) to have at least one value
  above certain pre-defined level within the given time range.
 How the models are combined:
  - every model is read with the same readers used in probmaps.py (read_gefs,
    read_mm), quality-controlled, converted to the output units and cropped to
    the domain;
  - the grids must be identical after the domain selection (same points and
    resolution). This is checked (check_grid) and a message is printed; if they
    do not match the code stops. No interpolation is applied. The first model in
    `models:` is the reference;
  - the n-max expansion (the nmax largest values in time, per member) is done
    PER MODEL and the results are appended along the member axis. This way the
    models do not need the same number of time steps;
  - the space-time cells and probabilities (stprob) then treat all appended
    members as one single ensemble.

USAGE:
 The information is passed to the script through 5 input arguments:
  1) .yaml configuration file;
  2) forecast cycle (YYYYMMDDHH) that will be used to compute the probabilities;
  3) initial day to define the time interval to calculate the statistics;
  4) final day to define the time interval to calculate the statistics;
  5) variable (select only one) to be processed: WS10 or Hs;
 This script must be run for each variable (WS10, Hs) separately.

 The .yaml is a copy of the operational probmaps_*.yaml where the keys that
  belong to one model (model, mpath, nenm, mvar_hs, mvar_wnd) are replaced by a
  `models:` list (nested keys are named `path` and `nenm`, not `mpath`, so
  shell scripts that grep the yaml for `mpath` do not find duplicates):

   models:
     - name: GEFS            # GEFS, ECMWF or EnvCanada
       path: /path/to/GEFS/archive/
       nenm: 31
     - name: ECMWF
       path: /path/to/ECMWF/archive/
       nenm: 51
       mvar_hs: <same value as in probmaps_ecmwf.yaml>
       mvar_wnd: <same value as in probmaps_ecmwf.yaml>
   label: MULTI              # optional, used in the output file names
   require_all_models: true  # optional; false = skip a model that cannot be
                             #  read and continue with the others (the figure
                             #  title lists the models actually used)

  All the remaining keys (ftag, mode, tres, nmax, spws, lonmin ..., plevels,
  qlev_*, qqvmax_*, vtickd_*, pctls, outpath ...) are shared by all models.

 Example (from linux terminal command line):
  python3 probmaps_multimodel.py probmaps_multimodel.yaml 2026101900 7 14 Hs

OUTPUT:
 Probability maps and percentile maps saved in the outpath directory:
  ProbMap_<var>_<level>_fcst07to14_<label>_<ftag>.png
  Pctl<NN>_<var>_fcst07to14_<label>_<ftag>.png

DEPENDENCIES:
 See the imports below.

AUTHOR and DATE:
 10/19/2026: Ricardo M. Campos, first version (adapted from probmaps.py v3.2).

PERSON OF CONTACT:
 Ricardo M Campos: ricardo.campos@noaa.gov

"""

# Pay attention to the pre-requisites and libraries
import matplotlib
matplotlib.use('Agg')
import xarray as xr
import matplotlib.pyplot as plt
import yaml
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
import numpy as np
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import sys
import warnings; warnings.filterwarnings("ignore")
# -------------------------
sl=13 # plot style configuration
matplotlib.rcParams.update({'font.size': sl}); plt.rc('font', size=sl)
matplotlib.rc('xtick', labelsize=sl); matplotlib.rc('ytick', labelsize=sl); matplotlib.rcParams.update({'font.size': sl})
# --------------------------------------------------------------------------

# Reading functions (same as probmaps.py)
def read_gefs(auxltime,fvarname,nenm,mpath,fcdate,fchour):
    '''
    Read GEFS grib2 ensemble forecast files.
    '''

    c=0
    for t in range(0,auxltime.shape[0]):
        for enm in range(0,nenm):

            fname=mpath+"GEFSv12Waves_"+fcdate+fchour+"/gefs.wave."+fcdate+"."+str(int(enm)).zfill(2)+".global.0p25.f"+str(int(auxltime[t])).zfill(3)+".grib2"

            if c==0:
                ds = xr.open_dataset(fname, engine='cfgrib')
                wtime = np.atleast_1d(np.array(ds.time.values))
                lat = np.array(ds.latitude.values); lat = np.sort(lat); lon = np.array(ds.longitude.values)
                fmod=np.zeros((auxltime.shape[0],nenm,lat.shape[0],lon.shape[0]),'f')*np.nan
                ds.close(); del ds

                if fvarname == 'Hs' or fvarname == 'HS':
                    xrvar='swh'
                else:
                    xrvar='ws'

            ds = xr.open_dataset(fname, engine='cfgrib')
            fmod[t,enm,:,:] = np.array(np.flip(ds[xrvar].values[:],axis=0)).astype('float')
            ds.close(); del ds

            c=c+1

        print(repr(t))

    return fmod, lat, lon, wtime

def read_mm(auxltime,model,fvarname,mvar,nenm,mpath,fcdate,fchour):
    '''
    Read ECMWF and EnvCanada netcdf ensemble forecast files.
    Downloaded and post-processed by
      download_ECMWF_ENS_wave.sh or download_ECMWF_ENS_wind.sh  (ECMWF)
      download_GEWPS_wave.sh or download_GEPS_wind.sh (EnvCanada)
    '''
    import netCDF4 as nc

    # variable
    if fvarname.upper() == "WS10" or fvarname.upper() == "WND" or fvarname.upper() == "U10":
        auxvarn = 'wind'
    else:
        auxvarn = 'wave'

    # model informed in the .yaml configuration file
    if model=='ECMWF':
        mtag='ECMWF_ENS'
    elif model=='EnvCanada':
        if auxvarn == 'wave':
            mtag='GEWPS'
        else:
            mtag='GEPS'

    omvar=mvar
    c=0 # loop through the ensemble members
    for enm in range(0,nenm):
        fname = mpath+auxvarn+"/"+fchour+"/"+mtag+"_"+auxvarn+"_"+fcdate+fchour+"."+str(enm).zfill(2)+".nc"
        try:
            f=nc.Dataset(fname)
            if omvar.split(' ')[0] in list(f.variables.keys()):

                if c==0:
                    ftime = np.array(f.variables['time'][:]).astype('int')
                    indt = np.where(np.isin(ftime, auxltime))[0]
                    lat = np.sort(np.array(f.variables['lat'][:]))
                    lon = np.array(f.variables['lon'][:]); lon=lon+180.; half = len(lon) // 2
                    fmod = np.zeros((len(indt),nenm,len(lat),len(lon)),'f')
                    wtime = np.atleast_1d(np.datetime64(pd.to_datetime(fcdate+fchour, format='%Y%m%d%H')))

                if fvarname.upper() == "WS10" or fvarname.upper() == "WND" or fvarname.upper() == "U10":
                    # u and v components
                    if c==0:
                        mvar=np.array(omvar.split(' ')).astype('str')

                    auxu = np.array(f.variables[mvar[0]][indt,:,:])
                    auxv = np.array(f.variables[mvar[1]][indt,:,:])
                    aux = np.sqrt(auxu**2 + auxv**2)[:,0,:,:]; del auxu,auxv
                    if model=='ECMWF':
                        aux = np.flip(aux,axis=1)

                else:
                    if model=='ECMWF':
                        aux = np.flip(np.array(f.variables[mvar][indt,:,:]), axis=1)
                    else:
                        aux = np.array(f.variables[mvar][indt,:,:])

                aux = np.roll(aux, shift=half, axis=2)
                aux[aux<-100]=np.nan; aux[aux>100]=np.nan
                fmod[:,enm,:,:] = aux
                del aux
                f.close()

            else:
                sys.exit(" Enviromental variable "+mvar+" not found in the file "+fname)

        except:
            sys.exit(" Could not open and read file "+fname)
        else:
            c=c+1

        print(repr(c))

    return fmod, lat, lon, wtime


# Multi-model: domain, common grid, and appending the members

def crop_domain(fmod,lat,lon,latmin,latmax,lonmin,lonmax):
    '''
    Select the domain of interest. fmod: (time, member, lat, lon).
    '''
    indlat=np.where((lat>=latmin)&(lat<=latmax))[0]
    indlon=np.where((lon>=lonmin)&(lon<=lonmax))[0]
    if indlon.size==0 or indlat.size==0:
        sys.exit(" Min/Max lat and lon incorrect when applied to the forecast file. Check longitude standards.")

    return np.ascontiguousarray(fmod[:,:,indlat,:][:,:,:,indlon]), lat[indlat], lon[indlon]


def same_grid(lat1,lon1,lat2,lon2,atol=1.e-3):
    '''
    True when two regular grids are the same.
    '''
    return (lat1.shape==lat2.shape) and (lon1.shape==lon2.shape) and np.allclose(lat1,lat2,atol=atol) and np.allclose(lon1,lon2,atol=atol)


def check_grid(lat_ref,lon_ref,lat,lon,name_ref,name,tol=1.e-3):
    '''
    The models are appended member by member, so they must share exactly the same
    grid points after the domain selection. Prints a confirmation message when they
    match, otherwise prints what differs and stops the code (no interpolation).
    '''
    if same_grid(lat_ref,lon_ref,lat,lon,atol=tol):
        print(" Grid check OK: "+name+" matches "+name_ref+" ("+str(lat.shape[0])+" lat x "+str(lon.shape[0])+" lon points, "
              +"resolution "+str(round(float(np.diff(lat).mean()),4))+" x "+str(round(float(np.diff(lon).mean()),4))+" deg).")
        return

    print(" "); print(" ERROR: the grids of "+name_ref+" and "+name+" do not match.")
    for nm,la,lo in ((name_ref,lat_ref,lon_ref),(name,lat,lon)):
        print("   "+nm+": "+str(la.shape[0])+" lat x "+str(lo.shape[0])+" lon points | lat "+str(round(float(la[0]),4))+" to "+str(round(float(la[-1]),4))
              +" (step "+str(round(float(np.diff(la).mean()),4))+") | lon "+str(round(float(lo[0]),4))+" to "+str(round(float(lo[-1]),4))
              +" (step "+str(round(float(np.diff(lo).mean()),4))+")")
    if lat_ref.shape==lat.shape:
        print("   max |lat difference| = "+str(float(np.abs(lat_ref-lat).max())))
    if lon_ref.shape==lon.shape:
        print("   max |lon difference| = "+str(float(np.abs(lon_ref-lon).max())))
    sys.exit(" Multi-model run stopped: the ensembles must be on the same grid (no interpolation is applied).")


def nmax_expand(fmod,nmax):
    '''
    n-max expansion of one model: for every member, the nmax largest values along the
    time axis. fmod: (time, member, lat, lon) -> (nmax*member, lat, lon).
    '''
    fmod=np.sort(fmod,axis=0)
    fmod=fmod[-nmax::,:,:,:]
    return np.array(fmod.reshape(fmod.shape[0]*fmod.shape[1],fmod.shape[2],fmod.shape[3]))


def stprob(fexp,lat,lon,spws,qlev,spctl):
    '''
    Space-Time Cells and Probabilities.
    fexp: n-max expanded members of ALL models appended, (nsamples, lat, lon),
    nsamples = nmax * (total number of members).
    '''

    nsamp=fexp.shape[0]

    gspws=int(np.floor(spws/np.diff(lat).mean())/2)

    probecdf=np.zeros((qlev.shape[0],lat.shape[0],lon.shape[0]),'f')
    for i in range(0,qlev.shape[0]):
        for j in range(0,lat.shape[0]):
            for k in range(0,lon.shape[0]):
                if np.any(fexp[:,j,k]>0.0):
                    if (j>=gspws) and (j<=lat.shape[0]-gspws) and (k>=gspws) and (k<=lon.shape[0]-gspws):
                        aux=np.array(fexp[:,(j-gspws):(j+gspws+1),:][:,:,(k-gspws):(k+gspws+1)])
                        aux=aux.reshape(nsamp,aux.shape[1]*aux.shape[2])
                        aux=np.sort(aux,axis=1)
                        ind=np.where(np.mean(aux,axis=0)>=0.)
                        if np.size(ind)>0:
                            aux=np.array(aux[:,ind[0]])
                            aux=np.array(aux[:,int(np.floor(aux.shape[1]*(spctl/100)))::])
                            aux=aux.reshape(aux.shape[0]*aux.shape[1])
                            probecdf[i,j,k] = np.size(aux[aux>qlev[i]]) / np.size(aux[aux>0.])

                        del aux

    return probecdf


def pooled_percentiles(arrs,pctls,lat_chunk=40):
    '''
    Percentiles over all time steps and members of all models together.
    arrs: list of (time, member, lat, lon) with the same lat/lon. Done in latitude
    blocks so the pooled array is never built at once. Returns (len(pctls), lat, lon).
    '''
    nlat=arrs[0].shape[2]
    out=np.full((len(pctls),nlat,arrs[0].shape[3]),np.nan,'f')
    for a0 in range(0,nlat,lat_chunk):
        sl_=slice(a0,min(a0+lat_chunk,nlat))
        nb=sl_.stop-sl_.start
        blk=np.concatenate([a[:,:,sl_,:].reshape(-1,nb,a.shape[3]) for a in arrs],axis=0)
        out[:,sl_,:]=np.nanpercentile(blk,pctls,axis=0)
        del blk
    return out


# Plot functions

def pctl_plot(pfields,wlevels,pctls,lat,lon,slonmin,slonmax,slatmin,slatmax,spws,outpath,fpname,mtxt,ftag,fvarname,funits,fcycle,trfi,wtime,vtickd,ltime1,ltime2):
    '''
    Percentile plots.
    '''

    # Percentiles informed in the .yaml configuration file.
    for i in range(0,pctls.shape[0]):
        plt.figure(figsize=(9,5.5))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=-90))
        ax.set_extent([slonmin+spws,slonmax-spws,slatmin+spws,slatmax-spws], crs=ccrs.PlateCarree())
        gl = ax.gridlines(crs=ccrs.PlateCarree(),  xlocs=range(-180,180, 20), draw_labels=True, linewidth=0.5, color='grey', alpha=0.5, linestyle='--')
        gl.xlabel_style = {'size': 9, 'color': 'k','rotation':0}; gl.ylabel_style = {'size': 9, 'color': 'k','rotation':0}
        cs=ax.contourf(lon,lat,pfields[i,:,:],levels=wlevels,alpha=0.7,cmap='jet',zorder=1,extend="max",transform = ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.OCEAN,facecolor=("white"))
        ax.add_feature(cartopy.feature.LAND,facecolor=("lightgrey"), edgecolor='grey',linewidth=0.5, zorder=2)
        ax.add_feature(cartopy.feature.BORDERS, edgecolor='grey', linestyle='-',linewidth=0.5, alpha=1, zorder=3)
        ax.coastlines(resolution='50m', color='dimgrey',linewidth=0.5, linestyle='-', alpha=1, zorder=4)
        title = "Percentile"+str(pctls[i]).zfill(2)+" "+fvarname+" ("+funits+"), Cycle "+fcycle[0:8]+" "+fcycle[8:10]+"Z ("+mtxt+") \n"
        title += r"$\bf{"+trfi+"Valid: "+pd.to_datetime(wtime[0]+np.timedelta64(ltime1,'D')).strftime('%B %d, %Y')+" - "
        title += pd.to_datetime(wtime[0]+np.timedelta64(ltime2,'D')).strftime('%B %d, %Y')+"}$"
        ax.set_title(title); del title
        plt.tight_layout()
        ax = plt.gca(); pos = ax.get_position(); l, b, w, h = pos.bounds; cax = plt.axes([l+0.06, b-0.07, w-0.12, 0.03]) # setup colorbar axes.
        cbar = plt.colorbar(cs,cax=cax, orientation='horizontal', format='%g')
        labels = np.arange(0, wlevels.max(),vtickd).astype('int'); ticks = np.arange(0, wlevels.max(),vtickd).astype('int')
        cbar.set_ticks(ticks); cbar.set_ticklabels(labels)
        plt.axes(ax); plt.tight_layout()
        plt.text(-90., 76., 'Experimental', color='k', fontsize=13, fontweight='bold')
        figname = outpath+"Pctl"+str(pctls[i]).zfill(2)+"_"+fvarname+"_fcst"+str(ltime1).zfill(2)+"to"+str(ltime2).zfill(2)+"_"+fpname
        plt.savefig(figname+".png", dpi=200, facecolor='w', edgecolor='w',
            orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)

        plt.close('all')

        del ax, figname
        print(" 2. Initial Plots. Percentile "+str(pctls[i]).zfill(2)+" ok.")


def pm_plot(probecdf,wtime,wconfig,mode,spws,plevels,qlev,hplevels,hpcolors,pcolors,lat,lon,slonmin,slonmax,slatmin,slatmax,fvarname,funits,fpname,mtxt,fcycle,outpath,ltime1,ltime2,gft,trfi):
    '''
    Probability Maps.
    '''

    # Probability levels and labels for the plots
    clabels=[];clevels=[]
    for j in range(0,np.size(plevels)-1):
        clabels=np.append(clabels,">"+str(int(plevels[j]*100)).zfill(2)+"%")
        clevels=np.append(clevels,(plevels[j]+plevels[j+1])/2)

    cmap = ListedColormap(pcolors)

    # If hurricane season (May 15th to Nov 31st), it masks (or cuts) the tropics for external plots
    mmdd = wtime[0].astype('datetime64[D]').astype(object).month * 100 + wtime[0].astype('datetime64[D]').astype(object).day
    if mode=='external' and (mmdd>=515 and mmdd<=1130):
        slatmin=wconfig['latminhs']-spws # min latitude for the hurricane season

    # Loop through the intensity levels
    for i in range(0,qlev.shape[0]):

        # Extra percentage level associated with the most extreme case
        if i == int(qlev.shape[0]-1):

            plevels = hplevels
            pcolors = hpcolors

            clabels=[];clevels=[]
            for j in range(0,np.size(plevels)-1):
                clabels=np.append(clabels,">"+str(int(plevels[j]*100))+"%")
                clevels=np.append(clevels,(plevels[j]+plevels[j+1])/2)

            cmap = ListedColormap(pcolors)

        # Figure
        plt.figure(figsize=(9,5.5))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=-90))
        ax.set_extent([slonmin+spws,slonmax-spws,slatmin+spws,slatmax-spws], crs=ccrs.PlateCarree())
        gl = ax.gridlines(crs=ccrs.PlateCarree(),  xlocs=range(-180,180, 20), draw_labels=True, linewidth=0.5, color='grey', alpha=0.5, linestyle='--')
        gl.xlabel_style = {'size': 9, 'color': 'k','rotation':0}; gl.ylabel_style = {'size': 9, 'color': 'k','rotation':0}
        ax.add_feature(cartopy.feature.OCEAN,facecolor=("white"))
        ax.add_feature(cartopy.feature.LAND,facecolor=("lightgrey"), edgecolor='grey',linewidth=0.5, zorder=2)
        ax.add_feature(cartopy.feature.BORDERS, edgecolor='grey', linestyle='-',linewidth=0.5, alpha=1, zorder=3)
        ax.coastlines(resolution='50m', color='dimgrey',linewidth=0.5, linestyle='-', alpha=1, zorder=4)
        title = "Prob "+fvarname+">"+str(qlev[i]).zfill(1)+funits+", Cycle "+fcycle[0:8]+" "+fcycle[8:10]+"Z ("+mtxt+") \n"
        title += r"$\bf{"+trfi+"Valid: "+pd.to_datetime(wtime[0]+np.timedelta64(ltime1,'D')).strftime('%B %d, %Y')+" - "
        title += pd.to_datetime(wtime[0]+np.timedelta64(ltime2,'D')).strftime('%B %d, %Y')+"}$"

        # If external and in the hurricane season, it masks (or cuts) the tropics.
        if mode=='external' and (mmdd>=515 and mmdd<=1130) and np.any((np.linspace(wconfig['latminhs'],slatmax,10)>-25.) & (np.linspace(wconfig['latminhs'],slatmax,10)<25.)):
            contour_data = gaussian_filter(probecdf[i, :, :], gft)
            tropical_mask = (lat >= -30) & (lat <= 30)
            # Expand tropical_mask to match the shape of contour_data
            tropical_mask_expanded = np.tile(tropical_mask[:, np.newaxis], (1, lon.shape[0]))
            # Add transparency to tropical areas
            cs = ax.contourf(lon, lat, np.ma.masked_where(tropical_mask_expanded, contour_data),
                                          levels=plevels, cmap=cmap, alpha=0.7, zorder=1, transform=ccrs.PlateCarree())

            ax.add_patch(plt.Rectangle((slonmin, -30), 360, 60, transform=ccrs.PlateCarree(), color='whitesmoke', alpha=0.4, zorder=2, hatch='//', edgecolor='none'))
            ax.add_patch(plt.Rectangle((slonmin, -30), 360, 60, transform=ccrs.PlateCarree(), color='whitesmoke', alpha=0.4, zorder=2, hatch='\\', edgecolor='none'))

            plt.text(slonmin+spws-270+1, 0., 'This product is not intended for tropical cyclones', color='k', fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), zorder=4)
            plt.text(slonmin+spws-270+1, -5., 'Tropical latitudes are masked during hurricane season.', color='k', fontsize=7, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),zorder=4)

        else:
            cs=ax.contourf(lon,lat,gaussian_filter(probecdf[i,:,:],gft),levels=plevels,alpha=0.7,cmap=cmap,zorder=1,transform = ccrs.PlateCarree())

        ax.set_title(title); del title
        plt.tight_layout()
        ax = plt.gca(); pos = ax.get_position(); l, b, w, h = pos.bounds; cax = plt.axes([l+0.06, b-0.07, w-0.12, 0.03]) # setup colorbar axes.
        cbar = plt.colorbar(cs,cax=cax, orientation='horizontal',ticks=clevels, format='%g')
        cbar.ax.set_xticklabels(clabels)
        cbar.ax.tick_params(length=0)
        for label in cbar.ax.get_xticklabels():
            label.set_weight('bold')

        plt.axes(ax); plt.tight_layout()
        plt.text(-90., 76., 'Experimental', color='k', fontsize=13, fontweight='bold')
        figname = outpath+"ProbMap_"+fvarname+"_"+str(qlev[i]).zfill(1)+"_fcst"+str(ltime1).zfill(2)+"to"+str(ltime2).zfill(2)+"_"+fpname
        plt.savefig(figname+".png", dpi=200, facecolor='w', edgecolor='w',
                orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)

        plt.close('all')

        del ax, figname
        print("   Plot ... qlev "+repr(qlev[i]))

# ------------------------------------------------------------

if __name__ == "__main__":

    # Input Arguments -----
    # .yaml configuration file
    fconfig=str(sys.argv[1])
    # Forecast Cycle
    fcycle=str(sys.argv[2])
    fcdate=str(fcycle[0:8]); fchour=str(fcycle[8:10])
    # Forecast Lead Time (Day) and intervall
    ltime1=int(sys.argv[3])
    ltime2=int(sys.argv[4])
    # forecast variable: WS10, Hs
    fvarname=str(sys.argv[5])

    # Fixed configuration variables, read yaml file -----------
    print(" "); print(" Reading yaml configuration file ...")
    with open(fconfig, 'r') as file:
        wconfig = yaml.safe_load(file)

    # string name to distinguish domains
    ftag=str(wconfig['ftag'])

    # Models to be combined
    models=wconfig['models']
    if not isinstance(models,list) or len(models)<1:
        sys.exit(" The yaml file needs a `models:` list (name, path, nenm).")
    for m in models:
        if str(m['name']) not in ('GEFS','ECMWF','EnvCanada'):
            sys.exit(" Model "+str(m['name'])+" not supported. Use GEFS, ECMWF or EnvCanada.")
    label=str(wconfig.get('label','MULTI'))
    require_all=bool(wconfig.get('require_all_models',True))

    # Internal or External (mask tropics during hurricane season)
    mode=str(wconfig['mode'])
    # time resolution
    tres=wconfig['tres']
    # n-max expansion
    nmax=wconfig['nmax']
    # spatial window size (diameter), square (degrees)
    spws=wconfig['spws']
    # Plotting Area
    slonmin=wconfig['lonmin']-spws; slonmax=wconfig['lonmax']+spws
    slatmin=wconfig['latmin']-spws; slatmax=wconfig['latmax']+spws
    # Spatial percentile (grid points)
    spctl=wconfig['spctl']
    # gaussian filter spatial plot smoothing
    gft=wconfig['gft']
    # percentiles for the initial plots
    pctls=np.array(wconfig['pctls']).astype('int')
    # Probability levels
    plevels = np.array(wconfig['plevels']).astype('float')
    hplevels = np.array(wconfig['hplevels']).astype('float')
    # Colors for the probability maps
    pcolors = np.array(wconfig['pcolors']).astype('str')
    hpcolors = np.array(wconfig['hpcolors']).astype('str')

    # output path
    outpath=str(wconfig['outpath'])
    if outpath[-1] != '/':
        outpath=outpath+"/"

    umf=1. # unit conversion, when necessary

    # maximum value allowed (quick quality control), and levels for the probability plot
    if fvarname.upper() == "WS10" or fvarname.upper() == "WND" or fvarname.upper() == "U10":
        iswind=True
        qqvmax=wconfig['qqvmax_wnd']
        qlev=np.array(wconfig['qlev_wnd']).astype('float')
        vtickd=int(wconfig['vtickd_wnd'])
        funits=str('knots')
        umf=1.94 # m/s to knots
    elif fvarname.upper() == "HS":
        iswind=False
        qqvmax=wconfig['qqvmax_hs']
        funits=str('m')
        qlev=np.array(wconfig['qlev_hs']).astype('float')
        vtickd=int(wconfig['vtickd_hs'])
    else:
        sys.exit(" Input variable "+fvarname+" not included in the list. Please select only one: WS10, Hs.")

    print(" Reading yaml configuration file, OK."); print(" ")

    # Time range forecast intervall string, for the plots
    if (ltime2-ltime1)<0:
        aux=np.copy(ltime1); ltime1=np.copy(ltime2)
        ltime2=np.copy(aux); del aux

    if ltime1==1 and (ltime2-ltime1)==7:
        trfi=str("Week 1 - ")
    elif ltime1==7 and (ltime2-ltime1)==7:
        trfi=str("Week 2 - ")
    elif ltime1==14 and (ltime2-ltime1)==7:
        trfi=str("Week 3 - ")
    elif ltime1==21 and (ltime2-ltime1)==7:
        trfi=str("Week 4 - ")
    elif ltime1==28 and (ltime2-ltime1)==7:
        trfi=str("Week 5 - ")
    elif (ltime2-ltime1)>0:
        trfi=str("Days "+str(ltime1)+"-"+str(ltime2)+" , ")
    elif ltime2==ltime1:
        trfi=str("Day "+str(ltime1)+" , ")
    else:
        trfi=''

    print(" "); print(" 1. Reading Forecast Data ...")

    auxltime = np.arange((ltime1-1)*24,((ltime2)*24)+1,tres)
    auxltime[auxltime>384]=384; auxltime[auxltime<0]=0

    # Domain used for all models (same selection as probmaps.py)
    bounds=(slatmin-spws,slatmax+spws,slonmin-spws,slonmax+spws)

    # Read every model; the reference model goes first and defines the common grid
    fall=[]; used=[]; nmem=[]
    lat=None; lon=None; wtime=None
    for m in models:
        name=str(m['name']); mpath=str(m['path']); nenm=int(m['nenm'])
        if mpath[-1] != '/':
            mpath=mpath+"/"
        mvar=m.get('mvar_wnd') if iswind else m.get('mvar_hs')

        print(" "); print(" Model "+name+" ("+str(nenm)+" members) ...")
        try:
            if name=='GEFS':
                fmod, la, lo, wt = read_gefs(auxltime,fvarname,nenm,mpath,fcdate,fchour)
            else:
                if mvar is None:
                    sys.exit(" Model "+name+" needs mvar_hs / mvar_wnd in its `models:` entry.")
                fmod, la, lo, wt = read_mm(auxltime,name,fvarname,str(mvar),nenm,mpath,fcdate,fchour)
        except (Exception, SystemExit) as e:
            if require_all:
                sys.exit(" Model "+name+" could not be read: "+str(e))
            print(" WARNING: model "+name+" skipped: "+str(e))
            continue

        # Quick simple quality control
        fmod[fmod>=qqvmax]=np.nan; fmod[fmod<0.]=np.nan
        # Unit conversion
        fmod=fmod*umf

        # Domain of interest. The first model that is read defines the grid; the
        # others must have exactly the same points, otherwise the code stops.
        fmod, la, lo = crop_domain(fmod,la,lo,*bounds)
        if lat is None:
            lat=la; lon=lo; wtime=wt; name_ref=name
        else:
            check_grid(lat,lon,la,lo,name_ref,name)

        del la, lo
        fall.append(fmod); used.append(name); nmem.append(nenm)
        print(" Model "+name+": "+str(fmod.shape[0])+" time steps, "+str(nenm)+" members OK")
        del fmod

    if len(fall)==0:
        sys.exit(" No model could be read.")

    mtxt="+".join(used)
    print(" "); print(" Combined ensemble: "+mtxt+", "+str(sum(nmem))+" members")
    print(" 1. Forecast Data ... OK"); print(" ")

    fpname=label+"_"+ftag

    print(" 2. Initial Plots ...")
    pfields=pooled_percentiles(fall,pctls)
    # color scale of the percentile maps, from a decimated sample of all models
    sub=np.concatenate([a[:,:,::4,::4].ravel() for a in fall])
    wlevels=np.linspace(0,np.nanpercentile(sub,99.99),101)
    del sub
    pctl_plot(pfields,wlevels,pctls,lat,lon,slonmin,slonmax,slatmin,slatmax,spws,outpath,fpname,mtxt,ftag,fvarname,funits,fcycle,trfi,wtime,vtickd,ltime1,ltime2)
    del pfields

    print(" "); print(" 3. Space-Time Cells and Probabilities ...")
    # n-max expansion per model, then append all members into one ensemble
    fexp=np.concatenate([nmax_expand(a,nmax) for a in fall],axis=0)
    del fall
    print(" Samples: "+str(fexp.shape[0])+" = "+str(nmax)+" x "+str(sum(nmem))+" members")
    probecdf = stprob(fexp,lat,lon,spws,qlev,spctl)
    print(" 3. Space-Time Cells and Probabilities ... OK"); print(" ")

    # MAIN PLOTS
    print(" 4. Probability Maps ...")
    pm_plot(probecdf,wtime,wconfig,mode,spws,plevels,qlev,hplevels,hpcolors,pcolors,lat,lon,slonmin,slonmax,slatmin,slatmax,fvarname,funits,fpname,mtxt,fcycle,outpath,ltime1,ltime2,gft,trfi)
    print(" 4. Probability Plots ... OK"); print(" ")


