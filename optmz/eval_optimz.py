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

    opath="/work/noaa/marine/ricardo.campos/work/analysis/4postproc/output/optmz/pickle"

    qlev_hs = np.array([ 4.,  6.,  9.]).astype('float')
    qlev_wnd = np.array([28.0, 34.0, 41.0, 48.0]).astype('float')/1.94
    plevels = np.array([0.15, 0.5, 0.65, 0.8, 1.0])

    values_spws = np.array([0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0]).astype('float')
    values_nmax = np.array([1,2,3,4]).astype('int')
    values_spctl = np.array([80,83,85,87,89,91,93,95,98]).astype('int')

    c=0
    for i in range(0,len(values_spws)):
        for j in range(0,len(values_nmax)):
            for k in range(0,len(values_spctl)):

                spws = values_spws[i]
                nmax = values_nmax[j]
                spctl = values_spctl[k]

                fname = opath+"/Optmz_nmax"+str(int(nmax))+"_spws"+str(int(spws*100)).zfill(3)+"_spctl"+str(int(spctl))+"_STAT.RESULTS.pkl"
                with open(fname,'rb') as f:
                    stat_results = pickle.load(f)

                if c==0:

                    POD_hs = np.zeros((len(values_spws),len(values_nmax),len(values_spctl),len(qlev_hs),len(plevels)-1),'f')*np.nan
                    FAR_hs = np.zeros((len(values_spws),len(values_nmax),len(values_spctl),len(qlev_hs),len(plevels)-1),'f')*np.nan
                    CSI_hs = np.zeros((len(values_spws),len(values_nmax),len(values_spctl),len(qlev_hs),len(plevels)-1),'f')*np.nan
                    POD_u10 = np.zeros((len(values_spws),len(values_nmax),len(values_spctl),len(qlev_wnd),len(plevels)-1),'f')*np.nan
                    FAR_u10 = np.zeros((len(values_spws),len(values_nmax),len(values_spctl),len(qlev_wnd),len(plevels)-1),'f')*np.nan
                    CSI_u10 = np.zeros((len(values_spws),len(values_nmax),len(values_spctl),len(qlev_wnd),len(plevels)-1),'f')*np.nan
                    ROCauc_hs = np.zeros((len(values_spws),len(values_nmax),len(values_spctl),len(qlev_hs)),'f')*np.nan
                    ROCauc_u10 = np.zeros((len(values_spws),len(values_nmax),len(values_spctl),len(qlev_wnd)),'f')*np.nan
                    BRIER_hs = np.zeros((len(values_spws),len(values_nmax),len(values_spctl),len(qlev_hs)),'f')*np.nan
                    BRIER_u10 = np.zeros((len(values_spws),len(values_nmax),len(values_spctl),len(qlev_wnd)),'f')*np.nan
                    CRPS_hs = np.zeros((len(values_spws),len(values_nmax),len(values_spctl)),'f')*np.nan
                    CRPS_u10 = np.zeros((len(values_spws),len(values_nmax),len(values_spctl)),'f')*np.nan


                POD_hs[i,j,k,:,:] = stat_results['ceval_gefs_hs'][:,:,0]
                FAR_hs[i,j,k,:,:] = stat_results['ceval_gefs_hs'][:,:,1] 
                CSI_hs[i,j,k,:,:] = stat_results['ceval_gefs_hs'][:,:,2]

                POD_u10[i,j,k,:,:] = stat_results['ceval_gefs_u10'][:,:,0]
                FAR_u10[i,j,k,:,:] = stat_results['ceval_gefs_u10'][:,:,1] 
                CSI_u10[i,j,k,:,:] = stat_results['ceval_gefs_u10'][:,:,2]

                ROCauc_hs[i,j,k,:] = stat_results['froc_hs'][:]
                ROCauc_u10[i,j,k,:] = stat_results['froc_u10'][:]

                BRIER_hs[i,j,k,:] = stat_results['fbriers_hs'][:]
                BRIER_u10[i,j,k,:] = stat_results['fbriers_u10'][:]

                CRPS_hs[i,j,k] = stat_results['mean_crps_hs']
                CRPS_u10[i,j,k] = stat_results['mean_crps_u10']

                c=c+1
                print(" Read "+fname); print(" "+repr(c)); print(" ")


    # ----- Search for optimal parameters:

    linestyle = np.array(['--','-.',':','--','-.',':','--','-.',':',])
    colors = np.array(['gold','lime','green','blue','navy','orange','red','firebrick','fuchsia'])

    # ============ U10 =====================
    #  ------------ POD ------------ 
    for i in range(0,len(qlev_wnd)):
        for j in range(0,len(plevels)-1):

            fig1, ax = plt.subplots(figsize=(6, 4))

            for k in range(0,len(values_spctl)):
                flabel = "PCTL"+repr(values_spctl[k])
                ax.plot(values_spws, POD_u10[:,1,k,i,j], color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
                ax.plot(values_spws, POD_u10[:,1,k,i,j], color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

            ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
            ax.set_ylabel("POD")
            ax.set_xlabel("SPWS (°)")
            ax.legend(fontsize=sl - 3)
            plt.tight_layout()
            plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
            fname = "Validation_POD_qvel"+repr(int(qlev_wnd[i]))+"_plevel"+repr(plevels[j])+".png"
            plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close(fig1)

    for i in range(0,len(qlev_wnd)):
        fig1, ax = plt.subplots(figsize=(6, 4))
        for k in range(0,len(values_spctl)):
            flabel = "PCTL"+repr(values_spctl[k])
            ax.plot(values_spws, np.nanmean(POD_u10[:,1,k,i,:],axis=1), color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
            ax.plot(values_spws, np.nanmean(POD_u10[:,1,k,i,:],axis=1), color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

        ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
        ax.set_ylabel("POD")
        ax.set_xlabel("SPWS (°)")
        ax.legend(fontsize=sl - 3)
        plt.tight_layout()
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
        fname = "Validation_POD_qvel"+repr(int(qlev_wnd[i]))+"_plevelMEAN.png"
        plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig1)

    # Best config
    fPOD_u10 = np.nanmean(POD_u10[:,:,:,:,:],axis=(3,4))  
    ind_fPOD_u10 = np.where(fPOD_u10 == np.nanmax(np.nanmean(POD_u10[:,:,:,:,:],axis=(3,4))))


    #  ------------ FAR ------------ 
    for i in range(0,len(qlev_wnd)):
        for j in range(0,len(plevels)-1):

            fig1, ax = plt.subplots(figsize=(6, 4))

            for k in range(0,len(values_spctl)):
                flabel = "PCTL"+repr(values_spctl[k])
                ax.plot(values_spws, FAR_u10[:,1,k,i,j], color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
                ax.plot(values_spws, FAR_u10[:,1,k,i,j], color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

            ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
            ax.set_ylabel("FAR")
            ax.set_xlabel("SPWS (°)")
            ax.legend(fontsize=sl - 3)
            plt.tight_layout()
            plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
            fname = "Validation_FAR_qvel"+repr(int(qlev_wnd[i]))+"_plevel"+repr(plevels[j])+".png"
            plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close(fig1)

    for i in range(0,len(qlev_wnd)):
        fig1, ax = plt.subplots(figsize=(6, 4))
        for k in range(0,len(values_spctl)):
            flabel = "PCTL"+repr(values_spctl[k])
            ax.plot(values_spws, np.nanmean(FAR_u10[:,1,k,i,:],axis=1), color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
            ax.plot(values_spws, np.nanmean(FAR_u10[:,1,k,i,:],axis=1), color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

        ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
        ax.set_ylabel("FAR")
        ax.set_xlabel("SPWS (°)")
        ax.legend(fontsize=sl - 3)
        plt.tight_layout()
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
        fname = "Validation_FAR_qvel"+repr(int(qlev_wnd[i]))+"_plevelMEAN.png"
        plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig1)

    # Best config
    fFAR_u10 = np.nanmean(FAR_u10[:,:,:,:,:],axis=(3,4))  
    ind_fFAR_u10 = np.where(fFAR_u10 == np.nanmin(np.nanmean(FAR_u10[:,:,:,:,:],axis=(3,4))))


    # ------------ CSI ------------ 
    for i in range(0,len(qlev_wnd)):
        for j in range(0,len(plevels)-1):

            fig1, ax = plt.subplots(figsize=(6, 4))

            for k in range(0,len(values_spctl)):
                flabel = "PCTL"+repr(values_spctl[k])
                ax.plot(values_spws, CSI_u10[:,1,k,i,j], color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
                ax.plot(values_spws, CSI_u10[:,1,k,i,j], color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

            ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
            ax.set_ylabel("CSI")
            ax.set_xlabel("SPWS (°)")
            ax.legend(fontsize=sl - 3)
            plt.tight_layout()
            plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
            fname = "Validation_CSI_qvel"+repr(int(qlev_wnd[i]))+"_plevel"+repr(plevels[j])+".png"
            plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close(fig1)

    for i in range(0,len(qlev_wnd)):
        fig1, ax = plt.subplots(figsize=(6, 4))
        for k in range(0,len(values_spctl)):
            flabel = "PCTL"+repr(values_spctl[k])
            ax.plot(values_spws, np.nanmean(CSI_u10[:,1,k,i,:],axis=1), color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
            ax.plot(values_spws, np.nanmean(CSI_u10[:,1,k,i,:],axis=1), color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

        ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
        ax.set_ylabel("CSI")
        ax.set_xlabel("SPWS (°)")
        ax.legend(fontsize=sl - 3)
        plt.tight_layout()
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
        fname = "Validation_CSI_qvel"+repr(int(qlev_wnd[i]))+"_plevelMEAN.png"
        plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig1)

    # Best config
    fCSI_u10 = np.nanmean(CSI_u10[:,:,:,:,:],axis=(3,4))  
    ind_fCSI_u10 = np.where(fCSI_u10 == np.nanmax(np.nanmean(CSI_u10[:,:,:,:,:],axis=(3,4))))

    # AUC ROC
    for i in range(0,len(qlev_wnd)):
        fig1, ax = plt.subplots(figsize=(6, 4))
        for k in range(0,len(values_spctl)):
            flabel = "PCTL"+repr(values_spctl[k])
            ax.plot(values_spws, ROCauc_u10[:,1,k,i], color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
            ax.plot(values_spws, ROCauc_u10[:,1,k,i], color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

        ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
        ax.set_ylabel("ROC AUC")
        ax.set_xlabel("SPWS (°)")
        ax.legend(fontsize=sl - 3)
        plt.tight_layout()
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
        fname = "Validation_ROCauc_qvel"+repr(int(qlev_wnd[i]))+".png"
        plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig1)

    # Best config
    fROCauc_u10 = np.nanmean(ROCauc_u10[:,:,:,:],axis=3)
    ind_fROCauc_u10 = np.where(fROCauc_u10 == np.nanmax(np.nanmean(ROCauc_u10[:,:,:,:],axis=3)) )

    # BRIER
    for i in range(0,len(qlev_wnd)):
        fig1, ax = plt.subplots(figsize=(6, 4))
        for k in range(0,len(values_spctl)):
            flabel = "PCTL"+repr(values_spctl[k])
            ax.plot(values_spws, BRIER_u10[:,1,k,i], color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
            ax.plot(values_spws, BRIER_u10[:,1,k,i], color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

        ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
        ax.set_ylabel("Brier Score")
        ax.set_xlabel("SPWS (°)")
        ax.legend(fontsize=sl - 3)
        plt.tight_layout()
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
        fname = "Validation_BRIER_qvel"+repr(int(qlev_wnd[i]))+".png"
        plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig1)

    fBRIER_u10 = np.nanmean(BRIER_u10[:,:,:,:],axis=3)
    ind_fBRIER_u10 = np.where(fBRIER_u10 == np.nanmin(np.nanmean(BRIER_u10[:,:,:,:],axis=3)) )

    # CRPS
    fig1, ax = plt.subplots(figsize=(6, 4))
    for k in range(0,len(values_spctl)):
        flabel = "PCTL"+repr(values_spctl[k])
        ax.plot(values_spws, CRPS_u10[:,1,k], color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
        ax.plot(values_spws, CRPS_u10[:,1,k], color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

    ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
    ax.set_ylabel("CRPS")
    ax.set_xlabel("SPWS (°)")
    ax.legend(fontsize=sl - 3)
    plt.tight_layout()
    plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
    fname = "Validation_CRPS.png"
    plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1)

    ind_fCRPS_u10 = np.where(CRPS_u10 == np.nanmin(CRPS_u10[:,:,:]) )

    # -------------------------------


    # ============ Hs =====================
    #  ------------ POD ------------ 
    for i in range(0,len(qlev_hs)):
        for j in range(0,len(plevels)-1):

            fig1, ax = plt.subplots(figsize=(6, 4))

            for k in range(0,len(values_spctl)):
                flabel = "PCTL"+repr(values_spctl[k])
                ax.plot(values_spws, POD_hs[:,1,k,i,j], color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
                ax.plot(values_spws, POD_hs[:,1,k,i,j], color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

            ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
            ax.set_ylabel("POD")
            ax.set_xlabel("SPWS (°)")
            ax.legend(fontsize=sl - 3)
            plt.tight_layout()
            plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
            fname = "Validation_POD_qvel"+repr(int(qlev_hs[i]))+"_plevel"+repr(plevels[j])+".png"
            plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close(fig1)

    for i in range(0,len(qlev_hs)):
        fig1, ax = plt.subplots(figsize=(6, 4))
        for k in range(0,len(values_spctl)):
            flabel = "PCTL"+repr(values_spctl[k])
            ax.plot(values_spws, np.nanmean(POD_hs[:,1,k,i,:],axis=1), color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
            ax.plot(values_spws, np.nanmean(POD_hs[:,1,k,i,:],axis=1), color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

        ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
        ax.set_ylabel("POD")
        ax.set_xlabel("SPWS (°)")
        ax.legend(fontsize=sl - 3)
        plt.tight_layout()
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
        fname = "Validation_POD_qvel"+repr(int(qlev_hs[i]))+"_plevelMEAN.png"
        plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig1)

    # Best config
    fPOD_hs = np.nanmean(POD_hs[:,:,:,:,:],axis=(3,4))  
    ind_fPOD_hs = np.where(fPOD_hs == np.nanmax(np.nanmean(POD_hs[:,:,:,:,:],axis=(3,4))))


    #  ------------ FAR ------------ 
    for i in range(0,len(qlev_hs)):
        for j in range(0,len(plevels)-1):

            fig1, ax = plt.subplots(figsize=(6, 4))

            for k in range(0,len(values_spctl)):
                flabel = "PCTL"+repr(values_spctl[k])
                ax.plot(values_spws, FAR_hs[:,1,k,i,j], color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
                ax.plot(values_spws, FAR_hs[:,1,k,i,j], color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

            ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
            ax.set_ylabel("FAR")
            ax.set_xlabel("SPWS (°)")
            ax.legend(fontsize=sl - 3)
            plt.tight_layout()
            plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
            fname = "Validation_FAR_qvel"+repr(int(qlev_hs[i]))+"_plevel"+repr(plevels[j])+".png"
            plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close(fig1)

    for i in range(0,len(qlev_hs)):
        fig1, ax = plt.subplots(figsize=(6, 4))
        for k in range(0,len(values_spctl)):
            flabel = "PCTL"+repr(values_spctl[k])
            ax.plot(values_spws, np.nanmean(FAR_hs[:,1,k,i,:],axis=1), color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
            ax.plot(values_spws, np.nanmean(FAR_hs[:,1,k,i,:],axis=1), color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

        ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
        ax.set_ylabel("FAR")
        ax.set_xlabel("SPWS (°)")
        ax.legend(fontsize=sl - 3)
        plt.tight_layout()
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
        fname = "Validation_FAR_qvel"+repr(int(qlev_hs[i]))+"_plevelMEAN.png"
        plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig1)

    # Best config
    fFAR_hs = np.nanmean(FAR_hs[:,:,:,:,:],axis=(3,4))  
    ind_fFAR_hs = np.where(fFAR_hs == np.nanmin(np.nanmean(FAR_hs[:,:,:,:,:],axis=(3,4))))


    # ------------ CSI ------------ 
    for i in range(0,len(qlev_hs)):
        for j in range(0,len(plevels)-1):

            fig1, ax = plt.subplots(figsize=(6, 4))

            for k in range(0,len(values_spctl)):
                flabel = "PCTL"+repr(values_spctl[k])
                ax.plot(values_spws, CSI_hs[:,1,k,i,j], color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
                ax.plot(values_spws, CSI_hs[:,1,k,i,j], color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

            ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
            ax.set_ylabel("CSI")
            ax.set_xlabel("SPWS (°)")
            ax.legend(fontsize=sl - 3)
            plt.tight_layout()
            plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
            fname = "Validation_CSI_qvel"+repr(int(qlev_hs[i]))+"_plevel"+repr(plevels[j])+".png"
            plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close(fig1)

    for i in range(0,len(qlev_hs)):
        fig1, ax = plt.subplots(figsize=(6, 4))
        for k in range(0,len(values_spctl)):
            flabel = "PCTL"+repr(values_spctl[k])
            ax.plot(values_spws, np.nanmean(CSI_hs[:,1,k,i,:],axis=1), color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
            ax.plot(values_spws, np.nanmean(CSI_hs[:,1,k,i,:],axis=1), color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

        ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
        ax.set_ylabel("CSI")
        ax.set_xlabel("SPWS (°)")
        ax.legend(fontsize=sl - 3)
        plt.tight_layout()
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
        fname = "Validation_CSI_qvel"+repr(int(qlev_hs[i]))+"_plevelMEAN.png"
        plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig1)

    # Best config
    fCSI_hs = np.nanmean(CSI_hs[:,:,:,:,:],axis=(3,4))  
    ind_fCSI_hs = np.where(fCSI_hs == np.nanmax(np.nanmean(CSI_hs[:,:,:,:,:],axis=(3,4))))

    # AUC ROC
    for i in range(0,len(qlev_hs)):
        fig1, ax = plt.subplots(figsize=(6, 4))
        for k in range(0,len(values_spctl)):
            flabel = "PCTL"+repr(values_spctl[k])
            ax.plot(values_spws, ROCauc_hs[:,1,k,i], color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
            ax.plot(values_spws, ROCauc_hs[:,1,k,i], color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

        ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
        ax.set_ylabel("ROC AUC")
        ax.set_xlabel("SPWS (°)")
        ax.legend(fontsize=sl - 3)
        plt.tight_layout()
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
        fname = "Validation_ROCauc_qvel"+repr(int(qlev_hs[i]))+".png"
        plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig1)

    # Best config
    fROCauc_hs = np.nanmean(ROCauc_hs[:,:,:,:],axis=3)
    ind_fROCauc_hs = np.where(fROCauc_hs == np.nanmax(np.nanmean(ROCauc_hs[:,:,:,:],axis=3)) )

    # BRIER
    for i in range(0,len(qlev_hs)):
        fig1, ax = plt.subplots(figsize=(6, 4))
        for k in range(0,len(values_spctl)):
            flabel = "PCTL"+repr(values_spctl[k])
            ax.plot(values_spws, BRIER_hs[:,1,k,i], color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
            ax.plot(values_spws, BRIER_hs[:,1,k,i], color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

        ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
        ax.set_ylabel("Brier Score")
        ax.set_xlabel("SPWS (°)")
        ax.legend(fontsize=sl - 3)
        plt.tight_layout()
        plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
        fname = "Validation_BRIER_qvel"+repr(int(qlev_hs[i]))+".png"
        plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig1)

    fBRIER_hs = np.nanmean(BRIER_hs[:,:,:,:],axis=3)
    ind_fBRIER_hs = np.where(fBRIER_hs == np.nanmin(np.nanmean(BRIER_hs[:,:,:,:],axis=3)) )

    # CRPS
    fig1, ax = plt.subplots(figsize=(6, 4))
    for k in range(0,len(values_spctl)):
        flabel = "PCTL"+repr(values_spctl[k])
        ax.plot(values_spws, CRPS_hs[:,1,k], color=colors[k], marker='.', linestyle='', linewidth=2., zorder=2)
        ax.plot(values_spws, CRPS_hs[:,1,k], color=colors[k], linestyle=linestyle[k], linewidth=2., label=flabel, alpha=0.8, zorder=3)

    ax.set_xlim(values_spws[0]-0.1, values_spws[-1]+0.1)
    ax.set_ylabel("CRPS")
    ax.set_xlabel("SPWS (°)")
    ax.legend(fontsize=sl - 3)
    plt.tight_layout()
    plt.grid(c='grey', ls='--', alpha=0.3, zorder=1)
    fname = "Validation_CRPS.png"
    plt.savefig(fname, dpi=200, facecolor='w', edgecolor='w', orientation='portrait', format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1)

    ind_fCRPS_hs = np.where(CRPS_hs == np.nanmin(CRPS_hs[:,:,:]) )

    # -------------------------------

