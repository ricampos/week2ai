#!/bin/bash

# 288 simulations

DIRJOUT="/work/noaa/marine/ricardo.campos/work/analysis/4postproc/jobs"
DIRSCRIPTS="/work/noaa/marine/ricardo.campos/work/analysis/4postproc"

cd ${DIRSCRIPTS}

spws_values="0.5 1.0 1.5 2.0 2.5 3.0 4.0 5.0"

for SPWS in $spws_values; do
  export SPWS=${SPWS}
  sbatch --output=${DIRJOUT}"/jrun_hyper_optimz_probmod_SPWS"${SPWS}".out" ${DIRSCRIPTS}"/run_hyper_optimz_probmod.sh"
  echo " job jrun_hyper_optimz_probmod_SPWS"${SPWS}" submitted OK at "$(date +"%T")
  sleep 2
done

