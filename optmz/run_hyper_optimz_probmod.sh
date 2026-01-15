#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -q batch
#SBATCH -t 03:50:00
#SBATCH -A marine-cpu
#SBATCH -p orion

# This job run the code hyper_optimz_probmod.py on Orion

ulimit -s unlimited
ulimit -c 0

export SPWS=${SPWS}

DIRSCRIPTS="/work/noaa/marine/ricardo.campos/work/analysis/4postproc"

echo " Starting at "$(date +"%T")

# python env
source /work/noaa/marine/ricardo.campos/progs/python/setanaconda3.sh
sh /work/noaa/marine/ricardo.campos/progs/python/setanaconda3.sh

# work dir
cd ${DIRSCRIPTS}

# run
/work/noaa/marine/ricardo.campos/progs/python/anaconda3/bin/python3 ${DIRSCRIPTS}/hyper_optimz_probmod.py ${SPWS}
wait $!
sleep 1

echo " Complete at "$(date +"%T")

