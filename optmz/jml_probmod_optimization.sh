#!/bin/bash --login
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=20
#SBATCH -q batch
#SBATCH -t 07:50:00
#SBATCH -A marine-cpu
#SBATCH -p orion

# This job script runs the code ml_probmod_optimization.py on Orion
#
# DIRJOUT="/work/noaa/marine/ricardo.campos/work/analysis/4postproc/jobs"
# sbatch --output=${DIRJOUT}/jml_probmod_optimization_XG.out jml_probmod_optimization.sh

ulimit -s unlimited
ulimit -c 0

DIRSCRIPTS="/work/noaa/marine/ricardo.campos/work/analysis/4postproc"

# python env
source /work/noaa/marine/ricardo.campos/progs/python/setanaconda3.sh
sh /work/noaa/marine/ricardo.campos/progs/python/setanaconda3.sh
# work dir
cd ${DIRSCRIPTS}
# run
/work/noaa/marine/ricardo.campos/progs/python/anaconda3/bin/python3 ${DIRSCRIPTS}/ml_probmod_optimization.py 2

wait $!
echo " Complete at "$(date +"%T")

