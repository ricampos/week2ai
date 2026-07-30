#!/bin/bash

##########################
# probmaps_envcanada.sh 
# based on probmaps_gefs.sh, adapted for EnvCanada (CMCE) data
##########################

set -euo pipefail
export USER_IS_ROOT=0
export MODULEPATH=/etc/scl/modulefiles:/apps/lmod/lmod/modulefiles/Core:/apps/modules/modulefiles/Linux:/apps/modules/modulefiles
source /apps/lmod/lmod/init/bash
module load cdo
module load nco

# INPUT ARGUMENT
# Forecast Cycle (00, 12)
HOUR="$1"
# Days into the past. pa=1 runs using yesterday's cycle
pa="$2"
# .yaml configuration file containing paths and information for this
#   shell script as well as for the python code.
PYCYAML="$3"
# PYCYAML="/scratch4/AOML/aoml-phod/Ricardo.Campos/week2_multimodel/probmaps_envcanada.yaml"

# Read the YAML as a text file:
#  Ensemble data path
mpath_line=$(grep 'mpath' "${PYCYAML}")
MDIR=$(echo "$mpath_line" | awk -F': ' '{print $2}')
#  Python script (probability maps)
pyscript_line=$(grep 'pyscript' "${PYCYAML}")
PYSCRIPT=$(echo "$pyscript_line" | awk -F': ' '{print $2}')
#  Variable names, for the python processing (probability maps)
mvars_line=$(grep 'mvars' "${PYCYAML}")
MVARS=$(echo "$mvars_line" | awk -F': ' '{gsub(/"/, "", $2); print $2}')
#  Output path
outpath_line=$(grep 'outpath' "${PYCYAML}")
OUTPATH=$(echo "$outpath_line" | awk -F': ' '{print $2}')

# Date / Forecast cycle
YEAR=`date --date=-$pa' day' '+%Y'`
MONTH=`date --date=-$pa' day' '+%m'`
DAY=`date --date=-$pa' day' '+%d'`

echo " "
echo " Looking for EnvCanada cycle ${YEAR}${MONTH}${DAY}${HOUR} ..."
echo " Expected file: $MDIR/wave/$HOUR/GEWPS_wave_$YEAR$MONTH$DAY$HOUR.20.nc"
echo " "
# Check ensemble is complete and ready.
# If not, it waits for 5 min and then try again (max 12 hours)
FSIZE=0
TRIES=1
while [ "$FSIZE" -lt 300000000 ] && [ "$TRIES" -le 144 ]; do
  # wait 5 minutes until next try
  if [ ${TRIES} -gt 5 ]; then
    sleep 300
  fi
  # Check if the last file is complete. Wave (Hs) is the most important variable
  if [ -f "$MDIR/wave/$HOUR/GEWPS_wave_$YEAR$MONTH$DAY$HOUR.20.nc" ]; then
    TE=0
  else
    TE=1
  fi
  if [ ${TE} -eq 1 ]; then
    FSIZE=0
  else
    FSIZE=$(du -sb "$MDIR/wave/$HOUR/GEWPS_wave_$YEAR$MONTH$DAY$HOUR.20.nc" | awk '{print $1}')
  fi
  echo "  Try ${TRIES}/144: file $([ ${TE} -eq 0 ] && echo "found (${FSIZE} bytes)" || echo "not found yet")."
  TRIES=`expr $TRIES + 1`
done

if [ "$FSIZE" -lt 300000000 ]; then
  echo " "
  echo " WARNING: gave up after ${TRIES} tries (~12h) - EnvCanada file never reached expected size."
  echo " Proceeding anyway, but downstream Python processing may fail or use incomplete data."
  echo " "
fi

# Module load python and activate environment when necessary.
source /home/Ricardo.Campos/python/envs/intelpy_env/bin/activate
echo "  "
echo " PYTHON PROCESSING: GLOBAL HAZARDS OUTLOOK - PROBABILITY MAPS, $YEAR$MONTH$DAY$HOUR "
echo "  "
# loop through variables
for WW3VAR in ${MVARS[*]}; do
  # 7 14 is the time intervall (days) for week 2
  python3 ${PYSCRIPT} ${PYCYAML} $YEAR$MONTH$DAY$HOUR 7 14 ${WW3VAR}
  echo " Probability maps for ${WW3VAR} Ok." 
done

echo "  "
echo " PYTHON PROCESSING COMPLETE."
# ----

cd ${OUTPATH}
mkdir -p $YEAR$MONTH$DAY$HOUR
mkdir -p $YEAR$MONTH$DAY$HOUR/Hs
mkdir -p $YEAR$MONTH$DAY$HOUR/WS10
mv *Hs* $YEAR$MONTH$DAY$HOUR/Hs/
mv *WS10* $YEAR$MONTH$DAY$HOUR/WS10/

