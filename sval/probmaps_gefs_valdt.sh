#!/bin/bash

########################################################################
# probmaps_gefs_valdt.sh
#
# VERSION AND LAST UPDATE:
#   v1.0  11/12/2025
#   v2.0  02/18/2026
#   v3.0  09/08/2026
#
# PURPOSE:
#  Automated spatial validation of the week-2 GEFS probabilistic forecast
#
# USAGE:
#  The first (and only) argument is the .yaml configuration file, same
#   probmaps_gefs*.yaml used operationally
#
#  Example:
#    bash probmaps_gefs_valdt.sh /scratch4/AOML/aoml-phod/Ricardo.Campos/week2_sval/probmaps_gefs.yaml
#
# OUTPUT:
#  Probability Map validation figures saved in outpath/<validated cycle>/<var>/
#
# DEPENDENCIES:
#  probmaps_gefs_valdt.py contains the module dependencies.
#
# AUTHOR and DATE:
#  11/12/2025: Ricardo M. Campos, first version.
#  02/18/2026: Ricardo M. Campos, improvement in the spatial validation.
#  09/08/2026: Ricardo M. Campos, fixed bugs, readiness check, and retention cutoff
#
# PERSON OF CONTACT:
#  Ricardo M. Campos: ricardo.campos@noaa.gov
#
########################################################################

set -euo pipefail
export USER_IS_ROOT=0
export MODULEPATH=/etc/scl/modulefiles:/apps/lmod/lmod/modulefiles/Core:/apps/modules/modulefiles/Linux:/apps/modules/modulefiles
source /apps/lmod/lmod/init/bash
module load cdo
module load nco

# INPUT ARGUMENT
# .yaml configuration file containing paths and information for this
#   shell script as well as for the python code.
PYCYAML="$1"

# Fixed validation window (week 2), matching operational probmaps.sh
LTIME1=7
LTIME2=14
# Cycle step to build reference
CYCLESTEP=12

# Read the YAML as a text file:
gefspath_line=$(grep 'gefspath' "${PYCYAML}")
GEFSARCHIVE=$(echo "$gefspath_line" | awk -F': ' '{print $2}')
pyscript_line=$(grep 'pyscript' "${PYCYAML}")
PYSCRIPT=$(echo "$pyscript_line" | awk -F': ' '{print $2}')
mvars_line=$(grep 'mvars' "${PYCYAML}")
MVARS=$(echo "$mvars_line" | awk -F': ' '{gsub(/"/, "", $2); print $2}')
outpath_line=$(grep 'outpath' "${PYCYAML}")
OUTPATH=$(echo "$outpath_line" | awk -F': ' '{print $2}')

# Forecast cycle being validated
pa=1 # expand the shift time
HOUR="00"
YEAR=$(date --date="-$((${LTIME2}+${pa})) day" '+%d')
MONTH=$(date --date="-$((${LTIME2}+${pa})) day" '+%d')
DAY=$(date --date="-$((${LTIME2}+${pa})) day" '+%d')
DAY=$(date --date="-$((${LTIME2}+${pa})) day" '+%d')
FCYCLE="${YEAR}${MONTH}${DAY}${HOUR}"

# Last archive cycle the truth window needs: fcycle + LTIME2 days, 12Z
# (the last 12h-cadence cycle inside/closing the window)
LASTCYCLE_DATE=$(date --date="-$((pa - LTIME2))"' day' '+%Y%m%d')
LASTCYCLE_HOUR="12"
LASTFILE="$GEFSARCHIVE/GEFSv12Waves_${LASTCYCLE_DATE}${LASTCYCLE_HOUR}/gefs.wave.${LASTCYCLE_DATE}.00.global.0p25.f012.grib2"

echo " "
echo " Validating GEFS cycle ${FCYCLE} (week 2: day ${LTIME1}-${LTIME2})."
echo " Waiting on last truth-window archive cycle: GEFSv12Waves_${LASTCYCLE_DATE}${LASTCYCLE_HOUR}"
echo " Expected file: $LASTFILE"
echo " "

# Check the last needed archive cycle is complete and ready.
# If not, wait 5 min and try again (max 12 hours).
FSIZE=0
TRIES=1
while [ "$FSIZE" -lt 1000000 ] && [ "$TRIES" -le 144 ]; do
  if [ ${TRIES} -gt 5 ]; then
    sleep 300
  fi
  if [ -f "$LASTFILE" ]; then
    FSIZE=$(du -sb "$LASTFILE" | awk '{print $1}')
  else
    FSIZE=0
  fi
  echo "  Try ${TRIES}/144: file $([ "$FSIZE" -gt 0 ] && echo "found (${FSIZE} bytes)" || echo "not found yet")."
  TRIES=$((TRIES + 1))
done

if [ "$FSIZE" -lt 1000000 ]; then
  echo " "
  echo " WARNING: gave up after ${TRIES} tries (~12h) - last truth-window archive cycle never reached expected size."
  echo " Proceeding anyway, but the ground truth may be incomplete for the most recent part of the window."
  echo " "
fi

# Module load python and activate environment when necessary.
source /home/Ricardo.Campos/python/envs/intelpy_env/bin/activate
# rm -f "$GEFSARCHIVE"/GEFSv12Waves_*/*.idx || true

echo "  "
echo " PYTHON PROCESSING: GLOBAL HAZARDS OUTLOOK (VALIDATION) - PROBABILITY MAPS, ${FCYCLE} "
echo "  "
for WW3VAR in ${MVARS[*]}; do
  python3 "${PYSCRIPT}" "${PYCYAML}" "${FCYCLE}" "${LTIME1}" "${LTIME2}" "${WW3VAR}"
  echo " Probability maps for ${WW3VAR} Ok."
done
echo "  "
echo " PYTHON PROCESSING COMPLETE."
# ----

cd "${OUTPATH}"
mkdir -p "${YEAR}${MONTH}${DAY}${HOUR}"
mkdir -p "${YEAR}${MONTH}${DAY}${HOUR}/Hs"
mkdir -p "${YEAR}${MONTH}${DAY}${HOUR}/WS10"
mv -f *Hs* "${YEAR}${MONTH}${DAY}${HOUR}/Hs/" 2>/dev/null || true
mv -f *WS10* "${YEAR}${MONTH}${DAY}${HOUR}/WS10/" 2>/dev/null || true

# Retention: never delete archive cycles that a validation run still needs.
RETAIN_BUFFER=3
CUTOFF=$(date -d "$((pa + LTIME2 + RETAIN_BUFFER)) days ago" +%Y%m%d)
for dir in "$GEFSARCHIVE"/GEFSv12Waves_*; do
    [ -d "$dir" ] || continue
    BASENAME=$(basename "$dir")
    DIR_DATE=${BASENAME#GEFSv12Waves_}
    DIR_DATE=${DIR_DATE:0:8}
    if [[ $DIR_DATE =~ ^[0-9]{8}$ ]]; then
        if [[ $DIR_DATE -lt $CUTOFF ]]; then
            echo "Deleting $dir (older than retention cutoff $CUTOFF)"
            rm -rf "$dir"
        fi
    fi
done

echo "  "
echo " Done."
echo "  "

