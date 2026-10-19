#!/bin/bash
set -euo pipefail

export USER_IS_ROOT=0
export MODULEPATH=/etc/scl/modulefiles:/apps/lmod/lmod/modulefiles/Core:/apps/modules/modulefiles/Linux:/apps/modules/modulefiles
source /apps/lmod/lmod/init/bash
module load cdo
module load nco

# INPUT ARGUMENT
# .yaml configuration file containing paths and model information.
PYCYAML="$1"

# Fixed validation window (week 2)
LTIME1=7
LTIME2=14

# Read YAML configuration options cleanly (stripping quotes, spaces, and carriage returns)
MODEL=$(grep 'model' "${PYCYAML}" | head -n 1 | awk -F':' '{print $2}' | tr -d ' "' | tr -d '\r')
ARCHIVE_PATH=$(grep 'mpath' "${PYCYAML}" | head -n 1 | awk -F':' '{print $2}' | tr -d ' "' | tr -d '\r')
PYSCRIPT=$(grep 'pyscript' "${PYCYAML}" | head -n 1 | awk -F':' '{print $2}' | tr -d ' "' | tr -d '\r')
MVARS=$(grep 'mvars' "${PYCYAML}" | head -n 1 | awk -F':' '{print $2}' | tr -d '"[]\r' | xargs)
OUTPATH=$(grep 'outpath' "${PYCYAML}" | head -n 1 | awk -F':' '{print $2}' | tr -d ' "' | tr -d '\r')

# Forecast cycle being validated
pa=1 # shift time buffer
HOUR="00"
YEAR=$(date --date="-$((LTIME2+pa)) day" '+%Y')
MONTH=$(date --date="-$((LTIME2+pa)) day" '+%m')
DAY=$(date --date="-$((LTIME2+pa)) day" '+%d')
FCYCLE="${YEAR}${MONTH}${DAY}${HOUR}"

LASTCYCLE_DATE=$(date --date="-${pa} day" '+%Y%m%d')
LASTCYCLE_HOUR="00"  # standard forecast cycle for checking truth availability

# Define model-dependent expected file check
if [ "${MODEL}" == "GEFS" ]; then
  EXPECTED_PATTERN="${ARCHIVE_PATH}/GEFSv12Waves_${LASTCYCLE_DATE}${LASTCYCLE_HOUR}/gefs.wave.${LASTCYCLE_DATE}.00.global.0p25.f012.grib2"
elif [ "${MODEL}" == "ECMWF" ]; then
  # Matches any ensemble member (e.g., ECMWF_ENS_wave_2026101800.*.nc)
  EXPECTED_PATTERN="${ARCHIVE_PATH}/wave/${LASTCYCLE_HOUR}/ECMWF_ENS_wave_${LASTCYCLE_DATE}${LASTCYCLE_HOUR}.*.nc"
else
  echo "ERROR: Model ${MODEL} is not supported. Use GEFS or ECMWF."
  exit 1
fi

echo " "
echo " Validating ${MODEL} cycle ${FCYCLE} (week 2: days ${LTIME1}-${LTIME2})."
echo " Waiting on last truth-window archive cycle: ${LASTCYCLE_DATE}${LASTCYCLE_HOUR}"
echo " Expected file pattern: ${EXPECTED_PATTERN}"
echo " "

# Helper function to find a valid file matching pattern
get_target_file() {
  ls -1 ${EXPECTED_PATTERN} 2>/dev/null | head -n 1 || echo ""
}

# Check if the required archive file exists and has reached minimum size
FSIZE=0
TRIES=1
while [ "$FSIZE" -lt 100000 ] && [ "$TRIES" -le 144 ]; do
  if [ ${TRIES} -gt 5 ]; then
    sleep 300
  fi
  
  TARGET_FILE=$(get_target_file)
  if [ -n "$TARGET_FILE" ] && [ -f "$TARGET_FILE" ]; then
    FSIZE=$(du -sb "$TARGET_FILE" | awk '{print $1}')
  else
    FSIZE=0
  fi

  echo "  Try ${TRIES}/144: file $([ "$FSIZE" -gt 0 ] && echo "found (${TARGET_FILE}, ${FSIZE} bytes)" || echo "not found yet")."
  TRIES=$((TRIES + 1))
done

if [ "$FSIZE" -lt 100000 ]; then
  echo " "
  echo " WARNING: gave up after ${TRIES} tries (~12h) - last truth-window archive cycle never reached expected size."
  echo " Proceeding anyway, but ground truth may be incomplete for the most recent window."
  echo " "
fi

# Load Python environment
source /home/Ricardo.Campos/python/envs/intelpy_env/bin/activate

echo " "
echo " PYTHON PROCESSING: GLOBAL HAZARDS OUTLOOK (VALIDATION) - ${MODEL} PROBABILITY MAPS, ${FCYCLE} "
echo " "
for WW3VAR in ${MVARS}; do
  python3 "${PYSCRIPT}" "${PYCYAML}" "${FCYCLE}" "${LTIME1}" "${LTIME2}" "${WW3VAR}"
  echo " Probability maps for ${WW3VAR} Ok."
done
echo " "
echo " PYTHON PROCESSING COMPLETE."

# Organize generated plots into target subdirectories
cd "${OUTPATH}"
mkdir -p "${YEAR}${MONTH}${DAY}${HOUR}/Hs"
mkdir -p "${YEAR}${MONTH}${DAY}${HOUR}/WS10"
mv -f *Hs* "${YEAR}${MONTH}${DAY}${HOUR}/Hs/" 2>/dev/null || true
mv -f *WS10* "${YEAR}${MONTH}${DAY}${HOUR}/WS10/" 2>/dev/null || true

echo " "
echo " Done."
echo " "

