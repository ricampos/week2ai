#!/bin/bash

########################################################################
# probmaps_multimodel.sh and probmaps_multimodel.py
#
# VERSION AND LAST UPDATE:
#   v1.0  10/20/2026
#
# PURPOSE:
#  Script to generate the multi-model Probability Maps (Global Hazard Outlooks)
#   of significant wave height (Hs) and 10-m wind speed (WS10) for the week-2
#   forecast. The members of the ensembles listed in the `models:` section of
#   the .yaml file (e.g. 31 GEFS + 51 ECMWF) are appended into one single
#   ensemble by probmaps_multimodel.py.
#
#  This shell script waits until the last forecast file of every model is
#   available, runs probmaps_multimodel.py for each variable (mvars in the
#   .yaml), and moves the figures to outpath/<YYYYMMDDHH>/{Hs,WS10}.
#
# USAGE:
#  The first argument is the forecast cycle (00,06,12,18)
#  The second argument is the time-delay, i.e. how many days backwards in time
#   (0 for today, 1 for yesterday etc)
#  The third argument is the configuration file probmaps_multimodel.yaml, which
#   must contain:
#     pyscript: path+name of probmaps_multimodel.py
#     mvars: "Hs WS10"
#     outpath: where the figures are saved
#     label: MULTI        (optional; used in the figure names, default MULTI)
#     models:             (name, path and number of members of every model)
#       - name: GEFS
#         path: /path/to/GEFS/archive/
#         nenm: 31
#       - name: ECMWF
#         path: /path/to/ECMWF/archive/
#         nenm: 51
#         mvar_hs: ...
#         mvar_wnd: ...
#     ... and the usual probmaps_*.yaml keys (see probmaps_multimodel.py).
#
#  This code expects the ensemble files (GEFS, ECMWF) were already downloaded
#   (download_GEFSwaves.sh, download_ECMWF_ENS_*.sh). If they are not complete yet,
#   it checks again every 5 minutes (max. 12 hours).
#
#  Example:
#    bash probmaps_multimodel.sh 00 0 /scratch4/AOML/aoml-phod/Ricardo.Campos/week2_multimodel/probmaps_multimodel.yaml
#
# OUTPUT:
#  Probability Map and Percentile figures saved in outpath/<YYYYMMDDHH>/Hs
#   and outpath/<YYYYMMDDHH>/WS10
#
# DEPENDENCIES:
#  The python code probmaps_multimodel.py contains the module dependencies.
#
# AUTHOR and DATE:
#  10/20/2026: Ricardo M. Campos, first version (adapted from probmaps.sh v2.1).
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
# Forecast Cycle (00,06,12,18)
HOUR="$1"
# Days into the past. pa=1 runs using yesterday's cycle
pa="$2"
# .yaml configuration file containing paths and information for this
#   shell script as well as for the python code.
PYCYAML="$3"

# Waiting for the files: number of checks, seconds between checks, and the minimum
#  file size (bytes) to consider a file complete.
MAX_TRIES=144
SLEEP_SECS=300
MIN_SIZE_GEFS=10000000      # same as probmaps_gefs.sh
MIN_SIZE_EC_WAVE=300000000  # same as probmaps_ecmwf.sh
MIN_SIZE_EC_WIND=1000000    # probmaps_ecmwf.sh only checks the wave file: raise to the size of a complete wind file
MIN_SIZE_NC=1000000         # EnvCanada (not used operationally here)

# Module load python and activate environment when necessary (also used to read the yaml)
source /home/Ricardo.Campos/python/envs/intelpy_env/bin/activate

# Read the .yaml (top-level keys). Lists are returned separated by spaces.
read_yaml() {
  python3 - "${PYCYAML}" "$1" <<'EOF'
import sys, yaml
with open(sys.argv[1], 'r') as f:
    c = yaml.safe_load(f)
v = c.get(sys.argv[2], '')
if isinstance(v, (list, tuple)):
    v = ' '.join(str(x) for x in v)
print('' if v is None else v)
EOF
}

#  Python script (probability maps)
PYSCRIPT=$(read_yaml pyscript)
#  Variable names, for the python processing (probability maps)
MVARS=$(read_yaml mvars)
#  Output path
OUTPATH=$(read_yaml outpath)
#  Label used in the figure names
LABEL=$(read_yaml label)
LABEL=${LABEL:-MULTI}
#  Models: name|path|number of members (one line per model)
MODELS=$(python3 - "${PYCYAML}" <<'EOF'
import sys, yaml
with open(sys.argv[1], 'r') as f:
    c = yaml.safe_load(f)
for m in c['models']:
    print(str(m['name']) + '|' + str(m['path']).rstrip('/') + '|' + str(m['nenm']))
EOF
)

# Date / Forecast cycle
YEAR=`date --date=-$pa' day' '+%Y'`
MONTH=`date --date=-$pa' day' '+%m'`
DAY=`date --date=-$pa' day' '+%d'`
DATE="${YEAR}${MONTH}${DAY}"
CYCLEDIR="${DATE}${HOUR}"

# Last file of every model. The last member of the last forecast lead (or the last
#  member file, for the netcdf models) is the final file to be written.
LASTFILES=(); LASTMIN=(); NAMES=()
while IFS='|' read -r MNAME MPATH MNENM; do
  [ -z "${MNAME}" ] && continue
  LASTMEM=$(printf '%02d' $((MNENM - 1)))
  case "${MNAME}" in
    GEFS)
      LASTFILES+=("${MPATH}/GEFSv12Waves_${DATE}${HOUR}/gefs.wave.${DATE}.${LASTMEM}.global.0p25.f384.grib2")
      LASTMIN+=("${MIN_SIZE_GEFS}"); NAMES+=("GEFS")
      ;;
    ECMWF)
      LASTFILES+=("${MPATH}/wave/${HOUR}/ECMWF_ENS_wave_${DATE}${HOUR}.${LASTMEM}.nc")
      LASTMIN+=("${MIN_SIZE_EC_WAVE}"); NAMES+=("ECMWF wave")
      LASTFILES+=("${MPATH}/wind/${HOUR}/ECMWF_ENS_wind_${DATE}${HOUR}.${LASTMEM}.nc")
      LASTMIN+=("${MIN_SIZE_EC_WIND}"); NAMES+=("ECMWF wind")
      ;;
    EnvCanada)
      LASTFILES+=("${MPATH}/wave/${HOUR}/GEWPS_wave_${DATE}${HOUR}.${LASTMEM}.nc")
      LASTMIN+=("${MIN_SIZE_NC}"); NAMES+=("EnvCanada wave")
      LASTFILES+=("${MPATH}/wind/${HOUR}/GEPS_wind_${DATE}${HOUR}.${LASTMEM}.nc")
      LASTMIN+=("${MIN_SIZE_NC}"); NAMES+=("EnvCanada wind")
      ;;
    *)
      echo " ERROR: model ${MNAME} not supported. Use GEFS, ECMWF or EnvCanada."
      exit 1
      ;;
  esac
done <<< "${MODELS}"

if [ ${#LASTFILES[@]} -eq 0 ]; then
  echo " ERROR: no model found in the models: list of ${PYCYAML}."
  exit 1
fi

echo " "
echo " Multi-model cycle ${CYCLEDIR}. Looking for the last file of every model ..."
for i in "${!LASTFILES[@]}"; do
  echo "  ${NAMES[$i]}: ${LASTFILES[$i]}"
done
echo " "

# Check all the ensembles are complete and ready.
# If not, it waits and then tries again (default: every 5 min, max 12 hours)
ALLREADY=0
TRIES=1
while [ "${TRIES}" -le "${MAX_TRIES}" ]; do
  NOTREADY=0
  for i in "${!LASTFILES[@]}"; do
    if [ -f "${LASTFILES[$i]}" ]; then
      FSIZE=$(du -sb "${LASTFILES[$i]}" | awk '{print $1}')
    else
      FSIZE=0
    fi
    if [ "${FSIZE}" -lt "${LASTMIN[$i]}" ]; then
      NOTREADY=$((NOTREADY + 1))
      echo "  Try ${TRIES}/${MAX_TRIES}: ${NAMES[$i]} not ready yet ($([ "${FSIZE}" -gt 0 ] && echo "${FSIZE} bytes" || echo "file not found"))."
    fi
  done

  if [ "${NOTREADY}" -eq 0 ]; then
    ALLREADY=1
    echo "  Try ${TRIES}/${MAX_TRIES}: all the files were found."
    break
  fi
  if [ "${TRIES}" -lt "${MAX_TRIES}" ]; then
    sleep "${SLEEP_SECS}"
  fi
  TRIES=$((TRIES + 1))
done

if [ "${ALLREADY}" -eq 0 ]; then
  echo " "
  echo " WARNING: gave up after ${MAX_TRIES} tries - at least one model never reached the expected size."
  echo " Proceeding anyway. The python code stops if a model cannot be read, unless"
  echo " require_all_models: false in the .yaml (then the missing model is skipped)."
  echo " "
fi

echo "  "
echo " PYTHON PROCESSING: GLOBAL HAZARDS OUTLOOK - MULTI-MODEL PROBABILITY MAPS, ${CYCLEDIR} "
echo "  "
# loop through variables
for WW3VAR in ${MVARS}; do
  # 7 14 is the time intervall (days) for week 2
  python3 "${PYSCRIPT}" "${PYCYAML}" "${CYCLEDIR}" 7 14 "${WW3VAR}"
  echo " Multi-model probability maps for ${WW3VAR} Ok."
done

echo "  "
echo " PYTHON PROCESSING COMPLETE."
# ----

# Move the figures of this run to outpath/<cycle>/<variable>. Only the files of this
#  multi-model product (label) are moved, so other figures in outpath are not touched.
cd "${OUTPATH}"
#  (if probmaps_gefs.sh / probmaps_ecmwf.sh ran meanwhile, their `mv *Hs*` may already have
#   moved these files to the same folder, so no match is not treated as an error)
shopt -s nullglob
for VARDIR in Hs WS10; do
  mkdir -p "${CYCLEDIR}/${VARDIR}"
  FIGS=(./*"${VARDIR}"*_"${LABEL}"_*)
  if [ ${#FIGS[@]} -gt 0 ]; then
    mv -f "${FIGS[@]}" "${CYCLEDIR}/${VARDIR}/"
  else
    echo " No ${VARDIR} ${LABEL} figures left in ${OUTPATH} (already moved?). Check ${CYCLEDIR}/${VARDIR}/"
  fi
done

echo "  "
echo " Figures saved in ${OUTPATH}/${CYCLEDIR}"
echo " Done."
echo "  "

