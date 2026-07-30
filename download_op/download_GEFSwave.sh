#!/bin/bash

########################################################################
# download_GEFSwave.sh
#
# VERSION AND LAST UPDATE:
#   v1.0  02/15/2023
#   v1.1  04/30/2025
#   v1.2  07/22/2026
#   v1.3  07/28/2026
#
# PURPOSE:
#  Script to download NOAA Global Ensemble Forecast System (GEFS), Wave 
#   Forecast from WAVEWATCH III operational. Download from AWS Open Data
#   S3 mirror and save the grib2 files without any conversion or
#   processing. It includes the control and all perturbed members of the
#   ensemble. Global wind and wave fields.
#
# USAGE:
#  Two input arguments, date and path, must be entered.
#  Example:
#    bash download_GEFSwave.sh 20220823 00 /home/ricardo/data/gefs
#
# OUTPUT:
#  Multiple grib2 files, for each time step and ensemble member.
#
# DEPENDENCIES:
#  wget, wgrib2 (for GRIB integrity validation - module load wgrib2 on Hera
#  if not already on PATH)
#
# AUTHOR and DATE:
#  02/15/2023: Ricardo M. Campos, first version 
#  04/30/2025: Ricardo M. Campos, flexible cycle time
#  07/22/2026: Ricardo M. Campos, fixed set -e early-exit bug on failed downloads
#  07/28/2026: Ricardo M. Campos, fixed 404 retry pileup / added error logging. Switched
#    to AWS S3 mirror + wgrib2 integrity check
#
# PERSON OF CONTACT:
#  Ricardo M. Campos: ricardo.campos@noaa.gov
#
#  If you are interested in operational forecasts from NOAA ftp, see:
#  https://github.com/NOAA-EMC/WW3-tools/tree/develop/opforecast
#
########################################################################

set -euo pipefail
export USER_IS_ROOT=0
export MODULEPATH=/etc/scl/modulefiles:/apps/lmod/lmod/modulefiles/Core:/apps/modules/modulefiles/Linux:/apps/modules/modulefiles
source /apps/lmod/lmod/init/bash

if ! command -v wgrib2 >/dev/null 2>&1; then
  module load wgrib2 2>/dev/null || true
fi
if ! command -v wgrib2 >/dev/null 2>&1; then
  echo "WARNING: wgrib2 not found on PATH. Falling back to size-only validation (less reliable)."
fi

# Two input arguments
# date
CTIME="$1"
# cycle 00,06,12,18
# HCYCLE="00"
HCYCLE="$2"
# destination path
DIRW="$3"
# server address
# Switched from NOMADS to the AWS Open Data S3 mirror: same directory
# structure (gefs.YYYYMMDD/HH/wave/gridded/...)
# SERVER=https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/   # old NOMADS source
SERVER=https://noaa-gefs-pds.s3.amazonaws.com/
# ensemble members
ensblm="`seq -f "%02g" 0 1 30`"
# Forecast lead time (hours) to download
fleads="`seq -f "%03g" 0 6 384`"
# minimum acceptable size (bytes) for a completed grib2 file
MINSIZE=10000000
# scratch file for capturing wget's stderr so we can tell a permanent
# 404 (data aged off the server) apart from a transient network hiccup
WGETERR=$(mktemp)
trap 'rm -f "$WGETERR"' EXIT
# track failures so we can report them at the end instead of finding out later
FAILED_FILES=()
# separately track permanent 404s (expected/benign - data no longer
# available) and files that failed GRIB integrity validation
NOTFOUND_FILES=()
CORRUPT_FILES=()

# Validate a downloaded grib2 file with wgrib2. Returns 0 (valid) if
# wgrib2 can parse it and it reports at least one message; returns 1
# (invalid) on any parse error or zero messages. Falls back to "assume
# valid" (size check only) if wgrib2 isn't available at all, so the
# script still runs, just with weaker guarantees.
validate_grib() {
  local f="$1"
  if ! command -v wgrib2 >/dev/null 2>&1; then
    return 0
  fi
  local nmsg
  nmsg=$(wgrib2 "$f" 2>/dev/null | wc -l)
  if [ "$nmsg" -gt 0 ]; then
    return 0
  else
    return 1
  fi
}

# Clean up any stray wget-log files from prior runs (harmless, but no
# reason to accumulate them now that -o above prevents new ones).
rm -f ${DIRW}/wget-log*

cd ${DIRW}
for h in $fleads;do
  echo " ======== GEFS Forecast, AWS S3 archive: ${CTIME} ${HCYCLE}Z $h ========"
  for e in $ensblm;do
    echo $e
    FILE=$DIRW/gefs.wave.${CTIME}.${e}.global.0p25.f$(printf "%03.f" $h).grib2
    # Skip if file exists, is large enough, AND passes GRIB integrity validation
    if [ -f "$FILE" ]; then
      TAM=$(du -sb "$FILE" | awk '{ print $1 }')
      if [ "$TAM" -ge "$MINSIZE" ] && validate_grib "$FILE"; then
        echo "File $FILE already exists, is large enough, and passes GRIB validation. Skipping download."
        continue
      elif [ "$TAM" -ge "$MINSIZE" ]; then
        echo "File $FILE is large enough but FAILED GRIB validation (likely truncated/corrupted). Re-downloading."
        rm -f "$FILE"
      fi
    fi
    # size TAM and tries TRIES will control the process
    TAM=0
    TRIES=1
    while [ $TAM -lt $MINSIZE ] && [ $TRIES -le 130 ]; do
      # sleep 5 minutes between attemps
      if [ ${TRIES} -gt 5 ]; then
        sleep 30
      fi
      if [ ${TAM} -lt $MINSIZE ]; then

          if [ ${e} == "00" ]; then
            wget --wait=1 --random-wait --limit-rate=5m -l1 -H -t1 -nd -N -np -erobots=off --tries=3 -o "$WGETERR" ${SERVER}gefs.${CTIME}/${HCYCLE}/wave/gridded/gefs.wave.t${HCYCLE}z.c${e}.global.0p25.f"$(printf "%03.f" $h)".grib2 -O $DIRW/gefs.wave.${CTIME}.${e}.global.0p25.f"$(printf "%03.f" $h)".grib2 || true
          else
            wget --wait=1 --random-wait --limit-rate=5m -l1 -H -t1 -nd -N -np -erobots=off --tries=3 -o "$WGETERR" ${SERVER}gefs.${CTIME}/${HCYCLE}/wave/gridded/gefs.wave.t${HCYCLE}z.p${e}.global.0p25.f"$(printf "%03.f" $h)".grib2 -O $DIRW/gefs.wave.${CTIME}.${e}.global.0p25.f"$(printf "%03.f" $h)".grib2 || true
          fi
          cat "$WGETERR"
          # test if the downloaded file exists
          # rewritten so a missing file (test -f returns 1) does NOT
          # trigger `set -e` and kill the script
          if [ -f $DIRW/gefs.wave.${CTIME}.${e}.global.0p25.f"$(printf "%03.f" $h)".grib2 ]; then
            TE=0
          else
            TE=1
          fi
          if [ ${TE} -eq 1 ]; then
            TAM=0
          else
            # check size of each file
            TAM=`du -sb $DIRW/gefs.wave.${CTIME}.${e}.global.0p25.f"$(printf "%03.f" $h)".grib2 | awk '{ print $1 }'`
          fi
          echo $DIRW/gefs.wave.${CTIME}.${e}.global.0p25.f"$(printf "%03.f" $h)".grib2

          if [ $TAM -lt $MINSIZE ] && grep -q "404 Not Found" "$WGETERR"; then
            echo "PERMANENT 404 for $FILE - not retrying."
            NOTFOUND_FILES+=("$FILE")
            TRIES=131
            TAM=0
            break
          fi

          if [ $TAM -ge $MINSIZE ]; then
            if ! validate_grib "$DIRW/gefs.wave.${CTIME}.${e}.global.0p25.f$(printf "%03.f" $h).grib2"; then
              echo "GRIB validation FAILED for $FILE (try ${TRIES}) - treating as incomplete, will retry."
              rm -f "$DIRW/gefs.wave.${CTIME}.${e}.global.0p25.f$(printf "%03.f" $h).grib2"
              TAM=0
            fi
          fi
	  # sleep 2
      fi
      TRIES=`expr $TRIES + 1`
      sleep 1
    done
    # If we exhausted all tries (or hit a permanent 404) and still don't
    # have a valid file, log it and move on instead of failing the whole run
    if [ $TAM -lt $MINSIZE ]; then
      echo "WARNING: failed to download $FILE after ${TRIES} tries."
      FAILED_FILES+=("$FILE")
    fi
  done
done
echo " Done ${CTIME}."
if [ ${#NOTFOUND_FILES[@]} -gt 0 ]; then
  echo " "
  echo "==== SUMMARY: ${#NOTFOUND_FILES[@]} file(s) returned 404 (not available) ===="
  for f in "${NOTFOUND_FILES[@]}"; do
    echo "  $f"
  done
fi
if [ ${#FAILED_FILES[@]} -gt 0 ]; then
  echo " "
  echo "==== SUMMARY: ${#FAILED_FILES[@]} file(s) failed to download or never passed GRIB validation ===="
  for f in "${FAILED_FILES[@]}"; do
    echo "  $f"
  done
fi

