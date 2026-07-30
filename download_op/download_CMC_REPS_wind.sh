#!/bin/bash

########################################################################
# download_CMC_REPS_wind.sh
#
# VERSION AND LAST UPDATE:
#   v1.0  09/03/2025
#
# PURPOSE:
#  Script to download Environment Canada / Canadian Meteorology Centre (CMC) wind ensemble forecast (regional)
#  This is used to obtain the 10-m winds for the Great Lakes
#   https://dd.meteo.gc.ca/
#   https://weather.gc.ca/ensemble/index_e.html
#   https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-10-0.shtml
#
# USAGE:
#  Three input arguments must be entered: cycle time (00 or 12), time lag(days), and the output path.
#  Example:
#    bash download_CMC_REPS_wind.sh 00 0 /home/ricardo/data/EnvCanada/wind
#  it will download the current day (00Z cycle)
#
# OUTPUT:
#  Multiple netcdf files containing the forecast steps, one file per ensemble member.
#
# DEPENDENCIES:
#  wget, CDO, NCO
#
# AUTHOR and DATE:
#  09/03/2025: Ricardo M. Campos, first version 
#
# PERSON OF CONTACT:
#  Ricardo M. Campos: ricardo.campos@noaa.gov
#
########################################################################

set -euo pipefail

# When working on the cluster
export USER_IS_ROOT=0
export MODULEPATH=/etc/scl/modulefiles:/apps/lmod/lmod/modulefiles/Core:/apps/modules/modulefiles/Linux:/apps/modules/modulefiles
source /apps/lmod/lmod/init/bash
module load cdo
module load nco

# Function to check if file has been previously downloaded
checkfile() {
  local FILE="$1"

  if [[ -f "$FILE" ]]; then
    local TAM
    TAM=$(stat -c%s "$FILE" 2>/dev/null)

    if [[ "$TAM" -ge 72000000 ]]; then
      echo "0"  # File exists and is big enough
      return
    fi
  fi

  echo "1"  # File missing or too small
}

# Cycle time
CHOUR="$1"
CHOUR=$(printf "%02.f" $CHOUR)
# time lag in days (past days)
pa="$2"
# Directory
WDIR="$3"

# Forecast cycle date
# DATE=$(date '+%Y%m%d')
DATE=$(date --date="-${pa} day" '+%Y%m%d')
exec > >(tee -a "$WDIR/download_CMC_REPS_wind_${DATE}${CHOUR}.log") 2>&1

# Wind variables: Hs, Dp, Tp, Tm
WVARS=("UGRD_AGL-10m" "VGRD_AGL-10m" "PRMSL_MSL")
# corresponding variable names after netcdf conversion
wvarname=("10u" "10v" "prmsl")

# Ensemble members
ensbl=($(seq 1 21))
ensblm=($(seq -f "%02g" 0 1 20))

cd $WDIR
if [ ! -d "$WDIR/work_${DATE}${CHOUR}" ]; then
  mkdir "$WDIR/work_${DATE}${CHOUR}"
fi
cd "$WDIR/work_${DATE}${CHOUR}"

# EnvCanada server address
SERVER=https://dd.meteo.gc.ca
# Forecast lead time
fleads=$(seq -f "%03g" 0 3 72)

# decimals/resolution for compression
dp=2

# Dowload loop
for h in $fleads;do

  for i in "${!WVARS[@]}"; do

    wv="${WVARS[$i]}"
    wvarn="${wvarname[$i]}"

    echo " ======== REPS Forecast: ${DATE} ${CHOUR}Z $h ${wv} ========"

    arqn="reps.wind.${DATE}T${CHOUR}Z_"${h}"H_${wvarn}"

    FILE="$WDIR/REPS_wind_${DATE}${CHOUR}.00.nc"
    cfile=$(checkfile "$FILE")
    if [ "$cfile" -eq 0 ]; then
      echo "File $FILE already exists and is large enough. Skipping download 1."
      continue
    fi

    wfile="${DATE}T${CHOUR}Z_MSC_REPS_${wv}_RLatLon0.09x0.09_PT"${h}"H.grib2"
    wget -l1 -H -t1 -nd -N -np -erobots=off --tries=3 "${SERVER}/${DATE}/WXO-DD/ensemble/reps/10km/grib2/${CHOUR}/"${h}"/${wfile}" -O "$WDIR/work_${DATE}${CHOUR}/${arqn}.grib2"
    echo "wget $WDIR/work_${DATE}${CHOUR}/${arqn}.grib2"

    test -f "$WDIR/work_${DATE}${CHOUR}/${arqn}.grib2"
    TE=$?
    if [ ${TE} -eq 0 ]; then

      # wgrib2 "${arqn}.grib2" -netcdf ${arqn}.saux.nc
      cdo -f nc4 copy "${arqn}.grib2" "${arqn}.saux.nc"

      for j in "${!ensbl[@]}"; do
        e="${ensbl[$j]}"
        ne="${ensblm[$j]}"

        if [ "$j" -eq 0 ]; then
          ncks -v ${wvarn} ${arqn}.saux.nc -o ${arqn}_${ne}.nc
        else
          ncks -v "${wvarn}_$e" ${arqn}.saux.nc -o ${arqn}_${ne}.saux.nc
          ncrename -v ${wvarn}_${e},${wvarn} ${arqn}_${ne}.saux.nc ${arqn}_${ne}.nc
        fi

        rm -f ${arqn}_${ne}.saux*.nc

      done

      rm -f $arqn.grib2
      rm -f $arqn.saux.nc

    else
      exit 1
    fi

  done

  for i in "${!ensbl[@]}"; do
    ne="${ensblm[$i]}"

    FILE="$WDIR/REPS_wind_${DATE}${CHOUR}.${ne}.nc"
    cfile=$(checkfile "$FILE")
    if [ "$cfile" -eq 0 ]; then
      echo "File $FILE already exists and is large enough. Skipping download 2."
      continue
    fi

    for j in "${!WVARS[@]}"; do
      wv="${WVARS[$j]}"
      wvarn="${wvarname[$j]}"
      if [ "$j" -eq 0 ]; then
        iname="reps.wind.${DATE}T${CHOUR}Z_${h}H_${ne}.nc"
        mv "reps.wind.${DATE}T${CHOUR}Z_${h}H_${wvarn}_${ne}.nc" "$iname"
      else
        src="reps.wind.${DATE}T${CHOUR}Z_${h}H_${wvarn}_${ne}.nc"
        ncks -A -v $wvarn "$src" "$iname"
        rm -f $src
      fi
    done

  done

done


for i in "${!ensbl[@]}"; do

  ne="${ensblm[$i]}"

  FILE="$WDIR/REPS_wind_${DATE}${CHOUR}.${ne}.nc"
  cfile=$(checkfile "$FILE")
  if [ "$cfile" -eq 0 ]; then
    echo "File $FILE already exists and is large enough. Skipping download 3."
    continue
  fi

  arqn="REPS_wind_"${DATE}${CHOUR}.${ne}
  ncrcat reps.wind.${DATE}T${CHOUR}Z_*H_${ne}.nc ${arqn}.saux.nc
  rm -f reps.wind.${DATE}T${CHOUR}Z_*H_${ne}.nc

  ncks -4 -L 1 ${arqn}".saux.nc" ${arqn}".saux1.nc"
  ncks --ppc default=.$dp ${arqn}".saux1.nc" "${arqn}.nc"
  ncatted -a _FillValue,,o,f,NaN "${arqn}.nc"
  chmod 775 "${arqn}.nc"
  mv "${arqn}.nc" $WDIR

  rm -f ${arqn}.saux*
  rm -f ${arqn}.*idx*
  rm -f *ncks* *tmp
  echo " File ${arqn} converted to netcdf and compressed with success. "
  sleep 1

done

sleep 1
cd $WDIR
rm -rf "work_${DATE}${CHOUR}"

echo " "
echo " Done download_CMC_REPS_wind.sh ${DATE} ${CHOUR}Z "

