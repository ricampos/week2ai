#!/bin/bash
set -e

# --- Configuration ---
REPO_DIR="/home/Ricardo.Campos/github/week2ai/wp/week2ai"
BASE_DIR="/scratch4/AOML/aoml-phod/Ricardo.Campos/week2_multimodel/results"
VAL_BASE_DIR="/scratch4/AOML/aoml-phod/Ricardo.Campos/week2_sval/results"

CYCLE="00"
TODAY=$(date -u +%Y%m%d)
TARGET_DIR="${BASE_DIR}/${TODAY}${CYCLE}"

# Validation: probmaps_gefs_valdt.sh validates the cycle issued (LTIME2 + pa) days ago
# (LTIME2=14, pa=1) and stores it in a directory named after that cycle.
# Keep VAL_LAG_DAYS in sync with LTIME2+pa in probmaps_gefs_valdt.sh.
VAL_LAG_DAYS=15
VAL_CYCLE="$(date -u -d "${VAL_LAG_DAYS} days ago" +%Y%m%d)${CYCLE}"
VAL_TARGET_DIR="${VAL_BASE_DIR}/${VAL_CYCLE}"

MAX_RETRIES=4          # Total attempts (Initial attempt + 3 retries, 1 hour apart)
SLEEP_DURATION=3600

# List of all expected Hs figures (12 total)
EXPECTED_HS=(
  "ProbMap_Hs_4.0_fcst07to14_GEFS_MAIN.png"
  "ProbMap_Hs_6.0_fcst07to14_GEFS_MAIN.png"
  "ProbMap_Hs_9.0_fcst07to14_GEFS_MAIN.png"
  "ProbMap_Hs_14.0_fcst07to14_GEFS_MAIN.png"
  "ProbMap_Hs_4.0_fcst07to14_ECMWF_MAIN.png"
  "ProbMap_Hs_6.0_fcst07to14_ECMWF_MAIN.png"
  "ProbMap_Hs_9.0_fcst07to14_ECMWF_MAIN.png"
  "ProbMap_Hs_14.0_fcst07to14_ECMWF_MAIN.png"
  "Pctl95_Hs_fcst07to14_GEFS_MAIN.png"
  "Pctl99_Hs_fcst07to14_GEFS_MAIN.png"
  "Pctl95_Hs_fcst07to14_ECMWF_MAIN.png"
  "Pctl99_Hs_fcst07to14_ECMWF_MAIN.png"
)

# List of all expected WS10 figures (10 total)
EXPECTED_WS10=(
  "ProbMap_WS10_34.0_fcst07to14_GEFS_MAIN.png"
  "ProbMap_WS10_48.0_fcst07to14_GEFS_MAIN.png"
  "ProbMap_WS10_64.0_fcst07to14_GEFS_MAIN.png"
  "ProbMap_WS10_34.0_fcst07to14_ECMWF_MAIN.png"
  "ProbMap_WS10_48.0_fcst07to14_ECMWF_MAIN.png"
  "ProbMap_WS10_64.0_fcst07to14_ECMWF_MAIN.png"
  "Pctl95_WS10_fcst07to14_GEFS_MAIN.png"
  "Pctl99_WS10_fcst07to14_GEFS_MAIN.png"
  "Pctl95_WS10_fcst07to14_ECMWF_MAIN.png"
  "Pctl99_WS10_fcst07to14_ECMWF_MAIN.png"
)

# Automatic validation figures (GEFS only): 4 Hs + 3 WS10
EXPECTED_VAL_HS=(
  "ProbMap_SpatialValidation_Hs_4.0_latest_GEFS_MAIN.png"
  "ProbMap_SpatialValidation_Hs_6.0_latest_GEFS_MAIN.png"
  "ProbMap_SpatialValidation_Hs_9.0_latest_GEFS_MAIN.png"
  "ProbMap_SpatialValidation_Hs_14.0_latest_GEFS_MAIN.png"
)
EXPECTED_VAL_WS10=(
  "ProbMap_SpatialValidation_WS10_34.0_latest_GEFS_MAIN.png"
  "ProbMap_SpatialValidation_WS10_48.0_latest_GEFS_MAIN.png"
  "ProbMap_SpatialValidation_WS10_64.0_latest_GEFS_MAIN.png"
)

# Function to check if all expected forecast figures exist
check_all_figures_exist() {
  local missing_count=0

  if [ ! -d "${TARGET_DIR}" ]; then
    echo "Directory ${TARGET_DIR} does not exist yet."
    return 1
  fi

  for file in "${EXPECTED_HS[@]}"; do
    if [ ! -f "${TARGET_DIR}/Hs/${file}" ]; then
      echo "  [MISSING] Hs/${file}"
      missing_count=$((missing_count + 1))
    fi
  done

  for file in "${EXPECTED_WS10[@]}"; do
    if [ ! -f "${TARGET_DIR}/WS10/${file}" ]; then
      echo "  [MISSING] WS10/${file}"
      missing_count=$((missing_count + 1))
    fi
  done

  return ${missing_count}
}

# Function to check if all expected validation figures exist
check_validation_figures_exist() {
  local missing_count=0

  if [ ! -d "${VAL_TARGET_DIR}" ]; then
    echo "Validation directory ${VAL_TARGET_DIR} does not exist yet."
    return 1
  fi

  for file in "${EXPECTED_VAL_HS[@]}"; do
    if [ ! -f "${VAL_TARGET_DIR}/Hs/${file}" ]; then
      echo "  [MISSING] (validation) Hs/${file}"
      missing_count=$((missing_count + 1))
    fi
  done

  for file in "${EXPECTED_VAL_WS10[@]}"; do
    if [ ! -f "${VAL_TARGET_DIR}/WS10/${file}" ]; then
      echo "  [MISSING] (validation) WS10/${file}"
      missing_count=$((missing_count + 1))
    fi
  done

  return ${missing_count}
}

# --- Retry Loop ---
# Forecast figures are required. Validation figures are best-effort: if they are
# still missing on the last attempt, the forecast figures are pushed anyway.
MAIN_OK=0
VAL_OK=0
attempt=1
while [ ${attempt} -le ${MAX_RETRIES} ]; do
  echo "=================================================="
  echo "Checking figure status (Attempt ${attempt}/${MAX_RETRIES}): $(date -u)"
  echo "Forecast directory:   ${TARGET_DIR}"
  echo "Validation directory: ${VAL_TARGET_DIR}"

  if check_all_figures_exist; then
    MAIN_OK=1
    echo "✓ All expected forecast figures are present!"
  else
    MAIN_OK=0
  fi

  if check_validation_figures_exist; then
    VAL_OK=1
    echo "✓ All expected validation figures are present!"
  else
    VAL_OK=0
  fi

  if [ ${MAIN_OK} -eq 1 ] && [ ${VAL_OK} -eq 1 ]; then
    break
  fi

  if [ ${attempt} -lt ${MAX_RETRIES} ]; then
    echo "Missing figures detected. Waiting 1 hour (3600s) before checking again..."
    sleep ${SLEEP_DURATION}
  else
    if [ ${MAIN_OK} -eq 0 ]; then
      echo "× Forecast figures still incomplete after ${MAX_RETRIES} attempts. Giving up for today."
      exit 1
    fi
    echo "! Validation figures still incomplete after ${MAX_RETRIES} attempts."
    echo "  Pushing forecast figures only; validation tab keeps the previous figures."
  fi
  attempt=$((attempt + 1))
done

# --- Copy & Push Operations ---
cd "${REPO_DIR}"
mkdir -p figures/Hs figures/WS10 figures/Validation/Hs figures/Validation/WS10

echo "Copying forecast figures..."
cp -u "${TARGET_DIR}/Hs"/*.png figures/Hs/
cp -u "${TARGET_DIR}/WS10"/*.png figures/WS10/

if [ ${VAL_OK} -eq 1 ]; then
  echo "Copying validation figures (validated cycle ${VAL_CYCLE})..."
  # Only the stable "latest" filenames are copied
  for file in "${EXPECTED_VAL_HS[@]}"; do
    cp "${VAL_TARGET_DIR}/Hs/${file}" figures/Validation/Hs/
  done
  for file in "${EXPECTED_VAL_WS10[@]}"; do
    cp "${VAL_TARGET_DIR}/WS10/${file}" figures/Validation/WS10/
  done
  # Small text file so the page can show which cycle was validated
  echo "${VAL_CYCLE}" > figures/Validation/validated_cycle.txt
fi

git add figures/

if git diff --staged --quiet; then
  echo "No new changes or new figures to commit."
else
  git commit -m "Auto-update multi-model figures for ${TODAY}${CYCLE} (validation: ${VAL_CYCLE}) [$(date -u '+%Y-%m-%d %H:%M UTC')]"
  git push origin main
  echo "Figures successfully pushed to GitHub!"
fi
