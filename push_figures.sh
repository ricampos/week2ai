#!/bin/bash
set -e

# --- Configuration ---
REPO_DIR="/home/Ricardo.Campos/github/week2ai/wp/week2ai"
BASE_DIR="/scratch4/AOML/aoml-phod/Ricardo.Campos/week2_multimodel/results"

CYCLE="00"
TODAY=$(date -u +%Y%m%d)
TARGET_DIR="${BASE_DIR}/${TODAY}${CYCLE}"

MAX_RETRIES=4          # Total attempts (Initial attempt + 1 retry after 1 hour)
SLEEP_DURATION=3600

# List of all expected Hs figures (11 total)
EXPECTED_HS=(
  "ProbMap_Hs_4.0_fcst07to14_GEFS_MAIN.png"
  "ProbMap_Hs_6.0_fcst07to14_GEFS_MAIN.png"
  "ProbMap_Hs_9.0_fcst07to14_GEFS_MAIN.png"
  "ProbMap_Hs_14.0_fcst07to14_GEFS_MAIN.png"
  "ProbMap_Hs_4.0_fcst07to14_ECMWF_MAIN.png"
  "ProbMap_Hs_6.0_fcst07to14_ECMWF_MAIN.png"
  "ProbMap_Hs_9.0_fcst07to14_ECMWF_MAIN.png"
  "ProbMap_Hs_14.0_fcst07to14_ECMWF_MAIN.png"
  "ProbMap_Hs_4.0_fcst07to14_EnvCanada_MAIN.png"
  "ProbMap_Hs_6.0_fcst07to14_EnvCanada_MAIN.png"
  "ProbMap_Hs_9.0_fcst07to14_EnvCanada_MAIN.png"
  "ProbMap_Hs_14.0_fcst07to14_EnvCanada_MAIN.png"
  "Pctl95_Hs_fcst07to14_GEFS_MAIN.png"
  "Pctl99_Hs_fcst07to14_GEFS_MAIN.png"
  "Pctl95_Hs_fcst07to14_ECMWF_MAIN.png"
  "Pctl99_Hs_fcst07to14_ECMWF_MAIN.png"
  "Pctl95_Hs_fcst07to14_EnvCanada_MAIN.png"
  "Pctl99_Hs_fcst07to14_EnvCanada_MAIN.png"
)

# List of all expected WS10 figures (11 total)
EXPECTED_WS10=(
  "ProbMap_WS10_34.0_fcst07to14_GEFS_MAIN.png"
  "ProbMap_WS10_48.0_fcst07to14_GEFS_MAIN.png"
  "ProbMap_WS10_64.0_fcst07to14_GEFS_MAIN.png"
  "ProbMap_WS10_34.0_fcst07to14_ECMWF_MAIN.png"
  "ProbMap_WS10_48.0_fcst07to14_ECMWF_MAIN.png"
  "ProbMap_WS10_64.0_fcst07to14_ECMWF_MAIN.png"
  "ProbMap_WS10_34.0_fcst07to14_EnvCanada_MAIN.png"
  "ProbMap_WS10_48.0_fcst07to14_EnvCanada_MAIN.png"
  "ProbMap_WS10_64.0_fcst07to14_EnvCanada_MAIN.png"
  "Pctl95_WS10_fcst07to14_GEFS_MAIN.png"
  "Pctl99_WS10_fcst07to14_GEFS_MAIN.png"
  "Pctl95_WS10_fcst07to14_ECMWF_MAIN.png"
  "Pctl99_WS10_fcst07to14_ECMWF_MAIN.png"
  "Pctl95_WS10_fcst07to14_EnvCanada_MAIN.png"
  "Pctl99_WS10_fcst07to14_EnvCanada_MAIN.png"
)

# Function to check if all expected files exist
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

# --- Retry Loop ---
attempt=1
while [ ${attempt} -le ${MAX_RETRIES} ]; do
  echo "=================================================="
  echo "Checking figure status (Attempt ${attempt}/${MAX_RETRIES}): $(date -u)"
  echo "Target directory: ${TARGET_DIR}"

  if check_all_figures_exist; then
    echo "✓ All expected figures are present!"
    break
  else
    if [ ${attempt} -lt ${MAX_RETRIES} ]; then
      echo "Missing figures detected. Waiting 1 hour (3600s) before checking again..."
      sleep ${SLEEP_DURATION}
    else
      echo "× Figures still incomplete after ${MAX_RETRIES} attempts. Giving up for today."
      exit 1
    fi
  fi
  attempt=$((attempt + 1))
done

# --- Copy & Push Operations ---
cd "${REPO_DIR}"
mkdir -p figures/Hs figures/WS10

echo "Copying figures..."
cp -u "${TARGET_DIR}/Hs"/*.png figures/Hs/
cp -u "${TARGET_DIR}/WS10"/*.png figures/WS10/

git add figures/

if git diff --staged --quiet; then
  echo "No new changes or new figures to commit."
else
  git commit -m "Auto-update complete multi-model figures for ${TODAY}${CYCLE} [$(date -u '+%Y-%m-%d %H:%M UTC')]"
  git push origin main
  echo "Figures successfully pushed to GitHub!"
fi

