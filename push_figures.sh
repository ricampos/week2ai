#!/bin/bash
set -e

# --- Configuration ---
REPO_DIR="/home/Ricardo.Campos/github/week2ai/wp/week2ai"
BASE_DIR="/scratch4/AOML/aoml-phod/Ricardo.Campos/week2_multimodel/results"

CYCLE="00"
TODAY=$(date -u +%Y%m%d)
TARGET_DIR="${BASE_DIR}/${TODAY}${CYCLE}"

echo "=================================================="
echo "Running push_figures.sh: $(date -u)"
echo "Target directory: ${TARGET_DIR}"

if [ ! -d "${TARGET_DIR}" ]; then
  echo "Warning: Directory ${TARGET_DIR} does not exist yet. Exiting."
  exit 0
fi

cd "${REPO_DIR}"

# Ensure directories exist
mkdir -p figures/Hs figures/WS10

echo "Copying figures..."
# Copy both ProbMap and Pctl figures for Hs and WS10
cp -u "${TARGET_DIR}/Hs"/ProbMap_*.png figures/Hs/ 2>/dev/null || echo "No Hs ProbMap figures found."
cp -u "${TARGET_DIR}/Hs"/Pctl*.png figures/Hs/ 2>/dev/null || echo "No Hs Pctl figures found."

cp -u "${TARGET_DIR}/WS10"/ProbMap_*.png figures/WS10/ 2>/dev/null || echo "No WS10 ProbMap figures found."
cp -u "${TARGET_DIR}/WS10"/Pctl*.png figures/WS10/ 2>/dev/null || echo "No WS10 Pctl figures found."

# Git push
git add figures/

if git diff --staged --quiet; then
  echo "No new changes or new figures to commit."
else
  git commit -m "Auto-update figures for ${TODAY}${CYCLE} [$(date -u '+%Y-%m-%d %H:%M UTC')]"
  git push origin main
  echo "Figures successfully pushed to GitHub!"
fi


