#!/usr/bin/env bash
set -euo pipefail

# Activate your env (adjust if needed)
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate clinical_annot

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/bootstrap_config_new.json}"

# Fix for R tempdir issues on WSL / full disks
# export TMPDIR="${TMPDIR:-/tmp}"
# mkdir -p "$TMPDIR"

TTS_SKIP_MAIN=1 Rscript "${SCRIPT_DIR}/bootstrap_comparer.r" "$CONFIG"
