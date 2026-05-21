#!/bin/bash
#source /data/weissjc/miniforge3/bin/activate base
#conda activate base

# Parameters:
# $1 = config (default: /data/weissjc/tta/scripts/compare_tts/config_simple.json)
# $2 = CUDA_VISIBLE_DEVICES (default: 5)
# ### TODO ### NOT CONFIGURED TO WORK $3 = port (default: 8000)

CONFIG=${1:-/Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/compare_tts/config_simple.json}

#CUDA_DEVICE=${2:-5}
# PORT=${3:-8000}

#CUDA_VISIBLE_DEVICES=$CUDA_DEVICE PYTHONPATH=/data/weissjc/tta/scripts/compare_tts/ uvicorn run_st_server:app --host localhost & # --port $PORT &

PYTHONPATH=/Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/compare_tts/ uvicorn run_st_server:app --host localhost & # --port $PORT &
UVICORN_PID=$!

Rscript /Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/compare_tts/compare_tts_2.r "$CONFIG"

# Kill the Uvicorn process after Rscript completes
kill $UVICORN_PID