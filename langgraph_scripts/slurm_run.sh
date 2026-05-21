#!/bin/bash
#SBATCH --job-name=run_llm_tta
#SBATCH --time=2-00:00:00
#SBATCH --array=0
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:8
#SBATCH --output=sbatch_logs/slurm_%A_%a.out
#SBATCH --error=sbatch_logs/slurm_%A_%a.err
#SBATCH --exclude=babel-o5-28,babel-p5-28,babel-q5-32

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tta_unimodal
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib":$LD_LIBRARY_PATH

# llama-server configuration
export LLAMA_SERVER="../unimodal/llama.cpp/build/bin/llama-server"
export LLAMA_LOG_DIR="./sbatch_logs"
export CUDA_DEVICES="0,1,2,3,4,5,6,7"
export LLAMA_TIMEOUT=1800

export MODEL_PATH=/data/user_data/juyongk/gguf/DeepSeek-V3.1-UD-Q2_K_XL/DeepSeek-V3.1-UD-Q2_K_XL-00001-of-00006.gguf
export MODEL_ALIAS="DeepSeek-V3.1"
export PORT=8001

export OUTPUT_DIR="outputs/$MODEL_ALIAS"

# Load start_server() and kill_server() functions
source llm_server_utils.sh

# Run the LLM server
start_server "$MODEL_PATH" "$MODEL_ALIAS" "$PORT"

if [ $? -eq 0 ]; then
    # Run langgraph
    bash run_langgraph_multimodal_v2.sh
    
    # Cleanup the server after client is done
    kill_server "$LLAMA_SERVER_PID" "$PORT"
else
    echo "Server failed to start."
    exit 1
fi