#!/bin/bash
#SBATCH --job-name=run_llm_tta
#SBATCH --time=2-00:00:00
#SBATCH --mem=128g
#SBATCH --partition=gpu
#SBATCH --qos=gpunlm2025.2
#SBATCH --gres=gpu:a100:4
#SBATCH --output=sbatch_output/slurm_%A_%a.out
#SBATCH --error=sbatch_error/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END  # Send email notifications
#SBATCH --mail-user=kumarsayantan94@gmail.com  # Replace with your actual email

# sinteractive --gres=gpu:a100:1 --mem=16g -c1 --time=2:00:00
module load CUDA/12.8.1 gcc
eval "$(conda shell.bash hook)"  # Ensures Conda is properly initialized
conda activate /data/kumars33/conda/envs/TTA

#Do tunning from local system
#ssh -L 8001:localhost:8001 -J $USER@biowulf.nih.gov cn0097


export LLAMA_SERVER="/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/unimodal/llama.cpp/build/bin/llama-server"
export LLAMA_LOG_DIR="./sbatch_logs"

export CUDA_DEVICES="0,1,2,3"
#export CUDA_DEVICES="0"

export LLAMA_TIMEOUT=1200

#export MODEL_PATH="/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3-32B-Q8_0.gguf"
export MODEL_PATH="/data/CHARM-MIMIC/.cache/huggingface/gguf/unsloth/DeepSeek-V3.2-GGUF/Q2_K/DeepSeek-V3.2-Q2_K-00001-of-00005.gguf"
export MODEL_ALIAS="DeepSeek-V3.2"

export PORT=8001

export OUTPUT_DIR="/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/multimodal_simple/outputs/tts_uncertain/dsv3/"

### IMPORTANT: avoid Biowulf Squid proxy for localhost health checks
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

# Load start_server() and kill_server() functions
source llm_server_utils.sh

# Run the LLM server
start_server "$MODEL_PATH" "$MODEL_ALIAS" "$PORT"

if [ $? -eq 0 ]; then
    # Run langgraph
    #bash run_langgraph_multimodal_v2.sh

    set -e

    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib":$LD_LIBRARY_PATH

    export summary_dir="/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/atp_tts/data/i2m4/notes"
    output_root=$OUTPUT_DIR

    # Get all .txt files in $summary_dir
    summary_files=("$summary_dir"/*.txt)

    mkdir -p "$output_root"

    # Print the ordered filenames, one per line
    for fname in "${summary_files[@]}"; do
        echo "Processing $fname"
        log_folder="${output_root}/$(basename "$fname" .txt)"
        hadm_id=$(basename "$fname" .txt | grep -oE '[0-9]+' | tail -n 1)
        python /data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/multimodal_simple/langgraph_runner_multimodal_v2_ablation_clean.py \
            --summary_file "$fname" \
            --hadm_id $hadm_id \
            --llm_endpoint "http://127.0.0.1:${PORT}" \
            --model_basename "$MODEL_ALIAS" \
            --log_folder "$log_folder" \
            --debug \
            --omit_tags think
    done
    
    # Cleanup the server after client is done
    kill_server "$LLAMA_SERVER_PID" "$PORT"
else
    echo "Server failed to start."
    exit 1
fi


###########################################

# mkdir -p /data/CHARM-MIMIC/.cache/huggingface/gguf/unsloth/DeepSeek-V3.2-GGUF/Q2_K
# cd /data/CHARM-MIMIC/.cache/huggingface/gguf/unsloth/DeepSeek-V3.2-GGUF/Q2_K

# BASE="https://huggingface.co/unsloth/DeepSeek-V3.2-GGUF/resolve/main/Q2_K"

# for i in 00001 00002 00003 00004 00005; do
#   wget -c --content-disposition \
#     "${BASE}/DeepSeek-V3.2-Q2_K-${i}-of-00005.gguf"
# done