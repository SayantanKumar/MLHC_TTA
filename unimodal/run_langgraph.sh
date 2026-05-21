set -e

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib":$LD_LIBRARY_PATH

summary_dir=../atp_tts/data/i2m4/notes
note_fname=2023-03-03_11-58-20_annotations_158331_yes_structured.txt
# note_fname=2024-01-12_11-16-46_annotations_105276.txt
# note_fname=2024-01-10_10-50-56_annotations_103859.txt

# 1. Running DS inside (single file). Change the GGUF path accordingly
# model_path=/usr3/data/LLM/gguf/DeepSeek-V3.1-UD-Q2_K_XL/DeepSeek-V3.1-UD-Q2_K_XL-00001-of-00006.gguf
# python langgraph_runner_unimodal.py \
#     --summary_file $summary_dir/$note_fname \
#     --model_path $model_path \
#     --log_folder "outputs/DeepSeek-V3.1/$(basename "$note_fname" .txt)" \
#     --debug \
#     --omit_tags think

# 2. Using LLM endpoint (single file)
python langgraph_runner_unimodal.py \
    --summary_file $summary_dir/$note_fname \
    --llm_endpoint http://localhost:8001 \
    --model_basename "DeepSeek-V3.1" \
    --log_folder "outputs/DeepSeek-V3.1/$(basename "$note_fname" .txt)" \
    --debug \
    --omit_tags think

# 3. Using LLM endpoint (multiple files)
# for fname in "$summary_dir"/*.txt; do
#     echo "Processing $fname"
#     log_folder="outputs/DeepSeek-V3.1/$(basename "$fname" .txt)"
#     # echo "$fname"
#     # echo $log_folder
#     python langgraph_runner_unimodal.py \
#         --summary_file "$fname" \
#         --llm_endpoint "http://localhost:8001" \
#         --model_basename "DeepSeek-V3.1" \
#         --log_folder $log_folder \
#         --debug \
#         --omit_tags think
# done