# LangGraph Timeline Reconstruction Scripts

This folder contains the Python and shell scripts used to run the timeline reconstruction workflow from the paper *Text Knows What, Tables Know When: Clinical Timeline Reconstruction via Retrieval-Augmented Multimodal Alignment*.

The scripts implement a graph-style clinical timeline pipeline:

1. Extract central timeline anchor events from a discharge summary.
2. Estimate pairwise time offsets between central events.
3. Reconstruct an initial central timeline.
4. Retrieve structured EHR rows for central events and update the central timeline.
5. Extract non-central events relative to central anchors.
6. Assemble a complete timeline.
7. Retrieve structured EHR rows for all events and update the final timeline.

The runners cache every stage as JSON, so interrupted or partially completed cases can resume from the first missing output file.

## Files

| File | Purpose |
| --- | --- |
| `langgraph_runner_multimodal_v3.py` | Main full multimodal multistep runner. Runs the default paper workflow with both central and final structured-EHR updates. |
| `langgraph_runner_multimodal_v3_ablations.py` | Variant runner for the full workflow, central-only update, and final-only update ablations. |
| `langgraph_runner_singlestep_ablation.py` | Single-step ablation runner. Extracts all events with timestamps in one prompt, then optionally updates the extracted timeline using structured EHR rows. |
| `mimic_struct_utils.py` | Loads MIMIC structured rows, converts row times to hours from admission, embeds structured rows, and retrieves top-k rows for an event. |
| `create_structured_timeline.py` | Post-processes case output folders into structured timeline JSON with `certain` and `certain_EHR` flags for downstream analysis. |
| `llm_server_utils.sh` | Starts and stops `llama-server` and waits until the OpenAI-compatible chat endpoint is ready. |
| `singlestep_anymodel.sh` | Slurm launcher for the single-step ablation. |
| `update_only_central_anymodel.sh` | Slurm launcher for the central-update-only ablation. |
| `update_only_final_anymodel.sh` | Slurm launcher for the final-update-only ablation. |
| `slurm_run.sh` | Older Slurm launcher template. It references legacy runner names and should be checked before use. |
| `test_server.sh` | Server test / older launch script. It also references older script paths and should be checked before use. |

Note: `langgraph_runner_multimodal.py` is not currently present in this `code_github/langgraph_scripts` folder. An older copy exists elsewhere in the repository tree under `multimodal_simple/`, but the current code here uses the `*_v3.py` runners.

## Relationship To The Paper

The default multimodal experiment corresponds to `langgraph_runner_multimodal_v3.py`:

| Paper step | Runner method | Output file |
| --- | --- | --- |
| Step 1: central-event extraction | `extract_central_events` | `central_events.json` |
| Step 2: pairwise central-event relations | `compute_time_distances` | `time_distances.json` |
| Step 3: initial central scaffold | `reconstruct_central_timeline` | `central_timeline.json` |
| Step 4: central scaffold calibration | `update_central_timeline` | `central_timeline_with_topk_rows.json`, `updated_central_timeline.json` |
| Step 5: non-central event extraction | `extract_non_central_events` | `non_central_events.json` |
| Step 6: full timeline assembly | `reconstruct_timeline` | `timeline.json` |
| Step 7: final timeline refinement | `update_timeline` | `timeline_with_topk_rows.json`, `updated_timeline.json` |

The ablation runner supports:

| Variant | Command value | Description |
| --- | --- | --- |
| Full multimodal | `--pipeline_variant full` | Updates both the central scaffold and final full timeline. |
| Central-only update | `--pipeline_variant update_only_central` | Updates the central scaffold, assembles the full timeline, and skips final full-timeline update. |
| Final-only update | `--pipeline_variant update_only_final` | Skips central scaffold update, assembles the full timeline from the initial central scaffold, then updates the final full timeline. |
| Single-step | `langgraph_runner_singlestep_ablation.py` | Bypasses central/non-central decomposition and extracts all events with time in one prompt. |

## Requirements

Python packages used across these scripts include:

- `langchain-community`
- `langchain-core`
- `requests`
- `pandas`
- `numpy`
- `torch`
- `transformers`
- `tqdm`
- `llama-cpp-python` when using local `LlamaCpp` instead of an HTTP endpoint

The shell launchers also assume:

- Slurm
- CUDA
- `llama-server` from `llama.cpp`
- `curl`
- `jq`
- GNU `timeout`

## Required Local Configuration

Several paths are hardcoded for the study environment and must be checked before running elsewhere.

### Structured EHR Data Paths

`mimic_struct_utils.py` loads these CSVs at import time:

- MIMIC-III structured timeline rows
- MIMIC-IV structured timeline rows
- MIMIC-III admissions
- MIMIC-IV admissions

Update the paths near the top of `mimic_struct_utils.py` before running in a new environment.

The structured retrieval code also loads the embedding model:

```text
mohammadkhodadad/MedTE-cl15-step-8000
```

It currently expects CUDA and calls `.cuda()` on the model and embeddings.

### Template Directory

The Python runners load prompts with:

```python
self.template_dir = "templates"
```

That path is resolved relative to the current working directory. In this `code_github` checkout, the prompt files are stored under `prompt_templates/`. Before running from `code_github`, either:

```bash
ln -s prompt_templates templates
```

or change `self.template_dir` in the runner to point to `prompt_templates`.

The single-step runner expects a template named:

```text
templates/extract_events_with_time.template
```

The code release also contains `prompt_templates/singlestep_extract_events_with_time.template`. If running the single-step script from `code_github`, make sure the expected filename exists or update `_load_template("extract_events_with_time")` accordingly.

### HADM IDs

For MIMIC-III, `hadm_id` is treated as a hospital admission ID. For the configured MIMIC-IV subset, the utility uses a small `subject_id -> hadm_id` mapping, so verify the identifier convention before adding new MIMIC-IV cases.

## Running A Single Case

Start an OpenAI-compatible LLM endpoint first, or let the Python script load a local GGUF with `LlamaCpp`.

Example with an existing `llama-server` endpoint:

```bash
python langgraph_scripts/langgraph_runner_multimodal_v3.py \
  --summary_file /path/to/case_123456.txt \
  --hadm_id 123456 \
  --llm_endpoint http://127.0.0.1:8001 \
  --model_basename DeepSeek-V3.2 \
  --log_folder /path/to/output/case_123456 \
  --omit_tags think
```

Example with a local GGUF model:

```bash
python langgraph_scripts/langgraph_runner_multimodal_v3.py \
  --summary_file /path/to/case_123456.txt \
  --hadm_id 123456 \
  --model_path /path/to/model.gguf \
  --log_folder /path/to/output/case_123456
```

For directory mode:

```bash
python langgraph_scripts/langgraph_runner_multimodal_v3.py \
  --summary_directory /path/to/notes \
  --summary_ext .txt \
  --llm_endpoint http://127.0.0.1:8001 \
  --model_basename DeepSeek-V3.2
```

In directory mode, the runner infers the ID from the last numeric substring in each filename.

## Running Ablations

Full multimodal pipeline:

```bash
python langgraph_scripts/langgraph_runner_multimodal_v3_ablations.py \
  --summary_file /path/to/case_123456.txt \
  --hadm_id 123456 \
  --llm_endpoint http://127.0.0.1:8001 \
  --model_basename DeepSeek-V3.2 \
  --log_folder /path/to/output/case_123456 \
  --pipeline_variant full
```

Central-only update:

```bash
python langgraph_scripts/langgraph_runner_multimodal_v3_ablations.py \
  --summary_file /path/to/case_123456.txt \
  --hadm_id 123456 \
  --llm_endpoint http://127.0.0.1:8001 \
  --model_basename DeepSeek-V3.2 \
  --log_folder /path/to/output/case_123456 \
  --pipeline_variant update_only_central
```

Final-only update:

```bash
python langgraph_scripts/langgraph_runner_multimodal_v3_ablations.py \
  --summary_file /path/to/case_123456.txt \
  --hadm_id 123456 \
  --llm_endpoint http://127.0.0.1:8001 \
  --model_basename DeepSeek-V3.2 \
  --log_folder /path/to/output/case_123456 \
  --pipeline_variant update_only_final
```

Single-step ablation:

```bash
python langgraph_scripts/langgraph_runner_singlestep_ablation.py \
  --summary_file /path/to/case_123456.txt \
  --hadm_id 123456 \
  --llm_endpoint http://127.0.0.1:8001 \
  --model_basename DeepSeek-V3.2 \
  --log_folder /path/to/output/case_123456 \
  --omit_tags think
```

The single-step runner writes `timeline.json`, retrieves structured rows for every extracted event, and writes `updated_timeline.json`.

## Using Slurm Launchers

The `*_anymodel.sh` scripts are examples for the Biowulf/Slurm environment used in the experiments. They:

1. Load CUDA and activate the TTA conda environment.
2. Configure `LLAMA_SERVER`, `MODEL_PATH`, `MODEL_ALIAS`, `PORT`, and `OUTPUT_DIR`.
3. Source `llm_server_utils.sh`.
4. Start `llama-server`.
5. Loop over note files.
6. Run the relevant Python runner.
7. Stop the server.

Before submitting, edit:

- `MODEL_PATH`
- `MODEL_ALIAS`
- `PORT`
- `OUTPUT_DIR`
- `summary_dir`
- the Python script path if running from `code_github` instead of `multimodal_simple`

Example:

```bash
sbatch update_only_central_anymodel.sh
```

## Output Files

### Full Multimodal Runner

`langgraph_runner_multimodal_v3.py` writes:

| File | Contents |
| --- | --- |
| `central_events.json` | List of central anchor events. |
| `time_distances.json` | Pairwise central-event offsets as `[event1, event2, e2_minus_e1, confidence]`. |
| `central_timeline.json` | Initial central event absolute times. |
| `non_central_events.json` | Non-central events with central anchor, relative time, and confidence. |
| `central_timeline_with_topk_rows.json` | Central timeline plus retrieved structured rows for each central event. |
| `updated_central_timeline.json` | Central timeline after structured-EHR calibration. |
| `timeline.json` | Complete assembled timeline before final structured-EHR refinement. |
| `timeline_with_topk_rows.json` | Complete timeline plus retrieved structured rows for every event. |
| `updated_timeline.json` | Final timeline after structured-EHR refinement. |

### Ablation Runner

The ablation runner writes the subset of output files required by the selected variant:

- `full`: same files as the full multimodal runner.
- `update_only_central`: stops after `timeline.json`; no `timeline_with_topk_rows.json` or `updated_timeline.json`.
- `update_only_final`: skips `central_timeline_with_topk_rows.json` and `updated_central_timeline.json`, but writes final `updated_timeline.json`.

### Single-Step Runner

`langgraph_runner_singlestep_ablation.py` writes:

- `timeline.json`
- `timeline_with_topk_rows.json`
- `updated_timeline.json`

It stores confidence from the initial single-step extraction and carries that confidence forward after structured-EHR time updates.

## Structured Timeline Post-Processing

`create_structured_timeline.py` converts per-case JSON outputs into structured records with analysis flags:

- `certain`: derived from model confidence, with high confidence treated as `1`.
- `certain_EHR`: `1` when the structured-EHR update changed the event time, otherwise `0`.

For each case folder it writes:

```text
structured_timelines_with_certain_and_ehr.json
```

At the parent folder level it writes:

```text
certain_summary_with_ehr.json
```

The script currently has a hardcoded `parent_folder` in `__main__`. Either edit that value or call `main(parent_folder)` from Python:

```bash
python -c "from create_structured_timeline import main; main('/path/to/output_root')"
```

Run that command from `langgraph_scripts` or add the folder to `PYTHONPATH`.

## Notes And Caveats

- The runners are graph-style step orchestrators, but the current v3 files use explicit `PIPELINE_STEPS` lists rather than importing LangGraph directly.
- Existing non-empty JSON files are preserved. Delete or empty a step output if you want that step to rerun.
- The prompt text says "Top-10 EHR rows", but the current v3 runners call `get_topk_struct_events(..., topk=5, max_times=5)`. Change those arguments if you need exactly ten retrieved rows.
- `mimic_struct_utils.py` loads large CSV files and the MedTE embedding model at import/runtime, so startup can be slow and GPU memory intensive.
- The shell scripts contain absolute `/data/CHARM-MIMIC/...` paths and may reference `multimodal_simple` paths. Adjust them before running from this `code_github` folder.
- If model output includes reasoning tags such as `<think>...</think>`, pass `--omit_tags think`; the parsers remove those tags before parsing BSV.
