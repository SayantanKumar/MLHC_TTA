# Text Knows What, Tables Know When: Clinical Timeline Reconstruction via Retrieval-Augmented Multimodal Alignment

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.x-yellow.svg)](https://huggingface.co/transformers)
[![R](https://img.shields.io/badge/R-4.x-276DC3.svg)](https://www.r-project.org/)

</div>

## Overview

This repository contains the code for reconstructing absolute clinical timelines from narrative notes and structured EHR data. The core idea is that clinical text often knows *what* happened, while structured tables often know *when* measurable events happened. The pipeline extracts clinically meaningful events from discharge summaries, builds a text-derived temporal scaffold, and then uses retrieved structured EHR rows as temporal evidence to refine event timestamps.

The method follows the study workflow from *Text Knows What, Tables Know When: Clinical Timeline Reconstruction via Retrieval-Augmented Multimodal Alignment*. It is evaluated on the i2m4 benchmark spanning MIMIC-III and MIMIC-IV notes and structured records.

### Key Contributions

- **Multistep Timeline Reconstruction**: Decomposes timeline generation into central anchor extraction, central-event temporal graph construction, non-central event attachment, full timeline assembly, and final refinement.
- **Retrieval-Augmented Multimodal Alignment**: Retrieves structured EHR rows for text-derived events and uses them to calibrate event timing without replacing the richer narrative event content.
- **Stage-Specific Ablations**: Supports text-only, central-update-only, final-update-only, full multimodal, and single-step baseline workflows.
- **Evaluation Utilities**: Provides event matching, concordance, and log-time discrepancy/AULTC comparison scripts for predicted vs. manual timelines.
- **Gap Analysis**: Includes tools for studying which text-derived clinical events are absent, delayed, semantically distant, or insufficiently detailed in structured EHR records.

## Repository Structure

```text
code_github/
├── prompt_templates/      # LLM prompt templates for each workflow node
├── langgraph_scripts/     # Python runners, structured retrieval utilities, and Slurm launchers
├── compare_tts/           # Evaluation scripts for comparing predicted timelines to manual references
├── gap_detection/         # Auxiliary textual-tabular gap analysis and tests
└── README.md              # This file
```

Each main folder has its own README with more detailed usage notes:

- [`prompt_templates/README.md`](prompt_templates/README.md)
- [`langgraph_scripts/README.md`](langgraph_scripts/README.md)
- [`compare_tts/README.md`](compare_tts/README.md)
- [`gap_detection/README.md`](gap_detection/README.md)

## Workflow

The default multimodal pipeline is implemented in `langgraph_scripts/langgraph_runner_multimodal_v3.py`:

1. **Extract central events** from a discharge summary.
2. **Compute pairwise time distances** between central events.
3. **Reconstruct the initial central timeline** from the central-event graph.
4. **Retrieve structured EHR rows for central events** and update the central timeline.
5. **Extract non-central events** and place them relative to central anchors.
6. **Reconstruct the full text-derived timeline** from central and non-central events.
7. **Retrieve structured EHR rows for all events** and update the final patient trajectory.

The main output is a timestamped textual time series:

```text
event | time
admitted to hospital | 0
fever | -72
discharged home | 48
```

All times are in hours relative to admission when admission is available. Events before the reference time have negative timestamps.

## Quick Start

### 1. Clone And Install

```bash
git clone <repo-url>
cd code_github

conda create -n tta python=3.10
conda activate tta
pip install langchain-community langchain-core requests pandas numpy torch transformers tqdm scikit-learn fastapi uvicorn pydantic
```

For the R-based evaluation scripts:

```r
install.packages(c(
  "tidyverse",
  "glmnet",
  "survival",
  "jsonlite",
  "stringdist",
  "reticulate",
  "TreeDist",
  "Matrix"
))
```

### 2. Configure Local Paths

Several scripts contain study-environment paths and must be edited before running on a new machine:

- `langgraph_scripts/mimic_struct_utils.py`: MIMIC structured timeline CSVs and admissions CSVs.
- `compare_tts/comparer_runner.sh`: local script paths for the evaluation server and R runner.
- `compare_tts/helper_comparer.r`: `reticulate::use_condaenv(...)` environment path.
- Slurm launchers in `langgraph_scripts/`: model paths, output paths, conda environment, and note directory.

The Python runners load templates from a folder named `templates` relative to the current working directory:

```python
self.template_dir = "templates"
```

In this release the templates are stored in `prompt_templates/`. Before running from the repository root, either create a symlink:

```bash
ln -s prompt_templates templates
```

or edit the runner to use `prompt_templates`.

### 3. Run The Default Multimodal Pipeline

Start an OpenAI-compatible LLM endpoint, for example with `llama-server`, then run:

```bash
python langgraph_scripts/langgraph_runner_multimodal_v3.py \
  --summary_file /path/to/case_123456.txt \
  --hadm_id 123456 \
  --llm_endpoint http://127.0.0.1:8001 \
  --model_basename DeepSeek-V3.2 \
  --log_folder outputs/case_123456 \
  --omit_tags think
```

The runner writes one JSON file per workflow node and resumes from existing non-empty files.

### 4. Run Ablation Variants

```bash
# Full multimodal: central update + final update
python langgraph_scripts/langgraph_runner_multimodal_v3_ablations.py \
  --summary_file /path/to/case_123456.txt \
  --hadm_id 123456 \
  --llm_endpoint http://127.0.0.1:8001 \
  --model_basename DeepSeek-V3.2 \
  --log_folder outputs/case_123456_full \
  --pipeline_variant full

# Central-update-only
python langgraph_scripts/langgraph_runner_multimodal_v3_ablations.py \
  --summary_file /path/to/case_123456.txt \
  --hadm_id 123456 \
  --llm_endpoint http://127.0.0.1:8001 \
  --model_basename DeepSeek-V3.2 \
  --log_folder outputs/case_123456_central_only \
  --pipeline_variant update_only_central

# Final-update-only
python langgraph_scripts/langgraph_runner_multimodal_v3_ablations.py \
  --summary_file /path/to/case_123456.txt \
  --hadm_id 123456 \
  --llm_endpoint http://127.0.0.1:8001 \
  --model_basename DeepSeek-V3.2 \
  --log_folder outputs/case_123456_final_only \
  --pipeline_variant update_only_final
```

Single-step baseline:

```bash
python langgraph_scripts/langgraph_runner_singlestep_ablation.py \
  --summary_file /path/to/case_123456.txt \
  --hadm_id 123456 \
  --llm_endpoint http://127.0.0.1:8001 \
  --model_basename DeepSeek-V3.2 \
  --log_folder outputs/case_123456_singlestep \
  --omit_tags think
```

### 5. Evaluate Predicted Timelines

The `compare_tts/` folder compares predicted timelines against manual annotations using embedding-based event matching and temporal metrics.

```bash
cd compare_tts
bash comparer_runner.sh config_simple.json
```

The evaluation writes matched-event CSVs and diagnostic plots, including match-rate, concordance, and time-discrepancy summaries. See [`compare_tts/README.md`](compare_tts/README.md) for configuration details.

### 6. Analyze Textual-Tabular Gaps

The `gap_detection/` folder analyzes where textual events are not well represented in structured EHR data.

```bash
cd gap_detection
python gap_detection.py /path/to/batch_output --output results/
python gap_analysis.py results/gap_detection_results.json reports/
```

Gap categories include:

- `well_captured`
- `complete_absence`
- `temporal_mismatch`
- `semantic_distance`
- `detail_gap`

See [`gap_detection/README.md`](gap_detection/README.md) for details and [`gap_detection/tests/README.md`](gap_detection/tests/README.md) for the test suite.

## Main Outputs

A full multimodal run produces these per-case JSON artifacts:

| File | Description |
| --- | --- |
| `central_events.json` | Extracted central timeline anchors. |
| `time_distances.json` | Pairwise central-event offsets with confidence. |
| `central_timeline.json` | Initial central event absolute times. |
| `central_timeline_with_topk_rows.json` | Central events with retrieved structured EHR evidence. |
| `updated_central_timeline.json` | Structured-EHR-calibrated central timeline. |
| `non_central_events.json` | Non-central events attached to central anchors. |
| `timeline.json` | Initial complete timeline. |
| `timeline_with_topk_rows.json` | Complete timeline with retrieved structured EHR evidence. |
| `updated_timeline.json` | Final multimodal reconstructed timeline. |

`langgraph_scripts/create_structured_timeline.py` can convert output folders into structured analysis files with `certain` and `certain_EHR` flags.

## Data

This repository does not include MIMIC-III, MIMIC-IV, i2m4 notes, or structured EHR tables. These data require the appropriate credentialing and data-use agreements. The code assumes local access to:

- discharge summary or case note `.txt` files
- MIMIC-III/MIMIC-IV admissions tables
- structured EHR rows converted to the expected timeline CSV format
- manual/reference timeline annotations for evaluation

## Models

The study evaluates instruction-tuned LLMs through an OpenAI-compatible local server and uses sentence embeddings for structured retrieval and timeline comparison. The scripts include references to:

- GGUF models served by `llama.cpp` / `llama-server`
- `mohammadkhodadad/MedTE-cl15-step-8000` for structured EHR row retrieval
- `pritamdeka/S-PubMedBert-MS-MARCO` for event-matching evaluation in `compare_tts/`

Model weights are not included in this repository.

## Testing

The included formal tests are for the gap detection module:

```bash
cd gap_detection
pytest tests/
```

The timeline reconstruction and evaluation scripts are workflow scripts and generally require local data paths, GPU resources, and an LLM endpoint to run end-to-end.

## Notes And Caveats

- Many scripts contain absolute paths from the original study environment. Treat them as examples and edit them before running.
- The Python workflow runners expect strict BSV outputs from the LLM. Extra Markdown or explanatory text can break parsing, though some parsers strip `<think>...</think>` sections when `--omit_tags think` is used.
- The prompt text often says "Top-10 EHR rows"; current v3 Python runners retrieve `topk=5` with up to five times per row unless changed.
- The final GitHub folder is named `prompt_templates/`, while the runners expect `templates/` at runtime.
- The Slurm scripts are useful templates but include cluster-specific paths, partitions, GPU requests, email settings, and model locations.

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{kumar2026text,
  title={Text Knows What, Tables Know When: Clinical Timeline Reconstruction via Retrieval-Augmented Multimodal Alignment},
  author={Kumar, Sayantan and Noroozizadeh, Shahriar and Kim, Juyong and Weiss, Jeremy C},
  journal={arXiv preprint arXiv:2605.15168},
  year={2026}
}
```

