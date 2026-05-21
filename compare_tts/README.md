# compare_tts

`compare_tts` compares manually annotated textual temporal annotations against one or more pilot/reference annotation directories. It matches event text across paired files, joins the matched events back to their times, computes comparison metrics, and writes diagnostic plots.

The main workflow is:

1. Read a JSON configuration file.
2. Pair manual and pilot files by shared basename.
3. Match manual events to pilot events using embedding cosine distance or string distance.
4. Join matched events back to their time values.
5. Compute match rate, time parse rate, concordance, and time discrepancy summaries.
6. Save event match CSV files and a multi-panel PDF of diagnostic plots.

## Files

| File | Purpose |
| --- | --- |
| `compare_tts_2.r` | Main R entrypoint. Reads config, validates inputs, runs or loads event matching, computes metrics, and writes plots. |
| `helper_comparer.r` | Matching utilities. Builds distance matrices, runs assignment matching, and supports string-only or featurized matching. |
| `distance_helper_v2.py` | Python helper called from R through `reticulate`; requests embeddings from the local server and writes cosine distance matrices. |
| `run_st_server.py` | FastAPI embedding server using `pritamdeka/S-PubMedBert-MS-MARCO`. |
| `comparer_runner.sh` | Convenience runner that starts the embedding server, runs the R comparison, then stops the server. |
| `config_simple.json` | Example configuration for the comparison pipeline. |

`bootstrap_comparer.r`, `bootstrap_runner.sh`, and `bootstrap_config_new.json` are related bootstrap-summary utilities in the same folder, but they are separate from the main comparison run documented here.

## Requirements

### R

The R scripts use these packages:

- `tidyverse`
- `glmnet`
- `survival`
- `jsonlite`
- `stringdist`
- `reticulate`
- `TreeDist`
- `Matrix`

Install missing packages from R, for example:

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

`helper_comparer.r` currently calls:

```r
use_condaenv("/opt/anaconda3/envs/Rtest")
```

Update that path or configure the matching conda environment before running on another machine.

### Python

The Python embedding server/helper uses:

- `fastapi`
- `uvicorn`
- `pydantic`
- `transformers`
- `torch`
- `numpy`
- `pandas`
- `requests`
- `scikit-learn`

Example conda setup:

```bash
conda create -n tts_compare python=3.10
conda activate tts_compare
pip install fastapi uvicorn pydantic transformers torch numpy pandas requests scikit-learn
```

The first server start may download the Hugging Face model `pritamdeka/S-PubMedBert-MS-MARCO`. Make sure the environment has access to the model cache or the internet for that initial download.

## Input Data

The pipeline compares one manual annotation directory with one or more pilot/reference directories.

Manual and pilot files are paired by basename after removing the configured suffix. For example:

```text
manual/
  patient_001.bsv
pilot/
  patient_001.bsv
```

Only files with matching basenames are compared.

### Standard `strcmp` Input

With the default `approach: "strcmp"`, each manual and pilot file is read as two columns:

| Column | Meaning |
| --- | --- |
| 1 | Event text |
| 2 | Event time |

For `.bsv` and `.bsv.gz` suffixes, the delimiter is `|`. Other suffixes are read with comma delimiters.

### Featurized Input

With `approach: "featurized"`, the pipeline also expects character-position columns:

- Manual files: event in column 1, time in column 2, plus `char_pos` and `char_pos_ub`.
- Pilot files: event in column 4, time in column 5, plus `char_pos` and `char_pos_ub`.

The featurized distance combines event-text distance with character-position distance using `approach.w`.

## Configuration

Run configuration is provided as JSON. `config_simple.json` is the main example.

```json
{
  "man_loc": "/path/to/manual/annotations/",
  "ref_locs": [
    "/path/to/pilot/annotations/"
  ],
  "pilot_names": [
    "model_name"
  ],
  "man_suffix": ".bsv",
  "ref_suffix": ".bsv",
  "event_match": true,
  "use_lap": true,
  "use_legacy_concordance": false,
  "out": {
    "relfolder": "matches/",
    "folder": "/tmp/",
    "event_out_locs": "/tmp/event_outputs/"
  },
  "distance": {
    "method": "embedding_cosine",
    "threshold": 0.1
  },
  "event_match_files": []
}
```

### Required Fields

| Field | Description |
| --- | --- |
| `man_loc` | Directory containing manual annotation files. |
| `ref_locs` | List of pilot/reference annotation directories to compare against `man_loc`. |
| `pilot_names` | Display names for each pilot/reference directory. Must have the same length as `ref_locs`. |

### Optional Fields

| Field | Default | Description |
| --- | --- | --- |
| `man_suffix` | `.bsv` | Suffix used to select manual files and strip basenames. |
| `ref_suffix` | `.bsv` | Suffix used to select pilot/reference files and strip basenames. |
| `man_header` | `false` | Whether manual files have column headers. |
| `ref_header` | `false` | Whether pilot/reference files have column headers. |
| `event_match` | No default in code; set explicitly | If `true`, compute new event matches. If `false`, load saved match files from `event_match_files`. |
| `event_match_files` | `[]` | CSV files containing previously saved event matches. Used when `event_match` is `false`. |
| `distance.method` | `embedding_cosine` | Distance method. Supported values are `embedding_cosine` and `string_similarity` in config validation; helper functions can also use `stringdist` methods such as `lv` if called directly. |
| `distance.threshold` | `0.1` | Maximum distance for a match to count as kept. Must be between 0 and 1. |
| `approach` | `strcmp` | Matching approach. Use `strcmp` for event text only or `featurized` for event text plus character positions. |
| `approach.w` | `NA` | Weights used by the featurized distance calculation, usually two values for text and position. |
| `out.relfolder` | `matches/` | Relative folder name used for match outputs. |
| `out.folder` | temp directory | Base output folder for figures. |
| `out.event_out_locs` | temp directory | Configured event output location; currently not central to the main output path. |
| `upper_limit` | `log(60*24*365.26)` | Upper limit used for time discrepancy AUC calculations. |
| `use_legacy_concordance` | `false` | If `true`, uses the older `tidyr::crossing` behavior for concordance pair enumeration. |
| `use_lap` | `true` in examples | Intended to control assignment matching. The current main call path uses LAPJV matching in `helper_comparer.r`. |

## Running

### Option 1: Convenience Runner

From this directory:

```bash
bash comparer_runner.sh path/to/config.json
```

The runner:

1. Starts `uvicorn run_st_server:app --host localhost`.
2. Runs `Rscript compare_tts_2.r "$CONFIG"`.
3. Kills the Uvicorn process after R finishes.

Check the hardcoded paths in `comparer_runner.sh` before using it on another machine or from a different repository location.

### Option 2: Manual Server and R Run

Start the embedding server in one terminal:

```bash
uvicorn run_st_server:app --host localhost --port 8000
```

Then run the comparison in another terminal:

```bash
Rscript compare_tts_2.r path/to/config.json
```

The embedding helper expects the server at:

```text
http://localhost:8000/embed
```

### Option 3: String Distance Without Embeddings

The matching helper can compute string-distance matrices with `stringdist`. For the main config path, use the configured distance method supported by `compare_tts_2.r` validation. If you adapt the helper functions directly, methods such as Levenshtein distance (`lv`) can be passed to `get_match_table()`.

## Outputs

### Event Match CSV

When `event_match` is `true`, the pipeline writes a match CSV inside each pilot/reference directory:

```text
<pilot_directory>/<out.relfolder>/best_matches2026-04-09.csv
```

The current filename is hardcoded in `compare_tts_2.r`.

Typical columns include:

| Column | Meaning |
| --- | --- |
| `common.bns` | Shared basename for the paired manual/pilot files. |
| `files` | Manual file path. |
| `pilot.files` | Pilot/reference file path. |
| `v1` | Manual event text. |
| `v2` | Matched pilot event text. |
| `error.rate` | Distance between matched events. |
| `idx` | Manual event row index. |
| `match.idx` | Pilot event row index. |

### Figures

The main script writes:

```text
<out.folder>/figures.pdf
```

The PDF includes:

- Event match distance ECDF.
- Concordance plot.
- Time discrepancy plot.
- Manual time vs. pilot time scatter plot.

### Returned Metrics

`compare_tts()` returns an R list with:

| Element | Description |
| --- | --- |
| `match_stats` | Match rate and matched-event count by pilot version. |
| `time_parse_rate` | Fraction of matched pilot times that parse as numeric. |
| `concordance_results` | Median and interquartile concordance by pilot version. |
| `time_discrepancy` | Log absolute time-error data and AUC summary inputs. |

When running with `Rscript`, these objects are returned inside R but are not automatically written as separate CSV files.

## Embedding API

`run_st_server.py` exposes one endpoint:

```http
POST /embed
```

Request body:

```json
{
  "sentences": ["event text one", "event text two"]
}
```

Response body:

```json
{
  "embeddings": [[0.01, 0.02], [0.03, 0.04]]
}
```

`distance_helper_v2.py` sends events to this endpoint, receives embeddings, computes pairwise cosine distances, and writes the resulting distance matrix to CSV for R to read.

## Troubleshooting

### Hardcoded Paths

Several scripts contain absolute local paths, including:

- `comparer_runner.sh`
- the fallback `script_dir` values in `compare_tts_2.r` and `helper_comparer.r`
- the conda environment path in `helper_comparer.r`

If files are not found, update these paths or run from the directory layout expected by the scripts.

### Port 8000 Already In Use

`distance_helper_v2.py` defaults to port `8000`, and `comparer_runner.sh` does not currently pass a custom port. Stop the process using port `8000`, or update both the server command and helper configuration.

### Embedding Model Download or Runtime Issues

The embedding server loads `pritamdeka/S-PubMedBert-MS-MARCO` through `transformers`. The first run can be slow because it may download model files. The server uses Apple Silicon `mps` when available and otherwise CPU.

### No Files Are Compared

Files are paired by basename after removing `man_suffix` and `ref_suffix`. Confirm that manual and pilot files share basenames and that the configured suffixes match the actual filenames.

### Time Metrics Are Missing or Sparse

Time metrics require numeric manual and pilot time values after matching. Non-numeric pilot times are excluded from parse-rate and concordance calculations or treated as missing depending on the metric.

### Header or Delimiter Problems

Set `man_header` and `ref_header` correctly for files with headers. Use `.bsv` or `.bsv.gz` suffixes for pipe-delimited files; other suffixes are read as comma-delimited.
