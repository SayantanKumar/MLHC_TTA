# Gap Detection Analysis for i2m4 Dataset: Design Document

**Date**: 2026-04-02
**Objective**: Characterize gaps in tabular data for forecasting applications using ground truth timelines, with sensitivity validation against GLM-5 extraction quality.

## Overview

This analysis compares textual-tabular gaps using two timeline sources:
1. **Ground truth**: Manually annotated timelines (n=20 cases)
2. **GLM-5 output**: Model-extracted timelines for the same cases

**Primary focus**: Characterize gaps in tabular data for forecasting tasks
**Secondary focus**: Validate whether GLM-5 extraction quality biases gap findings

## Architecture

### Working Directory Structure
```
gap_detection/
├── i2m4_analysis/
│   ├── ground_truth/
│   │   ├── timelines/          # Converted BSV files from ground truth CSV
│   │   └── results/            # Gap detection results for ground truth
│   ├── glm5_output/
│   │   ├── timelines/          # Symlink to i2m4b BSV files
│   │   └── results/            # Gap detection results for GLM-5
│   ├── comparison/
│   │   ├── extraction_quality/ # Recall, precision, mention-level comparison
│   │   ├── gap_consistency/    # Do same gaps appear?
│   │   └── reports/            # Final comparative analysis
│   └── scripts/
│       ├── convert_ground_truth_csv_to_bsv.py
│       ├── run_gap_detection_i2m4.py
│       ├── analyze_gaps_i2m4.py
│       ├── compare_extractions.py
│       ├── compare_gap_findings.py
│       └── generate_comparative_report.py
```

### Data Flow
1. Ground truth CSVs → converted to BSV → stored in `ground_truth/timelines/`
2. GLM-5 BSVs → symlinked from i2m4b to `glm5_output/timelines/`
3. Run gap detection on both → results in respective `results/` folders
4. Compare extractions and gaps → reports in `comparison/`

## Data Sources

### Ground Truth Data (Read-Only)
- **Timelines**: `/data/weissjc/data/mimic-iv/sample_i2m4/man_annotations_n20/charpos/*.csv.gz`
- **Tabular data**: `/data/weissjc/data/mimic-iv/sample_i2m4/man_annotations_n20/*.csv.gz`
- **Format**: CSV with columns `event,time,char_pos,char_pos_ub,subj_id`
- **Cases**: 20 patients

### GLM-5 Data (Read-Only, Symlinked)
- **Timelines**: `/data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/i2m4b/i2m4_batch_output_0001/bundle/charpos/*.bsv`
- **RAG results**: `/data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/i2m4b/i2m4_batch_output_0001/bundle/i2m4/i2m4/comparison_results/*_joined.csv`
- **Cases**: Same 20 patients as ground truth

## Implementation Components

### 1. Data Conversion
**Script**: `convert_ground_truth_csv_to_bsv.py`

Convert ground truth CSV to BSV format:
- Input: CSV files with `event,time,char_pos,char_pos_ub,subj_id`
- Output: BSV files with `uid4|char_pos|char_pos_ub|mention|time|bounds|known|context_uid4s`
- Keep ALL events including administrative (Admission, Discharge, etc.)
- Generate uid4 from char_pos (4-character hex)
- Set bounds = `[time, time]` (no uncertainty in ground truth)
- Set `known = 0`, `context_uid4s = []`

### 2. Gap Detection
**Script**: `run_gap_detection_i2m4.py`

Run gap detection pipeline on both timeline sources:
- Use existing `gap_detection.py` module
- Apply forecasting relevance scoring (same criteria as previous analysis)
- Generate RAG comparison results if needed
- Save results with same structure as previous analysis

### 3. Gap Analysis
**Script**: `analyze_gaps_i2m4.py`

Analogous to previous gap analysis, covering:
- Overall gap distribution
- Gap types by forecasting relevance (high/medium/low)
- Timing analysis (pre-admission, admission, early course, mid course, discharge)
- Gap categories: causal factors, temporal patterns, severity markers, denial statements
- Case-by-case gap profiles
- Value assessment for different forecasting tasks

Run separately for ground truth and GLM-5 results.

### 4. Extraction Comparison
**Script**: `compare_extractions.py`

Compare GLM-5 vs ground truth timelines:
- **Matching criteria**: (same text OR semantic similarity > threshold) AND time difference < 50% of smaller time value
- **Metrics**: Recall, precision, F1 by case and overall
- **Breakdowns**: By event type, timing, gap type
- **Outputs**:
  - Mentions only in ground truth (missed by GLM-5)
  - Mentions only in GLM-5 (hallucinations or over-extracted)
  - Mentions in both with different attributes

### 5. Gap Comparison
**Script**: `compare_gap_findings.py`

Compare gap detection results:
- Gap type distributions (chi-square test)
- Forecasting-relevant gap frequencies
- Identify gaps present in one analysis but not the other
- Case studies where conclusions differ
- Statistical significance tests

### 6. Report Generation
**Script**: `generate_comparative_report.py`

Synthesize findings into comprehensive report:
- **Primary**: Gap characterization using ground truth (gold standard)
  - What gaps exist in tabular data?
  - Which are forecasting-relevant?
  - When do they occur?
- **Secondary**: Sensitivity to GLM-5 extraction quality
  - Do findings hold up with GLM-5?
  - Where does extraction quality matter?
  - Bias assessment

## Analysis Methodology

### Primary Focus: Characterize Gaps in Tabular Data

Using ground truth as gold standard:
- What information is missing from tabular data?
- What is the distribution of gap types?
- Which clinical events are poorly captured?
- When do gaps occur in the clinical trajectory?
- Which gaps are forecasting-relevant?

### Secondary Focus: Sensitivity Validation

Compare GLM-5 vs ground truth to answer:
- **Robustness**: Do we identify the same gaps?
- **Impact**: Where does extraction quality affect conclusions?
- **Bias**: Does GLM-5 systematically miss certain gap types?

## Implementation Considerations

1. **Time units**: Ground truth times are in hours - verify consistency with GLM-5
2. **Embedding model**: Use same model for both to ensure comparable RAG scores
3. **Small dataset (n=20)**: Focus on qualitative insights; statistical tests may have low power
4. **Case IDs**: Some have DS suffixes (e.g., 10056200-DS-18) - handle correctly
5. **Read-only access**: No modifications to source data directories

## Testing Strategy

1. Test conversion on one case, verify BSV format
2. Run gap detection on converted ground truth, spot check results
3. Compare GLM-5 and ground truth for one case manually
4. Scale to all 20 cases

## Expected Outputs

1. Gap detection results for ground truth timelines
2. Gap detection results for GLM-5 timelines
3. Extraction quality metrics (recall, precision, F1)
4. Comparative analysis report
5. Visualization of gap distributions
6. Case studies illustrating key findings