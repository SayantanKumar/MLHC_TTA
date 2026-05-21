# Textual-Tabular Gap Detection

Multi-scale gap detection system for identifying textual mentions that are not well captured by tabular data in clinical records.

## Overview

This system analyzes batch_process_miv outputs to identify gaps between textual mentions in clinical narratives and their tabular counterparts in structured EHR data. The analysis focuses on gaps relevant for forecasting, particularly explanatory and causal factors.

## Gap Types

The system identifies four types of gaps:

1. **Complete Absence**: Textual mention has no corresponding tabular event
   - Example: "severe abdominal pain" mentioned in HPI but no symptom record in structured data

2. **Detail Gap**: Tabular event exists but lacks key details
   - Example: Surgery timestamp known but operation type/details missing
   - Detected missing attributes: severity, location, procedure_details

3. **Temporal Mismatch**: Event captured but timing misaligned (>12 hours difference)
   - Currently not detected in batch 0002 (no cases found)

4. **Semantic Distance**: Event exists but similarity below threshold
   - Example: Related tabular event exists but conceptually different enough that matching fails

## Results Summary (Batch 0002)

- **Total Cases**: 117
- **Total Mentions**: 20,815

### Gap Distribution
- **Complete Absence**: 10,792 (51.8%)
- **Well Captured**: 6,706 (32.2%)
- **Semantic Distance**: 2,650 (12.7%)
- **Detail Gap**: 667 (3.2%)

### Forecasting Relevance
- **High Relevance**: 1,435 (6.9%)
- **Medium Relevance**: 853 (4.1%)
- **Low Relevance**: 18,527 (89.0%)

### Common Missing Details
- Location: 370 occurrences
- Severity: 276 occurrences
- Procedure Details: 57 occurrences

## Usage

### Run Gap Detection

```bash
cd /data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/

# Run gap detection on batch 0002
python gap_detection.py symlinks/batch_output_0002 --output results/
```

### Generate Analysis Reports

```bash
# Generate per-patient reports, pattern analysis, and visualizations
python gap_analysis.py results/gap_detection_results.json reports/
```

## Output Files

### Gap Detection Results
- `results/gap_detection_results.json` - Full gap analysis results per case

### Analysis Reports
- `reports/per_patient_summary.csv` - Summary table with gap statistics per patient
- `reports/per_patient_details/` - Detailed JSON reports per patient
- `reports/pattern_analysis_report.md` - Pattern analysis across all cases
- `reports/gap_patterns.json` - Structured gap pattern data
- `reports/aggregate_statistics.json` - Overall statistics
- `reports/high_priority_cases_top20.json` - Top 20 cases with highest forecasting-relevant gaps
- `reports/summary_visualization.txt` - ASCII visualization of gap distribution

## Key Findings

### Symptoms
- 600 symptoms with complete absence gaps
- High forecasting relevance but often missing from tabular data
- Examples: "severe abdominal pain", "mild abdominal pain"

### Procedures
- 88 procedural mentions with complete absence
- 57 with detail gaps (missing procedure details)
- Examples: "Service: SURGERY" (detail gap), "seen by surgery" (complete absence)

### Temporal Patterns
- Past events (before -6h): 1,922 complete absence gaps
- Admission events (-6h to 6h): 5,134 complete absence gaps
- Future events (after 6h): 3,512 complete absence gaps

## Implementation

### Stage 1: Patient-Level Coverage Analysis
- Uses RAG scores from existing comparison results
- Threshold: score < 0.3 indicates no counterpart

### Stage 2: Temporal Alignment Analysis
- Compares textual time bounds with tabular timestamps
- Classifies as: aligned (≤6h), coarse (≤12h), misaligned (>12h)

### Stage 3: Detail Sufficiency Analysis
- Extracts attributes from textual mentions (severity, location, procedure details)
- Compares with tabular values
- Identifies missing attributes

### Stage 4: Gap Classification
- Combines results from stages 1-3 to classify gap type

### Stage 5: Forecasting Relevance Scoring
- High: symptoms, diagnoses, procedures, time-critical events
- Medium: lab results, medications, vital signs
- Low: background information, well-captured events

## Configuration

Thresholds can be adjusted in `gap_detection.py`:

```python
SEMANTIC_SIMILARITY_THRESHOLD = 0.7  # Cosine similarity for "good match"
TEMPORAL_ALIGNMENT_THRESHOLD = 6.0   # Hours for "aligned" events
TEMPORAL_COARSE_THRESHOLD = 12.0     # Hours for "coarse" events
RAG_SCORE_LOW_THRESHOLD = 0.3        # Low RAG score threshold
RAG_SCORE_MEDIUM_THRESHOLD = 0.6    # Medium RAG score threshold
```

## Dependencies

- Python 3.x
- pandas
- numpy
- Standard library: json, pathlib, collections, datetime, re

## Next Steps

1. **Validation**: Manual review of high-priority cases with clinicians
2. **Pattern Deep-Dive**: Analyze specific categories (e.g., all surgery-related gaps)
3. **Temporal Analysis**: Investigate why no temporal mismatches were detected
4. **Forecasting Integration**: Use gap classifications to weight textual vs tabular features
5. **Cross-Batch Analysis**: Apply to other batches to identify consistent patterns