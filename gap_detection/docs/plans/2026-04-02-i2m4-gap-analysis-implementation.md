# i2m4 Gap Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Characterize gaps in tabular data for forecasting using ground truth timelines and validate sensitivity to GLM-5 extraction quality.

**Architecture:** Convert ground truth CSVs to BSV format, run gap detection pipeline on both ground truth and GLM-5 timelines, compare extraction quality and gap findings to assess robustness.

**Tech Stack:** Python, pandas, numpy, existing gap_detection module, embedding extraction, RAG comparison

---

## Task 1: Set Up Directory Structure

**Files:**
- Create: `gap_detection/i2m4_analysis/ground_truth/timelines/`
- Create: `gap_detection/i2m4_analysis/ground_truth/results/`
- Create: `gap_detection/i2m4_analysis/glm5_output/timelines/`
- Create: `gap_detection/i2m4_analysis/glm5_output/results/`
- Create: `gap_detection/i2m4_analysis/comparison/extraction_quality/`
- Create: `gap_detection/i2m4_analysis/comparison/gap_consistency/`
- Create: `gap_detection/i2m4_analysis/comparison/reports/`
- Create: `gap_detection/i2m4_analysis/scripts/`

**Step 1: Create directory structure**

```bash
cd /data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection
mkdir -p i2m4_analysis/{ground_truth/{timelines,results},glm5_output/{timelines,results},comparison/{extraction_quality,gap_consistency,reports},scripts}
```

**Step 2: Verify directories created**

Run: `ls -R i2m4_analysis/`
Expected: Directory tree structure created

**Step 3: Commit**

```bash
git add i2m4_analysis/
git commit -m "chore: set up i2m4 analysis directory structure"
```

---

## Task 2: Create Symlinks to GLM-5 Data

**Files:**
- Create: `gap_detection/i2m4_analysis/glm5_output/timelines/*.bsv` (symlinks)
- Create: `gap_detection/i2m4_analysis/glm5_output/rag_results/` (symlink)

**Step 1: Create symlinks to GLM-5 timeline BSVs**

```bash
cd /data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis/glm5_output
ln -s /data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/i2m4b/i2m4_batch_output_0001/bundle/charpos/*.bsv timelines/
```

**Step 2: Create symlink to RAG results**

```bash
cd /data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis/glm5_output
ln -s /data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/i2m4b/i2m4_batch_output_0001/bundle/i2m4/i2m4/comparison_results rag_results
```

**Step 3: Verify symlinks**

Run: `ls -la timelines/ | head -5 && ls -la rag_results/ | head -5`
Expected: Symlinks pointing to correct locations

**Step 4: Commit**

```bash
git add i2m4_analysis/glm5_output/
git commit -m "chore: symlink GLM-5 data for i2m4 analysis"
```

---

## Task 3: Implement Ground Truth CSV to BSV Converter

**Files:**
- Create: `gap_detection/i2m4_analysis/scripts/convert_ground_truth_csv_to_bsv.py`

**Step 1: Write the converter script**

```python
#!/usr/bin/env python3
"""
Convert ground truth CSV timelines to BSV format.

Input: CSV files with columns: event,time,char_pos,char_pos_ub,subj_id
Output: BSV files with format: uid4|char_pos|char_pos_ub|mention|time|bounds|known|context_uid4s
"""

import gzip
import os
from pathlib import Path


def csv_to_bsv(csv_path: str, bsv_path: str) -> int:
    """
    Convert a single CSV timeline to BSV format.

    Args:
        csv_path: Path to input CSV file
        bsv_path: Path to output BSV file

    Returns:
        Number of events converted
    """
    events = []

    # Read CSV
    with gzip.open(csv_path, 'rt') if csv_path.endswith('.gz') else open(csv_path, 'r') as f:
        lines = f.readlines()

    # Skip header
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split(',')
        if len(parts) < 5:
            continue

        event = parts[0]
        time_str = parts[1]
        char_pos = int(parts[2])
        char_pos_ub = int(parts[3])
        # subj_id = parts[4]  # Not needed in BSV

        # Generate uid4 from char_pos (4-character hex)
        uid4 = format(char_pos, '04x')

        # Format time and bounds
        try:
            time_val = float(time_str)
            time_formatted = f"{time_val:.2f}"
            bounds = f"[{time_formatted},{time_formatted}]"
        except ValueError:
            time_formatted = "N/A"
            bounds = "[N/A,N/A]"

        # Format BSV line
        # Format: uid4|char_pos|char_pos_ub|mention|time|bounds|known|context_uid4s
        bsv_line = f"{uid4}|{char_pos}|{char_pos_ub}|{event}|{time_formatted}|{bounds}|0|[]"
        events.append(bsv_line)

    # Write BSV
    with open(bsv_path, 'w') as f:
        # Write header
        f.write("uid4|char_pos|char_pos_ub|mention|time|bounds|known|context_uid4s\n")
        # Write events
        for event in events:
            f.write(event + '\n')

    return len(events)


def main():
    """Convert all ground truth CSV timelines to BSV format."""
    # Paths
    csv_dir = Path("/data/weissjc/data/mimic-iv/sample_i2m4/man_annotations_n20/charpos")
    output_dir = Path(__file__).parent.parent / "ground_truth" / "timelines"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all CSV files
    csv_files = sorted(csv_dir.glob("*.csv.gz"))

    print(f"Found {len(csv_files)} ground truth CSV files")

    total_events = 0
    for csv_file in csv_files:
        # Output filename: replace .csv.gz with .bsv
        bsv_name = csv_file.stem.replace('.csv', '') + '.bsv'
        bsv_path = output_dir / bsv_name

        n_events = csv_to_bsv(str(csv_file), str(bsv_path))
        total_events += n_events

        print(f"Converted {csv_file.name} -> {bsv_name} ({n_events} events)")

    print(f"\nTotal: {len(csv_files)} files, {total_events} events")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
```

**Step 2: Run converter script**

Run: `cd /data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis/scripts && python convert_ground_truth_csv_to_bsv.py`
Expected: "Found 20 ground truth CSV files" followed by conversion messages

**Step 3: Verify output**

Run: `ls ../ground_truth/timelines/ | wc -l`
Expected: 20 BSV files

Run: `head -5 ../ground_truth/timelines/10056200-DS-18.bsv`
Expected: BSV format with header and event data

**Step 4: Commit**

```bash
git add i2m4_analysis/scripts/convert_ground_truth_csv_to_bsv.py
git add i2m4_analysis/ground_truth/timelines/
git commit -m "feat: convert ground truth CSV timelines to BSV format"
```

---

## Task 4: Implement Gap Detection Runner

**Files:**
- Create: `gap_detection/i2m4_analysis/scripts/run_gap_detection_i2m4.py`

**Step 1: Write the gap detection runner**

```python
#!/usr/bin/env python3
"""
Run gap detection on i2m4 timelines (both ground truth and GLM-5).
"""

import sys
from pathlib import Path

# Add parent directory to path to import gap_detection module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gap_detection import GapDetector


def run_gap_detection_on_timelines(timeline_dir: str, rag_dir: str, tabular_dir: str, output_dir: str, label: str):
    """
    Run gap detection on a set of timelines.

    Args:
        timeline_dir: Directory containing BSV timeline files
        rag_dir: Directory containing RAG comparison results (or None to compute)
        tabular_dir: Directory containing tabular data
        output_dir: Directory to save results
        label: Label for this run (e.g., "ground_truth" or "glm5")
    """
    import json
    import os

    timeline_dir = Path(timeline_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all BSV files
    bsv_files = sorted(timeline_dir.glob("*.bsv"))
    print(f"[{label}] Found {len(bsv_files)} timeline files")

    # For now, we'll use the existing gap detection infrastructure
    # by creating a temporary batch directory structure

    # This is a placeholder - the actual implementation will depend on
    # how we want to handle the RAG comparison step

    print(f"[{label}] Gap detection runner - placeholder for now")
    print(f"[{label}] Timeline dir: {timeline_dir}")
    print(f"[{label}] RAG dir: {rag_dir}")
    print(f"[{label}] Tabular dir: {tabular_dir}")
    print(f"[{label}] Output dir: {output_dir}")


def main():
    """Run gap detection on both ground truth and GLM-5 timelines."""

    # Ground truth
    run_gap_detection_on_timelines(
        timeline_dir="/data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis/ground_truth/timelines",
        rag_dir=None,  # Need to compute
        tabular_dir="/data/weissjc/data/mimic-iv/sample_i2m4/man_annotations_n20",
        output_dir="/data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis/ground_truth/results",
        label="ground_truth"
    )

    # GLM-5
    run_gap_detection_on_timelines(
        timeline_dir="/data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis/glm5_output/timelines",
        rag_dir="/data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis/glm5_output/rag_results",
        tabular_dir="/data/weissjc/data/mimic-iv/sample_i2m4/man_annotations_n20",
        output_dir="/data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis/glm5_output/results",
        label="glm5"
    )


if __name__ == "__main__":
    main()
```

**Step 2: Run placeholder script to verify paths**

Run: `cd /data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis/scripts && python run_gap_detection_i2m4.py`
Expected: Output showing placeholder messages with correct paths

**Step 3: Commit**

```bash
git add i2m4_analysis/scripts/run_gap_detection_i2m4.py
git commit -m "feat: add gap detection runner (placeholder)"
```

---

## Task 5: Analyze Ground Truth Timeline Structure

**Files:**
- Create: `gap_detection/i2m4_analysis/scripts/analyze_timelines.py`

**Step 1: Write analysis script**

```python
#!/usr/bin/env python3
"""
Analyze timeline structure for both ground truth and GLM-5.
"""

import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_loading import parse_bsv_timeline


def analyze_timeline_set(timeline_dir: str, label: str):
    """Analyze a set of BSV timelines."""
    timeline_dir = Path(timeline_dir)
    bsv_files = sorted(timeline_dir.glob("*.bsv"))

    print(f"\n{'='*80}")
    print(f"{label} TIMELINES")
    print(f"{'='*80}")
    print(f"Files: {len(bsv_files)}")

    all_events = []
    time_stats = []

    for bsv_file in bsv_files:
        events = parse_bsv_timeline(str(bsv_file))
        all_events.extend(events)

        for event in events:
            time_str = event['time']
            if time_str != 'N/A':
                try:
                    time_val = float(time_str)
                    time_stats.append(time_val)
                except ValueError:
                    pass

    print(f"Total events: {len(all_events)}")
    print(f"Events per file (avg): {len(all_events) / len(bsv_files):.1f}")

    if time_stats:
        print(f"\nTime statistics:")
        print(f"  Min: {min(time_stats):.2f} hours")
        print(f"  Max: {max(time_stats):.2f} hours")
        print(f"  Mean: {sum(time_stats)/len(time_stats):.2f} hours")

    # Event type distribution (top 20)
    event_types = Counter([e['mention'] for e in all_events])
    print(f"\nTop 20 event types:")
    for event, count in event_types.most_common(20):
        print(f"  {event[:50]:50s}: {count:5d}")


def main():
    """Analyze both timeline sets."""

    # Ground truth
    analyze_timeline_set(
        "/data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis/ground_truth/timelines",
        "GROUND TRUTH"
    )

    # GLM-5
    analyze_timeline_set(
        "/data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis/glm5_output/timelines",
        "GLM-5"
    )


if __name__ == "__main__":
    main()
```

**Step 2: Run analysis**

Run: `cd /data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis/scripts && python analyze_timelines.py`
Expected: Statistics for both timeline sets

**Step 3: Review output and note key differences**

Check the output to understand:
- How many events in each set?
- Are there systematic differences?
- What's the time range?

**Step 4: Commit**

```bash
git add i2m4_analysis/scripts/analyze_timelines.py
git commit -m "feat: add timeline analysis script"
```

---

## Task 6: Compare Extractions

**Files:**
- Create: `gap_detection/i2m4_analysis/scripts/compare_extractions.py`

**Step 1: Write comparison script**

```python
#!/usr/bin/env python3
"""
Compare GLM-5 extractions vs ground truth timelines.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_loading import parse_bsv_timeline
from embedding_extraction import get_embedding


def compute_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts."""
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)

    # Cosine similarity
    import numpy as np
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(similarity)


def match_events(gt_events: List[Dict], glm5_events: List[Dict]) -> Tuple[List, List, List]:
    """
    Match events between ground truth and GLM-5.

    Matching criteria: (same text OR semantic similarity > 0.8) AND time difference < 50% of smaller time

    Returns:
        - matched: List of (gt_event, glm5_event, similarity) tuples
        - gt_only: Events only in ground truth (missed by GLM-5)
        - glm5_only: Events only in GLM-5 (hallucinations)
    """
    matched = []
    gt_matched = set()
    glm5_matched = set()

    # For each GLM-5 event, find best matching ground truth event
    for i, glm5_event in enumerate(glm5_events):
        best_match = None
        best_similarity = 0.0

        glm5_time = glm5_event['time']
        if glm5_time == 'N/A':
            continue
        glm5_time = float(glm5_time)

        for j, gt_event in enumerate(gt_events):
            if j in gt_matched:
                continue

            gt_time = gt_event['time']
            if gt_time == 'N/A':
                continue
            gt_time = float(gt_time)

            # Check time constraint
            time_diff = abs(glm5_time - gt_time)
            min_time = min(abs(glm5_time), abs(gt_time))
            if min_time > 0 and time_diff / min_time > 0.5:
                continue

            # Check text similarity
            if glm5_event['mention'].lower() == gt_event['mention'].lower():
                similarity = 1.0
            else:
                similarity = compute_similarity(glm5_event['mention'], gt_event['mention'])

            if similarity > 0.8 and similarity > best_similarity:
                best_match = (j, gt_event, similarity)
                best_similarity = similarity

        if best_match:
            j, gt_event, sim = best_match
            matched.append((gt_event, glm5_event, sim))
            gt_matched.add(j)
            glm5_matched.add(i)

    # Events only in ground truth
    gt_only = [gt_events[i] for i in range(len(gt_events)) if i not in gt_matched]

    # Events only in GLM-5
    glm5_only = [glm5_events[i] for i in range(len(glm5_events)) if i not in glm5_matched]

    return matched, gt_only, glm5_only


def analyze_case(gt_bsv: str, glm5_bsv: str) -> Dict:
    """Analyze a single case."""
    gt_events = parse_bsv_timeline(gt_bsv)
    glm5_events = parse_bsv_timeline(glm5_bsv)

    matched, gt_only, glm5_only = match_events(gt_events, glm5_events)

    # Compute metrics
    recall = len(matched) / len(gt_events) if len(gt_events) > 0 else 0.0
    precision = len(matched) / len(glm5_events) if len(glm5_events) > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    return {
        'case_id': Path(gt_bsv).stem,
        'gt_events': len(gt_events),
        'glm5_events': len(glm5_events),
        'matched': len(matched),
        'gt_only': len(gt_only),
        'glm5_only': len(glm5_only),
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'matched_events': [(m[0]['mention'], m[1]['mention'], m[2]) for m in matched[:10]],  # Top 10
        'missed_events': [e['mention'] for e in gt_only[:10]],  # Top 10
        'hallucinated_events': [e['mention'] for e in glm5_only[:10]]  # Top 10
    }


def main():
    """Compare extractions for all cases."""

    gt_dir = Path("/data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis/ground_truth/timelines")
    glm5_dir = Path("/data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis/glm5_output/timelines")
    output_dir = Path(__file__).parent.parent / "comparison" / "extraction_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find matching pairs
    gt_files = {f.stem: f for f in gt_dir.glob("*.bsv")}
    glm5_files = {f.stem: f for f in glm5_dir.glob("*_positions_timeline_1.bsv")}

    # Extract case IDs (remove -DS-* suffix for matching)
    case_ids = set()
    for stem in gt_files.keys():
        case_id = stem.split('-DS-')[0] if '-DS-' in stem else stem
        case_ids.add(case_id)

    results = []

    for case_id in sorted(case_ids):
        # Find matching files
        gt_file = None
        for stem, path in gt_files.items():
            if stem == case_id or stem.startswith(case_id + '-DS-'):
                gt_file = path
                break

        glm5_file = None
        for stem, path in glm5_files.items():
            if stem.startswith(case_id):
                glm5_file = path
                break

        if gt_file and glm5_file:
            print(f"Analyzing {case_id}...")
            result = analyze_case(str(gt_file), str(glm5_file))
            results.append(result)

            print(f"  GT: {result['gt_events']}, GLM5: {result['glm5_events']}, Matched: {result['matched']}")
            print(f"  Recall: {result['recall']:.2%}, Precision: {result['precision']:.2%}, F1: {result['f1']:.2%}")

    # Save results
    output_file = output_dir / "extraction_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Compute overall metrics
    if results:
        avg_recall = sum(r['recall'] for r in results) / len(results)
        avg_precision = sum(r['precision'] for r in results) / len(results)
        avg_f1 = sum(r['f1'] for r in results) / len(results)

        print(f"\nOverall:")
        print(f"  Avg Recall: {avg_recall:.2%}")
        print(f"  Avg Precision: {avg_precision:.2%}")
        print(f"  Avg F1: {avg_f1:.2%}")


if __name__ == "__main__":
    main()
```

**Step 2: Run comparison**

Run: `cd /data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis/scripts && python compare_extractions.py`
Expected: Extraction quality metrics for each case

**Step 3: Commit**

```bash
git add i2m4_analysis/scripts/compare_extractions.py
git add i2m4_analysis/comparison/extraction_quality/
git commit -m "feat: implement extraction comparison analysis"
```

---

## Task 7: Generate Analysis Report

**Files:**
- Create: `gap_detection/i2m4_analysis/scripts/generate_report.py`

**Step 1: Write report generator**

```python
#!/usr/bin/env python3
"""
Generate comparative analysis report.
"""

import json
from pathlib import Path


def generate_report(extraction_results_file: str, output_file: str):
    """Generate markdown report."""

    with open(extraction_results_file, 'r') as f:
        results = json.load(f)

    report_lines = [
        "# i2m4 Gap Analysis Report",
        "",
        "## Extraction Quality: GLM-5 vs Ground Truth",
        "",
        "### Overall Metrics",
        "",
    ]

    if results:
        avg_recall = sum(r['recall'] for r in results) / len(results)
        avg_precision = sum(r['precision'] for r in results) / len(results)
        avg_f1 = sum(r['f1'] for r in results) / len(results)

        report_lines.extend([
            f"- **Average Recall**: {avg_recall:.2%}",
            f"- **Average Precision**: {avg_precision:.2%}",
            f"- **Average F1**: {avg_f1:.2%}",
            "",
            "### Per-Case Results",
            "",
            "| Case ID | GT Events | GLM5 Events | Matched | Recall | Precision | F1 |",
            "|---------|-----------|-------------|---------|--------|-----------|-----|",
        ])

        for r in results:
            report_lines.append(
                f"| {r['case_id']} | {r['gt_events']} | {r['glm5_events']} | {r['matched']} | "
                f"{r['recall']:.1%} | {r['precision']:.1%} | {r['f1']:.1%} |"
            )

        report_lines.extend([
            "",
            "### Key Findings",
            "",
            "#### Frequently Missed Events",
            "",
        ])

        # Collect missed events
        missed = []
        for r in results:
            missed.extend(r['missed_events'])

        from collections import Counter
        missed_counts = Counter(missed)

        report_lines.append("Top 10 events missed by GLM-5:")
        for event, count in missed_counts.most_common(10):
            report_lines.append(f"- {event}: {count} cases")

        report_lines.extend([
            "",
            "#### Hallucinated Events",
            "",
            "Top 10 events only in GLM-5:",
        ])

        hallucinated = []
        for r in results:
            hallucinated.extend(r['hallucinated_events'])

        hallucinated_counts = Counter(hallucinated)

        for event, count in hallucinated_counts.most_common(10):
            report_lines.append(f"- {event}: {count} cases")

    report_lines.extend([
        "",
        "## Next Steps",
        "",
        "1. Run gap detection on both ground truth and GLM-5 timelines",
        "2. Compare gap type distributions",
        "3. Assess whether extraction quality affects gap conclusions",
        "",
    ])

    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Report generated: {output_file}")


def main():
    """Generate the report."""

    extraction_file = Path(__file__).parent.parent / "comparison" / "extraction_quality" / "extraction_comparison.json"
    output_file = Path(__file__).parent.parent / "comparison" / "reports" / "i2m4_analysis_report.md"

    generate_report(str(extraction_file), str(output_file))


if __name__ == "__main__":
    main()
```

**Step 2: Generate report**

Run: `cd /data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/i2m4_analysis/scripts && python generate_report.py`
Expected: Report generated in comparison/reports/

**Step 3: Review report**

Run: `cat ../comparison/reports/i2m4_analysis_report.md`
Expected: Markdown report with extraction quality metrics

**Step 4: Commit**

```bash
git add i2m4_analysis/scripts/generate_report.py
git add i2m4_analysis/comparison/reports/
git commit -m "feat: generate comparative analysis report"
```

---

## Summary

This plan implements:
1. Directory structure setup
2. Symlinks to GLM-5 data
3. Ground truth CSV to BSV conversion
4. Gap detection runner (placeholder for extension)
5. Timeline analysis
6. Extraction comparison with semantic matching
7. Report generation

Next steps after this plan:
- Extend gap detection runner to actually run the pipeline
- Implement full gap analysis (analogous to previous work)
- Compare gap findings between ground truth and GLM-5
- Generate comprehensive comparative report with visualizations