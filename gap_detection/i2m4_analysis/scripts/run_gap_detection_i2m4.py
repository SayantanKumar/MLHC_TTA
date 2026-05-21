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