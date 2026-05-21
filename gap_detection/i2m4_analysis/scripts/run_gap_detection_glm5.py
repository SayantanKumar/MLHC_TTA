#!/usr/bin/env python3
"""
Run gap detection on GLM-5 timelines for i2m4 dataset.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import gap_detection module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gap_detection import GapDetector


def main():
    """Run gap detection on GLM-5 timelines."""

    # Path to the batch structure we created with symlinks
    batch_dir = Path(__file__).parent.parent / "glm5_batch"
    output_dir = Path(__file__).parent.parent / "glm5_output" / "gap_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running gap detection on GLM-5 timelines...")
    print(f"Batch dir: {batch_dir}")
    print(f"Output dir: {output_dir}")

    # Initialize gap detector
    detector = GapDetector(str(batch_dir))

    # Get all case IDs
    case_ids = sorted(detector.data['timelines'].keys())
    print(f"Found {len(case_ids)} cases\n")

    # Run gap detection on each case
    all_results = []

    for i, case_id in enumerate(case_ids, 1):
        print(f"[{i}/{len(case_ids)}] Processing {case_id}...")

        try:
            result = detector.detect_gaps_for_case(case_id)
            all_results.append(result)

            # Save individual case result
            case_file = output_dir / f"{case_id}_gap_analysis.json"
            with open(case_file, 'w') as f:
                json.dump(result, f, indent=2)

            # Print summary
            summary = result.get('gap_summary', {})
            print(f"  Total mentions: {summary.get('total_mentions', 0)}")
            print(f"  Complete absence: {summary.get('complete_absence_count', 0)}")
            print(f"  Well captured: {summary.get('well_captured_count', 0)}")
            print(f"  Detail gaps: {summary.get('detail_gap_count', 0)}")
            print(f"  Semantic distance: {summary.get('semantic_distance_count', 0)}\n")

        except Exception as e:
            print(f"  ERROR: {e}\n")
            import traceback
            traceback.print_exc()

    # Save aggregate results
    aggregate_file = output_dir / "aggregate_gap_analysis.json"
    with open(aggregate_file, 'w') as f:
        json.dump({
            'num_cases': len(all_results),
            'cases': all_results
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    # Print aggregate statistics
    if all_results:
        total_mentions = sum(r.get('gap_summary', {}).get('total_mentions', 0) for r in all_results)
        total_complete_absence = sum(r.get('gap_summary', {}).get('complete_absence_count', 0) for r in all_results)
        total_well_captured = sum(r.get('gap_summary', {}).get('well_captured_count', 0) for r in all_results)

        print(f"\n{'='*60}")
        print("AGGREGATE STATISTICS")
        print(f"{'='*60}")
        print(f"Total cases: {len(all_results)}")
        print(f"Total mentions: {total_mentions}")
        print(f"Complete absence: {total_complete_absence} ({total_complete_absence/total_mentions*100:.1f}%)")
        print(f"Well captured: {total_well_captured} ({total_well_captured/total_mentions*100:.1f}%)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()