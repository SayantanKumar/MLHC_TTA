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