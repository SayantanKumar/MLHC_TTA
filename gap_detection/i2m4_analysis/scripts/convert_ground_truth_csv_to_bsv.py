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

        # Try to parse char_pos and char_pos_ub
        # Some lines might have different formats
        try:
            char_pos = int(float(parts[2]))
            char_pos_ub = int(float(parts[3]))
        except (ValueError, IndexError):
            # Skip lines with invalid char_pos/char_pos_ub
            continue

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