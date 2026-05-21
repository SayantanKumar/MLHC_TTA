#!/usr/bin/env python3
"""
Data loading utilities for textual-tabular gap detection.

This module provides functions to load batch_process_miv outputs:
- BSV timeline files (charpos/*.bsv)
- RAG comparison results (joined.csv)
- Comparison statistics (comparison_results.json)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


def parse_bsv_timeline(bsv_path: str) -> List[Dict]:
    """
    Parse a BSV timeline file into a list of dictionaries.

    Args:
        bsv_path: Path to BSV file

    Returns:
        List of event dictionaries with keys:
        - uid4: 4-character hex identifier
        - char_pos: Character position start
        - char_pos_ub: Character position upper bound
        - mention: Event text
        - time: Relative time (hours from anchor, or N/A)
        - bounds: Time uncertainty bounds [lower, upper]
        - known: Whether time is known (0/1)
        - context_uid4s: Related event UIDs (list)
    """
    events = []
    with open(bsv_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        return events

    # Parse header
    header = lines[0].strip().split('|')

    # Parse events
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split('|')
        if len(parts) < 8:
            continue

        # Parse context_uid4s: format is [uid1, uid2, ...] where uids are hex strings
        context_uid4s = []
        if parts[7] and parts[7] != '[]':
            # Remove brackets and split by comma
            context_str = parts[7].strip('[]')
            if context_str:
                # Split by comma and extract hex strings
                context_uid4s = [uid.strip().strip("'\"") for uid in context_str.split(',') if uid.strip()]

        event = {
            'uid4': parts[0],
            'char_pos': int(parts[1]),
            'char_pos_ub': int(parts[2]),
            'mention': parts[3],
            'time': parts[4],
            'bounds': parts[5],
            'known': int(parts[6]),
            'context_uid4s': context_uid4s
        }
        events.append(event)

    return events


def load_all_timelines(batch_dir: str) -> Dict[str, List[Dict]]:
    """
    Load all timeline BSV files from a batch output directory.

    Args:
        batch_dir: Path to batch_output_* directory

    Returns:
        Dictionary mapping case_id to list of events
    """
    charpos_dir = Path(batch_dir) / 'bundle' / 'charpos'

    if not charpos_dir.exists():
        raise FileNotFoundError(f"charpos directory not found: {charpos_dir}")

    timelines = {}
    # Only load timeline_1 files (canonical/final version)
    for bsv_file in sorted(charpos_dir.glob('*_positions_timeline_1.bsv')):
        # Extract case_id from filename
        # Format: {case_id}_positions_timeline_1.bsv
        case_id = bsv_file.stem.replace('_positions_timeline_1', '')

        events = parse_bsv_timeline(str(bsv_file))
        if events:
            timelines[case_id] = events

    return timelines


def load_rag_results(batch_dir: str, case_id: str) -> pd.DataFrame:
    """
    Load RAG comparison results (joined.csv) for a specific case.

    Args:
        batch_dir: Path to batch_output_* directory
        case_id: Case identifier (e.g., "10004648-DS-13")

    Returns:
        DataFrame with columns:
        - anchor: Query anchor text
        - summary: Matched tabular event summary
        - score: Similarity score
        - bestscore: Best similarity score
        - event: Event identifier (e.g., "<uid4> mention")
        - summary_id: Unique summary identifier
        - row_index: Row index in tabular data
        - line_level_timestamp: Timestamp of tabular event
        - line_level_event: Event type (e.g., "lab:blood-chemistry-anion gap")
        - line_level_value: Event value
        - hadm_id: Hospital admission ID
        - cutoff_time: Admission cutoff time
        - note_id: Note identifier
    """
    joined_path = Path(batch_dir) / 'bundle' / 'miv' / 'uidtts' / 'comparison_results' / f'{case_id}_joined.csv'

    if not joined_path.exists():
        raise FileNotFoundError(f"RAG results not found: {joined_path}")

    df = pd.read_csv(joined_path)
    return df


def load_all_rag_results(batch_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all RAG comparison results (joined.csv) from a batch.

    Args:
        batch_dir: Path to batch_output_* directory

    Returns:
        Dictionary mapping case_id to DataFrame
    """
    # Try multiple possible locations for comparison_results
    possible_paths = [
        Path(batch_dir) / 'bundle' / 'miv' / 'uidtts' / 'comparison_results',
        Path(batch_dir) / 'bundle' / 'i2m4' / 'i2m4' / 'comparison_results',
    ]

    comparison_dir = None
    for path in possible_paths:
        if path.exists():
            comparison_dir = path
            break

    if comparison_dir is None:
        raise FileNotFoundError(
            f"Comparison results directory not found. Tried: {possible_paths}"
        )

    rag_results = {}
    for joined_file in sorted(comparison_dir.glob('*_joined.csv')):
        case_id = joined_file.stem.replace('_joined', '')
        df = pd.read_csv(joined_file)
        rag_results[case_id] = df

    return rag_results


def load_admission_times(i2m4_dir: str = None) -> Dict[str, 'datetime']:
    """
    Load admission times for i2m4 cases from note_link file.

    Args:
        i2m4_dir: Path to i2m4 data directory. If None, uses default path.

    Returns:
        Dictionary mapping case_id to admission datetime
    """
    if i2m4_dir is None:
        i2m4_dir = '/data/weissjc/data/mimic-iv/sample_i2m4'

    note_link_path = Path(i2m4_dir) / 'note_link_n20.csv.gz'

    if not note_link_path.exists():
        raise FileNotFoundError(f"note_link file not found: {note_link_path}")

    import gzip
    admission_times = {}

    with gzip.open(note_link_path, 'rt') as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row['note_id']
            admittime_str = row['admittime']
            # Parse ISO format timestamp
            from datetime import datetime
            admittime = datetime.fromisoformat(admittime_str.replace('Z', '+00:00'))
            admission_times[case_id] = admittime

    return admission_times


def load_comparison_stats(batch_dir: str) -> List[Dict]:
    """
    Load comparison statistics from batch processing.

    Args:
        batch_dir: Path to batch_output_* directory

    Returns:
        List of comparison statistics dictionaries
    """
    stats_path = Path(batch_dir) / 'logs' / 'comparison_results.json'

    if not stats_path.exists():
        raise FileNotFoundError(f"Comparison stats not found: {stats_path}")

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    return stats


def extract_uid4_from_anchor(anchor: str) -> Optional[str]:
    """
    Extract uid4 from anchor query text.

    Args:
        anchor: Anchor query text (e.g., "Information regarding: <64f1> Anion Gap 10, in particular: ...")

    Returns:
        uid4 string (e.g., "64f1") or None if not found
    """
    import re
    match = re.search(r'<([0-9a-f]{4})>', anchor)
    if match:
        return match.group(1)
    return None


def extract_mention_from_anchor(anchor: str) -> Optional[str]:
    """
    Extract mention text from anchor query.

    Args:
        anchor: Anchor query text (e.g., "Information regarding: <64f1> Anion Gap 10, in particular: ...")

    Returns:
        Mention text (e.g., "Anion Gap 10") or None if not found
    """
    import re
    match = re.search(r'<[0-9a-f]{4}>\s*([^,]+)', anchor)
    if match:
        return match.group(1).strip()
    return None


def get_case_patient_id(case_id: str) -> str:
    """
    Extract patient ID from case ID.

    Args:
        case_id: Case identifier (e.g., "10004648-DS-13")

    Returns:
        Patient ID (e.g., "10004648")
    """
    return case_id.split('-')[0]


def load_batch_data(batch_dir: str) -> Dict:
    """
    Load all batch data for gap detection.

    Args:
        batch_dir: Path to batch_output_* directory

    Returns:
        Dictionary with keys:
        - timelines: Dict mapping case_id to events
        - rag_results: Dict mapping case_id to DataFrame
        - comparison_stats: List of comparison statistics
    """
    print(f"Loading batch data from: {batch_dir}")

    # Load timelines
    print("  Loading timelines...")
    timelines = load_all_timelines(batch_dir)
    print(f"    Loaded {len(timelines)} timelines")

    # Load RAG results
    print("  Loading RAG results...")
    rag_results = load_all_rag_results(batch_dir)
    print(f"    Loaded {len(rag_results)} RAG result sets")

    # Load comparison stats
    print("  Loading comparison statistics...")
    comparison_stats = load_comparison_stats(batch_dir)
    print(f"    Loaded {len(comparison_stats)} comparison records")

    return {
        'timelines': timelines,
        'rag_results': rag_results,
        'comparison_stats': comparison_stats
    }


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_loading.py <batch_dir>")
        sys.exit(1)

    batch_dir = sys.argv[1]
    data = load_batch_data(batch_dir)

    # Print summary
    print("\n=== Batch Data Summary ===")
    print(f"Timelines: {len(data['timelines'])} cases")
    print(f"RAG results: {len(data['rag_results'])} cases")
    print(f"Comparison stats: {len(data['comparison_stats'])} records")

    # Show sample timeline
    if data['timelines']:
        sample_case = list(data['timelines'].keys())[0]
        print(f"\nSample timeline ({sample_case}):")
        for event in data['timelines'][sample_case][:3]:
            print(f"  {event['uid4']}: {event['mention']} @ {event['time']}h")

    # Show sample RAG results
    if data['rag_results']:
        sample_case = list(data['rag_results'].keys())[0]
        print(f"\nSample RAG results ({sample_case}):")
        df = data['rag_results'][sample_case]
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")
        if len(df) > 0:
            print(f"  First anchor: {df['anchor'].iloc[0][:80]}...")