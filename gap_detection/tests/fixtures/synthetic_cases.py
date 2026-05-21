"""Synthetic test fixtures for gap detection tests."""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import tempfile
import os


def create_mock_batch_dir(events_by_case: Dict[str, List[Dict]], rag_results_by_case: Dict[str, List[Dict]], base_dir: Optional[str] = None) -> str:
    """
    Create a temporary batch directory structure with synthetic data.

    Args:
        events_by_case: Dict mapping case_id to list of events
        rag_results_by_case: Dict mapping case_id to list of RAG results
        base_dir: Optional base directory (uses temp dir if None)

    Returns:
        Path to created batch directory
    """
    if base_dir is None:
        base_dir = tempfile.mkdtemp(prefix="gap_detection_test_")

    batch_path = Path(base_dir)

    # Create directory structure
    charpos_dir = batch_path / "bundle" / "charpos"
    charpos_dir.mkdir(parents=True, exist_ok=True)

    comparison_dir = batch_path / "bundle" / "anchor_queries" / "comparison_results"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Write BSV timeline files
    for case_id, events in events_by_case.items():
        bsv_path = charpos_dir / f"{case_id}_positions_timeline_1.bsv"
        with open(bsv_path, 'w') as f:
            # Write header
            f.write("uid4|char_pos|char_pos_ub|mention|time|bounds|known|context_uid4s\n")

            # Write events
            for event in events:
                bounds_str = f"[{event['bounds'][0]}, {event['bounds'][1]}]"
                context_str = str(event.get('context_uid4s', []))
                line = f"{event['uid4']}|{event['char_pos']}|{event['char_pos_ub']}|{event['mention']}|{event['time']}|{bounds_str}|{event['known']}|{context_str}\n"
                f.write(line)

    # Write RAG results CSV files
    import pandas as pd
    for case_id, rag_results in rag_results_by_case.items():
        csv_path = comparison_dir / f"{case_id}_joined.csv"

        # Build DataFrame
        rows = []
        for result in rag_results:
            rows.append({
                'query_uid4': result['query_uid4'],
                'score': result['score'],
                'timestamp': result['timestamp'],
                'match_text': result['match_text'],
                'match_type': result.get('match_type', 'test_match'),
                'attributes': str(result.get('attributes', {}))
            })

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

    return str(batch_path)


def create_mock_event(
    uid4: str,
    mention: str,
    time: float,
    bounds: Optional[List[float]] = None,
    context_uid4s: Optional[List[str]] = None
) -> Dict:
    """
    Create a single textual event with minimal required fields.

    Args:
        uid4: 4-character hex identifier
        mention: Textual mention string
        time: Time in hours relative to admission
        bounds: [lower, upper] time bounds (optional)
        context_uid4s: List of context event IDs (optional)

    Returns:
        Event dictionary matching BSV timeline format
    """
    if bounds is None:
        bounds = [time, time]

    if context_uid4s is None:
        context_uid4s = []

    return {
        'uid4': uid4,
        'char_pos': int(uid4, 16) * 10,  # Mock character position
        'char_pos_ub': int(uid4, 16) * 10 + len(mention),
        'mention': mention,
        'time': str(time),
        'bounds': bounds,
        'known': True,
        'context_uid4s': context_uid4s
    }


def create_mock_rag_result(
    uid4: str,
    score: float,
    timestamp: float,
    match_text: str,
    match_type: str = 'test_match',
    attributes: Optional[Dict] = None
) -> Dict:
    """
    Create a RAG match result with specified score and timing.

    Args:
        uid4: Query event ID
        score: RAG similarity score (0.0-1.0)
        timestamp: Tabular event timestamp (Unix epoch)
        match_text: Matched tabular text
        match_type: Type of match (e.g., 'lab', 'diagnosis')
        attributes: Additional attributes (severity, location, etc.)

    Returns:
        RAG result dictionary matching comparison results format
    """
    if attributes is None:
        attributes = {}

    return {
        'query_uid4': uid4,
        'score': score,
        'timestamp': timestamp,
        'match_text': match_text,
        'match_type': match_type,
        'attributes': attributes
    }


def create_mock_case(
    case_id: str,
    events: List[Dict],
    rag_results: List[Dict],
    admission_time: Optional[datetime] = None
) -> Dict:
    """
    Assemble a complete test case from events and RAG results.

    Args:
        case_id: Case identifier
        events: List of textual events
        rag_results: List of RAG match results
        admission_time: Admission datetime (default: 2023-06-15 10:30:00)

    Returns:
        Complete case data structure for GapDetector
    """
    if admission_time is None:
        admission_time = datetime(2023, 6, 15, 10, 30, 0)

    return {
        'case_id': case_id,
        'events': events,
        'rag_results': rag_results,
        'admission_time': admission_time
    }