#!/usr/bin/env python3
"""
Multi-scale textual-tabular gap detection algorithm.

This module implements a 5-stage approach to identify gaps between
textual mentions and tabular data:

Stage 1: Patient-Level Coverage Analysis
Stage 2: Temporal Alignment Analysis
Stage 3: Detail Sufficiency Analysis
Stage 4: Gap Classification
Stage 5: Forecasting Relevance Scoring
"""

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from data_loading import (
    load_batch_data,
    extract_uid4_from_anchor,
    extract_mention_from_anchor,
    get_case_patient_id
)


# Configuration thresholds
SEMANTIC_SIMILARITY_THRESHOLD = 0.7  # Cosine similarity threshold for "good match"
TEMPORAL_ALIGNMENT_THRESHOLD = 6.0   # Hours - events within this window are "aligned"
TEMPORAL_COARSE_THRESHOLD = 12.0     # Hours - events within this window are "coarse"
RAG_SCORE_LOW_THRESHOLD = 0.3        # Low RAG score indicates poor match
RAG_SCORE_MEDIUM_THRESHOLD = 0.6    # Medium RAG score indicates partial match


class GapDetector:
    """Multi-scale gap detection for textual-tabular analysis."""

    def __init__(self, batch_dir: str, i2m4_dir: str = None):
        """Initialize with batch data.

        Args:
            batch_dir: Path to batch output directory
            i2m4_dir: Path to i2m4 data directory (for admission times). If None, uses default.
        """
        self.batch_dir = batch_dir
        self.data = load_batch_data(batch_dir)

        # Load admission times for temporal alignment
        from data_loading import load_admission_times
        try:
            self.admission_times = load_admission_times(i2m4_dir)
        except FileNotFoundError:
            print("Warning: Could not load admission times. Will use cutoff_time from RAG data.")
            self.admission_times = None

        # Build lookup structures
        self._build_lookups()

    def _build_lookups(self):
        """Build lookup structures for efficient access."""
        # Map (case_id, uid4) -> event dict
        self.timeline_events = {}
        for case_id, events in self.data['timelines'].items():
            for event in events:
                self.timeline_events[(case_id, event['uid4'])] = event

        # Map case_id -> RAG DataFrame
        self.rag_dfs = self.data['rag_results']

        # Map (case_id, uid4) -> list of RAG matches
        self.rag_matches = defaultdict(list)
        for case_id, df in self.rag_dfs.items():
            for _, row in df.iterrows():
                uid4 = extract_uid4_from_anchor(row['anchor'])
                if uid4:
                    self.rag_matches[(case_id, uid4)].append(row.to_dict())

    def detect_gaps_for_case(self, case_id: str) -> Dict:
        """
        Run full gap detection pipeline for a single case.

        Returns:
            Dictionary with keys:
            - case_id: Case identifier
            - total_mentions: Total number of textual mentions
            - gap_analysis: List of gap analysis results per mention
            - gap_summary: Summary statistics
        """
        # Get timeline for this case
        if case_id not in self.data['timelines']:
            return {
                'case_id': case_id,
                'error': 'Timeline not found'
            }

        events = self.data['timelines'][case_id]
        gap_analysis = []

        for event in events:
            uid4 = event['uid4']
            analysis = self.analyze_mention(case_id, uid4, event)
            gap_analysis.append(analysis)

        # Compute summary statistics
        gap_summary = self._compute_gap_summary(gap_analysis)

        return {
            'case_id': case_id,
            'total_mentions': len(events),
            'gap_analysis': gap_analysis,
            'gap_summary': gap_summary
        }

    def analyze_mention(self, case_id: str, uid4: str, event: Dict) -> Dict:
        """
        Analyze a single textual mention for gaps.

        Returns:
            Dictionary with gap analysis results.
        """
        mention = event['mention']
        textual_time = event['time']

        # Stage 1: Patient-Level Coverage
        coverage = self._analyze_coverage(case_id, uid4, mention)

        # Stage 2: Temporal Alignment
        temporal = self._analyze_temporal_alignment(
            case_id, uid4, textual_time, coverage
        )

        # Stage 3: Detail Sufficiency
        details = self._analyze_detail_sufficiency(
            case_id, uid4, mention, coverage
        )

        # Stage 4: Gap Classification
        gap_type = self._classify_gap(coverage, temporal, details)

        # Stage 5: Forecasting Relevance
        relevance = self._score_forecasting_relevance(
            case_id, uid4, event, gap_type
        )

        return {
            'uid4': uid4,
            'mention': mention,
            'textual_time': textual_time,
            'coverage': coverage,
            'temporal': temporal,
            'details': details,
            'gap_type': gap_type,
            'forecasting_relevance': relevance
        }

    def _analyze_coverage(self, case_id: str, uid4: str, mention: str) -> Dict:
        """
        Stage 1: Check if tabular counterpart exists.

        Returns:
            Dictionary with:
            - has_counterpart: bool
            - best_score: float (RAG score)
            - num_matches: int
            - match_types: list of matched event types
        """
        matches = self.rag_matches.get((case_id, uid4), [])

        if not matches:
            return {
                'has_counterpart': False,
                'best_score': 0.0,
                'num_matches': 0,
                'match_types': []
            }

        # Find best match
        best_score = max(m['bestscore'] for m in matches)
        match_types = list(set(m['line_level_event'] for m in matches))

        # Determine if counterpart exists based on RAG score
        has_counterpart = best_score >= RAG_SCORE_LOW_THRESHOLD

        return {
            'has_counterpart': has_counterpart,
            'best_score': best_score,
            'num_matches': len(matches),
            'match_types': match_types
        }

    def _analyze_temporal_alignment(
        self,
        case_id: str,
        uid4: str,
        textual_time: str,
        coverage: Dict
    ) -> Dict:
        """
        Stage 2: Check temporal alignment between textual and tabular events.

        Returns:
            Dictionary with:
            - alignment: 'aligned' | 'coarse' | 'misaligned' | 'no_counterpart'
            - temporal_distance: float (hours) or None
            - tabular_timestamps: list of tabular timestamps
        """
        if not coverage['has_counterpart']:
            return {
                'alignment': 'no_counterpart',
                'temporal_distance': None,
                'tabular_timestamps': []
            }

        # Parse textual time
        try:
            textual_time_hours = float(textual_time) if textual_time != 'N/A' else None
        except (ValueError, TypeError):
            textual_time_hours = None

        if textual_time_hours is None:
            return {
                'alignment': 'unknown',
                'temporal_distance': None,
                'tabular_timestamps': []
            }

        # Get tabular timestamps
        matches = self.rag_matches.get((case_id, uid4), [])
        tabular_timestamps = []
        temporal_distances = []

        for match in matches:
            timestamp_val = match.get('line_level_timestamp')
            if not timestamp_val:
                continue

            # Parse timestamp (could be Unix timestamp in seconds or ISO format)
            try:
                # Handle numeric timestamps (Unix seconds, not milliseconds)
                if isinstance(timestamp_val, (int, float)):
                    ts = datetime.fromtimestamp(timestamp_val)
                    tabular_timestamps.append(str(timestamp_val))
                # Handle ISO format strings
                elif isinstance(timestamp_val, str):
                    ts = datetime.fromisoformat(timestamp_val.replace('Z', '+00:00'))
                    tabular_timestamps.append(timestamp_val)
                else:
                    continue

                # Use admission time for temporal alignment (not cutoff_time)
                if self.admission_times and case_id in self.admission_times:
                    # Use admission time as anchor for time=0
                    admission_time = self.admission_times[case_id]
                    # Make admission_time timezone-naive for comparison
                    if admission_time.tzinfo is not None:
                        admission_time = admission_time.replace(tzinfo=None)
                    tabular_time_hours = (ts - admission_time).total_seconds() / 3600
                    temporal_distances.append(abs(textual_time_hours - tabular_time_hours))
                else:
                    # Fallback to cutoff_time if admission times not available
                    cutoff_str = match.get('cutoff_time')
                    if cutoff_str:
                        # Handle cutoff time formats (ISO string or Unix seconds)
                        if isinstance(cutoff_str, (int, float)):
                            cutoff = datetime.fromtimestamp(cutoff_str)
                        elif isinstance(cutoff_str, str):
                            cutoff = datetime.fromisoformat(cutoff_str.replace(' ', 'T'))
                        else:
                            continue

                        tabular_time_hours = (ts - cutoff).total_seconds() / 3600
                        temporal_distances.append(abs(textual_time_hours - tabular_time_hours))
            except (ValueError, TypeError):
                continue

        if not temporal_distances:
            return {
                'alignment': 'no_timestamp',
                'temporal_distance': None,
                'tabular_timestamps': tabular_timestamps
            }

        min_distance = min(temporal_distances)

        # Classify alignment
        if min_distance <= TEMPORAL_ALIGNMENT_THRESHOLD:
            alignment = 'aligned'
        elif min_distance <= TEMPORAL_COARSE_THRESHOLD:
            alignment = 'coarse'
        else:
            alignment = 'misaligned'

        return {
            'alignment': alignment,
            'temporal_distance': min_distance,
            'tabular_timestamps': tabular_timestamps
        }

    def _analyze_detail_sufficiency(
        self,
        case_id: str,
        uid4: str,
        mention: str,
        coverage: Dict
    ) -> Dict:
        """
        Stage 3: Check if key details are captured in tabular data.

        Returns:
            Dictionary with:
            - has_detail_gap: bool
            - missing_attributes: list of missing detail categories
            - tabular_values: list of tabular values
        """
        if not coverage['has_counterpart']:
            return {
                'has_detail_gap': False,
                'missing_attributes': [],
                'tabular_values': []
            }

        # Extract potential attributes from mention
        attributes = self._extract_attributes(mention)

        # Get tabular values
        matches = self.rag_matches.get((case_id, uid4), [])
        tabular_values = [m.get('line_level_value') for m in matches if m.get('line_level_value')]

        # Check for missing attributes
        missing = []
        mention_lower = mention.lower()

        # Check for severity qualifiers
        if any(word in mention_lower for word in ['severe', 'mild', 'moderate', 'acute', 'chronic']):
            if not any(self._has_severity(tab_val) for tab_val in tabular_values):
                missing.append('severity')

        # Check for procedure types
        if 'surgery' in mention_lower or 'procedure' in mention_lower:
            if not any(self._has_procedure_details(tab_val) for tab_val in tabular_values):
                missing.append('procedure_details')

        # Check for anatomical locations
        if any(word in mention_lower for word in ['left', 'right', 'bilateral', 'upper', 'lower']):
            if not any(self._has_location(tab_val) for tab_val in tabular_values):
                missing.append('location')

        return {
            'has_detail_gap': len(missing) > 0,
            'missing_attributes': missing,
            'tabular_values': tabular_values[:5]  # Limit to first 5
        }

    def _extract_attributes(self, mention: str) -> Dict:
        """Extract key attributes from mention text."""
        attributes = {}

        # Extract severity
        severity_pattern = r'\b(severe|mild|moderate|acute|chronic|critical)\b'
        severity_match = re.search(severity_pattern, mention, re.IGNORECASE)
        if severity_match:
            attributes['severity'] = severity_match.group(1).lower()

        # Extract numeric values
        numeric_pattern = r'\b(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)?\b'
        numeric_matches = re.findall(numeric_pattern, mention)
        if numeric_matches:
            attributes['numeric_values'] = numeric_matches

        return attributes

    def _has_severity(self, value: str) -> bool:
        """Check if value contains severity information."""
        if not value:
            return False
        severity_words = ['severe', 'mild', 'moderate', 'acute', 'chronic', 'critical']
        return any(word in str(value).lower() for word in severity_words)

    def _has_procedure_details(self, value: str) -> bool:
        """Check if value contains procedure details."""
        if not value:
            return False
        # Heuristic: if value is longer than 20 chars, it might have details
        return len(str(value)) > 20

    def _has_location(self, value: str) -> bool:
        """Check if value contains location information."""
        if not value:
            return False
        location_words = ['left', 'right', 'bilateral', 'upper', 'lower', 'anterior', 'posterior']
        return any(word in str(value).lower() for word in location_words)

    def _classify_gap(
        self,
        coverage: Dict,
        temporal: Dict,
        details: Dict
    ) -> str:
        """
        Stage 4: Classify gap type.

        Returns:
            Gap type: 'complete_absence' | 'detail_gap' | 'temporal_mismatch' | 'semantic_distance' | 'well_captured'
        """
        # No tabular counterpart
        if not coverage['has_counterpart']:
            return 'complete_absence'

        # Has counterpart but missing details
        if details['has_detail_gap']:
            return 'detail_gap'

        # Temporal mismatch
        if temporal['alignment'] == 'misaligned':
            return 'temporal_mismatch'

        # Semantic distance (has counterpart but low similarity)
        if coverage['best_score'] < RAG_SCORE_MEDIUM_THRESHOLD:
            return 'semantic_distance'

        # Well captured
        return 'well_captured'

    def _score_forecasting_relevance(
        self,
        case_id: str,
        uid4: str,
        event: Dict,
        gap_type: str
    ) -> str:
        """
        Stage 5: Score forecasting relevance.

        Returns:
            Relevance score: 'high' | 'medium' | 'low'
        """
        # Well-captured events have low forecasting relevance
        if gap_type == 'well_captured':
            return 'low'

        mention = event['mention'].lower()

        # High relevance: symptoms, diagnoses, time-critical events
        high_keywords = [
            'symptom', 'pain', 'fever', 'weakness', 'nausea', 'vomiting',
            'diagnosis', 'condition', 'syndrome', 'failure',
            'emergency', 'urgent', 'acute', 'severe',
            'surgery', 'procedure', 'operation',
            'complication', 'adverse', 'reaction'
        ]

        if any(keyword in mention for keyword in high_keywords):
            return 'high'

        # Medium relevance: lab results, medications, vital signs
        medium_keywords = [
            'lab', 'test', 'result', 'level', 'count',
            'medication', 'drug', 'dose', 'prescription',
            'blood', 'pressure', 'heart', 'rate', 'temp'
        ]

        if any(keyword in mention for keyword in medium_keywords):
            return 'medium'

        # Default: low relevance
        return 'low'

    def _compute_gap_summary(self, gap_analysis: List[Dict]) -> Dict:
        """Compute summary statistics for gap analysis."""
        total = len(gap_analysis)

        gap_type_counts = defaultdict(int)
        relevance_counts = defaultdict(int)

        for analysis in gap_analysis:
            gap_type_counts[analysis['gap_type']] += 1
            relevance_counts[analysis['forecasting_relevance']] += 1

        return {
            'total_mentions': total,
            'gap_type_distribution': dict(gap_type_counts),
            'forecasting_relevance_distribution': dict(relevance_counts),
            'pct_with_gaps': sum(gap_type_counts.values()) / total if total > 0 else 0.0,
            'pct_high_relevance_gaps': (
                relevance_counts.get('high', 0) / total if total > 0 else 0.0
            )
        }

    def detect_all_gaps(self) -> Dict[str, Dict]:
        """Run gap detection on all cases in batch."""
        results = {}

        for case_id in sorted(self.data['timelines'].keys()):
            print(f"Processing {case_id}...")
            results[case_id] = self.detect_gaps_for_case(case_id)

        return results


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gap_detection.py <batch_dir> [--output output_dir]")
        sys.exit(1)

    batch_dir = sys.argv[1]
    output_dir = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == '--output' else 'results'

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"=== Running Gap Detection on {batch_dir} ===\n")

    # Initialize detector
    detector = GapDetector(batch_dir)

    # Run detection
    results = detector.detect_all_gaps()

    # Save results
    output_file = output_path / 'gap_detection_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Results saved to {output_file} ===")

    # Print summary
    print("\n=== Batch Summary ===")
    total_mentions = sum(r['total_mentions'] for r in results.values() if 'total_mentions' in r)

    gap_type_totals = defaultdict(int)
    relevance_totals = defaultdict(int)

    for result in results.values():
        if 'gap_summary' in result:
            for gap_type, count in result['gap_summary']['gap_type_distribution'].items():
                gap_type_totals[gap_type] += count
            for relevance, count in result['gap_summary']['forecasting_relevance_distribution'].items():
                relevance_totals[relevance] += count

    print(f"Total cases: {len(results)}")
    print(f"Total mentions: {total_mentions}")
    print(f"\nGap type distribution:")
    for gap_type, count in sorted(gap_type_totals.items()):
        pct = count / total_mentions * 100 if total_mentions > 0 else 0
        print(f"  {gap_type}: {count} ({pct:.1f}%)")

    print(f"\nForecasting relevance distribution:")
    for relevance, count in sorted(relevance_totals.items(), reverse=True):
        pct = count / total_mentions * 100 if total_mentions > 0 else 0
        print(f"  {relevance}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()