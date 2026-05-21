#!/usr/bin/env python3
"""
Analysis and reporting utilities for textual-tabular gap detection.

This module provides functions to:
- Analyze gap patterns across patients
- Generate per-patient reports
- Create aggregate statistics
- Produce visualizations
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_results(results_path: str) -> Dict:
    """Load gap detection results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def create_per_patient_report(results: Dict, output_dir: str):
    """
    Create detailed per-patient gap reports.

    Generates:
    - per_patient_summary.csv: Summary table with gap statistics per patient
    - per_patient_details/: Directory with detailed JSON reports per patient
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    details_dir = output_path / 'per_patient_details'
    details_dir.mkdir(exist_ok=True)

    # Create summary table
    summary_rows = []

    for case_id, result in results.items():
        if 'error' in result:
            continue

        summary = result['gap_summary']

        # Extract patient ID
        patient_id = case_id.split('-')[0]

        row = {
            'case_id': case_id,
            'patient_id': patient_id,
            'total_mentions': summary['total_mentions'],
            'complete_absence': summary['gap_type_distribution'].get('complete_absence', 0),
            'detail_gap': summary['gap_type_distribution'].get('detail_gap', 0),
            'temporal_mismatch': summary['gap_type_distribution'].get('temporal_mismatch', 0),
            'semantic_distance': summary['gap_type_distribution'].get('semantic_distance', 0),
            'well_captured': summary['gap_type_distribution'].get('well_captured', 0),
            'high_relevance_gaps': summary['forecasting_relevance_distribution'].get('high', 0),
            'medium_relevance_gaps': summary['forecasting_relevance_distribution'].get('medium', 0),
            'low_relevance_gaps': summary['forecasting_relevance_distribution'].get('low', 0),
            'pct_with_gaps': summary['pct_with_gaps'],
            'pct_high_relevance_gaps': summary['pct_high_relevance_gaps']
        }
        summary_rows.append(row)

        # Create detailed report
        detail_file = details_dir / f'{case_id}_gap_report.json'
        with open(detail_file, 'w') as f:
            json.dump(result, f, indent=2)

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_path / 'per_patient_summary.csv'
    summary_df.to_csv(summary_file, index=False)

    print(f"Per-patient summary saved to: {summary_file}")
    print(f"Per-patient details saved to: {details_dir}/")

    return summary_df


def analyze_gap_patterns(results: Dict) -> Dict:
    """
    Analyze patterns in gap types across the dataset.

    Returns:
        Dictionary with pattern analysis results.
    """
    patterns = {
        'common_missing_details': defaultdict(int),
        'gap_by_mention_type': defaultdict(lambda: defaultdict(int)),
        'temporal_patterns': {
            'past_events': defaultdict(int),
            'admission_events': defaultdict(int),
            'future_events': defaultdict(int)
        }
    }

    for case_id, result in results.items():
        if 'error' in result or 'gap_analysis' not in result:
            continue

        for analysis in result['gap_analysis']:
            gap_type = analysis['gap_type']
            mention = analysis['mention']
            details = analysis['details']

            # Track missing detail types
            if details.get('missing_attributes'):
                for attr in details['missing_attributes']:
                    patterns['common_missing_details'][attr] += 1

            # Categorize mentions by type
            mention_lower = mention.lower()

            # Identify mention category
            if any(word in mention_lower for word in ['surgery', 'procedure', 'operation']):
                category = 'procedures'
            elif any(word in mention_lower for word in ['symptom', 'pain', 'fever', 'weakness']):
                category = 'symptoms'
            elif any(word in mention_lower for word in ['lab', 'test', 'result', 'level']):
                category = 'lab_results'
            elif any(word in mention_lower for word in ['medication', 'drug', 'prescription']):
                category = 'medications'
            else:
                category = 'other'

            patterns['gap_by_mention_type'][category][gap_type] += 1

            # Temporal patterns
            try:
                time_hours = float(analysis['textual_time']) if analysis['textual_time'] != 'N/A' else None
                if time_hours is not None:
                    if time_hours < -6:
                        patterns['temporal_patterns']['past_events'][gap_type] += 1
                    elif time_hours <= 6:
                        patterns['temporal_patterns']['admission_events'][gap_type] += 1
                    else:
                        patterns['temporal_patterns']['future_events'][gap_type] += 1
            except (ValueError, TypeError):
                pass

    # Convert defaultdicts to regular dicts
    patterns['common_missing_details'] = dict(patterns['common_missing_details'])
    patterns['gap_by_mention_type'] = {
        k: dict(v) for k, v in patterns['gap_by_mention_type'].items()
    }
    for key in patterns['temporal_patterns']:
        patterns['temporal_patterns'][key] = dict(patterns['temporal_patterns'][key])

    return patterns


def create_pattern_report(patterns: Dict, output_dir: str):
    """Create detailed pattern analysis report."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    report_lines = []

    report_lines.append("# Gap Pattern Analysis Report\n")

    # Common missing details
    report_lines.append("## Common Missing Detail Types\n")
    if patterns['common_missing_details']:
        for detail, count in sorted(
            patterns['common_missing_details'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            report_lines.append(f"- {detail}: {count} occurrences\n")
    else:
        report_lines.append("No detail gaps found.\n")

    # Gap distribution by mention type
    report_lines.append("\n## Gap Distribution by Mention Type\n")
    for category in ['symptoms', 'procedures', 'lab_results', 'medications', 'other']:
        if category in patterns['gap_by_mention_type']:
            report_lines.append(f"\n### {category.title()}\n")
            for gap_type, count in patterns['gap_by_mention_type'][category].items():
                report_lines.append(f"- {gap_type}: {count}\n")

    # Temporal patterns
    report_lines.append("\n## Temporal Patterns\n")
    for time_period in ['past_events', 'admission_events', 'future_events']:
        if patterns['temporal_patterns'][time_period]:
            report_lines.append(f"\n### {time_period.replace('_', ' ').title()}\n")
            for gap_type, count in patterns['temporal_patterns'][time_period].items():
                report_lines.append(f"- {gap_type}: {count}\n")

    # Save report
    report_file = output_path / 'pattern_analysis_report.md'
    with open(report_file, 'w') as f:
        f.writelines(report_lines)

    print(f"Pattern analysis report saved to: {report_file}")

    # Also save as JSON
    patterns_file = output_path / 'gap_patterns.json'
    with open(patterns_file, 'w') as f:
        json.dump(patterns, f, indent=2)

    print(f"Gap patterns JSON saved to: {patterns_file}")


def create_aggregate_statistics(results: Dict, output_dir: str):
    """Create aggregate statistics across all patients."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    stats = {
        'overall': {},
        'by_gap_type': {},
        'by_relevance': {},
        'by_patient': {}
    }

    total_mentions = 0
    gap_type_totals = defaultdict(int)
    relevance_totals = defaultdict(int)

    patient_stats = []

    for case_id, result in results.items():
        if 'error' in result or 'gap_summary' not in result:
            continue

        summary = result['gap_summary']
        total_mentions += summary['total_mentions']

        for gap_type, count in summary['gap_type_distribution'].items():
            gap_type_totals[gap_type] += count

        for relevance, count in summary['forecasting_relevance_distribution'].items():
            relevance_totals[relevance] += count

        patient_id = case_id.split('-')[0]
        patient_stats.append({
            'patient_id': patient_id,
            'case_id': case_id,
            'total_mentions': summary['total_mentions'],
            'pct_high_relevance_gaps': summary['pct_high_relevance_gaps']
        })

    # Overall statistics
    stats['overall'] = {
        'total_cases': len(results),
        'total_mentions': total_mentions,
        'avg_mentions_per_case': total_mentions / len(results) if results else 0
    }

    # By gap type
    stats['by_gap_type'] = {
        gap_type: {
            'count': count,
            'percentage': count / total_mentions * 100 if total_mentions > 0 else 0
        }
        for gap_type, count in gap_type_totals.items()
    }

    # By relevance
    stats['by_relevance'] = {
        relevance: {
            'count': count,
            'percentage': count / total_mentions * 100 if total_mentions > 0 else 0
        }
        for relevance, count in relevance_totals.items()
    }

    # Save statistics
    stats_file = output_path / 'aggregate_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Aggregate statistics saved to: {stats_file}")

    return stats


def identify_high_priority_cases(results: Dict, output_dir: str, top_n: int = 20):
    """
    Identify cases with highest forecasting-relevant gaps.

    Returns:
        List of (case_id, high_relevance_gap_count) tuples
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    high_priority = []

    for case_id, result in results.items():
        if 'error' in result or 'gap_summary' not in result:
            continue

        high_relevance_gaps = result['gap_summary']['forecasting_relevance_distribution'].get('high', 0)
        high_priority.append((case_id, high_relevance_gaps))

    # Sort by number of high-relevance gaps
    high_priority.sort(key=lambda x: x[1], reverse=True)

    # Save top N
    top_cases = high_priority[:top_n]
    priority_file = output_path / f'high_priority_cases_top{top_n}.json'

    priority_data = [
        {
            'case_id': case_id,
            'high_relevance_gap_count': count,
            'result': results[case_id]
        }
        for case_id, count in top_cases
    ]

    with open(priority_file, 'w') as f:
        json.dump(priority_data, f, indent=2)

    print(f"Top {top_n} high-priority cases saved to: {priority_file}")

    return top_cases


def generate_summary_visualization(results: Dict, output_dir: str):
    """Generate text-based summary visualization."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Compute statistics
    total_mentions = sum(
        r['gap_summary']['total_mentions']
        for r in results.values()
        if 'gap_summary' in r
    )

    gap_type_totals = defaultdict(int)
    for result in results.values():
        if 'gap_summary' in result:
            for gap_type, count in result['gap_summary']['gap_type_distribution'].items():
                gap_type_totals[gap_type] += count

    # Create ASCII visualization
    viz_lines = []
    viz_lines.append("=" * 80)
    viz_lines.append("TEXTUAL-TABULAR GAP DETECTION SUMMARY")
    viz_lines.append("=" * 80)
    viz_lines.append(f"\nTotal Cases: {len(results)}")
    viz_lines.append(f"Total Mentions: {total_mentions}")
    viz_lines.append("\n" + "-" * 80)
    viz_lines.append("GAP TYPE DISTRIBUTION")
    viz_lines.append("-" * 80 + "\n")

    for gap_type in ['complete_absence', 'detail_gap', 'temporal_mismatch', 'semantic_distance', 'well_captured']:
        count = gap_type_totals.get(gap_type, 0)
        pct = count / total_mentions * 100 if total_mentions > 0 else 0
        bar_length = int(pct / 2)  # Scale to 50 chars max
        bar = '█' * bar_length

        viz_lines.append(f"{gap_type:20s} | {bar:50s} | {count:5d} ({pct:5.1f}%)")

    viz_lines.append("\n" + "=" * 80)

    # Save visualization
    viz_file = output_path / 'summary_visualization.txt'
    with open(viz_file, 'w') as f:
        f.write('\n'.join(viz_lines))

    print(f"Summary visualization saved to: {viz_file}")

    # Also print to console
    print('\n' + '\n'.join(viz_lines))


def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python gap_analysis.py <results_file> <output_dir>")
        sys.exit(1)

    results_file = sys.argv[1]
    output_dir = sys.argv[2]

    print(f"=== Analyzing Gap Detection Results ===\n")
    print(f"Results file: {results_file}")
    print(f"Output directory: {output_dir}\n")

    # Load results
    print("Loading results...")
    results = load_results(results_file)

    # Create per-patient reports
    print("\nCreating per-patient reports...")
    summary_df = create_per_patient_report(results, output_dir)

    # Analyze patterns
    print("\nAnalyzing gap patterns...")
    patterns = analyze_gap_patterns(results)
    create_pattern_report(patterns, output_dir)

    # Create aggregate statistics
    print("\nComputing aggregate statistics...")
    stats = create_aggregate_statistics(results, output_dir)

    # Identify high-priority cases
    print("\nIdentifying high-priority cases...")
    top_cases = identify_high_priority_cases(results, output_dir)

    # Generate visualization
    print("\nGenerating summary visualization...")
    generate_summary_visualization(results, output_dir)

    print("\n=== Analysis Complete ===")


if __name__ == '__main__':
    main()