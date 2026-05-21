#!/usr/bin/env python3
"""
Verification toolkit for data scientists to validate gap detection results.

This script provides tools to:
1. Verify input data integrity
2. Check implementation correctness
3. Perform spot-checks on results
4. Analyze parameter sensitivity
5. Generate validation reports
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).parent))
from data_loading import load_batch_data, extract_uid4_from_anchor


class GapDetectionValidator:
    """Toolkit for validating gap detection results."""

    def __init__(self, batch_dir: str, results_file: str):
        """Initialize with batch data and gap detection results."""
        self.batch_dir = batch_dir
        self.results_file = results_file

        print("Loading data for validation...")
        self.data = load_batch_data(batch_dir)

        with open(results_file, 'r') as f:
            self.results = json.load(f)

        print(f"Loaded {len(self.data['timelines'])} timelines")
        print(f"Loaded {len(self.results)} gap detection results")

    def verify_input_integrity(self):
        """Check 1: Verify all input data was processed correctly."""
        print("\n" + "="*80)
        print("CHECK 1: INPUT DATA INTEGRITY")
        print("="*80)

        # Count timelines
        timeline_count = len(self.data['timelines'])
        result_count = len(self.results)

        print(f"\nTimelines in batch: {timeline_count}")
        print(f"Results generated: {result_count}")

        if timeline_count != result_count:
            print(f"⚠️  MISMATCH: {timeline_count - result_count} cases missing from results")

            # Find missing cases
            timeline_cases = set(self.data['timelines'].keys())
            result_cases = set(self.results.keys())
            missing = timeline_cases - result_cases

            if missing:
                print(f"\nMissing cases (first 5): {list(missing)[:5]}")
        else:
            print("✓ All timelines processed")

        # Check RAG results alignment
        rag_count = len(self.data['rag_results'])
        print(f"\nRAG results in batch: {rag_count}")

        # Check for cases with RAG data but no timeline
        rag_cases = set(self.data['rag_results'].keys())
        timeline_cases = set(self.data['timelines'].keys())

        rag_only = rag_cases - timeline_cases
        timeline_only = timeline_cases - rag_cases

        if rag_only:
            print(f"⚠️  {len(rag_only)} cases have RAG data but no timeline")
        if timeline_only:
            print(f"⚠️  {len(timeline_only)} cases have timeline but no RAG data")

        return {
            'timeline_count': timeline_count,
            'result_count': result_count,
            'match': timeline_count == result_count,
            'rag_only_cases': list(rag_only)[:10],
            'timeline_only_cases': list(timeline_only)[:10]
        }

    def verify_gap_type_classification(self):
        """Check 2: Verify gap type classification logic."""
        print("\n" + "="*80)
        print("CHECK 2: GAP TYPE CLASSIFICATION LOGIC")
        print("="*80)

        # Sample cases for manual verification
        sample_checks = []

        for case_id, result in list(self.results.items())[:3]:
            if 'gap_analysis' not in result:
                continue

            for analysis in result['gap_analysis'][:5]:
                uid4 = analysis['uid4']
                mention = analysis['mention']
                gap_type = analysis['gap_type']
                has_counterpart = analysis['coverage']['has_counterpart']
                best_score = analysis['coverage']['best_score']

                # Expected gap type based on rules
                expected_type = self._classify_gap_expected(
                    has_counterpart,
                    best_score,
                    analysis['details'].get('has_detail_gap', False),
                    analysis['temporal'].get('alignment', 'unknown')
                )

                is_correct = gap_type == expected_type

                sample_checks.append({
                    'case_id': case_id,
                    'uid4': uid4,
                    'mention': mention,
                    'expected': expected_type,
                    'actual': gap_type,
                    'correct': is_correct,
                    'has_counterpart': has_counterpart,
                    'best_score': best_score
                })

        # Print sample
        print(f"\nSample of {len(sample_checks)} classifications:")
        for check in sample_checks[:10]:
            status = "✓" if check['correct'] else "✗"
            print(f"{status} {check['mention'][:40]:40s} -> {check['actual']:20s} (expected: {check['expected']})")

        incorrect = [c for c in sample_checks if not c['correct']]
        if incorrect:
            print(f"\n⚠️  {len(incorrect)} misclassifications found in sample")
        else:
            print("\n✓ All sampled classifications correct")

        return {
            'sample_size': len(sample_checks),
            'correct': len(sample_checks) - len(incorrect),
            'incorrect': len(incorrect),
            'details': sample_checks
        }

    def _classify_gap_expected(self, has_counterpart, best_score, has_detail_gap, alignment):
        """Expected classification logic (independent implementation)."""
        if not has_counterpart:
            return 'complete_absence'
        if has_detail_gap:
            return 'detail_gap'
        if alignment == 'misaligned':
            return 'temporal_mismatch'
        if best_score < 0.6:
            return 'semantic_distance'
        return 'well_captured'

    def spot_check_complete_absence(self, n_cases=5):
        """Check 3: Manually verify complete absence gaps."""
        print("\n" + "="*80)
        print("CHECK 3: SPOT-CHECK COMPLETE ABSENCE GAPS")
        print("="*80)

        # Find cases with complete_absence gaps
        complete_absence_samples = []

        for case_id, result in self.results.items():
            if 'gap_analysis' not in result:
                continue

            for analysis in result['gap_analysis']:
                if analysis['gap_type'] == 'complete_absence':
                    complete_absence_samples.append({
                        'case_id': case_id,
                        'uid4': analysis['uid4'],
                        'mention': analysis['mention'],
                        'coverage_score': analysis['coverage']['best_score']
                    })

                    if len(complete_absence_samples) >= n_cases:
                        break
            if len(complete_absence_samples) >= n_cases:
                break

        print(f"\nVerifying {len(complete_absence_samples)} complete_absence gaps:")
        print("\nManual verification steps:")
        print("1. Check RAG results file for each case")
        print("2. Search for uid4 in anchor field")
        print("3. Verify best_score is indeed < 0.3")

        for i, sample in enumerate(complete_absence_samples, 1):
            print(f"\n{i}. Case: {sample['case_id']}")
            print(f"   UID4: {sample['uid4']}")
            print(f"   Mention: {sample['mention']}")
            print(f"   Coverage Score: {sample['coverage_score']:.3f}")

            # Verify in RAG data
            if sample['case_id'] in self.data['rag_results']:
                df = self.data['rag_results'][sample['case_id']]

                # Find all anchors with this uid4
                matches = []
                for _, row in df.iterrows():
                    uid4 = extract_uid4_from_anchor(row['anchor'])
                    if uid4 == sample['uid4']:
                        matches.append({
                            'bestscore': row['bestscore'],
                            'event': row['line_level_event']
                        })

                if matches:
                    print(f"   Found {len(matches)} RAG matches:")
                    best_score = max(m['bestscore'] for m in matches)
                    print(f"   Best score from RAG: {best_score:.3f}")

                    if best_score < 0.3:
                        print("   ✓ Correctly classified as complete_absence")
                    else:
                        print(f"   ✗ ERROR: Score {best_score:.3f} >= 0.3, should not be complete_absence")
                else:
                    print("   No RAG matches found")
                    print("   ✓ Correctly classified as complete_absence")

        return complete_absence_samples

    def verify_relevance_scoring(self):
        """Check 4: Verify forecasting relevance scoring logic."""
        print("\n" + "="*80)
        print("CHECK 4: FORECASTING RELEVANCE SCORING")
        print("="*80)

        # Sample high-relevance gaps
        high_rel_samples = []

        for case_id, result in self.results.items():
            if 'gap_analysis' not in result:
                continue

            for analysis in result['gap_analysis']:
                if analysis['forecasting_relevance'] == 'high':
                    high_rel_samples.append({
                        'case_id': case_id,
                        'mention': analysis['mention'],
                        'gap_type': analysis['gap_type'],
                        'contains_keywords': any(
                            word in analysis['mention'].lower()
                            for word in ['symptom', 'pain', 'fever', 'weakness',
                                       'nausea', 'vomiting', 'diagnosis', 'surgery']
                        )
                    })

                    if len(high_rel_samples) >= 20:
                        break
            if len(high_rel_samples) >= 20:
                break

        print(f"\nVerifying {len(high_rel_samples)} high-relevance classifications:")
        print("\nExpected: High-relevance mentions should contain keywords like:")
        print("  symptom, pain, fever, weakness, nausea, vomiting, diagnosis, surgery")

        correct = 0
        for sample in high_rel_samples:
            if sample['contains_keywords']:
                correct += 1

        print(f"\n{correct}/{len(high_rel_samples)} ({correct/len(high_rel_samples)*100:.1f}%) contain expected keywords")

        # Show mismatches
        mismatches = [s for s in high_rel_samples if not s['contains_keywords']]
        if mismatches:
            print(f"\n⚠️  {len(mismatches)} high-relevance mentions without expected keywords:")
            for m in mismatches[:5]:
                print(f"  - {m['mention']}")

        return {
            'sample_size': len(high_rel_samples),
            'correct': correct,
            'accuracy': correct / len(high_rel_samples) if high_rel_samples else 0,
            'mismatches': mismatches[:10]
        }

    def analyze_parameter_sensitivity(self):
        """Check 5: Analyze sensitivity to threshold parameters."""
        print("\n" + "="*80)
        print("CHECK 5: PARAMETER SENSITIVITY ANALYSIS")
        print("="*80)

        # Test different RAG score thresholds
        thresholds = [0.2, 0.3, 0.4, 0.5]
        sensitivity_results = []

        for threshold in thresholds:
            complete_absence_count = 0

            for case_id, result in self.results.items():
                if 'gap_analysis' not in result:
                    continue

                for analysis in result['gap_analysis']:
                    best_score = analysis['coverage']['best_score']
                    if best_score < threshold:
                        complete_absence_count += 1

            total_mentions = sum(
                r['total_mentions'] for r in self.results.values()
                if 'total_mentions' in r
            )

            pct = complete_absence_count / total_mentions * 100 if total_mentions > 0 else 0

            sensitivity_results.append({
                'threshold': threshold,
                'complete_absence_count': complete_absence_count,
                'percentage': pct
            })

            print(f"\nThreshold {threshold}:")
            print(f"  Complete absence: {complete_absence_count} ({pct:.1f}%)")

        print(f"\nCurrent threshold: 0.3")
        print(f"Note: Higher threshold = more gaps classified as 'complete_absence'")

        return sensitivity_results

    def verify_statistical_distributions(self):
        """Check 6: Verify statistical distributions are sensible."""
        print("\n" + "="*80)
        print("CHECK 6: STATISTICAL DISTRIBUTION SANITY CHECKS")
        print("="*80)

        # Collect all gap types and relevance scores
        gap_type_counts = defaultdict(int)
        relevance_counts = defaultdict(int)

        for result in self.results.values():
            if 'gap_summary' not in result:
                continue

            for gap_type, count in result['gap_summary']['gap_type_distribution'].items():
                gap_type_counts[gap_type] += count

            for relevance, count in result['gap_summary']['forecasting_relevance_distribution'].items():
                relevance_counts[relevance] += count

        total = sum(gap_type_counts.values())

        print(f"\nTotal mentions: {total}")
        print(f"\nGap type distribution:")
        for gap_type, count in sorted(gap_type_counts.items()):
            pct = count / total * 100
            print(f"  {gap_type:25s}: {count:6d} ({pct:5.1f}%)")

        print(f"\nRelevance distribution:")
        for relevance, count in sorted(relevance_counts.items(), reverse=True):
            pct = count / total * 100
            print(f"  {relevance:25s}: {count:6d} ({pct:5.1f}%)")

        # Sanity checks
        issues = []

        # Check 1: Well_captured should be < 100%
        well_captured_pct = gap_type_counts.get('well_captured', 0) / total * 100
        if well_captured_pct > 90:
            issues.append(f"Well_captured percentage unusually high: {well_captured_pct:.1f}%")

        # Check 2: Complete_absence should be > 0
        if gap_type_counts.get('complete_absence', 0) == 0:
            issues.append("No complete_absence gaps found - unexpected")

        # Check 3: High-relevance gaps should be minority
        high_rel_pct = relevance_counts.get('high', 0) / total * 100
        if high_rel_pct > 50:
            issues.append(f"High-relevance gaps unusually high: {high_rel_pct:.1f}%")

        if issues:
            print(f"\n⚠️  Potential issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n✓ Distributions appear reasonable")

        return {
            'gap_type_counts': dict(gap_type_counts),
            'relevance_counts': dict(relevance_counts),
            'issues': issues
        }

    def generate_validation_report(self, output_file='validation_report.json'):
        """Generate comprehensive validation report."""
        print("\n" + "="*80)
        print("GENERATING VALIDATION REPORT")
        print("="*80)

        report = {
            'input_integrity': self.verify_input_integrity(),
            'classification_logic': self.verify_gap_type_classification(),
            'complete_absence_checks': self.spot_check_complete_absence(n_cases=5),
            'relevance_scoring': self.verify_relevance_scoring(),
            'parameter_sensitivity': self.analyze_parameter_sensitivity(),
            'statistical_distributions': self.verify_statistical_distributions()
        }

        # Save report
        output_path = Path(self.results_file).parent / output_file
        with open(output_path, 'w') as f:
            # Convert any non-serializable objects
            def convert(obj):
                if isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                if isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                return obj

            json.dump(report, f, indent=2, default=convert)

        print(f"\n✓ Validation report saved to: {output_path}")

        return report


def main():
    if len(sys.argv) < 3:
        print("Usage: python verification_checklist.py <batch_dir> <results_file>")
        print("\nExample:")
        print("  python verification_checklist.py symlinks/batch_output_0002 results/gap_detection_results.json")
        sys.exit(1)

    batch_dir = sys.argv[1]
    results_file = sys.argv[2]

    print("="*80)
    print("GAP DETECTION VALIDATION TOOLKIT")
    print("="*80)
    print(f"\nBatch directory: {batch_dir}")
    print(f"Results file: {results_file}")

    validator = GapDetectionValidator(batch_dir, results_file)
    report = validator.generate_validation_report()

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()