#!/usr/bin/env python3
"""
Generate comparative analysis report.
"""

import json
from pathlib import Path
from collections import Counter


def generate_report(extraction_results_file: str, output_file: str):
    """Generate markdown report."""

    with open(extraction_results_file, 'r') as f:
        data = json.load(f)

    report_lines = [
        "# i2m4 Gap Analysis Report",
        "",
        "## Extraction Quality: GLM-5 vs Ground Truth",
        "",
        "### Overall Metrics",
        "",
    ]

    if 'summary' in data:
        summary = data['summary']
        report_lines.extend([
            f"- **Average Recall**: {summary['avg_recall']:.2%}",
            f"- **Average Precision**: {summary['avg_precision']:.2%}",
            f"- **Average F1**: {summary['avg_f1']:.2%}",
            "",
            f"- **Number of cases analyzed**: {summary['num_cases']}",
            "",
            "### Per-Case Results",
            "",
            "| Case ID | GT Events | GLM5 Events | Matched | Recall | Precision | F1 |",
            "|---------|-----------|-------------|---------|--------|-----------|-----|",
        ])

        cases = data.get('cases', [])
        for case in cases:
            case_id = Path(case['gt_file']).stem
            report_lines.append(
                f"| {case_id} | {case['num_gt_events']} | {case['num_glm5_events']} | {case['num_matched']} | "
                f"{case['recall']:.1%} | {case['precision']:.1%} | {case['f1']:.1%} |"
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
        for case in cases:
            # unmatched_gt_indices means events in GT but not matched to GLM-5
            unmatched_indices = case.get('unmatched_gt_indices', [])
            if unmatched_indices and len(unmatched_indices) <= 10:
                # Just note how many were missed per case
                missed.append(f"Case {Path(case['gt_file']).stem}: {len(unmatched_indices)} events")

        report_lines.append(f"Total cases with missed events: {len(missed)}")
        for m in missed[:10]:
            report_lines.append(f"- {m}")

        report_lines.extend([
            "",
            "#### Hallucinated Events",
            "",
            "Events only in GLM-5 (not in ground truth):",
        ])

        hallucinated = []
        for case in cases:
            unmatched_glm5 = case.get('unmatched_glm5_indices', [])
            if unmatched_glm5:
                hallucinated.append(f"Case {Path(case['glm5_file']).stem}: {len(unmatched_glm5)} events")

        report_lines.append(f"Total cases with hallucinated events: {len(hallucinated)}")
        for h in hallucinated[:10]:
            report_lines.append(f"- {h}")

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