# Verification Guide for Data Scientists

## How to Review and Validate Gap Detection Results

This guide provides a systematic approach for data scientists to independently verify the gap detection implementation and results.

---

## Quick Start

```bash
cd /data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection/

# Run automated verification
python verification_checklist.py symlinks/batch_output_0002 results/gap_detection_results.json

# View validation report
cat results/validation_report.json | jq .
```

---

## Verification Protocol

### Phase 1: Data Integrity Checks

#### 1.1 Input Data Verification

**What to check:**
- All input files were read correctly
- No data corruption during loading
- Timeline counts match RAG result counts

**How to verify:**

```python
from data_loading import load_batch_data

data = load_batch_data('symlinks/batch_output_0002')

# Check timeline files
print(f"Timelines loaded: {len(data['timelines'])}")
for case_id, events in list(data['timelines'].items())[:3]:
    print(f"  {case_id}: {len(events)} events")
    if events:
        print(f"    Sample: {events[0]['mention']}")

# Check RAG results
print(f"\nRAG results loaded: {len(data['rag_results'])}")
for case_id, df in list(data['rag_results'].items())[:3]:
    print(f"  {case_id}: {len(df)} rows")
    print(f"    Columns: {list(df.columns)[:5]}")

# Check for missing data
timeline_cases = set(data['timelines'].keys())
rag_cases = set(data['rag_results'].keys())

missing_rag = timeline_cases - rag_cases
if missing_rag:
    print(f"\n⚠️  {len(missing_rag)} cases have timeline but no RAG data")
    print(f"    First 5: {list(missing_rag)[:5]}")
```

**Expected output:**
- 117 timelines loaded
- 103 RAG results loaded (14 cases have timeline but no RAG data is acceptable)
- All events have required fields (uid4, mention, time)

---

#### 1.2 Output Data Verification

**What to check:**
- All input cases were processed
- Output format is correct
- No missing fields

**How to verify:**

```python
import json

with open('results/gap_detection_results.json', 'r') as f:
    results = json.load(f)

# Check all cases processed
print(f"Input timelines: {len(data['timelines'])}")
print(f"Output results: {len(results)}")

# Check required fields in output
sample_case = list(results.keys())[0]
sample_result = results[sample_case]

print(f"\nSample result structure:")
print(f"  case_id: {sample_result.get('case_id')}")
print(f"  total_mentions: {sample_result.get('total_mentions')}")
print(f"  gap_analysis: {len(sample_result.get('gap_analysis', []))} analyses")
print(f"  gap_summary: {list(sample_result.get('gap_summary', {}).keys())}")

# Check each gap analysis has required fields
if sample_result.get('gap_analysis'):
    sample_analysis = sample_result['gap_analysis'][0]
    required_fields = ['uid4', 'mention', 'gap_type', 'forecasting_relevance',
                      'coverage', 'temporal', 'details']
    missing_fields = [f for f in required_fields if f not in sample_analysis]

    if missing_fields:
        print(f"\n⚠️  Missing fields: {missing_fields}")
    else:
        print(f"\n✓ All required fields present")
```

---

### Phase 2: Algorithm Correctness Checks

#### 2.1 Gap Type Classification Logic

**What to check:**
- Classification rules are implemented correctly
- Threshold values are appropriate
- No edge cases misclassified

**Manual verification approach:**

```python
# Independent implementation of classification logic
def classify_gap_independent(analysis):
    """Your own implementation to verify against."""
    has_counterpart = analysis['coverage']['has_counterpart']
    best_score = analysis['coverage']['best_score']
    has_detail_gap = analysis['details'].get('has_detail_gap', False)
    alignment = analysis['temporal'].get('alignment', 'unknown')

    # Expected rules:
    # 1. No counterpart -> complete_absence
    if not has_counterpart:
        return 'complete_absence'

    # 2. Has detail gap -> detail_gap
    if has_detail_gap:
        return 'detail_gap'

    # 3. Temporal mismatch -> temporal_mismatch
    if alignment == 'misaligned':
        return 'temporal_mismatch'

    # 4. Low score (< 0.6) -> semantic_distance
    if best_score < 0.6:
        return 'semantic_distance'

    # 5. Otherwise -> well_captured
    return 'well_captured'

# Verify against actual results
disagreements = []

for case_id, result in list(results.items())[:10]:
    if 'gap_analysis' not in result:
        continue

    for analysis in result['gap_analysis'][:10]:
        expected = classify_gap_independent(analysis)
        actual = analysis['gap_type']

        if expected != actual:
            disagreements.append({
                'case_id': case_id,
                'mention': analysis['mention'],
                'expected': expected,
                'actual': actual
            })

if disagreements:
    print(f"\n⚠️  Found {len(disagreements)} classification disagreements:")
    for d in disagreements[:5]:
        print(f"  {d['mention']}: expected {d['expected']}, got {d['actual']}")
else:
    print("\n✓ All classifications match expected logic")
```

---

#### 2.2 Coverage Score Threshold Verification

**What to check:**
- RAG score threshold (0.3) is appropriate
- Coverage scores are extracted correctly from RAG data

**How to verify:**

```python
# Spot-check coverage scores against original RAG data
import pandas as pd

case_id = '10004648-DS-13'
uid4 = '02bb'  # "No Known Allergies"

# Get from gap detection results
result = results[case_id]
analysis = next(a for a in result['gap_analysis'] if a['uid4'] == uid4)
coverage_score = analysis['coverage']['best_score']

print(f"Gap detection result:")
print(f"  Mention: {analysis['mention']}")
print(f"  Coverage score: {coverage_score}")

# Verify against RAG data
rag_df = data['rag_results'][case_id]

# Find all anchors with this uid4
matches = []
for _, row in rag_df.iterrows():
    anchor_uid4 = extract_uid4_from_anchor(row['anchor'])
    if anchor_uid4 == uid4:
        matches.append({
            'bestscore': row['bestscore'],
            'event': row['line_level_event']
        })

if matches:
    best_score_from_rag = max(m['bestscore'] for m in matches)
    print(f"\nFrom RAG data:")
    print(f"  Found {len(matches)} matches")
    print(f"  Best score: {best_score_from_rag}")

    if abs(coverage_score - best_score_from_rag) < 0.001:
        print(f"\n✓ Coverage score matches RAG data")
    else:
        print(f"\n⚠️  Score mismatch: {coverage_score} vs {best_score_from_rag}")
```

---

#### 2.3 Forecasting Relevance Scoring

**What to check:**
- Relevance classification is based on mention content
- High-relevance mentions contain appropriate keywords

**How to verify:**

```python
# Check high-relevance mentions
high_rel_mentions = []

for case_id, result in results.items():
    if 'gap_analysis' not in result:
        continue

    for analysis in result['gap_analysis']:
        if analysis['forecasting_relevance'] == 'high':
            high_rel_mentions.append(analysis['mention'])

print(f"Total high-relevance mentions: {len(high_rel_mentions)}")

# Check for expected keywords
high_rel_keywords = ['symptom', 'pain', 'fever', 'weakness', 'nausea',
                     'vomiting', 'diagnosis', 'surgery', 'procedure',
                     'complication', 'acute', 'severe']

contains_keywords = sum(
    1 for m in high_rel_mentions
    if any(kw in m.lower() for kw in high_rel_keywords)
)

print(f"Contains expected keywords: {contains_keywords}/{len(high_rel_mentions)} "
      f"({contains_keywords/len(high_rel_mentions)*100:.1f}%)")

# Show samples without expected keywords
no_keywords = [m for m in high_rel_mentions
               if not any(kw in m.lower() for kw in high_rel_keywords)]

if no_keywords:
    print(f"\n⚠️  {len(no_keywords)} high-relevance mentions without expected keywords:")
    for m in no_keywords[:10]:
        print(f"  - {m}")
```

---

### Phase 3: Statistical Validation

#### 3.1 Distribution Sanity Checks

**What to check:**
- Gap type distribution is reasonable
- No unexpected clustering
- Percentages make sense

**How to verify:**

```python
from collections import Counter

# Collect all gap types
gap_types = []
for result in results.values():
    if 'gap_analysis' not in result:
        continue
    for analysis in result['gap_analysis']:
        gap_types.append(analysis['gap_type'])

gap_counts = Counter(gap_types)
total = len(gap_types)

print(f"Total mentions: {total}\n")
print(f"Gap type distribution:")
for gap_type, count in sorted(gap_counts.items()):
    pct = count / total * 100
    print(f"  {gap_type:25s}: {count:6d} ({pct:5.1f}%)")

# Sanity checks
issues = []

# 1. Complete absence should be significant but not 100%
complete_absence_pct = gap_counts.get('complete_absence', 0) / total * 100
if complete_absence_pct < 10:
    issues.append(f"Complete absence unusually low: {complete_absence_pct:.1f}%")
elif complete_absence_pct > 80:
    issues.append(f"Complete absence unusually high: {complete_absence_pct:.1f}%")

# 2. Temporal mismatch should exist
if gap_counts.get('temporal_mismatch', 0) == 0:
    issues.append("No temporal mismatches found - investigate why")

# 3. Detail gaps should be present
if gap_counts.get('detail_gap', 0) == 0:
    issues.append("No detail gaps found - may indicate implementation issue")

if issues:
    print(f"\n⚠️  Potential issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print(f"\n✓ Distributions appear reasonable")
```

**Expected ranges:**
- Complete absence: 20-80%
- Well captured: 10-50%
- Semantic distance: 5-30%
- Detail gaps: 1-10%
- Temporal mismatch: 0-10% (may be 0 if timing is aligned)

---

#### 3.2 Cross-Validation with Known Cases

**What to check:**
- Known problematic cases show expected gaps
- Known well-captured cases show few gaps

**How to verify:**

```python
# Define known test cases (you should identify these from domain knowledge)
# Example: Surgery case should show detail gaps
surgery_case = None

for case_id, result in results.items():
    if 'gap_analysis' not in result:
        continue

    for analysis in result['gap_analysis']:
        if 'surgery' in analysis['mention'].lower():
            surgery_case = case_id
            break
    if surgery_case:
        break

if surgery_case:
    print(f"Found surgery case: {surgery_case}")
    result = results[surgery_case]

    surgery_analyses = [a for a in result['gap_analysis']
                       if 'surgery' in a['mention'].lower()]

    print(f"\nSurgery mentions ({len(surgery_analyses)}):")
    for a in surgery_analyses[:5]:
        print(f"  {a['mention']}")
        print(f"    Gap type: {a['gap_type']}")
        print(f"    Relevance: {a['forecasting_relevance']}")
        if a['gap_type'] == 'detail_gap':
            print(f"    Missing: {a['details'].get('missing_attributes', [])}")
```

---

### Phase 4: Spot-Check Methodology

#### 4.1 Random Sampling Verification

**Approach:** Select 20-30 random mentions and manually verify classification

```python
import random

# Select random samples
all_analyses = []
for case_id, result in results.items():
    if 'gap_analysis' not in result:
        continue
    for analysis in result['gap_analysis']:
        all_analyses.append({
            'case_id': case_id,
            'analysis': analysis
        })

samples = random.sample(all_analyses, min(30, len(all_analyses)))

print("Manual verification checklist:")
print("="*80)

for i, sample in enumerate(samples, 1):
    case_id = sample['case_id']
    a = sample['analysis']

    print(f"\n{i}. {a['mention']}")
    print(f"   Case: {case_id}")
    print(f"   Gap type: {a['gap_type']}")
    print(f"   Relevance: {a['forecasting_relevance']}")
    print(f"   Coverage score: {a['coverage']['best_score']:.3f}")

    print(f"\n   Verification steps:")
    print(f"   [ ] Check RAG file: bundle/miv/uidtts/comparison_results/{case_id}_joined.csv")
    print(f"   [ ] Search for uid4 '{a['uid4']}' in anchor field")
    print(f"   [ ] Verify best_score matches: {a['coverage']['best_score']:.3f}")
    print(f"   [ ] Confirm gap classification is correct")
    print(f"   [ ] Check if forecasting relevance is appropriate")
```

**Manual verification steps for each sample:**
1. Open RAG results file for case
2. Search for uid4 in anchor field
3. Verify best_score matches
4. Assess if gap classification makes sense given the score
5. Check if mention content justifies forecasting relevance

---

#### 4.2 Edge Case Verification

**Identify and verify edge cases:**

```python
# Edge case 1: High coverage score but classified as gap
edge_cases = []

for case_id, result in results.items():
    if 'gap_analysis' not in result:
        continue

    for analysis in result['gap_analysis']:
        score = analysis['coverage']['best_score']
        gap_type = analysis['gap_type']

        # Edge case: high score but not well_captured
        if score > 0.7 and gap_type != 'well_captured':
            edge_cases.append({
                'type': 'high_score_not_well_captured',
                'case_id': case_id,
                'mention': analysis['mention'],
                'score': score,
                'gap_type': gap_type,
                'reason': 'Has detail gap or temporal mismatch'
            })

        # Edge case: low score but classified as well_captured
        if score < 0.3 and gap_type == 'well_captured':
            edge_cases.append({
                'type': 'low_score_well_captured',
                'case_id': case_id,
                'mention': analysis['mention'],
                'score': score,
                'gap_type': gap_type,
                'reason': 'Should be complete_absence'
            })

if edge_cases:
    print(f"Found {len(edge_cases)} edge cases:\n")
    for ec in edge_cases[:10]:
        print(f"{ec['type']}: {ec['mention']}")
        print(f"  Score: {ec['score']:.3f}, Gap type: {ec['gap_type']}")
        print(f"  Reason: {ec['reason']}\n")
else:
    print("✓ No edge cases found")
```

---

### Phase 5: Parameter Sensitivity Analysis

#### 5.1 Threshold Impact Analysis

**What to check:**
- How sensitive are results to threshold choices?
- Is the chosen threshold (0.3) appropriate?

**How to analyze:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Test different thresholds
thresholds = np.arange(0.1, 0.7, 0.05)
complete_absence_counts = []

for threshold in thresholds:
    count = 0
    for result in results.values():
        if 'gap_analysis' not in result:
            continue
        for analysis in result['gap_analysis']:
            if analysis['coverage']['best_score'] < threshold:
                count += 1
    complete_absence_counts.append(count)

# Plot sensitivity
plt.figure(figsize=(10, 6))
plt.plot(thresholds, complete_absence_counts, marker='o')
plt.axvline(x=0.3, color='red', linestyle='--', label='Current threshold (0.3)')
plt.xlabel('Coverage Score Threshold')
plt.ylabel('Complete Absence Count')
plt.title('Sensitivity of Gap Detection to Threshold Choice')
plt.legend()
plt.grid(True)
plt.savefig('threshold_sensitivity.png')
print("Plot saved to: threshold_sensitivity.png")

# Calculate percentage change
pct_changes = []
for i in range(1, len(complete_absence_counts)):
    pct_change = abs(complete_absence_counts[i] - complete_absence_counts[i-1]) / complete_absence_counts[i-1] * 100
    pct_changes.append(pct_change)

avg_change = np.mean(pct_changes)
print(f"\nAverage sensitivity: {avg_change:.1f}% change per 0.05 threshold increment")

if avg_change > 20:
    print("⚠️  High sensitivity - threshold choice significantly impacts results")
else:
    print("✓ Moderate sensitivity - results are reasonably stable")
```

---

### Phase 6: Documentation Review

#### 6.1 Code Review Checklist

**Review the implementation files:**

```bash
# Read implementation
less gap_detection.py
less gap_analysis.py
less data_loading.py

# Check for:
# [ ] Clear function documentation
# [ ] Parameter explanations
# [ ] Edge case handling
# [ ] Error handling
# [ ] Reasonable variable names
# [ ] No hardcoded magic numbers (use constants)
# [ ] Threshold values are documented
```

#### 6.2 Threshold Rationale

**Document why thresholds were chosen:**

```python
# Review threshold choices in gap_detection.py
print("Threshold values and rationale:\n")
print("SEMANTIC_SIMILARITY_THRESHOLD = 0.7")
print("  Rationale: Standard cosine similarity threshold for 'good' match")
print()
print("RAG_SCORE_LOW_THRESHOLD = 0.3")
print("  Rationale: Below this score, RAG retrieval was unsuccessful")
print("  Validation: [Describe how you validated this choice]")
print()
print("RAG_SCORE_MEDIUM_THRESHOLD = 0.6")
print("  Rationale: Partial match threshold")
print("  Validation: [Describe how you validated this choice]")
print()
print("TEMPORAL_ALIGNMENT_THRESHOLD = 6.0 hours")
print("  Rationale: Events within 6h are considered temporally aligned")
print("  Validation: [Check if this matches clinical documentation patterns]")
```

---

## Expected Findings

### What Should Raise Concerns

1. **Classification Logic Errors**
   - Misclassified gaps (complete_absence when counterpart exists)
   - Coverage scores don't match RAG data
   - Inconsistent application of rules

2. **Data Quality Issues**
   - Missing cases in results
   - Corrupted data during loading
   - Incorrect field mapping

3. **Statistical Anomalies**
   - 100% of cases in one gap type
   - No temporal mismatches at all
   - Unrealistic distributions

4. **Implementation Issues**
   - Hardcoded values without documentation
   - No error handling
   - Missing edge case handling

### What Should NOT Raise Concerns

1. **Some disagreements in forecasting relevance**
   - Keyword-based heuristic, some subjectivity is expected
   - 60-70% keyword match is acceptable

2. **No temporal mismatches**
   - May indicate timing is generally aligned in this dataset
   - Investigate but not necessarily wrong

3. **Variation across cases**
   - Different patients have different gap patterns
   - This is expected

---

## Validation Report Template

After completing verification, fill out this template:

```markdown
# Validation Report

**Date:** [Date]
**Reviewer:** [Your name]
**Batch:** batch_output_0002
**Results file:** gap_detection_results.json

## Summary

- Total cases processed: [X]
- Total mentions analyzed: [X]
- Classification accuracy: [X]%
- Data integrity: [PASS/FAIL]

## Checks Performed

1. Input Data Integrity: [PASS/FAIL]
   - Notes: [Any issues found]

2. Gap Type Classification: [PASS/FAIL]
   - Sample size: [X]
   - Correct: [X]
   - Incorrect: [X]

3. Coverage Score Verification: [PASS/FAIL]
   - Spot-checks: [X]
   - All verified: [YES/NO]

4. Forecasting Relevance: [PASS/FAIL]
   - Keyword match rate: [X]%
   - Acceptable: [YES/NO]

5. Statistical Distributions: [PASS/FAIL]
   - Issues found: [List]

6. Parameter Sensitivity: [PASS/FAIL]
   - Sensitivity level: [LOW/MEDIUM/HIGH]

## Issues Found

[List any issues discovered during verification]

## Recommendations

[Suggestions for improvement or concerns to address]

## Approval

[APPROVED / NEEDS REVISION]
```

---

## Quick Commands

```bash
# Run all automated checks
python verification_checklist.py symlinks/batch_output_0002 results/gap_detection_results.json

# View validation report
cat results/validation_report.json | jq .

# Count gap types
jq '[.[] | .gap_analysis[] | .gap_type] | group_by(.) | map({type: .[0], count: length})' results/gap_detection_results.json

# Extract high-relevance gaps
jq '[.[] | .gap_analysis[] | select(.forecasting_relevance == "high") | .mention] | .[:20]' results/gap_detection_results.json

# Check coverage score distribution
jq '[.[] | .gap_analysis[] | .coverage.best_score] | sort | .[::100]' results/gap_detection_results.json
```

---

## Contact

For questions or issues with the verification process, contact the implementation team or create an issue in the project repository.