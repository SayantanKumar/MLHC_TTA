# Gap Detection Test Suite Design

**Date:** 2026-04-02
**Author:** Claude Code
**Status:** Ready for implementation

## Overview

Create a comprehensive test suite for the gap detection implementation to validate correctness against design specifications and document edge case behaviors.

## Goals

1. **Validate correctness** - Ensure implementation matches design spec (admission time anchoring, threshold logic, gap classification rules)
2. **Document behavior** - Use tests as executable documentation of edge cases and design decisions
3. **Protect against regressions** - Catch bugs like the 3.7-day temporal offset issue

## Test Framework

**Framework:** pytest
**Rationale:**
- Parametrization support for threshold boundary testing
- Readable assertions for documentation value
- Powerful fixture system for test data management
- Built-in test discovery and reporting

## Test Organization

### Directory Structure

```
gap_detection/
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures and environment checks
│   ├── test_complete_absence.py       # Gap type tests
│   ├── test_temporal_mismatch.py
│   ├── test_semantic_distance.py
│   ├── test_detail_gap.py
│   ├── test_well_captured.py
│   ├── test_thresholds.py             # Boundary condition tests
│   ├── test_timezone_handling.py      # Timezone edge cases
│   └── fixtures/
│       └── synthetic_cases.py         # Mock data generators
├── i2m4_analysis/                     # Real data (accessed via path resolution)
└── ...
```

### Test Modules

**Total: 7 modules, ~28-30 tests**

#### Module 1: test_complete_absence.py (~4 tests)
- Test: No RAG matches → complete_absence
- Test: RAG score below LOW_THRESHOLD (0.3) → complete_absence
- Test: Multiple low-score matches still → complete_absence
- Test: Real example - symptom mention with no tabular record

#### Module 2: test_temporal_mismatch.py (~5 tests)
- Test: Aligned timing (≤6h) → not temporal_mismatch
- Test: Coarse timing (6-12h) → not temporal_mismatch
- Test: Misaligned timing (>12h) with good scores → temporal_mismatch
- Test: Exactly 12.0 hours → not temporal_mismatch (boundary)
- Test: Real example - lab event with delayed tabular record

#### Module 3: test_semantic_distance.py (~4 tests)
- Test: Good timing + RAG ≥0.6 → not semantic_distance
- Test: Good timing + RAG <0.6 → semantic_distance
- Test: Exactly 0.6 score → not semantic_distance (boundary)
- Test: Real example - textual term with no direct tabular equivalent

#### Module 4: test_detail_gap.py (~5 tests)
- Test: Good timing + good score + missing severity → detail_gap
- Test: Good timing + good score + missing location → detail_gap
- Test: Good timing + good score + no missing details → well_captured
- Test: Multiple missing attributes → detail_gap
- Test: Real example - surgery without procedure details

#### Module 5: test_well_captured.py (~4 tests)
- Test: All conditions met → well_captured
- Test: Edge case - aligned timing + score exactly 0.6 + no details missing
- Test: Edge case - coarse timing (6h) + good score → well_captured
- Test: Real example - lab result with matching tabular event

#### Module 6: test_thresholds.py (~6 tests)
- Test: Temporal boundary at exactly 6.0 hours
- Test: Temporal boundary at exactly 12.0 hours
- Test: RAG score boundary at exactly 0.3
- Test: RAG score boundary at exactly 0.6
- Test: Multiple boundary conditions simultaneously
- Test: Edge case - time=0 (admission time) alignment

#### Module 7: test_timezone_handling.py (~4-5 tests)
- Test: Admission time with timezone info gets stripped correctly
- Test: Admission time without timezone passes through
- Test: Tabular timestamps with timezone handling
- Test: Mixed timezone scenarios (admission has tz, tabular doesn't)
- Test: DST boundary handling (if applicable)

## Test Data Strategy

### Synthetic Fixtures

**Location:** `tests/fixtures/synthetic_cases.py`

**Helper Functions:**
```python
def create_mock_event(uid4, mention, time, bounds=None):
    """Create a single textual event with minimal required fields"""

def create_mock_rag_result(uid4, score, timestamp, match_text, attributes=None):
    """Create a RAG match result with specified score and timing"""

def create_mock_case(case_id, events, rag_results, admission_time=None):
    """Assemble a complete test case from events and RAG results"""
```

**Advantages:**
- Self-contained and portable
- Easy to understand test scenarios
- Control over edge cases
- No privacy concerns

### Real Examples

**Location:** `i2m4_analysis/glm5_output/gap_results/` (existing data)

**Access Method:** Path resolution with environment checks
```python
@pytest.fixture
def real_case_complete_absence():
    """Load real case - requires i2m4 data access"""
    case_path = Path(__file__).parent.parent / "i2m4_analysis/glm5_output/gap_results/case_X.json"
    if not case_path.exists():
        pytest.skip("Real i2m4 data not available")
    with open(case_path) as f:
        return json.load(f)
```

**Advantages:**
- Tests realistic scenarios
- Catches real-world edge cases
- Gracefully skips if data unavailable
- Makes dependency on real data explicit

## Key Design Decisions to Test

### 1. Temporal Alignment with Admission Time Anchoring
- **Critical:** Time=0 is admission time (not RAG cutoff_time)
- **Bug to prevent:** 3.7-day offset from using wrong anchor
- **Tests:** `test_temporal_mismatch.py`, `test_timezone_handling.py`

### 2. Gap Classification Priority
- **Decision tree order:** complete_absence → detail_gap → temporal_mismatch → semantic_distance → well_captured
- **Tests:** All gap type modules document priority

### 3. Threshold Boundaries
- **Temporal:** 6h (aligned), 12h (coarse/misaligned boundary)
- **RAG scores:** 0.3 (low), 0.6 (medium/high boundary)
- **Semantic similarity:** 0.7 (coverage threshold)
- **Tests:** `test_thresholds.py`

### 4. Edge Cases
- Missing timestamps
- Unknown times
- Multiple matches with different scores
- Exactly-at-boundary values
- **Tests:** Distributed across modules

## Test Execution

```bash
# Run all tests
pytest gap_detection/tests/

# Run specific module
pytest gap_detection/tests/test_temporal_mismatch.py

# Run with verbose output
pytest -v gap_detection/tests/

# Run only threshold tests
pytest -v gap_detection/tests/test_thresholds.py

# Run with coverage report
pytest --cov=gap_detection gap_detection/tests/
```

## Implementation Plan

1. Create test directory structure
2. Implement synthetic fixture generators
3. Implement shared fixtures in conftest.py
4. Write tests for each gap type module
5. Write threshold boundary tests
6. Write timezone handling tests
7. Add real case fixtures with environment checks
8. Verify all tests pass
9. Add to CI/CD pipeline (if applicable)

## Success Criteria

- All ~28-30 tests pass
- Tests validate core design decisions
- Edge cases documented through test names and docstrings
- Real data tests gracefully skip when data unavailable
- Regression protection for critical bugs (admission time anchoring, timezone handling)

## Future Enhancements

- Integration tests for full pipeline
- Performance benchmarks
- Coverage targets (>90% for core logic)
- Mutation testing to ensure test quality