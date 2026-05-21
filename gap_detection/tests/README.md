# Gap Detection Test Suite

## Overview

Comprehensive test suite for the gap detection implementation. Tests validate correctness against design specifications and document edge case behaviors.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific module
pytest tests/test_temporal_mismatch.py

# Run with verbose output
pytest -v tests/

# Run with coverage
pytest --cov=gap_detection tests/

# Run specific test
pytest tests/test_thresholds.py::TestThresholds::test_temporal_12h_boundary -v
```

## Test Organization

- **test_complete_absence.py** - Tests for no tabular counterpart (4 tests)
- **test_temporal_mismatch.py** - Tests for timing differences >12h (5 tests)
- **test_semantic_distance.py** - Tests for low similarity matches (4 tests)
- **test_detail_gap.py** - Tests for missing attributes (5 tests)
- **test_well_captured.py** - Tests for successful matches (4 tests)
- **test_thresholds.py** - Boundary condition tests (6 tests)
- **test_timezone_handling.py** - Timezone edge case tests (5 tests)

**Total:** 33 tests

## Test Data

### Synthetic Fixtures

Mock data generators in `fixtures/synthetic_cases.py`:
- `create_mock_event()` - Create textual events
- `create_mock_rag_result()` - Create RAG matches
- `create_mock_case()` - Assemble complete test cases

### Real Cases

Integration tests use real i2m4 data. Tests gracefully skip if data unavailable.

## Key Design Decisions Tested

1. **Admission time anchoring** - Time=0 is admission time (not RAG cutoff_time)
2. **Gap classification priority** - Decision tree order validation
3. **Threshold boundaries** - Exactly at 6h, 12h, 0.3, 0.6
4. **Timezone handling** - DST, mixed timezones, tz stripping

## Requirements

- pytest
- pytest-cov (optional, for coverage)
- pytz (optional, for timezone tests)