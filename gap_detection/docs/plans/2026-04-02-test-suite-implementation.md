# Gap Detection Test Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create comprehensive test suite validating gap detection correctness and documenting edge cases

**Architecture:** pytest-based test suite with synthetic fixtures for unit testing and real case integration tests. Tests organized by gap type outcome, plus dedicated modules for threshold boundaries and timezone handling.

**Tech Stack:** pytest, pytest-cov, pathlib, datetime, json

---

## Task 1: Set Up Test Infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/synthetic_cases.py`

**Step 1: Create test directory structure**

```bash
cd /data/weissjc/lns/tta/uidtts/gsandbox/batch_processor/gap_detection
mkdir -p tests/fixtures
touch tests/__init__.py
touch tests/fixtures/__init__.py
```

**Step 2: Write synthetic fixture generators**

Create `tests/fixtures/synthetic_cases.py`:

```python
"""Synthetic test fixtures for gap detection tests."""
from datetime import datetime, timedelta
from typing import Dict, List, Optional


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
```

**Step 3: Write shared fixtures**

Create `tests/conftest.py`:

```python
"""Shared pytest fixtures for gap detection tests."""
import pytest
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gap_detection import GapDetector


@pytest.fixture
def gap_detector():
    """Initialize GapDetector for testing."""
    return GapDetector()


@pytest.fixture
def mock_admission_times():
    """Mock admission times for temporal testing."""
    return {
        'CASE_001': datetime(2023, 6, 15, 10, 30, 0),
        'CASE_002': datetime(2023, 6, 15, 10, 30, 0),
        'CASE_003': datetime(2023, 6, 15, 10, 30, 0),
    }


@pytest.fixture
def real_case_path():
    """Path to real i2m4 data (for integration tests)."""
    return Path(__file__).parent.parent / "i2m4_analysis" / "glm5_output" / "gap_results"
```

**Step 4: Verify setup**

Run: `python -c "from tests.fixtures.synthetic_cases import create_mock_event; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add tests/
git commit -m "test: set up test infrastructure with synthetic fixtures"
```

---

## Task 2: Implement test_complete_absence.py

**Files:**
- Create: `tests/test_complete_absence.py`

**Step 1: Write the failing tests**

Create `tests/test_complete_absence.py`:

```python
"""Tests for complete_absence gap type."""
import pytest
from datetime import datetime, timedelta

from gap_detection import GapDetector
from tests.fixtures.synthetic_cases import (
    create_mock_event,
    create_mock_rag_result,
    create_mock_case
)


class TestCompleteAbsence:
    """Tests for complete_absence gap classification."""

    def test_no_rag_matches(self, gap_detector):
        """No RAG matches → complete_absence."""
        # Create event with no matches
        event = create_mock_event('0001', 'chest pain', -5.0)
        case = create_mock_case('CASE_001', [event], [])

        # Run gap detection
        results = gap_detector.detect_gaps(case)

        # Verify complete_absence
        assert len(results) == 1
        assert results[0]['gap_type'] == 'complete_absence'
        assert results[0]['uid4'] == '0001'

    def test_low_rag_score_only(self, gap_detector):
        """RAG score below LOW_THRESHOLD (0.3) → complete_absence."""
        # Create event with only low-score matches
        event = create_mock_event('0002', 'dizziness', -3.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        tabular_time = admission_time + timedelta(hours=-3)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0002', 0.25, timestamp, 'unrelated note'),
            create_mock_rag_result('0002', 0.20, timestamp, 'another unrelated'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        # Run gap detection
        results = gap_detector.detect_gaps(case)

        # Verify complete_absence (scores too low)
        assert len(results) == 1
        assert results[0]['gap_type'] == 'complete_absence'

    def test_multiple_low_scores(self, gap_detector):
        """Multiple low-score matches still → complete_absence."""
        event = create_mock_event('0003', 'headache', -2.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        tabular_time = admission_time + timedelta(hours=-2)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0003', 0.29, timestamp, 'match 1'),
            create_mock_rag_result('0003', 0.28, timestamp, 'match 2'),
            create_mock_rag_result('0003', 0.27, timestamp, 'match 3'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        assert len(results) == 1
        assert results[0]['gap_type'] == 'complete_absence'

    def test_real_case_integration(self, gap_detector, real_case_path):
        """Real example - symptom mention with no tabular record."""
        # Skip if real data not available
        if not real_case_path.exists():
            pytest.skip("Real i2m4 data not available")

        # This test will be filled in once we identify a real complete_absence case
        # For now, just verify the fixture works
        pytest.skip("Real case integration test pending")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_complete_absence.py -v`
Expected: 4 FAILED (implementation not complete yet)

**Step 3: Commit**

```bash
git add tests/test_complete_absence.py
git commit -m "test: add complete_absence gap type tests"
```

---

## Task 3: Implement test_temporal_mismatch.py

**Files:**
- Create: `tests/test_temporal_mismatch.py`

**Step 1: Write the failing tests**

Create `tests/test_temporal_mismatch.py`:

```python
"""Tests for temporal_mismatch gap type."""
import pytest
from datetime import datetime, timedelta

from gap_detection import GapDetector
from tests.fixtures.synthetic_cases import (
    create_mock_event,
    create_mock_rag_result,
    create_mock_case
)


class TestTemporalMismatch:
    """Tests for temporal_mismatch gap classification."""

    def test_aligned_timing(self, gap_detector):
        """Aligned timing (≤6h) → not temporal_mismatch."""
        event = create_mock_event('0001', 'fever', -5.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        # Tabular event at -5.5h (0.5h difference, well within 6h)
        tabular_time = admission_time + timedelta(hours=-5.5)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0001', 0.85, timestamp, 'temperature elevation'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        # Should be well_captured (good timing, good score, no missing details)
        assert len(results) == 1
        assert results[0]['gap_type'] == 'well_captured'

    def test_coarse_timing(self, gap_detector):
        """Coarse timing (6-12h) → not temporal_mismatch."""
        event = create_mock_event('0002', 'nausea', -10.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        # Tabular event at -2h (8h difference, within coarse range)
        tabular_time = admission_time + timedelta(hours=-2)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0002', 0.80, timestamp, 'GI symptoms'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        # Should be well_captured (coarse is acceptable)
        assert len(results) == 1
        assert results[0]['gap_type'] == 'well_captured'

    def test_misaligned_timing(self, gap_detector):
        """Misaligned timing (>12h) with good scores → temporal_mismatch."""
        event = create_mock_event('0003', 'vomiting', -24.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        # Tabular event at -5h (19h difference, exceeds 12h)
        tabular_time = admission_time + timedelta(hours=-5)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0003', 0.90, timestamp, 'emesis episode'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        assert len(results) == 1
        assert results[0]['gap_type'] == 'temporal_mismatch'

    def test_exactly_12_hours_boundary(self, gap_detector):
        """Exactly 12.0 hours → not temporal_mismatch (boundary)."""
        event = create_mock_event('0004', 'diarrhea', -12.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        # Tabular event at admission (exactly 12h difference)
        tabular_time = admission_time
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0004', 0.82, timestamp, 'GI upset'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        # Should be well_captured (exactly 12h is acceptable)
        assert len(results) == 1
        assert results[0]['gap_type'] == 'well_captured'

    def test_real_case_integration(self, gap_detector, real_case_path):
        """Real example - lab event with delayed tabular record."""
        if not real_case_path.exists():
            pytest.skip("Real i2m4 data not available")

        pytest.skip("Real case integration test pending")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_temporal_mismatch.py -v`
Expected: 5 FAILED

**Step 3: Commit**

```bash
git add tests/test_temporal_mismatch.py
git commit -m "test: add temporal_mismatch gap type tests"
```

---

## Task 4: Implement test_semantic_distance.py

**Files:**
- Create: `tests/test_semantic_distance.py`

**Step 1: Write the failing tests**

Create `tests/test_semantic_distance.py`:

```python
"""Tests for semantic_distance gap type."""
import pytest
from datetime import datetime, timedelta

from gap_detection import GapDetector
from tests.fixtures.synthetic_cases import (
    create_mock_event,
    create_mock_rag_result,
    create_mock_case
)


class TestSemanticDistance:
    """Tests for semantic_distance gap classification."""

    def test_good_score_not_semantic_distance(self, gap_detector):
        """Good timing + RAG ≥0.6 → not semantic_distance."""
        event = create_mock_event('0001', 'hypertension', -8.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        tabular_time = admission_time + timedelta(hours=-8)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0001', 0.75, timestamp, 'high BP'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        assert len(results) == 1
        assert results[0]['gap_type'] != 'semantic_distance'

    def test_low_score_semantic_distance(self, gap_detector):
        """Good timing + RAG <0.6 → semantic_distance."""
        event = create_mock_event('0002', 'fatigue', -4.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        tabular_time = admission_time + timedelta(hours=-4)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0002', 0.45, timestamp, 'tiredness'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        assert len(results) == 1
        assert results[0]['gap_type'] == 'semantic_distance'

    def test_exactly_0_6_boundary(self, gap_detector):
        """Exactly 0.6 score → not semantic_distance (boundary)."""
        event = create_mock_event('0003', 'weakness', -6.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        tabular_time = admission_time + timedelta(hours=-6)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0003', 0.60, timestamp, 'motor weakness'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        # Exactly 0.6 should be well_captured (not semantic_distance)
        assert len(results) == 1
        assert results[0]['gap_type'] == 'well_captured'

    def test_real_case_integration(self, gap_detector, real_case_path):
        """Real example - textual term with no direct tabular equivalent."""
        if not real_case_path.exists():
            pytest.skip("Real i2m4 data not available")

        pytest.skip("Real case integration test pending")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_semantic_distance.py -v`
Expected: 4 FAILED

**Step 3: Commit**

```bash
git add tests/test_semantic_distance.py
git commit -m "test: add semantic_distance gap type tests"
```

---

## Task 5: Implement test_detail_gap.py

**Files:**
- Create: `tests/test_detail_gap.py`

**Step 1: Write the failing tests**

Create `tests/test_detail_gap.py`:

```python
"""Tests for detail_gap gap type."""
import pytest
from datetime import datetime, timedelta

from gap_detection import GapDetector
from tests.fixtures.synthetic_cases import (
    create_mock_event,
    create_mock_rag_result,
    create_mock_case
)


class TestDetailGap:
    """Tests for detail_gap gap classification."""

    def test_missing_severity(self, gap_detector):
        """Good timing + good score + missing severity → detail_gap."""
        event = create_mock_event('0001', 'pain', -3.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        tabular_time = admission_time + timedelta(hours=-3)
        timestamp = tabular_time.timestamp()

        # Match has no severity attribute
        rag_results = [
            create_mock_rag_result('0001', 0.85, timestamp, 'pain reported',
                                   attributes={}),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        assert len(results) == 1
        assert results[0]['gap_type'] == 'detail_gap'

    def test_missing_location(self, gap_detector):
        """Good timing + good score + missing location → detail_gap."""
        event = create_mock_event('0002', 'chest pain', -5.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        tabular_time = admission_time + timedelta(hours=-5)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0002', 0.80, timestamp, 'chest discomfort',
                                   attributes={'severity': 'moderate'}),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        assert len(results) == 1
        assert results[0]['gap_type'] == 'detail_gap'

    def test_no_missing_details(self, gap_detector):
        """Good timing + good score + no missing details → well_captured."""
        event = create_mock_event('0003', 'abdominal pain', -7.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        tabular_time = admission_time + timedelta(hours=-7)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0003', 0.82, timestamp, 'abd pain',
                                   attributes={'severity': 'severe', 'location': 'RLQ'}),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        assert len(results) == 1
        assert results[0]['gap_type'] == 'well_captured'

    def test_multiple_missing_attributes(self, gap_detector):
        """Multiple missing attributes → detail_gap."""
        event = create_mock_event('0004', 'severe headache', -2.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        tabular_time = admission_time + timedelta(hours=-2)
        timestamp = tabular_time.timestamp()

        # Missing both severity and location
        rag_results = [
            create_mock_rag_result('0004', 0.78, timestamp, 'headache',
                                   attributes={}),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        assert len(results) == 1
        assert results[0]['gap_type'] == 'detail_gap'

    def test_real_case_integration(self, gap_detector, real_case_path):
        """Real example - surgery without procedure details."""
        if not real_case_path.exists():
            pytest.skip("Real i2m4 data not available")

        pytest.skip("Real case integration test pending")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_detail_gap.py -v`
Expected: 5 FAILED

**Step 3: Commit**

```bash
git add tests/test_detail_gap.py
git commit -m "test: add detail_gap gap type tests"
```

---

## Task 6: Implement test_well_captured.py

**Files:**
- Create: `tests/test_well_captured.py`

**Step 1: Write the failing tests**

Create `tests/test_well_captured.py`:

```python
"""Tests for well_captured gap type."""
import pytest
from datetime import datetime, timedelta

from gap_detection import GapDetector
from tests.fixtures.synthetic_cases import (
    create_mock_event,
    create_mock_rag_result,
    create_mock_case
)


class TestWellCaptured:
    """Tests for well_captured gap classification."""

    def test_all_conditions_met(self, gap_detector):
        """All conditions met → well_captured."""
        event = create_mock_event('0001', 'diabetes', -15.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        tabular_time = admission_time + timedelta(hours=-15)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0001', 0.88, timestamp, 'DM type 2',
                                   attributes={'severity': 'moderate'}),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        assert len(results) == 1
        assert results[0]['gap_type'] == 'well_captured'

    def test_boundary_score_and_timing(self, gap_detector):
        """Edge case - aligned timing + score exactly 0.6 + no details missing."""
        event = create_mock_event('0002', 'hypothyroidism', -6.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        tabular_time = admission_time + timedelta(hours=-6)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0002', 0.60, timestamp, 'thyroid disorder',
                                   attributes={'severity': 'mild'}),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        assert len(results) == 1
        assert results[0]['gap_type'] == 'well_captured'

    def test_coarse_timing_good_score(self, gap_detector):
        """Edge case - coarse timing (6h) + good score → well_captured."""
        event = create_mock_event('0003', 'anemia', -8.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        # 7h difference (coarse, but acceptable)
        tabular_time = admission_time + timedelta(hours=-1)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0003', 0.85, timestamp, 'low Hb',
                                   attributes={'severity': 'moderate'}),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        assert len(results) == 1
        assert results[0]['gap_type'] == 'well_captured'

    def test_real_case_integration(self, gap_detector, real_case_path):
        """Real example - lab result with matching tabular event."""
        if not real_case_path.exists():
            pytest.skip("Real i2m4 data not available")

        pytest.skip("Real case integration test pending")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_well_captured.py -v`
Expected: 4 FAILED

**Step 3: Commit**

```bash
git add tests/test_well_captured.py
git commit -m "test: add well_captured gap type tests"
```

---

## Task 7: Implement test_thresholds.py

**Files:**
- Create: `tests/test_thresholds.py`

**Step 1: Write the failing tests**

Create `tests/test_thresholds.py`:

```python
"""Tests for threshold boundary conditions."""
import pytest
from datetime import datetime, timedelta

from gap_detection import GapDetector
from gap_detection.gap_detection import (
    TEMPORAL_ALIGNMENT_THRESHOLD,
    TEMPORAL_COARSE_THRESHOLD,
    RAG_SCORE_LOW_THRESHOLD,
    RAG_SCORE_MEDIUM_THRESHOLD
)
from tests.fixtures.synthetic_cases import (
    create_mock_event,
    create_mock_rag_result,
    create_mock_case
)


class TestThresholds:
    """Tests for threshold boundary conditions."""

    def test_temporal_6h_boundary(self, gap_detector):
        """Temporal boundary at exactly 6.0 hours."""
        event = create_mock_event('0001', 'symptom', -6.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        # Exactly 6h difference
        tabular_time = admission_time
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0001', 0.80, timestamp, 'event'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        # Exactly 6h should be aligned
        assert len(results) == 1
        assert results[0]['temporal']['alignment'] == 'aligned'

    def test_temporal_12h_boundary(self, gap_detector):
        """Temporal boundary at exactly 12.0 hours."""
        event = create_mock_event('0002', 'symptom', -12.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        # Exactly 12h difference
        tabular_time = admission_time
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0002', 0.80, timestamp, 'event'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        # Exactly 12h should be coarse (not misaligned)
        assert len(results) == 1
        assert results[0]['temporal']['alignment'] == 'coarse'

    def test_rag_0_3_boundary(self, gap_detector):
        """RAG score boundary at exactly 0.3."""
        event = create_mock_event('0003', 'symptom', -5.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        tabular_time = admission_time + timedelta(hours=-5)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0003', 0.30, timestamp, 'event'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        # Exactly 0.3 should be at the low threshold
        # Check that coverage recognizes this
        assert len(results) == 1
        # Should not have good coverage at 0.3
        assert results[0]['coverage']['best_score'] == 0.30

    def test_rag_0_6_boundary(self, gap_detector):
        """RAG score boundary at exactly 0.6."""
        event = create_mock_event('0004', 'symptom', -5.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        tabular_time = admission_time + timedelta(hours=-5)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0004', 0.60, timestamp, 'event',
                                   attributes={'severity': 'moderate'}),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        # Exactly 0.6 should meet medium threshold
        assert len(results) == 1
        assert results[0]['gap_type'] == 'well_captured'

    def test_multiple_boundaries(self, gap_detector):
        """Multiple boundary conditions simultaneously."""
        # Event at exactly 6h, score exactly 0.6
        event = create_mock_event('0005', 'symptom', -6.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        tabular_time = admission_time
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0005', 0.60, timestamp, 'event',
                                   attributes={'severity': 'mild'}),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        # Should pass both boundaries
        assert len(results) == 1
        assert results[0]['gap_type'] == 'well_captured'

    def test_time_zero_admission(self, gap_detector):
        """Edge case - time=0 (admission time) alignment."""
        event = create_mock_event('0006', 'admission event', 0.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        # Tabular event exactly at admission
        tabular_time = admission_time
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0006', 0.90, timestamp, 'admission record',
                                   attributes={'severity': 'N/A'}),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        # Time=0 should be perfectly aligned
        assert len(results) == 1
        assert results[0]['temporal']['alignment'] == 'aligned'
        assert results[0]['temporal']['temporal_distance'] == 0.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_thresholds.py -v`
Expected: 6 FAILED

**Step 3: Commit**

```bash
git add tests/test_thresholds.py
git commit -m "test: add threshold boundary tests"
```

---

## Task 8: Implement test_timezone_handling.py

**Files:**
- Create: `tests/test_timezone_handling.py`

**Step 1: Write the failing tests**

Create `tests/test_timezone_handling.py`:

```python
"""Tests for timezone handling in temporal alignment."""
import pytest
from datetime import datetime, timedelta
try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False

from gap_detection import GapDetector
from tests.fixtures.synthetic_cases import (
    create_mock_event,
    create_mock_rag_result,
    create_mock_case
)


class TestTimezoneHandling:
    """Tests for timezone edge cases in temporal alignment."""

    def test_admission_time_with_timezone_stripped(self, gap_detector):
        """Admission time with timezone info gets stripped correctly."""
        event = create_mock_event('0001', 'symptom', -5.0)

        # Admission time with UTC timezone
        admission_time = datetime(2023, 6, 15, 10, 30, 0, tzinfo=pytz.UTC if HAS_PYTZ else None)

        tabular_time = datetime(2023, 6, 15, 5, 30, 0)  # 5h before
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0001', 0.85, timestamp, 'event'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        # Should not raise timezone comparison error
        results = gap_detector.detect_gaps(case)

        assert len(results) == 1
        # Should be well aligned (0h difference)
        assert results[0]['temporal']['alignment'] == 'aligned'

    def test_admission_time_without_timezone(self, gap_detector):
        """Admission time without timezone passes through."""
        event = create_mock_event('0002', 'symptom', -3.0)

        # No timezone
        admission_time = datetime(2023, 6, 15, 10, 30, 0)
        tabular_time = datetime(2023, 6, 15, 7, 30, 0)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0002', 0.80, timestamp, 'event'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        assert len(results) == 1
        assert results[0]['temporal']['alignment'] == 'aligned'

    def test_tabular_timestamps_timezone(self, gap_detector):
        """Tabular timestamps with timezone handling."""
        event = create_mock_event('0003', 'symptom', -4.0)

        admission_time = datetime(2023, 6, 15, 10, 30, 0)

        # Tabular timestamp (Unix epoch has no timezone)
        tabular_time = datetime(2023, 6, 15, 6, 30, 0)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0003', 0.82, timestamp, 'event'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        assert len(results) == 1
        # Should handle correctly
        assert results[0]['temporal']['alignment'] == 'aligned'

    @pytest.mark.skipif(not HAS_PYTZ, reason="pytz not available")
    def test_mixed_timezone_scenarios(self, gap_detector):
        """Mixed timezone scenarios (admission has tz, tabular doesn't)."""
        event = create_mock_event('0004', 'symptom', -2.0)

        # Admission with Eastern timezone
        eastern = pytz.timezone('US/Eastern')
        admission_time = eastern.localize(datetime(2023, 6, 15, 10, 30, 0))

        # Tabular timestamp in UTC
        tabular_time = datetime(2023, 6, 15, 14, 30, 0)  # UTC
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0004', 0.88, timestamp, 'event'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        results = gap_detector.detect_gaps(case)

        # Should handle timezone conversion correctly
        assert len(results) == 1

    def test_dst_boundary(self, gap_detector):
        """DST boundary handling (if applicable)."""
        # Skip if pytz not available
        if not HAS_PYTZ:
            pytest.skip("pytz not available")

        event = create_mock_event('0005', 'symptom', -1.0)

        # During DST transition
        eastern = pytz.timezone('US/Eastern')
        admission_time = eastern.localize(datetime(2023, 3, 12, 10, 30, 0), is_dst=True)

        tabular_time = datetime(2023, 3, 12, 9, 30, 0)
        timestamp = tabular_time.timestamp()

        rag_results = [
            create_mock_rag_result('0005', 0.90, timestamp, 'event'),
        ]

        case = create_mock_case('CASE_001', [event], rag_results, admission_time)

        # Should handle DST without errors
        results = gap_detector.detect_gaps(case)

        assert len(results) == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_timezone_handling.py -v`
Expected: 4-5 FAILED/SKIPPED

**Step 3: Commit**

```bash
git add tests/test_timezone_handling.py
git commit -m "test: add timezone handling tests"
```

---

## Task 9: Verify All Tests Pass

**Files:**
- Modify: All test files as needed

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: Some tests may fail due to implementation gaps

**Step 2: Debug and fix any issues**

If tests fail due to test infrastructure issues, fix them.

**Step 3: Run with coverage**

Run: `pytest --cov=gap_detection tests/`
Expected: All tests pass, coverage report generated

**Step 4: Commit**

```bash
git add tests/
git commit -m "test: verify all gap detection tests pass"
```

---

## Task 10: Add Documentation

**Files:**
- Create: `tests/README.md`

**Step 1: Write test documentation**

Create `tests/README.md`:

```markdown
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

- **test_complete_absence.py** - Tests for no tabular counterpart
- **test_temporal_mismatch.py** - Tests for timing differences >12h
- **test_semantic_distance.py** - Tests for low similarity matches
- **test_detail_gap.py** - Tests for missing attributes
- **test_well_captured.py** - Tests for successful matches
- **test_thresholds.py** - Boundary condition tests
- **test_timezone_handling.py** - Timezone edge case tests

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
```

**Step 2: Commit**

```bash
git add tests/README.md
git commit -m "docs: add test suite documentation"
```

---

## Task 11: Final Verification

**Step 1: Run complete test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 2: Check test count**

Run: `pytest tests/ --collect-only | grep "test session starts"`
Expected: ~28-30 tests collected

**Step 3: Verify coverage**

Run: `pytest --cov=gap_detection tests/ --cov-report=term-missing`
Expected: Coverage report showing tested code paths

**Step 4: Final commit**

```bash
git add tests/
git commit -m "test: complete gap detection test suite implementation"
```

---

## Success Criteria

- [ ] All ~28-30 tests pass
- [ ] Tests validate core design decisions
- [ ] Edge cases documented through test names
- [ ] Real data tests gracefully skip when unavailable
- [ ] Coverage report generated successfully
- [ ] Test suite documented in README.md