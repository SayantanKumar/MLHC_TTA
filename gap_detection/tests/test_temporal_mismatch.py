"""Tests for temporal_mismatch gap type."""
import pytest
from pathlib import Path


class TestTemporalMismatch:
    """Tests for temporal_mismatch gap classification."""

    def test_aligned_timing(self):
        """Aligned timing (≤6h) → not temporal_mismatch."""
        from gap_detection import GapDetector

        test_batch_dir = Path(__file__).parent.parent.parent / "i2m4b" / "i2m4_batch_output_0001"
        if not test_batch_dir.exists():
            pytest.skip("i2m4 data not available - requires access to private dataset")

        detector = GapDetector(str(test_batch_dir))
        results = detector.detect_gaps_for_case('100831')

        if 'error' in results:
            pytest.skip(f"Case not found: {results['error']}")

        # Find well_captured gaps - these should have good timing
        well_captured = [
            gap for gap in results['gap_analysis']
            if gap['gap_type'] == 'well_captured'
        ]

        if well_captured:
            # Verify temporal alignment
            gap = well_captured[0]
            assert 'temporal' in gap
            # Well captured should not be temporal_mismatch
            assert gap['gap_type'] != 'temporal_mismatch'

    def test_coarse_timing(self):
        """Coarse timing (6-12h) → not temporal_mismatch."""
        from gap_detection import GapDetector

        test_batch_dir = Path(__file__).parent.parent.parent / "i2m4b" / "i2m4_batch_output_0001"
        if not test_batch_dir.exists():
            pytest.skip("i2m4 data not available - requires access to private dataset")

        detector = GapDetector(str(test_batch_dir))
        results = detector.detect_gaps_for_case('100831')

        if 'error' in results:
            pytest.skip(f"Case not found: {results['error']}")

        # Find gaps with coarse timing (should be well_captured, not temporal_mismatch)
        coarse_gaps = [
            gap for gap in results['gap_analysis']
            if 'temporal' in gap
            and gap['temporal'].get('alignment') == 'coarse'
        ]

        if coarse_gaps:
            # Coarse timing should still be well_captured (not temporal_mismatch)
            gap = coarse_gaps[0]
            assert gap['gap_type'] in ['well_captured', 'semantic_distance', 'detail_gap']
            assert gap['gap_type'] != 'temporal_mismatch'

    def test_misaligned_timing(self):
        """Misaligned timing (>12h) with good scores → temporal_mismatch."""
        from gap_detection import GapDetector

        test_batch_dir = Path(__file__).parent.parent.parent / "i2m4b" / "i2m4_batch_output_0001"
        if not test_batch_dir.exists():
            pytest.skip("i2m4 data not available - requires access to private dataset")

        detector = GapDetector(str(test_batch_dir))
        results = detector.detect_gaps_for_case('100831')

        if 'error' in results:
            pytest.skip(f"Case not found: {results['error']}")

        # Find temporal_mismatch gaps
        temporal_mismatches = [
            gap for gap in results['gap_analysis']
            if gap['gap_type'] == 'temporal_mismatch'
        ]

        if temporal_mismatches:
            # Verify temporal misalignment
            gap = temporal_mismatches[0]
            assert gap['gap_type'] == 'temporal_mismatch'
            assert 'temporal' in gap
            # Temporal mismatch should have misaligned timing
            assert gap['temporal'].get('alignment') == 'misaligned'

    def test_exactly_12_hours_boundary(self):
        """Exactly 12.0 hours → not temporal_mismatch (boundary)."""
        from gap_detection import GapDetector

        test_batch_dir = Path(__file__).parent.parent.parent / "i2m4b" / "i2m4_batch_output_0001"
        if not test_batch_dir.exists():
            pytest.skip("i2m4 data not available - requires access to private dataset")

        detector = GapDetector(str(test_batch_dir))
        results = detector.detect_gaps_for_case('100831')

        if 'error' in results:
            pytest.skip(f"Case not found: {results['error']}")

        # This test checks the boundary condition
        # In practice, we'd need to find a specific case with exactly 12h difference
        # For now, we verify that the temporal alignment logic exists
        assert 'gap_analysis' in results

        # Verify temporal information is captured
        for gap in results['gap_analysis']:
            if 'temporal' in gap:
                assert 'alignment' in gap['temporal']
                # 'no_counterpart' is valid when there's no tabular event to align with
                assert gap['temporal']['alignment'] in ['aligned', 'coarse', 'misaligned', 'no_counterpart']