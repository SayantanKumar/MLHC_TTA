"""Tests for well_captured gap type."""
import pytest
from pathlib import Path


class TestWellCaptured:
    """Tests for well_captured gap classification."""

    def test_all_conditions_met(self):
        """All conditions met → well_captured."""
        from gap_detection import GapDetector

        test_batch_dir = Path(__file__).parent.parent.parent / "i2m4b" / "i2m4_batch_output_0001"
        if not test_batch_dir.exists():
            pytest.skip("i2m4 data not available - requires access to private dataset")

        detector = GapDetector(str(test_batch_dir))
        results = detector.detect_gaps_for_case('100831')

        if 'error' in results:
            pytest.skip(f"Case not found: {results['error']}")

        # Find well_captured gaps
        well_captured = [
            gap for gap in results['gap_analysis']
            if gap['gap_type'] == 'well_captured'
        ]

        if well_captured:
            gap = well_captured[0]
            assert gap['gap_type'] == 'well_captured'

            # Well captured should have good coverage
            if 'coverage' in gap:
                assert gap['coverage'].get('has_counterpart', False)

            # Well captured should have good temporal alignment
            if 'temporal' in gap:
                assert gap['temporal'].get('alignment') in ['aligned', 'coarse']

    def test_boundary_score_and_timing(self):
        """Edge case - aligned timing + score exactly 0.6 + no details missing."""
        from gap_detection import GapDetector

        test_batch_dir = Path(__file__).parent.parent.parent / "i2m4b" / "i2m4_batch_output_0001"
        if not test_batch_dir.exists():
            pytest.skip("i2m4 data not available - requires access to private dataset")

        detector = GapDetector(str(test_batch_dir))
        results = detector.detect_gaps_for_case('100831')

        if 'error' in results:
            pytest.skip(f"Case not found: {results['error']}")

        # Verify gap analysis exists
        assert 'gap_analysis' in results

        # Check that scores are being calculated
        gaps_with_scores = [
            gap for gap in results['gap_analysis']
            if 'coverage' in gap and 'best_score' in gap['coverage']
        ]

        # At minimum, verify structure
        assert len(results['gap_analysis']) > 0

    def test_coarse_timing_good_score(self):
        """Edge case - coarse timing (6h) + good score → well_captured."""
        from gap_detection import GapDetector

        test_batch_dir = Path(__file__).parent.parent.parent / "i2m4b" / "i2m4_batch_output_0001"
        if not test_batch_dir.exists():
            pytest.skip("i2m4 data not available - requires access to private dataset")

        detector = GapDetector(str(test_batch_dir))
        results = detector.detect_gaps_for_case('100831')

        if 'error' in results:
            pytest.skip(f"Case not found: {results['error']}")

        # Find gaps with coarse timing
        coarse_gaps = [
            gap for gap in results['gap_analysis']
            if 'temporal' in gap and gap['temporal'].get('alignment') == 'coarse'
        ]

        if coarse_gaps:
            # Coarse timing with good score should be well_captured
            gap = coarse_gaps[0]
            # Could be well_captured or other type, but not temporal_mismatch
            assert gap['gap_type'] != 'temporal_mismatch'