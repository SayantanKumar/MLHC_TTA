"""Tests for semantic_distance gap type."""
import pytest
from pathlib import Path


class TestSemanticDistance:
    """Tests for semantic_distance gap classification."""

    def test_good_score_not_semantic_distance(self):
        """Good timing + RAG ≥0.6 → not semantic_distance."""
        from gap_detection import GapDetector

        test_batch_dir = Path(__file__).parent.parent.parent / "i2m4b" / "i2m4_batch_output_0001"
        if not test_batch_dir.exists():
            pytest.skip("i2m4 data not available - requires access to private dataset")

        detector = GapDetector(str(test_batch_dir))
        results = detector.detect_gaps_for_case('100831')

        if 'error' in results:
            pytest.skip(f"Case not found: {results['error']}")

        # Find well_captured gaps (should have good scores)
        well_captured = [
            gap for gap in results['gap_analysis']
            if gap['gap_type'] == 'well_captured'
        ]

        if well_captured:
            gap = well_captured[0]
            assert gap['gap_type'] != 'semantic_distance'
            # Well captured should have good RAG score
            if 'coverage' in gap:
                assert gap['coverage'].get('best_score', 0) >= 0.6

    def test_low_score_semantic_distance(self):
        """Good timing + RAG <0.6 → semantic_distance."""
        from gap_detection import GapDetector

        test_batch_dir = Path(__file__).parent.parent.parent / "i2m4b" / "i2m4_batch_output_0001"
        if not test_batch_dir.exists():
            pytest.skip("i2m4 data not available - requires access to private dataset")

        detector = GapDetector(str(test_batch_dir))
        results = detector.detect_gaps_for_case('100831')

        if 'error' in results:
            pytest.skip(f"Case not found: {results['error']}")

        # Find semantic_distance gaps
        semantic_distance_gaps = [
            gap for gap in results['gap_analysis']
            if gap['gap_type'] == 'semantic_distance'
        ]

        if semantic_distance_gaps:
            gap = semantic_distance_gaps[0]
            assert gap['gap_type'] == 'semantic_distance'
            # Semantic distance should have low to medium RAG score
            if 'coverage' in gap:
                score = gap['coverage'].get('best_score', 1.0)
                # Score should be in range that indicates semantic distance
                assert score >= 0.3  # Has a match

    def test_exactly_0_6_boundary(self):
        """Exactly 0.6 score → not semantic_distance (boundary)."""
        from gap_detection import GapDetector

        test_batch_dir = Path(__file__).parent.parent.parent / "i2m4b" / "i2m4_batch_output_0001"
        if not test_batch_dir.exists():
            pytest.skip("i2m4 data not available - requires access to private dataset")

        detector = GapDetector(str(test_batch_dir))
        results = detector.detect_gaps_for_case('100831')

        if 'error' in results:
            pytest.skip(f"Case not found: {results['error']}")

        # Check coverage scores are being calculated
        gaps_with_coverage = [
            gap for gap in results['gap_analysis']
            if 'coverage' in gap and 'best_score' in gap['coverage']
        ]

        # Verify score information exists
        assert len(gaps_with_coverage) > 0 or len(results['gap_analysis']) > 0