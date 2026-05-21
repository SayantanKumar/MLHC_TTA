"""Shared pytest fixtures for gap detection tests."""
import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Path to i2m4 test data directory (actual batch output)."""
    return Path(__file__).parent.parent.parent / "i2m4b" / "i2m4_batch_output_0001"


@pytest.fixture
def skip_if_no_data(test_data_dir):
    """Skip test if i2m4 data not available."""
    if not test_data_dir.exists():
        pytest.skip("i2m4 data not available - requires access to private dataset")
    return test_data_dir