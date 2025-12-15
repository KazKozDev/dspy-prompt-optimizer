"""
Pytest configuration and fixtures.
"""

import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return [
        {"input": "Great product!", "output": "positive"},
        {"input": "Terrible service", "output": "negative"},
        {"input": "It's okay", "output": "neutral"},
        {"input": "Love it!", "output": "positive"},
        {"input": "Worst ever", "output": "negative"},
    ]


@pytest.fixture
def sample_task():
    """Sample business task."""
    return "Classify customer reviews as positive, negative, or neutral"
