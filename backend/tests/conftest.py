"""Pytest configuration and fixtures."""

import os
import sys

# Add repo root and backend to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, BACKEND_DIR)

import pytest


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
