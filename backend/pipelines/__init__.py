"""DSPy Pipelines package.
Provides multi-stage pipeline building and templates.
"""

from .builder import PipelineBuilder, PipelineStage
from .templates import (
    PIPELINE_TEMPLATES,
    get_template,
    list_templates,
)

__all__ = [
    "PipelineBuilder",
    "PipelineStage",
    "get_template",
    "list_templates",
    "PIPELINE_TEMPLATES",
]
