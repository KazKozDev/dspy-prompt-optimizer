"""Tests for pipelines module."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines import PipelineBuilder, PipelineStage
from pipelines.templates import (
    get_template,
    list_templates,
    suggest_template,
    PIPELINE_TEMPLATES,
)


class TestPipelineStage:
    """Tests for PipelineStage dataclass."""

    def test_stage_creation(self):
        """Test creating a pipeline stage."""
        stage = PipelineStage(
            name="test_stage",
            module_type="predict",
            input_fields=["text"],
            output_fields=["result"],
            description="Test stage",
        )

        assert stage.name == "test_stage"
        assert stage.module_type == "predict"
        assert stage.input_fields == ["text"]
        assert stage.output_fields == ["result"]

    def test_stage_defaults(self):
        """Test stage default values."""
        stage = PipelineStage(
            name="test", module_type="predict", input_fields=["x"], output_fields=["y"]
        )

        assert stage.depends_on == []
        assert stage.config == {}


class TestPipelineBuilder:
    """Tests for PipelineBuilder."""

    def test_add_stage(self):
        """Test adding a stage."""
        builder = PipelineBuilder()
        builder.add_stage(
            name="step1",
            module_type="predict",
            input_fields=["text"],
            output_fields=["result"],
        )

        assert len(builder.stages) == 1
        assert builder.stages[0].name == "step1"

    def test_fluent_api(self):
        """Test fluent API chaining."""
        builder = (
            PipelineBuilder()
            .add_stage("step1", "predict", ["text"], ["intermediate"])
            .add_stage("step2", "predict", ["intermediate"], ["result"])
        )

        assert len(builder.stages) == 2

    def test_add_predict(self):
        """Test add_predict helper."""
        builder = PipelineBuilder()
        builder.add_predict("pred", ["text"], ["result"])

        assert builder.stages[0].module_type == "predict"

    def test_add_cot(self):
        """Test add_cot helper."""
        builder = PipelineBuilder()
        builder.add_cot("cot", ["text"], ["result"])

        assert builder.stages[0].module_type == "chain_of_thought"

    def test_add_react(self):
        """Test add_react helper."""
        builder = PipelineBuilder()
        builder.add_react("agent", ["task"], ["result"])

        assert builder.stages[0].module_type == "react"

    def test_add_retrieve(self):
        """Test add_retrieve helper."""
        builder = PipelineBuilder()
        builder.add_retrieve("retriever", k=10)

        assert builder.stages[0].module_type == "retrieve"
        assert builder.stages[0].config.get("k") == 10

    def test_with_tools(self):
        """Test adding tools."""
        builder = PipelineBuilder()
        builder.with_tools(["calculator", "web_search"])

        assert builder.tools == ["calculator", "web_search"]

    def test_to_dict(self):
        """Test serialization to dict."""
        builder = PipelineBuilder().add_stage(
            "step1", "predict", ["text"], ["result"], "Test step"
        )

        data = builder.to_dict()

        assert "stages" in data
        assert len(data["stages"]) == 1
        assert data["stages"][0]["name"] == "step1"

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "stages": [
                {
                    "name": "step1",
                    "module_type": "predict",
                    "input_fields": ["text"],
                    "output_fields": ["result"],
                    "description": "Test",
                }
            ]
        }

        builder = PipelineBuilder.from_dict(data)

        assert len(builder.stages) == 1
        assert builder.stages[0].name == "step1"

    def test_build_empty_raises(self):
        """Test that building empty pipeline raises error."""
        builder = PipelineBuilder()

        with pytest.raises(ValueError):
            builder.build()


class TestPipelineTemplates:
    """Tests for pipeline templates."""

    def test_templates_exist(self):
        """Test that templates are defined."""
        assert len(PIPELINE_TEMPLATES) > 0

    def test_get_template(self):
        """Test getting a template."""
        template = get_template("chain_of_thought")

        assert template is not None
        assert "name" in template
        assert "stages" in template

    def test_get_nonexistent_template(self):
        """Test getting nonexistent template."""
        template = get_template("nonexistent")

        assert template is None

    def test_list_templates(self):
        """Test listing templates."""
        templates = list_templates()

        assert len(templates) > 0
        assert all("name" in t for t in templates)
        assert all("description" in t for t in templates)

    def test_suggest_template_classification(self):
        """Test template suggestion for classification."""
        template = suggest_template("classification")

        assert template == "classify_explain"

    def test_suggest_template_with_retrieval(self):
        """Test template suggestion with retrieval."""
        template = suggest_template("qa", needs_retrieval=True)

        assert template == "rag_basic"

    def test_suggest_template_with_tools(self):
        """Test template suggestion with tools."""
        template = suggest_template("reasoning", needs_tools=True)

        assert template == "react_agent"

    def test_suggest_template_generation(self):
        """Test template suggestion for generation."""
        template = suggest_template("generation")

        assert template == "outline_draft_revise"

    def test_all_templates_have_required_fields(self):
        """Test that all templates have required fields."""
        for name, template in PIPELINE_TEMPLATES.items():
            assert "name" in template, f"Template {name} missing 'name'"
            assert "description" in template, f"Template {name} missing 'description'"
            assert "stages" in template, f"Template {name} missing 'stages'"
            assert len(template["stages"]) > 0, f"Template {name} has no stages"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
