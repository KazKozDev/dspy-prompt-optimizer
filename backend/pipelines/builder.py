"""Pipeline Builder - Fluent API for building multi-stage DSPy pipelines."""

from dataclasses import dataclass, field
from typing import Any

dspy = None


def _load_dspy():
    """Lazy load DSPy."""
    global dspy
    if dspy is None:
        import dspy as _dspy

        dspy = _dspy
    return dspy


@dataclass
class PipelineStage:
    """Definition of a single pipeline stage."""

    name: str
    module_type: str  # "predict", "chain_of_thought", "retrieve", "react", "custom"
    input_fields: list[str]
    output_fields: list[str]
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


class MultiStageProgram:
    """Dynamic multi-stage DSPy program.

    Composes multiple DSPy modules into a single program.
    """

    def __init__(self, stages: list[PipelineStage], tools: list | None = None):
        """Initialize multi-stage program.

        Args:
            stages: List of pipeline stages
            tools: Optional tools for ReAct stages
        """
        dspy = _load_dspy()

        self.stages = stages
        self.tools = tools or []
        self.modules: dict[str, Any] = {}

        for stage in stages:
            module = self._create_module(stage)
            self.modules[stage.name] = module

    def _create_signature(self, stage: PipelineStage) -> type:
        """Create dynamic signature for stage."""
        dspy = _load_dspy()

        sig_fields = {}
        for field_name in stage.input_fields:
            sig_fields[field_name] = dspy.InputField(desc=f"{field_name} input")
        for field_name in stage.output_fields:
            sig_fields[field_name] = dspy.OutputField(desc=f"{field_name} output")

        return type(
            f"{stage.name}Signature",
            (dspy.Signature,),
            {"__doc__": stage.description, **sig_fields},
        )

    def _create_module(self, stage: PipelineStage) -> Any:
        """Create DSPy module for stage."""
        dspy = _load_dspy()

        signature = self._create_signature(stage)

        if stage.module_type == "predict":
            return dspy.Predict(signature)
        elif stage.module_type == "chain_of_thought":
            return dspy.ChainOfThought(signature)
        elif stage.module_type == "react":
            if hasattr(dspy, "ReAct"):
                return dspy.ReAct(signature, tools=self.tools)
            return dspy.ChainOfThought(signature)
        elif stage.module_type == "retrieve":
            k = stage.config.get("k", 5)
            return dspy.Retrieve(k=k)
        else:
            return dspy.Predict(signature)

    def forward(self, **kwargs) -> dict[str, Any]:
        """Execute the pipeline.

        Args:
            **kwargs: Input fields for the first stage

        Returns:
            Dict with outputs from all stages
        """
        context = dict(kwargs)
        results = {}

        for stage in self.stages:
            stage_inputs = {}
            for field_name in stage.input_fields:
                if field_name in context:
                    stage_inputs[field_name] = context[field_name]

            module = self.modules[stage.name]

            if stage.module_type == "retrieve":
                query = stage_inputs.get("query", stage_inputs.get("text", ""))
                result = module(query)
                context["passages"] = (
                    result.passages if hasattr(result, "passages") else result
                )
                results[stage.name] = {"passages": context["passages"]}
            else:
                result = module(**stage_inputs)

                for field_name in stage.output_fields:
                    if hasattr(result, field_name):
                        context[field_name] = getattr(result, field_name)

                results[stage.name] = {
                    field_name: getattr(result, field_name, None)
                    for field_name in stage.output_fields
                }

        return results

    def __call__(self, **kwargs) -> dict[str, Any]:
        """Call the pipeline."""
        return self.forward(**kwargs)


class PipelineBuilder:
    """Fluent API for building multi-stage DSPy pipelines.

    Example:
        pipeline = (
            PipelineBuilder()
            .add_stage("outline", "chain_of_thought", ["topic"], ["outline"])
            .add_stage("draft", "chain_of_thought", ["outline"], ["draft"])
            .add_stage("revise", "chain_of_thought", ["draft"], ["final"])
            .build()
        )
    """

    def __init__(self):
        """Initialize empty pipeline builder."""
        self.stages: list[PipelineStage] = []
        self.tools: list = []

    def add_stage(
        self,
        name: str,
        module_type: str,
        input_fields: list[str],
        output_fields: list[str],
        description: str = "",
        depends_on: list[str] | None = None,
        **config,
    ) -> "PipelineBuilder":
        """Add a stage to the pipeline.

        Args:
            name: Unique name for the stage
            module_type: Type of DSPy module ("predict", "chain_of_thought", "react", "retrieve")
            input_fields: List of input field names
            output_fields: List of output field names
            description: Optional description
            depends_on: Optional list of stage names this depends on
            **config: Additional configuration for the module

        Returns:
            self for chaining
        """
        stage = PipelineStage(
            name=name,
            module_type=module_type,
            input_fields=input_fields,
            output_fields=output_fields,
            description=description,
            depends_on=depends_on or [],
            config=config,
        )
        self.stages.append(stage)
        return self

    def add_predict(
        self,
        name: str,
        input_fields: list[str],
        output_fields: list[str],
        description: str = "",
    ) -> "PipelineBuilder":
        """Add a Predict stage."""
        return self.add_stage(name, "predict", input_fields, output_fields, description)

    def add_cot(
        self,
        name: str,
        input_fields: list[str],
        output_fields: list[str],
        description: str = "",
    ) -> "PipelineBuilder":
        """Add a ChainOfThought stage."""
        return self.add_stage(
            name, "chain_of_thought", input_fields, output_fields, description
        )

    def add_react(
        self,
        name: str,
        input_fields: list[str],
        output_fields: list[str],
        description: str = "",
    ) -> "PipelineBuilder":
        """Add a ReAct stage."""
        return self.add_stage(name, "react", input_fields, output_fields, description)

    def add_retrieve(self, name: str = "retrieve", k: int = 5) -> "PipelineBuilder":
        """Add a Retrieve stage."""
        return self.add_stage(
            name, "retrieve", ["query"], ["passages"], "Retrieve relevant passages", k=k
        )

    def with_tools(self, tools: list) -> "PipelineBuilder":
        """Add tools for ReAct stages.

        Args:
            tools: List of tools

        Returns:
            self for chaining
        """
        self.tools = tools
        return self

    def build(self) -> MultiStageProgram:
        """Build the pipeline.

        Returns:
            MultiStageProgram instance
        """
        if not self.stages:
            raise ValueError("Pipeline must have at least one stage")

        return MultiStageProgram(self.stages, self.tools)

    def to_dict(self) -> dict[str, Any]:
        """Convert pipeline definition to dict."""
        return {
            "stages": [
                {
                    "name": s.name,
                    "module_type": s.module_type,
                    "input_fields": s.input_fields,
                    "output_fields": s.output_fields,
                    "description": s.description,
                    "depends_on": s.depends_on,
                    "config": s.config,
                }
                for s in self.stages
            ],
            "has_tools": len(self.tools) > 0,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineBuilder":
        """Create pipeline builder from dict."""
        builder = cls()
        for stage_data in data.get("stages", []):
            builder.add_stage(
                name=stage_data["name"],
                module_type=stage_data["module_type"],
                input_fields=stage_data["input_fields"],
                output_fields=stage_data["output_fields"],
                description=stage_data.get("description", ""),
                depends_on=stage_data.get("depends_on"),
                **stage_data.get("config", {}),
            )
        return builder
