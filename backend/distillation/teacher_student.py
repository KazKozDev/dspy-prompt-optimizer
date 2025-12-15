"""Teacher-Student Distillation - Transfer knowledge from large to small models."""

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy

from utils.settings import get_settings

dspy = None


def _load_dspy():
    """Lazy load DSPy."""
    global dspy
    if dspy is None:
        import dspy as _dspy

        dspy = _dspy
    return dspy


@dataclass
class DistillationConfig:
    """Configuration for teacher-student distillation."""

    teacher_model: str = "openai/gpt-5"
    student_model: str = ""

    num_samples: int = 100
    temperature: float = 0.7

    batch_size: int = 10
    max_retries: int = 3

    save_intermediate: bool = True
    output_dir: str = "data/distillation"

    def __post_init__(self) -> None:
        if not self.student_model:
            settings = get_settings()
            self.student_model = f"openai/{settings.model_defaults.openai_chat}"


@dataclass
class DistillationResult:
    """Result of distillation process."""

    num_generated: int
    num_successful: int
    generated_data: list[dict[str, str]]
    teacher_model: str
    student_model: str
    duration_seconds: float
    output_path: str | None = None


class TeacherStudentDistiller:
    """Teacher-Student Distillation for DSPy.

    Uses a powerful "teacher" model to generate high-quality training data,
    then optimizes a smaller "student" model on this data.

    Benefits:
    - Reduce inference costs by using smaller models
    - Transfer reasoning capabilities from large to small models
    - Generate labeled data for tasks without ground truth
    """

    def __init__(self, config: DistillationConfig | None = None):
        """Initialize distiller.

        Args:
            config: Distillation configuration
        """
        self.config = config or DistillationConfig()

        self._teacher_lm = None
        self._student_lm = None

    def _get_lm(self, model_string: str):
        """Get DSPy LM from model string."""
        dspy = _load_dspy()

        if "/" in model_string:
            provider, model_name = model_string.split("/", 1)
        else:
            provider = "openai"
            model_name = model_string

        if provider == "ollama":
            base_url = os.getenv(
                "OLLAMA_BASE_URL", get_settings().endpoints.ollama_base_url
            )
            return dspy.LM(f"ollama_chat/{model_name}", api_base=base_url)
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            return dspy.LM(f"anthropic/{model_name}", api_key=api_key)
        elif provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            return dspy.LM(f"google/{model_name}", api_key=api_key)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            return dspy.LM(f"openai/{model_name}", api_key=api_key)

    def _get_teacher(self):
        """Get teacher LM."""
        if self._teacher_lm is None:
            self._teacher_lm = self._get_lm(self.config.teacher_model)
        return self._teacher_lm

    def _get_student(self):
        """Get student LM."""
        if self._student_lm is None:
            self._student_lm = self._get_lm(self.config.student_model)
        return self._student_lm

    def generate_training_data(
        self,
        task_description: str,
        input_examples: list[str],
        input_field: str = "input",
        output_field: str = "output",
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, str]]:
        """Generate training data using teacher model.

        Args:
            task_description: Description of the task
            input_examples: List of input examples (unlabeled)
            input_field: Name of input field
            output_field: Name of output field
            progress_callback: Optional callback(current, total)

        Returns:
            List of {input, output} dicts
        """
        dspy = _load_dspy()
        teacher = self._get_teacher()

        # Create signature for teacher
        class TeacherSignature(dspy.Signature):
            """Generate output for the given task."""

            task: str = dspy.InputField(desc="Task description")
            input_text: str = dspy.InputField(desc="Input to process")
            output: str = dspy.OutputField(desc="Generated output")

        # Configure DSPy with teacher
        dspy.configure(lm=teacher)
        predictor = dspy.ChainOfThought(TeacherSignature)

        generated_data = []
        total = len(input_examples)

        for i, input_text in enumerate(input_examples):
            try:
                result = predictor(task=task_description, input_text=input_text)

                generated_data.append(
                    {input_field: input_text, output_field: result.output}
                )

            except Exception as e:
                print(f"Error generating for example {i}: {e}")
                continue

            if progress_callback:
                progress_callback(i + 1, total)

        return generated_data

    async def generate_training_data_async(
        self,
        task_description: str,
        input_examples: list[str],
        input_field: str = "input",
        output_field: str = "output",
    ) -> list[dict[str, str]]:
        """Async version of generate_training_data.

        Processes examples in batches for better performance.
        """
        dspy = _load_dspy()
        teacher = self._get_teacher()

        class TeacherSignature(dspy.Signature):
            """Generate output for the given task."""

            task: str = dspy.InputField(desc="Task description")
            input_text: str = dspy.InputField(desc="Input to process")
            output: str = dspy.OutputField(desc="Generated output")

        dspy.configure(lm=teacher)
        predictor = dspy.ChainOfThought(TeacherSignature)

        generated_data = []
        batch_size = self.config.batch_size

        for batch_start in range(0, len(input_examples), batch_size):
            batch = input_examples[batch_start : batch_start + batch_size]

            for input_text in batch:
                try:
                    result = predictor(task=task_description, input_text=input_text)
                    generated_data.append(
                        {input_field: input_text, output_field: result.output}
                    )
                except Exception as e:
                    print(f"Error: {e}")
                    continue

            # Small delay between batches
            await asyncio.sleep(0.1)

        return generated_data

    def distill(
        self,
        task_description: str,
        unlabeled_inputs: list[str],
        metric: Callable | None = None,
        optimizer_type: str = "BootstrapFewShot",
    ) -> dict[str, Any]:
        """Full distillation pipeline.

        1. Generate training data with teacher
        2. Optimize student on generated data

        Args:
            task_description: Task description
            unlabeled_inputs: Unlabeled input examples
            metric: Evaluation metric
            optimizer_type: DSPy optimizer to use

        Returns:
            Dict with results and optimized program
        """
        import time

        start_time = time.time()

        dspy = _load_dspy()
        from dspy import teleprompt

        # Step 1: Generate training data with teacher
        print(f"Generating training data with {self.config.teacher_model}...")

        num_samples = min(self.config.num_samples, len(unlabeled_inputs))
        samples = unlabeled_inputs[:num_samples]

        generated_data = self.generate_training_data(
            task_description=task_description, input_examples=samples
        )

        if not generated_data:
            raise ValueError("Failed to generate any training data")

        print(f"Generated {len(generated_data)} training examples")

        # Save generated data
        if self.config.save_intermediate:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_path = output_dir / f"distillation_data_{timestamp}.json"

            with open(data_path, "w") as f:
                json.dump(
                    {
                        "task": task_description,
                        "teacher": self.config.teacher_model,
                        "student": self.config.student_model,
                        "data": generated_data,
                    },
                    f,
                    indent=2,
                )

        # Step 2: Create training set
        trainset = []
        for item in generated_data:
            example = dspy.Example(
                input=item.get("input", ""), output=item.get("output", "")
            ).with_inputs("input")
            trainset.append(example)

        # Step 3: Create student program
        class StudentSignature(dspy.Signature):
            """Process input and generate output."""

            input: str = dspy.InputField()
            output: str = dspy.OutputField()

        student = self._get_student()
        dspy.configure(lm=student)

        student_program = dspy.ChainOfThought(StudentSignature)

        # Step 4: Create metric if not provided
        if metric is None:

            def default_metric(example, pred, trace=None):
                gold = str(getattr(example, "output", "")).strip().lower()
                predicted = str(getattr(pred, "output", "")).strip().lower()

                if gold == predicted:
                    return 1.0
                if gold in predicted or predicted in gold:
                    return 0.5

                # Token overlap
                gold_tokens = set(gold.split())
                pred_tokens = set(predicted.split())
                if not gold_tokens or not pred_tokens:
                    return 0.0
                overlap = len(gold_tokens & pred_tokens)
                return overlap / max(len(gold_tokens), len(pred_tokens))

            metric = default_metric

        # Step 5: Optimize student
        print(f"Optimizing student model {self.config.student_model}...")

        BootstrapFewShot = getattr(teleprompt, "BootstrapFewShot", None)

        if BootstrapFewShot:
            optimizer = BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=min(16, len(trainset)),
                max_rounds=1,
            )

            optimized_student = optimizer.compile(student_program, trainset=trainset)
        else:
            optimized_student = student_program

        duration = time.time() - start_time

        # Evaluate
        correct = 0
        eval_count = min(10, len(trainset))
        for example in trainset[:eval_count]:
            try:
                pred = optimized_student(input=example.input)
                score = metric(example, pred)
                correct += score
            except Exception:
                pass

        final_score = correct / eval_count if eval_count > 0 else 0.0

        return {
            "status": "success",
            "num_generated": len(generated_data),
            "num_training": len(trainset),
            "teacher_model": self.config.teacher_model,
            "student_model": self.config.student_model,
            "metric_score": round(final_score, 3),
            "duration_seconds": round(duration, 2),
            "optimized_program": optimized_student,
            "generated_data": generated_data,
        }

    def save_generated_data(self, data: list[dict[str, str]], path: str) -> None:
        """Save generated training data to file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_generated_data(self, path: str) -> list[dict[str, str]]:
        """Load generated training data from file."""
        with open(path) as f:
            return json.load(f)
