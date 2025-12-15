"""
Optimizer Selector - Selects appropriate DSPy optimizer.
"""

from typing import Optional

from ..config import (
    TaskAnalysis, OptimizerConfig, OptimizerType, ComplexityLevel
)


class OptimizerSelector:
    """
    Selects DSPy optimizer based on task analysis and data characteristics.
    """
    
    def select(
        self,
        task_analysis: TaskAnalysis,
        dataset_size: int,
        quality_profile: str = "BALANCED",
        target_model: Optional[str] = None,
        optimizer_model: Optional[str] = None
    ) -> OptimizerConfig:
        """
        Select optimizer configuration.
        
        Args:
            task_analysis: Result of task analysis
            dataset_size: Number of training examples
            quality_profile: Quality profile
            target_model: Target model for inference
            optimizer_model: Model for optimization
            
        Returns:
            OptimizerConfig with selected settings
        """
        optimizer_type = self._select_optimizer_type(
            task_analysis, dataset_size, quality_profile
        )
        
        params = self._select_params(optimizer_type, dataset_size, quality_profile)
        
        teacher_model = None
        student_model = None
        distillation_samples = 100
        
        if optimizer_type == OptimizerType.DISTILLATION:
            teacher_model = optimizer_model or "openai/gpt-4o"
            student_model = target_model
            distillation_samples = min(200, dataset_size * 2)
        
        return OptimizerConfig(
            optimizer_type=optimizer_type,
            max_bootstrapped_demos=params["max_bootstrapped_demos"],
            max_labeled_demos=params["max_labeled_demos"],
            max_rounds=params["max_rounds"],
            num_candidates=params["num_candidates"],
            teacher_model=teacher_model,
            student_model=student_model,
            distillation_samples=distillation_samples
        )
    
    def _select_optimizer_type(
        self,
        task_analysis: TaskAnalysis,
        dataset_size: int,
        quality_profile: str
    ) -> OptimizerType:
        """Select optimizer type."""
        if quality_profile == "FAST_CHEAP":
            return OptimizerType.BOOTSTRAP_FEW_SHOT
        
        if dataset_size < 20:
            return OptimizerType.BOOTSTRAP_FEW_SHOT
        
        if quality_profile == "HIGH_QUALITY":
            if dataset_size >= 50:
                return OptimizerType.MIPRO_V2
            if dataset_size >= 30:
                return OptimizerType.BOOTSTRAP_RANDOM_SEARCH
        
        if task_analysis.complexity == ComplexityLevel.HIGH:
            if dataset_size >= 30:
                return OptimizerType.MIPRO_V2
        
        if dataset_size >= 30:
            return OptimizerType.BOOTSTRAP_RANDOM_SEARCH
        
        return OptimizerType.BOOTSTRAP_FEW_SHOT
    
    def _select_params(
        self,
        optimizer_type: OptimizerType,
        dataset_size: int,
        quality_profile: str
    ) -> dict:
        """Select optimizer parameters."""
        base_params = {
            OptimizerType.BOOTSTRAP_FEW_SHOT: {
                "max_bootstrapped_demos": 2,
                "max_labeled_demos": 4,
                "max_rounds": 1,
                "num_candidates": 1
            },
            OptimizerType.BOOTSTRAP_RANDOM_SEARCH: {
                "max_bootstrapped_demos": 3,
                "max_labeled_demos": 8,
                "max_rounds": 1,
                "num_candidates": 12
            },
            OptimizerType.MIPRO_V2: {
                "max_bootstrapped_demos": 4,
                "max_labeled_demos": 4,
                "max_rounds": 1,
                "num_candidates": 16
            },
            OptimizerType.COPRO: {
                "max_bootstrapped_demos": 0,
                "max_labeled_demos": 0,
                "max_rounds": 3,
                "num_candidates": 10
            },
            OptimizerType.DISTILLATION: {
                "max_bootstrapped_demos": 4,
                "max_labeled_demos": 16,
                "max_rounds": 1,
                "num_candidates": 1
            }
        }
        
        params = base_params.get(optimizer_type, base_params[OptimizerType.BOOTSTRAP_FEW_SHOT])
        
        if quality_profile == "HIGH_QUALITY":
            params["max_bootstrapped_demos"] = min(params["max_bootstrapped_demos"] + 2, 8)
            params["max_labeled_demos"] = min(params["max_labeled_demos"] + 4, 16)
            params["num_candidates"] = min(params["num_candidates"] + 4, 24)
        elif quality_profile == "FAST_CHEAP":
            params["max_bootstrapped_demos"] = max(params["max_bootstrapped_demos"] - 1, 1)
            params["max_labeled_demos"] = max(params["max_labeled_demos"] - 2, 2)
            params["num_candidates"] = max(params["num_candidates"] // 2, 4)
        
        max_demos = min(dataset_size // 2, params["max_labeled_demos"])
        params["max_labeled_demos"] = max(max_demos, 2)
        
        return params
