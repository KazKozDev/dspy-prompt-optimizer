"""DSPy Distillation package.
Provides teacher-student distillation for model optimization.
"""

from .teacher_student import DistillationConfig, TeacherStudentDistiller

__all__ = [
    "TeacherStudentDistiller",
    "DistillationConfig",
]
