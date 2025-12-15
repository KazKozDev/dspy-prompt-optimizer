"""
DSPy Distillation package.
Provides teacher-student distillation for model optimization.
"""

from .teacher_student import TeacherStudentDistiller, DistillationConfig

__all__ = [
    "TeacherStudentDistiller",
    "DistillationConfig",
]
