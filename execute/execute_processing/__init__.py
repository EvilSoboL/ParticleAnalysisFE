"""
Модуль обработки данных для GUI ParticleAnalysis.

Содержит:
- VectorFilterExecutor: Фильтрация векторов по U и V
- VectorFilterParameters: Параметры фильтрации
- VectorFilterResult: Результат фильтрации
"""

from .vector_filter import (
    VectorFilterExecutor,
    VectorFilterParameters,
    VectorFilterResult,
    run_vector_filter
)

__all__ = [
    'VectorFilterExecutor',
    'VectorFilterParameters',
    'VectorFilterResult',
    'run_vector_filter',
]
