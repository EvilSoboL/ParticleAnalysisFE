"""
Модуль обработки данных для GUI ParticleAnalysis.

Содержит:
- VectorFilterExecutor: Фильтрация векторов по U и V
- VectorFilterParameters: Параметры фильтрации
- VectorFilterResult: Результат фильтрации
- VectorAverageExecutor: Усреднение векторов по ячейкам сетки
- VectorAverageParameters: Параметры усреднения
- VectorAverageResult: Результат усреднения
"""

from .vector_filter import (
    VectorFilterExecutor,
    VectorFilterParameters,
    VectorFilterResult,
    run_vector_filter
)

from .vector_average import (
    VectorAverageExecutor,
    VectorAverageParameters,
    VectorAverageResult,
    run_vector_average
)

__all__ = [
    'VectorFilterExecutor',
    'VectorFilterParameters',
    'VectorFilterResult',
    'run_vector_filter',
    'VectorAverageExecutor',
    'VectorAverageParameters',
    'VectorAverageResult',
    'run_vector_average',
]
