"""
Модуль execute для выполнения операций ParticleAnalysis.

Содержит готовые к использованию скрипты с параметрами для GUI.

Структура:
- execute_filter/  - модули фильтрации (сортировка, бинаризация, intensity filter)
- execute_analysis/ - модули анализа и визуализации (PTV, PIV)
"""

__version__ = "2.0.0"

# Реэкспорт из подмодулей для обратной совместимости
from .execute_filter import (
    SortingExecutor,
    SortingParameters,
    BinarizationExecutor,
    BinarizationParameters,
    IntensityFilterExecutor,
    IntensityFilterParameters,
)

from .execute_analysis import (
    PTVExecutor,
    PTVParameters,
    PIVExecutor,
    PIVParameters,
    VisualizationExecutor,
    VisualizationParameters,
    VectorFieldExecutor,
    VectorFieldParameters,
    PIVVisualizationExecutor,
    PIVVisualizationParameters,
    PIVVectorFieldExecutor,
    PIVVectorFieldParameters,
)

__all__ = [
    # Filters
    'SortingExecutor',
    'SortingParameters',
    'BinarizationExecutor',
    'BinarizationParameters',
    'IntensityFilterExecutor',
    'IntensityFilterParameters',
    # Analysis
    'PTVExecutor',
    'PTVParameters',
    'PIVExecutor',
    'PIVParameters',
    # Visualization
    'VisualizationExecutor',
    'VisualizationParameters',
    'VectorFieldExecutor',
    'VectorFieldParameters',
    'PIVVisualizationExecutor',
    'PIVVisualizationParameters',
    'PIVVectorFieldExecutor',
    'PIVVectorFieldParameters',
]
