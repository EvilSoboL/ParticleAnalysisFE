"""
Модуль визуализации для ParticleAnalysis.

Предоставляет функциональность для:
- Визуализации детектированных частиц
- Отображения центров и окружностей с эквивалентным диаметром
- Визуализации векторных полей PIV и Farneback
- Сохранения визуализированных изображений
"""

from .particle_visualization import (
    ParticleVisualizer,
    VisualizationProgress,
    VisualizationResult,
    VisualizationConfig
)

from .farneback_vector_field import (
    FarnebackVectorFieldVisualizer,
    FarnebackVectorFieldResult,
    FarnebackVectorFieldConfig
)

__all__ = [
    'ParticleVisualizer',
    'VisualizationProgress',
    'VisualizationResult',
    'VisualizationConfig',
    'FarnebackVectorFieldVisualizer',
    'FarnebackVectorFieldResult',
    'FarnebackVectorFieldConfig',
]
