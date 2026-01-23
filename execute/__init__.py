"""
Модуль execute для выполнения операций ParticleAnalysis.

Содержит готовые к использованию скрипты с параметрами для GUI.
"""

__version__ = "1.0.0"

# Импорты основных модулей
from .execute_sorting import SortingExecutor, SortingParameters
from .execute_binarization import BinarizationExecutor, BinarizationParameters
from .execute_intensity_filter import IntensityFilterExecutor, IntensityFilterParameters
from .execute_ptv_analysis import PTVExecutor, PTVParameters
from .execute_piv_analysis import PIVExecutor, PIVParameters
from .execute_visualization import VisualizationExecutor, VisualizationParameters
from .execute_vector_field import VectorFieldExecutor, VectorFieldParameters

__all__ = [
    'SortingExecutor',
    'SortingParameters',
    'BinarizationExecutor',
    'BinarizationParameters',
    'IntensityFilterExecutor',
    'IntensityFilterParameters',
    'PTVExecutor',
    'PTVParameters',
    'PIVExecutor',
    'PIVParameters',
    'VisualizationExecutor',
    'VisualizationParameters',
    'VectorFieldExecutor',
    'VectorFieldParameters',
]
