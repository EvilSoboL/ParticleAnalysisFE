"""
Модули выполнения фильтрации изображений.

Содержит executors для:
- Сортировки изображений по камерам
- Бинаризации изображений
- Фильтрации по интенсивности
- Farneback/Kanade фильтрации для оптического потока
"""

from .execute_sorting import SortingExecutor, SortingParameters
from .execute_binarization import BinarizationExecutor, BinarizationParameters
from .execute_intensity_filter import IntensityFilterExecutor, IntensityFilterParameters
from .execute_farneback_kanade import FarnebackKanadeExecutor, FarnebackKanadeParameters

__all__ = [
    'SortingExecutor',
    'SortingParameters',
    'BinarizationExecutor',
    'BinarizationParameters',
    'IntensityFilterExecutor',
    'IntensityFilterParameters',
    'FarnebackKanadeExecutor',
    'FarnebackKanadeParameters',
]
