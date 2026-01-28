"""
Модуль анализа оптического потока методом Farneback.

Содержит:
- FarnebackAnalyzer: Анализ оптического потока для пар изображений
- FarnebackConfig: Конфигурация алгоритма Farneback
- FarnebackResult: Результат анализа
- FarnebackProgress: Прогресс обработки
"""

from .farneback_analysis import (
    FarnebackAnalyzer,
    FarnebackConfig,
    FarnebackResult,
    FarnebackProgress
)

__all__ = [
    'FarnebackAnalyzer',
    'FarnebackConfig',
    'FarnebackResult',
    'FarnebackProgress',
]
