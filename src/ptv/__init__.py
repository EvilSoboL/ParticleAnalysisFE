"""
Модуль PTV (Particle Tracking Velocimetry) анализа.

Предоставляет функциональность для:
- Детектирования частиц на бинаризованных изображениях
- Сопоставления частиц между последовательными кадрами
- Расчета векторов смещения и скоростей
- Создания суммарных CSV файлов с парами
"""

from .ptv_analysis import (
    PTVAnalyzer,
    PTVProgress,
    PTVResult,
    Particle,
    ParticlePair,
    DetectionConfig,
    MatchingConfig
)

__all__ = [
    'PTVAnalyzer',
    'PTVProgress',
    'PTVResult',
    'Particle',
    'ParticlePair',
    'DetectionConfig',
    'MatchingConfig'
]
