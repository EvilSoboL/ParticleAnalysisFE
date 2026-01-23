"""
Модуль выполнения PIV анализа для GUI ParticleAnalysis.

Этот модуль предоставляет готовую к использованию структуру для интеграции
с графическим интерфейсом. Все параметры четко определены и могут быть
легко привязаны к элементам GUI.

PIV (Particle Image Velocimetry):
- Вычисление векторных полей скоростей между последовательными кадрами
- Использование кросс-корреляции для определения смещений
- Валидация и фильтрация векторов
- Экспорт результатов в CSV формат

Требования:
    pip install openpiv

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, Dict
import logging
import numpy as np

# Добавление пути к модулям проекта
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.piv.piv_analysis import (
    PIVAnalyzer,
    PIVProgress,
    PIVResult
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PIVParameters:
    """
    Параметры PIV анализа для GUI.

    Все параметры должны устанавливаться через GUI элементы:
    - input_folder: путь к папке intensity_filtered_XXXX (через file dialog)
    - window_size: размер окна корреляции (combobox: 16, 32, 64, 128)
    - overlap: перекрытие окон (spinbox)
    - search_area_size: размер области поиска (spinbox)
    - dt: временной интервал между кадрами (spinbox)
    - scaling_factor: масштабный коэффициент (spinbox)
    - sig2noise_threshold: порог signal-to-noise (spinbox)
    - enable_progress_callback: включить обратную связь прогресса (checkbox)
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    input_folder: str  # Путь к папке intensity_filtered_XXXX с изображениями

    # ПАРАМЕТРЫ ОКНА КОРРЕЛЯЦИИ
    window_size: int = 32  # Размер окна (16, 32, 64, 128)
    overlap: int = 16  # Перекрытие окон (обычно window_size // 2)
    search_area_size: int = 64  # Размер области поиска

    # ФИЗИЧЕСКИЕ ПАРАМЕТРЫ
    dt: float = 1.0  # Временной интервал между кадрами (мс)
    scaling_factor: float = 1.0  # Масштабный коэффициент (пиксели -> мм)

    # ПАРАМЕТРЫ ВАЛИДАЦИИ
    sig2noise_threshold: float = 1.3  # Порог signal-to-noise

    # ОПЦИОНАЛЬНЫЕ ПАРАМЕТРЫ
    enable_progress_callback: bool = True  # Включить callback для прогресса

    # GUI ПОДСКАЗКИ (не используются в обработке, только для GUI)
    window_size_options: tuple = (16, 32, 64, 128)  # Доступные размеры окон
    window_size_default: int = 32

    overlap_min: int = 0
    overlap_max: int = 64
    overlap_default: int = 16

    search_area_size_min: int = 16
    search_area_size_max: int = 256
    search_area_size_default: int = 64

    dt_min: float = 0.001
    dt_max: float = 1000.0
    dt_default: float = 1.0

    scaling_factor_min: float = 0.001
    scaling_factor_max: float = 100.0
    scaling_factor_default: float = 1.0

    sig2noise_threshold_min: float = 1.0
    sig2noise_threshold_max: float = 10.0
    sig2noise_threshold_default: float = 1.3

    def validate(self) -> tuple[bool, str]:
        """
        Валидация параметров.

        Returns:
            tuple[bool, str]: (success, error_message)
        """
        # Проверка входной папки
        input_path = Path(self.input_folder)
        if not input_path.exists():
            return False, f"Входная папка не существует: {self.input_folder}"

        cam1_path = input_path / "cam_1"
        cam2_path = input_path / "cam_2"

        if not cam1_path.exists():
            return False, f"Не найдена папка cam_1 в {self.input_folder}"
        if not cam2_path.exists():
            return False, f"Не найдена папка cam_2 в {self.input_folder}"

        # Проверка размера окна
        if self.window_size not in self.window_size_options:
            return False, f"window_size должен быть одним из {self.window_size_options}"

        # Проверка перекрытия
        if not (0 <= self.overlap < self.window_size):
            return False, f"overlap должен быть в диапазоне [0, {self.window_size})"

        # Проверка размера области поиска
        if self.search_area_size < self.window_size:
            return False, (
                f"search_area_size должен быть >= window_size: "
                f"{self.search_area_size} < {self.window_size}"
            )

        # Проверка временного интервала
        if self.dt <= 0:
            return False, f"dt должен быть > 0: {self.dt}"

        # Проверка масштабного коэффициента
        if self.scaling_factor <= 0:
            return False, f"scaling_factor должен быть > 0: {self.scaling_factor}"

        # Проверка порога signal-to-noise
        if self.sig2noise_threshold <= 0:
            return False, f"sig2noise_threshold должен быть > 0: {self.sig2noise_threshold}"

        return True, ""


class PIVExecutor:
    """
    Класс для выполнения PIV анализа с параметрами для GUI.

    Использование:
        1. Создать экземпляр PIVExecutor
        2. Задать параметры через PIVParameters
        3. (Опционально) Установить callback для прогресса
        4. Вызвать execute() для запуска PIV анализа
        5. Получить результат PIVResult
    """

    def __init__(self):
        """Инициализация исполнителя PIV анализа."""
        try:
            self.analyzer = PIVAnalyzer()
            self.parameters: Optional[PIVParameters] = None
            self._progress_callback: Optional[Callable[[PIVProgress], None]] = None

            logger.info("Инициализирован PIVExecutor")
        except ImportError as e:
            logger.error(f"Ошибка импорта PIVAnalyzer: {e}")
            raise

    def set_parameters(self, parameters: PIVParameters) -> tuple[bool, str]:
        """
        Установка параметров PIV анализа.

        Args:
            parameters: Параметры PIV анализа

        Returns:
            tuple[bool, str]: (success, error_message)
        """
        # Валидация параметров
        success, error_msg = parameters.validate()
        if not success:
            logger.error(f"Ошибка валидации параметров: {error_msg}")
            return False, error_msg

        self.parameters = parameters

        # Применение параметров к анализатору
        if not self.analyzer.set_input_folder(parameters.input_folder):
            return False, "Не удалось установить входную папку"

        if not self.analyzer.set_piv_config(
            window_size=parameters.window_size,
            overlap=parameters.overlap,
            search_area_size=parameters.search_area_size,
            dt=parameters.dt,
            scaling_factor=parameters.scaling_factor,
            sig2noise_threshold=parameters.sig2noise_threshold
        ):
            return False, "Не удалось установить параметры PIV"

        logger.info(
            f"Параметры установлены: "
            f"window_size={parameters.window_size}, "
            f"overlap={parameters.overlap}, "
            f"dt={parameters.dt}, "
            f"scaling={parameters.scaling_factor}"
        )
        return True, ""

    def set_progress_callback(self, callback: Callable[[PIVProgress], None]) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция обратного вызова для GUI
                     Принимает PIVProgress с полями:
                     - current_pair: текущая пара изображений
                     - total_pairs: общее количество пар
                     - processed_pairs: обработано пар
                     - current_camera: текущая камера (cam_1 или cam_2)
                     - percentage: процент выполнения (0-100)
                     - message: текстовое сообщение
        """
        self._progress_callback = callback

        if self.parameters and self.parameters.enable_progress_callback:
            self.analyzer.set_progress_callback(callback)
            logger.info("Установлен callback для прогресса")

    def cancel(self) -> None:
        """
        Отмена выполнения PIV анализа.

        Этот метод следует вызвать из GUI при нажатии кнопки "Отмена".
        """
        self.analyzer.cancel_processing()
        logger.info("Запрошена отмена обработки")

    def execute(self) -> PIVResult:
        """
        Выполнение PIV анализа.

        Returns:
            PIVResult с результатами анализа:
            - success: успешность выполнения
            - total_pairs_processed: общее количество обработанных пар
            - cam1_vectors_count: количество векторов cam_1
            - cam2_vectors_count: количество векторов cam_2
            - errors: список ошибок
            - warnings: список предупреждений
            - output_folder: путь к выходной папке
        """
        if self.parameters is None:
            logger.error("Параметры не установлены")
            return PIVResult(
                success=False,
                total_pairs_processed=0,
                cam1_vectors_count=0,
                cam2_vectors_count=0,
                errors=["Параметры не установлены"],
                warnings=[],
                output_folder=""
            )

        logger.info("=" * 60)
        logger.info("ЗАПУСК PIV АНАЛИЗА")
        logger.info(f"Входная папка: {self.parameters.input_folder}")
        logger.info(f"Параметры окна: size={self.parameters.window_size}, "
                   f"overlap={self.parameters.overlap}")
        logger.info(f"Физические параметры: dt={self.parameters.dt}, "
                   f"scaling={self.parameters.scaling_factor}")
        logger.info("=" * 60)

        # Выполнение анализа
        result = self.analyzer.process_all()

        logger.info("=" * 60)
        logger.info("ЗАВЕРШЕНИЕ PIV АНАЛИЗА")
        logger.info(f"Успешно: {result.success}")
        logger.info(f"Обработано пар: {result.total_pairs_processed}")
        logger.info(f"  cam_1: {result.cam1_vectors_count} векторов")
        logger.info(f"  cam_2: {result.cam2_vectors_count} векторов")
        logger.info(f"Ошибок: {len(result.errors)}")
        logger.info(f"Предупреждений: {len(result.warnings)}")
        logger.info(f"Выходная папка: {result.output_folder}")
        logger.info("=" * 60)

        return result

    def get_output_folder(self) -> Optional[str]:
        """
        Получение пути к выходной папке.

        Returns:
            Путь к выходной папке или None
        """
        if self.analyzer.output_folder:
            return str(self.analyzer.output_folder)
        return None

    def get_preview(
        self,
        camera_name: str,
        pair_index: int = 0
    ) -> Optional[Dict[str, any]]:
        """
        Получение предварительного просмотра PIV анализа.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)
            pair_index: Индекс пары для предпросмотра

        Returns:
            Словарь с результатами или None
        """
        preview = self.analyzer.get_preview(camera_name, pair_index)

        if preview is None:
            logger.warning(f"Не удалось создать предпросмотр для {camera_name}")
            return None

        logger.info(
            f"Создан предпросмотр для {camera_name}: "
            f"{preview['vectors_count']} векторов, "
            f"средняя магнитуда: {preview['mean_magnitude']:.2f}"
        )

        return preview


def run_piv_analysis(
    input_folder: str,
    window_size: int = 32,
    overlap: int = 16,
    search_area_size: int = 64,
    dt: float = 1.0,
    scaling_factor: float = 1.0,
    sig2noise_threshold: float = 1.3,
    progress_callback: Optional[Callable] = None
) -> PIVResult:
    """
    Удобная функция для запуска PIV анализа без создания объектов.

    Args:
        input_folder: Путь к папке intensity_filtered_XXXX
        window_size: Размер окна корреляции (по умолчанию 32)
        overlap: Перекрытие окон (по умолчанию 16)
        search_area_size: Размер области поиска (по умолчанию 64)
        dt: Временной интервал между кадрами (по умолчанию 1.0)
        scaling_factor: Масштабный коэффициент (по умолчанию 1.0)
        sig2noise_threshold: Порог signal-to-noise (по умолчанию 1.3)
        progress_callback: Callback функция для прогресса (опционально)

    Returns:
        PIVResult с результатами

    Example:
        >>> result = run_piv_analysis(
        ...     input_folder="path/to/intensity_filtered_3240",
        ...     window_size=32,
        ...     overlap=16
        ... )
        >>> print(f"Векторов: {result.cam1_vectors_count + result.cam2_vectors_count}")
    """
    # Создание параметров
    params = PIVParameters(
        input_folder=input_folder,
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
        dt=dt,
        scaling_factor=scaling_factor,
        sig2noise_threshold=sig2noise_threshold,
        enable_progress_callback=progress_callback is not None
    )

    # Создание исполнителя
    try:
        executor = PIVExecutor()
    except ImportError as e:
        logger.error(f"Ошибка инициализации PIVExecutor: {e}")
        return PIVResult(
            success=False,
            total_pairs_processed=0,
            cam1_vectors_count=0,
            cam2_vectors_count=0,
            errors=[str(e)],
            warnings=[],
            output_folder=""
        )

    # Установка параметров
    success, error_msg = executor.set_parameters(params)
    if not success:
        logger.error(f"Ошибка установки параметров: {error_msg}")
        return PIVResult(
            success=False,
            total_pairs_processed=0,
            cam1_vectors_count=0,
            cam2_vectors_count=0,
            errors=[error_msg],
            warnings=[],
            output_folder=""
        )

    # Установка callback
    if progress_callback:
        executor.set_progress_callback(progress_callback)

    # Выполнение
    return executor.execute()


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ ДЛЯ GUI
# ============================================================================

def example_gui_usage():
    """
    Пример использования модуля с GUI.

    Этот пример показывает, как интегрировать модуль с графическим интерфейсом.
    """
    print("\n" + "=" * 60)
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ С GUI - PIV АНАЛИЗ")
    print("=" * 60)

    # === ШАГ 1: Задание параметров (из GUI элементов) ===
    parameters = PIVParameters(
        input_folder=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\intensity_filtered_3240",
        window_size=32,
        overlap=16,
        search_area_size=64,
        dt=1.0,
        scaling_factor=1.0,
        sig2noise_threshold=1.3,
        enable_progress_callback=True
    )

    print(f"\nПараметры:")
    print(f"  Входная папка: {parameters.input_folder}")
    print(f"  Размер окна: {parameters.window_size} пикс.")
    print(f"  Перекрытие: {parameters.overlap} пикс.")
    print(f"  Размер области поиска: {parameters.search_area_size} пикс.")
    print(f"  Временной интервал: {parameters.dt} мс")
    print(f"  Масштабный коэффициент: {parameters.scaling_factor}")
    print(f"  Порог signal-to-noise: {parameters.sig2noise_threshold}")

    # === ШАГ 2: Создание исполнителя ===
    try:
        executor = PIVExecutor()
    except ImportError as e:
        print(f"\nОШИБКА: {e}")
        print("\nУстановите OpenPIV: pip install openpiv")
        return

    # === ШАГ 3: Валидация и установка параметров ===
    success, error_msg = executor.set_parameters(parameters)
    if not success:
        print(f"\nОШИБКА: {error_msg}")
        return

    print("\n✓ Параметры валидны")

    # === ШАГ 4: Предварительный просмотр ===
    print("\nПредварительный просмотр PIV анализа...")
    preview = executor.get_preview("cam_1", pair_index=0)
    if preview:
        print(f"\nРезультаты предпросмотра:")
        print(f"  Пара: {preview['pair_names'][0]} - {preview['pair_names'][1]}")
        print(f"  Векторов: {preview['vectors_count']}")
        print(f"  Средняя магнитуда: {preview['mean_magnitude']:.3f}")
        print(f"  Макс. магнитуда: {preview['max_magnitude']:.3f}")

    # === ШАГ 5: Установка callback для прогресса (для GUI progress bar) ===
    def progress_callback(progress: PIVProgress):
        """Callback для обновления GUI."""
        print(
            f"  [{progress.current_camera}] "
            f"{progress.percentage:.1f}% - {progress.message}"
        )

    executor.set_progress_callback(progress_callback)

    # === ШАГ 6: Выполнение PIV анализа ===
    print("\nЗапуск PIV анализа...")
    result = executor.execute()

    # === ШАГ 7: Обработка результата ===
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Успешно: {result.success}")
    print(f"Обработано пар: {result.total_pairs_processed}")
    print(f"  cam_1: {result.cam1_vectors_count} векторов")
    print(f"  cam_2: {result.cam2_vectors_count} векторов")
    print(f"Ошибок: {len(result.errors)}")
    print(f"Предупреждений: {len(result.warnings)}")
    print(f"Выходная папка: {result.output_folder}")

    if result.errors:
        print("\nОшибки:")
        for error in result.errors[:5]:  # Показываем первые 5 ошибок
            print(f"  - {error}")

    if result.warnings:
        print("\nПредупреждения:")
        for warning in result.warnings[:5]:  # Показываем первые 5 предупреждений
            print(f"  - {warning}")

    print("=" * 60)


if __name__ == "__main__":
    # При запуске модуля напрямую - показать пример использования
    example_gui_usage()
