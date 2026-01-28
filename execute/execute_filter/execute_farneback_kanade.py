"""
Модуль выполнения Farneback/Kanade фильтрации для GUI ParticleAnalysis.

Этот модуль предоставляет готовую к использованию структуру для интеграции
с графическим интерфейсом. Подготавливает пары изображений для алгоритмов
оптического потока (Farneback, Lucas-Kanade).

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import logging
import numpy as np

# Добавление пути к модулям проекта
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.filters.filter_farneback_kanade import (
    FarnebackKanadeFilter,
    FilterProgress,
    FilterResult,
    PairStatistics
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FarnebackKanadeParameters:
    """
    Параметры Farneback/Kanade фильтрации для GUI.

    Все параметры должны устанавливаться через GUI элементы:
    - input_folder: путь к папке cam_sorted (через file dialog)
    - threshold: пороговое значение (через slider или spinbox)
    - output_min/output_max: диапазон нормализации (через spinbox)
    - intensity_boost: коэффициент усиления (через slider)
    - enable_progress_callback: включить обратную связь прогресса (checkbox)
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    input_folder: str  # Путь к папке cam_sorted с подпапками cam_1 и cam_2

    # ПАРАМЕТРЫ ФИЛЬТРАЦИИ
    threshold: int = 2000  # Пороговое значение фильтрации (0-65535)
    output_min: int = 50  # Минимальное значение нормализации (для ненулевых пикселей)
    output_max: int = 255  # Максимальное значение нормализации
    intensity_boost: float = 1.5  # Коэффициент усиления интенсивности

    # ОПЦИОНАЛЬНЫЕ ПАРАМЕТРЫ
    enable_progress_callback: bool = True  # Включить callback для прогресса

    # GUI ПОДСКАЗКИ (не используются в обработке, только для GUI)
    threshold_min: int = 0  # Минимальное значение для slider
    threshold_max: int = 65535  # Максимальное значение для slider
    threshold_default: int = 2000  # Значение по умолчанию
    threshold_step: int = 100  # Шаг изменения в slider

    output_range_min: int = 0  # Минимальное значение диапазона
    output_range_max: int = 255  # Максимальное значение диапазона

    intensity_boost_min: float = 0.5  # Минимальный коэффициент усиления
    intensity_boost_max: float = 3.0  # Максимальный коэффициент усиления
    intensity_boost_default: float = 1.5  # Значение по умолчанию
    intensity_boost_step: float = 0.1  # Шаг изменения

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

        # Проверка порога
        if not (self.threshold_min <= self.threshold <= self.threshold_max):
            return False, f"Порог должен быть в диапазоне [{self.threshold_min}, {self.threshold_max}]"

        # Проверка диапазона нормализации
        if not (0 <= self.output_min < self.output_max <= 255):
            return False, f"Некорректный диапазон нормализации: [{self.output_min}, {self.output_max}]"

        # Проверка коэффициента усиления
        if self.intensity_boost <= 0:
            return False, f"Коэффициент усиления должен быть положительным: {self.intensity_boost}"

        return True, ""


class FarnebackKanadeExecutor:
    """
    Класс для выполнения Farneback/Kanade фильтрации с параметрами для GUI.

    Использование:
        1. Создать экземпляр FarnebackKanadeExecutor
        2. Задать параметры через FarnebackKanadeParameters
        3. (Опционально) Установить callback для прогресса
        4. Вызвать execute() для запуска обработки
        5. Получить результат FilterResult
    """

    def __init__(self):
        """Инициализация исполнителя Farneback/Kanade фильтрации."""
        self.filter = FarnebackKanadeFilter()
        self.parameters: Optional[FarnebackKanadeParameters] = None
        self._progress_callback: Optional[Callable[[FilterProgress], None]] = None

        logger.info("Инициализирован FarnebackKanadeExecutor")

    def set_parameters(self, parameters: FarnebackKanadeParameters) -> tuple[bool, str]:
        """
        Установка параметров фильтрации.

        Args:
            parameters: Параметры фильтрации

        Returns:
            tuple[bool, str]: (success, error_message)
        """
        # Валидация параметров
        success, error_msg = parameters.validate()
        if not success:
            logger.error(f"Ошибка валидации параметров: {error_msg}")
            return False, error_msg

        self.parameters = parameters

        # Применение параметров к фильтру
        if not self.filter.set_input_folder(parameters.input_folder):
            return False, "Не удалось установить входную папку"

        if not self.filter.set_threshold(parameters.threshold):
            return False, "Не удалось установить пороговое значение"

        if not self.filter.set_output_range(parameters.output_min, parameters.output_max):
            return False, "Не удалось установить диапазон нормализации"

        if not self.filter.set_intensity_boost(parameters.intensity_boost):
            return False, "Не удалось установить коэффициент усиления"

        logger.info(f"Параметры установлены: threshold={parameters.threshold}, "
                    f"output_range=[{parameters.output_min}, {parameters.output_max}], "
                    f"intensity_boost={parameters.intensity_boost}")
        return True, ""

    def set_progress_callback(self, callback: Callable[[FilterProgress], None]) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция обратного вызова для GUI
                     Принимает FilterProgress с полями:
                     - current_file: имя текущего файла
                     - total_files: общее количество пар
                     - processed_files: обработано пар
                     - current_camera: текущая камера (cam_1 или cam_2)
                     - percentage: процент выполнения (0-100)
                     - message: текстовое сообщение
        """
        self._progress_callback = callback

        if self.parameters and self.parameters.enable_progress_callback:
            self.filter.set_progress_callback(callback)
            logger.info("Установлен callback для прогресса")

    def cancel(self) -> None:
        """
        Отмена выполнения обработки.

        Этот метод следует вызвать из GUI при нажатии кнопки "Отмена".
        """
        self.filter.cancel_processing()
        logger.info("Запрошена отмена обработки")

    def execute(self) -> FilterResult:
        """
        Выполнение Farneback/Kanade фильтрации.

        Returns:
            FilterResult с результатами обработки:
            - success: успешность выполнения
            - total_processed: общее количество обработанных пар
            - cam1_processed: обработано пар cam_1
            - cam2_processed: обработано пар cam_2
            - errors: список ошибок
            - output_folder: путь к выходной папке
            - threshold: использованный порог
            - output_range: использованный диапазон нормализации
        """
        if self.parameters is None:
            logger.error("Параметры не установлены")
            return FilterResult(
                success=False,
                total_processed=0,
                cam1_processed=0,
                cam2_processed=0,
                errors=["Параметры не установлены"],
                output_folder="",
                threshold=0,
                output_range=(0, 0)
            )

        logger.info("=" * 60)
        logger.info("ЗАПУСК FARNEBACK/KANADE ФИЛЬТРАЦИИ")
        logger.info(f"Входная папка: {self.parameters.input_folder}")
        logger.info(f"Порог: {self.parameters.threshold}")
        logger.info(f"Диапазон нормализации: [{self.parameters.output_min}, {self.parameters.output_max}]")
        logger.info(f"Коэффициент усиления: {self.parameters.intensity_boost}")
        logger.info("=" * 60)

        # Выполнение обработки
        result = self.filter.process_all()

        logger.info("=" * 60)
        logger.info("ЗАВЕРШЕНИЕ FARNEBACK/KANADE ФИЛЬТРАЦИИ")
        logger.info(f"Успешно: {result.success}")
        logger.info(f"Обработано пар: {result.total_processed}")
        logger.info(f"cam_1: {result.cam1_processed}")
        logger.info(f"cam_2: {result.cam2_processed}")
        logger.info(f"Ошибок: {len(result.errors)}")
        logger.info(f"Выходная папка: {result.output_folder}")
        logger.info("=" * 60)

        return result

    def get_output_folder(self) -> Optional[str]:
        """
        Получение пути к выходной папке.

        Returns:
            Путь к выходной папке или None
        """
        if self.filter.output_folder:
            return str(self.filter.output_folder)
        return None

    def get_preview(self, camera_name: str, pair_index: int = 0) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]]:
        """
        Получение предварительного просмотра фильтрации пары изображений.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)
            pair_index: Индекс пары для предпросмотра

        Returns:
            Tuple[original_a, original_b, filtered_a, filtered_b, stats] или None
            - original_a: оригинальное изображение _a (16-bit)
            - original_b: оригинальное изображение _b (16-bit)
            - filtered_a: отфильтрованное изображение _a (8-bit)
            - filtered_b: отфильтрованное изображение _b (8-bit)
            - stats: статистика обработки пары
        """
        if self.filter.input_folder is None:
            logger.error("Входная папка не установлена")
            return None

        camera_path = self.filter.input_folder / camera_name
        if not camera_path.exists():
            logger.error(f"Папка {camera_name} не найдена")
            return None

        # Поиск пар изображений
        a_files = sorted(camera_path.glob("*_a.png"))
        if not a_files or pair_index >= len(a_files):
            logger.error(f"Пара с индексом {pair_index} не найдена в {camera_name}")
            return None

        img_a_path = a_files[pair_index]
        img_b_path = camera_path / img_a_path.name.replace("_a.png", "_b.png")

        if not img_b_path.exists():
            logger.error(f"Не найден парный файл для {img_a_path.name}")
            return None

        result = self.filter.preview_pair(img_a_path, img_b_path)

        if result is None:
            return None

        original_a, original_b, filtered_a, filtered_b, statistics = result

        stats_dict = {
            'mean_a_original': statistics.mean_a_original,
            'mean_b_original': statistics.mean_b_original,
            'mean_a_after_threshold': statistics.mean_a_after_threshold,
            'mean_b_after_threshold': statistics.mean_b_after_threshold,
            'scale_factor': statistics.scale_factor,
            'global_max': statistics.global_max,
            'pixels_zeroed_a_percent': statistics.pixels_zeroed_a_percent,
            'pixels_zeroed_b_percent': statistics.pixels_zeroed_b_percent
        }

        logger.info(f"Создан предпросмотр для {camera_name}/{img_a_path.stem}")
        return original_a, original_b, filtered_a, filtered_b, stats_dict

    def get_pair_count(self, camera_name: str) -> int:
        """
        Получение количества пар изображений в камере.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Количество пар или 0
        """
        if self.filter.input_folder is None:
            return 0

        camera_path = self.filter.input_folder / camera_name
        if not camera_path.exists():
            return 0

        a_files = list(camera_path.glob("*_a.png"))
        return len(a_files)


def run_farneback_kanade(input_folder: str,
                         threshold: int = 2000,
                         output_min: int = 50,
                         output_max: int = 255,
                         intensity_boost: float = 1.5,
                         progress_callback: Optional[Callable] = None) -> FilterResult:
    """
    Удобная функция для запуска Farneback/Kanade фильтрации без создания объектов.

    Args:
        input_folder: Путь к папке cam_sorted
        threshold: Пороговое значение (по умолчанию 2000)
        output_min: Минимальное значение нормализации (по умолчанию 50)
        output_max: Максимальное значение нормализации (по умолчанию 255)
        intensity_boost: Коэффициент усиления (по умолчанию 1.5)
        progress_callback: Callback функция для прогресса (опционально)

    Returns:
        FilterResult с результатами

    Example:
        >>> result = run_farneback_kanade(
        ...     input_folder="path/to/cam_sorted",
        ...     threshold=2500,
        ...     intensity_boost=1.8
        ... )
        >>> print(f"Обработано пар: {result.total_processed}")
    """
    # Создание параметров
    params = FarnebackKanadeParameters(
        input_folder=input_folder,
        threshold=threshold,
        output_min=output_min,
        output_max=output_max,
        intensity_boost=intensity_boost,
        enable_progress_callback=progress_callback is not None
    )

    # Создание исполнителя
    executor = FarnebackKanadeExecutor()

    # Установка параметров
    success, error_msg = executor.set_parameters(params)
    if not success:
        logger.error(f"Ошибка установки параметров: {error_msg}")
        return FilterResult(
            success=False,
            total_processed=0,
            cam1_processed=0,
            cam2_processed=0,
            errors=[error_msg],
            output_folder="",
            threshold=threshold,
            output_range=(output_min, output_max)
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
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ С GUI")
    print("=" * 60)

    # === ШАГ 1: Задание параметров (из GUI элементов) ===
    parameters = FarnebackKanadeParameters(
        input_folder=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted",
        threshold=2000,
        output_min=50,
        output_max=255,
        intensity_boost=1.5,
        enable_progress_callback=True
    )

    print(f"\nПараметры:")
    print(f"  Входная папка: {parameters.input_folder}")
    print(f"  Порог: {parameters.threshold}")
    print(f"  Диапазон нормализации: [{parameters.output_min}, {parameters.output_max}]")
    print(f"  Коэффициент усиления: {parameters.intensity_boost}")

    # === ШАГ 2: Создание исполнителя ===
    executor = FarnebackKanadeExecutor()

    # === ШАГ 3: Валидация и установка параметров ===
    success, error_msg = executor.set_parameters(parameters)
    if not success:
        print(f"\nОШИБКА: {error_msg}")
        return

    print("\n[OK] Параметры валидны")

    # === ШАГ 4: Получение количества пар для обработки ===
    cam1_pairs = executor.get_pair_count("cam_1")
    cam2_pairs = executor.get_pair_count("cam_2")
    print(f"\nКоличество пар:")
    print(f"  cam_1: {cam1_pairs}")
    print(f"  cam_2: {cam2_pairs}")

    # === ШАГ 5: Предварительный просмотр фильтрации ===
    print("\nПредварительный просмотр фильтрации...")
    preview_result = executor.get_preview("cam_1", pair_index=0)
    if preview_result:
        original_a, original_b, filtered_a, filtered_b, stats = preview_result
        print(f"\nРезультаты предпросмотра:")
        print(f"  Средняя интенсивность _a (оригинал): {stats['mean_a_original']:.1f}")
        print(f"  Средняя интенсивность _b (оригинал): {stats['mean_b_original']:.1f}")
        print(f"  Средняя интенсивность _a (после порога): {stats['mean_a_after_threshold']:.1f}")
        print(f"  Средняя интенсивность _b (после порога): {stats['mean_b_after_threshold']:.1f}")
        print(f"  Коэффициент масштабирования: {stats['scale_factor']:.3f}")
        print(f"  Глобальный максимум: {stats['global_max']:.1f}")
        print(f"  Обнулено пикселей _a: {stats['pixels_zeroed_a_percent']:.2f}%")
        print(f"  Обнулено пикселей _b: {stats['pixels_zeroed_b_percent']:.2f}%")

    # === ШАГ 6: Установка callback для прогресса (для GUI progress bar) ===
    def progress_callback(progress: FilterProgress):
        """Callback для обновления GUI."""
        print(f"  [{progress.current_camera}] {progress.percentage:.1f}% - {progress.message}")

    executor.set_progress_callback(progress_callback)

    # === ШАГ 7: Выполнение фильтрации ===
    print("\nЗапуск Farneback/Kanade фильтрации...")
    result = executor.execute()

    # === ШАГ 8: Обработка результата ===
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Успешно: {result.success}")
    print(f"Обработано пар: {result.total_processed}")
    print(f"  cam_1: {result.cam1_processed}")
    print(f"  cam_2: {result.cam2_processed}")
    print(f"Ошибок: {len(result.errors)}")
    print(f"Выходная папка: {result.output_folder}")

    if result.errors:
        print("\nОшибки:")
        for error in result.errors[:5]:  # Показываем первые 5 ошибок
            print(f"  - {error}")

    print("=" * 60)


if __name__ == "__main__":
    # При запуске модуля напрямую - показать пример использования
    example_gui_usage()
