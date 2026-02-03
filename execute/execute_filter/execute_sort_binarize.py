"""
Модуль выполнения комбинированной сортировки и бинаризации для GUI ParticleAnalysis.

Объединяет сортировку по камерам и бинаризацию в один проход,
минуя промежуточные 16-bit копии и экономя дисковое пространство.

Алгоритм:
- Первые 2 фото → cam_1 (с отражением по горизонтали)
- Следующие 2 фото → cam_2 (без изменений)
- Бинаризация: пиксели >= порога → 255, иначе → 0
- Результат: 8-bit PNG в {input}_cam_sorted/binary_filter_{threshold}/
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
import logging

# Добавление пути к модулям проекта
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.sort_and_binarize import (
    SortAndBinarize,
    SortBinarizeProgress,
    SortBinarizeResult
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SortBinarizeParameters:
    """
    Параметры комбинированной сортировки и бинаризации для GUI.

    Все параметры должны устанавливаться через GUI элементы:
    - input_folder: путь к папке с сырыми PNG изображениями (через file dialog)
    - threshold: пороговое значение бинаризации (через slider или spinbox)
    - validate_format: валидировать ли формат изображений (checkbox)
    - enable_progress_callback: включить обратную связь прогресса (checkbox)
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    input_folder: str  # Путь к папке с сырыми 16-bit PNG изображениями

    # ОПЦИОНАЛЬНЫЕ ПАРАМЕТРЫ
    threshold: int = 10000  # Пороговое значение бинаризации (0-65535)
    validate_format: bool = True  # Валидировать формат изображений (16-bit PNG)
    enable_progress_callback: bool = True  # Включить callback для прогресса

    # GUI ПОДСКАЗКИ (не используются в обработке, только для GUI)
    threshold_min: int = 0  # Минимальное значение для slider
    threshold_max: int = 65535  # Максимальное значение для slider
    threshold_default: int = 10000  # Значение по умолчанию
    threshold_step: int = 100  # Шаг изменения в slider

    def validate(self) -> tuple[bool, str]:
        """
        Валидация параметров.

        Returns:
            tuple[bool, str]: (success, error_message)
        """
        input_path = Path(self.input_folder)
        if not input_path.exists():
            return False, f"Входная папка не существует: {self.input_folder}"

        if not input_path.is_dir():
            return False, f"Указанный путь не является папкой: {self.input_folder}"

        # Проверка наличия PNG файлов
        png_files = list(input_path.glob("*.png"))
        if not png_files:
            return False, f"В папке {self.input_folder} нет PNG файлов"

        # Проверка количества файлов (должно быть кратно 4)
        if len(png_files) % 4 != 0:
            return False, (
                f"Количество изображений ({len(png_files)}) не кратно 4. "
                f"Алгоритм работает с циклом из 4 изображений."
            )

        # Проверка порога
        if not (self.threshold_min <= self.threshold <= self.threshold_max):
            return False, f"Порог должен быть в диапазоне [{self.threshold_min}, {self.threshold_max}]"

        return True, ""


class SortBinarizeExecutor:
    """
    Класс для выполнения комбинированной сортировки и бинаризации с параметрами для GUI.

    Использование:
        1. Создать экземпляр SortBinarizeExecutor
        2. Задать параметры через SortBinarizeParameters
        3. (Опционально) Установить callback для прогресса
        4. Вызвать execute() для запуска обработки
        5. Получить результат SortBinarizeResult
    """

    def __init__(self):
        """Инициализация исполнителя."""
        self.processor: Optional[SortAndBinarize] = None
        self.parameters: Optional[SortBinarizeParameters] = None
        self._progress_callback: Optional[Callable[[SortBinarizeProgress], None]] = None

        logger.info("Инициализирован SortBinarizeExecutor")

    def set_parameters(self, parameters: SortBinarizeParameters) -> tuple[bool, str]:
        """
        Установка параметров обработки.

        Args:
            parameters: Параметры сортировки и бинаризации

        Returns:
            tuple[bool, str]: (success, error_message)
        """
        # Валидация параметров
        success, error_msg = parameters.validate()
        if not success:
            logger.error(f"Ошибка валидации параметров: {error_msg}")
            return False, error_msg

        self.parameters = parameters

        # Создание процессора
        try:
            self.processor = SortAndBinarize(
                input_folder=parameters.input_folder,
                threshold=parameters.threshold,
                validate_format=parameters.validate_format
            )
            logger.info(
                f"Параметры установлены: input_folder={parameters.input_folder}, "
                f"threshold={parameters.threshold}"
            )
            return True, ""
        except Exception as e:
            error_msg = f"Ошибка создания процессора: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def set_progress_callback(self, callback: Callable[[SortBinarizeProgress], None]) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция обратного вызова для GUI
                     Принимает SortBinarizeProgress с полями:
                     - current_file: имя текущего файла
                     - total_files: общее количество файлов
                     - processed_files: обработано файлов
                     - current_camera: текущая камера (cam_1 или cam_2)
                     - percentage: процент выполнения (0-100)
                     - message: текстовое сообщение
        """
        self._progress_callback = callback

        if self.processor and self.parameters and self.parameters.enable_progress_callback:
            self.processor.set_progress_callback(callback)
            logger.info("Установлен callback для прогресса")

    def cancel(self) -> None:
        """
        Отмена выполнения обработки.

        Этот метод следует вызвать из GUI при нажатии кнопки "Отмена".
        """
        if self.processor:
            self.processor.cancel_processing()
        logger.info("Запрошена отмена обработки")

    def execute(self) -> SortBinarizeResult:
        """
        Выполнение комбинированной сортировки и бинаризации.

        Returns:
            SortBinarizeResult с результатами обработки
        """
        if self.parameters is None or self.processor is None:
            logger.error("Параметры не установлены")
            return SortBinarizeResult(
                success=False,
                cam1_count=0,
                cam2_count=0,
                total_processed=0,
                output_folder="",
                threshold=0,
                errors=["Параметры не установлены"]
            )

        logger.info("=" * 60)
        logger.info("ЗАПУСК СОРТИРОВКИ И БИНАРИЗАЦИИ")
        logger.info(f"Входная папка: {self.parameters.input_folder}")
        logger.info(f"Порог: {self.parameters.threshold}")
        logger.info(f"Валидация формата: {'Да' if self.parameters.validate_format else 'Нет'}")
        logger.info("=" * 60)

        # Установка callback если ещё не установлен
        if self._progress_callback and self.parameters.enable_progress_callback:
            self.processor.set_progress_callback(self._progress_callback)

        # Выполнение обработки
        result = self.processor.process()

        logger.info("=" * 60)
        logger.info("ЗАВЕРШЕНИЕ СОРТИРОВКИ И БИНАРИЗАЦИИ")
        logger.info(f"Успешно: {result.success}")
        logger.info(f"cam_1: {result.cam1_count} файлов")
        logger.info(f"cam_2: {result.cam2_count} файлов")
        logger.info(f"Всего: {result.total_processed}")
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
        if self.processor:
            return str(self.processor.output_folder)
        return None

    def get_input_statistics(self) -> Optional[dict]:
        """
        Получение статистики входной папки до обработки.

        Returns:
            Словарь со статистикой или None
        """
        if self.parameters is None:
            return None

        input_path = Path(self.parameters.input_folder)
        png_files = list(input_path.glob("*.png"))

        return {
            'total_files': len(png_files),
            'expected_cam1_files': len(png_files) // 2,
            'expected_cam2_files': len(png_files) // 2,
            'expected_cam1_pairs': len(png_files) // 4,
            'expected_cam2_pairs': len(png_files) // 4,
            'input_folder': str(input_path),
            'output_folder': f"{input_path}_cam_sorted/binary_filter_{self.parameters.threshold}",
            'threshold': self.parameters.threshold
        }


def run_sort_and_binarize(input_folder: str, threshold: int = 10000,
                          validate_format: bool = True,
                          progress_callback: Optional[Callable] = None) -> SortBinarizeResult:
    """
    Удобная функция для запуска сортировки и бинаризации без создания объектов.

    Args:
        input_folder: Путь к папке с сырыми 16-bit PNG изображениями
        threshold: Пороговое значение бинаризации (по умолчанию 10000)
        validate_format: Валидировать ли формат изображений (по умолчанию True)
        progress_callback: Callback функция для прогресса (опционально)

    Returns:
        SortBinarizeResult с результатами

    Example:
        >>> result = run_sort_and_binarize(
        ...     input_folder="path/to/raw_images",
        ...     threshold=2000
        ... )
        >>> print(f"Обработано: {result.total_processed}")
    """
    params = SortBinarizeParameters(
        input_folder=input_folder,
        threshold=threshold,
        validate_format=validate_format,
        enable_progress_callback=progress_callback is not None
    )

    executor = SortBinarizeExecutor()

    success, error_msg = executor.set_parameters(params)
    if not success:
        logger.error(f"Ошибка установки параметров: {error_msg}")
        return SortBinarizeResult(
            success=False,
            cam1_count=0,
            cam2_count=0,
            total_processed=0,
            output_folder="",
            threshold=threshold,
            errors=[error_msg]
        )

    if progress_callback:
        executor.set_progress_callback(progress_callback)

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
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ С GUI - СОРТИРОВКА + БИНАРИЗАЦИЯ")
    print("=" * 60)

    # === ШАГ 1: Задание параметров (из GUI элементов) ===
    parameters = SortBinarizeParameters(
        input_folder=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit",
        threshold=2_000,
        validate_format=True,
        enable_progress_callback=True
    )

    print(f"\nПараметры:")
    print(f"  Входная папка: {parameters.input_folder}")
    print(f"  Порог бинаризации: {parameters.threshold}")
    print(f"  Валидация формата: {'Да' if parameters.validate_format else 'Нет'}")

    # === ШАГ 2: Создание исполнителя ===
    executor = SortBinarizeExecutor()

    # === ШАГ 3: Валидация и установка параметров ===
    success, error_msg = executor.set_parameters(parameters)
    if not success:
        print(f"\nОШИБКА: {error_msg}")
        return

    print("\nПараметры валидны")

    # === ШАГ 4: Получение статистики до обработки ===
    input_stats = executor.get_input_statistics()
    if input_stats:
        print(f"\nСтатистика входной папки:")
        print(f"  Всего файлов: {input_stats['total_files']}")
        print(f"  Ожидаемо пар в cam_1: {input_stats['expected_cam1_pairs']}")
        print(f"  Ожидаемо пар в cam_2: {input_stats['expected_cam2_pairs']}")
        print(f"  Выходная папка: {input_stats['output_folder']}")

    # === ШАГ 5: Установка callback для прогресса (для GUI progress bar) ===
    def progress_callback(progress: SortBinarizeProgress):
        """Callback для обновления GUI."""
        print(f"  [{progress.current_camera}] {progress.percentage:.1f}% - {progress.message}")

    executor.set_progress_callback(progress_callback)

    # === ШАГ 6: Выполнение обработки ===
    print("\nЗапуск сортировки и бинаризации...")
    result = executor.execute()

    # === ШАГ 7: Обработка результата ===
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Успешно: {result.success}")
    print(f"cam_1: {result.cam1_count} файлов ({result.cam1_count // 2} пар)")
    print(f"cam_2: {result.cam2_count} файлов ({result.cam2_count // 2} пар)")
    print(f"Всего: {result.total_processed}")
    print(f"Порог: {result.threshold}")
    print(f"Выходная папка: {result.output_folder}")

    if result.errors:
        print(f"\nОшибки:")
        for error in result.errors[:5]:
            print(f"  - {error}")

    print("=" * 60)


if __name__ == "__main__":
    example_gui_usage()
