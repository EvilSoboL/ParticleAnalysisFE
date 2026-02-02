"""
Модуль выполнения бинарной фильтрации для GUI ParticleAnalysis.

Этот модуль предоставляет готовую к использованию структуру для интеграции
с графическим интерфейсом. Все параметры четко определены и могут быть
легко привязаны к элементам GUI.

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
import logging

# Добавление пути к модулям проекта
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.filters.binarization_filter import (
    BinarizationFilter,
    BinarizationProgress,
    BinarizationResult
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BinarizationParameters:
    """
    Параметры бинарной фильтрации для GUI.

    Все параметры должны устанавливаться через GUI элементы:
    - input_folder: путь к папке cam_sorted (через file dialog)
    - threshold: пороговое значение (через slider или spinbox)
    - enable_progress_callback: включить обратную связь прогресса (checkbox)
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    input_folder: str  # Путь к папке cam_sorted с подпапками cam_1 и cam_2
    threshold: int = 10000  # Пороговое значение бинаризации (0-65535)

    # ОПЦИОНАЛЬНЫЕ ПАРАМЕТРЫ
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

        return True, ""


class BinarizationExecutor:
    """
    Класс для выполнения бинарной фильтрации с параметрами для GUI.

    Использование:
        1. Создать экземпляр BinarizationExecutor
        2. Задать параметры через BinarizationParameters
        3. (Опционально) Установить callback для прогресса
        4. Вызвать execute() для запуска обработки
        5. Получить результат BinarizationResult
    """

    def __init__(self):
        """Инициализация исполнителя бинаризации."""
        self.filter = BinarizationFilter()
        self.parameters: Optional[BinarizationParameters] = None
        self._progress_callback: Optional[Callable[[BinarizationProgress], None]] = None

        logger.info("Инициализирован BinarizationExecutor")

    def set_parameters(self, parameters: BinarizationParameters) -> tuple[bool, str]:
        """
        Установка параметров бинаризации.

        Args:
            parameters: Параметры бинаризации

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

        logger.info(f"Параметры установлены: threshold={parameters.threshold}")
        return True, ""

    def set_progress_callback(self, callback: Callable[[BinarizationProgress], None]) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция обратного вызова для GUI
                     Принимает BinarizationProgress с полями:
                     - current_file: имя текущего файла
                     - total_files: общее количество файлов
                     - processed_files: обработано файлов
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

    def execute(self) -> BinarizationResult:
        """
        Выполнение бинарной фильтрации.

        Returns:
            BinarizationResult с результатами обработки:
            - success: успешность выполнения
            - total_processed: общее количество обработанных файлов
            - cam1_processed: обработано файлов cam_1
            - cam2_processed: обработано файлов cam_2
            - errors: список ошибок
            - output_folder: путь к выходной папке
            - threshold: использованный порог
        """
        if self.parameters is None:
            logger.error("Параметры не установлены")
            return BinarizationResult(
                success=False,
                total_processed=0,
                cam1_processed=0,
                cam2_processed=0,
                errors=["Параметры не установлены"],
                output_folder="",
                threshold=0
            )

        logger.info("=" * 60)
        logger.info("ЗАПУСК БИНАРНОЙ ФИЛЬТРАЦИИ")
        logger.info(f"Входная папка: {self.parameters.input_folder}")
        logger.info(f"Порог: {self.parameters.threshold}")
        logger.info("=" * 60)

        # Выполнение обработки
        result = self.filter.process_all()

        logger.info("=" * 60)
        logger.info("ЗАВЕРШЕНИЕ БИНАРНОЙ ФИЛЬТРАЦИИ")
        logger.info(f"Успешно: {result.success}")
        logger.info(f"Обработано файлов: {result.total_processed}")
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

    def get_image_statistics(self, camera_name: str, sample_size: int = 5) -> Optional[dict]:
        """
        Получение статистики изображений для помощи в выборе порога.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)
            sample_size: Количество изображений для анализа

        Returns:
            Словарь со статистикой или None
        """
        stats = self.filter.get_camera_statistics(camera_name, sample_size)

        if stats is None:
            return None

        return {
            'min_value': stats.min_value,
            'max_value': stats.max_value,
            'mean_value': stats.mean_value,
            'median_value': stats.median_value,
            'std_value': stats.std_value,
            'non_zero_pixels': stats.non_zero_pixels,
            'total_pixels': stats.total_pixels,
            'recommended_threshold_low': int(stats.mean_value - stats.std_value),
            'recommended_threshold_high': int(stats.mean_value + stats.std_value),
            'recommended_threshold_median': int(stats.median_value)
        }


def run_binarization(input_folder: str, threshold: int = 10000,
                    progress_callback: Optional[Callable] = None) -> BinarizationResult:
    """
    Удобная функция для запуска бинаризации без создания объектов.

    Args:
        input_folder: Путь к папке cam_sorted
        threshold: Пороговое значение (по умолчанию 10000)
        progress_callback: Callback функция для прогресса (опционально)

    Returns:
        BinarizationResult с результатами

    Example:
        >>> result = run_binarization(
        ...     input_folder="path/to/cam_sorted",
        ...     threshold=12000
        ... )
        >>> print(f"Обработано: {result.total_processed}")
    """
    # Создание параметров
    params = BinarizationParameters(
        input_folder=input_folder,
        threshold=threshold,
        enable_progress_callback=progress_callback is not None
    )

    # Создание исполнителя
    executor = BinarizationExecutor()

    # Установка параметров
    success, error_msg = executor.set_parameters(params)
    if not success:
        logger.error(f"Ошибка установки параметров: {error_msg}")
        return BinarizationResult(
            success=False,
            total_processed=0,
            cam1_processed=0,
            cam2_processed=0,
            errors=[error_msg],
            output_folder="",
            threshold=threshold
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
    parameters = BinarizationParameters(
        input_folder=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\without_filter",
        threshold=2_000,
        enable_progress_callback=True
    )

    print(f"\nПараметры:")
    print(f"  Входная папка: {parameters.input_folder}")
    print(f"  Порог: {parameters.threshold}")

    # === ШАГ 2: Создание исполнителя ===
    executor = BinarizationExecutor()

    # === ШАГ 3: Валидация и установка параметров ===
    success, error_msg = executor.set_parameters(parameters)
    if not success:
        print(f"\nОШИБКА: {error_msg}")
        return

    print("\n✓ Параметры валидны")

    # === ШАГ 4: Получение статистики для рекомендации порога ===
    print("\nАнализ статистики для рекомендации порога...")
    cam1_stats = executor.get_image_statistics("cam_1", sample_size=3)
    if cam1_stats:
        print(f"\nСтатистика cam_1:")
        print(f"  Минимум: {cam1_stats['min_value']}")
        print(f"  Максимум: {cam1_stats['max_value']}")
        print(f"  Среднее: {cam1_stats['mean_value']:.1f}")
        print(f"  Медиана: {cam1_stats['median_value']:.1f}")
        print(f"  Рекомендуемый порог (низкий): {cam1_stats['recommended_threshold_low']}")
        print(f"  Рекомендуемый порог (медиана): {cam1_stats['recommended_threshold_median']}")
        print(f"  Рекомендуемый порог (высокий): {cam1_stats['recommended_threshold_high']}")

    # === ШАГ 5: Установка callback для прогресса (для GUI progress bar) ===
    def progress_callback(progress: BinarizationProgress):
        """Callback для обновления GUI."""
        print(f"  [{progress.current_camera}] {progress.percentage:.1f}% - {progress.message}")

    executor.set_progress_callback(progress_callback)

    # === ШАГ 6: Выполнение бинаризации ===
    print("\nЗапуск бинаризации...")
    result = executor.execute()

    # === ШАГ 7: Обработка результата ===
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Успешно: {result.success}")
    print(f"Обработано файлов: {result.total_processed}")
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
