"""
Модуль выполнения PTV анализа для GUI ParticleAnalysis.

Этот модуль предоставляет готовую к использованию структуру для интеграции
с графическим интерфейсом. Все параметры четко определены и могут быть
легко привязаны к элементам GUI.

PTV (Particle Tracking Velocimetry):
- Детектирование частиц на бинаризованных изображениях
- Сопоставление частиц между последовательными кадрами
- Расчет векторов смещения и скоростей

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

from src.ptv.ptv_analysis import (
    PTVAnalyzer,
    PTVProgress,
    PTVResult
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PTVParameters:
    """
    Параметры PTV анализа для GUI.

    Все параметры должны устанавливаться через GUI элементы:
    - input_folder: путь к папке binary_filter_XXXX (через file dialog)
    - detection_min_area: минимальная площадь частицы (spinbox)
    - detection_max_area: максимальная площадь частицы (spinbox)
    - matching_max_distance: максимальный радиус поиска (spinbox)
    - matching_max_diameter_diff: максимальная разница диаметров (spinbox)
    - enable_progress_callback: включить обратную связь прогресса (checkbox)
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    input_folder: str  # Путь к папке binary_filter_XXXX с бинаризованными изображениями

    # ПАРАМЕТРЫ ДЕТЕКТИРОВАНИЯ
    detection_min_area: int = 4  # Минимальная площадь частицы (пикс.)
    detection_max_area: int = 150  # Максимальная площадь частицы (пикс.)

    # ПАРАМЕТРЫ СОПОСТАВЛЕНИЯ
    matching_max_distance: float = 30.0  # Максимальный радиус поиска (пикс.)
    matching_max_diameter_diff: float = 2.0  # Максимальная разница диаметров (пикс.)

    # ОПЦИОНАЛЬНЫЕ ПАРАМЕТРЫ
    enable_progress_callback: bool = True  # Включить callback для прогресса

    # GUI ПОДСКАЗКИ (не используются в обработке, только для GUI)
    detection_min_area_min: int = 1
    detection_min_area_max: int = 1000
    detection_min_area_default: int = 4

    detection_max_area_min: int = 1
    detection_max_area_max: int = 1000
    detection_max_area_default: int = 150

    matching_max_distance_min: float = 1.0
    matching_max_distance_max: float = 100.0
    matching_max_distance_default: float = 30.0

    matching_max_diameter_diff_min: float = 0.0
    matching_max_diameter_diff_max: float = 10.0
    matching_max_diameter_diff_default: float = 2.0

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

        # Проверка параметров детектирования
        if self.detection_min_area < 1:
            return False, f"Минимальная площадь должна быть >= 1: {self.detection_min_area}"

        if self.detection_max_area < self.detection_min_area:
            return False, (
                f"Максимальная площадь должна быть >= минимальной: "
                f"{self.detection_max_area} < {self.detection_min_area}"
            )

        # Проверка параметров сопоставления
        if self.matching_max_distance <= 0:
            return False, f"Максимальное расстояние должно быть > 0: {self.matching_max_distance}"

        if self.matching_max_diameter_diff < 0:
            return False, (
                f"Максимальная разница диаметров должна быть >= 0: "
                f"{self.matching_max_diameter_diff}"
            )

        return True, ""


class PTVExecutor:
    """
    Класс для выполнения PTV анализа с параметрами для GUI.

    Использование:
        1. Создать экземпляр PTVExecutor
        2. Задать параметры через PTVParameters
        3. (Опционально) Установить callback для прогресса
        4. Вызвать execute() для запуска PTV анализа
        5. Получить результат PTVResult
    """

    def __init__(self):
        """Инициализация исполнителя PTV анализа."""
        self.analyzer = PTVAnalyzer()
        self.parameters: Optional[PTVParameters] = None
        self._progress_callback: Optional[Callable[[PTVProgress], None]] = None

        logger.info("Инициализирован PTVExecutor")

    def set_parameters(self, parameters: PTVParameters) -> tuple[bool, str]:
        """
        Установка параметров PTV анализа.

        Args:
            parameters: Параметры PTV анализа

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

        if not self.analyzer.set_detection_config(
            min_area=parameters.detection_min_area,
            max_area=parameters.detection_max_area
        ):
            return False, "Не удалось установить параметры детектирования"

        if not self.analyzer.set_matching_config(
            max_distance=parameters.matching_max_distance,
            max_diameter_diff=parameters.matching_max_diameter_diff
        ):
            return False, "Не удалось установить параметры сопоставления"

        logger.info(
            f"Параметры установлены: "
            f"detection(min={parameters.detection_min_area}, max={parameters.detection_max_area}), "
            f"matching(dist={parameters.matching_max_distance}, "
            f"diam_diff={parameters.matching_max_diameter_diff})"
        )
        return True, ""

    def set_progress_callback(self, callback: Callable[[PTVProgress], None]) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция обратного вызова для GUI
                     Принимает PTVProgress с полями:
                     - current_file: имя текущего файла
                     - total_files: общее количество файлов
                     - processed_files: обработано файлов
                     - current_camera: текущая камера (cam_1 или cam_2)
                     - current_stage: стадия ('detection' или 'matching')
                     - percentage: процент выполнения (0-100)
                     - message: текстовое сообщение
        """
        self._progress_callback = callback

        if self.parameters and self.parameters.enable_progress_callback:
            self.analyzer.set_progress_callback(callback)
            logger.info("Установлен callback для прогресса")

    def cancel(self) -> None:
        """
        Отмена выполнения PTV анализа.

        Этот метод следует вызвать из GUI при нажатии кнопки "Отмена".
        """
        self.analyzer.cancel_processing()
        logger.info("Запрошена отмена обработки")

    def execute(self) -> PTVResult:
        """
        Выполнение PTV анализа.

        Returns:
            PTVResult с результатами анализа:
            - success: успешность выполнения
            - total_images_processed: общее количество обработанных изображений
            - total_particles_detected: общее количество обнаруженных частиц
            - total_pairs_matched: общее количество сопоставленных пар
            - cam1_pairs_count: количество пар в cam_1
            - cam2_pairs_count: количество пар в cam_2
            - errors: список ошибок
            - warnings: список предупреждений
            - output_folder: путь к выходной папке
        """
        if self.parameters is None:
            logger.error("Параметры не установлены")
            return PTVResult(
                success=False,
                total_images_processed=0,
                total_particles_detected=0,
                total_pairs_matched=0,
                cam1_pairs_count=0,
                cam2_pairs_count=0,
                errors=["Параметры не установлены"],
                warnings=[],
                output_folder=""
            )

        logger.info("=" * 60)
        logger.info("ЗАПУСК PTV АНАЛИЗА")
        logger.info(f"Входная папка: {self.parameters.input_folder}")
        logger.info(f"Детектирование: min_area={self.parameters.detection_min_area}, "
                   f"max_area={self.parameters.detection_max_area}")
        logger.info(f"Сопоставление: max_distance={self.parameters.matching_max_distance}, "
                   f"max_diameter_diff={self.parameters.matching_max_diameter_diff}")
        logger.info("=" * 60)

        # Выполнение анализа
        result = self.analyzer.process_all()

        logger.info("=" * 60)
        logger.info("ЗАВЕРШЕНИЕ PTV АНАЛИЗА")
        logger.info(f"Успешно: {result.success}")
        logger.info(f"Обработано изображений: {result.total_images_processed}")
        logger.info(f"Обнаружено частиц: {result.total_particles_detected}")
        logger.info(f"Сопоставлено пар: {result.total_pairs_matched}")
        logger.info(f"  cam_1: {result.cam1_pairs_count} пар")
        logger.info(f"  cam_2: {result.cam2_pairs_count} пар")
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

    def get_detection_preview(self, image_path: str) -> Optional[dict]:
        """
        Получение предварительного просмотра детектирования для одного изображения.

        Args:
            image_path: Путь к изображению

        Returns:
            Словарь с результатами или None
        """
        result = self.analyzer.get_detection_preview(Path(image_path))

        if result is None:
            return None

        image_array, particles = result

        return {
            'image': image_array,
            'particles': particles,
            'particles_count': len(particles),
            'min_area': self.parameters.detection_min_area if self.parameters else 4,
            'max_area': self.parameters.detection_max_area if self.parameters else 150
        }

    def get_matching_preview(self, path_a: str, path_b: str) -> Optional[dict]:
        """
        Получение предварительного просмотра сопоставления для пары изображений.

        Args:
            path_a: Путь к изображению a
            path_b: Путь к изображению b

        Returns:
            Словарь с результатами или None
        """
        return self.analyzer.get_matching_preview(Path(path_a), Path(path_b))


def run_ptv_analysis(
    input_folder: str,
    detection_min_area: int = 4,
    detection_max_area: int = 150,
    matching_max_distance: float = 30.0,
    matching_max_diameter_diff: float = 2.0,
    progress_callback: Optional[Callable] = None
) -> PTVResult:
    """
    Удобная функция для запуска PTV анализа без создания объектов.

    Args:
        input_folder: Путь к папке binary_filter_XXXX
        detection_min_area: Минимальная площадь частицы (по умолчанию 4)
        detection_max_area: Максимальная площадь частицы (по умолчанию 150)
        matching_max_distance: Максимальный радиус поиска (по умолчанию 30.0)
        matching_max_diameter_diff: Максимальная разница диаметров (по умолчанию 2.0)
        progress_callback: Callback функция для прогресса (опционально)

    Returns:
        PTVResult с результатами

    Example:
        >>> result = run_ptv_analysis(
        ...     input_folder="path/to/binary_filter_10000",
        ...     detection_min_area=5,
        ...     detection_max_area=200
        ... )
        >>> print(f"Сопоставлено пар: {result.total_pairs_matched}")
    """
    # Создание параметров
    params = PTVParameters(
        input_folder=input_folder,
        detection_min_area=detection_min_area,
        detection_max_area=detection_max_area,
        matching_max_distance=matching_max_distance,
        matching_max_diameter_diff=matching_max_diameter_diff,
        enable_progress_callback=progress_callback is not None
    )

    # Создание исполнителя
    executor = PTVExecutor()

    # Установка параметров
    success, error_msg = executor.set_parameters(params)
    if not success:
        logger.error(f"Ошибка установки параметров: {error_msg}")
        return PTVResult(
            success=False,
            total_images_processed=0,
            total_particles_detected=0,
            total_pairs_matched=0,
            cam1_pairs_count=0,
            cam2_pairs_count=0,
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
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ С GUI - PTV АНАЛИЗ")
    print("=" * 60)

    # === ШАГ 1: Задание параметров (из GUI элементов) ===
    parameters = PTVParameters(
        input_folder=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\without_filter\binary_filter_2000",
        detection_min_area=4,
        detection_max_area=150,
        matching_max_distance=50.0,
        matching_max_diameter_diff=4.0,
        enable_progress_callback=True
    )

    print(f"\nПараметры:")
    print(f"  Входная папка: {parameters.input_folder}")
    print(f"  Детектирование:")
    print(f"    Минимальная площадь: {parameters.detection_min_area} пикс.")
    print(f"    Максимальная площадь: {parameters.detection_max_area} пикс.")
    print(f"  Сопоставление:")
    print(f"    Максимальное расстояние: {parameters.matching_max_distance} пикс.")
    print(f"    Макс. разница диаметров: {parameters.matching_max_diameter_diff} пикс.")

    # === ШАГ 2: Создание исполнителя ===
    executor = PTVExecutor()

    # === ШАГ 3: Валидация и установка параметров ===
    success, error_msg = executor.set_parameters(parameters)
    if not success:
        print(f"\nОШИБКА: {error_msg}")
        return

    print("\n✓ Параметры валидны")

    # === ШАГ 4: Установка callback для прогресса (для GUI progress bar) ===
    def progress_callback(progress: PTVProgress):
        """Callback для обновления GUI."""
        print(
            f"  [{progress.current_camera}] [{progress.current_stage}] "
            f"{progress.percentage:.1f}% - {progress.message}"
        )

    executor.set_progress_callback(progress_callback)

    # === ШАГ 5: Выполнение PTV анализа ===
    print("\nЗапуск PTV анализа...")
    result = executor.execute()

    # === ШАГ 6: Обработка результата ===
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Успешно: {result.success}")
    print(f"Обработано изображений: {result.total_images_processed}")
    print(f"Обнаружено частиц: {result.total_particles_detected}")
    print(f"Сопоставлено пар: {result.total_pairs_matched}")
    print(f"  cam_1: {result.cam1_pairs_count} пар")
    print(f"  cam_2: {result.cam2_pairs_count} пар")
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
