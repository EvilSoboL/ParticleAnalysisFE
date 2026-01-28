"""
Модуль выполнения визуализации PTV результатов для GUI ParticleAnalysis.

Этот модуль предоставляет готовую к использованию структуру для интеграции
с графическим интерфейсом. Все параметры четко определены и могут быть
легко привязаны к элементам GUI.

Визуализация One-to-One Matching:
- Наложение центров частиц на исходные изображения
- Отображение векторов смещения между сопоставленными частицами
- Цветовая маркировка: зелёный (кадр A), красный (кадр B), оранжевый (связи)

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import logging

# Добавление пути к модулям проекта
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.one_to_one_visualization import (
    PTVVisualizer,
    VisualizationProgress,
    VisualizationResult
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VisualizationParameters:
    """
    Параметры визуализации PTV результатов для GUI.

    Все параметры должны устанавливаться через GUI элементы:
    - original_folder: путь к папке cam_sorted с исходными изображениями (file dialog)
    - ptv_folder: путь к папке PTV_XXXX с результатами анализа (file dialog)
    - particle_a_color: цвет частиц кадра A (color picker) - зелёный по умолчанию
    - particle_b_color: цвет частиц кадра B (color picker) - красный по умолчанию
    - line_color: цвет соединительных линий (color picker) - оранжевый по умолчанию
    - line_thickness: толщина линий (spinbox)
    - enable_progress_callback: включить обратную связь прогресса (checkbox)
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    original_folder: str  # Путь к папке cam_sorted с исходными изображениями
    ptv_folder: str  # Путь к папке PTV_XXXX с результатами PTV анализа

    # ПАРАМЕТРЫ ВИЗУАЛИЗАЦИИ
    particle_a_color: Tuple[int, int, int] = (0, 255, 0)  # Зелёный (BGR) - кадр A
    particle_b_color: Tuple[int, int, int] = (0, 0, 255)  # Красный (BGR) - кадр B
    line_color: Tuple[int, int, int] = (0, 165, 255)  # Оранжевый (BGR) - связи
    line_thickness: int = 1  # Толщина линии

    # ОПЦИОНАЛЬНЫЕ ПАРАМЕТРЫ
    enable_progress_callback: bool = True  # Включить callback для прогресса

    # GUI ПОДСКАЗКИ (не используются в обработке, только для GUI)
    line_thickness_min: int = 1
    line_thickness_max: int = 5
    line_thickness_default: int = 1

    # Информация для пользователя
    color_info: str = (
        "Цветовая схема:\n"
        "• Зелёный - частицы в кадре A (начальная позиция)\n"
        "• Красный - частицы в кадре B (конечная позиция)\n"
        "• Оранжевый - линии связи между сопоставленными частицами"
    )

    def validate(self) -> tuple[bool, str]:
        """
        Валидация параметров.

        Returns:
            tuple[bool, str]: (success, error_message)
        """
        # Проверка папки с исходными изображениями
        original_path = Path(self.original_folder)
        if not original_path.exists():
            return False, f"Папка исходных изображений не существует: {self.original_folder}"

        cam1_path = original_path / "cam_1"
        cam2_path = original_path / "cam_2"

        if not cam1_path.exists():
            return False, f"Не найдена папка cam_1 в {self.original_folder}"
        if not cam2_path.exists():
            return False, f"Не найдена папка cam_2 в {self.original_folder}"

        # Проверка папки с результатами PTV
        ptv_path = Path(self.ptv_folder)
        if not ptv_path.exists():
            return False, f"Папка PTV результатов не существует: {self.ptv_folder}"

        cam1_pairs = ptv_path / "cam_1_pairs"
        cam2_pairs = ptv_path / "cam_2_pairs"

        if not cam1_pairs.exists() and not cam2_pairs.exists():
            return False, (
                f"Папка {self.ptv_folder} должна содержать cam_1_pairs и/или cam_2_pairs"
            )

        # Проверка параметров визуализации
        if self.line_thickness < 1 or self.line_thickness > 10:
            return False, f"Толщина линии должна быть в диапазоне [1, 10]: {self.line_thickness}"

        # Проверка цветов (BGR: 0-255 для каждого канала)
        for color_name, color in [
            ("particle_a_color", self.particle_a_color),
            ("particle_b_color", self.particle_b_color),
            ("line_color", self.line_color)
        ]:
            if len(color) != 3:
                return False, f"{color_name} должен содержать 3 значения (B, G, R)"
            if not all(0 <= c <= 255 for c in color):
                return False, f"{color_name} должен содержать значения в диапазоне [0, 255]"

        return True, ""


class VisualizationExecutor:
    """
    Класс для выполнения визуализации PTV результатов с параметрами для GUI.

    Использование:
        1. Создать экземпляр VisualizationExecutor
        2. Задать параметры через VisualizationParameters
        3. (Опционально) Установить callback для прогресса
        4. Вызвать execute() для запуска визуализации
        5. Получить результат VisualizationResult
    """

    def __init__(self):
        """Инициализация исполнителя визуализации."""
        self.visualizer = PTVVisualizer()
        self.parameters: Optional[VisualizationParameters] = None
        self._progress_callback: Optional[Callable[[VisualizationProgress], None]] = None

        logger.info("Инициализирован VisualizationExecutor")

    def set_parameters(self, parameters: VisualizationParameters) -> tuple[bool, str]:
        """
        Установка параметров визуализации.

        Args:
            parameters: Параметры визуализации

        Returns:
            tuple[bool, str]: (success, error_message)
        """
        # Валидация параметров
        success, error_msg = parameters.validate()
        if not success:
            logger.error(f"Ошибка валидации параметров: {error_msg}")
            return False, error_msg

        self.parameters = parameters

        # Применение параметров к визуализатору
        if not self.visualizer.set_original_folder(parameters.original_folder):
            return False, "Не удалось установить папку с исходными изображениями"

        if not self.visualizer.set_ptv_folder(parameters.ptv_folder):
            return False, "Не удалось установить папку с PTV результатами"

        # Установка цветовых параметров
        self.visualizer.set_visualization_config(
            particle_a_color=parameters.particle_a_color,
            particle_b_color=parameters.particle_b_color,
            line_color=parameters.line_color,
            line_thickness=parameters.line_thickness
        )

        logger.info(
            f"Параметры установлены: "
            f"original={parameters.original_folder}, "
            f"ptv={parameters.ptv_folder}, "
            f"line_thickness={parameters.line_thickness}"
        )
        return True, ""

    def set_progress_callback(self, callback: Callable[[VisualizationProgress], None]) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция обратного вызова для GUI
                     Принимает VisualizationProgress с полями:
                     - current_file: имя текущего файла
                     - total_files: общее количество файлов
                     - processed_files: обработано файлов
                     - current_camera: текущая камера (cam_1 или cam_2)
                     - percentage: процент выполнения (0-100)
                     - message: текстовое сообщение
        """
        self._progress_callback = callback

        if self.parameters and self.parameters.enable_progress_callback:
            self.visualizer.set_progress_callback(callback)
            logger.info("Установлен callback для прогресса")

    def cancel(self) -> None:
        """
        Отмена выполнения визуализации.

        Этот метод следует вызвать из GUI при нажатии кнопки "Отмена".
        """
        self.visualizer.cancel_processing()
        logger.info("Запрошена отмена визуализации")

    def execute(self) -> VisualizationResult:
        """
        Выполнение визуализации PTV результатов.

        Returns:
            VisualizationResult с результатами визуализации:
            - success: успешность выполнения
            - total_pairs_processed: общее количество обработанных пар
            - cam1_visualizations: количество визуализаций cam_1
            - cam2_visualizations: количество визуализаций cam_2
            - errors: список ошибок
            - output_folder: путь к выходной папке
        """
        if self.parameters is None:
            logger.error("Параметры не установлены")
            return VisualizationResult(
                success=False,
                total_pairs_processed=0,
                cam1_visualizations=0,
                cam2_visualizations=0,
                errors=["Параметры не установлены"],
                output_folder=""
            )

        logger.info("=" * 60)
        logger.info("ЗАПУСК ВИЗУАЛИЗАЦИИ PTV РЕЗУЛЬТАТОВ")
        logger.info(f"Исходные изображения: {self.parameters.original_folder}")
        logger.info(f"PTV результаты: {self.parameters.ptv_folder}")
        logger.info(f"Цвет частиц A (BGR): {self.parameters.particle_a_color}")
        logger.info(f"Цвет частиц B (BGR): {self.parameters.particle_b_color}")
        logger.info(f"Цвет линий (BGR): {self.parameters.line_color}")
        logger.info(f"Толщина линий: {self.parameters.line_thickness}")
        logger.info("=" * 60)

        # Выполнение визуализации
        result = self.visualizer.process_all()

        logger.info("=" * 60)
        logger.info("ЗАВЕРШЕНИЕ ВИЗУАЛИЗАЦИИ")
        logger.info(f"Успешно: {result.success}")
        logger.info(f"Обработано пар: {result.total_pairs_processed}")
        logger.info(f"Создано визуализаций:")
        logger.info(f"  cam_1: {result.cam1_visualizations}")
        logger.info(f"  cam_2: {result.cam2_visualizations}")
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
        if self.visualizer.output_folder:
            return str(self.visualizer.output_folder)
        return None

    def get_preview(self, camera_name: str, pair_number: int) -> Optional[dict]:
        """
        Получение предварительного просмотра визуализации для одной пары.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)
            pair_number: Номер пары

        Returns:
            Словарь с результатами или None
        """
        result = self.visualizer.get_preview(camera_name, pair_number)

        if result is None:
            return None

        vis_a, vis_b = result

        return {
            'vis_a': vis_a,
            'vis_b': vis_b,
            'camera': camera_name,
            'pair_number': pair_number
        }

    def get_pair_statistics(self, camera_name: str, pair_number: int) -> Optional[dict]:
        """
        Получение статистики для одной пары из CSV.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)
            pair_number: Номер пары

        Returns:
            Словарь со статистикой или None
        """
        return self.visualizer.get_pair_statistics(camera_name, pair_number)


def run_visualization(
    original_folder: str,
    ptv_folder: str,
    particle_a_color: Tuple[int, int, int] = (0, 255, 0),
    particle_b_color: Tuple[int, int, int] = (0, 0, 255),
    line_color: Tuple[int, int, int] = (0, 165, 255),
    line_thickness: int = 1,
    progress_callback: Optional[Callable] = None
) -> VisualizationResult:
    """
    Удобная функция для запуска визуализации без создания объектов.

    Args:
        original_folder: Путь к папке cam_sorted с исходными изображениями
        ptv_folder: Путь к папке PTV_XXXX с результатами
        particle_a_color: Цвет частиц кадра A (BGR) - зелёный по умолчанию
        particle_b_color: Цвет частиц кадра B (BGR) - красный по умолчанию
        line_color: Цвет соединительных линий (BGR) - оранжевый по умолчанию
        line_thickness: Толщина линий (по умолчанию 1)
        progress_callback: Callback функция для прогресса (опционально)

    Returns:
        VisualizationResult с результатами

    Example:
        >>> result = run_visualization(
        ...     original_folder="path/to/cam_sorted",
        ...     ptv_folder="path/to/PTV_10000",
        ...     line_thickness=2
        ... )
        >>> print(f"Создано визуализаций: {result.cam1_visualizations + result.cam2_visualizations}")
    """
    # Создание параметров
    params = VisualizationParameters(
        original_folder=original_folder,
        ptv_folder=ptv_folder,
        particle_a_color=particle_a_color,
        particle_b_color=particle_b_color,
        line_color=line_color,
        line_thickness=line_thickness,
        enable_progress_callback=progress_callback is not None
    )

    # Создание исполнителя
    executor = VisualizationExecutor()

    # Установка параметров
    success, error_msg = executor.set_parameters(params)
    if not success:
        logger.error(f"Ошибка установки параметров: {error_msg}")
        return VisualizationResult(
            success=False,
            total_pairs_processed=0,
            cam1_visualizations=0,
            cam2_visualizations=0,
            errors=[error_msg],
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
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ С GUI - ВИЗУАЛИЗАЦИЯ PTV")
    print("=" * 60)

    # === ШАГ 1: Задание параметров (из GUI элементов) ===
    parameters = VisualizationParameters(
        original_folder=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted",
        ptv_folder=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\PTV_3240",
        particle_a_color=(0, 255, 0),  # Зелёный (BGR)
        particle_b_color=(0, 0, 255),  # Красный (BGR)
        line_color=(0, 165, 255),  # Оранжевый (BGR)
        line_thickness=1,
        enable_progress_callback=True
    )

    print(f"\nПараметры:")
    print(f"  Исходные изображения: {parameters.original_folder}")
    print(f"  PTV результаты: {parameters.ptv_folder}")
    print(f"  Цвет частиц A (BGR): {parameters.particle_a_color}")
    print(f"  Цвет частиц B (BGR): {parameters.particle_b_color}")
    print(f"  Цвет линий (BGR): {parameters.line_color}")
    print(f"  Толщина линий: {parameters.line_thickness}")
    print(f"\n{parameters.color_info}")

    # === ШАГ 2: Создание исполнителя ===
    executor = VisualizationExecutor()

    # === ШАГ 3: Валидация и установка параметров ===
    success, error_msg = executor.set_parameters(parameters)
    if not success:
        print(f"\nОШИБКА: {error_msg}")
        return

    print("\n✓ Параметры валидны")

    # === ШАГ 4: Получение статистики для примера ===
    print("\nПолучение статистики для первой пары cam_1...")
    stats = executor.get_pair_statistics("cam_1", 1)
    if stats:
        print(f"  Сопоставлено пар: {stats['pairs_count']}")
        print(f"  Средний диаметр: {stats['mean_diameter']:.2f} пикс.")
        if 'mean_displacement' in stats:
            print(f"  Среднее смещение: {stats['mean_displacement']:.2f} пикс.")
            print(f"  Среднее dx: {stats.get('mean_dx', 0):.2f} пикс.")
            print(f"  Среднее dy: {stats.get('mean_dy', 0):.2f} пикс.")

    # === ШАГ 5: Установка callback для прогресса (для GUI progress bar) ===
    def progress_callback(progress: VisualizationProgress):
        """Callback для обновления GUI."""
        print(f"  [{progress.current_camera}] {progress.percentage:.1f}% - {progress.message}")

    executor.set_progress_callback(progress_callback)

    # === ШАГ 6: Выполнение визуализации ===
    print("\nЗапуск визуализации...")
    result = executor.execute()

    # === ШАГ 7: Обработка результата ===
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Успешно: {result.success}")
    print(f"Обработано пар: {result.total_pairs_processed}")
    print(f"Создано визуализаций:")
    print(f"  cam_1: {result.cam1_visualizations}")
    print(f"  cam_2: {result.cam2_visualizations}")
    print(f"  Всего: {result.cam1_visualizations + result.cam2_visualizations}")
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
