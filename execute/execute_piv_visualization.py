"""
Модуль выполнения визуализации PIV результатов для GUI ParticleAnalysis.

Этот модуль предоставляет готовую к использованию структуру для интеграции
с графическим интерфейсом. Все параметры четко определены и могут быть
легко привязаны к элементам GUI.

Визуализация PIV (первые 10 пар):
- Наложение векторов скоростей (стрелок) на исходные изображения
- Цветовая маркировка по магнитуде
- Настраиваемые параметры отображения

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

from src.visualization.piv_visualization import (
    PIVVisualizer,
    PIVVisualizationProgress,
    PIVVisualizationResult
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PIVVisualizationParameters:
    """
    Параметры визуализации PIV результатов для GUI.

    Все параметры должны устанавливаться через GUI элементы:
    - original_folder: путь к папке с исходными изображениями (file dialog)
    - piv_folder: путь к папке PIV_XXXX с результатами анализа (file dialog)
    - arrow_color: цвет стрелок (color picker) - зелёный по умолчанию
    - arrow_thickness: толщина стрелок (spinbox)
    - arrow_scale: масштаб стрелок (slider или spinbox)
    - show_magnitude_color: цвет по магнитуде (checkbox)
    - max_pairs: количество пар для обработки (spinbox)
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    original_folder: str  # Путь к папке с исходными изображениями (intensity_filtered_XXXX или cam_sorted)
    piv_folder: str  # Путь к папке PIV_XXXX с результатами PIV анализа

    # ПАРАМЕТРЫ ВИЗУАЛИЗАЦИИ
    arrow_color: Tuple[int, int, int] = (0, 255, 0)  # Зелёный (BGR)
    arrow_thickness: int = 1  # Толщина стрелок
    arrow_scale: float = 5.0  # Масштаб стрелок (увеличение длины)
    arrow_tip_length: float = 0.3  # Длина наконечника относительно стрелки
    show_magnitude_color: bool = True  # Использовать цвет по магнитуде

    # ПАРАМЕТРЫ ОБРАБОТКИ
    max_pairs: int = 10  # Максимальное количество пар для обработки

    # ОПЦИОНАЛЬНЫЕ ПАРАМЕТРЫ
    enable_progress_callback: bool = True  # Включить callback для прогресса

    # GUI ПОДСКАЗКИ (не используются в обработке, только для GUI)
    arrow_thickness_min: int = 1
    arrow_thickness_max: int = 5
    arrow_scale_min: float = 1.0
    arrow_scale_max: float = 20.0
    max_pairs_min: int = 1
    max_pairs_max: int = 100

    # Информация для пользователя
    visualization_info: str = (
        "Визуализация PIV:\n"
        "• Стрелки показывают направление и величину скорости\n"
        "• Цвет по магнитуде: синий (мин) -> красный (макс)\n"
        "• Обрабатываются первые N пар изображений"
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

        if not cam1_path.exists() and not cam2_path.exists():
            return False, f"Не найдены папки cam_1 или cam_2 в {self.original_folder}"

        # Проверка папки с результатами PIV
        piv_path = Path(self.piv_folder)
        if not piv_path.exists():
            return False, f"Папка PIV результатов не существует: {self.piv_folder}"

        cam1_piv = piv_path / "cam_1"
        cam2_piv = piv_path / "cam_2"

        if not cam1_piv.exists() and not cam2_piv.exists():
            return False, f"Не найдены папки cam_1 или cam_2 с PIV результатами в {self.piv_folder}"

        # Проверка параметров визуализации
        if self.arrow_thickness < 1 or self.arrow_thickness > 10:
            return False, f"Толщина стрелки должна быть в диапазоне [1, 10]: {self.arrow_thickness}"

        if self.arrow_scale < 0.1 or self.arrow_scale > 100:
            return False, f"Масштаб стрелки должен быть в диапазоне [0.1, 100]: {self.arrow_scale}"

        if self.max_pairs < 1 or self.max_pairs > 1000:
            return False, f"Количество пар должно быть в диапазоне [1, 1000]: {self.max_pairs}"

        # Проверка цвета (BGR: 0-255 для каждого канала)
        if len(self.arrow_color) != 3:
            return False, "arrow_color должен содержать 3 значения (B, G, R)"
        if not all(0 <= c <= 255 for c in self.arrow_color):
            return False, "arrow_color должен содержать значения в диапазоне [0, 255]"

        return True, ""


class PIVVisualizationExecutor:
    """
    Класс для выполнения визуализации PIV результатов с параметрами для GUI.

    Использование:
        1. Создать экземпляр PIVVisualizationExecutor
        2. Задать параметры через PIVVisualizationParameters
        3. (Опционально) Установить callback для прогресса
        4. Вызвать execute() для запуска визуализации
        5. Получить результат PIVVisualizationResult
    """

    def __init__(self):
        """Инициализация исполнителя визуализации PIV."""
        self.visualizer = PIVVisualizer()
        self.parameters: Optional[PIVVisualizationParameters] = None
        self._progress_callback: Optional[Callable[[PIVVisualizationProgress], None]] = None

        logger.info("Инициализирован PIVVisualizationExecutor")

    def set_parameters(self, parameters: PIVVisualizationParameters) -> tuple[bool, str]:
        """
        Установка параметров визуализации PIV.

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

        if not self.visualizer.set_piv_folder(parameters.piv_folder):
            return False, "Не удалось установить папку с PIV результатами"

        # Установка параметров визуализации
        self.visualizer.set_visualization_config(
            arrow_color=parameters.arrow_color,
            arrow_thickness=parameters.arrow_thickness,
            arrow_scale=parameters.arrow_scale,
            arrow_tip_length=parameters.arrow_tip_length,
            show_magnitude_color=parameters.show_magnitude_color,
            max_pairs=parameters.max_pairs
        )

        logger.info(
            f"Параметры установлены: "
            f"original={parameters.original_folder}, "
            f"piv={parameters.piv_folder}, "
            f"max_pairs={parameters.max_pairs}, "
            f"scale={parameters.arrow_scale}"
        )
        return True, ""

    def set_progress_callback(self, callback: Callable[[PIVVisualizationProgress], None]) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция обратного вызова для GUI
                     Принимает PIVVisualizationProgress с полями:
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
        logger.info("Запрошена отмена визуализации PIV")

    def execute(self) -> PIVVisualizationResult:
        """
        Выполнение визуализации PIV результатов.

        Returns:
            PIVVisualizationResult с результатами визуализации:
            - success: успешность выполнения
            - total_pairs_processed: общее количество обработанных пар
            - cam1_visualizations: количество визуализаций cam_1
            - cam2_visualizations: количество визуализаций cam_2
            - errors: список ошибок
            - output_folder: путь к выходной папке
        """
        if self.parameters is None:
            logger.error("Параметры не установлены")
            return PIVVisualizationResult(
                success=False,
                total_pairs_processed=0,
                cam1_visualizations=0,
                cam2_visualizations=0,
                errors=["Параметры не установлены"],
                output_folder=""
            )

        logger.info("=" * 60)
        logger.info("ЗАПУСК ВИЗУАЛИЗАЦИИ PIV РЕЗУЛЬТАТОВ")
        logger.info(f"Исходные изображения: {self.parameters.original_folder}")
        logger.info(f"PIV результаты: {self.parameters.piv_folder}")
        logger.info(f"Максимум пар: {self.parameters.max_pairs}")
        logger.info(f"Масштаб стрелок: {self.parameters.arrow_scale}")
        logger.info(f"Цвет по магнитуде: {self.parameters.show_magnitude_color}")
        logger.info("=" * 60)

        # Выполнение визуализации
        result = self.visualizer.process_all()

        logger.info("=" * 60)
        logger.info("ЗАВЕРШЕНИЕ ВИЗУАЛИЗАЦИИ PIV")
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
        Получение предварительного просмотра визуализации PIV для одной пары.

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
        Получение статистики векторов для одной пары.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)
            pair_number: Номер пары

        Returns:
            Словарь со статистикой или None
        """
        return self.visualizer.get_pair_statistics(camera_name, pair_number)


def run_piv_visualization(
    original_folder: str,
    piv_folder: str,
    arrow_color: Tuple[int, int, int] = (0, 255, 0),
    arrow_thickness: int = 1,
    arrow_scale: float = 5.0,
    show_magnitude_color: bool = True,
    max_pairs: int = 10,
    progress_callback: Optional[Callable] = None
) -> PIVVisualizationResult:
    """
    Удобная функция для запуска визуализации PIV без создания объектов.

    Args:
        original_folder: Путь к папке с исходными изображениями
        piv_folder: Путь к папке PIV_XXXX с результатами
        arrow_color: Цвет стрелок (BGR) - зелёный по умолчанию
        arrow_thickness: Толщина стрелок (по умолчанию 1)
        arrow_scale: Масштаб стрелок (по умолчанию 5.0)
        show_magnitude_color: Цвет по магнитуде (по умолчанию True)
        max_pairs: Количество пар для обработки (по умолчанию 10)
        progress_callback: Callback функция для прогресса (опционально)

    Returns:
        PIVVisualizationResult с результатами

    Example:
        >>> result = run_piv_visualization(
        ...     original_folder="path/to/intensity_filtered_3240",
        ...     piv_folder="path/to/PIV_3240",
        ...     max_pairs=10,
        ...     arrow_scale=5.0
        ... )
        >>> print(f"Создано визуализаций: {result.cam1_visualizations + result.cam2_visualizations}")
    """
    # Создание параметров
    params = PIVVisualizationParameters(
        original_folder=original_folder,
        piv_folder=piv_folder,
        arrow_color=arrow_color,
        arrow_thickness=arrow_thickness,
        arrow_scale=arrow_scale,
        show_magnitude_color=show_magnitude_color,
        max_pairs=max_pairs,
        enable_progress_callback=progress_callback is not None
    )

    # Создание исполнителя
    executor = PIVVisualizationExecutor()

    # Установка параметров
    success, error_msg = executor.set_parameters(params)
    if not success:
        logger.error(f"Ошибка установки параметров: {error_msg}")
        return PIVVisualizationResult(
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
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ С GUI - ВИЗУАЛИЗАЦИЯ PIV")
    print("=" * 60)

    # === ШАГ 1: Задание параметров (из GUI элементов) ===
    # ВНИМАНИЕ: Замените пути на реальные пути к папкам
    parameters = PIVVisualizationParameters(
        original_folder=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\intensity_filtered_3240",
        piv_folder=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\PIV_3240",
        arrow_color=(0, 255, 0),  # Зелёный (BGR)
        arrow_thickness=1,
        arrow_scale=5.0,
        show_magnitude_color=True,
        max_pairs=10,
        enable_progress_callback=True
    )

    print(f"\nПараметры:")
    print(f"  Исходные изображения: {parameters.original_folder}")
    print(f"  PIV результаты: {parameters.piv_folder}")
    print(f"  Цвет стрелок (BGR): {parameters.arrow_color}")
    print(f"  Толщина стрелок: {parameters.arrow_thickness}")
    print(f"  Масштаб стрелок: {parameters.arrow_scale}")
    print(f"  Цвет по магнитуде: {parameters.show_magnitude_color}")
    print(f"  Максимум пар: {parameters.max_pairs}")
    print(f"\n{parameters.visualization_info}")

    # === ШАГ 2: Создание исполнителя ===
    executor = PIVVisualizationExecutor()

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
        print(f"  Количество векторов: {stats['vectors_count']}")
        if 'mean_magnitude' in stats:
            print(f"  Средняя магнитуда: {stats['mean_magnitude']:.4f}")
            print(f"  Макс магнитуда: {stats.get('max_magnitude', 0):.4f}")
            print(f"  Среднее U: {stats.get('mean_u', 0):.4f}")
            print(f"  Среднее V: {stats.get('mean_v', 0):.4f}")

    # === ШАГ 5: Установка callback для прогресса (для GUI progress bar) ===
    def progress_callback(progress: PIVVisualizationProgress):
        """Callback для обновления GUI."""
        print(f"  [{progress.current_camera}] {progress.percentage:.1f}% - {progress.message}")

    executor.set_progress_callback(progress_callback)

    # === ШАГ 6: Выполнение визуализации ===
    print("\nЗапуск визуализации PIV...")
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
