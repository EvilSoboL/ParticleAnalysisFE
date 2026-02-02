"""
Модуль выполнения анализа оптического потока Lucas-Kanade для GUI ParticleAnalysis.

Этот модуль предоставляет готовую к использованию структуру для интеграции
с графическим интерфейсом. Все параметры четко определены и могут быть
легко привязаны к элементам GUI.

Анализ оптического потока:
- Lucas-Kanade (sparse) - вычисление потока на характерных точках (частицах)
- Экспорт результатов в CSV формат

Входные данные: изображения после фильтрации (8-bit PNG, пары _a и _b)

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import logging

# Добавление пути к модулям проекта
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.kanade.kanade_analysis import (
    LucasKanadeAnalyzer,
    LucasKanadeProgress,
    LucasKanadeResult
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class KanadeAnalysisParameters:
    """
    Параметры анализа оптического потока Lucas-Kanade для GUI.

    Все параметры должны устанавливаться через GUI элементы:
    - input_folder: путь к папке с изображениями (через file dialog)
    - Параметры Lucas-Kanade (через spinbox/slider)
    - enable_progress_callback: включить обратную связь прогресса (checkbox)
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    input_folder: str  # Путь к папке с изображениями

    # ПАРАМЕТРЫ ДЕТЕКЦИИ ТОЧЕК (goodFeaturesToTrack)
    max_corners: int = 500  # Максимальное количество точек для трекинга
    quality_level: float = 0.01  # Минимальное качество угла (0-1)
    min_distance: int = 10  # Минимальное расстояние между точками
    block_size: int = 7  # Размер блока для вычисления производных

    # ПАРАМЕТРЫ LUCAS-KANADE (calcOpticalFlowPyrLK)
    win_size: int = 21  # Размер окна для вычисления потока (нечётное)
    max_level: int = 2  # Количество уровней пирамиды
    max_iterations: int = 30  # Максимальное количество итераций
    epsilon: float = 0.01  # Порог сходимости

    # ПАРАМЕТРЫ ФИЛЬТРАЦИИ
    min_magnitude: float = 0.0  # Минимальная магнитуда для фильтрации
    max_magnitude: float = 100.0  # Максимальная магнитуда для фильтрации

    # ФИЗИЧЕСКИЕ ПАРАМЕТРЫ
    dt: float = 0.002  # Временной интервал между кадрами в паре (мс)
    scaling_factor: float = 1.0  # Масштабный коэффициент (пиксели -> мм)

    # ОПЦИОНАЛЬНЫЕ ПАРАМЕТРЫ
    enable_progress_callback: bool = True  # Включить callback для прогресса

    # GUI ПОДСКАЗКИ - ДЕТЕКЦИЯ ТОЧЕК
    max_corners_min: int = 50
    max_corners_max: int = 5000
    max_corners_default: int = 500
    max_corners_step: int = 50

    quality_level_min: float = 0.001
    quality_level_max: float = 0.1
    quality_level_default: float = 0.01
    quality_level_step: float = 0.001

    min_distance_min: int = 1
    min_distance_max: int = 50
    min_distance_default: int = 10

    block_size_options: tuple = (3, 5, 7, 9, 11)
    block_size_default: int = 7

    # GUI ПОДСКАЗКИ - LUCAS-KANADE
    win_size_options: tuple = (5, 7, 11, 15, 21, 31)
    win_size_default: int = 21

    max_level_min: int = 0
    max_level_max: int = 5
    max_level_default: int = 2

    max_iterations_min: int = 10
    max_iterations_max: int = 100
    max_iterations_default: int = 30

    epsilon_min: float = 0.001
    epsilon_max: float = 0.1
    epsilon_default: float = 0.01
    epsilon_step: float = 0.001

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

        # Проверка параметров детекции точек
        if self.max_corners < 1:
            return False, f"max_corners должен быть >= 1: {self.max_corners}"

        if not (0 < self.quality_level <= 1):
            return False, f"quality_level должен быть в диапазоне (0, 1]: {self.quality_level}"

        if self.min_distance < 1:
            return False, f"min_distance должен быть >= 1: {self.min_distance}"

        if self.block_size < 3 or self.block_size % 2 == 0:
            return False, f"block_size должен быть нечётным и >= 3: {self.block_size}"

        # Проверка параметров Lucas-Kanade
        if self.win_size < 3 or self.win_size % 2 == 0:
            return False, f"win_size должен быть нечётным и >= 3: {self.win_size}"

        if self.max_level < 0:
            return False, f"max_level должен быть >= 0: {self.max_level}"

        if self.max_iterations < 1:
            return False, f"max_iterations должен быть >= 1: {self.max_iterations}"

        if self.epsilon <= 0:
            return False, f"epsilon должен быть > 0: {self.epsilon}"

        # Проверка параметров фильтрации
        if self.min_magnitude < 0:
            return False, f"min_magnitude должен быть >= 0: {self.min_magnitude}"

        if self.max_magnitude <= 0:
            return False, f"max_magnitude должен быть > 0: {self.max_magnitude}"

        # Проверка физических параметров
        if self.dt <= 0:
            return False, f"dt должен быть > 0: {self.dt}"

        if self.scaling_factor <= 0:
            return False, f"scaling_factor должен быть > 0: {self.scaling_factor}"

        return True, ""


class KanadeAnalysisExecutor:
    """
    Класс для выполнения анализа оптического потока Lucas-Kanade с параметрами для GUI.

    Использование:
        1. Создать экземпляр KanadeAnalysisExecutor
        2. Задать параметры через KanadeAnalysisParameters
        3. (Опционально) Установить callback для прогресса
        4. Вызвать execute() для запуска анализа
        5. Получить результат LucasKanadeResult
    """

    def __init__(self):
        """Инициализация исполнителя анализа оптического потока."""
        self.analyzer = LucasKanadeAnalyzer()
        self.parameters: Optional[KanadeAnalysisParameters] = None
        self._progress_callback: Optional[Callable[[LucasKanadeProgress], None]] = None

        logger.info("Инициализирован KanadeAnalysisExecutor")

    def set_parameters(self, parameters: KanadeAnalysisParameters) -> tuple[bool, str]:
        """
        Установка параметров анализа.

        Args:
            parameters: Параметры анализа

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

        # Установка параметров Lucas-Kanade
        if not self.analyzer.set_config(
            max_corners=parameters.max_corners,
            quality_level=parameters.quality_level,
            min_distance=parameters.min_distance,
            block_size=parameters.block_size,
            win_size=parameters.win_size,
            max_level=parameters.max_level,
            max_iterations=parameters.max_iterations,
            epsilon=parameters.epsilon,
            min_magnitude=parameters.min_magnitude,
            max_magnitude=parameters.max_magnitude,
            dt=parameters.dt,
            scaling_factor=parameters.scaling_factor
        ):
            return False, "Не удалось установить параметры Lucas-Kanade"

        logger.info("Параметры Lucas-Kanade установлены")
        return True, ""

    def set_progress_callback(self, callback: Callable[[LucasKanadeProgress], None]) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция обратного вызова для GUI
                     Принимает LucasKanadeProgress с полями:
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
        Отмена выполнения анализа.

        Этот метод следует вызвать из GUI при нажатии кнопки "Отмена".
        """
        self.analyzer.cancel_processing()
        logger.info("Запрошена отмена обработки")

    def execute(self) -> LucasKanadeResult:
        """
        Выполнение анализа оптического потока.

        Returns:
            LucasKanadeResult с результатами анализа:
            - success: успешность выполнения
            - total_pairs_processed: общее количество обработанных пар
            - cam1_pairs_count: количество пар cam_1
            - cam2_pairs_count: количество пар cam_2
            - total_vectors_count: общее количество векторов
            - total_points_detected: всего найдено точек
            - total_points_tracked: всего отслежено точек
            - errors: список ошибок
            - warnings: список предупреждений
            - output_folder: путь к выходной папке
        """
        if self.parameters is None:
            logger.error("Параметры не установлены")
            return LucasKanadeResult(
                success=False,
                total_pairs_processed=0,
                cam1_pairs_count=0,
                cam2_pairs_count=0,
                total_vectors_count=0,
                total_points_detected=0,
                total_points_tracked=0,
                errors=["Параметры не установлены"],
                warnings=[],
                output_folder=""
            )

        logger.info("=" * 60)
        logger.info("ЗАПУСК АНАЛИЗА ОПТИЧЕСКОГО ПОТОКА LUCAS-KANADE")
        logger.info(f"Входная папка: {self.parameters.input_folder}")
        logger.info(f"Параметры: max_corners={self.parameters.max_corners}, "
                   f"quality_level={self.parameters.quality_level}, "
                   f"win_size={self.parameters.win_size}, "
                   f"max_level={self.parameters.max_level}")
        logger.info("=" * 60)

        # Выполнение анализа
        result = self.analyzer.process_all()

        logger.info("=" * 60)
        logger.info("ЗАВЕРШЕНИЕ АНАЛИЗА LUCAS-KANADE")
        logger.info(f"Успешно: {result.success}")
        logger.info(f"Обработано пар: {result.total_pairs_processed}")
        logger.info(f"  cam_1: {result.cam1_pairs_count} пар")
        logger.info(f"  cam_2: {result.cam2_pairs_count} пар")
        logger.info(f"Всего векторов: {result.total_vectors_count}")
        logger.info(f"Всего точек найдено: {result.total_points_detected}")
        logger.info(f"Всего точек отслежено: {result.total_points_tracked}")
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
    ) -> Optional[Dict[str, Any]]:
        """
        Получение предварительного просмотра анализа.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)
            pair_index: Индекс пары для предпросмотра

        Returns:
            Словарь с результатами или None
        """
        if self.parameters is None:
            logger.warning("Параметры не установлены для предпросмотра")
            return None

        preview = self.analyzer.get_preview(camera_name, pair_index)

        if preview is None:
            logger.warning(f"Не удалось создать предпросмотр для {camera_name}")
            return None

        logger.info(
            f"Создан предпросмотр для {camera_name} (Lucas-Kanade): "
            f"найдено {preview['total_detected']} точек, "
            f"отслежено {preview['total_tracked']}, "
            f"средняя магнитуда: {preview['mean_magnitude']:.2f}"
        )

        return preview

    def get_pair_count(self, camera_name: str) -> int:
        """
        Получение количества пар изображений в камере.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Количество пар или 0
        """
        return self.analyzer.get_pair_count(camera_name)


def run_kanade_analysis(
    input_folder: str,
    # Параметры детекции точек
    max_corners: int = 500,
    quality_level: float = 0.01,
    min_distance: int = 10,
    block_size: int = 7,
    # Параметры Lucas-Kanade
    win_size: int = 21,
    max_level: int = 2,
    max_iterations: int = 30,
    epsilon: float = 0.01,
    # Параметры фильтрации
    min_magnitude: float = 0.0,
    max_magnitude: float = 100.0,
    # Физические параметры
    dt: float = 0.002,
    scaling_factor: float = 1.0,
    # Callback
    progress_callback: Optional[Callable] = None
) -> LucasKanadeResult:
    """
    Удобная функция для запуска анализа оптического потока Lucas-Kanade без создания объектов.

    Args:
        input_folder: Путь к папке с изображениями
        max_corners: Максимальное количество точек для трекинга
        quality_level: Минимальное качество угла (0-1)
        min_distance: Минимальное расстояние между точками
        block_size: Размер блока для вычисления производных
        win_size: Размер окна для вычисления потока
        max_level: Количество уровней пирамиды
        max_iterations: Максимальное количество итераций
        epsilon: Порог сходимости
        min_magnitude: Минимальная магнитуда для фильтрации
        max_magnitude: Максимальная магнитуда для фильтрации
        dt: Временной интервал между кадрами
        scaling_factor: Масштабный коэффициент
        progress_callback: Callback функция для прогресса

    Returns:
        LucasKanadeResult с результатами

    Example:
        >>> result = run_kanade_analysis(
        ...     input_folder="path/to/filtered_images",
        ...     max_corners=1000,
        ...     win_size=21
        ... )
        >>> print(f"Векторов: {result.total_vectors_count}")
        >>> print(f"Отслежено точек: {result.total_points_tracked}")
    """
    # Создание параметров
    params = KanadeAnalysisParameters(
        input_folder=input_folder,
        max_corners=max_corners,
        quality_level=quality_level,
        min_distance=min_distance,
        block_size=block_size,
        win_size=win_size,
        max_level=max_level,
        max_iterations=max_iterations,
        epsilon=epsilon,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        dt=dt,
        scaling_factor=scaling_factor,
        enable_progress_callback=progress_callback is not None
    )

    # Создание исполнителя
    executor = KanadeAnalysisExecutor()

    # Установка параметров
    success, error_msg = executor.set_parameters(params)
    if not success:
        logger.error(f"Ошибка установки параметров: {error_msg}")
        return LucasKanadeResult(
            success=False,
            total_pairs_processed=0,
            cam1_pairs_count=0,
            cam2_pairs_count=0,
            total_vectors_count=0,
            total_points_detected=0,
            total_points_tracked=0,
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
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ С GUI - АНАЛИЗ LUCAS-KANADE")
    print("=" * 60)

    # === ШАГ 1: Задание параметров (из GUI элементов) ===
    parameters = KanadeAnalysisParameters(
        input_folder=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\without_filter",
        # Параметры детекции точек
        max_corners=10_000,
        quality_level=0.01,
        min_distance=10,
        block_size=7,
        # Параметры Lucas-Kanade
        win_size=41,
        max_level=1,
        max_iterations=5,
        epsilon=0.01,
        # Физические параметры
        dt=0.000002,
        scaling_factor=7.5,
        enable_progress_callback=True
    )

    print(f"\nМетод: Lucas-Kanade (sparse optical flow)")
    print(f"\nПараметры:")
    print(f"  Входная папка: {parameters.input_folder}")
    print(f"  max_corners: {parameters.max_corners}")
    print(f"  quality_level: {parameters.quality_level}")
    print(f"  min_distance: {parameters.min_distance}")
    print(f"  win_size: {parameters.win_size}")
    print(f"  max_level: {parameters.max_level}")

    # === ШАГ 2: Создание исполнителя ===
    executor = KanadeAnalysisExecutor()

    # === ШАГ 3: Валидация и установка параметров ===
    success, error_msg = executor.set_parameters(parameters)
    if not success:
        print(f"\nОШИБКА: {error_msg}")
        return

    print("\n[OK] Параметры валидны")

    # === ШАГ 4: Количество пар для обработки ===
    cam1_pairs = executor.get_pair_count("cam_1")
    cam2_pairs = executor.get_pair_count("cam_2")
    print(f"\nКоличество пар:")
    print(f"  cam_1: {cam1_pairs}")
    print(f"  cam_2: {cam2_pairs}")

    # === ШАГ 5: Предварительный просмотр ===
    print("\nПредварительный просмотр анализа...")
    preview = executor.get_preview("cam_1", pair_index=0)
    if preview:
        print(f"\nРезультаты предпросмотра:")
        print(f"  Пара: {preview['pair_names'][0]} - {preview['pair_names'][1]}")
        print(f"  Найдено точек: {preview['total_detected']}")
        print(f"  Отслежено точек: {preview['total_tracked']}")
        print(f"  Средняя магнитуда: {preview['mean_magnitude']:.3f}")
        print(f"  Макс. магнитуда: {preview['max_magnitude']:.3f}")

    # === ШАГ 6: Установка callback для прогресса (для GUI progress bar) ===
    def progress_callback(progress: LucasKanadeProgress):
        """Callback для обновления GUI."""
        print(
            f"  [{progress.current_camera}] "
            f"{progress.percentage:.1f}% - {progress.message}"
        )

    executor.set_progress_callback(progress_callback)

    # === ШАГ 7: Выполнение анализа ===
    print("\nЗапуск анализа Lucas-Kanade...")
    result = executor.execute()

    # === ШАГ 8: Обработка результата ===
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Успешно: {result.success}")
    print(f"Обработано пар: {result.total_pairs_processed}")
    print(f"  cam_1: {result.cam1_pairs_count} пар")
    print(f"  cam_2: {result.cam2_pairs_count} пар")
    print(f"Всего векторов: {result.total_vectors_count}")
    print(f"Всего точек найдено: {result.total_points_detected}")
    print(f"Всего точек отслежено: {result.total_points_tracked}")
    print(f"Ошибок: {len(result.errors)}")
    print(f"Предупреждений: {len(result.warnings)}")
    print(f"Выходная папка: {result.output_folder}")

    if result.errors:
        print("\nОшибки:")
        for error in result.errors[:5]:
            print(f"  - {error}")

    if result.warnings:
        print("\nПредупреждения:")
        for warning in result.warnings[:5]:
            print(f"  - {warning}")

    print("=" * 60)


if __name__ == "__main__":
    # При запуске модуля напрямую - показать пример использования
    example_gui_usage()
