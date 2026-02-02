"""
Модуль выполнения анализа оптического потока Farneback для GUI ParticleAnalysis.

Этот модуль предоставляет готовую к использованию структуру для интеграции
с графическим интерфейсом. Все параметры четко определены и могут быть
легко привязаны к элементам GUI.

Анализ оптического потока:
- Farneback (dense) - вычисление потока для каждого пикселя
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

from src.farneback.farneback_analysis import (
    FarnebackAnalyzer,
    FarnebackProgress,
    FarnebackResult
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FarnebackAnalysisParameters:
    """
    Параметры анализа оптического потока Farneback для GUI.

    Все параметры должны устанавливаться через GUI элементы:
    - input_folder: путь к папке farneback_filtered_XXXX (через file dialog)
    - Параметры Farneback (через spinbox/slider)
    - enable_progress_callback: включить обратную связь прогресса (checkbox)
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    input_folder: str  # Путь к папке farneback_filtered_XXXX с изображениями

    # ПАРАМЕТРЫ FARNEBACK
    pyr_scale: float = 0.5  # Масштаб пирамиды (0-1)
    levels: int = 3  # Количество уровней пирамиды
    winsize: int = 15  # Размер окна усреднения (нечётное)
    iterations: int = 3  # Количество итераций на каждом уровне
    poly_n: int = 5  # Размер окрестности для полиномиальной аппроксимации (5 или 7)
    poly_sigma: float = 1.2  # Стандартное отклонение гауссиана

    # ПАРАМЕТРЫ ФИЛЬТРАЦИИ FARNEBACK
    min_magnitude: float = 0.0  # Минимальная магнитуда для фильтрации
    max_magnitude: float = 100.0  # Максимальная магнитуда для фильтрации
    grid_step: int = 10  # Шаг сетки для экспорта векторов

    # ФИЗИЧЕСКИЕ ПАРАМЕТРЫ
    dt: float = 0.002  # Временной интервал между кадрами в паре (мс)
    scaling_factor: float = 1.0  # Масштабный коэффициент (пиксели -> мм)

    # ОПЦИОНАЛЬНЫЕ ПАРАМЕТРЫ
    enable_progress_callback: bool = True  # Включить callback для прогресса

    # GUI ПОДСКАЗКИ - FARNEBACK
    pyr_scale_min: float = 0.1
    pyr_scale_max: float = 0.9
    pyr_scale_default: float = 0.5
    pyr_scale_step: float = 0.1

    levels_min: int = 1
    levels_max: int = 10
    levels_default: int = 3

    winsize_options: tuple = (5, 7, 9, 11, 13, 15, 17, 19, 21)
    winsize_default: int = 15

    iterations_min: int = 1
    iterations_max: int = 10
    iterations_default: int = 3

    poly_n_options: tuple = (5, 7)
    poly_n_default: int = 5

    poly_sigma_min: float = 0.5
    poly_sigma_max: float = 2.0
    poly_sigma_default: float = 1.2
    poly_sigma_step: float = 0.1

    grid_step_min: int = 1
    grid_step_max: int = 50
    grid_step_default: int = 10

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

        # Проверка параметров Farneback
        if not (0 < self.pyr_scale < 1):
            return False, f"pyr_scale должен быть в диапазоне (0, 1): {self.pyr_scale}"

        if self.levels < 1:
            return False, f"levels должен быть >= 1: {self.levels}"

        if self.winsize < 1 or self.winsize % 2 == 0:
            return False, f"winsize должен быть нечётным и >= 1: {self.winsize}"

        if self.iterations < 1:
            return False, f"iterations должен быть >= 1: {self.iterations}"

        if self.poly_n not in (5, 7):
            return False, f"poly_n должен быть 5 или 7: {self.poly_n}"

        if self.poly_sigma <= 0:
            return False, f"poly_sigma должен быть > 0: {self.poly_sigma}"

        if self.grid_step < 1:
            return False, f"grid_step должен быть >= 1: {self.grid_step}"

        # Проверка физических параметров
        if self.dt <= 0:
            return False, f"dt должен быть > 0: {self.dt}"

        if self.scaling_factor <= 0:
            return False, f"scaling_factor должен быть > 0: {self.scaling_factor}"

        return True, ""


class FarnebackAnalysisExecutor:
    """
    Класс для выполнения анализа оптического потока с параметрами для GUI.

    Использование:
        1. Создать экземпляр FarnebackAnalysisExecutor
        2. Задать параметры через FarnebackAnalysisParameters
        3. (Опционально) Установить callback для прогресса
        4. Вызвать execute() для запуска анализа
        5. Получить результат FarnebackResult
    """

    def __init__(self):
        """Инициализация исполнителя анализа оптического потока."""
        self.analyzer = FarnebackAnalyzer()
        self.parameters: Optional[FarnebackAnalysisParameters] = None
        self._progress_callback: Optional[Callable[[FarnebackProgress], None]] = None

        logger.info("Инициализирован FarnebackAnalysisExecutor")

    def set_parameters(self, parameters: FarnebackAnalysisParameters) -> tuple[bool, str]:
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

        # Установка параметров Farneback
        if not self.analyzer.set_farneback_config(
            pyr_scale=parameters.pyr_scale,
            levels=parameters.levels,
            winsize=parameters.winsize,
            iterations=parameters.iterations,
            poly_n=parameters.poly_n,
            poly_sigma=parameters.poly_sigma,
            min_magnitude=parameters.min_magnitude,
            max_magnitude=parameters.max_magnitude,
            grid_step=parameters.grid_step,
            dt=parameters.dt,
            scaling_factor=parameters.scaling_factor
        ):
            return False, "Не удалось установить параметры Farneback"

        logger.info("Параметры Farneback установлены")
        return True, ""

    def set_progress_callback(self, callback: Callable[[FarnebackProgress], None]) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция обратного вызова для GUI
                     Принимает FarnebackProgress с полями:
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

    def execute(self) -> FarnebackResult:
        """
        Выполнение анализа оптического потока.

        Returns:
            FarnebackResult с результатами анализа:
            - success: успешность выполнения
            - total_pairs_processed: общее количество обработанных пар
            - cam1_pairs_count: количество пар cam_1
            - cam2_pairs_count: количество пар cam_2
            - total_vectors_count: общее количество векторов
            - errors: список ошибок
            - warnings: список предупреждений
            - output_folder: путь к выходной папке
        """
        if self.parameters is None:
            logger.error("Параметры не установлены")
            return FarnebackResult(
                success=False,
                total_pairs_processed=0,
                cam1_pairs_count=0,
                cam2_pairs_count=0,
                total_vectors_count=0,
                errors=["Параметры не установлены"],
                warnings=[],
                output_folder=""
            )

        logger.info("=" * 60)
        logger.info("ЗАПУСК АНАЛИЗА ОПТИЧЕСКОГО ПОТОКА FARNEBACK")
        logger.info(f"Входная папка: {self.parameters.input_folder}")
        logger.info(f"Параметры: pyr_scale={self.parameters.pyr_scale}, "
                   f"levels={self.parameters.levels}, "
                   f"winsize={self.parameters.winsize}, "
                   f"grid_step={self.parameters.grid_step}")
        logger.info("=" * 60)

        # Выполнение анализа
        result = self.analyzer.process_all()

        logger.info("=" * 60)
        logger.info("ЗАВЕРШЕНИЕ АНАЛИЗА FARNEBACK")
        logger.info(f"Успешно: {result.success}")
        logger.info(f"Обработано пар: {result.total_pairs_processed}")
        logger.info(f"  cam_1: {result.cam1_pairs_count} пар")
        logger.info(f"  cam_2: {result.cam2_pairs_count} пар")
        logger.info(f"Всего векторов: {result.total_vectors_count}")
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
            f"Создан предпросмотр для {camera_name} (Farneback): "
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


def run_farneback_analysis(
    input_folder: str,
    # Параметры Farneback
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    grid_step: int = 10,
    # Физические параметры
    dt: float = 0.002,
    scaling_factor: float = 1.0,
    # Callback
    progress_callback: Optional[Callable] = None
) -> FarnebackResult:
    """
    Удобная функция для запуска анализа оптического потока Farneback без создания объектов.

    Args:
        input_folder: Путь к папке farneback_filtered_XXXX
        pyr_scale: Масштаб пирамиды (0-1)
        levels: Количество уровней пирамиды
        winsize: Размер окна усреднения
        iterations: Количество итераций
        poly_n: Размер окрестности (5 или 7)
        poly_sigma: Стандартное отклонение гауссиана
        grid_step: Шаг сетки для экспорта
        dt: Временной интервал между кадрами
        scaling_factor: Масштабный коэффициент
        progress_callback: Callback функция для прогресса

    Returns:
        FarnebackResult с результатами

    Example:
        >>> result = run_farneback_analysis(
        ...     input_folder="path/to/farneback_filtered_2000",
        ...     winsize=15,
        ...     grid_step=10
        ... )
        >>> print(f"Векторов: {result.total_vectors_count}")
    """
    # Создание параметров
    params = FarnebackAnalysisParameters(
        input_folder=input_folder,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        grid_step=grid_step,
        dt=dt,
        scaling_factor=scaling_factor,
        enable_progress_callback=progress_callback is not None
    )

    # Создание исполнителя
    executor = FarnebackAnalysisExecutor()

    # Установка параметров
    success, error_msg = executor.set_parameters(params)
    if not success:
        logger.error(f"Ошибка установки параметров: {error_msg}")
        return FarnebackResult(
            success=False,
            total_pairs_processed=0,
            cam1_pairs_count=0,
            cam2_pairs_count=0,
            total_vectors_count=0,
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
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ С GUI - АНАЛИЗ ОПТИЧЕСКОГО ПОТОКА")
    print("=" * 60)

    # === ШАГ 1: Задание параметров (из GUI элементов) ===
    parameters = FarnebackAnalysisParameters(
        input_folder=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\small_without_filter",
        # Параметры Farneback
        pyr_scale=0.5,
        levels=4,
        winsize=9,
        iterations=3,
        poly_n=5,
        poly_sigma=1.5,
        grid_step=30,
        # Физические параметры
        dt=0.002,
        scaling_factor=1.0,
        enable_progress_callback=True
    )

    print(f"\nМетод: Farneback")
    print(f"\nПараметры:")
    print(f"  Входная папка: {parameters.input_folder}")
    print(f"  pyr_scale: {parameters.pyr_scale}")
    print(f"  levels: {parameters.levels}")
    print(f"  winsize: {parameters.winsize}")
    print(f"  iterations: {parameters.iterations}")
    print(f"  grid_step: {parameters.grid_step}")

    # === ШАГ 2: Создание исполнителя ===
    executor = FarnebackAnalysisExecutor()

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
        print(f"  Метод: {preview['method']}")
        print(f"  Средняя магнитуда: {preview['mean_magnitude']:.3f}")
        print(f"  Макс. магнитуда: {preview['max_magnitude']:.3f}")

    # === ШАГ 6: Установка callback для прогресса (для GUI progress bar) ===
    def progress_callback(progress: FarnebackProgress):
        """Callback для обновления GUI."""
        print(
            f"  [{progress.current_camera}] "
            f"{progress.percentage:.1f}% - {progress.message}"
        )

    executor.set_progress_callback(progress_callback)

    # === ШАГ 7: Выполнение анализа ===
    print("\nЗапуск анализа Farneback...")
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
