"""
Модуль выполнения анализа оптического потока Farneback для GUI ParticleAnalysis.

Этот модуль предоставляет готовую к использованию структуру для интеграции
с графическим интерфейсом. Все параметры четко определены и могут быть
легко привязаны к элементам GUI.

Анализ оптического потока:
- Farneback (dense) - вычисление потока для каждого пикселя
- Lucas-Kanade (sparse) - вычисление потока на выбранных точках (частицах)
- Экспорт результатов в CSV формат

Входные данные: изображения после filter_farneback_kanade (8-bit PNG, пары _a и _b)

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
    - use_lucas_kanade: использовать Lucas-Kanade вместо Farneback (checkbox)
    - Параметры Farneback (через spinbox/slider)
    - Параметры Lucas-Kanade (через spinbox/slider)
    - enable_progress_callback: включить обратную связь прогресса (checkbox)
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    input_folder: str  # Путь к папке farneback_filtered_XXXX с изображениями

    # ВЫБОР МЕТОДА
    use_lucas_kanade: bool = False  # True = Lucas-Kanade (sparse), False = Farneback (dense)

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

    # ПАРАМЕТРЫ LUCAS-KANADE
    max_corners: int = 500  # Максимальное количество точек для трекинга
    quality_level: float = 0.01  # Минимальное качество угла (0-1)
    min_distance: int = 10  # Минимальное расстояние между точками
    lk_win_size: int = 21  # Размер окна для вычисления потока (нечётное)
    max_level: int = 2  # Количество уровней пирамиды

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

    # GUI ПОДСКАЗКИ - LUCAS-KANADE
    max_corners_min: int = 50
    max_corners_max: int = 5000
    max_corners_default: int = 500

    quality_level_min: float = 0.001
    quality_level_max: float = 0.1
    quality_level_default: float = 0.01
    quality_level_step: float = 0.001

    min_distance_min: int = 1
    min_distance_max: int = 50
    min_distance_default: int = 10

    lk_win_size_options: tuple = (5, 7, 11, 15, 21, 31)
    lk_win_size_default: int = 21

    max_level_min: int = 0
    max_level_max: int = 5
    max_level_default: int = 2

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

        # Проверка параметров Lucas-Kanade
        if self.max_corners < 1:
            return False, f"max_corners должен быть >= 1: {self.max_corners}"

        if not (0 < self.quality_level <= 1):
            return False, f"quality_level должен быть в диапазоне (0, 1]: {self.quality_level}"

        if self.min_distance < 1:
            return False, f"min_distance должен быть >= 1: {self.min_distance}"

        if self.lk_win_size < 3 or self.lk_win_size % 2 == 0:
            return False, f"lk_win_size должен быть нечётным и >= 3: {self.lk_win_size}"

        if self.max_level < 0:
            return False, f"max_level должен быть >= 0: {self.max_level}"

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

        # Установка параметров Lucas-Kanade
        if not self.analyzer.set_lucas_kanade_config(
            max_corners=parameters.max_corners,
            quality_level=parameters.quality_level,
            min_distance=parameters.min_distance,
            win_size=parameters.lk_win_size,
            max_level=parameters.max_level,
            dt=parameters.dt,
            scaling_factor=parameters.scaling_factor
        ):
            return False, "Не удалось установить параметры Lucas-Kanade"

        method = "Lucas-Kanade" if parameters.use_lucas_kanade else "Farneback"
        logger.info(f"Параметры установлены. Метод: {method}")
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

        method = "Lucas-Kanade" if self.parameters.use_lucas_kanade else "Farneback"

        logger.info("=" * 60)
        logger.info(f"ЗАПУСК АНАЛИЗА ОПТИЧЕСКОГО ПОТОКА ({method})")
        logger.info(f"Входная папка: {self.parameters.input_folder}")

        if self.parameters.use_lucas_kanade:
            logger.info(f"Параметры LK: max_corners={self.parameters.max_corners}, "
                       f"quality_level={self.parameters.quality_level}, "
                       f"win_size={self.parameters.lk_win_size}")
        else:
            logger.info(f"Параметры FB: pyr_scale={self.parameters.pyr_scale}, "
                       f"levels={self.parameters.levels}, "
                       f"winsize={self.parameters.winsize}, "
                       f"grid_step={self.parameters.grid_step}")

        logger.info("=" * 60)

        # Выполнение анализа
        result = self.analyzer.process_all(use_lucas_kanade=self.parameters.use_lucas_kanade)

        logger.info("=" * 60)
        logger.info(f"ЗАВЕРШЕНИЕ АНАЛИЗА ({method})")
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

        preview = self.analyzer.get_preview(
            camera_name,
            pair_index,
            use_lucas_kanade=self.parameters.use_lucas_kanade
        )

        if preview is None:
            logger.warning(f"Не удалось создать предпросмотр для {camera_name}")
            return None

        method = preview.get('method', 'unknown')
        logger.info(
            f"Создан предпросмотр для {camera_name} ({method}): "
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
    use_lucas_kanade: bool = False,
    # Параметры Farneback
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    grid_step: int = 10,
    # Параметры Lucas-Kanade
    max_corners: int = 500,
    quality_level: float = 0.01,
    min_distance: int = 10,
    lk_win_size: int = 21,
    max_level: int = 2,
    # Физические параметры
    dt: float = 0.002,
    scaling_factor: float = 1.0,
    # Callback
    progress_callback: Optional[Callable] = None
) -> FarnebackResult:
    """
    Удобная функция для запуска анализа оптического потока без создания объектов.

    Args:
        input_folder: Путь к папке farneback_filtered_XXXX
        use_lucas_kanade: Использовать Lucas-Kanade вместо Farneback
        pyr_scale: Масштаб пирамиды (0-1)
        levels: Количество уровней пирамиды
        winsize: Размер окна усреднения
        iterations: Количество итераций
        poly_n: Размер окрестности (5 или 7)
        poly_sigma: Стандартное отклонение гауссиана
        grid_step: Шаг сетки для экспорта
        max_corners: Максимальное количество точек (LK)
        quality_level: Минимальное качество угла (LK)
        min_distance: Минимальное расстояние между точками (LK)
        lk_win_size: Размер окна для LK
        max_level: Количество уровней пирамиды (LK)
        dt: Временной интервал между кадрами
        scaling_factor: Масштабный коэффициент
        progress_callback: Callback функция для прогресса

    Returns:
        FarnebackResult с результатами

    Example:
        >>> result = run_farneback_analysis(
        ...     input_folder="path/to/farneback_filtered_2000",
        ...     use_lucas_kanade=False,
        ...     winsize=15,
        ...     grid_step=10
        ... )
        >>> print(f"Векторов: {result.total_vectors_count}")
    """
    # Создание параметров
    params = FarnebackAnalysisParameters(
        input_folder=input_folder,
        use_lucas_kanade=use_lucas_kanade,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        grid_step=grid_step,
        max_corners=max_corners,
        quality_level=quality_level,
        min_distance=min_distance,
        lk_win_size=lk_win_size,
        max_level=max_level,
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
        input_folder=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\farneback_filtered_2000",
        use_lucas_kanade=False,  # Farneback (dense)
        # Параметры Farneback
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        grid_step=10,
        # Физические параметры
        dt=0.002,
        scaling_factor=1.0,
        enable_progress_callback=True
    )

    method = "Lucas-Kanade" if parameters.use_lucas_kanade else "Farneback"
    print(f"\nМетод: {method}")
    print(f"\nПараметры:")
    print(f"  Входная папка: {parameters.input_folder}")

    if not parameters.use_lucas_kanade:
        print(f"  pyr_scale: {parameters.pyr_scale}")
        print(f"  levels: {parameters.levels}")
        print(f"  winsize: {parameters.winsize}")
        print(f"  iterations: {parameters.iterations}")
        print(f"  grid_step: {parameters.grid_step}")
    else:
        print(f"  max_corners: {parameters.max_corners}")
        print(f"  quality_level: {parameters.quality_level}")
        print(f"  lk_win_size: {parameters.lk_win_size}")

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

        if preview['method'] == 'lucas_kanade':
            print(f"  Найдено точек: {preview['total_detected']}")
            print(f"  Отслежено точек: {preview['total_tracked']}")

    # === ШАГ 6: Установка callback для прогресса (для GUI progress bar) ===
    def progress_callback(progress: FarnebackProgress):
        """Callback для обновления GUI."""
        print(
            f"  [{progress.current_camera}] "
            f"{progress.percentage:.1f}% - {progress.message}"
        )

    executor.set_progress_callback(progress_callback)

    # === ШАГ 7: Выполнение анализа ===
    print(f"\nЗапуск анализа ({method})...")
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
