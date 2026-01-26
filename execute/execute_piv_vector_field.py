"""
Модуль выполнения визуализации векторного поля для PIV анализа для GUI ParticleAnalysis.

Этот модуль предоставляет готовую к использованию структуру для интеграции
с графическим интерфейсом. Все параметры четко определены и могут быть
легко привязаны к элементам GUI.

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import logging
import numpy as np

# Добавление пути к модулям проекта
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.piv_vector_field import (
    PIVVectorFieldVisualizer,
    PIVVectorFieldResult
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PIVVectorFieldParameters:
    """
    Параметры визуализации векторного поля для PIV анализа для GUI.

    Все параметры должны устанавливаться через GUI элементы:
    - piv_folder: путь к папке PIV_XXXX (через file dialog)
    - nx, ny: разрешение сетки (через spinbox)
    - scale: масштаб стрелок (через slider или spinbox)
    - width: толщина стрелок (через slider или spinbox)
    - cmap: цветовая карта (через combobox)
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    piv_folder: str  # Путь к папке PIV_XXXX с cam_X/cam_X_vectors_sum.csv файлами

    # ПАРАМЕТРЫ СЕТКИ
    nx: int = 73  # Количество ячеек по X (горизонталь)
    ny: int = 50  # Количество ячеек по Y (вертикаль)

    # ПАРАМЕТРЫ QUIVER
    scale: float = 20  # Масштаб стрелок (меньше = длиннее стрелки)
    width: float = 0.005  # Толщина стрелок

    # ПАРАМЕТРЫ ЦВЕТОВОЙ КАРТЫ
    cmap: str = "jet"  # Название цветовой карты matplotlib
    vmin: Optional[float] = None  # Минимум для colorbar (None = авто)
    vmax: Optional[float] = None  # Максимум для colorbar (None = авто)

    # ПАРАМЕТРЫ СЕТКИ
    show_grid: bool = True  # Показывать сетку на графике
    grid_color: str = "black"  # Цвет линий сетки
    grid_alpha: float = 0.25  # Прозрачность сетки (0-1)
    grid_linewidth: float = 0.4  # Толщина линий сетки

    # ПАРАМЕТРЫ ОСЕЙ И ЗАГОЛОВКА
    xlabel: str = "X, px"  # Подпись оси X
    ylabel: str = "Y, px"  # Подпись оси Y
    figsize: Tuple[float, float] = (9, 6)  # Размер фигуры (ширина, высота) в дюймах

    # ОПЦИОНАЛЬНЫЕ ПАРАМЕТРЫ
    enable_progress_callback: bool = False  # Включить callback для прогресса (зарезервировано)

    # GUI ПОДСКАЗКИ (не используются в обработке, только для GUI)
    nx_min: int = 10  # Минимальное количество ячеек по X
    nx_max: int = 200  # Максимальное количество ячеек по X
    ny_min: int = 10  # Минимальное количество ячеек по Y
    ny_max: int = 200  # Максимальное количество ячеек по Y
    scale_min: float = 1  # Минимальный масштаб
    scale_max: float = 200  # Максимальный масштаб
    width_min: float = 0.001  # Минимальная толщина стрелок
    width_max: float = 0.02  # Максимальная толщина стрелок

    # Доступные цветовые карты для combobox
    available_cmaps: Tuple[str, ...] = (
        "jet", "viridis", "plasma", "inferno", "magma", "cividis",
        "twilight", "turbo", "hot", "cool", "spring", "summer",
        "autumn", "winter", "RdYlBu", "RdYlGn", "Spectral"
    )

    def validate(self) -> tuple[bool, str]:
        """
        Валидация параметров.

        Returns:
            tuple[bool, str]: (success, error_message)
        """
        # Проверка входной папки
        piv_path = Path(self.piv_folder)
        if not piv_path.exists():
            return False, f"Папка PIV не существует: {self.piv_folder}"

        cam1_sum = piv_path / "cam_1" / "cam_1_vectors_sum.csv"
        cam2_sum = piv_path / "cam_2" / "cam_2_vectors_sum.csv"

        if not cam1_sum.exists() and not cam2_sum.exists():
            return False, f"Не найдены файлы cam_X/cam_X_vectors_sum.csv в {self.piv_folder}"

        # Проверка параметров сетки
        if not (self.nx_min <= self.nx <= self.nx_max):
            return False, f"nx должно быть в диапазоне [{self.nx_min}, {self.nx_max}]"
        if not (self.ny_min <= self.ny <= self.ny_max):
            return False, f"ny должно быть в диапазоне [{self.ny_min}, {self.ny_max}]"

        # Проверка параметров quiver
        if not (self.scale_min <= self.scale <= self.scale_max):
            return False, f"scale должен быть в диапазоне [{self.scale_min}, {self.scale_max}]"
        if not (self.width_min <= self.width <= self.width_max):
            return False, f"width должна быть в диапазоне [{self.width_min}, {self.width_max}]"

        # Проверка цветовой карты
        if self.cmap not in self.available_cmaps:
            return False, f"Неизвестная цветовая карта: {self.cmap}"

        # Проверка vmin/vmax
        if self.vmin is not None and self.vmax is not None:
            if self.vmin >= self.vmax:
                return False, "vmin должен быть меньше vmax"

        # Проверка параметров сетки
        if not (0 <= self.grid_alpha <= 1):
            return False, "grid_alpha должна быть в диапазоне [0, 1]"

        return True, ""


class PIVVectorFieldExecutor:
    """
    Класс для выполнения визуализации векторного поля PIV с параметрами для GUI.

    Использование:
        1. Создать экземпляр PIVVectorFieldExecutor
        2. Задать параметры через PIVVectorFieldParameters
        3. (Опционально) Установить callback для прогресса
        4. Вызвать execute() для запуска создания векторных полей
        5. Получить результат PIVVectorFieldResult
    """

    def __init__(self):
        """Инициализация исполнителя визуализации векторного поля PIV."""
        self.visualizer = PIVVectorFieldVisualizer()
        self.parameters: Optional[PIVVectorFieldParameters] = None
        self._progress_callback: Optional[Callable] = None
        self._cancelled: bool = False

        logger.info("Инициализирован PIVVectorFieldExecutor")

    def set_parameters(self, parameters: PIVVectorFieldParameters) -> tuple[bool, str]:
        """
        Установка параметров визуализации векторного поля PIV.

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
        if not self.visualizer.set_piv_folder(parameters.piv_folder):
            return False, "Не удалось установить папку PIV"

        self.visualizer.set_config(
            nx=parameters.nx,
            ny=parameters.ny,
            scale=parameters.scale,
            width=parameters.width,
            cmap=parameters.cmap,
            vmin=parameters.vmin,
            vmax=parameters.vmax,
            show_grid=parameters.show_grid,
            xlabel=parameters.xlabel,
            ylabel=parameters.ylabel,
            figsize=parameters.figsize
        )

        logger.info(f"Параметры установлены: nx={parameters.nx}, ny={parameters.ny}, "
                   f"scale={parameters.scale}, cmap={parameters.cmap}")
        return True, ""

    def set_progress_callback(self, callback: Callable) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция обратного вызова для GUI
                     (в текущей версии не используется, зарезервировано для будущего)
        """
        self._progress_callback = callback
        logger.info("Установлен callback для прогресса (зарезервировано)")

    def cancel(self) -> None:
        """
        Отмена выполнения обработки.

        Этот метод следует вызвать из GUI при нажатии кнопки "Отмена".
        (в текущей версии не используется, зарезервировано для будущего)
        """
        self._cancelled = True
        logger.info("Запрошена отмена обработки")

    def execute(self) -> PIVVectorFieldResult:
        """
        Выполнение визуализации векторного поля PIV.

        Returns:
            PIVVectorFieldResult с результатами создания:
            - success: успешность выполнения
            - cam1_vectors_count: количество векторов cam_1
            - cam2_vectors_count: количество векторов cam_2
            - errors: список ошибок
            - output_folder: путь к выходной папке
        """
        if self.parameters is None:
            logger.error("Параметры не установлены")
            return PIVVectorFieldResult(
                success=False,
                cam1_vectors_count=0,
                cam2_vectors_count=0,
                errors=["Параметры не установлены"],
                output_folder=""
            )

        logger.info("=" * 60)
        logger.info("ЗАПУСК ВИЗУАЛИЗАЦИИ ВЕКТОРНОГО ПОЛЯ PIV")
        logger.info(f"PIV папка: {self.parameters.piv_folder}")
        logger.info(f"Сетка: {self.parameters.nx}×{self.parameters.ny}")
        logger.info(f"Масштаб: {self.parameters.scale}")
        logger.info(f"Цветовая карта: {self.parameters.cmap}")
        logger.info("=" * 60)

        # Выполнение обработки
        result = self.visualizer.process_all()

        logger.info("=" * 60)
        logger.info("ЗАВЕРШЕНИЕ ВИЗУАЛИЗАЦИИ ВЕКТОРНОГО ПОЛЯ PIV")
        logger.info(f"Успешно: {result.success}")
        logger.info(f"cam_1 векторов: {result.cam1_vectors_count}")
        logger.info(f"cam_2 векторов: {result.cam2_vectors_count}")
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

    def get_preview(self, camera_name: str) -> Optional[np.ndarray]:
        """
        Получение предварительного просмотра векторного поля для одной камеры.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Изображение векторного поля (BGR) или None
        """
        if camera_name not in ["cam_1", "cam_2"]:
            logger.error(f"Неверное название камеры: {camera_name}")
            return None

        preview = self.visualizer.get_preview(camera_name)

        if preview is None:
            logger.warning(f"Не удалось создать предпросмотр для {camera_name}")
        else:
            logger.info(f"Создан предпросмотр для {camera_name}")

        return preview

    def get_statistics(self, camera_name: str) -> Optional[dict]:
        """
        Получение статистики векторов для камеры.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Словарь со статистикой:
            - vectors_count: общее количество векторов
            - vectors_with_velocity: векторы с ненулевой скоростью
            - mean_magnitude: средняя магнитуда вектора
            - max_magnitude: максимальная магнитуда
            - min_magnitude: минимальная магнитуда
            - std_magnitude: стандартное отклонение магнитуды
            - mean_u, mean_v: среднее значение компонент скорости
            - std_u, std_v: стандартное отклонение компонент
        """
        if camera_name not in ["cam_1", "cam_2"]:
            logger.error(f"Неверное название камеры: {camera_name}")
            return None

        stats = self.visualizer.get_statistics(camera_name)

        if stats is None:
            logger.warning(f"Не удалось получить статистику для {camera_name}")
        else:
            logger.info(f"Получена статистика для {camera_name}: {stats.get('vectors_count', 0)} векторов")

        return stats


def run_piv_vector_field(piv_folder: str, nx: int = 73, ny: int = 50,
                          scale: float = 20, width: float = 0.005,
                          cmap: str = "jet") -> PIVVectorFieldResult:
    """
    Удобная функция для запуска визуализации векторного поля PIV без создания объектов.

    Args:
        piv_folder: Путь к папке PIV_XXXX
        nx: Количество ячеек сетки по X (по умолчанию 73)
        ny: Количество ячеек сетки по Y (по умолчанию 50)
        scale: Масштаб стрелок (по умолчанию 20)
        width: Толщина стрелок (по умолчанию 0.005)
        cmap: Цветовая карта (по умолчанию "jet")

    Returns:
        PIVVectorFieldResult с результатами

    Example:
        >>> result = run_piv_vector_field(
        ...     piv_folder="path/to/PIV_3240",
        ...     nx=73,
        ...     ny=50
        ... )
        >>> print(f"cam_1 векторов: {result.cam1_vectors_count}")
    """
    # Создание параметров
    params = PIVVectorFieldParameters(
        piv_folder=piv_folder,
        nx=nx,
        ny=ny,
        scale=scale,
        width=width,
        cmap=cmap
    )

    # Создание исполнителя
    executor = PIVVectorFieldExecutor()

    # Установка параметров
    success, error_msg = executor.set_parameters(params)
    if not success:
        logger.error(f"Ошибка установки параметров: {error_msg}")
        return PIVVectorFieldResult(
            success=False,
            cam1_vectors_count=0,
            cam2_vectors_count=0,
            errors=[error_msg],
            output_folder=""
        )

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
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ С GUI - PIV ВЕКТОРНОЕ ПОЛЕ")
    print("=" * 60)

    # === ШАГ 1: Задание параметров (из GUI элементов) ===
    # ВНИМАНИЕ: Замените путь на реальный путь к папке PIV_XXXX
    parameters = PIVVectorFieldParameters(
        piv_folder=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\PIV_3240",  # Путь к папке с результатами PIV анализа
        nx=73,
        ny=50,
        scale=200,
        width=0.005,
        cmap="jet",
        show_grid=True,
        xlabel="X, px",
        ylabel="Y, px"
    )

    print(f"\nПараметры:")
    print(f"  PIV папка: {parameters.piv_folder}")
    print(f"  Сетка: {parameters.nx}×{parameters.ny}")
    print(f"  Масштаб: {parameters.scale}")
    print(f"  Толщина стрелок: {parameters.width}")
    print(f"  Цветовая карта: {parameters.cmap}")

    # === ШАГ 2: Создание исполнителя ===
    executor = PIVVectorFieldExecutor()

    # === ШАГ 3: Валидация и установка параметров ===
    success, error_msg = executor.set_parameters(parameters)
    if not success:
        print(f"\nОШИБКА: {error_msg}")
        return

    print("\n✓ Параметры валидны")

    # === ШАГ 4: Получение статистики ===
    print("\nПолучение статистики...")
    cam1_stats = executor.get_statistics("cam_1")
    if cam1_stats:
        print(f"\nСтатистика cam_1:")
        print(f"  Всего векторов: {cam1_stats['vectors_count']}")
        if cam1_stats.get('vectors_with_velocity'):
            print(f"  Векторов с скоростью: {cam1_stats['vectors_with_velocity']}")
            print(f"  Средняя магнитуда: {cam1_stats.get('mean_magnitude', 0):.2f}")
            print(f"  Макс магнитуда: {cam1_stats.get('max_magnitude', 0):.2f}")
            print(f"  Мин магнитуда: {cam1_stats.get('min_magnitude', 0):.2f}")
            print(f"  Среднее U: {cam1_stats.get('mean_u', 0):.2f}")
            print(f"  Среднее V: {cam1_stats.get('mean_v', 0):.2f}")

    cam2_stats = executor.get_statistics("cam_2")
    if cam2_stats:
        print(f"\nСтатистика cam_2:")
        print(f"  Всего векторов: {cam2_stats['vectors_count']}")
        if cam2_stats.get('vectors_with_velocity'):
            print(f"  Векторов с скоростью: {cam2_stats['vectors_with_velocity']}")
            print(f"  Средняя магнитуда: {cam2_stats.get('mean_magnitude', 0):.2f}")

    # === ШАГ 5: Выполнение визуализации ===
    print("\nЗапуск визуализации векторного поля PIV...")
    result = executor.execute()

    # === ШАГ 6: Обработка результата ===
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Успешно: {result.success}")
    print(f"cam_1 векторов: {result.cam1_vectors_count}")
    print(f"cam_2 векторов: {result.cam2_vectors_count}")
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
