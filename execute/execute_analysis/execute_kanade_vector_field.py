"""
Модуль выполнения визуализации векторного поля для анализа Lucas-Kanade для GUI ParticleAnalysis.

Этот модуль предоставляет готовую к использованию структуру для интеграции
с графическим интерфейсом. Все параметры четко определены и могут быть
легко привязаны к элементам GUI.

Визуализирует результаты sparse optical flow анализа (Lucas-Kanade):
- Отображение векторов смещения частиц
- Усреднение по ячейкам сетки
- Цветовое кодирование по магнитуде

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, List
import logging
import numpy as np
import csv

# Добавление пути к модулям проекта
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.farneback_vector_field import (
    FarnebackVectorFieldVisualizer,
    FarnebackVectorFieldResult,
    FlowVectorData
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class KanadeVectorFieldParameters:
    """
    Параметры визуализации векторного поля для анализа Lucas-Kanade для GUI.

    Все параметры должны устанавливаться через GUI элементы:
    - kanade_folder: путь к папке LucasKanade_XXXX (через file dialog)
    - nx, ny: разрешение сетки (через spinbox)
    - scale: масштаб стрелок (через slider или spinbox)
    - width: толщина стрелок (через slider или spinbox)
    - cmap: цветовая карта (через combobox)
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    kanade_folder: str  # Путь к папке LucasKanade_XXXX с результатами анализа

    # ПАРАМЕТРЫ СЕТКИ
    nx: int = 50  # Количество ячеек по X (горизонталь)
    ny: int = 50  # Количество ячеек по Y (вертикаль)

    # ПАРАМЕТРЫ QUIVER
    scale: float = 20  # Масштаб стрелок (меньше = длиннее стрелки)
    width: float = 0.005  # Толщина стрелок

    # ПАРАМЕТРЫ ЦВЕТОВОЙ КАРТЫ
    cmap: str = "jet"  # Название цветовой карты matplotlib
    vmin: Optional[float] = None  # Минимум для colorbar (None = авто)
    vmax: Optional[float] = None  # Максимум для colorbar (None = авто)

    # ПАРАМЕТРЫ СЕТКИ ОТОБРАЖЕНИЯ
    show_grid: bool = True  # Показывать сетку на графике
    grid_color: str = "black"  # Цвет линий сетки
    grid_alpha: float = 0.25  # Прозрачность сетки (0-1)
    grid_linewidth: float = 0.4  # Толщина линий сетки

    # ПАРАМЕТРЫ ОСЕЙ И ЗАГОЛОВКА
    xlabel: str = "X, px"  # Подпись оси X
    ylabel: str = "Y, px"  # Подпись оси Y
    figsize: Tuple[float, float] = (9, 6)  # Размер фигуры (ширина, высота) в дюймах

    # ОПЦИОНАЛЬНЫЕ ПАРАМЕТРЫ
    enable_progress_callback: bool = False  # Включить callback для прогресса

    # GUI ПОДСКАЗКИ
    nx_min: int = 5
    nx_max: int = 200
    ny_min: int = 5
    ny_max: int = 200

    scale_min: float = 1
    scale_max: float = 200
    width_min: float = 0.001
    width_max: float = 0.02

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
        kanade_path = Path(self.kanade_folder)
        if not kanade_path.exists():
            return False, f"Папка Lucas-Kanade не существует: {self.kanade_folder}"

        # Проверяем наличие суммарных CSV файлов
        cam1_lk = kanade_path / "cam_1" / "cam_1_lucas_kanade_sum.csv"
        cam2_lk = kanade_path / "cam_2" / "cam_2_lucas_kanade_sum.csv"

        has_any = cam1_lk.exists() or cam2_lk.exists()

        if not has_any:
            return False, f"Не найдены файлы cam_X_lucas_kanade_sum.csv в {self.kanade_folder}"

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


@dataclass
class KanadeVectorFieldResult:
    """Результат создания векторного поля Lucas-Kanade."""
    success: bool
    cam1_vectors_count: int
    cam2_vectors_count: int
    cam1_points_count: int  # Количество уникальных точек cam_1
    cam2_points_count: int  # Количество уникальных точек cam_2
    errors: List[str]
    output_folder: str


class KanadeVectorFieldExecutor:
    """
    Класс для выполнения визуализации векторного поля Lucas-Kanade с параметрами для GUI.

    Использование:
        1. Создать экземпляр KanadeVectorFieldExecutor
        2. Задать параметры через KanadeVectorFieldParameters
        3. Вызвать execute() для запуска создания векторных полей
        4. Получить результат KanadeVectorFieldResult
    """

    def __init__(self):
        """Инициализация исполнителя визуализации векторного поля Lucas-Kanade."""
        self.visualizer = FarnebackVectorFieldVisualizer()
        self.parameters: Optional[KanadeVectorFieldParameters] = None
        self._progress_callback: Optional[Callable] = None
        self._cancelled: bool = False

        logger.info("Инициализирован KanadeVectorFieldExecutor")

    def set_parameters(self, parameters: KanadeVectorFieldParameters) -> tuple[bool, str]:
        """
        Установка параметров визуализации векторного поля Lucas-Kanade.

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
        if not self.visualizer.set_farneback_folder(parameters.kanade_folder):
            return False, "Не удалось установить папку Lucas-Kanade"

        # Принудительно устанавливаем метод lucas_kanade
        if not self.visualizer.set_method("lucas_kanade"):
            return False, "Не удалось установить метод Lucas-Kanade"

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
        """
        self._progress_callback = callback
        logger.info("Установлен callback для прогресса")

    def cancel(self) -> None:
        """
        Отмена выполнения обработки.

        Этот метод следует вызвать из GUI при нажатии кнопки "Отмена".
        """
        self._cancelled = True
        logger.info("Запрошена отмена обработки")

    def _count_unique_points(self, csv_path: Path) -> int:
        """
        Подсчёт уникальных точек в CSV файле Lucas-Kanade.

        Args:
            csv_path: Путь к CSV файлу

        Returns:
            Количество уникальных точек
        """
        try:
            points = set()
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                for row in reader:
                    if 'X0' in row and 'Y0' in row:
                        x = float(row['X0'])
                        y = float(row['Y0'])
                        points.add((round(x, 2), round(y, 2)))
            return len(points)
        except Exception as e:
            logger.warning(f"Ошибка подсчёта точек: {e}")
            return 0

    def execute(self) -> KanadeVectorFieldResult:
        """
        Выполнение визуализации векторного поля Lucas-Kanade.

        Returns:
            KanadeVectorFieldResult с результатами создания
        """
        if self.parameters is None:
            logger.error("Параметры не установлены")
            return KanadeVectorFieldResult(
                success=False,
                cam1_vectors_count=0,
                cam2_vectors_count=0,
                cam1_points_count=0,
                cam2_points_count=0,
                errors=["Параметры не установлены"],
                output_folder=""
            )

        logger.info("=" * 60)
        logger.info("ЗАПУСК ВИЗУАЛИЗАЦИИ ВЕКТОРНОГО ПОЛЯ LUCAS-KANADE")
        logger.info(f"Папка результатов: {self.parameters.kanade_folder}")
        logger.info(f"Сетка: {self.parameters.nx}x{self.parameters.ny}")
        logger.info(f"Масштаб: {self.parameters.scale}")
        logger.info(f"Цветовая карта: {self.parameters.cmap}")
        logger.info("=" * 60)

        # Выполнение обработки через базовый визуализатор
        fb_result = self.visualizer.process_all()

        # Подсчёт уникальных точек
        kanade_path = Path(self.parameters.kanade_folder)
        cam1_csv = kanade_path / "cam_1" / "cam_1_lucas_kanade_sum.csv"
        cam2_csv = kanade_path / "cam_2" / "cam_2_lucas_kanade_sum.csv"

        cam1_points = self._count_unique_points(cam1_csv) if cam1_csv.exists() else 0
        cam2_points = self._count_unique_points(cam2_csv) if cam2_csv.exists() else 0

        logger.info("=" * 60)
        logger.info("ЗАВЕРШЕНИЕ ВИЗУАЛИЗАЦИИ ВЕКТОРНОГО ПОЛЯ LUCAS-KANADE")
        logger.info(f"Успешно: {fb_result.success}")
        logger.info(f"cam_1: {fb_result.cam1_vectors_count} векторов, {cam1_points} уникальных точек")
        logger.info(f"cam_2: {fb_result.cam2_vectors_count} векторов, {cam2_points} уникальных точек")
        logger.info(f"Ошибок: {len(fb_result.errors)}")
        logger.info(f"Выходная папка: {fb_result.output_folder}")
        logger.info("=" * 60)

        return KanadeVectorFieldResult(
            success=fb_result.success,
            cam1_vectors_count=fb_result.cam1_vectors_count,
            cam2_vectors_count=fb_result.cam2_vectors_count,
            cam1_points_count=cam1_points,
            cam2_points_count=cam2_points,
            errors=fb_result.errors,
            output_folder=fb_result.output_folder
        )

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
            - mean_angle: средний угол направления
        """
        if camera_name not in ["cam_1", "cam_2"]:
            logger.error(f"Неверное название камеры: {camera_name}")
            return None

        stats = self.visualizer.get_statistics(camera_name)

        if stats is None:
            logger.warning(f"Не удалось получить статистику для {camera_name}")
        else:
            # Добавляем информацию о методе
            stats['method'] = 'lucas_kanade'
            logger.info(f"Получена статистика для {camera_name}: {stats.get('vectors_count', 0)} векторов")

        return stats


def run_kanade_vector_field(
    kanade_folder: str,
    nx: int = 50,
    ny: int = 50,
    scale: float = 20,
    width: float = 0.005,
    cmap: str = "jet"
) -> KanadeVectorFieldResult:
    """
    Удобная функция для запуска визуализации векторного поля Lucas-Kanade без создания объектов.

    Args:
        kanade_folder: Путь к папке LucasKanade_XXXX
        nx: Количество ячеек сетки по X (по умолчанию 50)
        ny: Количество ячеек сетки по Y (по умолчанию 50)
        scale: Масштаб стрелок (по умолчанию 20)
        width: Толщина стрелок (по умолчанию 0.005)
        cmap: Цветовая карта (по умолчанию "jet")

    Returns:
        KanadeVectorFieldResult с результатами

    Example:
        >>> result = run_kanade_vector_field(
        ...     kanade_folder="path/to/LucasKanade_2000",
        ...     nx=50,
        ...     ny=50
        ... )
        >>> print(f"cam_1 векторов: {result.cam1_vectors_count}")
        >>> print(f"cam_1 уникальных точек: {result.cam1_points_count}")
    """
    # Создание параметров
    params = KanadeVectorFieldParameters(
        kanade_folder=kanade_folder,
        nx=nx,
        ny=ny,
        scale=scale,
        width=width,
        cmap=cmap
    )

    # Создание исполнителя
    executor = KanadeVectorFieldExecutor()

    # Установка параметров
    success, error_msg = executor.set_parameters(params)
    if not success:
        logger.error(f"Ошибка установки параметров: {error_msg}")
        return KanadeVectorFieldResult(
            success=False,
            cam1_vectors_count=0,
            cam2_vectors_count=0,
            cam1_points_count=0,
            cam2_points_count=0,
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
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ С GUI - LUCAS-KANADE ВЕКТОРНОЕ ПОЛЕ")
    print("=" * 60)

    # === ШАГ 1: Задание параметров (из GUI элементов) ===
    parameters = KanadeVectorFieldParameters(
        kanade_folder=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\LucasKanade_2000",
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
    print(f"  Папка результатов: {parameters.kanade_folder}")
    print(f"  Сетка: {parameters.nx}x{parameters.ny}")
    print(f"  Масштаб: {parameters.scale}")
    print(f"  Толщина стрелок: {parameters.width}")
    print(f"  Цветовая карта: {parameters.cmap}")

    # === ШАГ 2: Создание исполнителя ===
    executor = KanadeVectorFieldExecutor()

    # === ШАГ 3: Валидация и установка параметров ===
    success, error_msg = executor.set_parameters(parameters)
    if not success:
        print(f"\nОШИБКА: {error_msg}")
        return

    print("\n[OK] Параметры валидны")

    # === ШАГ 4: Получение статистики ===
    print("\nПолучение статистики...")
    cam1_stats = executor.get_statistics("cam_1")
    if cam1_stats:
        print(f"\nСтатистика cam_1:")
        print(f"  Всего векторов: {cam1_stats['vectors_count']}")
        if cam1_stats.get('vectors_with_velocity'):
            print(f"  Векторов с скоростью: {cam1_stats['vectors_with_velocity']}")
            print(f"  Средняя магнитуда: {cam1_stats.get('mean_magnitude', 0):.4f}")
            print(f"  Макс магнитуда: {cam1_stats.get('max_magnitude', 0):.4f}")
            print(f"  Мин магнитуда: {cam1_stats.get('min_magnitude', 0):.4f}")
            print(f"  Среднее U: {cam1_stats.get('mean_u', 0):.4f}")
            print(f"  Среднее V: {cam1_stats.get('mean_v', 0):.4f}")
            print(f"  Средний угол: {cam1_stats.get('mean_angle', 0):.1f} deg")

    cam2_stats = executor.get_statistics("cam_2")
    if cam2_stats:
        print(f"\nСтатистика cam_2:")
        print(f"  Всего векторов: {cam2_stats['vectors_count']}")
        if cam2_stats.get('vectors_with_velocity'):
            print(f"  Векторов с скоростью: {cam2_stats['vectors_with_velocity']}")
            print(f"  Средняя магнитуда: {cam2_stats.get('mean_magnitude', 0):.4f}")

    # === ШАГ 5: Выполнение визуализации ===
    print("\nЗапуск визуализации векторного поля Lucas-Kanade...")
    result = executor.execute()

    # === ШАГ 6: Обработка результата ===
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Успешно: {result.success}")
    print(f"cam_1: {result.cam1_vectors_count} векторов, {result.cam1_points_count} уникальных точек")
    print(f"cam_2: {result.cam2_vectors_count} векторов, {result.cam2_points_count} уникальных точек")
    print(f"Ошибок: {len(result.errors)}")
    print(f"Выходная папка: {result.output_folder}")

    if result.errors:
        print("\nОшибки:")
        for error in result.errors[:5]:
            print(f"  - {error}")

    print("=" * 60)


if __name__ == "__main__":
    # При запуске модуля напрямую - показать пример использования
    example_gui_usage()
