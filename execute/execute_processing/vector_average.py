"""
Модуль усреднения векторов по ячейкам сетки для GUI ParticleAnalysis.

Этот модуль позволяет усреднять векторы оптического потока по ячейкам
регулярной сетки, заданной на плоскости.

Входной формат CSV (разделитель ;):
    X0;Y0;X1;Y1;U;V;Magnitude;Angle
    или минимально:
    X0;Y0;U;V

Выходной формат CSV:
    X_center;Y_center;U_avg;V_avg;count
    где:
    - X_center, Y_center - координаты центра ячейки
    - U_avg, V_avg - усреднённые значения компонент скорости
    - count - количество точек в ячейке
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import logging
import csv
import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VectorAverageResult:
    """Результат усреднения векторов."""
    success: bool
    input_vectors: int  # Количество векторов на входе
    output_cells: int  # Количество непустых ячеек на выходе
    empty_cells: int  # Количество пустых ячеек
    total_cells: int  # Общее количество ячеек в сетке
    grid_size: Tuple[int, int]  # Размер сетки (nx, ny)
    min_points_per_cell: int  # Минимальное количество точек в непустой ячейке
    max_points_per_cell: int  # Максимальное количество точек в ячейке
    avg_points_per_cell: float  # Среднее количество точек на непустую ячейку
    errors: List[str]
    output_file: str


@dataclass
class VectorAverageParameters:
    """
    Параметры усреднения векторов для GUI.

    Все параметры должны устанавливаться через GUI элементы:
    - input_file: путь к CSV файлу (через file dialog)
    - plane_width, plane_height: размер плоскости (через spinbox)
    - cell_width, cell_height: размер ячейки (через spinbox)
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    input_file: str  # Путь к CSV файлу

    # ПАРАМЕТРЫ ПЛОСКОСТИ
    plane_width: float  # Ширина плоскости
    plane_height: float  # Высота плоскости

    # ПАРАМЕТРЫ СЕТКИ
    cell_width: float  # Ширина ячейки
    cell_height: float  # Высота ячейки

    # ПАРАМЕТРЫ НАЧАЛА КООРДИНАТ (опционально)
    origin_x: float = 0.0  # X координата начала плоскости
    origin_y: float = 0.0  # Y координата начала плоскости

    # ПАРАМЕТРЫ ФИЛЬТРАЦИИ
    min_points_in_cell: int = 1  # Минимальное количество точек для записи ячейки

    # ПАРАМЕТРЫ ВЫХОДА
    output_folder: Optional[str] = None  # Папка для выходного файла (None = та же папка)
    suffix: str = "_averaged"  # Суффикс для выходного файла

    # КОЛОНКИ ДЛЯ КООРДИНАТ (по умолчанию X0, Y0)
    x_column: str = "X0"  # Имя колонки для X координаты
    y_column: str = "Y0"  # Имя колонки для Y координаты

    def validate(self) -> Tuple[bool, str]:
        """
        Валидация параметров.

        Returns:
            tuple[bool, str]: (success, error_message)
        """
        # Проверка входного файла
        input_path = Path(self.input_file)
        if not input_path.exists():
            return False, f"Файл не существует: {self.input_file}"

        if not input_path.is_file():
            return False, f"Путь не является файлом: {self.input_file}"

        if input_path.suffix.lower() != '.csv':
            return False, f"Файл должен иметь расширение .csv: {self.input_file}"

        # Проверка размеров плоскости
        if self.plane_width <= 0:
            return False, f"Ширина плоскости должна быть > 0: {self.plane_width}"

        if self.plane_height <= 0:
            return False, f"Высота плоскости должна быть > 0: {self.plane_height}"

        # Проверка размеров ячейки
        if self.cell_width <= 0:
            return False, f"Ширина ячейки должна быть > 0: {self.cell_width}"

        if self.cell_height <= 0:
            return False, f"Высота ячейки должна быть > 0: {self.cell_height}"

        if self.cell_width > self.plane_width:
            return False, f"Ширина ячейки ({self.cell_width}) больше ширины плоскости ({self.plane_width})"

        if self.cell_height > self.plane_height:
            return False, f"Высота ячейки ({self.cell_height}) больше высоты плоскости ({self.plane_height})"

        # Проверка минимального количества точек
        if self.min_points_in_cell < 1:
            return False, f"Минимальное количество точек должно быть >= 1: {self.min_points_in_cell}"

        # Проверка выходной папки
        if self.output_folder:
            output_path = Path(self.output_folder)
            if output_path.exists() and not output_path.is_dir():
                return False, f"Выходной путь не является папкой: {self.output_folder}"

        return True, ""

    def get_grid_info(self) -> Dict[str, Any]:
        """Получение информации о сетке."""
        nx = int(np.ceil(self.plane_width / self.cell_width))
        ny = int(np.ceil(self.plane_height / self.cell_height))
        return {
            'nx': nx,
            'ny': ny,
            'total_cells': nx * ny,
            'cell_width': self.cell_width,
            'cell_height': self.cell_height,
            'plane_width': self.plane_width,
            'plane_height': self.plane_height
        }


class VectorAverageExecutor:
    """
    Исполнитель усреднения векторов по ячейкам сетки.

    Использование:
        1. Создать экземпляр VectorAverageExecutor
        2. Задать параметры через VectorAverageParameters
        3. Вызвать execute() для запуска усреднения
        4. Получить результат VectorAverageResult
    """

    def __init__(self):
        """Инициализация исполнителя усреднения векторов."""
        self.parameters: Optional[VectorAverageParameters] = None
        logger.info("Инициализирован VectorAverageExecutor")

    def set_parameters(self, parameters: VectorAverageParameters) -> Tuple[bool, str]:
        """
        Установка параметров усреднения.

        Args:
            parameters: Параметры усреднения

        Returns:
            Tuple[bool, str]: (success, error_message)
        """
        # Валидация
        is_valid, error_msg = parameters.validate()
        if not is_valid:
            logger.error(f"Ошибка валидации: {error_msg}")
            return False, error_msg

        self.parameters = parameters

        grid_info = parameters.get_grid_info()
        logger.info(f"Параметры установлены:")
        logger.info(f"  Входной файл: {parameters.input_file}")
        logger.info(f"  Плоскость: {parameters.plane_width} x {parameters.plane_height}")
        logger.info(f"  Ячейка: {parameters.cell_width} x {parameters.cell_height}")
        logger.info(f"  Сетка: {grid_info['nx']} x {grid_info['ny']} = {grid_info['total_cells']} ячеек")
        logger.info(f"  Мин. точек в ячейке: {parameters.min_points_in_cell}")

        return True, ""

    def _get_output_path(self) -> Path:
        """
        Получение пути к выходному файлу.

        Returns:
            Путь к выходному файлу
        """
        input_path = Path(self.parameters.input_file)
        suffix = self.parameters.suffix if self.parameters else "_averaged"

        if self.parameters and self.parameters.output_folder:
            output_dir = Path(self.parameters.output_folder)
        else:
            output_dir = input_path.parent

        output_name = f"{input_path.stem}{suffix}.csv"
        return output_dir / output_name

    def _get_cell_index(self, x: float, y: float) -> Tuple[int, int]:
        """
        Определение индекса ячейки для точки.

        Args:
            x: X координата точки
            y: Y координата точки

        Returns:
            Tuple[int, int]: (ix, iy) индексы ячейки
        """
        # Смещение относительно начала координат
        x_rel = x - self.parameters.origin_x
        y_rel = y - self.parameters.origin_y

        # Индекс ячейки
        ix = int(x_rel / self.parameters.cell_width)
        iy = int(y_rel / self.parameters.cell_height)

        return ix, iy

    def _get_cell_center(self, ix: int, iy: int) -> Tuple[float, float]:
        """
        Получение координат центра ячейки.

        Args:
            ix: Индекс ячейки по X
            iy: Индекс ячейки по Y

        Returns:
            Tuple[float, float]: (x_center, y_center)
        """
        x_center = self.parameters.origin_x + (ix + 0.5) * self.parameters.cell_width
        y_center = self.parameters.origin_y + (iy + 0.5) * self.parameters.cell_height
        return x_center, y_center

    def execute(self) -> VectorAverageResult:
        """
        Выполнение усреднения векторов.

        Returns:
            VectorAverageResult с результатами усреднения
        """
        if self.parameters is None:
            logger.error("Параметры не установлены")
            return VectorAverageResult(
                success=False,
                input_vectors=0,
                output_cells=0,
                empty_cells=0,
                total_cells=0,
                grid_size=(0, 0),
                min_points_per_cell=0,
                max_points_per_cell=0,
                avg_points_per_cell=0.0,
                errors=["Параметры не установлены"],
                output_file=""
            )

        input_path = Path(self.parameters.input_file)
        output_path = self._get_output_path()

        # Расчёт размера сетки
        nx = int(np.ceil(self.parameters.plane_width / self.parameters.cell_width))
        ny = int(np.ceil(self.parameters.plane_height / self.parameters.cell_height))
        total_cells = nx * ny

        logger.info("=" * 60)
        logger.info("ЗАПУСК УСРЕДНЕНИЯ ВЕКТОРОВ")
        logger.info(f"Входной файл: {input_path}")
        logger.info(f"Выходной файл: {output_path}")
        logger.info(f"Плоскость: {self.parameters.plane_width} x {self.parameters.plane_height}")
        logger.info(f"Начало координат: ({self.parameters.origin_x}, {self.parameters.origin_y})")
        logger.info(f"Размер ячейки: {self.parameters.cell_width} x {self.parameters.cell_height}")
        logger.info(f"Сетка: {nx} x {ny} = {total_cells} ячеек")
        logger.info("=" * 60)

        errors = []

        try:
            # Чтение входного файла
            with open(input_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                fieldnames = reader.fieldnames

                if fieldnames is None:
                    errors.append("Не удалось прочитать заголовки файла")
                    return VectorAverageResult(
                        success=False,
                        input_vectors=0,
                        output_cells=0,
                        empty_cells=total_cells,
                        total_cells=total_cells,
                        grid_size=(nx, ny),
                        min_points_per_cell=0,
                        max_points_per_cell=0,
                        avg_points_per_cell=0.0,
                        errors=errors,
                        output_file=""
                    )

                # Проверка обязательных столбцов
                required_columns = [self.parameters.x_column, self.parameters.y_column, 'U', 'V']
                for col in required_columns:
                    if col not in fieldnames:
                        errors.append(f"Отсутствует обязательный столбец: {col}")
                        return VectorAverageResult(
                            success=False,
                            input_vectors=0,
                            output_cells=0,
                            empty_cells=total_cells,
                            total_cells=total_cells,
                            grid_size=(nx, ny),
                            min_points_per_cell=0,
                            max_points_per_cell=0,
                            avg_points_per_cell=0.0,
                            errors=errors,
                            output_file=""
                        )

                rows = list(reader)
                input_vectors = len(rows)

            # Словарь для накопления данных по ячейкам
            # Ключ: (ix, iy), Значение: {'u_sum': float, 'v_sum': float, 'count': int}
            cells: Dict[Tuple[int, int], Dict[str, float]] = {}

            # Распределение точек по ячейкам
            x_col = self.parameters.x_column
            y_col = self.parameters.y_column

            for row in rows:
                try:
                    x = float(row[x_col].replace(',', '.'))
                    y = float(row[y_col].replace(',', '.'))
                    u = float(row['U'].replace(',', '.'))
                    v = float(row['V'].replace(',', '.'))

                    # Проверка, что точка внутри плоскости
                    x_rel = x - self.parameters.origin_x
                    y_rel = y - self.parameters.origin_y

                    if x_rel < 0 or x_rel >= self.parameters.plane_width:
                        continue
                    if y_rel < 0 or y_rel >= self.parameters.plane_height:
                        continue

                    ix, iy = self._get_cell_index(x, y)

                    # Ограничение индексов
                    ix = max(0, min(ix, nx - 1))
                    iy = max(0, min(iy, ny - 1))

                    key = (ix, iy)
                    if key not in cells:
                        cells[key] = {'u_sum': 0.0, 'v_sum': 0.0, 'count': 0}

                    cells[key]['u_sum'] += u
                    cells[key]['v_sum'] += v
                    cells[key]['count'] += 1

                except (ValueError, KeyError) as e:
                    logger.warning(f"Ошибка парсинга строки: {e}")
                    continue

            # Формирование выходных данных
            output_rows = []
            point_counts = []

            for (ix, iy), data in cells.items():
                count = data['count']
                if count >= self.parameters.min_points_in_cell:
                    x_center, y_center = self._get_cell_center(ix, iy)
                    u_avg = data['u_sum'] / count
                    v_avg = data['v_sum'] / count

                    output_rows.append({
                        'X_center': f"{x_center:.6f}",
                        'Y_center': f"{y_center:.6f}",
                        'U_avg': f"{u_avg:.6f}",
                        'V_avg': f"{v_avg:.6f}",
                        'count': count
                    })
                    point_counts.append(count)

            output_cells = len(output_rows)
            empty_cells = total_cells - len(cells)

            # Статистика
            if point_counts:
                min_points = min(point_counts)
                max_points = max(point_counts)
                avg_points = np.mean(point_counts)
            else:
                min_points = 0
                max_points = 0
                avg_points = 0.0

            # Запись выходного файла
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                fieldnames = ['X_center', 'Y_center', 'U_avg', 'V_avg', 'count']
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                writer.writerows(output_rows)

            logger.info("=" * 60)
            logger.info("ЗАВЕРШЕНИЕ УСРЕДНЕНИЯ ВЕКТОРОВ")
            logger.info(f"Векторов на входе: {input_vectors}")
            logger.info(f"Непустых ячеек: {output_cells}")
            logger.info(f"Пустых ячеек: {empty_cells}")
            logger.info(f"Точек в ячейке: min={min_points}, max={max_points}, avg={avg_points:.1f}")
            logger.info(f"Выходной файл: {output_path}")
            logger.info("=" * 60)

            return VectorAverageResult(
                success=True,
                input_vectors=input_vectors,
                output_cells=output_cells,
                empty_cells=empty_cells,
                total_cells=total_cells,
                grid_size=(nx, ny),
                min_points_per_cell=min_points,
                max_points_per_cell=max_points,
                avg_points_per_cell=avg_points,
                errors=[],
                output_file=str(output_path)
            )

        except Exception as e:
            logger.error(f"Ошибка обработки: {e}")
            errors.append(str(e))
            return VectorAverageResult(
                success=False,
                input_vectors=0,
                output_cells=0,
                empty_cells=total_cells,
                total_cells=total_cells,
                grid_size=(nx, ny),
                min_points_per_cell=0,
                max_points_per_cell=0,
                avg_points_per_cell=0.0,
                errors=errors,
                output_file=""
            )

    def get_preview(self) -> Optional[Dict[str, Any]]:
        """
        Получение предварительного просмотра усреднения.

        Returns:
            Словарь со статистикой предпросмотра или None
        """
        if self.parameters is None:
            return None

        input_path = Path(self.parameters.input_file)

        if not input_path.exists():
            return None

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                rows = list(reader)

            total_vectors = len(rows)
            grid_info = self.parameters.get_grid_info()

            # Подсчёт точек в границах плоскости
            x_col = self.parameters.x_column
            y_col = self.parameters.y_column
            vectors_in_bounds = 0

            x_values = []
            y_values = []

            for row in rows:
                try:
                    x = float(row[x_col].replace(',', '.'))
                    y = float(row[y_col].replace(',', '.'))

                    x_rel = x - self.parameters.origin_x
                    y_rel = y - self.parameters.origin_y

                    if 0 <= x_rel < self.parameters.plane_width and 0 <= y_rel < self.parameters.plane_height:
                        vectors_in_bounds += 1
                        x_values.append(x)
                        y_values.append(y)

                except (ValueError, KeyError):
                    continue

            return {
                'file': input_path.name,
                'total_vectors': total_vectors,
                'vectors_in_bounds': vectors_in_bounds,
                'vectors_out_of_bounds': total_vectors - vectors_in_bounds,
                'grid_nx': grid_info['nx'],
                'grid_ny': grid_info['ny'],
                'total_cells': grid_info['total_cells'],
                'avg_vectors_per_cell': vectors_in_bounds / grid_info['total_cells'] if grid_info['total_cells'] > 0 else 0,
                'x_min': min(x_values) if x_values else 0,
                'x_max': max(x_values) if x_values else 0,
                'y_min': min(y_values) if y_values else 0,
                'y_max': max(y_values) if y_values else 0
            }

        except Exception as e:
            logger.error(f"Ошибка предпросмотра: {e}")
            return None


def run_vector_average(
    input_file: str,
    plane_width: float,
    plane_height: float,
    cell_width: float,
    cell_height: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    min_points_in_cell: int = 1,
    output_folder: Optional[str] = None,
    suffix: str = "_averaged"
) -> VectorAverageResult:
    """
    Удобная функция для запуска усреднения векторов без создания объектов.

    Args:
        input_file: Путь к CSV файлу
        plane_width: Ширина плоскости
        plane_height: Высота плоскости
        cell_width: Ширина ячейки
        cell_height: Высота ячейки
        origin_x: X координата начала плоскости
        origin_y: Y координата начала плоскости
        min_points_in_cell: Минимальное количество точек для записи ячейки
        output_folder: Папка для выходного файла (None = та же папка)
        suffix: Суффикс для выходного файла

    Returns:
        VectorAverageResult с результатами

    Example:
        >>> result = run_vector_average(
        ...     input_file="path/to/vectors_filtered.csv",
        ...     plane_width=1920,
        ...     plane_height=1080,
        ...     cell_width=64,
        ...     cell_height=64
        ... )
        >>> print(f"Непустых ячеек: {result.output_cells}")
    """
    params = VectorAverageParameters(
        input_file=input_file,
        plane_width=plane_width,
        plane_height=plane_height,
        cell_width=cell_width,
        cell_height=cell_height,
        origin_x=origin_x,
        origin_y=origin_y,
        min_points_in_cell=min_points_in_cell,
        output_folder=output_folder,
        suffix=suffix
    )

    executor = VectorAverageExecutor()
    success, error_msg = executor.set_parameters(params)

    if not success:
        logger.error(f"Ошибка установки параметров: {error_msg}")
        return VectorAverageResult(
            success=False,
            input_vectors=0,
            output_cells=0,
            empty_cells=0,
            total_cells=0,
            grid_size=(0, 0),
            min_points_per_cell=0,
            max_points_per_cell=0,
            avg_points_per_cell=0.0,
            errors=[error_msg],
            output_file=""
        )

    return executor.execute()


def example_gui_usage():
    """
    Пример использования модуля в GUI.

    Эта функция демонстрирует типичный сценарий использования
    модуля усреднения векторов в графическом интерфейсе.
    """
    print("=" * 60)
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ VECTOR AVERAGE В GUI")
    print("=" * 60)

    # === ШАГ 1: Задание параметров (из GUI элементов) ===
    parameters = VectorAverageParameters(
        input_file=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\LucasKanade_2000\cam_1\cam_1_lucas_kanade_sum_filtered.csv",
        plane_width=1920.0,  # Ширина изображения
        plane_height=1080.0,  # Высота изображения
        cell_width=64.0,  # Размер ячейки 64x64
        cell_height=64.0,
        origin_x=0.0,
        origin_y=0.0,
        min_points_in_cell=3,  # Минимум 3 точки для усреднения
        suffix="_averaged"
    )

    print(f"\nПараметры:")
    print(f"  Входной файл: {parameters.input_file}")
    print(f"  Плоскость: {parameters.plane_width} x {parameters.plane_height}")
    print(f"  Ячейка: {parameters.cell_width} x {parameters.cell_height}")
    print(f"  Мин. точек: {parameters.min_points_in_cell}")

    grid_info = parameters.get_grid_info()
    print(f"  Сетка: {grid_info['nx']} x {grid_info['ny']} = {grid_info['total_cells']} ячеек")

    # === ШАГ 2: Валидация параметров ===
    is_valid, error_msg = parameters.validate()
    print(f"\nВалидация: {'OK' if is_valid else 'ОШИБКА'}")
    if not is_valid:
        print(f"  Ошибка: {error_msg}")
        return

    # === ШАГ 3: Создание и настройка исполнителя ===
    executor = VectorAverageExecutor()
    success, error = executor.set_parameters(parameters)

    if not success:
        print(f"Ошибка установки параметров: {error}")
        return

    # === ШАГ 4: Предварительный просмотр ===
    preview = executor.get_preview()
    if preview:
        print(f"\nПредпросмотр:")
        print(f"  Файл: {preview['file']}")
        print(f"  Всего векторов: {preview['total_vectors']}")
        print(f"  В границах плоскости: {preview['vectors_in_bounds']}")
        print(f"  За границами: {preview['vectors_out_of_bounds']}")
        print(f"  Сетка: {preview['grid_nx']} x {preview['grid_ny']}")
        print(f"  Среднее векторов на ячейку: {preview['avg_vectors_per_cell']:.1f}")

    # === ШАГ 5: Выполнение усреднения ===
    print("\nЗапуск усреднения...")
    result = executor.execute()

    # === ШАГ 6: Обработка результата ===
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Успешно: {result.success}")
    print(f"Векторов на входе: {result.input_vectors}")
    print(f"Сетка: {result.grid_size[0]} x {result.grid_size[1]} = {result.total_cells} ячеек")
    print(f"Непустых ячеек: {result.output_cells}")
    print(f"Пустых ячеек: {result.empty_cells}")
    print(f"Точек в ячейке: min={result.min_points_per_cell}, max={result.max_points_per_cell}, avg={result.avg_points_per_cell:.1f}")
    print(f"Выходной файл: {result.output_file}")

    if result.errors:
        print("\nОшибки:")
        for error in result.errors:
            print(f"  - {error}")

    print("=" * 60)


if __name__ == "__main__":
    example_gui_usage()
