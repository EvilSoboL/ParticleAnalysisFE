"""
Модуль визуализации векторного поля для GUI ParticleAnalysis.

Этот модуль строит график векторного поля по CSV файлу
с усреднёнными векторами.

Входной формат CSV (разделитель ;):
    X_center;Y_center;U_avg;V_avg;count
    или:
    X_center;Y_center;U_avg;V_avg
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import logging
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VectorPlotResult:
    """Результат построения графика."""
    success: bool
    vectors_count: int  # Количество отрисованных векторов
    output_file: str  # Путь к сохранённому изображению
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    magnitude_min: float
    magnitude_max: float
    errors: List[str]


@dataclass
class VectorPlotParameters:
    """
    Параметры визуализации векторного поля для GUI.

    Все параметры должны устанавливаться через GUI элементы:
    - input_file: путь к CSV файлу (через file dialog)
    - figure_width, figure_height: размер изображения в дюймах
    - arrow_scale: масштаб стрелок
    - colormap: цветовая карта для магнитуды
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    input_file: str  # Путь к CSV файлу

    # ПАРАМЕТРЫ ИЗОБРАЖЕНИЯ
    figure_width: float = 16.0  # Ширина в дюймах
    figure_height: float = 12.0  # Высота в дюймах
    dpi: int = 150  # Разрешение

    # ПАРАМЕТРЫ СТРЕЛОК
    arrow_scale: float = 1.0  # Масштаб стрелок (меньше = длиннее)
    arrow_width: float = 0.003  # Ширина стрелок
    arrow_headwidth: float = 4.0  # Ширина наконечника
    arrow_headlength: float = 5.0  # Длина наконечника

    # ПАРАМЕТРЫ ЦВЕТА
    colormap: str = "jet"  # Цветовая карта (jet, viridis, plasma, coolwarm, etc.)
    color_by: str = "magnitude"  # По чему красить: magnitude, u, v, angle
    show_colorbar: bool = True  # Показывать colorbar

    # ПАРАМЕТРЫ ОСЕЙ
    title: str = "Vector Field"  # Заголовок графика
    xlabel: str = "X"  # Подпись оси X
    ylabel: str = "Y"  # Подпись оси Y
    invert_y: bool = True  # Инвертировать ось Y (для изображений)
    equal_aspect: bool = True  # Равный масштаб осей
    grid: bool = False  # Показывать сетку

    # ПАРАМЕТРЫ ФОНА
    background_image: Optional[str] = None  # Путь к фоновому изображению
    background_alpha: float = 0.5  # Прозрачность фона

    # ПАРАМЕТРЫ ВЫХОДА
    output_folder: Optional[str] = None  # Папка для выходного файла
    suffix: str = "_plot"  # Суффикс для выходного файла
    output_format: str = "png"  # Формат: png, jpg, svg, pdf

    # КОЛОНКИ CSV
    x_column: str = "X_center"
    y_column: str = "Y_center"
    u_column: str = "U_avg"
    v_column: str = "V_avg"

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

        # Проверка размеров
        if self.figure_width <= 0:
            return False, f"Ширина изображения должна быть > 0: {self.figure_width}"

        if self.figure_height <= 0:
            return False, f"Высота изображения должна быть > 0: {self.figure_height}"

        if self.dpi <= 0:
            return False, f"DPI должен быть > 0: {self.dpi}"

        # Проверка масштаба стрелок
        if self.arrow_scale <= 0:
            return False, f"Масштаб стрелок должен быть > 0: {self.arrow_scale}"

        # Проверка colormap
        valid_cmaps = ['jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                       'coolwarm', 'RdYlBu', 'RdBu', 'Spectral', 'hsv', 'rainbow']
        if self.colormap not in valid_cmaps and self.colormap not in plt.colormaps():
            return False, f"Неизвестная цветовая карта: {self.colormap}"

        # Проверка color_by
        valid_color_by = ['magnitude', 'u', 'v', 'angle']
        if self.color_by not in valid_color_by:
            return False, f"Недопустимое значение color_by: {self.color_by}. Допустимые: {valid_color_by}"

        # Проверка формата
        valid_formats = ['png', 'jpg', 'jpeg', 'svg', 'pdf']
        if self.output_format.lower() not in valid_formats:
            return False, f"Недопустимый формат: {self.output_format}. Допустимые: {valid_formats}"

        # Проверка фонового изображения
        if self.background_image:
            bg_path = Path(self.background_image)
            if not bg_path.exists():
                return False, f"Фоновое изображение не существует: {self.background_image}"

        return True, ""


class VectorPlotExecutor:
    """
    Исполнитель визуализации векторного поля.

    Использование:
        1. Создать экземпляр VectorPlotExecutor
        2. Задать параметры через VectorPlotParameters
        3. Вызвать execute() для построения графика
        4. Получить результат VectorPlotResult
    """

    def __init__(self):
        """Инициализация исполнителя визуализации."""
        self.parameters: Optional[VectorPlotParameters] = None
        self._figure: Optional[Figure] = None
        logger.info("Инициализирован VectorPlotExecutor")

    def set_parameters(self, parameters: VectorPlotParameters) -> Tuple[bool, str]:
        """
        Установка параметров визуализации.

        Args:
            parameters: Параметры визуализации

        Returns:
            Tuple[bool, str]: (success, error_message)
        """
        # Валидация
        is_valid, error_msg = parameters.validate()
        if not is_valid:
            logger.error(f"Ошибка валидации: {error_msg}")
            return False, error_msg

        self.parameters = parameters

        logger.info(f"Параметры установлены:")
        logger.info(f"  Входной файл: {parameters.input_file}")
        logger.info(f"  Размер: {parameters.figure_width}x{parameters.figure_height} дюймов, {parameters.dpi} DPI")
        logger.info(f"  Цветовая карта: {parameters.colormap}, по {parameters.color_by}")

        return True, ""

    def _get_output_path(self) -> Path:
        """
        Получение пути к выходному файлу.

        Returns:
            Путь к выходному файлу
        """
        input_path = Path(self.parameters.input_file)
        suffix = self.parameters.suffix if self.parameters else "_plot"
        fmt = self.parameters.output_format if self.parameters else "png"

        if self.parameters and self.parameters.output_folder:
            output_dir = Path(self.parameters.output_folder)
        else:
            output_dir = input_path.parent

        output_name = f"{input_path.stem}{suffix}.{fmt}"
        return output_dir / output_name

    def _read_csv(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Чтение CSV файла.

        Returns:
            Tuple[X, Y, U, V, errors]
        """
        input_path = Path(self.parameters.input_file)
        errors = []

        x_list = []
        y_list = []
        u_list = []
        v_list = []

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                fieldnames = reader.fieldnames

                if fieldnames is None:
                    errors.append("Не удалось прочитать заголовки файла")
                    return np.array([]), np.array([]), np.array([]), np.array([]), errors

                # Проверка обязательных столбцов
                required = [self.parameters.x_column, self.parameters.y_column,
                           self.parameters.u_column, self.parameters.v_column]
                for col in required:
                    if col not in fieldnames:
                        errors.append(f"Отсутствует столбец: {col}")
                        return np.array([]), np.array([]), np.array([]), np.array([]), errors

                for row in reader:
                    try:
                        x = float(row[self.parameters.x_column].replace(',', '.'))
                        y = float(row[self.parameters.y_column].replace(',', '.'))
                        u = float(row[self.parameters.u_column].replace(',', '.'))
                        v = float(row[self.parameters.v_column].replace(',', '.'))

                        x_list.append(x)
                        y_list.append(y)
                        u_list.append(u)
                        v_list.append(v)

                    except (ValueError, KeyError) as e:
                        logger.warning(f"Ошибка парсинга строки: {e}")
                        continue

        except Exception as e:
            errors.append(str(e))

        return np.array(x_list), np.array(y_list), np.array(u_list), np.array(v_list), errors

    def execute(self) -> VectorPlotResult:
        """
        Выполнение построения графика.

        Returns:
            VectorPlotResult с результатами
        """
        if self.parameters is None:
            logger.error("Параметры не установлены")
            return VectorPlotResult(
                success=False,
                vectors_count=0,
                output_file="",
                u_min=0, u_max=0,
                v_min=0, v_max=0,
                magnitude_min=0, magnitude_max=0,
                errors=["Параметры не установлены"]
            )

        output_path = self._get_output_path()

        logger.info("=" * 60)
        logger.info("ПОСТРОЕНИЕ ГРАФИКА ВЕКТОРНОГО ПОЛЯ")
        logger.info(f"Входной файл: {self.parameters.input_file}")
        logger.info(f"Выходной файл: {output_path}")
        logger.info("=" * 60)

        # Чтение данных
        X, Y, U, V, errors = self._read_csv()

        if errors:
            return VectorPlotResult(
                success=False,
                vectors_count=0,
                output_file="",
                u_min=0, u_max=0,
                v_min=0, v_max=0,
                magnitude_min=0, magnitude_max=0,
                errors=errors
            )

        if len(X) == 0:
            return VectorPlotResult(
                success=False,
                vectors_count=0,
                output_file="",
                u_min=0, u_max=0,
                v_min=0, v_max=0,
                magnitude_min=0, magnitude_max=0,
                errors=["Нет данных для отображения"]
            )

        # Вычисление магнитуды и угла
        magnitude = np.sqrt(U**2 + V**2)
        angle = np.arctan2(V, U)

        # Определение цвета
        if self.parameters.color_by == 'magnitude':
            colors = magnitude
            colorbar_label = 'Magnitude'
        elif self.parameters.color_by == 'u':
            colors = U
            colorbar_label = 'U'
        elif self.parameters.color_by == 'v':
            colors = V
            colorbar_label = 'V'
        else:  # angle
            colors = np.degrees(angle)
            colorbar_label = 'Angle (degrees)'

        # Создание графика
        fig, ax = plt.subplots(figsize=(self.parameters.figure_width, self.parameters.figure_height))
        self._figure = fig

        # Фоновое изображение
        if self.parameters.background_image:
            try:
                bg_img = plt.imread(self.parameters.background_image)
                ax.imshow(bg_img, alpha=self.parameters.background_alpha, aspect='auto',
                         extent=[X.min(), X.max(), Y.max(), Y.min()] if self.parameters.invert_y
                         else [X.min(), X.max(), Y.min(), Y.max()])
            except Exception as e:
                logger.warning(f"Не удалось загрузить фоновое изображение: {e}")

        # Построение quiver
        quiver = ax.quiver(
            X, Y, U, V,
            colors,
            cmap=self.parameters.colormap,
            scale=self.parameters.arrow_scale * 50,  # Нормализация масштаба
            width=self.parameters.arrow_width,
            headwidth=self.parameters.arrow_headwidth,
            headlength=self.parameters.arrow_headlength,
            angles='xy',
            scale_units='xy'
        )

        # Colorbar
        if self.parameters.show_colorbar:
            cbar = plt.colorbar(quiver, ax=ax, label=colorbar_label)

        # Настройки осей
        ax.set_title(self.parameters.title, fontsize=14)
        ax.set_xlabel(self.parameters.xlabel, fontsize=12)
        ax.set_ylabel(self.parameters.ylabel, fontsize=12)

        if self.parameters.invert_y:
            ax.invert_yaxis()

        if self.parameters.equal_aspect:
            ax.set_aspect('equal')

        if self.parameters.grid:
            ax.grid(True, alpha=0.3)

        # Сохранение
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.parameters.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info("=" * 60)
        logger.info("ГРАФИК ПОСТРОЕН")
        logger.info(f"Векторов: {len(X)}")
        logger.info(f"U: [{U.min():.3f}, {U.max():.3f}]")
        logger.info(f"V: [{V.min():.3f}, {V.max():.3f}]")
        logger.info(f"Magnitude: [{magnitude.min():.3f}, {magnitude.max():.3f}]")
        logger.info(f"Выходной файл: {output_path}")
        logger.info("=" * 60)

        return VectorPlotResult(
            success=True,
            vectors_count=len(X),
            output_file=str(output_path),
            u_min=float(U.min()),
            u_max=float(U.max()),
            v_min=float(V.min()),
            v_max=float(V.max()),
            magnitude_min=float(magnitude.min()),
            magnitude_max=float(magnitude.max()),
            errors=[]
        )

    def get_figure(self) -> Optional[Figure]:
        """
        Получение объекта Figure для встраивания в GUI.

        Returns:
            matplotlib Figure или None
        """
        if self.parameters is None:
            return None

        # Чтение данных
        X, Y, U, V, errors = self._read_csv()

        if errors or len(X) == 0:
            return None

        magnitude = np.sqrt(U**2 + V**2)
        angle = np.arctan2(V, U)

        # Определение цвета
        if self.parameters.color_by == 'magnitude':
            colors = magnitude
            colorbar_label = 'Magnitude'
        elif self.parameters.color_by == 'u':
            colors = U
            colorbar_label = 'U'
        elif self.parameters.color_by == 'v':
            colors = V
            colorbar_label = 'V'
        else:
            colors = np.degrees(angle)
            colorbar_label = 'Angle (degrees)'

        fig, ax = plt.subplots(figsize=(self.parameters.figure_width, self.parameters.figure_height))

        if self.parameters.background_image:
            try:
                bg_img = plt.imread(self.parameters.background_image)
                ax.imshow(bg_img, alpha=self.parameters.background_alpha, aspect='auto',
                         extent=[X.min(), X.max(), Y.max(), Y.min()] if self.parameters.invert_y
                         else [X.min(), X.max(), Y.min(), Y.max()])
            except Exception as e:
                logger.warning(f"Не удалось загрузить фоновое изображение: {e}")

        quiver = ax.quiver(
            X, Y, U, V,
            colors,
            cmap=self.parameters.colormap,
            scale=self.parameters.arrow_scale * 50,
            width=self.parameters.arrow_width,
            headwidth=self.parameters.arrow_headwidth,
            headlength=self.parameters.arrow_headlength,
            angles='xy',
            scale_units='xy'
        )

        if self.parameters.show_colorbar:
            plt.colorbar(quiver, ax=ax, label=colorbar_label)

        ax.set_title(self.parameters.title, fontsize=14)
        ax.set_xlabel(self.parameters.xlabel, fontsize=12)
        ax.set_ylabel(self.parameters.ylabel, fontsize=12)

        if self.parameters.invert_y:
            ax.invert_yaxis()

        if self.parameters.equal_aspect:
            ax.set_aspect('equal')

        if self.parameters.grid:
            ax.grid(True, alpha=0.3)

        self._figure = fig
        return fig

    def get_preview(self) -> Optional[Dict[str, Any]]:
        """
        Получение предварительного просмотра.

        Returns:
            Словарь со статистикой или None
        """
        if self.parameters is None:
            return None

        X, Y, U, V, errors = self._read_csv()

        if errors or len(X) == 0:
            return None

        magnitude = np.sqrt(U**2 + V**2)

        return {
            'file': Path(self.parameters.input_file).name,
            'vectors_count': len(X),
            'x_range': (float(X.min()), float(X.max())),
            'y_range': (float(Y.min()), float(Y.max())),
            'u_range': (float(U.min()), float(U.max())),
            'v_range': (float(V.min()), float(V.max())),
            'magnitude_range': (float(magnitude.min()), float(magnitude.max())),
            'u_mean': float(U.mean()),
            'v_mean': float(V.mean()),
            'magnitude_mean': float(magnitude.mean())
        }


def run_vector_plot(
    input_file: str,
    output_folder: Optional[str] = None,
    suffix: str = "_plot",
    output_format: str = "png",
    figure_width: float = 16.0,
    figure_height: float = 12.0,
    dpi: int = 150,
    arrow_scale: float = 1.0,
    colormap: str = "jet",
    color_by: str = "magnitude",
    title: str = "Vector Field",
    invert_y: bool = True,
    background_image: Optional[str] = None
) -> VectorPlotResult:
    """
    Удобная функция для построения графика без создания объектов.

    Args:
        input_file: Путь к CSV файлу
        output_folder: Папка для выходного файла (None = та же папка)
        suffix: Суффикс для выходного файла
        output_format: Формат изображения (png, jpg, svg, pdf)
        figure_width: Ширина в дюймах
        figure_height: Высота в дюймах
        dpi: Разрешение
        arrow_scale: Масштаб стрелок
        colormap: Цветовая карта
        color_by: По чему красить (magnitude, u, v, angle)
        title: Заголовок графика
        invert_y: Инвертировать ось Y
        background_image: Путь к фоновому изображению

    Returns:
        VectorPlotResult с результатами

    Example:
        >>> result = run_vector_plot(
        ...     input_file="vectors_averaged.csv",
        ...     colormap="coolwarm",
        ...     title="Velocity Field"
        ... )
        >>> print(f"Сохранено: {result.output_file}")
    """
    params = VectorPlotParameters(
        input_file=input_file,
        output_folder=output_folder,
        suffix=suffix,
        output_format=output_format,
        figure_width=figure_width,
        figure_height=figure_height,
        dpi=dpi,
        arrow_scale=arrow_scale,
        colormap=colormap,
        color_by=color_by,
        title=title,
        invert_y=invert_y,
        background_image=background_image
    )

    executor = VectorPlotExecutor()
    success, error_msg = executor.set_parameters(params)

    if not success:
        logger.error(f"Ошибка установки параметров: {error_msg}")
        return VectorPlotResult(
            success=False,
            vectors_count=0,
            output_file="",
            u_min=0, u_max=0,
            v_min=0, v_max=0,
            magnitude_min=0, magnitude_max=0,
            errors=[error_msg]
        )

    return executor.execute()


def example_gui_usage():
    """
    Пример использования модуля в GUI.
    """
    print("=" * 60)
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ VECTOR PLOT В GUI")
    print("=" * 60)

    # === ШАГ 1: Задание параметров ===
    parameters = VectorPlotParameters(
        input_file=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\LucasKanade_2000\cam_1\cam_1_lucas_kanade_sum_filtered_averaged.csv",
        figure_width=16.0,
        figure_height=12.0,
        dpi=150,
        arrow_scale=1.0,
        colormap="jet",
        color_by="magnitude",
        title="Vector Field",
        xlabel="X (pixels)",
        ylabel="Y (pixels)",
        invert_y=True,
        equal_aspect=True,
        show_colorbar=True,
        suffix="_plot",
        output_format="png"
    )

    print(f"\nПараметры:")
    print(f"  Входной файл: {parameters.input_file}")
    print(f"  Размер: {parameters.figure_width}x{parameters.figure_height} дюймов")
    print(f"  DPI: {parameters.dpi}")
    print(f"  Colormap: {parameters.colormap}")
    print(f"  Color by: {parameters.color_by}")

    # === ШАГ 2: Валидация ===
    is_valid, error_msg = parameters.validate()
    print(f"\nВалидация: {'OK' if is_valid else 'ОШИБКА'}")
    if not is_valid:
        print(f"  Ошибка: {error_msg}")
        return

    # === ШАГ 3: Создание исполнителя ===
    executor = VectorPlotExecutor()
    success, error = executor.set_parameters(parameters)

    if not success:
        print(f"Ошибка: {error}")
        return

    # === ШАГ 4: Предпросмотр ===
    preview = executor.get_preview()
    if preview:
        print(f"\nПредпросмотр:")
        print(f"  Файл: {preview['file']}")
        print(f"  Векторов: {preview['vectors_count']}")
        print(f"  X: [{preview['x_range'][0]:.1f}, {preview['x_range'][1]:.1f}]")
        print(f"  Y: [{preview['y_range'][0]:.1f}, {preview['y_range'][1]:.1f}]")
        print(f"  U: [{preview['u_range'][0]:.3f}, {preview['u_range'][1]:.3f}]")
        print(f"  V: [{preview['v_range'][0]:.3f}, {preview['v_range'][1]:.3f}]")
        print(f"  Magnitude: [{preview['magnitude_range'][0]:.3f}, {preview['magnitude_range'][1]:.3f}]")

    # === ШАГ 5: Построение графика ===
    print("\nПостроение графика...")
    result = executor.execute()

    # === ШАГ 6: Результат ===
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Успешно: {result.success}")
    print(f"Векторов: {result.vectors_count}")
    print(f"U: [{result.u_min:.3f}, {result.u_max:.3f}]")
    print(f"V: [{result.v_min:.3f}, {result.v_max:.3f}]")
    print(f"Magnitude: [{result.magnitude_min:.3f}, {result.magnitude_max:.3f}]")
    print(f"Выходной файл: {result.output_file}")

    if result.errors:
        print("\nОшибки:")
        for error in result.errors:
            print(f"  - {error}")

    print("=" * 60)


if __name__ == "__main__":
    example_gui_usage()
