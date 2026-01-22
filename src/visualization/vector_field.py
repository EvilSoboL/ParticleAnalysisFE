"""
Модуль визуализации векторного поля для PTV анализа.

Создает векторное поле смещений частиц для каждой камеры на основе
суммарных CSV файлов (cam_1_pairs_sum.csv, cam_2_pairs_sum.csv).

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging
import csv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VectorData:
    """Данные вектора смещения частицы."""
    x0: float
    y0: float
    dx: float
    dy: float
    length: float


@dataclass
class VectorFieldConfig:
    """Конфигурация визуализации векторного поля."""
    image_width: int = 1024
    image_height: int = 1024
    # Параметры quiver
    nx: int = 54  # Количество ячеек по X
    ny: int = 54  # Количество ячеек по Y
    scale: float = 20  # Масштаб quiver (меньше = длиннее стрелки)
    width: float = 0.005  # Толщина стрелок
    cmap: str = "jet"  # Цветовая карта
    vmin: Optional[float] = None  # Минимум для colorbar
    vmax: Optional[float] = None  # Максимум для colorbar
    # Параметры сетки
    show_grid: bool = True  # Показывать сетку
    grid_color: str = "black"  # Цвет сетки
    grid_alpha: float = 0.25  # Прозрачность сетки
    grid_linewidth: float = 0.4  # Толщина линий сетки
    # Параметры осей и заголовка
    xlabel: str = "r, mm"  # Подпись оси X
    ylabel: str = "z, mm"  # Подпись оси Y
    title: Optional[str] = None  # Заголовок графика
    figsize: Tuple[float, float] = (9, 6)  # Размер фигуры в дюймах
    # Устаревшие параметры (для совместимости)
    background_color: Tuple[int, int, int] = (255, 255, 255)
    arrow_color: Tuple[int, int, int] = (0, 0, 255)
    arrow_thickness: int = 2
    arrow_tip_length: float = 0.3
    scale_factor: float = 1.0
    draw_start_points: bool = False
    start_point_color: Tuple[int, int, int] = (0, 255, 0)
    start_point_radius: int = 2


@dataclass
class VectorFieldResult:
    """Результат создания векторного поля."""
    success: bool
    cam1_vectors_count: int
    cam2_vectors_count: int
    errors: List[str]
    output_folder: str


class VectorFieldVisualizer:
    """
    Класс для создания визуализации векторного поля.

    Читает суммарные CSV файлы с парами частиц и создает
    изображения с векторами смещения.
    """

    def __init__(self):
        """Инициализация визуализатора векторного поля."""
        self.ptv_folder: Optional[Path] = None
        self.output_folder: Optional[Path] = None
        self.config = VectorFieldConfig()
        self.original_folder: Optional[Path] = None

        logger.info("Инициализирован модуль визуализации векторного поля")

    def set_ptv_folder(self, folder_path: str) -> bool:
        """
        Установка папки с результатами PTV анализа.

        Args:
            folder_path: Путь к папке PTV_XXXX

        Returns:
            bool: True если папка валидна
        """
        path = Path(folder_path)

        if not path.exists():
            logger.error(f"Папка не существует: {folder_path}")
            return False

        # Проверяем наличие суммарных CSV файлов
        cam1_sum = path / "cam_1_pairs_sum.csv"
        cam2_sum = path / "cam_2_pairs_sum.csv"

        if not cam1_sum.exists() and not cam2_sum.exists():
            logger.error("Папка должна содержать cam_1_pairs_sum.csv и/или cam_2_pairs_sum.csv")
            return False

        self.ptv_folder = path
        self.output_folder = path / "vector_field"
        logger.info(f"Установлена папка PTV результатов: {path}")
        logger.info(f"Выходная папка: {self.output_folder}")

        return True

    def set_original_folder(self, folder_path: str) -> bool:
        """
        Установка папки с исходными изображениями для определения размеров.

        Args:
            folder_path: Путь к папке cam_sorted

        Returns:
            bool: True если папка валидна
        """
        path = Path(folder_path)

        if not path.exists():
            logger.error(f"Папка не существует: {folder_path}")
            return False

        self.original_folder = path
        logger.info(f"Установлена папка исходных изображений: {path}")

        # Попытка определить размеры изображения из первого файла
        cam1_folder = path / "cam_1"
        if cam1_folder.exists():
            png_files = list(cam1_folder.glob("*.png"))
            if png_files:
                try:
                    img = cv2.imread(str(png_files[0]), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        h, w = img.shape[:2]
                        self.config.image_height = h
                        self.config.image_width = w
                        logger.info(f"Размеры изображения определены: {w}x{h}")
                except Exception as e:
                    logger.warning(f"Не удалось определить размеры изображения: {e}")

        return True

    def set_config(self,
                   image_width: Optional[int] = None,
                   image_height: Optional[int] = None,
                   nx: Optional[int] = None,
                   ny: Optional[int] = None,
                   scale: Optional[float] = None,
                   width: Optional[float] = None,
                   cmap: Optional[str] = None,
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   show_grid: Optional[bool] = None,
                   xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None,
                   title: Optional[str] = None,
                   figsize: Optional[Tuple[float, float]] = None,
                   # Устаревшие параметры для обратной совместимости
                   arrow_color: Optional[Tuple[int, int, int]] = None,
                   arrow_thickness: Optional[int] = None,
                   scale_factor: Optional[float] = None,
                   background_color: Optional[Tuple[int, int, int]] = None,
                   draw_start_points: Optional[bool] = None,
                   start_point_color: Optional[Tuple[int, int, int]] = None) -> None:
        """
        Установка параметров визуализации.

        Args:
            image_width: Ширина выходного изображения
            image_height: Высота выходного изображения
            nx: Количество ячеек сетки по X
            ny: Количество ячеек сетки по Y
            scale: Масштаб quiver (меньше = длиннее стрелки)
            width: Толщина стрелок quiver
            cmap: Название цветовой карты matplotlib
            vmin: Минимальное значение для colorbar
            vmax: Максимальное значение для colorbar
            show_grid: Показывать ли сетку
            xlabel: Подпись оси X
            ylabel: Подпись оси Y
            title: Заголовок графика
            figsize: Размер фигуры (ширина, высота) в дюймах
            arrow_color: [устарело] Цвет стрелок (BGR)
            arrow_thickness: [устарело] Толщина стрелок
            scale_factor: [устарело] Масштаб векторов
            background_color: [устарело] Цвет фона (BGR)
            draw_start_points: [устарело] Рисовать ли начальные точки
            start_point_color: [устарело] Цвет начальных точек (BGR)
        """
        if image_width is not None:
            self.config.image_width = image_width
        if image_height is not None:
            self.config.image_height = image_height
        if nx is not None:
            self.config.nx = nx
        if ny is not None:
            self.config.ny = ny
        if scale is not None:
            self.config.scale = scale
        if width is not None:
            self.config.width = width
        if cmap is not None:
            self.config.cmap = cmap
        if vmin is not None:
            self.config.vmin = vmin
        if vmax is not None:
            self.config.vmax = vmax
        if show_grid is not None:
            self.config.show_grid = show_grid
        if xlabel is not None:
            self.config.xlabel = xlabel
        if ylabel is not None:
            self.config.ylabel = ylabel
        if title is not None:
            self.config.title = title
        if figsize is not None:
            self.config.figsize = figsize
        # Устаревшие параметры
        if arrow_color is not None:
            self.config.arrow_color = arrow_color
        if arrow_thickness is not None:
            self.config.arrow_thickness = arrow_thickness
        if scale_factor is not None:
            self.config.scale_factor = scale_factor
        if background_color is not None:
            self.config.background_color = background_color
        if draw_start_points is not None:
            self.config.draw_start_points = draw_start_points
        if start_point_color is not None:
            self.config.start_point_color = start_point_color

        logger.info("Обновлены параметры визуализации векторного поля")

    def _load_vectors_csv(self, csv_path: Path) -> List[VectorData]:
        """
        Загрузка векторов смещения из суммарного CSV файла.

        Args:
            csv_path: Путь к CSV файлу (cam_X_pairs_sum.csv)

        Returns:
            Список векторов
        """
        vectors = []

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')

                for row in reader:
                    try:
                        vector = VectorData(
                            x0=float(row['X0']),
                            y0=float(row['Y0']),
                            dx=float(row['dx']),
                            dy=float(row['dy']),
                            length=float(row['L'])
                        )
                        vectors.append(vector)
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Ошибка парсинга строки в {csv_path.name}: {e}")
                        continue

            logger.info(f"Загружено {len(vectors)} векторов из {csv_path.name}")

        except Exception as e:
            logger.error(f"Ошибка чтения CSV {csv_path}: {e}")

        return vectors

    def create_vector_field(self, vectors: List[VectorData]) -> np.ndarray:
        """
        Создание изображения векторного поля с усреднением по ячейкам сетки.

        Args:
            vectors: Список векторов смещения

        Returns:
            Изображение с векторным полем (BGR)
        """
        if not vectors:
            raise RuntimeError("Список векторов пуст")

        cfg = self.config

        # Преобразование в массивы numpy
        x0_arr = np.array([v.x0 for v in vectors])
        y0_arr = np.array([v.y0 for v in vectors])
        dx_arr = np.array([v.dx for v in vectors])
        dy_arr = np.array([v.dy for v in vectors])

        # ---------- ГРАНИЦЫ ----------
        x_min, x_max = x0_arr.min(), x0_arr.max()
        y_min, y_max = y0_arr.min(), y0_arr.max()

        # ---------- СЕТКА ----------
        x_edges = np.linspace(x_min, x_max, cfg.nx + 1)
        y_edges = np.linspace(y_min, y_max, cfg.ny + 1)

        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # ---------- ЯЧЕЙКИ ----------
        ix = np.digitize(x0_arr, x_edges) - 1
        iy = np.digitize(y0_arr, y_edges) - 1

        mask = (ix >= 0) & (ix < cfg.nx) & (iy >= 0) & (iy < cfg.ny)

        x0_arr = x0_arr[mask]
        y0_arr = y0_arr[mask]
        dx_arr = dx_arr[mask]
        dy_arr = dy_arr[mask]
        ix = ix[mask]
        iy = iy[mask]

        # ---------- УСРЕДНЕНИЕ ----------
        # Используем словарь для группировки
        cell_data = {}
        for i in range(len(x0_arr)):
            key = (ix[i], iy[i])
            if key not in cell_data:
                cell_data[key] = {'dx': [], 'dy': []}
            cell_data[key]['dx'].append(dx_arr[i])
            cell_data[key]['dy'].append(dy_arr[i])

        if not cell_data:
            raise RuntimeError("Нет данных для усреднения после фильтрации")

        # Вычисление средних значений
        X = []
        Y = []
        dx_mean = []
        dy_mean = []

        for (i_x, i_y), data in cell_data.items():
            X.append(x_centers[i_x])
            Y.append(y_centers[i_y])
            dx_mean.append(np.mean(data['dx']))
            dy_mean.append(np.mean(data['dy']))

        X = np.array(X)
        Y = np.array(Y)
        dx_mean = np.array(dx_mean)
        dy_mean = np.array(dy_mean)
        L = np.sqrt(dx_mean ** 2 + dy_mean ** 2)

        # ---------- НОРМАЛИЗАЦИЯ ----------
        from matplotlib.colors import Normalize
        norm = None
        if cfg.vmin is not None or cfg.vmax is not None:
            norm = Normalize(vmin=cfg.vmin, vmax=cfg.vmax)

        # ---------- ОТРИСОВКА ----------
        fig = Figure(figsize=cfg.figsize, dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        # Рисуем векторное поле с цветовой картой
        q = ax.quiver(
            X, Y, dx_mean, dy_mean, L,
            cmap=cfg.cmap,
            norm=norm,
            scale=cfg.scale,
            width=cfg.width
        )

        # Colorbar
        cbar = fig.colorbar(q, ax=ax, label="|V| (bin mean)")

        # Сетка
        if cfg.show_grid:
            for x in x_edges:
                ax.axvline(x, color=cfg.grid_color, lw=cfg.grid_linewidth, alpha=cfg.grid_alpha)
            for y in y_edges:
                ax.axhline(y, color=cfg.grid_color, lw=cfg.grid_linewidth, alpha=cfg.grid_alpha)

        # Заголовок и подписи
        if cfg.title:
            ax.set_title(cfg.title)
        ax.set_xlabel(cfg.xlabel)
        ax.set_ylabel(cfg.ylabel)

        fig.tight_layout()

        # Преобразование в изображение numpy
        canvas.draw()
        width, height = fig.get_size_inches() * fig.dpi
        width, height = int(width), int(height)

        image_rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        image_rgba = image_rgba.reshape(height, width, 4)

        # Конвертация RGBA в BGR для совместимости с OpenCV
        image_bgr = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGR)

        plt.close(fig)

        return image_bgr

    def _save_vector_field(self, image: np.ndarray, output_path: Path) -> bool:
        """
        Сохранение изображения векторного поля.

        Args:
            image: Изображение для сохранения
            output_path: Путь к выходному файлу

        Returns:
            bool: True если успешно
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)
            logger.info(f"Сохранено: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения {output_path}: {e}")
            return False

    def process_camera(self, camera_name: str) -> Tuple[int, List[str]]:
        """
        Создание векторного поля для одной камеры.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Tuple[vectors_count, errors]
        """
        if self.ptv_folder is None:
            return 0, ["Папка PTV не установлена"]

        # Путь к суммарному CSV файлу
        csv_path = self.ptv_folder / f"{camera_name}_pairs_sum.csv"

        if not csv_path.exists():
            return 0, [f"Не найден файл: {csv_path.name}"]

        # Загрузка векторов
        vectors = self._load_vectors_csv(csv_path)

        if not vectors:
            return 0, [f"Нет данных в {csv_path.name}"]

        # Установка заголовка графика
        self.config.title = f"Усреднение по ячейкам ({self.config.nx}×{self.config.ny})\n{camera_name}"

        # Создание векторного поля
        vector_field_image = self.create_vector_field(vectors)

        # Сохранение изображения
        output_path = self.output_folder / f"{camera_name}_vector_field.png"

        if not self._save_vector_field(vector_field_image, output_path):
            return 0, [f"Ошибка сохранения {output_path.name}"]

        return len(vectors), []

    def process_all(self) -> VectorFieldResult:
        """
        Создание векторных полей для всех камер.

        Returns:
            VectorFieldResult с результатами
        """
        if self.ptv_folder is None:
            return VectorFieldResult(
                success=False,
                cam1_vectors_count=0,
                cam2_vectors_count=0,
                errors=["Папка PTV результатов не установлена"],
                output_folder=""
            )

        logger.info("=" * 60)
        logger.info("СОЗДАНИЕ ВЕКТОРНЫХ ПОЛЕЙ")
        logger.info(f"PTV результаты: {self.ptv_folder}")
        logger.info(f"Выходная папка: {self.output_folder}")
        logger.info(f"Сетка: {self.config.nx}×{self.config.ny}")
        logger.info(f"Масштаб quiver: {self.config.scale}")
        logger.info(f"Цветовая карта: {self.config.cmap}")
        logger.info("=" * 60)

        all_errors = []

        # Обработка cam_1
        cam1_count = 0
        cam1_sum_path = self.ptv_folder / "cam_1_pairs_sum.csv"
        if cam1_sum_path.exists():
            logger.info("\n--- Создание векторного поля для cam_1 ---")
            cam1_count, cam1_errors = self.process_camera("cam_1")
            all_errors.extend(cam1_errors)
        else:
            logger.warning("cam_1_pairs_sum.csv не найден")

        # Обработка cam_2
        cam2_count = 0
        cam2_sum_path = self.ptv_folder / "cam_2_pairs_sum.csv"
        if cam2_sum_path.exists():
            logger.info("\n--- Создание векторного поля для cam_2 ---")
            cam2_count, cam2_errors = self.process_camera("cam_2")
            all_errors.extend(cam2_errors)
        else:
            logger.warning("cam_2_pairs_sum.csv не найден")

        success = len(all_errors) == 0 and (cam1_count > 0 or cam2_count > 0)

        logger.info("\n" + "=" * 60)
        logger.info("РЕЗУЛЬТАТЫ СОЗДАНИЯ ВЕКТОРНЫХ ПОЛЕЙ")
        logger.info("=" * 60)
        logger.info(f"cam_1: {cam1_count} векторов")
        logger.info(f"cam_2: {cam2_count} векторов")
        logger.info(f"Выходная папка: {self.output_folder}")
        logger.info("=" * 60)

        return VectorFieldResult(
            success=success,
            cam1_vectors_count=cam1_count,
            cam2_vectors_count=cam2_count,
            errors=all_errors,
            output_folder=str(self.output_folder) if self.output_folder else ""
        )

    def get_preview(self, camera_name: str) -> Optional[np.ndarray]:
        """
        Получение предварительного просмотра векторного поля.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Изображение векторного поля или None
        """
        if self.ptv_folder is None:
            return None

        csv_path = self.ptv_folder / f"{camera_name}_pairs_sum.csv"

        if not csv_path.exists():
            return None

        vectors = self._load_vectors_csv(csv_path)

        if not vectors:
            return None

        return self.create_vector_field(vectors)

    def get_statistics(self, camera_name: str) -> Optional[dict]:
        """
        Получение статистики векторов для камеры.

        Args:
            camera_name: Название камеры

        Returns:
            Словарь со статистикой или None
        """
        if self.ptv_folder is None:
            return None

        csv_path = self.ptv_folder / f"{camera_name}_pairs_sum.csv"

        if not csv_path.exists():
            return None

        vectors = self._load_vectors_csv(csv_path)

        if not vectors:
            return {'vectors_count': 0}

        lengths = [v.length for v in vectors]
        dx_values = [v.dx for v in vectors]
        dy_values = [v.dy for v in vectors]

        non_zero_lengths = [l for l in lengths if l > 0]

        stats = {
            'vectors_count': len(vectors),
            'vectors_with_displacement': len(non_zero_lengths)
        }

        if non_zero_lengths:
            stats.update({
                'mean_length': np.mean(non_zero_lengths),
                'max_length': np.max(non_zero_lengths),
                'min_length': np.min(non_zero_lengths),
                'std_length': np.std(non_zero_lengths),
                'mean_dx': np.mean([dx for dx, l in zip(dx_values, lengths) if l > 0]),
                'mean_dy': np.mean([dy for dy, l in zip(dy_values, lengths) if l > 0]),
                'std_dx': np.std([dx for dx, l in zip(dx_values, lengths) if l > 0]),
                'std_dy': np.std([dy for dy, l in zip(dy_values, lengths) if l > 0])
            })
        else:
            stats.update({
                'mean_length': 0,
                'mean_dx': 0,
                'mean_dy': 0
            })

        return stats
