"""
Модуль визуализации векторного поля для анализа оптического потока Farneback.

Создает векторное поле скоростей для каждой камеры на основе
суммарных CSV файлов (cam_X_farneback_sum.csv или cam_X_lucas_kanade_sum.csv).

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from typing import Optional, List, Tuple, Dict
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
class FlowVectorData:
    """Данные вектора оптического потока."""
    x: float
    y: float
    u: float
    v: float
    magnitude: float
    angle: float


@dataclass
class FarnebackVectorFieldConfig:
    """Конфигурация визуализации векторного поля Farneback."""
    image_width: int = 1024
    image_height: int = 1024
    # Параметры quiver
    nx: int = 50  # Количество ячеек по X
    ny: int = 50  # Количество ячеек по Y
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
    xlabel: str = "X, px"  # Подпись оси X
    ylabel: str = "Y, px"  # Подпись оси Y
    title: Optional[str] = None  # Заголовок графика
    figsize: Tuple[float, float] = (9, 6)  # Размер фигуры в дюймах


@dataclass
class FarnebackVectorFieldResult:
    """Результат создания векторного поля Farneback."""
    success: bool
    cam1_vectors_count: int
    cam2_vectors_count: int
    method: str  # 'farneback' или 'lucas_kanade'
    errors: List[str]
    output_folder: str


class FarnebackVectorFieldVisualizer:
    """
    Класс для создания визуализации векторного поля Farneback.

    Читает суммарные CSV файлы с векторами оптического потока и создает
    изображения с усреднёнными векторами по ячейкам сетки.
    """

    def __init__(self):
        """Инициализация визуализатора векторного поля Farneback."""
        self.farneback_folder: Optional[Path] = None
        self.output_folder: Optional[Path] = None
        self.config = FarnebackVectorFieldConfig()
        self.method: str = "farneback"  # 'farneback' или 'lucas_kanade'

        logger.info("Инициализирован модуль визуализации векторного поля Farneback")

    def set_farneback_folder(self, folder_path: str) -> bool:
        """
        Установка папки с результатами анализа Farneback.

        Args:
            folder_path: Путь к папке Farneback_XXXX

        Returns:
            bool: True если папка валидна
        """
        path = Path(folder_path)

        if not path.exists():
            logger.error(f"Папка не существует: {folder_path}")
            return False

        # Проверяем наличие суммарных CSV файлов (farneback или lucas_kanade)
        cam1_fb = path / "cam_1" / "cam_1_farneback_sum.csv"
        cam2_fb = path / "cam_2" / "cam_2_farneback_sum.csv"
        cam1_lk = path / "cam_1" / "cam_1_lucas_kanade_sum.csv"
        cam2_lk = path / "cam_2" / "cam_2_lucas_kanade_sum.csv"

        has_farneback = cam1_fb.exists() or cam2_fb.exists()
        has_lucas_kanade = cam1_lk.exists() or cam2_lk.exists()

        if not has_farneback and not has_lucas_kanade:
            logger.error("Папка должна содержать cam_X/cam_X_farneback_sum.csv или cam_X/cam_X_lucas_kanade_sum.csv")
            return False

        # Определяем метод по наличию файлов
        if has_farneback:
            self.method = "farneback"
        else:
            self.method = "lucas_kanade"

        self.farneback_folder = path
        self.output_folder = path / "vector_field"
        logger.info(f"Установлена папка Farneback результатов: {path}")
        logger.info(f"Метод: {self.method}")
        logger.info(f"Выходная папка: {self.output_folder}")

        return True

    def set_method(self, method: str) -> bool:
        """
        Установка метода анализа (farneback или lucas_kanade).

        Args:
            method: 'farneback' или 'lucas_kanade'

        Returns:
            bool: True если метод валиден
        """
        if method not in ("farneback", "lucas_kanade"):
            logger.error(f"Неверный метод: {method}. Допустимые: farneback, lucas_kanade")
            return False

        self.method = method
        logger.info(f"Установлен метод: {method}")
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
                   figsize: Optional[Tuple[float, float]] = None) -> None:
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

        logger.info("Обновлены параметры визуализации векторного поля Farneback")

    def _load_vectors_csv(self, csv_path: Path) -> List[FlowVectorData]:
        """
        Загрузка векторов оптического потока из суммарного CSV файла.

        Args:
            csv_path: Путь к CSV файлу

        Returns:
            Список векторов
        """
        vectors = []

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')

                for row in reader:
                    try:
                        # Farneback формат: X, Y, U, V, Magnitude, Angle
                        # Lucas-Kanade формат: X0, Y0, X1, Y1, U, V, Magnitude, Angle
                        if 'X0' in row:
                            # Lucas-Kanade формат
                            x = float(row['X0'])
                            y = float(row['Y0'])
                        else:
                            # Farneback формат
                            x = float(row['X'])
                            y = float(row['Y'])

                        u = float(row['U'])
                        v = float(row['V'])
                        magnitude = float(row['Magnitude'])
                        angle = float(row.get('Angle', 0))

                        # Пропускаем NaN значения
                        if np.isnan(x) or np.isnan(y) or np.isnan(u) or np.isnan(v) or np.isnan(magnitude):
                            continue

                        vector = FlowVectorData(
                            x=x,
                            y=y,
                            u=u,
                            v=v,
                            magnitude=magnitude,
                            angle=angle
                        )
                        vectors.append(vector)
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Ошибка парсинга строки в {csv_path.name}: {e}")
                        continue

            logger.info(f"Загружено {len(vectors)} векторов из {csv_path.name}")

        except Exception as e:
            logger.error(f"Ошибка чтения CSV {csv_path}: {e}")

        return vectors

    def create_vector_field(self, vectors: List[FlowVectorData], camera_name: str = "") -> np.ndarray:
        """
        Создание изображения векторного поля с усреднением по ячейкам сетки.

        Args:
            vectors: Список векторов оптического потока
            camera_name: Название камеры для заголовка

        Returns:
            Изображение с векторным полем (BGR)
        """
        if not vectors:
            raise RuntimeError("Список векторов пуст")

        cfg = self.config

        # Преобразование в массивы numpy
        x_arr = np.array([v.x for v in vectors])
        y_arr = np.array([v.y for v in vectors])
        u_arr = np.array([v.u for v in vectors])
        v_arr = np.array([v.v for v in vectors])

        # ---------- ГРАНИЦЫ ----------
        x_min, x_max = x_arr.min(), x_arr.max()
        y_min, y_max = y_arr.min(), y_arr.max()

        # ---------- СЕТКА ----------
        x_edges = np.linspace(x_min, x_max, cfg.nx + 1)
        y_edges = np.linspace(y_min, y_max, cfg.ny + 1)

        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # ---------- ЯЧЕЙКИ ----------
        ix = np.digitize(x_arr, x_edges) - 1
        iy = np.digitize(y_arr, y_edges) - 1

        mask = (ix >= 0) & (ix < cfg.nx) & (iy >= 0) & (iy < cfg.ny)

        x_arr = x_arr[mask]
        y_arr = y_arr[mask]
        u_arr = u_arr[mask]
        v_arr = v_arr[mask]
        ix = ix[mask]
        iy = iy[mask]

        # ---------- УСРЕДНЕНИЕ ПО ЯЧЕЙКАМ ----------
        cell_data: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
        for i in range(len(x_arr)):
            key = (ix[i], iy[i])
            if key not in cell_data:
                cell_data[key] = {'u': [], 'v': []}
            cell_data[key]['u'].append(u_arr[i])
            cell_data[key]['v'].append(v_arr[i])

        if not cell_data:
            raise RuntimeError("Нет данных для усреднения после фильтрации")

        # Вычисление средних значений
        X = []
        Y = []
        u_mean = []
        v_mean = []

        for (i_x, i_y), data in cell_data.items():
            X.append(x_centers[i_x])
            Y.append(y_centers[i_y])
            u_mean.append(np.mean(data['u']))
            v_mean.append(np.mean(data['v']))

        X = np.array(X)
        Y = np.array(Y)
        u_mean = np.array(u_mean)
        v_mean = np.array(v_mean)
        magnitude = np.sqrt(u_mean ** 2 + v_mean ** 2)

        # ---------- НОРМАЛИЗАЦИЯ ----------
        norm = None
        if cfg.vmin is not None or cfg.vmax is not None:
            norm = Normalize(vmin=cfg.vmin, vmax=cfg.vmax)

        # ---------- ОТРИСОВКА ----------
        fig = Figure(figsize=cfg.figsize, dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        # Рисуем векторное поле с цветовой картой
        q = ax.quiver(
            X, Y, u_mean, v_mean, magnitude,
            cmap=cfg.cmap,
            norm=norm,
            scale=cfg.scale,
            width=cfg.width
        )

        # Colorbar
        method_name = "Farneback" if self.method == "farneback" else "Lucas-Kanade"
        cbar = fig.colorbar(q, ax=ax, label="Velocity Magnitude (bin mean)")

        # Сетка
        if cfg.show_grid:
            for x in x_edges:
                ax.axvline(x, color=cfg.grid_color, lw=cfg.grid_linewidth, alpha=cfg.grid_alpha)
            for y in y_edges:
                ax.axhline(y, color=cfg.grid_color, lw=cfg.grid_linewidth, alpha=cfg.grid_alpha)

        # Заголовок и подписи
        title = cfg.title
        if title is None:
            title = f"{method_name} Velocity Field ({cfg.nx}×{cfg.ny} bins)"
            if camera_name:
                title += f"\n{camera_name}"
        ax.set_title(title)
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

    def _get_sum_csv_path(self, camera_name: str) -> Optional[Path]:
        """
        Получение пути к суммарному CSV файлу.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Путь к CSV файлу или None
        """
        if self.farneback_folder is None:
            return None

        # Проверяем оба возможных пути
        fb_path = self.farneback_folder / camera_name / f"{camera_name}_farneback_sum.csv"
        lk_path = self.farneback_folder / camera_name / f"{camera_name}_lucas_kanade_sum.csv"

        if self.method == "farneback" and fb_path.exists():
            return fb_path
        elif self.method == "lucas_kanade" and lk_path.exists():
            return lk_path
        elif fb_path.exists():
            self.method = "farneback"
            return fb_path
        elif lk_path.exists():
            self.method = "lucas_kanade"
            return lk_path

        return None

    def process_camera(self, camera_name: str) -> Tuple[int, List[str]]:
        """
        Создание векторного поля для одной камеры.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Tuple[vectors_count, errors]
        """
        if self.farneback_folder is None:
            return 0, ["Папка Farneback не установлена"]

        # Путь к суммарному CSV файлу
        csv_path = self._get_sum_csv_path(camera_name)

        if csv_path is None or not csv_path.exists():
            return 0, [f"Не найден суммарный CSV файл для {camera_name}"]

        # Загрузка векторов
        vectors = self._load_vectors_csv(csv_path)

        if not vectors:
            return 0, [f"Нет данных в {csv_path.name}"]

        # Создание векторного поля
        try:
            vector_field_image = self.create_vector_field(vectors, camera_name)
        except RuntimeError as e:
            return 0, [str(e)]

        # Сохранение изображения
        method_suffix = "fb" if self.method == "farneback" else "lk"
        output_path = self.output_folder / f"{camera_name}_{method_suffix}_vector_field.png"

        if not self._save_vector_field(vector_field_image, output_path):
            return 0, [f"Ошибка сохранения {output_path.name}"]

        return len(vectors), []

    def process_all(self) -> FarnebackVectorFieldResult:
        """
        Создание векторных полей для всех камер.

        Returns:
            FarnebackVectorFieldResult с результатами
        """
        if self.farneback_folder is None:
            return FarnebackVectorFieldResult(
                success=False,
                cam1_vectors_count=0,
                cam2_vectors_count=0,
                method=self.method,
                errors=["Папка Farneback результатов не установлена"],
                output_folder=""
            )

        method_name = "Farneback" if self.method == "farneback" else "Lucas-Kanade"

        logger.info("=" * 60)
        logger.info(f"СОЗДАНИЕ ВЕКТОРНЫХ ПОЛЕЙ {method_name.upper()}")
        logger.info(f"Farneback результаты: {self.farneback_folder}")
        logger.info(f"Выходная папка: {self.output_folder}")
        logger.info(f"Сетка: {self.config.nx}×{self.config.ny}")
        logger.info(f"Масштаб quiver: {self.config.scale}")
        logger.info(f"Цветовая карта: {self.config.cmap}")
        logger.info("=" * 60)

        all_errors = []

        # Обработка cam_1
        cam1_count = 0
        cam1_csv = self._get_sum_csv_path("cam_1")
        if cam1_csv and cam1_csv.exists():
            logger.info(f"\n--- Создание векторного поля {method_name} для cam_1 ---")
            cam1_count, cam1_errors = self.process_camera("cam_1")
            all_errors.extend(cam1_errors)
        else:
            logger.warning(f"cam_1: суммарный CSV файл не найден")

        # Обработка cam_2
        cam2_count = 0
        cam2_csv = self._get_sum_csv_path("cam_2")
        if cam2_csv and cam2_csv.exists():
            logger.info(f"\n--- Создание векторного поля {method_name} для cam_2 ---")
            cam2_count, cam2_errors = self.process_camera("cam_2")
            all_errors.extend(cam2_errors)
        else:
            logger.warning(f"cam_2: суммарный CSV файл не найден")

        success = len(all_errors) == 0 and (cam1_count > 0 or cam2_count > 0)

        logger.info("\n" + "=" * 60)
        logger.info(f"РЕЗУЛЬТАТЫ СОЗДАНИЯ ВЕКТОРНЫХ ПОЛЕЙ {method_name.upper()}")
        logger.info("=" * 60)
        logger.info(f"cam_1: {cam1_count} векторов")
        logger.info(f"cam_2: {cam2_count} векторов")
        logger.info(f"Выходная папка: {self.output_folder}")
        logger.info("=" * 60)

        return FarnebackVectorFieldResult(
            success=success,
            cam1_vectors_count=cam1_count,
            cam2_vectors_count=cam2_count,
            method=self.method,
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
        if self.farneback_folder is None:
            return None

        csv_path = self._get_sum_csv_path(camera_name)

        if csv_path is None or not csv_path.exists():
            return None

        vectors = self._load_vectors_csv(csv_path)

        if not vectors:
            return None

        try:
            return self.create_vector_field(vectors, camera_name)
        except RuntimeError:
            return None

    def get_statistics(self, camera_name: str) -> Optional[dict]:
        """
        Получение статистики векторов для камеры.

        Args:
            camera_name: Название камеры

        Returns:
            Словарь со статистикой или None
        """
        if self.farneback_folder is None:
            return None

        csv_path = self._get_sum_csv_path(camera_name)

        if csv_path is None or not csv_path.exists():
            return None

        vectors = self._load_vectors_csv(csv_path)

        if not vectors:
            return {'vectors_count': 0}

        magnitudes = [v.magnitude for v in vectors]
        u_values = [v.u for v in vectors]
        v_values = [v.v for v in vectors]
        angles = [v.angle for v in vectors]

        non_zero_magnitudes = [m for m in magnitudes if m > 0]

        stats = {
            'vectors_count': len(vectors),
            'vectors_with_velocity': len(non_zero_magnitudes),
            'method': self.method
        }

        if non_zero_magnitudes:
            stats.update({
                'mean_magnitude': float(np.mean(non_zero_magnitudes)),
                'max_magnitude': float(np.max(non_zero_magnitudes)),
                'min_magnitude': float(np.min(non_zero_magnitudes)),
                'std_magnitude': float(np.std(non_zero_magnitudes)),
                'mean_u': float(np.mean([u for u, m in zip(u_values, magnitudes) if m > 0])),
                'mean_v': float(np.mean([v for v, m in zip(v_values, magnitudes) if m > 0])),
                'std_u': float(np.std([u for u, m in zip(u_values, magnitudes) if m > 0])),
                'std_v': float(np.std([v for v, m in zip(v_values, magnitudes) if m > 0])),
                'mean_angle': float(np.mean([a for a, m in zip(angles, magnitudes) if m > 0])),
                'std_angle': float(np.std([a for a, m in zip(angles, magnitudes) if m > 0]))
            })
        else:
            stats.update({
                'mean_magnitude': 0,
                'mean_u': 0,
                'mean_v': 0,
                'mean_angle': 0
            })

        return stats
