"""
Модуль визуализации векторного поля для PTV анализа.

Создает векторное поле смещений частиц для каждой камеры на основе
суммарных CSV файлов (cam_1_pairs_sum.csv, cam_2_pairs_sum.csv).

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import numpy as np
import cv2
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
    background_color: Tuple[int, int, int] = (255, 255, 255)  # Белый фон (BGR)
    arrow_color: Tuple[int, int, int] = (0, 0, 255)  # Красные стрелки (BGR)
    arrow_thickness: int = 2
    arrow_tip_length: float = 0.3  # Длина наконечника стрелки (доля от длины)
    scale_factor: float = 1.0  # Масштаб для векторов (1.0 = без изменений)
    draw_start_points: bool = True  # Рисовать ли начальные точки
    start_point_color: Tuple[int, int, int] = (0, 255, 0)  # Зеленые точки (BGR)
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
            arrow_color: Цвет стрелок (BGR)
            arrow_thickness: Толщина стрелок
            scale_factor: Масштаб векторов
            background_color: Цвет фона (BGR)
            draw_start_points: Рисовать ли начальные точки
            start_point_color: Цвет начальных точек (BGR)
        """
        if image_width is not None:
            self.config.image_width = image_width
        if image_height is not None:
            self.config.image_height = image_height
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
        Создание изображения векторного поля.

        Args:
            vectors: Список векторов смещения

        Returns:
            Изображение с векторным полем (BGR)
        """
        # Создание пустого изображения с фоном
        image = np.full(
            (self.config.image_height, self.config.image_width, 3),
            self.config.background_color,
            dtype=np.uint8
        )

        cfg = self.config

        # Рисуем стрелки для каждого вектора
        for vector in vectors:
            # Начальная точка
            x_start = int(round(vector.x0))
            y_start = int(round(vector.y0))

            # Конечная точка с учетом масштаба
            x_end = int(round(vector.x0 + vector.dx * cfg.scale_factor))
            y_end = int(round(vector.y0 + vector.dy * cfg.scale_factor))

            # Проверка, что точки находятся в пределах изображения
            if (0 <= x_start < cfg.image_width and
                0 <= y_start < cfg.image_height):

                # Рисуем начальную точку, если включено
                if cfg.draw_start_points:
                    cv2.circle(image, (x_start, y_start),
                             cfg.start_point_radius,
                             cfg.start_point_color, -1)

                # Рисуем стрелку только если есть смещение
                if vector.length > 0:
                    # Ограничиваем конечную точку границами изображения
                    x_end = max(0, min(x_end, cfg.image_width - 1))
                    y_end = max(0, min(y_end, cfg.image_height - 1))

                    cv2.arrowedLine(
                        image,
                        (x_start, y_start),
                        (x_end, y_end),
                        cfg.arrow_color,
                        cfg.arrow_thickness,
                        tipLength=cfg.arrow_tip_length
                    )

        return image

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
        logger.info(f"Размер изображения: {self.config.image_width}x{self.config.image_height}")
        logger.info(f"Масштаб векторов: {self.config.scale_factor}")
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
