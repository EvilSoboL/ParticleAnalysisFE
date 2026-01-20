"""
Модуль визуализации частиц для GUI приложения ParticleAnalysis.

Этот модуль предназначен для визуализации результатов детектирования частиц:
- Загрузка бинарных изображений из binary_filter_XXXX
- Чтение координат частиц из CSV файлов PTV анализа (PTV_XXXX)
- Отображение центров частиц и окружностей с эквивалентным диаметром
- Сохранение визуализированных изображений

Автор: ParticleAnalysis Team
Версия: 2.0
"""

import numpy as np
import cv2
from PIL import Image
from typing import Optional, Callable, List, Tuple, Dict
from pathlib import Path
from dataclasses import dataclass
import csv
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Particle:
    """Класс для хранения информации о частице."""
    id: int
    area: int
    center_x: float
    center_y: float
    diameter: float


@dataclass
class VisualizationProgress:
    """Класс для передачи информации о прогрессе визуализации."""
    current_file: str
    total_files: int
    processed_files: int
    current_camera: str
    percentage: float
    message: str


@dataclass
class VisualizationResult:
    """Результат визуализации."""
    success: bool
    total_processed: int
    cam1_processed: int
    cam2_processed: int
    total_particles_visualized: int
    errors: List[str]
    output_folder: str


@dataclass
class VisualizationConfig:
    """Конфигурация визуализации."""
    # Цвета в формате BGR (для OpenCV)
    center_color: Tuple[int, int, int] = (0, 0, 255)  # Красный для центра
    circle_color: Tuple[int, int, int] = (0, 255, 0)  # Зеленый для окружности
    center_radius: int = 2  # Радиус точки центра
    circle_thickness: int = 1  # Толщина линии окружности
    max_images: int = 10  # Максимальное количество изображений для обработки


class ParticleVisualizer:
    """
    Класс для визуализации детектированных частиц.

    Загружает бинарные изображения и накладывает на них информацию
    о частицах из CSV файлов PTV анализа.
    """

    def __init__(self):
        """Инициализация модуля визуализации."""
        self.binary_folder: Optional[Path] = None  # Папка с бинарными изображениями
        self.ptv_folder: Optional[Path] = None  # Папка с результатами PTV
        self.output_folder: Optional[Path] = None
        self.config = VisualizationConfig()
        self._cancel_requested: bool = False
        self._progress_callback: Optional[Callable[[VisualizationProgress], None]] = None

        logger.info("Инициализирован модуль визуализации частиц")

    def set_input_folders(
        self,
        binary_folder_path: str,
        ptv_folder_path: str
    ) -> bool:
        """
        Установка входных папок.

        Args:
            binary_folder_path: Путь к папке с бинарными изображениями (binary_filter_XXXX)
            ptv_folder_path: Путь к папке с результатами PTV (PTV_XXXX)

        Returns:
            bool: True если обе папки валидны, False иначе
        """
        binary_path = Path(binary_folder_path)
        ptv_path = Path(ptv_folder_path)

        # Проверка папки с бинарными изображениями
        if not binary_path.exists():
            logger.error(f"Папка с бинарными изображениями не существует: {binary_folder_path}")
            return False

        binary_cam1 = binary_path / "cam_1"
        binary_cam2 = binary_path / "cam_2"

        if not binary_cam1.exists() or not binary_cam2.exists():
            logger.error("Папка binary_filter должна содержать подпапки cam_1 и cam_2")
            return False

        # Проверка папки с результатами PTV
        if not ptv_path.exists():
            logger.error(f"Папка с результатами PTV не существует: {ptv_folder_path}")
            return False

        ptv_cam1 = ptv_path / "cam_1"
        ptv_cam2 = ptv_path / "cam_2"

        if not ptv_cam1.exists() or not ptv_cam2.exists():
            logger.error("Папка PTV должна содержать подпапки cam_1 и cam_2")
            return False

        self.binary_folder = binary_path
        self.ptv_folder = ptv_path
        self._update_output_folder()

        logger.info(f"Папка бинарных изображений: {self.binary_folder}")
        logger.info(f"Папка результатов PTV: {self.ptv_folder}")

        return True

    def _update_output_folder(self) -> None:
        """Обновление пути выходной папки."""
        if self.binary_folder is not None:
            self.output_folder = self.binary_folder.parent / "particle_visualization"
            logger.info(f"Выходная папка: {self.output_folder}")

    def set_visualization_config(
        self,
        center_color: Tuple[int, int, int] = (0, 0, 255),
        circle_color: Tuple[int, int, int] = (0, 255, 0),
        center_radius: int = 2,
        circle_thickness: int = 1,
        max_images: int = 10
    ) -> None:
        """
        Установка параметров визуализации.

        Args:
            center_color: Цвет центра частицы (BGR)
            circle_color: Цвет окружности (BGR)
            center_radius: Радиус точки центра в пикселях
            circle_thickness: Толщина линии окружности в пикселях
            max_images: Максимальное количество изображений для обработки
        """
        self.config = VisualizationConfig(
            center_color=center_color,
            circle_color=circle_color,
            center_radius=center_radius,
            circle_thickness=circle_thickness,
            max_images=max_images
        )

        logger.info(
            f"Параметры визуализации: center_color={center_color}, "
            f"circle_color={circle_color}, max_images={max_images}"
        )

    def set_progress_callback(
        self,
        callback: Callable[[VisualizationProgress], None]
    ) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция, принимающая VisualizationProgress
        """
        self._progress_callback = callback
        logger.debug("Установлен callback для прогресса")

    def cancel_processing(self) -> None:
        """Запрос на отмену обработки."""
        self._cancel_requested = True
        logger.info("Запрошена отмена обработки")

    def _load_binary_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Загрузка бинаризованного 8-битного PNG изображения.

        Args:
            image_path: Путь к изображению

        Returns:
            numpy.ndarray или None в случае ошибки
        """
        try:
            img = Image.open(image_path)

            if img.format != 'PNG':
                logger.warning(f"{image_path.name} не является PNG файлом")
                return None

            img_array = np.array(img)

            if img_array.dtype != np.uint8:
                logger.warning(
                    f"{image_path.name} не является 8-битным изображением"
                )
                return None

            return img_array

        except Exception as e:
            logger.error(f"Ошибка загрузки {image_path.name}: {e}")
            return None

    def _load_particles_from_csv(self, csv_path: Path) -> List[Particle]:
        """
        Загрузка информации о частицах из CSV файла.

        Args:
            csv_path: Путь к CSV файлу

        Returns:
            Список частиц
        """
        particles = []

        if not csv_path.exists():
            logger.warning(f"CSV файл не найден: {csv_path}")
            return particles

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')

                for row in reader:
                    particle = Particle(
                        id=int(row['ID']),
                        area=int(row['Area']),
                        center_x=float(row['Center_X']),
                        center_y=float(row['Center_Y']),
                        diameter=float(row['Diameter'])
                    )
                    particles.append(particle)

        except Exception as e:
            logger.error(f"Ошибка чтения CSV {csv_path.name}: {e}")

        return particles

    def _convert_to_color(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Преобразование grayscale изображения в цветное (BGR).

        Args:
            gray_image: Grayscale изображение

        Returns:
            Цветное BGR изображение
        """
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    def _draw_particles(
        self,
        image: np.ndarray,
        particles: List[Particle]
    ) -> np.ndarray:
        """
        Рисование частиц на изображении.

        Args:
            image: Цветное BGR изображение
            particles: Список частиц

        Returns:
            Изображение с нарисованными частицами
        """
        result = image.copy()

        for particle in particles:
            # Координаты центра (округляем до целых)
            center_x = int(round(particle.center_x))
            center_y = int(round(particle.center_y))
            center = (center_x, center_y)

            # Радиус окружности (половина эквивалентного диаметра)
            radius = int(round(particle.diameter / 2))

            # Рисуем окружность
            cv2.circle(
                result,
                center,
                radius,
                self.config.circle_color,
                self.config.circle_thickness
            )

            # Рисуем центр (заполненный круг)
            cv2.circle(
                result,
                center,
                self.config.center_radius,
                self.config.center_color,
                -1  # Заполненный круг
            )

        return result

    def _save_image(self, image: np.ndarray, output_path: Path) -> bool:
        """
        Сохранение изображения в файл.

        Args:
            image: Изображение для сохранения
            output_path: Путь для сохранения

        Returns:
            bool: True если успешно
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения {output_path.name}: {e}")
            return False

    def _get_matching_files(self, camera_name: str) -> List[Tuple[Path, Path]]:
        """
        Получение списка соответствующих файлов (изображение + CSV).

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Список кортежей (путь_к_изображению, путь_к_csv)
        """
        if self.binary_folder is None or self.ptv_folder is None:
            return []

        binary_camera = self.binary_folder / camera_name
        ptv_camera = self.ptv_folder / camera_name

        if not binary_camera.exists() or not ptv_camera.exists():
            return []

        # Получаем все PNG файлы
        png_files = sorted(binary_camera.glob("*.png"))[:self.config.max_images]

        matching_files = []

        for png_path in png_files:
            # Формируем имя соответствующего CSV файла
            # Например: 1_a.png -> 1_a.csv
            csv_name = png_path.stem + ".csv"
            csv_path = ptv_camera / csv_name

            if csv_path.exists():
                matching_files.append((png_path, csv_path))
            else:
                logger.warning(f"CSV файл не найден для {png_path.name}")

        return matching_files

    def visualize_single_image(
        self,
        image_path: Path,
        csv_path: Path
    ) -> Optional[Tuple[np.ndarray, List[Particle]]]:
        """
        Визуализация частиц на одном изображении.

        Args:
            image_path: Путь к бинарному изображению
            csv_path: Путь к CSV файлу с частицами

        Returns:
            Tuple[визуализированное_изображение, список_частиц] или None
        """
        # Загрузка изображения
        gray_image = self._load_binary_image(image_path)

        if gray_image is None:
            return None

        # Загрузка частиц из CSV
        particles = self._load_particles_from_csv(csv_path)

        # Преобразование в цветное изображение
        color_image = self._convert_to_color(gray_image)

        # Рисование частиц
        visualized = self._draw_particles(color_image, particles)

        return (visualized, particles)

    def process_camera(
        self,
        camera_name: str
    ) -> Tuple[int, int, List[str]]:
        """
        Обработка одной камеры.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Tuple[processed_count, particles_count, errors]
        """
        if self.binary_folder is None or self.ptv_folder is None:
            return 0, 0, ["Папки не установлены"]

        if self.output_folder is None:
            return 0, 0, ["Выходная папка не установлена"]

        camera_output = self.output_folder / camera_name

        # Получение списка файлов для обработки
        matching_files = self._get_matching_files(camera_name)
        total_files = len(matching_files)

        if total_files == 0:
            return 0, 0, [f"Нет файлов для обработки в {camera_name}"]

        processed = 0
        total_particles = 0
        errors = []

        for idx, (img_path, csv_path) in enumerate(matching_files):
            if self._cancel_requested:
                logger.info(f"Обработка {camera_name} отменена")
                break

            # Прогресс
            if self._progress_callback:
                progress = VisualizationProgress(
                    current_file=img_path.name,
                    total_files=total_files,
                    processed_files=idx,
                    current_camera=camera_name,
                    percentage=(idx / total_files) * 100,
                    message=f"{camera_name}: визуализация {img_path.name}"
                )
                self._progress_callback(progress)

            # Визуализация
            result = self.visualize_single_image(img_path, csv_path)

            if result is None:
                errors.append(f"Ошибка обработки: {img_path.name}")
                continue

            visualized_image, particles = result
            total_particles += len(particles)

            # Сохранение
            output_path = camera_output / f"vis_{img_path.name}"

            if self._save_image(visualized_image, output_path):
                processed += 1
                logger.info(
                    f"  {img_path.name}: {len(particles)} частиц → {output_path.name}"
                )
            else:
                errors.append(f"Ошибка сохранения: {img_path.name}")

        # Финальный прогресс
        if self._progress_callback and not self._cancel_requested:
            progress = VisualizationProgress(
                current_file="",
                total_files=total_files,
                processed_files=total_files,
                current_camera=camera_name,
                percentage=100.0,
                message=f"{camera_name}: завершено"
            )
            self._progress_callback(progress)

        return processed, total_particles, errors

    def process_all(self) -> VisualizationResult:
        """
        Обработка всех изображений с отслеживанием прогресса.

        Обрабатывает первые N изображений из cam_1 и cam_2.

        Returns:
            VisualizationResult с результатами обработки
        """
        if self.binary_folder is None or self.ptv_folder is None:
            return VisualizationResult(
                success=False,
                total_processed=0,
                cam1_processed=0,
                cam2_processed=0,
                total_particles_visualized=0,
                errors=["Входные папки не установлены"],
                output_folder=""
            )

        self._update_output_folder()

        logger.info("=" * 60)
        logger.info("НАЧАЛО ВИЗУАЛИЗАЦИИ ЧАСТИЦ")
        logger.info(f"Папка бинарных изображений: {self.binary_folder}")
        logger.info(f"Папка результатов PTV: {self.ptv_folder}")
        logger.info(f"Выходная папка: {self.output_folder}")
        logger.info(f"Макс. изображений: {self.config.max_images}")
        logger.info("=" * 60)

        self._cancel_requested = False
        all_errors = []

        # Обработка cam_1
        logger.info("\n--- Обработка cam_1 ---")
        cam1_processed, cam1_particles, cam1_errors = self.process_camera("cam_1")
        all_errors.extend(cam1_errors)

        # Обработка cam_2
        cam2_processed = 0
        cam2_particles = 0

        if not self._cancel_requested:
            logger.info("\n--- Обработка cam_2 ---")
            cam2_processed, cam2_particles, cam2_errors = self.process_camera("cam_2")
            all_errors.extend(cam2_errors)

        # Итоговые результаты
        total_processed = cam1_processed + cam2_processed
        total_particles = cam1_particles + cam2_particles
        success = not self._cancel_requested and len(all_errors) == 0

        logger.info("\n" + "=" * 60)
        logger.info("РЕЗУЛЬТАТЫ ВИЗУАЛИЗАЦИИ")
        logger.info("=" * 60)
        logger.info(f"Обработано изображений: {total_processed}")
        logger.info(f"  cam_1: {cam1_processed} изображений, {cam1_particles} частиц")
        logger.info(f"  cam_2: {cam2_processed} изображений, {cam2_particles} частиц")
        logger.info(f"Всего частиц визуализировано: {total_particles}")
        logger.info(f"Ошибок: {len(all_errors)}")
        logger.info(f"Выходная папка: {self.output_folder}")

        if self._cancel_requested:
            logger.info("Обработка была отменена")

        logger.info("=" * 60)

        return VisualizationResult(
            success=success,
            total_processed=total_processed,
            cam1_processed=cam1_processed,
            cam2_processed=cam2_processed,
            total_particles_visualized=total_particles,
            errors=all_errors,
            output_folder=str(self.output_folder) if self.output_folder else ""
        )

    def get_preview(
        self,
        image_path: Path,
        csv_path: Path
    ) -> Optional[Dict]:
        """
        Получение предварительного просмотра визуализации.

        Args:
            image_path: Путь к бинарному изображению
            csv_path: Путь к CSV файлу с частицами

        Returns:
            Словарь с результатами или None
        """
        result = self.visualize_single_image(image_path, csv_path)

        if result is None:
            return None

        visualized_image, particles = result

        return {
            'image': visualized_image,
            'particles': particles,
            'particles_count': len(particles),
            'image_shape': visualized_image.shape,
            'config': {
                'center_color': self.config.center_color,
                'circle_color': self.config.circle_color,
                'center_radius': self.config.center_radius,
                'circle_thickness': self.config.circle_thickness
            }
        }

    def create_comparison_image(
        self,
        image_path: Path,
        csv_path: Path
    ) -> Optional[np.ndarray]:
        """
        Создание изображения сравнения (оригинал и визуализация рядом).

        Args:
            image_path: Путь к бинарному изображению
            csv_path: Путь к CSV файлу с частицами

        Returns:
            Объединенное изображение или None
        """
        # Загрузка оригинала
        gray_image = self._load_binary_image(image_path)

        if gray_image is None:
            return None

        # Визуализация
        result = self.visualize_single_image(image_path, csv_path)

        if result is None:
            return None

        visualized_image, _ = result

        # Преобразование оригинала в цветное
        original_color = self._convert_to_color(gray_image)

        # Объединение изображений горизонтально
        comparison = np.hstack([original_color, visualized_image])

        return comparison
