"""
Модуль бинаризации изображений для GUI приложения ParticleAnalysis.

Этот модуль предназначен для интеграции с графическим интерфейсом и предоставляет:
- Бинаризацию 16-битных изображений по пороговому значению
- Преобразование в формат 0/255 для последующего PTV анализа
- Callback функции для отслеживания прогресса
- Методы для предварительного просмотра результатов
- Пошаговую обработку с возможностью отмены

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import numpy as np
from PIL import Image
from typing import Optional, Callable, Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BinarizationProgress:
    """Класс для передачи информации о прогрессе обработки."""
    current_file: str
    total_files: int
    processed_files: int
    current_camera: str
    percentage: float
    message: str


@dataclass
class BinarizationResult:
    """Результат бинаризации."""
    success: bool
    total_processed: int
    cam1_processed: int
    cam2_processed: int
    errors: List[str]
    output_folder: str
    threshold: int


@dataclass
class ImageStatistics:
    """Статистика изображения."""
    min_value: int
    max_value: int
    mean_value: float
    median_value: float
    std_value: float
    non_zero_pixels: int
    total_pixels: int


@dataclass
class BinarizationStatistics:
    """Статистика бинаризации."""
    white_pixels: int
    black_pixels: int
    total_pixels: int
    white_percentage: float
    black_percentage: float


class BinarizationFilter:
    """
    Класс для бинаризации изображений с поддержкой GUI.

    Бинаризация выполняется по пороговому значению:
    - Пиксели с интенсивностью >= порога → белые (255)
    - Пиксели с интенсивностью < порога → черные (0)

    Выходные изображения сохраняются в 8-битном формате PNG.
    """

    def __init__(self):
        """Инициализация модуля бинаризации."""
        self.input_folder: Optional[Path] = None
        self.output_folder: Optional[Path] = None
        self.threshold: int = 10_000
        self._cancel_requested: bool = False
        self._progress_callback: Optional[Callable[[BinarizationProgress], None]] = None

        logger.info("Инициализирован модуль бинаризации")

    def set_input_folder(self, folder_path: str) -> bool:
        """
        Установка входной папки (папка cam_sorted).

        Args:
            folder_path: Путь к папке с отсортированными изображениями

        Returns:
            bool: True если папка валидна, False иначе
        """
        path = Path(folder_path)

        if not path.exists():
            logger.error(f"Папка не существует: {folder_path}")
            return False

        cam1_path = path / "cam_1"
        cam2_path = path / "cam_2"

        if not cam1_path.exists() or not cam2_path.exists():
            logger.error("Папка должна содержать подпапки cam_1 и cam_2")
            return False

        self.input_folder = path
        self._update_output_folder()
        logger.info(f"Установлена входная папка: {self.input_folder}")

        return True

    def set_threshold(self, threshold: int) -> bool:
        """
        Установка порогового значения бинаризации.

        Args:
            threshold: Пороговое значение (0-65535).
                      Пиксели >= порога → белые (255)
                      Пиксели < порога → черные (0)

        Returns:
            bool: True если значение валидно, False иначе
        """
        if not (0 <= threshold <= 65535):
            logger.error(f"Порог вне диапазона [0, 65535]: {threshold}")
            return False

        self.threshold = threshold
        self._update_output_folder()

        logger.info(f"Установлен порог бинаризации: {threshold}")
        return True

    def _update_output_folder(self) -> None:
        """Обновление пути выходной папки на основе порогового значения."""
        if self.input_folder is not None:
            output_name = f"binary_filter_{self.threshold}"
            self.output_folder = self.input_folder / output_name
            logger.info(f"Выходная папка: {self.output_folder}")

    def set_progress_callback(self, callback: Callable[[BinarizationProgress], None]) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция, принимающая BinarizationProgress
        """
        self._progress_callback = callback
        logger.debug("Установлен callback для прогресса")

    def cancel_processing(self) -> None:
        """Запрос на отмену обработки."""
        self._cancel_requested = True
        logger.info("Запрошена отмена обработки")

    def _load_16bit_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Загрузка 16-битного PNG изображения.

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

            if img_array.dtype != np.uint16:
                logger.warning(f"{image_path.name} не является 16-битным изображением")
                return None

            return img_array

        except Exception as e:
            logger.error(f"Ошибка загрузки {image_path.name}: {e}")
            return None

    def _apply_binarization(self, image_array: np.ndarray) -> np.ndarray:
        """
        Применение бинаризации к изображению.

        Пиксели >= порога → 255 (белые)
        Пиксели < порога → 0 (черные)

        Args:
            image_array: Массив 16-битного изображения

        Returns:
            Бинаризованный 8-битный массив (0 или 255)
        """
        binary = np.where(image_array >= self.threshold, 255, 0).astype(np.uint8)
        return binary

    def _save_8bit_image(self, image_array: np.ndarray, output_path: Path) -> bool:
        """
        Сохранение 8-битного PNG изображения.

        Args:
            image_array: Массив изображения (8-бит)
            output_path: Путь для сохранения

        Returns:
            bool: True если успешно
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.fromarray(image_array, mode='L')
            img.save(output_path)
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения {output_path.name}: {e}")
            return False

    def get_image_statistics(self, image_path: Path) -> Optional[ImageStatistics]:
        """
        Получение статистики изображения.

        Args:
            image_path: Путь к изображению

        Returns:
            ImageStatistics или None
        """
        img_array = self._load_16bit_image(image_path)

        if img_array is None:
            return None

        non_zero_pixels = img_array[img_array > 0]

        if len(non_zero_pixels) == 0:
            logger.warning(f"Нет ненулевых пикселей в {image_path.name}")
            return None

        return ImageStatistics(
            min_value=int(np.min(non_zero_pixels)),
            max_value=int(np.max(non_zero_pixels)),
            mean_value=float(np.mean(non_zero_pixels)),
            median_value=float(np.median(non_zero_pixels)),
            std_value=float(np.std(non_zero_pixels)),
            non_zero_pixels=len(non_zero_pixels),
            total_pixels=img_array.size
        )

    def get_camera_statistics(self, camera_name: str,
                              sample_size: int = 5) -> Optional[ImageStatistics]:
        """
        Получение усредненной статистики для камеры.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)
            sample_size: Количество изображений для анализа

        Returns:
            ImageStatistics или None
        """
        if self.input_folder is None:
            logger.error("Входная папка не установлена")
            return None

        camera_path = self.input_folder / camera_name

        if not camera_path.exists():
            logger.error(f"Папка {camera_name} не найдена")
            return None

        png_files = sorted(camera_path.glob("*.png"))[:sample_size]

        if not png_files:
            logger.warning(f"В папке {camera_name} нет PNG файлов")
            return None

        logger.info(f"Анализ статистики {camera_name} ({len(png_files)} изображений)")

        all_values = []

        for img_path in png_files:
            img_array = self._load_16bit_image(img_path)
            if img_array is not None:
                non_zero = img_array[img_array > 0]
                all_values.extend(non_zero.flatten())

        if not all_values:
            return None

        all_values = np.array(all_values)

        return ImageStatistics(
            min_value=int(np.min(all_values)),
            max_value=int(np.max(all_values)),
            mean_value=float(np.mean(all_values)),
            median_value=float(np.median(all_values)),
            std_value=float(np.std(all_values)),
            non_zero_pixels=len(all_values),
            total_pixels=len(png_files) * all_values.size
        )

    def preview_binarization(self, image_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Предварительный просмотр бинаризации для одного изображения.

        Args:
            image_path: Путь к изображению

        Returns:
            Tuple[original, binarized] или None
        """
        original = self._load_16bit_image(image_path)

        if original is None:
            return None

        binarized = self._apply_binarization(original)

        return (original, binarized)

    def get_binarization_statistics(self, original: np.ndarray,
                                    binarized: np.ndarray) -> BinarizationStatistics:
        """
        Получение статистики бинаризации.

        Args:
            original: Оригинальное 16-битное изображение
            binarized: Бинаризованное 8-битное изображение

        Returns:
            BinarizationStatistics со статистикой
        """
        total_pixels = binarized.size
        white_pixels = np.count_nonzero(binarized)
        black_pixels = total_pixels - white_pixels

        white_percentage = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        black_percentage = (black_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        return BinarizationStatistics(
            white_pixels=white_pixels,
            black_pixels=black_pixels,
            total_pixels=total_pixels,
            white_percentage=white_percentage,
            black_percentage=black_percentage
        )

    def get_preview_statistics(self, original: np.ndarray,
                               binarized: np.ndarray) -> Dict[str, any]:
        """
        Получение статистики для предварительного просмотра (совместимость с GUI).

        Args:
            original: Оригинальное изображение
            binarized: Бинаризованное изображение

        Returns:
            Словарь со статистикой
        """
        stats = self.get_binarization_statistics(original, binarized)
        original_nonzero = np.count_nonzero(original)

        return {
            'original_nonzero': original_nonzero,
            'white_pixels': stats.white_pixels,
            'black_pixels': stats.black_pixels,
            'white_percentage': stats.white_percentage,
            'black_percentage': stats.black_percentage,
            'total_pixels': stats.total_pixels,
            'threshold': self.threshold
        }

    def process_camera(self, camera_name: str) -> Tuple[int, List[str]]:
        """
        Обработка одной камеры с отслеживанием прогресса.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Tuple[processed_count, errors]
        """
        if self.input_folder is None or self.output_folder is None:
            logger.error("Входная или выходная папка не установлена")
            return 0, ["Папки не установлены"]

        camera_input = self.input_folder / camera_name
        camera_output = self.output_folder / camera_name

        if not camera_input.exists():
            logger.warning(f"Папка {camera_name} не найдена")
            return 0, [f"Папка {camera_name} не найдена"]

        png_files = sorted(camera_input.glob("*.png"))
        total_files = len(png_files)

        if total_files == 0:
            logger.warning(f"В {camera_name} нет PNG файлов")
            return 0, [f"Нет PNG файлов в {camera_name}"]

        processed = 0
        errors = []

        for idx, img_path in enumerate(png_files):
            if self._cancel_requested:
                logger.info(f"Обработка {camera_name} отменена")
                break

            if self._progress_callback:
                progress = BinarizationProgress(
                    current_file=img_path.name,
                    total_files=total_files,
                    processed_files=idx,
                    current_camera=camera_name,
                    percentage=(idx / total_files) * 100,
                    message=f"Бинаризация {camera_name}: {img_path.name}"
                )
                self._progress_callback(progress)

            img_array = self._load_16bit_image(img_path)

            if img_array is None:
                errors.append(f"Ошибка загрузки: {img_path.name}")
                continue

            binarized = self._apply_binarization(img_array)
            output_path = camera_output / img_path.name

            if self._save_8bit_image(binarized, output_path):
                processed += 1
            else:
                errors.append(f"Ошибка сохранения: {img_path.name}")

        if self._progress_callback and not self._cancel_requested:
            progress = BinarizationProgress(
                current_file="",
                total_files=total_files,
                processed_files=total_files,
                current_camera=camera_name,
                percentage=100.0,
                message=f"{camera_name}: завершено"
            )
            self._progress_callback(progress)

        return processed, errors

    def process_all(self) -> BinarizationResult:
        """
        Обработка всех изображений с отслеживанием прогресса.

        Returns:
            BinarizationResult с результатами обработки
        """
        if self.input_folder is None:
            return BinarizationResult(
                success=False,
                total_processed=0,
                cam1_processed=0,
                cam2_processed=0,
                errors=["Входная папка не установлена"],
                output_folder="",
                threshold=self.threshold
            )

        self._update_output_folder()

        logger.info("=" * 60)
        logger.info("НАЧАЛО БИНАРИЗАЦИИ")
        logger.info(f"Входная папка: {self.input_folder}")
        logger.info(f"Выходная папка: {self.output_folder}")
        logger.info(f"Порог: {self.threshold}")
        logger.info("Пиксели >= порога → 255 (белые)")
        logger.info("Пиксели < порога → 0 (черные)")
        logger.info("=" * 60)

        self._cancel_requested = False

        logger.info("\n--- Обработка cam_1 ---")
        cam1_processed, cam1_errors = self.process_camera("cam_1")

        cam2_processed = 0
        cam2_errors = []

        if not self._cancel_requested:
            logger.info("\n--- Обработка cam_2 ---")
            cam2_processed, cam2_errors = self.process_camera("cam_2")

        total_processed = cam1_processed + cam2_processed
        all_errors = cam1_errors + cam2_errors
        success = not self._cancel_requested and len(all_errors) == 0

        logger.info("\n" + "=" * 60)
        logger.info("РЕЗУЛЬТАТЫ БИНАРИЗАЦИИ")
        logger.info("=" * 60)
        logger.info(f"cam_1: обработано {cam1_processed}, ошибок: {len(cam1_errors)}")
        logger.info(f"cam_2: обработано {cam2_processed}, ошибок: {len(cam2_errors)}")
        logger.info(f"Всего: {total_processed}")
        logger.info(f"Выходная папка: {self.output_folder}")

        if self._cancel_requested:
            logger.info("Обработка была отменена")

        logger.info("=" * 60)

        return BinarizationResult(
            success=success,
            total_processed=total_processed,
            cam1_processed=cam1_processed,
            cam2_processed=cam2_processed,
            errors=all_errors,
            output_folder=str(self.output_folder),
            threshold=self.threshold
        )
