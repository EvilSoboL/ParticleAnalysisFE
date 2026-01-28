"""
Модуль фильтрации изображений для алгоритма Farneback/Lucas-Kanade.

Этот модуль предназначен для подготовки пар изображений к оптическому потоку:
- Пороговая фильтрация (зануление значений ниже порога)
- Нормализация _a относительно _b по средней интенсивности
- Нормализация в заданный диапазон (по умолчанию 50-255)
- Преобразование из 16-бит в 8-бит

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import numpy as np
from PIL import Image
from typing import Optional, Callable, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FilterProgress:
    """Класс для передачи информации о прогрессе обработки."""
    current_file: str
    total_files: int
    processed_files: int
    current_camera: str
    percentage: float
    message: str


@dataclass
class FilterResult:
    """Результат фильтрации."""
    success: bool
    total_processed: int
    cam1_processed: int
    cam2_processed: int
    errors: List[str]
    output_folder: str
    threshold: int
    output_range: Tuple[int, int]


@dataclass
class PairStatistics:
    """Статистика пары изображений после обработки."""
    mean_a_original: float
    mean_b_original: float
    mean_a_after_threshold: float
    mean_b_after_threshold: float
    scale_factor: float
    global_max: float
    pixels_zeroed_a_percent: float
    pixels_zeroed_b_percent: float


class FarnebackKanadeFilter:
    """
    Класс для подготовки пар изображений к алгоритмам оптического потока.

    Выполняет:
    1. Пороговую фильтрацию (значения ниже порога обнуляются)
    2. Нормализацию _a относительно _b по средней интенсивности
    3. Нормализацию обоих изображений в заданный диапазон
    4. Преобразование в 8-битный формат
    """

    def __init__(self):
        """Инициализация модуля фильтрации."""
        self.input_folder: Optional[Path] = None
        self.output_folder: Optional[Path] = None
        self.threshold: int = 2000
        self.output_range: Tuple[int, int] = (50, 255)
        self.intensity_boost: float = 1.5
        self._cancel_requested: bool = False
        self._progress_callback: Optional[Callable[[FilterProgress], None]] = None

        logger.info("Инициализирован модуль Farneback/Kanade фильтрации")

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
        Установка порогового значения фильтрации.

        Args:
            threshold: Пороговое значение (0-65535).
                      Пиксели с интенсивностью ниже порога станут нулевыми.

        Returns:
            bool: True если значение валидно, False иначе
        """
        if not (0 <= threshold <= 65535):
            logger.error(f"Порог вне диапазона [0, 65535]: {threshold}")
            return False

        self.threshold = threshold
        self._update_output_folder()

        logger.info(f"Установлен порог фильтрации: {threshold}")
        return True

    def set_output_range(self, out_min: int, out_max: int) -> bool:
        """
        Установка диапазона нормализации выходных значений.

        Args:
            out_min: Минимальное значение (для ненулевых пикселей)
            out_max: Максимальное значение

        Returns:
            bool: True если значения валидны, False иначе
        """
        if not (0 <= out_min < out_max <= 255):
            logger.error(f"Некорректный диапазон: [{out_min}, {out_max}]")
            return False

        self.output_range = (out_min, out_max)
        logger.info(f"Установлен диапазон нормализации: {self.output_range}")
        return True

    def set_intensity_boost(self, boost: float) -> bool:
        """
        Установка коэффициента усиления интенсивности.

        Args:
            boost: Коэффициент усиления (обычно 1.0-2.0)

        Returns:
            bool: True если значение валидно, False иначе
        """
        if boost <= 0:
            logger.error(f"Коэффициент усиления должен быть положительным: {boost}")
            return False

        self.intensity_boost = boost
        logger.info(f"Установлен коэффициент усиления: {boost}")
        return True

    def _update_output_folder(self) -> None:
        """Обновление пути выходной папки на основе параметров."""
        if self.input_folder is not None:
            output_name = f"farneback_filtered_{self.threshold}"
            self.output_folder = self.input_folder / output_name
            logger.info(f"Выходная папка: {self.output_folder}")

    def set_progress_callback(self, callback: Callable[[FilterProgress], None]) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция, принимающая FilterProgress
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

    def _save_8bit_image(self, image_array: np.ndarray, output_path: Path) -> bool:
        """
        Сохранение 8-битного PNG изображения.

        Args:
            image_array: Массив изображения (uint8)
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

    def process_pair(self, img_a_path: Path, img_b_path: Path,
                     output_dir: Optional[Path] = None) -> Optional[Tuple[np.ndarray, np.ndarray, PairStatistics]]:
        """
        Обработка пары изображений (_a и _b).

        Шаги обработки:
        1. Загрузка 16-битных изображений
        2. Зануление значений ниже порога
        3. Нормализация _a относительно _b по средней интенсивности
        4. Нормализация обоих изображений в output_range
        5. Преобразование в 8-битный формат

        Args:
            img_a_path: Путь к первому изображению (..._a.png)
            img_b_path: Путь ко второму изображению (..._b.png)
            output_dir: Директория для сохранения результатов (опционально)

        Returns:
            Tuple[img_a_8bit, img_b_8bit, statistics] или None при ошибке
        """
        # Шаг 1: Загрузка 16-битных изображений
        img_a = self._load_16bit_image(img_a_path)
        img_b = self._load_16bit_image(img_b_path)

        if img_a is None or img_b is None:
            return None

        # Шаг 2: Зануление значений ниже порога
        img_a_thresh = np.where(img_a < self.threshold, 0, img_a).astype(np.float64)
        img_b_thresh = np.where(img_b < self.threshold, 0, img_b).astype(np.float64)

        pixels_zeroed_a = np.sum(img_a < self.threshold)
        pixels_zeroed_b = np.sum(img_b < self.threshold)
        total_pixels = img_a.size

        pixels_zeroed_a_percent = 100 * pixels_zeroed_a / total_pixels
        pixels_zeroed_b_percent = 100 * pixels_zeroed_b / total_pixels

        # Шаг 3: Нормализация _a относительно _b по средней интенсивности
        mask_a_nonzero = img_a_thresh > 0
        mask_b_nonzero = img_b_thresh > 0

        mean_a_original = float(img_a[img_a > 0].mean()) if (img_a > 0).any() else 0.0
        mean_b_original = float(img_b[img_b > 0].mean()) if (img_b > 0).any() else 0.0

        mean_a = float(img_a_thresh[mask_a_nonzero].mean()) if mask_a_nonzero.any() else 0.0
        mean_b = float(img_b_thresh[mask_b_nonzero].mean()) if mask_b_nonzero.any() else 0.0

        if mean_a > 0:
            scale_factor = (mean_b / mean_a) * self.intensity_boost
            img_a_scaled = img_a_thresh * scale_factor
        else:
            scale_factor = self.intensity_boost
            img_a_scaled = img_a_thresh * self.intensity_boost

        img_b_scaled = img_b_thresh.copy()

        # Шаг 4: Нормализация в диапазон output_range
        out_min, out_max = self.output_range
        global_max = max(img_a_scaled.max(), img_b_scaled.max())

        if global_max > 0:
            # Нормализуем: ненулевые значения переводим в [out_min, out_max]
            mask_a = img_a_scaled > 0
            img_a_norm = np.zeros_like(img_a_scaled)
            if mask_a.any():
                img_a_norm[mask_a] = out_min + (img_a_scaled[mask_a] / global_max) * (out_max - out_min)

            mask_b = img_b_scaled > 0
            img_b_norm = np.zeros_like(img_b_scaled)
            if mask_b.any():
                img_b_norm[mask_b] = out_min + (img_b_scaled[mask_b] / global_max) * (out_max - out_min)
        else:
            img_a_norm = img_a_scaled.copy()
            img_b_norm = img_b_scaled.copy()

        # Шаг 5: Преобразование в 8-бит
        img_a_8bit = np.clip(img_a_norm, 0, 255).astype(np.uint8)
        img_b_8bit = np.clip(img_b_norm, 0, 255).astype(np.uint8)

        # Статистика
        statistics = PairStatistics(
            mean_a_original=mean_a_original,
            mean_b_original=mean_b_original,
            mean_a_after_threshold=mean_a,
            mean_b_after_threshold=mean_b,
            scale_factor=scale_factor,
            global_max=float(global_max),
            pixels_zeroed_a_percent=pixels_zeroed_a_percent,
            pixels_zeroed_b_percent=pixels_zeroed_b_percent
        )

        # Сохранение результатов
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            name_a = img_a_path.stem + ".png"
            name_b = img_b_path.stem + ".png"

            self._save_8bit_image(img_a_8bit, output_dir / name_a)
            self._save_8bit_image(img_b_8bit, output_dir / name_b)

        return img_a_8bit, img_b_8bit, statistics

    def _find_image_pairs(self, camera_path: Path) -> List[Tuple[Path, Path]]:
        """
        Поиск пар изображений (_a и _b) в папке камеры.

        Args:
            camera_path: Путь к папке камеры

        Returns:
            Список пар (path_a, path_b)
        """
        pairs = []
        a_files = sorted(camera_path.glob("*_a.png"))

        for a_file in a_files:
            b_file = camera_path / a_file.name.replace("_a.png", "_b.png")
            if b_file.exists():
                pairs.append((a_file, b_file))
            else:
                logger.warning(f"Не найден парный файл для {a_file.name}")

        return pairs

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

        pairs = self._find_image_pairs(camera_input)
        total_pairs = len(pairs)

        if total_pairs == 0:
            logger.warning(f"В {camera_name} нет пар изображений")
            return 0, [f"Нет пар изображений в {camera_name}"]

        processed = 0
        errors = []

        for idx, (img_a_path, img_b_path) in enumerate(pairs):
            if self._cancel_requested:
                logger.info(f"Обработка {camera_name} отменена")
                break

            if self._progress_callback:
                progress = FilterProgress(
                    current_file=img_a_path.stem,
                    total_files=total_pairs,
                    processed_files=idx,
                    current_camera=camera_name,
                    percentage=(idx / total_pairs) * 100,
                    message=f"Обработка {camera_name}: пара {img_a_path.stem}"
                )
                self._progress_callback(progress)

            result = self.process_pair(img_a_path, img_b_path, camera_output)

            if result is not None:
                processed += 1
            else:
                errors.append(f"Ошибка обработки пары: {img_a_path.stem}")

        if self._progress_callback and not self._cancel_requested:
            progress = FilterProgress(
                current_file="",
                total_files=total_pairs,
                processed_files=total_pairs,
                current_camera=camera_name,
                percentage=100.0,
                message=f"{camera_name}: завершено"
            )
            self._progress_callback(progress)

        return processed, errors

    def process_all(self) -> FilterResult:
        """
        Обработка всех изображений с отслеживанием прогресса.

        Returns:
            FilterResult с результатами обработки
        """
        if self.input_folder is None:
            return FilterResult(
                success=False,
                total_processed=0,
                cam1_processed=0,
                cam2_processed=0,
                errors=["Входная папка не установлена"],
                output_folder="",
                threshold=self.threshold,
                output_range=self.output_range
            )

        self._update_output_folder()

        logger.info("=" * 60)
        logger.info("НАЧАЛО FARNEBACK/KANADE ФИЛЬТРАЦИИ")
        logger.info(f"Входная папка: {self.input_folder}")
        logger.info(f"Выходная папка: {self.output_folder}")
        logger.info(f"Порог: {self.threshold}")
        logger.info(f"Диапазон нормализации: {self.output_range}")
        logger.info(f"Коэффициент усиления: {self.intensity_boost}")
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
        logger.info("РЕЗУЛЬТАТЫ ФИЛЬТРАЦИИ")
        logger.info("=" * 60)
        logger.info(f"cam_1: обработано {cam1_processed} пар, ошибок: {len(cam1_errors)}")
        logger.info(f"cam_2: обработано {cam2_processed} пар, ошибок: {len(cam2_errors)}")
        logger.info(f"Всего пар: {total_processed}")
        logger.info(f"Выходная папка: {self.output_folder}")

        if self._cancel_requested:
            logger.info("Обработка была отменена")

        logger.info("=" * 60)

        return FilterResult(
            success=success,
            total_processed=total_processed,
            cam1_processed=cam1_processed,
            cam2_processed=cam2_processed,
            errors=all_errors,
            output_folder=str(self.output_folder),
            threshold=self.threshold,
            output_range=self.output_range
        )

    def preview_pair(self, img_a_path: Path, img_b_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PairStatistics]]:
        """
        Предварительный просмотр обработки пары изображений.

        Args:
            img_a_path: Путь к первому изображению
            img_b_path: Путь ко второму изображению

        Returns:
            Tuple[original_a, original_b, filtered_a, filtered_b, statistics] или None
        """
        original_a = self._load_16bit_image(img_a_path)
        original_b = self._load_16bit_image(img_b_path)

        if original_a is None or original_b is None:
            return None

        result = self.process_pair(img_a_path, img_b_path)

        if result is None:
            return None

        filtered_a, filtered_b, statistics = result

        return original_a, original_b, filtered_a, filtered_b, statistics
