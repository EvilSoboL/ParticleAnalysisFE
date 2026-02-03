"""
Комбинированный модуль сортировки и бинаризации изображений.

Объединяет функциональность CamSorter и BinarizationFilter в один проход:
- Читает сырые 16-bit PNG изображения
- Сортирует по камерам (cam_1 / cam_2) по алгоритму чередования
- Для cam_1 применяет отражение по горизонтальной оси
- Бинаризует по пороговому значению
- Сохраняет 8-bit PNG (0/255) напрямую, без промежуточных 16-bit копий

Выходная структура совместима с существующим PTV пайплайном.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass, field
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SortBinarizeProgress:
    """Информация о прогрессе обработки для GUI."""
    current_file: str
    total_files: int
    processed_files: int
    current_camera: str
    percentage: float
    message: str


@dataclass
class SortBinarizeResult:
    """Результат комбинированной сортировки и бинаризации."""
    success: bool
    cam1_count: int
    cam2_count: int
    total_processed: int
    output_folder: str
    threshold: int
    errors: List[str] = field(default_factory=list)


class SortAndBinarize:
    """
    Комбинированный класс: сортировка по камерам + бинаризация за один проход.

    Алгоритм:
    - Первые 2 фото в цикле → cam_1 (с отражением по горизонтали)
    - Следующие 2 фото → cam_2 (без изменений)
    - Цикл повторяется

    Бинаризация:
    - Пиксели >= порога → 255 (белые)
    - Пиксели < порога → 0 (черные)
    - Результат сохраняется как 8-bit PNG
    """

    def __init__(self, input_folder: str, threshold: int = 10000,
                 validate_format: bool = True):
        """
        Инициализация.

        Args:
            input_folder: Путь к папке с сырыми 16-bit PNG изображениями
            threshold: Пороговое значение бинаризации (0-65535)
            validate_format: Проверять ли что изображения 16-bit
        """
        self.input_folder = Path(input_folder)
        self.threshold = threshold
        self.validate_format = validate_format
        self._cancel_requested = False
        self._progress_callback: Optional[Callable[[SortBinarizeProgress], None]] = None

        if not self.input_folder.exists():
            raise FileNotFoundError(f"Входная папка не найдена: {input_folder}")

        # Выходная структура: {input}_cam_sorted/binary_filter_{threshold}/cam_1|cam_2
        self.sorted_folder = Path(f"{input_folder}_cam_sorted")
        self.output_folder = self.sorted_folder / f"binary_filter_{threshold}"
        self.cam1_folder = self.output_folder / "cam_1"
        self.cam2_folder = self.output_folder / "cam_2"

    def set_progress_callback(self, callback: Callable[[SortBinarizeProgress], None]) -> None:
        """Установка callback функции для отслеживания прогресса."""
        self._progress_callback = callback

    def cancel_processing(self) -> None:
        """Запрос на отмену обработки."""
        self._cancel_requested = True
        logger.info("Запрошена отмена обработки")

    def _create_output_structure(self) -> None:
        """Создает структуру выходных папок."""
        self.sorted_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
        self.cam1_folder.mkdir(exist_ok=True)
        self.cam2_folder.mkdir(exist_ok=True)
        logger.info(f"Создана структура папок: {self.output_folder}")

    def _get_sorted_images(self) -> List[Path]:
        """Получает отсортированный список PNG изображений из входной папки."""
        images = sorted(self.input_folder.glob("*.png"))
        if not images:
            raise ValueError(f"В папке {self.input_folder} не найдено PNG изображений")
        logger.info(f"Найдено {len(images)} изображений")
        return images

    def _generate_new_filename(self, original_name: str, pair_number: int) -> str:
        """
        Генерирует новое имя файла на основе номера пары.

        Args:
            original_name: Оригинальное имя файла
            pair_number: Номер пары (1, 2, 3, ...)

        Returns:
            Новое имя файла в формате "pair_letter.png"
            Например: "1_a.png", "1_b.png"
        """
        name_without_ext = original_name.rsplit('.', 1)[0]
        if '_' in name_without_ext:
            suffix = name_without_ext.split('_')[-1].lower()
        else:
            suffix = 'x'
        return f"{pair_number}_{suffix}.png"

    def _validate_image(self, image_path: Path) -> bool:
        """Проверяет, что изображение имеет 16-битный формат PNG."""
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error(f"Не удалось загрузить изображение: {image_path}")
            return False
        if img.dtype != np.uint16:
            logger.error(
                f"Изображение {image_path.name} не является 16-битным. "
                f"Тип данных: {img.dtype}"
            )
            return False
        return True

    def process(self) -> SortBinarizeResult:
        """
        Выполняет сортировку и бинаризацию за один проход.

        Returns:
            SortBinarizeResult с результатами обработки
        """
        self._cancel_requested = False
        errors: List[str] = []

        # Создание структуры папок
        self._create_output_structure()

        # Получение списка изображений
        try:
            images = self._get_sorted_images()
        except ValueError as e:
            return SortBinarizeResult(
                success=False, cam1_count=0, cam2_count=0,
                total_processed=0, output_folder=str(self.output_folder),
                threshold=self.threshold, errors=[str(e)]
            )

        total_files = len(images)

        # Валидация формата
        if self.validate_format:
            logger.info("Проверка формата изображений...")
            for img_path in images:
                if not self._validate_image(img_path):
                    return SortBinarizeResult(
                        success=False, cam1_count=0, cam2_count=0,
                        total_processed=0, output_folder=str(self.output_folder),
                        threshold=self.threshold,
                        errors=[f"Изображение {img_path.name} не прошло валидацию формата"]
                    )
            logger.info("Все изображения прошли валидацию формата")

        # Счётчики
        cam1_count = 0
        cam2_count = 0
        cam1_pair_counter = 1
        cam2_pair_counter = 1

        logger.info("Начало сортировки и бинаризации...")
        logger.info(f"Порог бинаризации: {self.threshold}")

        for i, img_path in enumerate(images):
            if self._cancel_requested:
                logger.info("Обработка отменена пользователем")
                errors.append("Обработка отменена пользователем")
                break

            position_in_cycle = i % 4
            is_cam1 = position_in_cycle < 2
            camera_name = "cam_1" if is_cam1 else "cam_2"

            # Прогресс
            if self._progress_callback:
                percentage = (i / total_files) * 100
                progress = SortBinarizeProgress(
                    current_file=img_path.name,
                    total_files=total_files,
                    processed_files=i,
                    current_camera=camera_name,
                    percentage=percentage,
                    message=f"Обработка {camera_name}: {img_path.name}"
                )
                self._progress_callback(progress)

            # Загрузка 16-bit изображения
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                errors.append(f"Не удалось загрузить: {img_path.name}")
                continue

            # Для cam_1: отражение по горизонтальной оси
            if is_cam1:
                img = cv2.flip(img, 0)

            # Бинаризация: 16-bit → 8-bit (0 или 255)
            binary = np.where(img >= self.threshold, 255, 0).astype(np.uint8)

            # Определение выходной папки и имени файла
            if is_cam1:
                new_filename = self._generate_new_filename(img_path.name, cam1_pair_counter)
                output_path = self.cam1_folder / new_filename
                cam1_count += 1
                if position_in_cycle == 1:
                    cam1_pair_counter += 1
            else:
                new_filename = self._generate_new_filename(img_path.name, cam2_pair_counter)
                output_path = self.cam2_folder / new_filename
                cam2_count += 1
                if position_in_cycle == 3:
                    cam2_pair_counter += 1

            # Сохранение 8-bit PNG
            success = cv2.imwrite(str(output_path), binary)
            if not success:
                errors.append(f"Ошибка сохранения: {new_filename}")
                if is_cam1:
                    cam1_count -= 1
                else:
                    cam2_count -= 1

        # Финальный прогресс
        if self._progress_callback and not self._cancel_requested:
            progress = SortBinarizeProgress(
                current_file="",
                total_files=total_files,
                processed_files=total_files,
                current_camera="completed",
                percentage=100.0,
                message="Обработка завершена"
            )
            self._progress_callback(progress)

        total_processed = cam1_count + cam2_count
        success = not self._cancel_requested and len(errors) == 0

        # Вывод результатов
        logger.info("=" * 60)
        logger.info("Сортировка и бинаризация завершены!")
        logger.info(f"Выходная папка: {self.output_folder}")
        logger.info(f"Порог: {self.threshold}")
        logger.info(f"cam_1: {cam1_count} файлов ({cam1_count // 2} пар)")
        logger.info(f"cam_2: {cam2_count} файлов ({cam2_count // 2} пар)")
        logger.info(f"Всего: {total_processed}")
        if errors:
            logger.info(f"Ошибок: {len(errors)}")
        logger.info("=" * 60)

        return SortBinarizeResult(
            success=success,
            cam1_count=cam1_count,
            cam2_count=cam2_count,
            total_processed=total_processed,
            output_folder=str(self.output_folder),
            threshold=self.threshold,
            errors=errors
        )
