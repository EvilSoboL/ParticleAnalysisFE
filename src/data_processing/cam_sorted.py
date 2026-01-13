"""
Модуль начальной обработки данных для проекта ParticleAnalysis.

Отвечает за сортировку изображений по камерам и предварительную обработку:
- Создание структуры папок cam_1/ и cam_2/
- Сортировка изображений по алгоритму чередования
- Отражение изображений cam_1 по горизонтальной оси
- Валидация формата изображений (16-битный PNG)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CamSorter:
    """
    Класс для сортировки и обработки изображений с двух камер.
    """

    def __init__(self, input_folder: str):
        """
        Инициализация сортировщика камер.

        Args:
            input_folder: Путь к входной папке с изображениями
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(f"{input_folder}_cam_sorted")
        self.cam1_folder = self.output_folder / "cam_1"
        self.cam2_folder = self.output_folder / "cam_2"

        if not self.input_folder.exists():
            raise FileNotFoundError(f"Входная папка не найдена: {input_folder}")

    def _create_output_structure(self) -> None:
        """
        Создает структуру выходных папок.
        """
        self.output_folder.mkdir(exist_ok=True)
        self.cam1_folder.mkdir(exist_ok=True)
        self.cam2_folder.mkdir(exist_ok=True)
        logger.info(f"Создана структура папок: {self.output_folder}")

    def _get_sorted_images(self) -> List[Path]:
        """
        Получает отсортированный список PNG изображений из входной папки.

        Returns:
            Список путей к изображениям, отсортированный по имени
        """
        images = sorted(self.input_folder.glob("*.png"))

        if not images:
            raise ValueError(f"В папке {self.input_folder} не найдено PNG изображений")

        logger.info(f"Найдено {len(images)} изображений")
        return images

    def _validate_image_format(self, image_path: Path) -> bool:
        """
        Проверяет, что изображение имеет 16-битный формат PNG.

        Args:
            image_path: Путь к изображению

        Returns:
            True, если формат корректный
        """
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

    def _flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        """
        Отражает изображение относительно горизонтальной оси (flip по оси X).

        Args:
            image: Входное изображение

        Returns:
            Отраженное изображение
        """
        return cv2.flip(image, 0)  # 0 - отражение по горизонтальной оси

    def _process_and_save_image(
        self,
        image_path: Path,
        output_path: Path,
        flip: bool = False
    ) -> bool:
        """
        Обрабатывает и сохраняет изображение.

        Args:
            image_path: Путь к входному изображению
            output_path: Путь для сохранения
            flip: Применять ли отражение по горизонтали

        Returns:
            True, если обработка прошла успешно
        """
        # Загрузка 16-битного изображения
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            logger.error(f"Не удалось загрузить изображение: {image_path}")
            return False

        # Применение отражения, если требуется
        if flip:
            img = self._flip_horizontal(img)

        # Сохранение изображения
        cv2.imwrite(str(output_path), img)
        return True

    def sort_images(self, validate_format: bool = True) -> Tuple[int, int]:
        """
        Сортирует изображения по камерам согласно алгоритму.

        Алгоритм:
        - Первые две фотографии → cam_1/ (с отражением)
        - Следующие две фотографии → cam_2/ (без изменений)
        - Цикл повторяется

        Args:
            validate_format: Выполнять ли валидацию формата изображений

        Returns:
            Кортеж (количество пар в cam_1, количество пар в cam_2)
        """
        # Создание структуры папок
        self._create_output_structure()

        # Получение списка изображений
        images = self._get_sorted_images()

        # Валидация формата, если требуется
        if validate_format:
            logger.info("Проверка формата изображений...")
            for img_path in images:
                if not self._validate_image_format(img_path):
                    raise ValueError(
                        f"Изображение {img_path.name} не прошло валидацию формата"
                    )
            logger.info("Все изображения прошли валидацию формата ✓")

        # Счетчики для cam_1 и cam_2
        cam1_count = 0
        cam2_count = 0

        # Сортировка изображений
        logger.info("Начало сортировки изображений...")

        for i, img_path in enumerate(images):
            # Определяем, в какую группу попадает изображение
            # i % 4 даёт позицию в цикле из 4 изображений
            position_in_cycle = i % 4

            if position_in_cycle < 2:
                # Первые две позиции → cam_1 с отражением
                output_path = self.cam1_folder / img_path.name
                success = self._process_and_save_image(img_path, output_path, flip=True)

                if success:
                    cam1_count += 1
                    logger.debug(
                        f"{img_path.name} → cam_1/ (отражено по горизонтали)"
                    )
            else:
                # Следующие две позиции → cam_2 без изменений
                output_path = self.cam2_folder / img_path.name
                success = self._process_and_save_image(img_path, output_path, flip=False)

                if success:
                    cam2_count += 1
                    logger.debug(f"{img_path.name} → cam_2/ (без изменений)")

        # Вычисление количества пар
        cam1_pairs = cam1_count // 2
        cam2_pairs = cam2_count // 2

        # Вывод результатов
        logger.info("=" * 60)
        logger.info("Сортировка завершена!")
        logger.info(f"Выходная папка: {self.output_folder}")
        logger.info("-" * 60)
        logger.info(f"cam_1: {cam1_pairs} пар изображений ({cam1_count} файлов)")
        logger.info("       Все изображения отзеркалены относительно горизонтальной оси")
        logger.info(f"cam_2: {cam2_pairs} пар изображений ({cam2_count} файлов)")
        logger.info("       Изображения без изменений")
        logger.info("=" * 60)

        return cam1_pairs, cam2_pairs

    def get_statistics(self) -> dict:
        """
        Возвращает статистику обработанных изображений.

        Returns:
            Словарь со статистикой
        """
        cam1_images = list(self.cam1_folder.glob("*.png"))
        cam2_images = list(self.cam2_folder.glob("*.png"))

        return {
            'output_folder': str(self.output_folder),
            'cam1_count': len(cam1_images),
            'cam1_pairs': len(cam1_images) // 2,
            'cam2_count': len(cam2_images),
            'cam2_pairs': len(cam2_images) // 2,
            'total_images': len(cam1_images) + len(cam2_images)
        }
