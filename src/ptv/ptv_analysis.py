"""
Модуль трассерной визуализации частиц (PTV) для GUI приложения ParticleAnalysis.

Этот модуль предназначен для интеграции с графическим интерфейсом и предоставляет:
- Детектирование частиц на бинаризованных изображениях
- Сопоставление частиц между последовательными кадрами
- Расчет векторов смещения и скоростей
- Callback функции для отслеживания прогресса
- Пошаговую обработку с возможностью отмены

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import numpy as np
from PIL import Image
from typing import Optional, Callable, List, Tuple, Dict
from pathlib import Path
from dataclasses import dataclass, field
from scipy import ndimage
from scipy.spatial import KDTree
import csv
import logging
import re


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
class ParticlePair:
    """Класс для хранения информации о сопоставленной паре частиц."""
    id: int
    x0: float
    y0: float
    dx: float
    dy: float
    length: float
    diameter: float
    area: int


@dataclass
class PTVProgress:
    """Класс для передачи информации о прогрессе обработки."""
    current_file: str
    total_files: int
    processed_files: int
    current_camera: str
    current_stage: str  # 'detection' или 'matching'
    percentage: float
    message: str


@dataclass
class PTVResult:
    """Результат PTV анализа."""
    success: bool
    total_images_processed: int
    total_particles_detected: int
    total_pairs_matched: int
    cam1_pairs_count: int
    cam2_pairs_count: int
    errors: List[str]
    warnings: List[str]
    output_folder: str


@dataclass
class DetectionConfig:
    """Конфигурация детектирования частиц."""
    min_area: int = 4
    max_area: int = 150


@dataclass
class MatchingConfig:
    """Конфигурация сопоставления частиц."""
    max_distance: float = 30.0
    max_diameter_diff: float = 2.0


class PTVAnalyzer:
    """
    Класс для PTV анализа с поддержкой GUI.

    Выполняет детектирование частиц на бинаризованных изображениях
    и сопоставление частиц между последовательными кадрами.
    """

    def __init__(self):
        """Инициализация модуля PTV анализа."""
        self.input_folder: Optional[Path] = None
        self.output_folder: Optional[Path] = None
        self.detection_config = DetectionConfig()
        self.matching_config = MatchingConfig()
        self._cancel_requested: bool = False
        self._progress_callback: Optional[Callable[[PTVProgress], None]] = None

        logger.info("Инициализирован модуль PTV анализа")

    def set_input_folder(self, folder_path: str) -> bool:
        """
        Установка входной папки (папка binary_filter_XXXX).

        Args:
            folder_path: Путь к папке с бинаризованными изображениями

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

        # Проверка формата имени папки
        if not path.name.startswith("binary_filter_"):
            logger.warning(
                f"Имя папки не соответствует формату binary_filter_XXXX: {path.name}"
            )

        self.input_folder = path
        self._update_output_folder()
        logger.info(f"Установлена входная папка: {self.input_folder}")

        return True

    def _extract_threshold_from_folder_name(self) -> str:
        """
        Извлечение значения порога из имени входной папки.

        Returns:
            str: Значение порога или 'unknown'
        """
        if self.input_folder is None:
            return "unknown"

        folder_name = self.input_folder.name
        match = re.search(r'binary_filter_(\d+)', folder_name)

        if match:
            return match.group(1)

        return "unknown"

    def _update_output_folder(self) -> None:
        """Обновление пути выходной папки."""
        if self.input_folder is not None:
            threshold = self._extract_threshold_from_folder_name()
            output_name = f"PTV_{threshold}"
            # Выходная папка создается рядом с входной
            self.output_folder = self.input_folder.parent / output_name
            logger.info(f"Выходная папка: {self.output_folder}")

    def set_detection_config(
        self,
        min_area: int = 4,
        max_area: int = 150
    ) -> bool:
        """
        Установка параметров детектирования частиц.

        Args:
            min_area: Минимальная площадь частицы (пикс.)
            max_area: Максимальная площадь частицы (пикс.)

        Returns:
            bool: True если параметры валидны
        """
        if min_area < 1:
            logger.error(f"min_area должна быть >= 1: {min_area}")
            return False

        if max_area < min_area:
            logger.error(f"max_area должна быть >= min_area: {max_area} < {min_area}")
            return False

        self.detection_config = DetectionConfig(
            min_area=min_area,
            max_area=max_area
        )

        logger.info(
            f"Параметры детектирования: min_area={min_area}, max_area={max_area}"
        )
        return True

    def set_matching_config(
        self,
        max_distance: float = 30.0,
        max_diameter_diff: float = 2.0
    ) -> bool:
        """
        Установка параметров сопоставления частиц.

        Args:
            max_distance: Максимальный радиус поиска соответствия (пикс.)
            max_diameter_diff: Максимальная разница диаметров (пикс.)

        Returns:
            bool: True если параметры валидны
        """
        if max_distance <= 0:
            logger.error(f"max_distance должен быть > 0: {max_distance}")
            return False

        if max_diameter_diff < 0:
            logger.error(f"max_diameter_diff должен быть >= 0: {max_diameter_diff}")
            return False

        self.matching_config = MatchingConfig(
            max_distance=max_distance,
            max_diameter_diff=max_diameter_diff
        )

        logger.info(
            f"Параметры сопоставления: max_distance={max_distance}, "
            f"max_diameter_diff={max_diameter_diff}"
        )
        return True

    def set_progress_callback(
        self,
        callback: Callable[[PTVProgress], None]
    ) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция, принимающая PTVProgress
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

    def detect_particles(self, image_array: np.ndarray) -> List[Particle]:
        """
        Детектирование частиц на бинаризованном изображении.

        Использует 4-связность для поиска связных компонент.

        Args:
            image_array: Бинаризованное изображение (0 или 255)

        Returns:
            Список обнаруженных частиц
        """
        # Преобразование в бинарную маску (0 и 1)
        binary_mask = (image_array > 0).astype(np.uint8)

        # Поиск связных компонент с 4-связностью
        # structure определяет связность: 4-связность = только по сторонам
        structure = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=np.uint8)

        labeled_array, num_features = ndimage.label(binary_mask, structure=structure)

        particles = []
        particle_id = 1

        for label_idx in range(1, num_features + 1):
            # Получение маски текущей компоненты
            component_mask = (labeled_array == label_idx)

            # Вычисление площади
            area = np.sum(component_mask)

            # Фильтрация по площади
            if area < self.detection_config.min_area:
                continue
            if area > self.detection_config.max_area:
                continue

            # Получение координат пикселей компоненты
            coords = np.argwhere(component_mask)

            # Вычисление центра масс
            center_y = np.mean(coords[:, 0])
            center_x = np.mean(coords[:, 1])

            # Вычисление эквивалентного диаметра: D = 2 * sqrt(S / pi)
            diameter = 2 * np.sqrt(area / np.pi)

            particle = Particle(
                id=particle_id,
                area=int(area),
                center_x=float(center_x),
                center_y=float(center_y),
                diameter=float(diameter)
            )

            particles.append(particle)
            particle_id += 1

        return particles

    def match_particles(
        self,
        particles_a: List[Particle],
        particles_b: List[Particle]
    ) -> List[ParticlePair]:
        """
        Сопоставление частиц между двумя кадрами.

        Использует метод One-to-One matching с KD-деревом.

        Args:
            particles_a: Частицы из кадра a (момент t₀)
            particles_b: Частицы из кадра b (момент t₀ + Δt)

        Returns:
            Список сопоставленных пар
        """
        if not particles_a or not particles_b:
            return []

        # Построение KD-дерева по координатам частиц кадра b
        coords_b = np.array([
            [p.center_x, p.center_y] for p in particles_b
        ])
        tree = KDTree(coords_b)

        # Множество использованных частиц из кадра b
        used_b_indices = set()

        pairs = []
        pair_id = 1

        for particle_a in particles_a:
            # Поиск кандидатов в радиусе max_distance
            point_a = [particle_a.center_x, particle_a.center_y]
            candidate_indices = tree.query_ball_point(
                point_a,
                self.matching_config.max_distance
            )

            best_candidate = None
            best_score = float('inf')

            for idx in candidate_indices:
                # Пропускаем уже использованные частицы
                if idx in used_b_indices:
                    continue

                particle_b = particles_b[idx]

                # Фильтрация по разнице диаметров
                diameter_diff = abs(particle_a.diameter - particle_b.diameter)
                if diameter_diff > self.matching_config.max_diameter_diff:
                    continue

                # Вычисление расстояния
                dx = particle_b.center_x - particle_a.center_x
                dy = particle_b.center_y - particle_a.center_y
                distance = np.sqrt(dx**2 + dy**2)

                # Комбинированная метрика: score = dist + Δd × 5
                score = distance + diameter_diff * 5

                if score < best_score:
                    best_score = score
                    best_candidate = (idx, particle_b, dx, dy)

            if best_candidate is not None:
                idx, particle_b, dx, dy = best_candidate
                used_b_indices.add(idx)

                # Вычисление модуля вектора смещения
                length = np.sqrt(dx**2 + dy**2)

                pair = ParticlePair(
                    id=pair_id,
                    x0=particle_a.center_x,
                    y0=particle_a.center_y,
                    dx=dx,
                    dy=dy,
                    length=length,
                    diameter=particle_a.diameter,
                    area=particle_a.area
                )

                pairs.append(pair)
                pair_id += 1

        return pairs

    def _save_particles_csv(
        self,
        particles: List[Particle],
        output_path: Path
    ) -> bool:
        """
        Сохранение списка частиц в CSV файл.

        Args:
            particles: Список частиц
            output_path: Путь для сохранения

        Returns:
            bool: True если успешно
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')

                # Заголовки
                writer.writerow(['ID', 'Area', 'Center_X', 'Center_Y', 'Diameter'])

                # Данные
                for p in particles:
                    writer.writerow([
                        p.id,
                        p.area,
                        f"{p.center_x:.2f}",
                        f"{p.center_y:.2f}",
                        f"{p.diameter:.2f}"
                    ])

            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения {output_path.name}: {e}")
            return False

    def _save_pairs_csv(
        self,
        pairs: List[ParticlePair],
        output_path: Path
    ) -> bool:
        """
        Сохранение списка сопоставленных пар в CSV файл.

        Args:
            pairs: Список пар
            output_path: Путь для сохранения

        Returns:
            bool: True если успешно
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')

                # Заголовки
                writer.writerow([
                    'ID', 'X0', 'Y0', 'dx', 'dy', 'L', 'Diameter', 'Area'
                ])

                # Данные
                for p in pairs:
                    writer.writerow([
                        p.id,
                        f"{p.x0:.2f}",
                        f"{p.y0:.2f}",
                        f"{p.dx:.2f}",
                        f"{p.dy:.2f}",
                        f"{p.length:.2f}",
                        f"{p.diameter:.2f}",
                        p.area
                    ])

            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения {output_path.name}: {e}")
            return False

    def _save_summary_pairs_csv(
        self,
        pairs_folder: Path,
        output_path: Path
    ) -> bool:
        """
        Создание суммарного CSV файла со всеми парами из папки.

        Читает все CSV файлы из указанной папки, объединяет их в один файл
        с перенумерацией ID.

        Args:
            pairs_folder: Папка с CSV файлами пар (например, cam_1_pairs)
            output_path: Путь для сохранения суммарного файла

        Returns:
            bool: True если успешно
        """
        try:
            if not pairs_folder.exists():
                logger.warning(f"Папка {pairs_folder} не существует")
                return False

            # Получение всех CSV файлов пар
            pair_files = sorted(pairs_folder.glob("*_pair.csv"))

            if not pair_files:
                logger.warning(f"Нет файлов пар в {pairs_folder}")
                return False

            all_pairs_data = []
            pair_id = 1

            # Чтение всех файлов пар
            for pair_file in pair_files:
                try:
                    with open(pair_file, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f, delimiter=';')
                        next(reader)  # Пропускаем заголовок

                        for row in reader:
                            if len(row) == 8:  # Проверка корректности строки
                                # Перенумеровываем ID и сохраняем остальные данные
                                new_row = [str(pair_id)] + row[1:]
                                all_pairs_data.append(new_row)
                                pair_id += 1

                except Exception as e:
                    logger.error(f"Ошибка чтения {pair_file.name}: {e}")
                    continue

            # Сохранение суммарного файла
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')

                # Заголовки
                writer.writerow([
                    'ID', 'X0', 'Y0', 'dx', 'dy', 'L', 'Diameter', 'Area'
                ])

                # Данные
                writer.writerows(all_pairs_data)

            logger.info(
                f"Создан суммарный файл: {output_path.name} "
                f"(всего пар: {len(all_pairs_data)})"
            )
            return True

        except Exception as e:
            logger.error(f"Ошибка создания суммарного файла {output_path.name}: {e}")
            return False

    def _get_image_pairs(self, camera_folder: Path) -> List[Tuple[Path, Path]]:
        """
        Получение списка пар изображений (a, b) из папки камеры.

        Args:
            camera_folder: Путь к папке камеры

        Returns:
            Список кортежей (путь_к_a, путь_к_b)
        """
        png_files = sorted(camera_folder.glob("*.png"))

        # Группировка по номеру пары
        pairs_dict: Dict[int, Dict[str, Path]] = {}

        for file_path in png_files:
            # Парсинг имени файла: N_X.png
            match = re.match(r'(\d+)_([ab])\.png', file_path.name, re.IGNORECASE)

            if match:
                pair_num = int(match.group(1))
                frame_type = match.group(2).lower()

                if pair_num not in pairs_dict:
                    pairs_dict[pair_num] = {}

                pairs_dict[pair_num][frame_type] = file_path

        # Формирование списка полных пар
        complete_pairs = []

        for pair_num in sorted(pairs_dict.keys()):
            pair_files = pairs_dict[pair_num]

            if 'a' in pair_files and 'b' in pair_files:
                complete_pairs.append((pair_files['a'], pair_files['b']))
            else:
                missing = 'a' if 'a' not in pair_files else 'b'
                logger.warning(
                    f"Пара {pair_num}: отсутствует файл {missing}, пропускаем"
                )

        return complete_pairs

    def process_camera(
        self,
        camera_name: str
    ) -> Tuple[int, int, int, List[str], List[str]]:
        """
        Обработка одной камеры.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Tuple[images_processed, particles_detected, pairs_matched, errors, warnings]
        """
        if self.input_folder is None or self.output_folder is None:
            return 0, 0, 0, ["Папки не установлены"], []

        camera_input = self.input_folder / camera_name
        camera_output = self.output_folder / camera_name
        pairs_output = self.output_folder / f"{camera_name}_pairs"

        if not camera_input.exists():
            return 0, 0, 0, [f"Папка {camera_name} не найдена"], []

        # Получение пар изображений
        image_pairs = self._get_image_pairs(camera_input)
        total_pairs = len(image_pairs)

        if total_pairs == 0:
            return 0, 0, 0, [], [f"Нет полных пар изображений в {camera_name}"]

        images_processed = 0
        total_particles = 0
        total_matched_pairs = 0
        errors = []
        warnings = []

        for pair_idx, (path_a, path_b) in enumerate(image_pairs):
            if self._cancel_requested:
                logger.info(f"Обработка {camera_name} отменена")
                break

            # Прогресс
            if self._progress_callback:
                progress = PTVProgress(
                    current_file=path_a.name,
                    total_files=total_pairs * 2,
                    processed_files=pair_idx * 2,
                    current_camera=camera_name,
                    current_stage='detection',
                    percentage=(pair_idx / total_pairs) * 100,
                    message=f"{camera_name}: обработка пары {pair_idx + 1}/{total_pairs}"
                )
                self._progress_callback(progress)

            # Загрузка изображений
            img_a = self._load_binary_image(path_a)
            img_b = self._load_binary_image(path_b)

            if img_a is None:
                errors.append(f"Ошибка загрузки: {path_a.name}")
                continue

            if img_b is None:
                errors.append(f"Ошибка загрузки: {path_b.name}")
                continue

            # Детектирование частиц
            particles_a = self.detect_particles(img_a)
            particles_b = self.detect_particles(img_b)

            total_particles += len(particles_a) + len(particles_b)
            images_processed += 2

            # Извлечение номера пары из имени файла
            match = re.match(r'(\d+)_[ab]\.png', path_a.name, re.IGNORECASE)
            pair_num = match.group(1) if match else str(pair_idx + 1)

            # Сохранение CSV с частицами
            csv_a_path = camera_output / f"{pair_num}_a.csv"
            csv_b_path = camera_output / f"{pair_num}_b.csv"

            if not self._save_particles_csv(particles_a, csv_a_path):
                errors.append(f"Ошибка сохранения: {csv_a_path.name}")

            if not self._save_particles_csv(particles_b, csv_b_path):
                errors.append(f"Ошибка сохранения: {csv_b_path.name}")

            # Сопоставление частиц
            if self._progress_callback:
                progress = PTVProgress(
                    current_file=path_a.name,
                    total_files=total_pairs * 2,
                    processed_files=pair_idx * 2 + 1,
                    current_camera=camera_name,
                    current_stage='matching',
                    percentage=((pair_idx + 0.5) / total_pairs) * 100,
                    message=f"{camera_name}: сопоставление пары {pair_idx + 1}"
                )
                self._progress_callback(progress)

            matched_pairs = self.match_particles(particles_a, particles_b)
            total_matched_pairs += len(matched_pairs)

            # Сохранение CSV с парами
            pairs_csv_path = pairs_output / f"{pair_num}_pair.csv"

            if not self._save_pairs_csv(matched_pairs, pairs_csv_path):
                errors.append(f"Ошибка сохранения: {pairs_csv_path.name}")

            if len(particles_a) == 0 and len(particles_b) == 0:
                warnings.append(f"Пара {pair_num}: частицы не обнаружены")

        # Создание суммарного файла с парами
        if not self._cancel_requested and total_matched_pairs > 0:
            summary_output_path = self.output_folder / f"{camera_name}_pairs_sum.csv"
            if not self._save_summary_pairs_csv(pairs_output, summary_output_path):
                errors.append(f"Ошибка создания суммарного файла: {summary_output_path.name}")

        # Финальный прогресс
        if self._progress_callback and not self._cancel_requested:
            progress = PTVProgress(
                current_file="",
                total_files=total_pairs * 2,
                processed_files=total_pairs * 2,
                current_camera=camera_name,
                current_stage='complete',
                percentage=100.0,
                message=f"{camera_name}: завершено"
            )
            self._progress_callback(progress)

        return images_processed, total_particles, total_matched_pairs, errors, warnings

    def process_all(self) -> PTVResult:
        """
        Обработка всех изображений с отслеживанием прогресса.

        Returns:
            PTVResult с результатами обработки
        """
        if self.input_folder is None:
            return PTVResult(
                success=False,
                total_images_processed=0,
                total_particles_detected=0,
                total_pairs_matched=0,
                cam1_pairs_count=0,
                cam2_pairs_count=0,
                errors=["Входная папка не установлена"],
                warnings=[],
                output_folder=""
            )

        self._update_output_folder()

        logger.info("=" * 60)
        logger.info("НАЧАЛО PTV АНАЛИЗА")
        logger.info(f"Входная папка: {self.input_folder}")
        logger.info(f"Выходная папка: {self.output_folder}")
        logger.info(f"Детектирование: min_area={self.detection_config.min_area}, "
                    f"max_area={self.detection_config.max_area}")
        logger.info(f"Сопоставление: max_distance={self.matching_config.max_distance}, "
                    f"max_diameter_diff={self.matching_config.max_diameter_diff}")
        logger.info("=" * 60)

        self._cancel_requested = False
        all_errors = []
        all_warnings = []

        # Обработка cam_1
        logger.info("\n--- Обработка cam_1 ---")
        (cam1_images, cam1_particles, cam1_pairs,
         cam1_errors, cam1_warnings) = self.process_camera("cam_1")
        all_errors.extend(cam1_errors)
        all_warnings.extend(cam1_warnings)

        # Обработка cam_2
        cam2_images = 0
        cam2_particles = 0
        cam2_pairs = 0

        if not self._cancel_requested:
            logger.info("\n--- Обработка cam_2 ---")
            (cam2_images, cam2_particles, cam2_pairs,
             cam2_errors, cam2_warnings) = self.process_camera("cam_2")
            all_errors.extend(cam2_errors)
            all_warnings.extend(cam2_warnings)

        # Итоговые результаты
        total_images = cam1_images + cam2_images
        total_particles = cam1_particles + cam2_particles
        total_pairs = cam1_pairs + cam2_pairs
        success = not self._cancel_requested and len(all_errors) == 0

        logger.info("\n" + "=" * 60)
        logger.info("РЕЗУЛЬТАТЫ PTV АНАЛИЗА")
        logger.info("=" * 60)
        logger.info(f"Обработано изображений: {total_images}")
        logger.info(f"Обнаружено частиц: {total_particles}")
        logger.info(f"Сопоставлено пар: {total_pairs}")
        logger.info(f"  cam_1: {cam1_pairs} пар")
        logger.info(f"  cam_2: {cam2_pairs} пар")
        logger.info(f"Ошибок: {len(all_errors)}")
        logger.info(f"Предупреждений: {len(all_warnings)}")
        logger.info(f"Выходная папка: {self.output_folder}")

        if self._cancel_requested:
            logger.info("Обработка была отменена")

        logger.info("=" * 60)

        return PTVResult(
            success=success,
            total_images_processed=total_images,
            total_particles_detected=total_particles,
            total_pairs_matched=total_pairs,
            cam1_pairs_count=cam1_pairs,
            cam2_pairs_count=cam2_pairs,
            errors=all_errors,
            warnings=all_warnings,
            output_folder=str(self.output_folder) if self.output_folder else ""
        )

    def get_detection_preview(
        self,
        image_path: Path
    ) -> Optional[Tuple[np.ndarray, List[Particle]]]:
        """
        Предварительный просмотр детектирования для одного изображения.

        Args:
            image_path: Путь к изображению

        Returns:
            Tuple[image_array, particles] или None
        """
        img_array = self._load_binary_image(image_path)

        if img_array is None:
            return None

        particles = self.detect_particles(img_array)

        return (img_array, particles)

    def get_matching_preview(
        self,
        path_a: Path,
        path_b: Path
    ) -> Optional[Dict]:
        """
        Предварительный просмотр сопоставления для пары изображений.

        Args:
            path_a: Путь к изображению a
            path_b: Путь к изображению b

        Returns:
            Словарь с результатами или None
        """
        img_a = self._load_binary_image(path_a)
        img_b = self._load_binary_image(path_b)

        if img_a is None or img_b is None:
            return None

        particles_a = self.detect_particles(img_a)
        particles_b = self.detect_particles(img_b)
        matched_pairs = self.match_particles(particles_a, particles_b)

        return {
            'image_a': img_a,
            'image_b': img_b,
            'particles_a': particles_a,
            'particles_b': particles_b,
            'matched_pairs': matched_pairs,
            'particles_a_count': len(particles_a),
            'particles_b_count': len(particles_b),
            'matched_count': len(matched_pairs),
            'unmatched_a_count': len(particles_a) - len(matched_pairs),
            'unmatched_b_count': len(particles_b) - len(matched_pairs)
        }
