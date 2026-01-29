"""
Модуль анализа оптического потока методом Lucas-Kanade для GUI приложения ParticleAnalysis.

Этот модуль предназначен для интеграции с графическим интерфейсом и предоставляет:
- Анализ оптического потока методом Lucas-Kanade (sparse optical flow)
- Вычисление векторов смещения на отслеживаемых точках (частицах)
- Экспорт результатов в CSV формат
- Callback функции для отслеживания прогресса
- Пошаговую обработку с возможностью отмены

Входные данные: изображения после фильтрации (8-bit PNG, пары _a и _b)

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Callable, List, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import csv
import logging
import re


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LucasKanadeProgress:
    """Класс для передачи информации о прогрессе обработки."""
    current_pair: str
    total_pairs: int
    processed_pairs: int
    current_camera: str
    percentage: float
    message: str


@dataclass
class LucasKanadeResult:
    """Результат анализа оптического потока Lucas-Kanade."""
    success: bool
    total_pairs_processed: int
    cam1_pairs_count: int
    cam2_pairs_count: int
    total_vectors_count: int
    total_points_detected: int
    total_points_tracked: int
    errors: List[str]
    warnings: List[str]
    output_folder: str


@dataclass
class FlowStatistics:
    """Статистика оптического потока для пары изображений."""
    mean_magnitude: float
    max_magnitude: float
    min_magnitude: float
    std_magnitude: float
    mean_angle: float
    std_angle: float
    points_detected: int
    points_tracked: int


@dataclass
class LucasKanadeConfig:
    """Конфигурация алгоритма Lucas-Kanade."""
    # Параметры детекции точек (goodFeaturesToTrack)
    max_corners: int = 500  # Максимальное количество точек для трекинга
    quality_level: float = 0.01  # Минимальное качество угла (0-1)
    min_distance: int = 10  # Минимальное расстояние между точками
    block_size: int = 7  # Размер блока для вычисления производных

    # Параметры Lucas-Kanade (calcOpticalFlowPyrLK)
    win_size: int = 21  # Размер окна для вычисления потока
    max_level: int = 2  # Количество уровней пирамиды
    max_iterations: int = 30  # Максимальное количество итераций
    epsilon: float = 0.01  # Порог сходимости

    # Параметры фильтрации результатов
    min_magnitude: float = 0.0  # Минимальная магнитуда для фильтрации
    max_magnitude: float = 100.0  # Максимальная магнитуда для фильтрации

    # Физические параметры
    dt: float = 0.002  # Временной интервал между кадрами в паре (мс)
    scaling_factor: float = 1.0  # Масштабный коэффициент (пиксели -> мм)


class LucasKanadeAnalyzer:
    """
    Класс для анализа оптического потока методом Lucas-Kanade с поддержкой GUI.

    Выполняет sparse optical flow анализ между последовательными
    изображениями, отслеживая характерные точки (частицы).
    """

    def __init__(self):
        """Инициализация модуля анализа оптического потока Lucas-Kanade."""
        self.input_folder: Optional[Path] = None
        self.output_folder: Optional[Path] = None
        self.config = LucasKanadeConfig()
        self._cancel_requested: bool = False
        self._progress_callback: Optional[Callable[[LucasKanadeProgress], None]] = None

        logger.info("Инициализирован модуль анализа оптического потока Lucas-Kanade")

    def set_input_folder(self, folder_path: str) -> bool:
        """
        Установка входной папки с отфильтрованными изображениями.

        Args:
            folder_path: Путь к папке с изображениями

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

    def _extract_threshold_from_folder_name(self) -> str:
        """
        Извлечение значения порога из имени входной папки.

        Returns:
            str: Значение порога или 'unknown'
        """
        if self.input_folder is None:
            return "unknown"

        folder_name = self.input_folder.name
        # Поддержка разных форматов имён папок
        match = re.search(r'filtered_(\d+)', folder_name)

        if match:
            return match.group(1)

        return "unknown"

    def _update_output_folder(self) -> None:
        """Обновление пути выходной папки."""
        if self.input_folder is not None:
            threshold = self._extract_threshold_from_folder_name()
            output_name = f"LucasKanade_{threshold}"
            self.output_folder = self.input_folder.parent / output_name
            logger.info(f"Выходная папка: {self.output_folder}")

    def set_config(
        self,
        max_corners: Optional[int] = None,
        quality_level: Optional[float] = None,
        min_distance: Optional[int] = None,
        block_size: Optional[int] = None,
        win_size: Optional[int] = None,
        max_level: Optional[int] = None,
        max_iterations: Optional[int] = None,
        epsilon: Optional[float] = None,
        min_magnitude: Optional[float] = None,
        max_magnitude: Optional[float] = None,
        dt: Optional[float] = None,
        scaling_factor: Optional[float] = None
    ) -> bool:
        """
        Установка параметров алгоритма Lucas-Kanade.

        Args:
            max_corners: Максимальное количество точек для трекинга
            quality_level: Минимальное качество угла (0-1)
            min_distance: Минимальное расстояние между точками
            block_size: Размер блока для вычисления производных
            win_size: Размер окна для вычисления потока
            max_level: Количество уровней пирамиды
            max_iterations: Максимальное количество итераций
            epsilon: Порог сходимости
            min_magnitude: Минимальная магнитуда для фильтрации
            max_magnitude: Максимальная магнитуда для фильтрации
            dt: Временной интервал между кадрами (мс)
            scaling_factor: Масштабный коэффициент (пиксели -> мм)

        Returns:
            bool: True если параметры валидны
        """
        if max_corners is not None:
            if max_corners < 1:
                logger.error(f"max_corners должен быть >= 1: {max_corners}")
                return False
            self.config.max_corners = max_corners

        if quality_level is not None:
            if not (0 < quality_level <= 1):
                logger.error(f"quality_level должен быть в диапазоне (0, 1]: {quality_level}")
                return False
            self.config.quality_level = quality_level

        if min_distance is not None:
            if min_distance < 1:
                logger.error(f"min_distance должен быть >= 1: {min_distance}")
                return False
            self.config.min_distance = min_distance

        if block_size is not None:
            if block_size < 3 or block_size % 2 == 0:
                logger.error(f"block_size должен быть нечётным и >= 3: {block_size}")
                return False
            self.config.block_size = block_size

        if win_size is not None:
            if win_size < 3 or win_size % 2 == 0:
                logger.error(f"win_size должен быть нечётным и >= 3: {win_size}")
                return False
            self.config.win_size = win_size

        if max_level is not None:
            if max_level < 0:
                logger.error(f"max_level должен быть >= 0: {max_level}")
                return False
            self.config.max_level = max_level

        if max_iterations is not None:
            if max_iterations < 1:
                logger.error(f"max_iterations должен быть >= 1: {max_iterations}")
                return False
            self.config.max_iterations = max_iterations

        if epsilon is not None:
            if epsilon <= 0:
                logger.error(f"epsilon должен быть > 0: {epsilon}")
                return False
            self.config.epsilon = epsilon

        if min_magnitude is not None:
            if min_magnitude < 0:
                logger.error(f"min_magnitude должен быть >= 0: {min_magnitude}")
                return False
            self.config.min_magnitude = min_magnitude

        if max_magnitude is not None:
            if max_magnitude <= 0:
                logger.error(f"max_magnitude должен быть > 0: {max_magnitude}")
                return False
            self.config.max_magnitude = max_magnitude

        if dt is not None:
            if dt <= 0:
                logger.error(f"dt должен быть > 0: {dt}")
                return False
            self.config.dt = dt

        if scaling_factor is not None:
            if scaling_factor <= 0:
                logger.error(f"scaling_factor должен быть > 0: {scaling_factor}")
                return False
            self.config.scaling_factor = scaling_factor

        logger.info(
            f"Параметры Lucas-Kanade установлены: max_corners={self.config.max_corners}, "
            f"quality_level={self.config.quality_level}, win_size={self.config.win_size}, "
            f"max_level={self.config.max_level}"
        )
        return True

    def set_progress_callback(self, callback: Callable[[LucasKanadeProgress], None]) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция, принимающая LucasKanadeProgress
        """
        self._progress_callback = callback
        logger.debug("Установлен callback для прогресса")

    def cancel_processing(self) -> None:
        """Запрос на отмену обработки."""
        self._cancel_requested = True
        logger.info("Запрошена отмена обработки")

    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Загрузка 8-битного изображения.

        Args:
            image_path: Путь к изображению

        Returns:
            numpy.ndarray (grayscale) или None в случае ошибки
        """
        try:
            img = Image.open(image_path)
            img_array = np.array(img)

            # Конвертация в grayscale если необходимо
            if len(img_array.shape) == 3:
                img_array = np.mean(img_array, axis=2).astype(np.uint8)

            # Преобразование в uint8 если необходимо
            if img_array.dtype == np.uint16:
                img_array = (img_array / 256).astype(np.uint8)
            elif img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)

            return img_array

        except Exception as e:
            logger.error(f"Ошибка загрузки {image_path.name}: {e}")
            return None

    def compute_flow(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Вычисление оптического потока методом Lucas-Kanade (sparse).

        Args:
            img1: Первый кадр (grayscale)
            img2: Второй кадр (grayscale)

        Returns:
            dict с ключами:
                - 'points_old': координаты на кадре 1 (N, 2)
                - 'points_new': координаты на кадре 2 (N, 2)
                - 'u': смещения по X (N,)
                - 'v': смещения по Y (N,)
                - 'magnitude': магнитуда смещений (N,)
                - 'angle': углы направления в градусах (N,)
                - 'total_detected': общее количество найденных точек
                - 'total_tracked': количество успешно отслеженных точек
            None если точки не найдены
        """
        cfg = self.config

        # Параметры детекции точек (goodFeaturesToTrack)
        feature_params = dict(
            maxCorners=cfg.max_corners,
            qualityLevel=cfg.quality_level,
            minDistance=cfg.min_distance,
            blockSize=cfg.block_size
        )

        # Параметры Lucas-Kanade (calcOpticalFlowPyrLK)
        lk_params = dict(
            winSize=(cfg.win_size, cfg.win_size),
            maxLevel=cfg.max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                cfg.max_iterations,
                cfg.epsilon
            )
        )

        # Находим характерные точки на первом кадре
        p0 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)

        if p0 is None or len(p0) == 0:
            logger.warning("Не найдены характерные точки на изображении")
            return None

        total_detected = len(p0)

        # Вычисляем optical flow
        p1, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)

        # Отбираем успешно отслеженные точки
        good_old = p0[status == 1]
        good_new = p1[status == 1]

        if len(good_old) == 0:
            logger.warning("Не удалось отследить ни одной точки")
            return None

        # Вычисляем смещения
        displacement = good_new - good_old
        u = displacement[:, 0]
        v = displacement[:, 1]
        magnitude = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, u) * 180 / np.pi

        # Фильтрация по магнитуде
        mask = (magnitude >= cfg.min_magnitude) & (magnitude <= cfg.max_magnitude)
        if not np.any(mask):
            logger.warning("Все точки отфильтрованы по магнитуде")
            return None

        good_old = good_old[mask]
        good_new = good_new[mask]
        u = u[mask]
        v = v[mask]
        magnitude = magnitude[mask]
        angle = angle[mask]

        # Применение масштабного коэффициента
        if cfg.scaling_factor != 1.0:
            u = u * cfg.scaling_factor
            v = v * cfg.scaling_factor
            magnitude = magnitude * cfg.scaling_factor

        return {
            'points_old': good_old,
            'points_new': good_new,
            'u': u,
            'v': v,
            'magnitude': magnitude,
            'angle': angle,
            'total_detected': total_detected,
            'total_tracked': len(good_old)
        }

    def _compute_flow_statistics(
        self,
        flow_result: Dict[str, np.ndarray]
    ) -> FlowStatistics:
        """
        Вычисление статистики оптического потока.

        Args:
            flow_result: Результат compute_flow

        Returns:
            FlowStatistics
        """
        magnitude = flow_result['magnitude']
        angle = flow_result['angle']

        return FlowStatistics(
            mean_magnitude=float(np.mean(magnitude)),
            max_magnitude=float(np.max(magnitude)),
            min_magnitude=float(np.min(magnitude)),
            std_magnitude=float(np.std(magnitude)),
            mean_angle=float(np.mean(angle)),
            std_angle=float(np.std(angle)),
            points_detected=flow_result['total_detected'],
            points_tracked=flow_result['total_tracked']
        )

    def _save_csv(
        self,
        output_path: Path,
        flow_result: Dict[str, np.ndarray]
    ) -> int:
        """
        Сохранение результатов Lucas-Kanade в CSV файл.

        Args:
            output_path: Путь к выходному CSV файлу
            flow_result: Результат compute_flow

        Returns:
            int: Количество сохранённых векторов
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            points_old = flow_result['points_old']
            points_new = flow_result['points_new']
            u = flow_result['u']
            v = flow_result['v']
            magnitude = flow_result['magnitude']
            angle = flow_result['angle']

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(['X0', 'Y0', 'X1', 'Y1', 'U', 'V', 'Magnitude', 'Angle'])

                for i in range(len(points_old)):
                    writer.writerow([
                        f"{points_old[i, 0]:.2f}",
                        f"{points_old[i, 1]:.2f}",
                        f"{points_new[i, 0]:.2f}",
                        f"{points_new[i, 1]:.2f}",
                        f"{u[i]:.6f}",
                        f"{v[i]:.6f}",
                        f"{magnitude[i]:.6f}",
                        f"{angle[i]:.2f}"
                    ])

            logger.info(f"Сохранено {len(points_old)} векторов в {output_path.name}")
            return len(points_old)

        except Exception as e:
            logger.error(f"Ошибка сохранения CSV {output_path}: {e}")
            return 0

    def _save_summary_csv(
        self,
        camera_folder: Path,
        camera_name: str,
        all_files: List[Path]
    ) -> bool:
        """
        Сохранение суммарного CSV файла со всеми векторами.

        Args:
            camera_folder: Папка камеры
            camera_name: Название камеры
            all_files: Список файлов для объединения

        Returns:
            bool: True если успешно
        """
        summary_path = camera_folder / f"{camera_name}_lucas_kanade_sum.csv"

        try:
            with open(summary_path, 'w', newline='', encoding='utf-8') as f_out:
                writer = csv.writer(f_out, delimiter=';')
                header_written = False
                total_vectors = 0

                for file_path in all_files:
                    if not file_path.exists():
                        continue

                    with open(file_path, 'r', encoding='utf-8') as f_in:
                        reader = csv.reader(f_in, delimiter=';')
                        header = next(reader)

                        if not header_written:
                            writer.writerow(header)
                            header_written = True

                        for row in reader:
                            writer.writerow(row)
                            total_vectors += 1

            logger.info(f"Сохранён суммарный файл: {summary_path.name} ({total_vectors} векторов)")
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения суммарного CSV: {e}")
            return False

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

    def process_camera(
        self,
        camera_name: str
    ) -> Tuple[int, int, int, int, List[str], List[str]]:
        """
        Обработка одной камеры с отслеживанием прогресса.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Tuple[pairs_processed, vectors_count, points_detected, points_tracked, errors, warnings]
        """
        if self.input_folder is None or self.output_folder is None:
            logger.error("Входная или выходная папка не установлена")
            return 0, 0, 0, 0, ["Папки не установлены"], []

        camera_input = self.input_folder / camera_name
        camera_output = self.output_folder / camera_name

        if not camera_input.exists():
            logger.warning(f"Папка {camera_name} не найдена")
            return 0, 0, 0, 0, [f"Папка {camera_name} не найдена"], []

        # Поиск пар изображений
        pairs = self._find_image_pairs(camera_input)
        total_pairs = len(pairs)

        if total_pairs == 0:
            logger.warning(f"В {camera_name} нет пар изображений")
            return 0, 0, 0, 0, [], [f"Нет пар изображений в {camera_name}"]

        processed = 0
        total_vectors = 0
        total_detected = 0
        total_tracked = 0
        errors = []
        warnings = []
        output_files = []

        for idx, (path_a, path_b) in enumerate(pairs):
            if self._cancel_requested:
                logger.info(f"Обработка {camera_name} отменена")
                break

            if self._progress_callback:
                progress = LucasKanadeProgress(
                    current_pair=f"{path_a.name} - {path_b.name}",
                    total_pairs=total_pairs,
                    processed_pairs=idx,
                    current_camera=camera_name,
                    percentage=(idx / total_pairs) * 100,
                    message=f"Обработка {camera_name}: пара {idx+1}/{total_pairs}"
                )
                self._progress_callback(progress)

            # Загрузка изображений
            img_a = self._load_image(path_a)
            img_b = self._load_image(path_b)

            if img_a is None or img_b is None:
                errors.append(f"Ошибка загрузки пары: {path_a.name} - {path_b.name}")
                continue

            # Вычисление оптического потока
            try:
                flow_result = self.compute_flow(img_a, img_b)

                if flow_result is None:
                    warnings.append(f"Не найдены/отслежены точки для {path_a.name}")
                    continue

                # Накопление статистики
                total_detected += flow_result['total_detected']
                total_tracked += flow_result['total_tracked']

                # Сохранение результата
                pair_name = path_a.stem.replace("_a", "")
                output_filename = f"lk_{pair_name}.csv"
                output_path = camera_output / output_filename

                vectors = self._save_csv(output_path, flow_result)

                if vectors > 0:
                    output_files.append(output_path)
                    total_vectors += vectors
                    processed += 1
                else:
                    errors.append(f"Ошибка сохранения: {output_filename}")

            except Exception as e:
                errors.append(f"Ошибка анализа пары {path_a.name}: {e}")
                logger.error(f"Ошибка анализа пары {path_a.name}: {e}")

        # Сохранение суммарного файла
        if output_files and not self._cancel_requested:
            self._save_summary_csv(camera_output, camera_name, output_files)

        if self._progress_callback and not self._cancel_requested:
            progress = LucasKanadeProgress(
                current_pair="",
                total_pairs=total_pairs,
                processed_pairs=total_pairs,
                current_camera=camera_name,
                percentage=100.0,
                message=f"{camera_name}: завершено"
            )
            self._progress_callback(progress)

        return processed, total_vectors, total_detected, total_tracked, errors, warnings

    def process_all(self) -> LucasKanadeResult:
        """
        Обработка всех камер с отслеживанием прогресса.

        Returns:
            LucasKanadeResult с результатами обработки
        """
        if self.input_folder is None:
            return LucasKanadeResult(
                success=False,
                total_pairs_processed=0,
                cam1_pairs_count=0,
                cam2_pairs_count=0,
                total_vectors_count=0,
                total_points_detected=0,
                total_points_tracked=0,
                errors=["Входная папка не установлена"],
                warnings=[],
                output_folder=""
            )

        self._update_output_folder()

        logger.info("=" * 60)
        logger.info("НАЧАЛО АНАЛИЗА ОПТИЧЕСКОГО ПОТОКА LUCAS-KANADE")
        logger.info(f"Входная папка: {self.input_folder}")
        logger.info(f"Выходная папка: {self.output_folder}")
        logger.info(f"Параметры: max_corners={self.config.max_corners}, "
                   f"quality_level={self.config.quality_level}, "
                   f"win_size={self.config.win_size}, max_level={self.config.max_level}")
        logger.info("=" * 60)

        self._cancel_requested = False

        logger.info("\n--- Обработка cam_1 ---")
        cam1_pairs, cam1_vectors, cam1_detected, cam1_tracked, cam1_errors, cam1_warnings = \
            self.process_camera("cam_1")

        cam2_pairs = 0
        cam2_vectors = 0
        cam2_detected = 0
        cam2_tracked = 0
        cam2_errors = []
        cam2_warnings = []

        if not self._cancel_requested:
            logger.info("\n--- Обработка cam_2 ---")
            cam2_pairs, cam2_vectors, cam2_detected, cam2_tracked, cam2_errors, cam2_warnings = \
                self.process_camera("cam_2")

        total_pairs = cam1_pairs + cam2_pairs
        total_vectors = cam1_vectors + cam2_vectors
        total_detected = cam1_detected + cam2_detected
        total_tracked = cam1_tracked + cam2_tracked
        all_errors = cam1_errors + cam2_errors
        all_warnings = cam1_warnings + cam2_warnings
        success = not self._cancel_requested and len(all_errors) == 0

        logger.info("\n" + "=" * 60)
        logger.info("РЕЗУЛЬТАТЫ АНАЛИЗА LUCAS-KANADE")
        logger.info("=" * 60)
        logger.info(f"cam_1: {cam1_pairs} пар, {cam1_vectors} векторов")
        logger.info(f"cam_2: {cam2_pairs} пар, {cam2_vectors} векторов")
        logger.info(f"Всего пар: {total_pairs}")
        logger.info(f"Всего векторов: {total_vectors}")
        logger.info(f"Всего точек найдено: {total_detected}")
        logger.info(f"Всего точек отслежено: {total_tracked}")
        logger.info(f"Ошибок: {len(all_errors)}")
        logger.info(f"Выходная папка: {self.output_folder}")

        if self._cancel_requested:
            logger.info("Обработка была отменена")

        logger.info("=" * 60)

        return LucasKanadeResult(
            success=success,
            total_pairs_processed=total_pairs,
            cam1_pairs_count=cam1_pairs,
            cam2_pairs_count=cam2_pairs,
            total_vectors_count=total_vectors,
            total_points_detected=total_detected,
            total_points_tracked=total_tracked,
            errors=all_errors,
            warnings=all_warnings,
            output_folder=str(self.output_folder)
        )

    def get_preview(
        self,
        camera_name: str,
        pair_index: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Получение предварительного просмотра анализа для пары изображений.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)
            pair_index: Индекс пары для предпросмотра

        Returns:
            Словарь с результатами или None
        """
        if self.input_folder is None:
            return None

        camera_path = self.input_folder / camera_name
        if not camera_path.exists():
            return None

        # Поиск пар изображений
        pairs = self._find_image_pairs(camera_path)
        if not pairs or pair_index >= len(pairs):
            return None

        path_a, path_b = pairs[pair_index]

        # Загрузка изображений
        img_a = self._load_image(path_a)
        img_b = self._load_image(path_b)

        if img_a is None or img_b is None:
            return None

        try:
            flow_result = self.compute_flow(img_a, img_b)
            if flow_result is None:
                return None

            stats = self._compute_flow_statistics(flow_result)

            return {
                'image_a': img_a,
                'image_b': img_b,
                'points_old': flow_result['points_old'],
                'points_new': flow_result['points_new'],
                'u': flow_result['u'],
                'v': flow_result['v'],
                'magnitude': flow_result['magnitude'],
                'angle': flow_result['angle'],
                'total_detected': flow_result['total_detected'],
                'total_tracked': flow_result['total_tracked'],
                'statistics': stats,
                'mean_magnitude': stats.mean_magnitude,
                'max_magnitude': stats.max_magnitude,
                'pair_names': (path_a.name, path_b.name)
            }

        except Exception as e:
            logger.error(f"Ошибка создания предпросмотра: {e}")
            return None

    def get_pair_count(self, camera_name: str) -> int:
        """
        Получение количества пар изображений в камере.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Количество пар или 0
        """
        if self.input_folder is None:
            return 0

        camera_path = self.input_folder / camera_name
        if not camera_path.exists():
            return 0

        pairs = self._find_image_pairs(camera_path)
        return len(pairs)
