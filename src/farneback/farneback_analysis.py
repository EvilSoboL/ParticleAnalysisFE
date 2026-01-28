"""
Модуль анализа оптического потока методом Farneback для GUI приложения ParticleAnalysis.

Этот модуль предназначен для интеграции с графическим интерфейсом и предоставляет:
- Анализ оптического потока методом Farneback (dense optical flow)
- Анализ оптического потока методом Lucas-Kanade (sparse optical flow)
- Вычисление векторных полей скоростей
- Экспорт результатов в CSV формат
- Callback функции для отслеживания прогресса
- Пошаговую обработку с возможностью отмены

Входные данные: изображения после filter_farneback_kanade (8-bit PNG, пары _a и _b)

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
class FarnebackProgress:
    """Класс для передачи информации о прогрессе обработки."""
    current_pair: str
    total_pairs: int
    processed_pairs: int
    current_camera: str
    percentage: float
    message: str


@dataclass
class FarnebackResult:
    """Результат анализа оптического потока."""
    success: bool
    total_pairs_processed: int
    cam1_pairs_count: int
    cam2_pairs_count: int
    total_vectors_count: int
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
    nonzero_pixels: int
    total_pixels: int


@dataclass
class FarnebackConfig:
    """Конфигурация алгоритма Farneback."""
    # Параметры алгоритма Farneback
    pyr_scale: float = 0.5  # Масштаб пирамиды (< 1)
    levels: int = 3  # Количество уровней пирамиды
    winsize: int = 15  # Размер окна усреднения
    iterations: int = 3  # Количество итераций на каждом уровне
    poly_n: int = 5  # Размер окрестности для полиномиальной аппроксимации
    poly_sigma: float = 1.2  # Стандартное отклонение гауссиана для сглаживания

    # Параметры фильтрации результатов
    min_magnitude: float = 0.0  # Минимальная магнитуда для фильтрации
    max_magnitude: float = 100.0  # Максимальная магнитуда для фильтрации

    # Параметры сетки для экспорта (прореживание dense flow)
    grid_step: int = 10  # Шаг сетки для экспорта векторов

    # Физические параметры
    dt: float = 0.002  # Временной интервал между кадрами в паре (мс)
    scaling_factor: float = 1.0  # Масштабный коэффициент (пиксели -> мм)


@dataclass
class LucasKanadeConfig:
    """Конфигурация алгоритма Lucas-Kanade."""
    # Параметры детекции точек
    max_corners: int = 500  # Максимальное количество точек для трекинга
    quality_level: float = 0.01  # Минимальное качество угла (0-1)
    min_distance: int = 10  # Минимальное расстояние между точками
    block_size: int = 7  # Размер блока для вычисления производных

    # Параметры Lucas-Kanade
    win_size: int = 21  # Размер окна для вычисления потока
    max_level: int = 2  # Количество уровней пирамиды

    # Физические параметры
    dt: float = 0.002  # Временной интервал между кадрами в паре (мс)
    scaling_factor: float = 1.0  # Масштабный коэффициент (пиксели -> мм)


class FarnebackAnalyzer:
    """
    Класс для анализа оптического потока с поддержкой GUI.

    Выполняет анализ оптического потока между последовательными
    изображениями методами Farneback (dense) и Lucas-Kanade (sparse).
    """

    def __init__(self):
        """Инициализация модуля анализа оптического потока."""
        self.input_folder: Optional[Path] = None
        self.output_folder: Optional[Path] = None
        self.farneback_config = FarnebackConfig()
        self.lucas_kanade_config = LucasKanadeConfig()
        self._cancel_requested: bool = False
        self._progress_callback: Optional[Callable[[FarnebackProgress], None]] = None

        logger.info("Инициализирован модуль анализа оптического потока Farneback")

    def set_input_folder(self, folder_path: str) -> bool:
        """
        Установка входной папки (папка farneback_filtered_XXXX).

        Args:
            folder_path: Путь к папке с отфильтрованными изображениями

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
        match = re.search(r'farneback_filtered_(\d+)', folder_name)

        if match:
            return match.group(1)

        return "unknown"

    def _update_output_folder(self) -> None:
        """Обновление пути выходной папки."""
        if self.input_folder is not None:
            threshold = self._extract_threshold_from_folder_name()
            output_name = f"Farneback_{threshold}"
            self.output_folder = self.input_folder.parent / output_name
            logger.info(f"Выходная папка: {self.output_folder}")

    def set_farneback_config(
        self,
        pyr_scale: Optional[float] = None,
        levels: Optional[int] = None,
        winsize: Optional[int] = None,
        iterations: Optional[int] = None,
        poly_n: Optional[int] = None,
        poly_sigma: Optional[float] = None,
        min_magnitude: Optional[float] = None,
        max_magnitude: Optional[float] = None,
        grid_step: Optional[int] = None,
        dt: Optional[float] = None,
        scaling_factor: Optional[float] = None
    ) -> bool:
        """
        Установка параметров алгоритма Farneback.

        Args:
            pyr_scale: Масштаб пирамиды (0-1)
            levels: Количество уровней пирамиды
            winsize: Размер окна усреднения
            iterations: Количество итераций на каждом уровне
            poly_n: Размер окрестности для полиномиальной аппроксимации
            poly_sigma: Стандартное отклонение гауссиана
            min_magnitude: Минимальная магнитуда для фильтрации
            max_magnitude: Максимальная магнитуда для фильтрации
            grid_step: Шаг сетки для экспорта
            dt: Временной интервал между кадрами (мс)
            scaling_factor: Масштабный коэффициент (пиксели -> мм)

        Returns:
            bool: True если параметры валидны
        """
        if pyr_scale is not None:
            if not (0 < pyr_scale < 1):
                logger.error(f"pyr_scale должен быть в диапазоне (0, 1): {pyr_scale}")
                return False
            self.farneback_config.pyr_scale = pyr_scale

        if levels is not None:
            if levels < 1:
                logger.error(f"levels должен быть >= 1: {levels}")
                return False
            self.farneback_config.levels = levels

        if winsize is not None:
            if winsize < 1 or winsize % 2 == 0:
                logger.error(f"winsize должен быть нечётным и >= 1: {winsize}")
                return False
            self.farneback_config.winsize = winsize

        if iterations is not None:
            if iterations < 1:
                logger.error(f"iterations должен быть >= 1: {iterations}")
                return False
            self.farneback_config.iterations = iterations

        if poly_n is not None:
            if poly_n not in [5, 7]:
                logger.error(f"poly_n должен быть 5 или 7: {poly_n}")
                return False
            self.farneback_config.poly_n = poly_n

        if poly_sigma is not None:
            if poly_sigma <= 0:
                logger.error(f"poly_sigma должен быть > 0: {poly_sigma}")
                return False
            self.farneback_config.poly_sigma = poly_sigma

        if min_magnitude is not None:
            if min_magnitude < 0:
                logger.error(f"min_magnitude должен быть >= 0: {min_magnitude}")
                return False
            self.farneback_config.min_magnitude = min_magnitude

        if max_magnitude is not None:
            if max_magnitude <= 0:
                logger.error(f"max_magnitude должен быть > 0: {max_magnitude}")
                return False
            self.farneback_config.max_magnitude = max_magnitude

        if grid_step is not None:
            if grid_step < 1:
                logger.error(f"grid_step должен быть >= 1: {grid_step}")
                return False
            self.farneback_config.grid_step = grid_step

        if dt is not None:
            if dt <= 0:
                logger.error(f"dt должен быть > 0: {dt}")
                return False
            self.farneback_config.dt = dt

        if scaling_factor is not None:
            if scaling_factor <= 0:
                logger.error(f"scaling_factor должен быть > 0: {scaling_factor}")
                return False
            self.farneback_config.scaling_factor = scaling_factor

        logger.info(
            f"Параметры Farneback установлены: pyr_scale={self.farneback_config.pyr_scale}, "
            f"levels={self.farneback_config.levels}, winsize={self.farneback_config.winsize}, "
            f"iterations={self.farneback_config.iterations}"
        )
        return True

    def set_lucas_kanade_config(
        self,
        max_corners: Optional[int] = None,
        quality_level: Optional[float] = None,
        min_distance: Optional[int] = None,
        win_size: Optional[int] = None,
        max_level: Optional[int] = None,
        dt: Optional[float] = None,
        scaling_factor: Optional[float] = None
    ) -> bool:
        """
        Установка параметров алгоритма Lucas-Kanade.

        Args:
            max_corners: Максимальное количество точек
            quality_level: Минимальное качество угла (0-1)
            min_distance: Минимальное расстояние между точками
            win_size: Размер окна для вычисления потока
            max_level: Количество уровней пирамиды
            dt: Временной интервал между кадрами (мс)
            scaling_factor: Масштабный коэффициент (пиксели -> мм)

        Returns:
            bool: True если параметры валидны
        """
        if max_corners is not None:
            if max_corners < 1:
                logger.error(f"max_corners должен быть >= 1: {max_corners}")
                return False
            self.lucas_kanade_config.max_corners = max_corners

        if quality_level is not None:
            if not (0 < quality_level <= 1):
                logger.error(f"quality_level должен быть в диапазоне (0, 1]: {quality_level}")
                return False
            self.lucas_kanade_config.quality_level = quality_level

        if min_distance is not None:
            if min_distance < 1:
                logger.error(f"min_distance должен быть >= 1: {min_distance}")
                return False
            self.lucas_kanade_config.min_distance = min_distance

        if win_size is not None:
            if win_size < 3 or win_size % 2 == 0:
                logger.error(f"win_size должен быть нечётным и >= 3: {win_size}")
                return False
            self.lucas_kanade_config.win_size = win_size

        if max_level is not None:
            if max_level < 0:
                logger.error(f"max_level должен быть >= 0: {max_level}")
                return False
            self.lucas_kanade_config.max_level = max_level

        if dt is not None:
            if dt <= 0:
                logger.error(f"dt должен быть > 0: {dt}")
                return False
            self.lucas_kanade_config.dt = dt

        if scaling_factor is not None:
            if scaling_factor <= 0:
                logger.error(f"scaling_factor должен быть > 0: {scaling_factor}")
                return False
            self.lucas_kanade_config.scaling_factor = scaling_factor

        logger.info(
            f"Параметры Lucas-Kanade установлены: max_corners={self.lucas_kanade_config.max_corners}, "
            f"quality_level={self.lucas_kanade_config.quality_level}, "
            f"win_size={self.lucas_kanade_config.win_size}"
        )
        return True

    def set_progress_callback(self, callback: Callable[[FarnebackProgress], None]) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция, принимающая FarnebackProgress
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

    def compute_farneback_flow(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Вычисление оптического потока методом Farneback (dense).

        Args:
            img1: Первый кадр (grayscale)
            img2: Второй кадр (grayscale)

        Returns:
            dict с ключами:
                - 'flow': поле скоростей (H, W, 2)
                - 'u': компонента X скорости (H, W)
                - 'v': компонента Y скорости (H, W)
                - 'magnitude': магнитуда (H, W)
                - 'angle': угол направления в градусах (H, W)
        """
        cfg = self.farneback_config

        flow = cv2.calcOpticalFlowFarneback(
            img1, img2, None,
            pyr_scale=cfg.pyr_scale,
            levels=cfg.levels,
            winsize=cfg.winsize,
            iterations=cfg.iterations,
            poly_n=cfg.poly_n,
            poly_sigma=cfg.poly_sigma,
            flags=0
        )

        u = flow[:, :, 0]
        v = flow[:, :, 1]
        magnitude = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, u) * 180 / np.pi

        # Применение масштабного коэффициента
        if cfg.scaling_factor != 1.0:
            u = u * cfg.scaling_factor
            v = v * cfg.scaling_factor
            magnitude = magnitude * cfg.scaling_factor

        return {
            'flow': flow,
            'u': u,
            'v': v,
            'magnitude': magnitude,
            'angle': angle
        }

    def compute_lucas_kanade_flow(
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
        cfg = self.lucas_kanade_config

        # Параметры детекции точек
        feature_params = dict(
            maxCorners=cfg.max_corners,
            qualityLevel=cfg.quality_level,
            minDistance=cfg.min_distance,
            blockSize=cfg.block_size
        )

        # Параметры Lucas-Kanade
        lk_params = dict(
            winSize=(cfg.win_size, cfg.win_size),
            maxLevel=cfg.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Находим точки на первом кадре
        p0 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)

        if p0 is None or len(p0) == 0:
            return None

        total_detected = len(p0)

        # Вычисляем optical flow
        p1, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)

        # Отбираем успешно отслеженные точки
        good_old = p0[status == 1]
        good_new = p1[status == 1]

        if len(good_old) == 0:
            return None

        # Вычисляем смещения
        displacement = good_new - good_old
        u = displacement[:, 0]
        v = displacement[:, 1]
        magnitude = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, u) * 180 / np.pi

        # Применение масштабного коэффициента
        if cfg.scaling_factor != 1.0:
            good_old = good_old * cfg.scaling_factor
            good_new = good_new * cfg.scaling_factor
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
        flow_result: Dict[str, np.ndarray],
        img: np.ndarray,
        intensity_threshold: int = 30
    ) -> FlowStatistics:
        """
        Вычисление статистики оптического потока.

        Args:
            flow_result: Результат compute_farneback_flow
            img: Исходное изображение для маскирования
            intensity_threshold: Порог интенсивности для маски

        Returns:
            FlowStatistics
        """
        magnitude = flow_result['magnitude']
        angle = flow_result['angle']

        # Маска по интенсивности (только на частицах)
        mask = img > intensity_threshold
        nonzero_pixels = mask.sum()
        total_pixels = img.size

        if nonzero_pixels > 0:
            mag_masked = magnitude[mask]
            angle_masked = angle[mask]
        else:
            mag_masked = magnitude.flatten()
            angle_masked = angle.flatten()

        return FlowStatistics(
            mean_magnitude=float(np.mean(mag_masked)),
            max_magnitude=float(np.max(mag_masked)),
            min_magnitude=float(np.min(mag_masked)),
            std_magnitude=float(np.std(mag_masked)),
            mean_angle=float(np.mean(angle_masked)),
            std_angle=float(np.std(angle_masked)),
            nonzero_pixels=int(nonzero_pixels),
            total_pixels=int(total_pixels)
        )

    def _save_farneback_csv(
        self,
        output_path: Path,
        flow_result: Dict[str, np.ndarray],
        img_shape: Tuple[int, int]
    ) -> int:
        """
        Сохранение результатов Farneback в CSV файл.

        Использует grid_step для прореживания dense flow.

        Args:
            output_path: Путь к выходному CSV файлу
            flow_result: Результат compute_farneback_flow
            img_shape: Размер изображения (H, W)

        Returns:
            int: Количество сохранённых векторов
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            cfg = self.farneback_config
            step = cfg.grid_step
            h, w = img_shape

            u = flow_result['u']
            v = flow_result['v']
            magnitude = flow_result['magnitude']
            angle = flow_result['angle']

            vectors_count = 0

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(['X', 'Y', 'U', 'V', 'Magnitude', 'Angle'])

                for y in range(step // 2, h, step):
                    for x in range(step // 2, w, step):
                        mag = magnitude[y, x]

                        # Фильтрация по магнитуде
                        if mag < cfg.min_magnitude or mag > cfg.max_magnitude:
                            continue

                        writer.writerow([
                            f"{x:.1f}",
                            f"{y:.1f}",
                            f"{u[y, x]:.6f}",
                            f"{v[y, x]:.6f}",
                            f"{mag:.6f}",
                            f"{angle[y, x]:.2f}"
                        ])
                        vectors_count += 1

            logger.info(f"Сохранено {vectors_count} векторов в {output_path.name}")
            return vectors_count

        except Exception as e:
            logger.error(f"Ошибка сохранения CSV {output_path}: {e}")
            return 0

    def _save_lucas_kanade_csv(
        self,
        output_path: Path,
        lk_result: Dict[str, np.ndarray]
    ) -> int:
        """
        Сохранение результатов Lucas-Kanade в CSV файл.

        Args:
            output_path: Путь к выходному CSV файлу
            lk_result: Результат compute_lucas_kanade_flow

        Returns:
            int: Количество сохранённых векторов
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            points_old = lk_result['points_old']
            points_new = lk_result['points_new']
            u = lk_result['u']
            v = lk_result['v']
            magnitude = lk_result['magnitude']
            angle = lk_result['angle']

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
        method: str,
        all_files: List[Path]
    ) -> bool:
        """
        Сохранение суммарного CSV файла со всеми векторами.

        Args:
            camera_folder: Папка камеры
            camera_name: Название камеры
            method: Метод ('farneback' или 'lucas_kanade')
            all_files: Список файлов для объединения

        Returns:
            bool: True если успешно
        """
        summary_path = camera_folder / f"{camera_name}_{method}_sum.csv"

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
        camera_name: str,
        use_lucas_kanade: bool = False
    ) -> Tuple[int, int, List[str], List[str]]:
        """
        Обработка одной камеры с отслеживанием прогресса.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)
            use_lucas_kanade: Использовать Lucas-Kanade вместо Farneback

        Returns:
            Tuple[pairs_processed, vectors_count, errors, warnings]
        """
        if self.input_folder is None or self.output_folder is None:
            logger.error("Входная или выходная папка не установлена")
            return 0, 0, ["Папки не установлены"], []

        camera_input = self.input_folder / camera_name
        method = "lucas_kanade" if use_lucas_kanade else "farneback"
        camera_output = self.output_folder / camera_name / method

        if not camera_input.exists():
            logger.warning(f"Папка {camera_name} не найдена")
            return 0, 0, [f"Папка {camera_name} не найдена"], []

        # Поиск пар изображений
        pairs = self._find_image_pairs(camera_input)
        total_pairs = len(pairs)

        if total_pairs == 0:
            logger.warning(f"В {camera_name} нет пар изображений")
            return 0, 0, [], [f"Нет пар изображений в {camera_name}"]

        processed = 0
        total_vectors = 0
        errors = []
        warnings = []
        output_files = []

        for idx, (path_a, path_b) in enumerate(pairs):
            if self._cancel_requested:
                logger.info(f"Обработка {camera_name} отменена")
                break

            if self._progress_callback:
                progress = FarnebackProgress(
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
                if use_lucas_kanade:
                    flow_result = self.compute_lucas_kanade_flow(img_a, img_b)
                    if flow_result is None:
                        warnings.append(f"Не найдены точки для {path_a.name}")
                        continue

                    # Сохранение результата
                    pair_name = path_a.stem.replace("_a", "")
                    output_filename = f"lk_{pair_name}.csv"
                    output_path = camera_output / output_filename

                    vectors = self._save_lucas_kanade_csv(output_path, flow_result)
                else:
                    flow_result = self.compute_farneback_flow(img_a, img_b)

                    # Сохранение результата
                    pair_name = path_a.stem.replace("_a", "")
                    output_filename = f"fb_{pair_name}.csv"
                    output_path = camera_output / output_filename

                    vectors = self._save_farneback_csv(output_path, flow_result, img_a.shape)

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
            self._save_summary_csv(camera_output.parent, camera_name, method, output_files)

        if self._progress_callback and not self._cancel_requested:
            progress = FarnebackProgress(
                current_pair="",
                total_pairs=total_pairs,
                processed_pairs=total_pairs,
                current_camera=camera_name,
                percentage=100.0,
                message=f"{camera_name}: завершено"
            )
            self._progress_callback(progress)

        return processed, total_vectors, errors, warnings

    def process_all(self, use_lucas_kanade: bool = False) -> FarnebackResult:
        """
        Обработка всех камер с отслеживанием прогресса.

        Args:
            use_lucas_kanade: Использовать Lucas-Kanade вместо Farneback

        Returns:
            FarnebackResult с результатами обработки
        """
        if self.input_folder is None:
            return FarnebackResult(
                success=False,
                total_pairs_processed=0,
                cam1_pairs_count=0,
                cam2_pairs_count=0,
                total_vectors_count=0,
                errors=["Входная папка не установлена"],
                warnings=[],
                output_folder=""
            )

        self._update_output_folder()
        method = "Lucas-Kanade" if use_lucas_kanade else "Farneback"

        logger.info("=" * 60)
        logger.info(f"НАЧАЛО АНАЛИЗА ОПТИЧЕСКОГО ПОТОКА ({method})")
        logger.info(f"Входная папка: {self.input_folder}")
        logger.info(f"Выходная папка: {self.output_folder}")

        if use_lucas_kanade:
            cfg = self.lucas_kanade_config
            logger.info(f"Параметры: max_corners={cfg.max_corners}, "
                       f"quality_level={cfg.quality_level}, win_size={cfg.win_size}")
        else:
            cfg = self.farneback_config
            logger.info(f"Параметры: pyr_scale={cfg.pyr_scale}, levels={cfg.levels}, "
                       f"winsize={cfg.winsize}, iterations={cfg.iterations}")

        logger.info("=" * 60)

        self._cancel_requested = False

        logger.info("\n--- Обработка cam_1 ---")
        cam1_pairs, cam1_vectors, cam1_errors, cam1_warnings = self.process_camera(
            "cam_1", use_lucas_kanade
        )

        cam2_pairs = 0
        cam2_vectors = 0
        cam2_errors = []
        cam2_warnings = []

        if not self._cancel_requested:
            logger.info("\n--- Обработка cam_2 ---")
            cam2_pairs, cam2_vectors, cam2_errors, cam2_warnings = self.process_camera(
                "cam_2", use_lucas_kanade
            )

        total_pairs = cam1_pairs + cam2_pairs
        total_vectors = cam1_vectors + cam2_vectors
        all_errors = cam1_errors + cam2_errors
        all_warnings = cam1_warnings + cam2_warnings
        success = not self._cancel_requested and len(all_errors) == 0

        logger.info("\n" + "=" * 60)
        logger.info(f"РЕЗУЛЬТАТЫ АНАЛИЗА ({method})")
        logger.info("=" * 60)
        logger.info(f"cam_1: {cam1_pairs} пар, {cam1_vectors} векторов")
        logger.info(f"cam_2: {cam2_pairs} пар, {cam2_vectors} векторов")
        logger.info(f"Всего пар: {total_pairs}")
        logger.info(f"Всего векторов: {total_vectors}")
        logger.info(f"Ошибок: {len(all_errors)}")
        logger.info(f"Выходная папка: {self.output_folder}")

        if self._cancel_requested:
            logger.info("Обработка была отменена")

        logger.info("=" * 60)

        return FarnebackResult(
            success=success,
            total_pairs_processed=total_pairs,
            cam1_pairs_count=cam1_pairs,
            cam2_pairs_count=cam2_pairs,
            total_vectors_count=total_vectors,
            errors=all_errors,
            warnings=all_warnings,
            output_folder=str(self.output_folder)
        )

    def get_preview(
        self,
        camera_name: str,
        pair_index: int = 0,
        use_lucas_kanade: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Получение предварительного просмотра анализа для пары изображений.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)
            pair_index: Индекс пары для предпросмотра
            use_lucas_kanade: Использовать Lucas-Kanade вместо Farneback

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
            if use_lucas_kanade:
                flow_result = self.compute_lucas_kanade_flow(img_a, img_b)
                if flow_result is None:
                    return None

                return {
                    'image_a': img_a,
                    'image_b': img_b,
                    'method': 'lucas_kanade',
                    'points_old': flow_result['points_old'],
                    'points_new': flow_result['points_new'],
                    'u': flow_result['u'],
                    'v': flow_result['v'],
                    'magnitude': flow_result['magnitude'],
                    'angle': flow_result['angle'],
                    'total_detected': flow_result['total_detected'],
                    'total_tracked': flow_result['total_tracked'],
                    'mean_magnitude': float(np.mean(flow_result['magnitude'])),
                    'max_magnitude': float(np.max(flow_result['magnitude'])),
                    'pair_names': (path_a.name, path_b.name)
                }
            else:
                flow_result = self.compute_farneback_flow(img_a, img_b)
                stats = self._compute_flow_statistics(flow_result, img_a)

                return {
                    'image_a': img_a,
                    'image_b': img_b,
                    'method': 'farneback',
                    'flow': flow_result['flow'],
                    'u': flow_result['u'],
                    'v': flow_result['v'],
                    'magnitude': flow_result['magnitude'],
                    'angle': flow_result['angle'],
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
