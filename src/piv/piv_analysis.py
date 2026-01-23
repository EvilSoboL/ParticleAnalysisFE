"""
Модуль PIV (Particle Image Velocimetry) анализа для GUI приложения ParticleAnalysis.

Этот модуль предназначен для интеграции с графическим интерфейсом и предоставляет:
- PIV анализ последовательных изображений с использованием OpenPIV
- Вычисление векторных полей скоростей
- Экспорт результатов в CSV формат
- Callback функции для отслеживания прогресса
- Пошаговую обработку с возможностью отмены

Требования:
    pip install openpiv

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import numpy as np
from PIL import Image
from typing import Optional, Callable, List, Tuple, Dict
from pathlib import Path
from dataclasses import dataclass
import csv
import logging
import re

try:
    from openpiv import tools, pyprocess, validation, filters, scaling
    OPENPIV_AVAILABLE = True
except ImportError:
    OPENPIV_AVAILABLE = False
    logging.warning(
        "OpenPIV не установлен. Установите: pip install openpiv\n"
        "Модуль PIV анализа не будет работать без этой библиотеки."
    )


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PIVProgress:
    """Класс для передачи информации о прогрессе обработки."""
    current_pair: str
    total_pairs: int
    processed_pairs: int
    current_camera: str
    percentage: float
    message: str


@dataclass
class PIVResult:
    """Результат PIV анализа."""
    success: bool
    total_pairs_processed: int
    cam1_vectors_count: int
    cam2_vectors_count: int
    errors: List[str]
    warnings: List[str]
    output_folder: str


@dataclass
class PIVConfig:
    """Конфигурация PIV анализа."""
    # Параметры окна корреляции
    window_size: int = 32  # Размер окна (обычно 16, 32, 64, 128)
    overlap: int = 16  # Перекрытие окон (обычно window_size // 2)
    search_area_size: int = 64  # Размер области поиска

    # Параметры валидации
    validation_method: str = "sig2noise"  # Метод валидации: 'sig2noise', 'mean_velocity', 'median_velocity'
    sig2noise_threshold: float = 1.3  # Порог для signal-to-noise

    # Параметры фильтрации
    filter_method: str = "localmean"  # Метод фильтрации: 'localmean', 'median'
    max_filter_iteration: int = 3  # Максимальное количество итераций фильтрации
    filter_kernel_size: int = 2  # Размер ядра фильтра

    # Физические параметры
    dt: float = 0.002  # Временной интервал между кадрами в паре (_a и _b): 2 мкс = 0.002 мс
    scaling_factor: float = 1.0  # Масштабный коэффициент (пиксели -> мм)


class PIVAnalyzer:
    """
    Класс для PIV анализа с поддержкой GUI.

    Выполняет анализ векторного поля скоростей между последовательными
    изображениями с использованием библиотеки OpenPIV.
    """

    def __init__(self):
        """Инициализация модуля PIV анализа."""
        if not OPENPIV_AVAILABLE:
            raise ImportError(
                "OpenPIV не установлен. Установите библиотеку командой:\n"
                "pip install openpiv"
            )

        self.input_folder: Optional[Path] = None
        self.output_folder: Optional[Path] = None
        self.config = PIVConfig()
        self._cancel_requested: bool = False
        self._progress_callback: Optional[Callable[[PIVProgress], None]] = None

        logger.info("Инициализирован модуль PIV анализа")

    def set_input_folder(self, folder_path: str) -> bool:
        """
        Установка входной папки (папка intensity_filtered_XXXX).

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
        match = re.search(r'intensity_filtered_(\d+)', folder_name)

        if match:
            return match.group(1)

        return "unknown"

    def _update_output_folder(self) -> None:
        """Обновление пути выходной папки."""
        if self.input_folder is not None:
            threshold = self._extract_threshold_from_folder_name()
            output_name = f"PIV_{threshold}"
            # Выходная папка создается рядом с входной
            self.output_folder = self.input_folder.parent / output_name
            logger.info(f"Выходная папка: {self.output_folder}")

    def set_piv_config(
        self,
        window_size: Optional[int] = None,
        overlap: Optional[int] = None,
        search_area_size: Optional[int] = None,
        dt: Optional[float] = None,
        scaling_factor: Optional[float] = None,
        sig2noise_threshold: Optional[float] = None
    ) -> bool:
        """
        Установка параметров PIV анализа.

        Args:
            window_size: Размер окна корреляции (16, 32, 64, 128)
            overlap: Перекрытие окон (обычно window_size // 2)
            search_area_size: Размер области поиска
            dt: Временной интервал между кадрами (мс)
            scaling_factor: Масштабный коэффициент (пиксели -> мм)
            sig2noise_threshold: Порог signal-to-noise для валидации

        Returns:
            bool: True если параметры валидны
        """
        if window_size is not None:
            if window_size not in [16, 32, 64, 128]:
                logger.error(f"window_size должен быть 16, 32, 64 или 128: {window_size}")
                return False
            self.config.window_size = window_size

        if overlap is not None:
            if overlap < 0 or overlap >= self.config.window_size:
                logger.error(
                    f"overlap должен быть в диапазоне [0, {self.config.window_size}): {overlap}"
                )
                return False
            self.config.overlap = overlap

        if search_area_size is not None:
            if search_area_size < self.config.window_size:
                logger.error(
                    f"search_area_size должен быть >= window_size: "
                    f"{search_area_size} < {self.config.window_size}"
                )
                return False
            self.config.search_area_size = search_area_size

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

        if sig2noise_threshold is not None:
            if sig2noise_threshold <= 0:
                logger.error(f"sig2noise_threshold должен быть > 0: {sig2noise_threshold}")
                return False
            self.config.sig2noise_threshold = sig2noise_threshold

        logger.info(
            f"Параметры PIV установлены: window_size={self.config.window_size}, "
            f"overlap={self.config.overlap}, dt={self.config.dt}, "
            f"scaling={self.config.scaling_factor}"
        )
        return True

    def set_progress_callback(self, callback: Callable[[PIVProgress], None]) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция, принимающая PIVProgress
        """
        self._progress_callback = callback
        logger.debug("Установлен callback для прогресса")

    def cancel_processing(self) -> None:
        """Запрос на отмену обработки."""
        self._cancel_requested = True
        logger.info("Запрошена отмена обработки")

    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Загрузка изображения для PIV анализа.

        Args:
            image_path: Путь к изображению

        Returns:
            numpy.ndarray или None в случае ошибки
        """
        try:
            img = Image.open(image_path)
            img_array = np.array(img)

            # Конвертация в grayscale если необходимо
            if len(img_array.shape) == 3:
                img_array = np.mean(img_array, axis=2)

            # Нормализация к uint8 для OpenPIV
            if img_array.dtype == np.uint16:
                img_array = (img_array / 256).astype(np.uint8)

            return img_array

        except Exception as e:
            logger.error(f"Ошибка загрузки {image_path.name}: {e}")
            return None

    def _piv_analysis_pair(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Выполнение PIV анализа для пары изображений.

        Args:
            image_a: Первое изображение
            image_b: Второе изображение

        Returns:
            Tuple[x, y, u, v, sig2noise]: Координаты и векторы скоростей
        """
        cfg = self.config

        # Базовый PIV анализ
        u, v, sig2noise = pyprocess.extended_search_area_piv(
            image_a.astype(np.int32),
            image_b.astype(np.int32),
            window_size=cfg.window_size,
            overlap=cfg.overlap,
            dt=cfg.dt,
            search_area_size=cfg.search_area_size,
            sig2noise_method='peak2peak'
        )

        # Координаты центров окон
        x, y = pyprocess.get_coordinates(
            image_size=image_a.shape,
            search_area_size=cfg.search_area_size,
            overlap=cfg.overlap
        )

        # Валидация векторов
        if cfg.validation_method == 'sig2noise':
            # sig2noise_val возвращает только маску
            mask = validation.sig2noise_val(
                sig2noise,
                threshold=cfg.sig2noise_threshold
            )
        else:
            # global_val требует пороги для u и v
            # Используем широкие пороги по умолчанию
            mask = validation.global_val(
                u, v,
                u_thresholds=(-100, 100),
                v_thresholds=(-100, 100)
            )

        # Замена невалидных векторов
        # replace_outliers возвращает (u, v, flags, ...) или (u, v, ...)
        result = filters.replace_outliers(
            u, v,
            mask,
            method=cfg.filter_method,
            max_iter=cfg.max_filter_iteration,
            kernel_size=cfg.filter_kernel_size
        )
        u, v = result[0], result[1]

        # Применение масштабного коэффициента
        if cfg.scaling_factor != 1.0:
            x = x * cfg.scaling_factor
            y = y * cfg.scaling_factor
            u = u * cfg.scaling_factor
            v = v * cfg.scaling_factor

        return x, y, u, v, sig2noise

    def _process_image_pair(
        self,
        path_a: Path,
        path_b: Path
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Обработка пары изображений.

        Args:
            path_a: Путь к первому изображению
            path_b: Путь к второму изображению

        Returns:
            Tuple[x, y, u, v] или None при ошибке
        """
        # Загрузка изображений
        image_a = self._load_image(path_a)
        image_b = self._load_image(path_b)

        if image_a is None or image_b is None:
            return None

        # PIV анализ
        try:
            x, y, u, v, _ = self._piv_analysis_pair(image_a, image_b)
            return x, y, u, v
        except Exception as e:
            logger.error(f"Ошибка PIV анализа для пары {path_a.name} - {path_b.name}: {e}")
            return None

    def _save_vectors_csv(
        self,
        output_path: Path,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray
    ) -> bool:
        """
        Сохранение векторов в CSV файл.

        Args:
            output_path: Путь к выходному CSV файлу
            x: X координаты
            y: Y координаты
            u: U компоненты скорости
            v: V компоненты скорости

        Returns:
            bool: True если успешно
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(['X', 'Y', 'U', 'V', 'Magnitude'])

                # Разворачиваем массивы в 1D
                x_flat = x.flatten()
                y_flat = y.flatten()
                u_flat = u.flatten()
                v_flat = v.flatten()

                for i in range(len(x_flat)):
                    magnitude = np.sqrt(u_flat[i]**2 + v_flat[i]**2)
                    writer.writerow([
                        f"{x_flat[i]:.3f}",
                        f"{y_flat[i]:.3f}",
                        f"{u_flat[i]:.6f}",
                        f"{v_flat[i]:.6f}",
                        f"{magnitude:.6f}"
                    ])

            logger.info(f"Сохранено {len(x_flat)} векторов в {output_path.name}")
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения CSV {output_path}: {e}")
            return False

    def _save_summary_csv(
        self,
        camera_folder: Path,
        camera_name: str,
        all_vectors: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> bool:
        """
        Сохранение суммарного CSV файла со всеми векторами.

        Args:
            camera_folder: Папка камеры
            camera_name: Название камеры
            all_vectors: Список всех векторов

        Returns:
            bool: True если успешно
        """
        summary_path = camera_folder / f"{camera_name}_vectors_sum.csv"

        try:
            with open(summary_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(['X', 'Y', 'U', 'V', 'Magnitude'])

                total_vectors = 0
                for x, y, u, v in all_vectors:
                    x_flat = x.flatten()
                    y_flat = y.flatten()
                    u_flat = u.flatten()
                    v_flat = v.flatten()

                    for i in range(len(x_flat)):
                        magnitude = np.sqrt(u_flat[i]**2 + v_flat[i]**2)
                        writer.writerow([
                            f"{x_flat[i]:.3f}",
                            f"{y_flat[i]:.3f}",
                            f"{u_flat[i]:.6f}",
                            f"{v_flat[i]:.6f}",
                            f"{magnitude:.6f}"
                        ])
                        total_vectors += 1

            logger.info(f"Сохранен суммарный файл: {summary_path.name} ({total_vectors} векторов)")
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения суммарного CSV: {e}")
            return False

    def process_camera(self, camera_name: str) -> Tuple[int, List[str], List[str]]:
        """
        Обработка одной камеры с отслеживанием прогресса.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Tuple[vectors_count, errors, warnings]
        """
        if self.input_folder is None or self.output_folder is None:
            logger.error("Входная или выходная папка не установлена")
            return 0, ["Папки не установлены"], []

        camera_input = self.input_folder / camera_name
        camera_output = self.output_folder / camera_name

        if not camera_input.exists():
            logger.warning(f"Папка {camera_name} не найдена")
            return 0, [f"Папка {camera_name} не найдена"], []

        # Получение списка изображений
        png_files = sorted(camera_input.glob("*.png"))
        total_files = len(png_files)

        if total_files < 2:
            logger.warning(f"В {camera_name} недостаточно файлов для PIV анализа")
            return 0, [f"Недостаточно файлов в {camera_name}"], []

        # Создание пар изображений _a и _b
        # Разделяем файлы на _a и _b
        files_a = sorted([f for f in png_files if f.stem.endswith('_a')])
        files_b = sorted([f for f in png_files if f.stem.endswith('_b')])

        # Создаем пары по номерам (1_a с 1_b, 2_a с 2_b и т.д.)
        pairs = []
        for file_a in files_a:
            # Извлекаем номер из имени файла (например, "100_a" -> "100")
            number = file_a.stem.rsplit('_', 1)[0]
            # Ищем соответствующий _b файл
            file_b = camera_input / f"{number}_b.png"
            if file_b.exists():
                pairs.append((file_a, file_b))
            else:
                logger.warning(f"Не найден парный файл для {file_a.name}")

        total_pairs = len(pairs)

        if total_pairs == 0:
            logger.warning(f"В {camera_name} не найдено валидных пар изображений (_a и _b)")
            return 0, [f"Нет валидных пар в {camera_name}"], []

        processed = 0
        errors = []
        warnings = []
        all_vectors = []

        for idx, (path_a, path_b) in enumerate(pairs):
            if self._cancel_requested:
                logger.info(f"Обработка {camera_name} отменена")
                break

            if self._progress_callback:
                progress = PIVProgress(
                    current_pair=f"{path_a.name} - {path_b.name}",
                    total_pairs=total_pairs,
                    processed_pairs=idx,
                    current_camera=camera_name,
                    percentage=(idx / total_pairs) * 100,
                    message=f"Обработка {camera_name}: пара {idx+1}/{total_pairs}"
                )
                self._progress_callback(progress)

            # Обработка пары
            result = self._process_image_pair(path_a, path_b)

            if result is None:
                errors.append(f"Ошибка обработки пары: {path_a.name} - {path_b.name}")
                continue

            x, y, u, v = result

            # Сохранение результата для пары
            output_filename = f"piv_{path_a.stem}_to_{path_b.stem}.csv"
            output_path = camera_output / output_filename

            if self._save_vectors_csv(output_path, x, y, u, v):
                all_vectors.append((x, y, u, v))
                processed += 1
            else:
                errors.append(f"Ошибка сохранения: {output_filename}")

        # Сохранение суммарного файла
        if all_vectors:
            self._save_summary_csv(camera_output, camera_name, all_vectors)

        if self._progress_callback and not self._cancel_requested:
            progress = PIVProgress(
                current_pair="",
                total_pairs=total_pairs,
                processed_pairs=total_pairs,
                current_camera=camera_name,
                percentage=100.0,
                message=f"{camera_name}: завершено"
            )
            self._progress_callback(progress)

        # Подсчет общего количества векторов
        total_vectors = sum(x.size for x, _, _, _ in all_vectors)

        return total_vectors, errors, warnings

    def process_all(self) -> PIVResult:
        """
        Обработка всех камер с отслеживанием прогресса.

        Returns:
            PIVResult с результатами обработки
        """
        if self.input_folder is None:
            return PIVResult(
                success=False,
                total_pairs_processed=0,
                cam1_vectors_count=0,
                cam2_vectors_count=0,
                errors=["Входная папка не установлена"],
                warnings=[],
                output_folder=""
            )

        self._update_output_folder()

        logger.info("=" * 60)
        logger.info("НАЧАЛО PIV АНАЛИЗА")
        logger.info(f"Входная папка: {self.input_folder}")
        logger.info(f"Выходная папка: {self.output_folder}")
        logger.info(f"Параметры: window_size={self.config.window_size}, "
                   f"overlap={self.config.overlap}, dt={self.config.dt}")
        logger.info("=" * 60)

        self._cancel_requested = False

        logger.info("\n--- Обработка cam_1 ---")
        cam1_vectors, cam1_errors, cam1_warnings = self.process_camera("cam_1")

        cam2_vectors = 0
        cam2_errors = []
        cam2_warnings = []

        if not self._cancel_requested:
            logger.info("\n--- Обработка cam_2 ---")
            cam2_vectors, cam2_errors, cam2_warnings = self.process_camera("cam_2")

        total_vectors = cam1_vectors + cam2_vectors
        all_errors = cam1_errors + cam2_errors
        all_warnings = cam1_warnings + cam2_warnings
        success = not self._cancel_requested and len(all_errors) == 0

        logger.info("\n" + "=" * 60)
        logger.info("РЕЗУЛЬТАТЫ PIV АНАЛИЗА")
        logger.info("=" * 60)
        logger.info(f"cam_1: {cam1_vectors} векторов, ошибок: {len(cam1_errors)}")
        logger.info(f"cam_2: {cam2_vectors} векторов, ошибок: {len(cam2_errors)}")
        logger.info(f"Всего векторов: {total_vectors}")
        logger.info(f"Выходная папка: {self.output_folder}")

        if self._cancel_requested:
            logger.info("Обработка была отменена")

        logger.info("=" * 60)

        return PIVResult(
            success=success,
            total_pairs_processed=total_vectors,
            cam1_vectors_count=cam1_vectors,
            cam2_vectors_count=cam2_vectors,
            errors=all_errors,
            warnings=all_warnings,
            output_folder=str(self.output_folder)
        )

    def get_preview(
        self,
        camera_name: str,
        pair_index: int = 0
    ) -> Optional[Dict[str, any]]:
        """
        Получение предварительного просмотра PIV анализа для пары изображений.

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

        # Получение пар _a и _b файлов
        png_files = sorted(camera_path.glob("*.png"))
        files_a = sorted([f for f in png_files if f.stem.endswith('_a')])

        if not files_a or pair_index >= len(files_a):
            return None

        # Получение пары по индексу
        path_a = files_a[pair_index]
        number = path_a.stem.rsplit('_', 1)[0]
        path_b = camera_path / f"{number}_b.png"

        if not path_b.exists():
            return None

        # Загрузка изображений
        image_a = self._load_image(path_a)
        image_b = self._load_image(path_b)

        if image_a is None or image_b is None:
            return None

        # PIV анализ
        try:
            x, y, u, v, sig2noise = self._piv_analysis_pair(image_a, image_b)

            magnitude = np.sqrt(u**2 + v**2)

            return {
                'image_a': image_a,
                'image_b': image_b,
                'x': x,
                'y': y,
                'u': u,
                'v': v,
                'magnitude': magnitude,
                'sig2noise': sig2noise,
                'vectors_count': x.size,
                'mean_magnitude': np.mean(magnitude),
                'max_magnitude': np.max(magnitude),
                'pair_names': (path_a.name, path_b.name)
            }
        except Exception as e:
            logger.error(f"Ошибка создания предпросмотра: {e}")
            return None
