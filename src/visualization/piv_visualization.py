"""
Модуль визуализации результатов PIV анализа для GUI приложения ParticleAnalysis.

Этот модуль предназначен для визуализации результатов PIV анализа (первые 10 пар):
- Чтение векторов скоростей из CSV файлов
- Наложение векторов (стрелок) на исходные изображения
- Callback функции для отслеживания прогресса
- Предварительный просмотр для GUI

Входные данные:
- Папка с исходными изображениями (intensity_filtered_XXXX или cam_sorted)
- Папка с результатами PIV анализа (PIV_XXXX): cam_1/piv_1_a_to_1_b.csv, ...

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import numpy as np
import cv2
from typing import Optional, Callable, List, Tuple, Dict
from pathlib import Path
from dataclasses import dataclass
import logging
import csv
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PIVVector:
    """Вектор скорости из PIV анализа."""
    x: float
    y: float
    u: float
    v: float
    magnitude: float


@dataclass
class PIVVisualizationProgress:
    """Информация о прогрессе визуализации PIV."""
    current_file: str
    total_files: int
    processed_files: int
    current_camera: str
    percentage: float
    message: str


@dataclass
class PIVVisualizationConfig:
    """Конфигурация визуализации PIV."""
    arrow_color: Tuple[int, int, int] = (0, 255, 0)  # Зелёный (BGR)
    arrow_thickness: int = 1  # Толщина стрелки
    arrow_scale: float = 5.0  # Масштаб стрелок (увеличение длины)
    arrow_tip_length: float = 0.3  # Длина наконечника относительно стрелки
    show_magnitude_color: bool = True  # Цвет по магнитуде
    colormap: int = cv2.COLORMAP_JET  # Цветовая карта для магнитуды
    max_pairs: int = 10  # Максимальное количество пар для обработки


@dataclass
class PIVVisualizationResult:
    """Результат визуализации PIV."""
    success: bool
    total_pairs_processed: int
    cam1_visualizations: int
    cam2_visualizations: int
    errors: List[str]
    output_folder: str


class PIVVisualizer:
    """
    Класс для визуализации результатов PIV анализа.

    Читает CSV файлы с векторами скоростей и накладывает
    стрелки на исходные изображения (первые 10 пар).
    """

    def __init__(self):
        """Инициализация визуализатора PIV."""
        self.original_folder: Optional[Path] = None  # Папка с исходными изображениями
        self.piv_folder: Optional[Path] = None  # Папка PIV_XXXX с результатами
        self.output_folder: Optional[Path] = None  # Папка для визуализаций

        self.config = PIVVisualizationConfig()

        self._cancel_requested: bool = False
        self._progress_callback: Optional[Callable[[PIVVisualizationProgress], None]] = None

        logger.info("Инициализирован модуль визуализации PIV")

    def set_original_folder(self, folder_path: str) -> bool:
        """
        Установка папки с исходными изображениями.

        Args:
            folder_path: Путь к папке с изображениями (intensity_filtered_XXXX или cam_sorted)

        Returns:
            bool: True если папка валидна
        """
        path = Path(folder_path)

        if not path.exists():
            logger.error(f"Папка не существует: {folder_path}")
            return False

        cam1 = path / "cam_1"
        cam2 = path / "cam_2"

        if not cam1.exists() and not cam2.exists():
            logger.error("Папка должна содержать подпапки cam_1 и/или cam_2")
            return False

        self.original_folder = path
        logger.info(f"Установлена папка исходных изображений: {path}")

        return True

    def set_piv_folder(self, folder_path: str) -> bool:
        """
        Установка папки с результатами PIV анализа.

        Args:
            folder_path: Путь к папке PIV_XXXX

        Returns:
            bool: True если папка валидна
        """
        path = Path(folder_path)

        if not path.exists():
            logger.error(f"Папка не существует: {folder_path}")
            return False

        # Проверяем наличие папок с результатами PIV
        cam1_piv = path / "cam_1"
        cam2_piv = path / "cam_2"

        if not cam1_piv.exists() and not cam2_piv.exists():
            logger.error("Папка должна содержать cam_1 и/или cam_2 с результатами PIV")
            return False

        self.piv_folder = path
        self._update_output_folder()
        logger.info(f"Установлена папка PIV результатов: {path}")

        return True

    def _update_output_folder(self) -> None:
        """Обновление пути выходной папки для визуализаций."""
        if self.piv_folder is not None:
            self.output_folder = self.piv_folder / "piv_visualization"
            logger.info(f"Выходная папка визуализаций: {self.output_folder}")

    def set_visualization_config(self,
                                  arrow_color: Tuple[int, int, int] = (0, 255, 0),
                                  arrow_thickness: int = 1,
                                  arrow_scale: float = 5.0,
                                  arrow_tip_length: float = 0.3,
                                  show_magnitude_color: bool = True,
                                  max_pairs: int = 10) -> None:
        """
        Установка параметров визуализации.

        Args:
            arrow_color: Цвет стрелок (BGR) - используется если show_magnitude_color=False
            arrow_thickness: Толщина стрелок
            arrow_scale: Масштаб стрелок (увеличение длины)
            arrow_tip_length: Длина наконечника относительно стрелки
            show_magnitude_color: Использовать цвет по магнитуде
            max_pairs: Максимальное количество пар для обработки
        """
        self.config = PIVVisualizationConfig(
            arrow_color=arrow_color,
            arrow_thickness=arrow_thickness,
            arrow_scale=arrow_scale,
            arrow_tip_length=arrow_tip_length,
            show_magnitude_color=show_magnitude_color,
            max_pairs=max_pairs
        )
        logger.info(f"Обновлены параметры визуализации: scale={arrow_scale}, max_pairs={max_pairs}")

    def set_progress_callback(self, callback: Callable[[PIVVisualizationProgress], None]) -> None:
        """Установка callback для отслеживания прогресса."""
        self._progress_callback = callback

    def cancel_processing(self) -> None:
        """Запрос на отмену обработки."""
        self._cancel_requested = True
        logger.info("Запрошена отмена визуализации PIV")

    def _load_original_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Загрузка исходного изображения и конвертация для визуализации.

        Args:
            image_path: Путь к изображению

        Returns:
            numpy.ndarray (8-бит BGR) или None
        """
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.error(f"Не удалось загрузить: {image_path}")
                return None

            # Конвертация 16-бит в 8-бит
            if img.dtype == np.uint16:
                img_min = img.min()
                img_max = img.max()
                if img_max > img_min:
                    img_8bit = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_8bit = np.zeros_like(img, dtype=np.uint8)
            else:
                img_8bit = img

            # Конвертация в BGR для рисования цветных стрелок
            if len(img_8bit.shape) == 2:
                img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img_8bit

            return img_bgr

        except Exception as e:
            logger.error(f"Ошибка загрузки {image_path}: {e}")
            return None

    def _load_piv_csv(self, csv_path: Path) -> List[PIVVector]:
        """
        Загрузка векторов скоростей из CSV файла PIV.

        Args:
            csv_path: Путь к CSV файлу

        Returns:
            Список векторов
        """
        vectors = []

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')

                for row in reader:
                    try:
                        vector = PIVVector(
                            x=float(row['X']),
                            y=float(row['Y']),
                            u=float(row['U']),
                            v=float(row['V']),
                            magnitude=float(row['Magnitude'])
                        )
                        vectors.append(vector)
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Ошибка парсинга строки в {csv_path.name}: {e}")
                        continue

            logger.debug(f"Загружено {len(vectors)} векторов из {csv_path.name}")

        except Exception as e:
            logger.error(f"Ошибка чтения CSV {csv_path}: {e}")

        return vectors

    def _get_color_by_magnitude(self, magnitude: float, min_mag: float, max_mag: float) -> Tuple[int, int, int]:
        """
        Получение цвета по магнитуде вектора.

        Args:
            magnitude: Магнитуда вектора
            min_mag: Минимальная магнитуда
            max_mag: Максимальная магнитуда

        Returns:
            Цвет (BGR)
        """
        if max_mag <= min_mag:
            return (0, 255, 0)  # Зелёный по умолчанию

        # Нормализация магнитуды к диапазону [0, 255]
        normalized = int(255 * (magnitude - min_mag) / (max_mag - min_mag))
        normalized = max(0, min(255, normalized))

        # Создание цветовой карты
        colormap_img = np.zeros((1, 256, 3), dtype=np.uint8)
        for i in range(256):
            colormap_img[0, i] = [i, i, i]
        colormap_img = cv2.applyColorMap(colormap_img, self.config.colormap)

        # Получение цвета
        color = tuple(int(c) for c in colormap_img[0, normalized])
        return color

    def create_visualization(self,
                            original_image: np.ndarray,
                            vectors: List[PIVVector]) -> np.ndarray:
        """
        Создание визуализации PIV на исходном изображении.

        На изображение накладываются стрелки векторов скоростей.

        Args:
            original_image: Исходное изображение (BGR)
            vectors: Список векторов скоростей

        Returns:
            Изображение с визуализацией (BGR)
        """
        vis_image = original_image.copy()
        cfg = self.config
        h, w = vis_image.shape[:2]

        if not vectors:
            return vis_image

        # Вычисление диапазона магнитуд для цветовой карты
        magnitudes = [v.magnitude for v in vectors if v.magnitude > 0]
        if magnitudes:
            min_mag = min(magnitudes)
            max_mag = max(magnitudes)
        else:
            min_mag, max_mag = 0, 1

        # Рисуем стрелки
        for vector in vectors:
            # Начальная точка (позиция вектора)
            x_start = int(round(vector.x))
            y_start = int(round(vector.y))

            # Конечная точка (с учётом масштаба)
            x_end = int(round(vector.x + vector.u * cfg.arrow_scale))
            y_end = int(round(vector.y + vector.v * cfg.arrow_scale))

            # Проверка границ изображения
            if not (0 <= x_start < w and 0 <= y_start < h):
                continue

            # Пропускаем нулевые векторы
            if vector.u == 0 and vector.v == 0:
                continue

            # Определение цвета
            if cfg.show_magnitude_color:
                color = self._get_color_by_magnitude(vector.magnitude, min_mag, max_mag)
            else:
                color = cfg.arrow_color

            # Рисуем стрелку
            cv2.arrowedLine(
                vis_image,
                (x_start, y_start),
                (x_end, y_end),
                color,
                cfg.arrow_thickness,
                tipLength=cfg.arrow_tip_length
            )

        return vis_image

    def _save_visualization(self, image: np.ndarray, output_path: Path) -> bool:
        """
        Сохранение изображения визуализации.

        Args:
            image: Изображение для сохранения
            output_path: Путь к выходному файлу

        Returns:
            bool: True если успешно
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения визуализации {output_path}: {e}")
            return False

    def _extract_pair_number(self, csv_filename: str) -> Optional[int]:
        """
        Извлечение номера пары из имени CSV файла.

        Args:
            csv_filename: Имя файла (например, "piv_1_a_to_1_b.csv")

        Returns:
            Номер пары или None
        """
        # Паттерн: piv_NUMBER_a_to_NUMBER_b.csv
        match = re.match(r'piv_(\d+)_a_to_\d+_b\.csv', csv_filename)
        if match:
            return int(match.group(1))
        return None

    def process_pair(self, camera_name: str, pair_number: int, csv_path: Path) -> Dict:
        """
        Визуализация одной пары изображений PIV.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)
            pair_number: Номер пары
            csv_path: Путь к CSV файлу с векторами

        Returns:
            Словарь с результатами
        """
        result = {
            'success': False,
            'pair_number': pair_number,
            'camera': camera_name,
            'vectors_count': 0,
            'visualizations_created': 0,
            'errors': []
        }

        # Пути к файлам
        original_cam = self.original_folder / camera_name
        output_cam = self.output_folder / camera_name

        # Пути к исходным изображениям
        original_a_path = original_cam / f"{pair_number}_a.png"
        original_b_path = original_cam / f"{pair_number}_b.png"

        # Загрузка векторов из CSV
        vectors = self._load_piv_csv(csv_path)
        result['vectors_count'] = len(vectors)

        if not vectors:
            result['errors'].append(f"Нет данных в {csv_path.name}")
            return result

        # Визуализация для кадра A
        if original_a_path.exists():
            original_a = self._load_original_image(original_a_path)
            if original_a is not None:
                vis_a = self.create_visualization(original_a, vectors)
                output_a_path = output_cam / f"{pair_number}_a_piv_vis.png"
                if self._save_visualization(vis_a, output_a_path):
                    result['visualizations_created'] += 1
            else:
                result['errors'].append(f"Ошибка загрузки {original_a_path.name}")
        else:
            result['errors'].append(f"Не найден файл: {original_a_path.name}")

        # Визуализация для кадра B
        if original_b_path.exists():
            original_b = self._load_original_image(original_b_path)
            if original_b is not None:
                vis_b = self.create_visualization(original_b, vectors)
                output_b_path = output_cam / f"{pair_number}_b_piv_vis.png"
                if self._save_visualization(vis_b, output_b_path):
                    result['visualizations_created'] += 1
            else:
                result['errors'].append(f"Ошибка загрузки {original_b_path.name}")
        else:
            result['errors'].append(f"Не найден файл: {original_b_path.name}")

        result['success'] = result['visualizations_created'] > 0
        return result

    def process_camera(self, camera_name: str) -> Tuple[int, int, List[str]]:
        """
        Визуализация первых N пар для одной камеры.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Tuple[pairs_processed, visualizations_created, errors]
        """
        if self.original_folder is None or self.piv_folder is None:
            return 0, 0, ["Папки не установлены"]

        piv_cam_folder = self.piv_folder / camera_name
        if not piv_cam_folder.exists():
            return 0, 0, [f"Папка {camera_name} не найдена в PIV результатах"]

        # Находим все CSV файлы с результатами PIV
        csv_files = sorted(piv_cam_folder.glob("piv_*_a_to_*_b.csv"))

        if not csv_files:
            return 0, 0, [f"Нет CSV файлов PIV в {camera_name}"]

        # Собираем пары с номерами
        pairs_info = []
        for csv_file in csv_files:
            pair_num = self._extract_pair_number(csv_file.name)
            if pair_num is not None:
                pairs_info.append((pair_num, csv_file))

        # Сортируем по номеру пары и берём первые N
        pairs_info.sort(key=lambda x: x[0])
        pairs_info = pairs_info[:self.config.max_pairs]

        total_files = len(pairs_info)
        pairs_processed = 0
        visualizations_created = 0
        errors = []

        for idx, (pair_num, csv_file) in enumerate(pairs_info):
            if self._cancel_requested:
                break

            if self._progress_callback:
                progress = PIVVisualizationProgress(
                    current_file=csv_file.name,
                    total_files=total_files,
                    processed_files=idx,
                    current_camera=camera_name,
                    percentage=(idx / total_files) * 100,
                    message=f"Визуализация PIV {camera_name}: пара {pair_num}"
                )
                self._progress_callback(progress)

            result = self.process_pair(camera_name, pair_num, csv_file)

            if result['success']:
                pairs_processed += 1
                visualizations_created += result['visualizations_created']

            errors.extend(result['errors'])

        if self._progress_callback and not self._cancel_requested:
            progress = PIVVisualizationProgress(
                current_file="",
                total_files=total_files,
                processed_files=total_files,
                current_camera=camera_name,
                percentage=100.0,
                message=f"{camera_name}: завершено"
            )
            self._progress_callback(progress)

        return pairs_processed, visualizations_created, errors

    def process_all(self) -> PIVVisualizationResult:
        """
        Визуализация первых N пар для всех камер.

        Returns:
            PIVVisualizationResult с результатами
        """
        if self.original_folder is None:
            return PIVVisualizationResult(
                success=False,
                total_pairs_processed=0,
                cam1_visualizations=0,
                cam2_visualizations=0,
                errors=["Папка исходных изображений не установлена"],
                output_folder=""
            )

        if self.piv_folder is None:
            return PIVVisualizationResult(
                success=False,
                total_pairs_processed=0,
                cam1_visualizations=0,
                cam2_visualizations=0,
                errors=["Папка PIV результатов не установлена"],
                output_folder=""
            )

        self._update_output_folder()

        logger.info("=" * 60)
        logger.info("НАЧАЛО ВИЗУАЛИЗАЦИИ PIV")
        logger.info(f"Исходные изображения: {self.original_folder}")
        logger.info(f"PIV результаты: {self.piv_folder}")
        logger.info(f"Выходная папка: {self.output_folder}")
        logger.info(f"Максимум пар: {self.config.max_pairs}")
        logger.info("=" * 60)

        self._cancel_requested = False

        # Обработка cam_1
        cam1_pairs = 0
        cam1_vis = 0
        cam1_errors = []

        cam1_piv_folder = self.piv_folder / "cam_1"
        if cam1_piv_folder.exists():
            logger.info("\n--- Визуализация PIV cam_1 ---")
            cam1_pairs, cam1_vis, cam1_errors = self.process_camera("cam_1")

        # Обработка cam_2
        cam2_pairs = 0
        cam2_vis = 0
        cam2_errors = []

        if not self._cancel_requested:
            cam2_piv_folder = self.piv_folder / "cam_2"
            if cam2_piv_folder.exists():
                logger.info("\n--- Визуализация PIV cam_2 ---")
                cam2_pairs, cam2_vis, cam2_errors = self.process_camera("cam_2")

        total_pairs = cam1_pairs + cam2_pairs
        all_errors = cam1_errors + cam2_errors

        # Фильтруем критические ошибки
        critical_errors = [e for e in all_errors if "Ошибка" in e]
        success = not self._cancel_requested and len(critical_errors) == 0 and total_pairs > 0

        logger.info("\n" + "=" * 60)
        logger.info("РЕЗУЛЬТАТЫ ВИЗУАЛИЗАЦИИ PIV")
        logger.info("=" * 60)
        logger.info(f"cam_1: {cam1_pairs} пар, {cam1_vis} визуализаций")
        logger.info(f"cam_2: {cam2_pairs} пар, {cam2_vis} визуализаций")
        logger.info(f"Всего пар: {total_pairs}")
        logger.info(f"Выходная папка: {self.output_folder}")

        if self._cancel_requested:
            logger.info("Визуализация была отменена")

        logger.info("=" * 60)

        return PIVVisualizationResult(
            success=success,
            total_pairs_processed=total_pairs,
            cam1_visualizations=cam1_vis,
            cam2_visualizations=cam2_vis,
            errors=all_errors,
            output_folder=str(self.output_folder) if self.output_folder else ""
        )

    def get_preview(self, camera_name: str, pair_number: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Получение предварительного просмотра визуализации PIV для GUI.

        Args:
            camera_name: Название камеры
            pair_number: Номер пары

        Returns:
            Tuple[vis_a, vis_b] или None
        """
        if self.original_folder is None or self.piv_folder is None:
            return None

        original_cam = self.original_folder / camera_name
        piv_cam = self.piv_folder / camera_name

        # Путь к CSV с векторами
        csv_path = piv_cam / f"piv_{pair_number}_a_to_{pair_number}_b.csv"
        if not csv_path.exists():
            return None

        vectors = self._load_piv_csv(csv_path)
        if not vectors:
            return None

        # Загружаем и визуализируем изображения
        original_a = self._load_original_image(original_cam / f"{pair_number}_a.png")
        original_b = self._load_original_image(original_cam / f"{pair_number}_b.png")

        if original_a is None or original_b is None:
            return None

        vis_a = self.create_visualization(original_a, vectors)
        vis_b = self.create_visualization(original_b, vectors)

        return vis_a, vis_b

    def get_pair_statistics(self, camera_name: str, pair_number: int) -> Optional[Dict]:
        """
        Получение статистики векторов для пары.

        Args:
            camera_name: Название камеры
            pair_number: Номер пары

        Returns:
            Словарь со статистикой или None
        """
        if self.piv_folder is None:
            return None

        piv_cam = self.piv_folder / camera_name
        csv_path = piv_cam / f"piv_{pair_number}_a_to_{pair_number}_b.csv"

        if not csv_path.exists():
            return None

        vectors = self._load_piv_csv(csv_path)

        if not vectors:
            return {'vectors_count': 0}

        magnitudes = [v.magnitude for v in vectors]
        u_values = [v.u for v in vectors]
        v_values = [v.v for v in vectors]

        # Фильтруем ненулевые магнитуды
        non_zero_magnitudes = [m for m in magnitudes if m > 0]

        stats = {
            'vectors_count': len(vectors),
            'vectors_with_velocity': len(non_zero_magnitudes)
        }

        if non_zero_magnitudes:
            stats.update({
                'mean_magnitude': np.mean(non_zero_magnitudes),
                'max_magnitude': np.max(non_zero_magnitudes),
                'min_magnitude': np.min(non_zero_magnitudes),
                'std_magnitude': np.std(non_zero_magnitudes),
                'mean_u': np.mean([u for u, m in zip(u_values, magnitudes) if m > 0]),
                'mean_v': np.mean([v for v, m in zip(v_values, magnitudes) if m > 0]),
                'std_u': np.std([u for u, m in zip(u_values, magnitudes) if m > 0]),
                'std_v': np.std([v for v, m in zip(v_values, magnitudes) if m > 0])
            })
        else:
            stats.update({
                'mean_magnitude': 0,
                'mean_u': 0,
                'mean_v': 0
            })

        return stats
