"""
Модуль визуализации результатов PTV анализа для GUI приложения ParticleAnalysis.

Этот модуль предназначен для визуализации уже выполненного one-to-one сопоставления:
- Чтение готовых результатов сопоставления из CSV файлов
- Наложение центров частиц и векторов смещения на исходные изображения
- Callback функции для отслеживания прогресса
- Предварительный просмотр для GUI

Входные данные:
- Папка с исходными изображениями (cam_sorted): cam_1/1_a.png, cam_1/1_b.png, ...
- Папка с результатами PTV сопоставления: cam_1_pairs/1_pair.csv, ...

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MatchedPair:
    """Сопоставленная пара частиц из CSV."""
    id: int
    x0: float  # X-координата в кадре A
    y0: float  # Y-координата в кадре A
    dx: float  # Смещение по X
    dy: float  # Смещение по Y
    length: float  # Модуль вектора смещения
    diameter: float  # Диаметр частицы
    area: int  # Площадь частицы


@dataclass
class VisualizationProgress:
    """Информация о прогрессе визуализации."""
    current_file: str
    total_files: int
    processed_files: int
    current_camera: str
    percentage: float
    message: str


@dataclass
class VisualizationConfig:
    """Конфигурация визуализации."""
    particle_a_color: Tuple[int, int, int] = (0, 255, 0)  # Зелёный (BGR) - частицы кадра A
    particle_b_color: Tuple[int, int, int] = (0, 0, 255)  # Красный (BGR) - частицы кадра B
    line_color: Tuple[int, int, int] = (0, 165, 255)  # Оранжевый (BGR) - линии связи
    line_thickness: int = 1  # Толщина линии


@dataclass
class VisualizationResult:
    """Результат визуализации."""
    success: bool
    total_pairs_processed: int
    cam1_visualizations: int
    cam2_visualizations: int
    errors: List[str]
    output_folder: str


class PTVVisualizer:
    """
    Класс для визуализации результатов PTV сопоставления.

    Читает готовые CSV файлы с результатами сопоставления и накладывает
    центры частиц и векторы смещения на исходные изображения.
    """

    def __init__(self):
        """Инициализация визуализатора."""
        self.original_folder: Optional[Path] = None  # Папка cam_sorted с исходниками
        self.ptv_folder: Optional[Path] = None  # Папка PTV_XXXX с результатами
        self.output_folder: Optional[Path] = None  # Папка для визуализаций

        self.config = VisualizationConfig()

        self._cancel_requested: bool = False
        self._progress_callback: Optional[Callable[[VisualizationProgress], None]] = None

        logger.info("Инициализирован модуль визуализации PTV")

    def set_original_folder(self, folder_path: str) -> bool:
        """
        Установка папки с исходными изображениями (cam_sorted).

        Args:
            folder_path: Путь к папке cam_sorted

        Returns:
            bool: True если папка валидна
        """
        path = Path(folder_path)

        if not path.exists():
            logger.error(f"Папка не существует: {folder_path}")
            return False

        cam1 = path / "cam_1"
        cam2 = path / "cam_2"

        if not cam1.exists() or not cam2.exists():
            logger.error("Папка должна содержать подпапки cam_1 и cam_2")
            return False

        self.original_folder = path
        logger.info(f"Установлена папка исходных изображений: {path}")

        return True

    def set_ptv_folder(self, folder_path: str) -> bool:
        """
        Установка папки с результатами PTV анализа.

        Args:
            folder_path: Путь к папке PTV_XXXX

        Returns:
            bool: True если папка валидна
        """
        path = Path(folder_path)

        if not path.exists():
            logger.error(f"Папка не существует: {folder_path}")
            return False

        # Проверяем наличие папок с парами
        cam1_pairs = path / "cam_1_pairs"
        cam2_pairs = path / "cam_2_pairs"

        if not cam1_pairs.exists() and not cam2_pairs.exists():
            logger.error("Папка должна содержать cam_1_pairs и/или cam_2_pairs")
            return False

        self.ptv_folder = path
        self._update_output_folder()
        logger.info(f"Установлена папка PTV результатов: {path}")

        return True

    def _update_output_folder(self) -> None:
        """Обновление пути выходной папки для визуализаций."""
        if self.ptv_folder is not None:
            self.output_folder = self.ptv_folder / "one_to_one_visualization"
            logger.info(f"Выходная папка визуализаций: {self.output_folder}")

    def set_visualization_config(self,
                                  particle_a_color: Tuple[int, int, int] = (0, 255, 0),
                                  particle_b_color: Tuple[int, int, int] = (0, 0, 255),
                                  line_color: Tuple[int, int, int] = (0, 165, 255),
                                  line_thickness: int = 1) -> None:
        """
        Установка параметров визуализации.

        Args:
            particle_a_color: Цвет частиц кадра A (BGR) - по умолчанию зелёный
            particle_b_color: Цвет частиц кадра B (BGR) - по умолчанию красный
            line_color: Цвет соединительных линий (BGR) - по умолчанию оранжевый
            line_thickness: Толщина линии
        """
        self.config = VisualizationConfig(
            particle_a_color=particle_a_color,
            particle_b_color=particle_b_color,
            line_color=line_color,
            line_thickness=line_thickness
        )
        logger.info("Обновлены параметры визуализации")

    def set_progress_callback(self, callback: Callable[[VisualizationProgress], None]) -> None:
        """Установка callback для отслеживания прогресса."""
        self._progress_callback = callback

    def cancel_processing(self) -> None:
        """Запрос на отмену обработки."""
        self._cancel_requested = True
        logger.info("Запрошена отмена визуализации")

    def _load_original_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Загрузка исходного 16-битного изображения и конвертация для визуализации.

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

            # Конвертация в BGR для рисования цветных маркеров
            if len(img_8bit.shape) == 2:
                img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img_8bit

            return img_bgr

        except Exception as e:
            logger.error(f"Ошибка загрузки {image_path}: {e}")
            return None

    def _load_pairs_csv(self, csv_path: Path) -> List[MatchedPair]:
        """
        Загрузка результатов сопоставления из CSV файла.

        Args:
            csv_path: Путь к CSV файлу

        Returns:
            Список сопоставленных пар
        """
        pairs = []

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')

                for row in reader:
                    try:
                        pair = MatchedPair(
                            id=int(row['ID']),
                            x0=float(row['X0']),
                            y0=float(row['Y0']),
                            dx=float(row['dx']),
                            dy=float(row['dy']),
                            length=float(row['L']),
                            diameter=float(row['Diameter']),
                            area=int(row['Area'])
                        )
                        pairs.append(pair)
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Ошибка парсинга строки в {csv_path.name}: {e}")
                        continue

            logger.debug(f"Загружено {len(pairs)} пар из {csv_path.name}")

        except Exception as e:
            logger.error(f"Ошибка чтения CSV {csv_path}: {e}")

        return pairs

    def create_visualization(self,
                            original_image: np.ndarray,
                            pairs: List[MatchedPair]) -> np.ndarray:
        """
        Создание визуализации сопоставления на исходном изображении.

        На изображение накладываются:
        - Зелёные точки (1 пиксель): центры частиц в кадре A (X0, Y0)
        - Красные точки (1 пиксель): центры частиц в кадре B (X0+dx, Y0+dy)
        - Оранжевые линии: соединяют сопоставленные пары

        Args:
            original_image: Исходное изображение (BGR)
            pairs: Список сопоставленных пар

        Returns:
            Изображение с визуализацией (BGR)
        """
        vis_image = original_image.copy()
        cfg = self.config
        h, w = vis_image.shape[:2]

        # Рисуем линии связи
        for pair in pairs:
            # Координаты частицы A
            x_a, y_a = int(round(pair.x0)), int(round(pair.y0))
            # Координаты частицы B (смещённая позиция)
            x_b, y_b = int(round(pair.x0 + pair.dx)), int(round(pair.y0 + pair.dy))

            # Рисуем линию только если есть смещение
            if pair.dx != 0 or pair.dy != 0:
                cv2.line(vis_image, (x_a, y_a), (x_b, y_b),
                        cfg.line_color, cfg.line_thickness)

        # Рисуем точки частиц A (красные, 1 пиксель)
        for pair in pairs:
            x, y = int(round(pair.x0)), int(round(pair.y0))
            # Проверка границ изображения
            if 0 <= x < w and 0 <= y < h:
                vis_image[y, x] = cfg.particle_a_color

        # Рисуем точки частиц B (синие, 1 пиксель)
        for pair in pairs:
            x, y = int(round(pair.x0 + pair.dx)), int(round(pair.y0 + pair.dy))
            # Проверка границ изображения
            if 0 <= x < w and 0 <= y < h:
                vis_image[y, x] = cfg.particle_b_color

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

    def process_pair(self, camera_name: str, pair_number: int) -> Dict:
        """
        Визуализация одной пары изображений.

        Создаёт визуализацию для обоих кадров (1_a и 1_b) с наложением
        результатов сопоставления.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)
            pair_number: Номер пары (1, 2, 3, ...)

        Returns:
            Словарь с результатами
        """
        result = {
            'success': False,
            'pair_number': pair_number,
            'camera': camera_name,
            'pairs_count': 0,
            'visualizations_created': 0,
            'errors': []
        }

        # Пути к файлам
        original_cam = self.original_folder / camera_name
        pairs_folder = self.ptv_folder / f"{camera_name}_pairs"
        output_cam = self.output_folder / camera_name

        # Пути к исходным изображениям
        original_a_path = original_cam / f"{pair_number}_a.png"
        original_b_path = original_cam / f"{pair_number}_b.png"

        # Путь к CSV с парами
        pairs_csv_path = pairs_folder / f"{pair_number}_pair.csv"

        # Проверка существования CSV
        if not pairs_csv_path.exists():
            result['errors'].append(f"Не найден файл: {pairs_csv_path.name}")
            return result

        # Загрузка пар из CSV
        pairs = self._load_pairs_csv(pairs_csv_path)
        result['pairs_count'] = len(pairs)

        if not pairs:
            result['errors'].append(f"Нет данных в {pairs_csv_path.name}")
            return result

        # Визуализация для кадра A
        if original_a_path.exists():
            original_a = self._load_original_image(original_a_path)
            if original_a is not None:
                vis_a = self.create_visualization(original_a, pairs)
                output_a_path = output_cam / f"{pair_number}_a_vis.png"
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
                vis_b = self.create_visualization(original_b, pairs)
                output_b_path = output_cam / f"{pair_number}_b_vis.png"
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
        Визуализация всех пар для одной камеры.

        Args:
            camera_name: Название камеры (cam_1 или cam_2)

        Returns:
            Tuple[pairs_processed, visualizations_created, errors]
        """
        if self.original_folder is None or self.ptv_folder is None:
            return 0, 0, ["Папки не установлены"]

        pairs_folder = self.ptv_folder / f"{camera_name}_pairs"
        if not pairs_folder.exists():
            return 0, 0, [f"Папка {camera_name}_pairs не найдена"]

        # Находим все CSV файлы с парами
        csv_files = sorted(pairs_folder.glob("*_pair.csv"))

        if not csv_files:
            return 0, 0, [f"Нет CSV файлов в {camera_name}_pairs"]

        # Извлекаем номера пар из имён файлов
        pair_numbers = []
        for f in csv_files:
            try:
                # Имя файла: "1_pair.csv" -> номер 1
                pair_num = int(f.stem.split('_')[0])
                pair_numbers.append(pair_num)
            except (ValueError, IndexError):
                continue

        total_files = len(pair_numbers)
        pairs_processed = 0
        visualizations_created = 0
        errors = []

        for idx, pair_num in enumerate(pair_numbers):
            if self._cancel_requested:
                break

            if self._progress_callback:
                progress = VisualizationProgress(
                    current_file=f"{pair_num}_pair.csv",
                    total_files=total_files,
                    processed_files=idx,
                    current_camera=camera_name,
                    percentage=(idx / total_files) * 100,
                    message=f"Визуализация {camera_name}: пара {pair_num}"
                )
                self._progress_callback(progress)

            result = self.process_pair(camera_name, pair_num)

            if result['success']:
                pairs_processed += 1
                visualizations_created += result['visualizations_created']

            errors.extend(result['errors'])

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

        return pairs_processed, visualizations_created, errors

    def process_all(self) -> VisualizationResult:
        """
        Визуализация всех результатов PTV.

        Returns:
            VisualizationResult с результатами
        """
        if self.original_folder is None:
            return VisualizationResult(
                success=False,
                total_pairs_processed=0,
                cam1_visualizations=0,
                cam2_visualizations=0,
                errors=["Папка исходных изображений не установлена"],
                output_folder=""
            )

        if self.ptv_folder is None:
            return VisualizationResult(
                success=False,
                total_pairs_processed=0,
                cam1_visualizations=0,
                cam2_visualizations=0,
                errors=["Папка PTV результатов не установлена"],
                output_folder=""
            )

        self._update_output_folder()

        logger.info("=" * 60)
        logger.info("НАЧАЛО ВИЗУАЛИЗАЦИИ PTV")
        logger.info(f"Исходные изображения: {self.original_folder}")
        logger.info(f"PTV результаты: {self.ptv_folder}")
        logger.info(f"Выходная папка: {self.output_folder}")
        logger.info("=" * 60)

        self._cancel_requested = False

        # Обработка cam_1
        cam1_pairs = 0
        cam1_vis = 0
        cam1_errors = []

        cam1_pairs_folder = self.ptv_folder / "cam_1_pairs"
        if cam1_pairs_folder.exists():
            logger.info("\n--- Визуализация cam_1 ---")
            cam1_pairs, cam1_vis, cam1_errors = self.process_camera("cam_1")

        # Обработка cam_2
        cam2_pairs = 0
        cam2_vis = 0
        cam2_errors = []

        if not self._cancel_requested:
            cam2_pairs_folder = self.ptv_folder / "cam_2_pairs"
            if cam2_pairs_folder.exists():
                logger.info("\n--- Визуализация cam_2 ---")
                cam2_pairs, cam2_vis, cam2_errors = self.process_camera("cam_2")

        total_pairs = cam1_pairs + cam2_pairs
        all_errors = cam1_errors + cam2_errors

        # Фильтруем критические ошибки (не считаем отсутствие файлов критичным)
        critical_errors = [e for e in all_errors if "Ошибка" in e]
        success = not self._cancel_requested and len(critical_errors) == 0 and total_pairs > 0

        logger.info("\n" + "=" * 60)
        logger.info("РЕЗУЛЬТАТЫ ВИЗУАЛИЗАЦИИ")
        logger.info("=" * 60)
        logger.info(f"cam_1: {cam1_pairs} пар, {cam1_vis} визуализаций")
        logger.info(f"cam_2: {cam2_pairs} пар, {cam2_vis} визуализаций")
        logger.info(f"Всего пар: {total_pairs}")
        logger.info(f"Выходная папка: {self.output_folder}")

        if self._cancel_requested:
            logger.info("Визуализация была отменена")

        logger.info("=" * 60)

        return VisualizationResult(
            success=success,
            total_pairs_processed=total_pairs,
            cam1_visualizations=cam1_vis,
            cam2_visualizations=cam2_vis,
            errors=all_errors,
            output_folder=str(self.output_folder) if self.output_folder else ""
        )

    def get_preview(self, camera_name: str,
                   pair_number: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Получение предварительного просмотра визуализации для GUI.

        Args:
            camera_name: Название камеры
            pair_number: Номер пары

        Returns:
            Tuple[vis_a, vis_b] или None
        """
        if self.original_folder is None or self.ptv_folder is None:
            return None

        original_cam = self.original_folder / camera_name
        pairs_folder = self.ptv_folder / f"{camera_name}_pairs"

        # Загружаем пары
        pairs_csv_path = pairs_folder / f"{pair_number}_pair.csv"
        if not pairs_csv_path.exists():
            return None

        pairs = self._load_pairs_csv(pairs_csv_path)
        if not pairs:
            return None

        # Загружаем и визуализируем изображения
        original_a = self._load_original_image(original_cam / f"{pair_number}_a.png")
        original_b = self._load_original_image(original_cam / f"{pair_number}_b.png")

        if original_a is None or original_b is None:
            return None

        vis_a = self.create_visualization(original_a, pairs)
        vis_b = self.create_visualization(original_b, pairs)

        return vis_a, vis_b

    def get_pair_statistics(self, camera_name: str, pair_number: int) -> Optional[Dict]:
        """
        Получение статистики для пары из CSV.

        Args:
            camera_name: Название камеры
            pair_number: Номер пары

        Returns:
            Словарь со статистикой или None
        """
        if self.ptv_folder is None:
            return None

        pairs_folder = self.ptv_folder / f"{camera_name}_pairs"
        pairs_csv_path = pairs_folder / f"{pair_number}_pair.csv"

        if not pairs_csv_path.exists():
            return None

        pairs = self._load_pairs_csv(pairs_csv_path)

        if not pairs:
            return {'pairs_count': 0}

        lengths = [p.length for p in pairs]
        dx_values = [p.dx for p in pairs]
        dy_values = [p.dy for p in pairs]
        diameters = [p.diameter for p in pairs]

        # Фильтруем нулевые смещения для статистики
        non_zero_lengths = [l for l in lengths if l > 0]

        stats = {
            'pairs_count': len(pairs),
            'mean_diameter': np.mean(diameters),
            'min_diameter': np.min(diameters),
            'max_diameter': np.max(diameters),
        }

        if non_zero_lengths:
            stats.update({
                'mean_displacement': np.mean(non_zero_lengths),
                'max_displacement': np.max(non_zero_lengths),
                'min_displacement': np.min(non_zero_lengths),
                'std_displacement': np.std(non_zero_lengths),
                'mean_dx': np.mean([dx for dx, l in zip(dx_values, lengths) if l > 0]),
                'mean_dy': np.mean([dy for dy, l in zip(dy_values, lengths) if l > 0]),
                'matched_with_displacement': len(non_zero_lengths)
            })
        else:
            stats.update({
                'mean_displacement': 0,
                'matched_with_displacement': 0
            })

        return stats