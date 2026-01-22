"""
Тесты для модуля PTVAnalyzer.

Тестирование проводится на реальных тестовых данных.
Тесты вызываются из кода, не из CLI.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import csv
from src.ptv.ptv_analysis import (
    PTVAnalyzer,
    PTVProgress,
    PTVResult,
    Particle,
    ParticlePair,
    DetectionConfig,
    MatchingConfig
)


class TestPTVAnalyzer:
    """Тесты для класса PTVAnalyzer."""

    def __init__(self, test_data_path: str, cleanup: bool = True):
        """
        Инициализация тестов.

        Args:
            test_data_path: Путь к папке с тестовыми данными (binary_filter_XXXX)
            cleanup: Удалять ли выходные папки после тестов (по умолчанию True)
        """
        self.test_data_path = Path(test_data_path)
        self.cleanup = cleanup
        self.results = []

    def setup(self) -> bool:
        """
        Подготовка к тестированию.

        Returns:
            bool: True если подготовка успешна
        """
        if not self.test_data_path.exists():
            print(f"ОШИБКА: Папка с тестовыми данными не найдена: {self.test_data_path}")
            return False

        cam1 = self.test_data_path / "cam_1"
        cam2 = self.test_data_path / "cam_2"

        if not cam1.exists() or not cam2.exists():
            print("ОШИБКА: Тестовая папка должна содержать cam_1 и cam_2")
            return False

        print(f"Тестовые данные: {self.test_data_path}")
        print(f"  cam_1: {len(list(cam1.glob('*.png')))} файлов")
        print(f"  cam_2: {len(list(cam2.glob('*.png')))} файлов")

        return True

    def teardown(self) -> None:
        """Очистка после тестов."""
        pass

    def _log_result(self, test_name: str, passed: bool, message: str = "") -> None:
        """Логирование результата теста."""
        status = "✓ PASSED" if passed else "✗ FAILED"
        self.results.append((test_name, passed, message))
        print(f"  {status}: {test_name}")
        if message and not passed:
            print(f"    → {message}")

    def test_set_input_folder_valid(self) -> bool:
        """Тест установки валидной входной папки."""
        analyzer = PTVAnalyzer()
        result = analyzer.set_input_folder(str(self.test_data_path))

        passed = result is True and analyzer.input_folder == self.test_data_path
        self._log_result("test_set_input_folder_valid", passed)
        return passed

    def test_set_input_folder_invalid(self) -> bool:
        """Тест установки невалидной входной папки."""
        analyzer = PTVAnalyzer()
        result = analyzer.set_input_folder("/nonexistent/path")

        passed = result is False
        self._log_result("test_set_input_folder_invalid", passed)
        return passed

    def test_output_folder_naming(self) -> bool:
        """Тест формирования имени выходной папки."""
        analyzer = PTVAnalyzer()
        analyzer.set_input_folder(str(self.test_data_path))

        # Извлекаем порог из имени входной папки
        folder_name = self.test_data_path.name
        if "binary_filter_" in folder_name:
            threshold = folder_name.replace("binary_filter_", "")
        else:
            threshold = "unknown"

        expected_name = f"PTV_{threshold}"
        expected_path = self.test_data_path.parent / expected_name

        passed = analyzer.output_folder == expected_path
        message = f"Ожидалось: {expected_path}, получено: {analyzer.output_folder}"
        self._log_result("test_output_folder_naming", passed, message if not passed else "")
        return passed

    def test_set_detection_config_valid(self) -> bool:
        """Тест установки валидных параметров детектирования."""
        analyzer = PTVAnalyzer()

        result = analyzer.set_detection_config(min_area=5, max_area=200)

        passed = (
            result is True and
            analyzer.detection_config.min_area == 5 and
            analyzer.detection_config.max_area == 200
        )
        self._log_result("test_set_detection_config_valid", passed)
        return passed

    def test_set_detection_config_invalid(self) -> bool:
        """Тест установки невалидных параметров детектирования."""
        analyzer = PTVAnalyzer()

        result_negative = analyzer.set_detection_config(min_area=-1, max_area=100)
        result_reversed = analyzer.set_detection_config(min_area=100, max_area=50)

        passed = result_negative is False and result_reversed is False
        self._log_result("test_set_detection_config_invalid", passed)
        return passed

    def test_set_matching_config_valid(self) -> bool:
        """Тест установки валидных параметров сопоставления."""
        analyzer = PTVAnalyzer()

        result = analyzer.set_matching_config(max_distance=50.0, max_diameter_diff=3.0)

        passed = (
            result is True and
            analyzer.matching_config.max_distance == 50.0 and
            analyzer.matching_config.max_diameter_diff == 3.0
        )
        self._log_result("test_set_matching_config_valid", passed)
        return passed

    def test_set_matching_config_invalid(self) -> bool:
        """Тест установки невалидных параметров сопоставления."""
        analyzer = PTVAnalyzer()

        result_negative_dist = analyzer.set_matching_config(max_distance=-10, max_diameter_diff=2.0)
        result_negative_diff = analyzer.set_matching_config(max_distance=30, max_diameter_diff=-1.0)

        passed = result_negative_dist is False and result_negative_diff is False
        self._log_result("test_set_matching_config_invalid", passed)
        return passed

    def test_load_binary_image(self) -> bool:
        """Тест загрузки бинаризованного изображения."""
        analyzer = PTVAnalyzer()

        cam1_path = self.test_data_path / "cam_1"
        png_files = list(cam1_path.glob("*.png"))

        if not png_files:
            self._log_result("test_load_binary_image", False, "Нет PNG файлов для теста")
            return False

        img_array = analyzer._load_binary_image(png_files[0])

        passed = (
            img_array is not None and
            isinstance(img_array, np.ndarray) and
            img_array.dtype == np.uint8
        )
        self._log_result("test_load_binary_image", passed)
        return passed

    def test_detect_particles_synthetic(self) -> bool:
        """Тест детектирования частиц на синтетическом изображении."""
        analyzer = PTVAnalyzer()
        analyzer.set_detection_config(min_area=4, max_area=100)

        # Создаем синтетическое бинарное изображение с двумя частицами
        test_image = np.zeros((100, 100), dtype=np.uint8)

        # Частица 1: квадрат 5x5 в позиции (10, 10)
        test_image[10:15, 10:15] = 255

        # Частица 2: квадрат 4x4 в позиции (50, 50)
        test_image[50:54, 50:54] = 255

        particles = analyzer.detect_particles(test_image)

        passed = (
            len(particles) == 2 and
            all(isinstance(p, Particle) for p in particles) and
            particles[0].area == 25 and  # 5x5
            particles[1].area == 16      # 4x4
        )
        self._log_result("test_detect_particles_synthetic", passed)
        return passed

    def test_detect_particles_area_filter(self) -> bool:
        """Тест фильтрации частиц по площади."""
        analyzer = PTVAnalyzer()
        analyzer.set_detection_config(min_area=10, max_area=50)

        test_image = np.zeros((100, 100), dtype=np.uint8)

        # Частица 1: слишком маленькая (4 пикселя)
        test_image[10:12, 10:12] = 255

        # Частица 2: в диапазоне (25 пикселей)
        test_image[30:35, 30:35] = 255

        # Частица 3: слишком большая (100 пикселей)
        test_image[60:70, 60:70] = 255

        particles = analyzer.detect_particles(test_image)

        passed = (
            len(particles) == 1 and
            particles[0].area == 25
        )
        self._log_result("test_detect_particles_area_filter", passed)
        return passed

    def test_detect_particles_real_image(self) -> bool:
        """Тест детектирования частиц на реальном изображении."""
        analyzer = PTVAnalyzer()
        analyzer.set_detection_config(min_area=4, max_area=150)

        cam1_path = self.test_data_path / "cam_1"
        png_files = list(cam1_path.glob("*.png"))

        if not png_files:
            self._log_result("test_detect_particles_real_image", False, "Нет PNG файлов для теста")
            return False

        img_array = analyzer._load_binary_image(png_files[0])

        if img_array is None:
            self._log_result("test_detect_particles_real_image", False, "Ошибка загрузки изображения")
            return False

        particles = analyzer.detect_particles(img_array)

        # Проверяем что детектирование работает (хотя бы что-то найдено или пустой список)
        passed = (
            isinstance(particles, list) and
            all(isinstance(p, Particle) for p in particles)
        )

        if passed and len(particles) > 0:
            # Дополнительная проверка корректности данных
            passed = all(
                p.area >= 4 and p.area <= 150 and
                p.center_x >= 0 and p.center_y >= 0 and
                p.diameter > 0
                for p in particles
            )

        self._log_result("test_detect_particles_real_image", passed)
        return passed

    def test_match_particles_synthetic(self) -> bool:
        """Тест сопоставления частиц на синтетических данных."""
        analyzer = PTVAnalyzer()
        analyzer.set_matching_config(max_distance=30.0, max_diameter_diff=2.0)

        # Частицы кадра a
        particles_a = [
            Particle(id=1, area=25, center_x=10.0, center_y=10.0, diameter=5.64),
            Particle(id=2, area=16, center_x=50.0, center_y=50.0, diameter=4.51),
        ]

        # Частицы кадра b (сдвинуты на небольшое расстояние)
        particles_b = [
            Particle(id=1, area=24, center_x=15.0, center_y=12.0, diameter=5.52),  # Сдвиг (5, 2)
            Particle(id=2, area=17, center_x=55.0, center_y=53.0, diameter=4.65),  # Сдвиг (5, 3)
        ]

        pairs = analyzer.match_particles(particles_a, particles_b)

        passed = (
            len(pairs) == 2 and
            all(isinstance(p, ParticlePair) for p in pairs)
        )

        if passed:
            # Проверка первой пары
            pair1 = pairs[0]
            passed = (
                abs(pair1.dx - 5.0) < 0.01 and
                abs(pair1.dy - 2.0) < 0.01
            )

        self._log_result("test_match_particles_synthetic", passed)
        return passed

    def test_match_particles_no_match(self) -> bool:
        """Тест сопоставления когда частицы слишком далеко."""
        analyzer = PTVAnalyzer()
        analyzer.set_matching_config(max_distance=10.0, max_diameter_diff=2.0)

        particles_a = [
            Particle(id=1, area=25, center_x=10.0, center_y=10.0, diameter=5.64),
        ]

        particles_b = [
            Particle(id=1, area=25, center_x=100.0, center_y=100.0, diameter=5.64),  # Слишком далеко
        ]

        pairs = analyzer.match_particles(particles_a, particles_b)

        passed = len(pairs) == 0
        self._log_result("test_match_particles_no_match", passed)
        return passed

    def test_match_particles_diameter_filter(self) -> bool:
        """Тест фильтрации по разнице диаметров."""
        analyzer = PTVAnalyzer()
        analyzer.set_matching_config(max_distance=50.0, max_diameter_diff=1.0)

        particles_a = [
            Particle(id=1, area=25, center_x=10.0, center_y=10.0, diameter=5.0),
        ]

        particles_b = [
            Particle(id=1, area=100, center_x=15.0, center_y=15.0, diameter=11.0),  # Большая разница диаметров
        ]

        pairs = analyzer.match_particles(particles_a, particles_b)

        passed = len(pairs) == 0
        self._log_result("test_match_particles_diameter_filter", passed)
        return passed

    def test_get_image_pairs(self) -> bool:
        """Тест получения списка пар изображений."""
        analyzer = PTVAnalyzer()

        cam1_path = self.test_data_path / "cam_1"

        if not cam1_path.exists():
            self._log_result("test_get_image_pairs", False, "Папка cam_1 не найдена")
            return False

        pairs = analyzer._get_image_pairs(cam1_path)

        # Проверяем что пары найдены и структура корректна
        passed = (
            isinstance(pairs, list) and
            (len(pairs) == 0 or all(
                isinstance(p, tuple) and len(p) == 2 and
                isinstance(p[0], Path) and isinstance(p[1], Path)
                for p in pairs
            ))
        )

        self._log_result("test_get_image_pairs", passed)
        return passed

    def test_save_particles_csv(self) -> bool:
        """Тест сохранения частиц в CSV."""
        analyzer = PTVAnalyzer()

        particles = [
            Particle(id=1, area=25, center_x=10.5, center_y=20.3, diameter=5.64),
            Particle(id=2, area=16, center_x=50.0, center_y=60.0, diameter=4.51),
        ]

        output_path = Path("/home/claude/test_particles.csv")

        result = analyzer._save_particles_csv(particles, output_path)

        passed = result is True and output_path.exists()

        if passed:
            # Проверяем содержимое CSV
            with open(output_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=';')
                rows = list(reader)

            passed = (
                len(rows) == 3 and  # Заголовок + 2 строки
                rows[0] == ['ID', 'Area', 'Center_X', 'Center_Y', 'Diameter'] and
                rows[1][0] == '1' and rows[1][1] == '25'
            )

            # Очистка
            output_path.unlink()

        self._log_result("test_save_particles_csv", passed)
        return passed

    def test_save_pairs_csv(self) -> bool:
        """Тест сохранения пар в CSV."""
        analyzer = PTVAnalyzer()

        pairs = [
            ParticlePair(id=1, x0=10.0, y0=20.0, dx=5.0, dy=3.0, length=5.83, diameter=5.64, area=25),
        ]

        output_path = Path("/home/claude/test_pairs.csv")

        result = analyzer._save_pairs_csv(pairs, output_path)

        passed = result is True and output_path.exists()

        if passed:
            # Проверяем содержимое CSV
            with open(output_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=';')
                rows = list(reader)

            passed = (
                len(rows) == 2 and  # Заголовок + 1 строка
                rows[0] == ['ID', 'X0', 'Y0', 'dx', 'dy', 'L', 'Diameter', 'Area']
            )

            # Очистка
            output_path.unlink()

        self._log_result("test_save_pairs_csv", passed)
        return passed

    def test_process_all(self) -> bool:
        """Тест полной обработки всех изображений."""
        analyzer = PTVAnalyzer()
        analyzer.set_input_folder(str(self.test_data_path))
        analyzer.set_detection_config(min_area=4, max_area=150)
        analyzer.set_matching_config(max_distance=30.0, max_diameter_diff=2.0)

        progress_messages = []

        def progress_callback(progress: PTVProgress):
            progress_messages.append(progress.message)

        analyzer.set_progress_callback(progress_callback)

        result = analyzer.process_all()

        output_folder = Path(result.output_folder)

        passed = (
            isinstance(result, PTVResult) and
            result.total_images_processed >= 0 and
            result.total_particles_detected >= 0 and
            result.total_pairs_matched >= 0
        )

        # Если обработка прошла успешно, проверяем структуру выходных папок
        if passed and output_folder.exists():
            cam1_output = output_folder / "cam_1"
            cam2_output = output_folder / "cam_2"
            cam1_pairs = output_folder / "cam_1_pairs"
            cam2_pairs = output_folder / "cam_2_pairs"

            # Проверяем что папки созданы (если были обработаны данные)
            if result.total_images_processed > 0:
                passed = (
                    cam1_output.exists() or cam2_output.exists()
                )

        if self.cleanup and output_folder.exists():
            shutil.rmtree(output_folder)
            print(f"    Очищена выходная папка: {output_folder}")

        self._log_result("test_process_all", passed)
        return passed

    def test_cancel_processing(self) -> bool:
        """Тест отмены обработки."""
        analyzer = PTVAnalyzer()
        analyzer.set_input_folder(str(self.test_data_path))

        def progress_callback(progress: PTVProgress):
            if progress.processed_files >= 1:
                analyzer.cancel_processing()

        analyzer.set_progress_callback(progress_callback)

        result = analyzer.process_all()

        output_folder = Path(result.output_folder)
        if self.cleanup and output_folder.exists():
            shutil.rmtree(output_folder)

        passed = True  # Тест успешен если не было исключений
        self._log_result("test_cancel_processing", passed)
        return passed

    def test_get_detection_preview(self) -> bool:
        """Тест предварительного просмотра детектирования."""
        analyzer = PTVAnalyzer()
        analyzer.set_detection_config(min_area=4, max_area=150)

        cam1_path = self.test_data_path / "cam_1"
        png_files = list(cam1_path.glob("*.png"))

        if not png_files:
            self._log_result("test_get_detection_preview", False, "Нет PNG файлов для теста")
            return False

        result = analyzer.get_detection_preview(png_files[0])

        passed = (
            result is not None and
            len(result) == 2 and
            isinstance(result[0], np.ndarray) and
            isinstance(result[1], list)
        )

        self._log_result("test_get_detection_preview", passed)
        return passed

    def test_get_matching_preview(self) -> bool:
        """Тест предварительного просмотра сопоставления."""
        analyzer = PTVAnalyzer()
        analyzer.set_detection_config(min_area=4, max_area=150)
        analyzer.set_matching_config(max_distance=30.0, max_diameter_diff=2.0)

        cam1_path = self.test_data_path / "cam_1"
        pairs = analyzer._get_image_pairs(cam1_path)

        if not pairs:
            self._log_result("test_get_matching_preview", False, "Нет пар изображений для теста")
            return False

        path_a, path_b = pairs[0]
        result = analyzer.get_matching_preview(path_a, path_b)

        passed = (
            result is not None and
            'image_a' in result and
            'image_b' in result and
            'particles_a' in result and
            'particles_b' in result and
            'matched_pairs' in result and
            'particles_a_count' in result and
            'matched_count' in result
        )

        self._log_result("test_get_matching_preview", passed)
        return passed

    def test_csv_output_format(self) -> bool:
        """Тест формата выходных CSV файлов."""
        analyzer = PTVAnalyzer()
        analyzer.set_input_folder(str(self.test_data_path))

        result = analyzer.process_all()

        output_folder = Path(result.output_folder)

        passed = True

        if output_folder.exists() and result.total_images_processed > 0:
            # Проверяем CSV файлы с частицами
            cam1_output = output_folder / "cam_1"
            if cam1_output.exists():
                csv_files = list(cam1_output.glob("*.csv"))
                if csv_files:
                    with open(csv_files[0], 'r', encoding='utf-8') as f:
                        reader = csv.reader(f, delimiter=';')
                        header = next(reader)
                        passed = header == ['ID', 'Area', 'Center_X', 'Center_Y', 'Diameter']

            # Проверяем CSV файлы с парами
            cam1_pairs = output_folder / "cam_1_pairs"
            if cam1_pairs.exists() and passed:
                csv_files = list(cam1_pairs.glob("*.csv"))
                if csv_files:
                    with open(csv_files[0], 'r', encoding='utf-8') as f:
                        reader = csv.reader(f, delimiter=';')
                        header = next(reader)
                        passed = header == ['ID', 'X0', 'Y0', 'dx', 'dy', 'L', 'Diameter', 'Area']

        if self.cleanup and output_folder.exists():
            shutil.rmtree(output_folder)

        self._log_result("test_csv_output_format", passed)
        return passed

    def test_summary_pairs_csv(self) -> bool:
        """Тест создания суммарных CSV файлов с парами."""
        analyzer = PTVAnalyzer()
        analyzer.set_input_folder(str(self.test_data_path))

        result = analyzer.process_all()

        output_folder = Path(result.output_folder)

        passed = False

        if output_folder.exists() and result.total_images_processed > 0:
            # Проверяем наличие суммарных файлов
            cam1_summary = output_folder / "cam_1_pairs_sum.csv"
            cam2_summary = output_folder / "cam_2_pairs_sum.csv"

            # Проверяем что хотя бы один суммарный файл создан
            if cam1_summary.exists() or cam2_summary.exists():
                passed = True

                # Проверяем формат суммарного файла cam_1
                if cam1_summary.exists():
                    with open(cam1_summary, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f, delimiter=';')
                        header = next(reader)

                        # Проверка заголовка
                        if header != ['ID', 'X0', 'Y0', 'dx', 'dy', 'L', 'Diameter', 'Area']:
                            passed = False

                        # Проверка что есть данные и ID начинается с 1
                        rows = list(reader)
                        if len(rows) > 0:
                            if rows[0][0] != '1':
                                passed = False
                        else:
                            # Если нет строк, это тоже валидный случай (нет пар)
                            pass

                # Проверяем формат суммарного файла cam_2
                if cam2_summary.exists() and passed:
                    with open(cam2_summary, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f, delimiter=';')
                        header = next(reader)

                        if header != ['ID', 'X0', 'Y0', 'dx', 'dy', 'L', 'Diameter', 'Area']:
                            passed = False

        if self.cleanup and output_folder.exists():
            shutil.rmtree(output_folder)

        self._log_result("test_summary_pairs_csv", passed)
        return passed

    def test_particle_diameter_calculation(self) -> bool:
        """Тест корректности вычисления диаметра частицы."""
        analyzer = PTVAnalyzer()
        analyzer.set_detection_config(min_area=1, max_area=1000)

        # Создаем круглую частицу с известной площадью
        test_image = np.zeros((100, 100), dtype=np.uint8)

        # Квадрат 10x10 = площадь 100
        test_image[45:55, 45:55] = 255

        particles = analyzer.detect_particles(test_image)

        if len(particles) != 1:
            self._log_result("test_particle_diameter_calculation", False, "Не найдена одна частица")
            return False

        particle = particles[0]

        # Эквивалентный диаметр: D = 2 * sqrt(S / pi) = 2 * sqrt(100 / pi) ≈ 11.28
        expected_diameter = 2 * np.sqrt(100 / np.pi)

        passed = abs(particle.diameter - expected_diameter) < 0.01
        self._log_result("test_particle_diameter_calculation", passed)
        return passed

    def test_4_connectivity(self) -> bool:
        """Тест 4-связности при детектировании."""
        analyzer = PTVAnalyzer()
        analyzer.set_detection_config(min_area=1, max_area=100)

        # Создаем изображение с двумя частицами, соединенными диагонально
        test_image = np.zeros((20, 20), dtype=np.uint8)

        # Частица 1
        test_image[5, 5] = 255
        test_image[5, 6] = 255
        test_image[6, 5] = 255

        # Частица 2 (диагонально от первой - не должна объединяться при 4-связности)
        test_image[7, 7] = 255
        test_image[7, 8] = 255
        test_image[8, 7] = 255

        particles = analyzer.detect_particles(test_image)

        # При 4-связности должно быть 2 отдельные частицы
        passed = len(particles) == 2
        self._log_result("test_4_connectivity", passed)
        return passed

    def run_all_tests(self) -> dict:
        """
        Запуск всех тестов.

        Returns:
            dict: Результаты тестирования
        """
        print("\n" + "=" * 60)
        print("ЗАПУСК ТЕСТОВ PTVAnalyzer")
        print("=" * 60)

        if not self.setup():
            return {'success': False, 'message': 'Ошибка подготовки тестов'}

        print("\n--- Тесты инициализации ---")
        self.test_set_input_folder_valid()
        self.test_set_input_folder_invalid()
        self.test_output_folder_naming()
        self.test_set_detection_config_valid()
        self.test_set_detection_config_invalid()
        self.test_set_matching_config_valid()
        self.test_set_matching_config_invalid()

        print("\n--- Тесты загрузки изображений ---")
        self.test_load_binary_image()
        self.test_get_image_pairs()

        print("\n--- Тесты детектирования частиц ---")
        self.test_detect_particles_synthetic()
        self.test_detect_particles_area_filter()
        self.test_detect_particles_real_image()
        self.test_particle_diameter_calculation()
        self.test_4_connectivity()

        print("\n--- Тесты сопоставления частиц ---")
        self.test_match_particles_synthetic()
        self.test_match_particles_no_match()
        self.test_match_particles_diameter_filter()

        print("\n--- Тесты сохранения CSV ---")
        self.test_save_particles_csv()
        self.test_save_pairs_csv()
        self.test_csv_output_format()
        self.test_summary_pairs_csv()

        print("\n--- Тесты полной обработки ---")
        self.test_process_all()
        self.test_cancel_processing()

        print("\n--- Тесты предварительного просмотра ---")
        self.test_get_detection_preview()
        self.test_get_matching_preview()

        self.teardown()

        passed_count = sum(1 for _, passed, _ in self.results if passed)
        total_count = len(self.results)

        print("\n" + "=" * 60)
        print(f"РЕЗУЛЬТАТЫ: {passed_count}/{total_count} тестов пройдено")
        print("=" * 60)

        if passed_count < total_count:
            print("\nНеудачные тесты:")
            for name, passed, message in self.results:
                if not passed:
                    print(f"  - {name}: {message}")

        return {
            'success': passed_count == total_count,
            'passed': passed_count,
            'total': total_count,
            'results': self.results
        }


def run_tests(test_data_path: str, cleanup: bool = True) -> dict:
    """
    Запуск тестов PTVAnalyzer.

    Args:
        test_data_path: Путь к папке с тестовыми данными (binary_filter_XXXX)
        cleanup: Удалять ли выходные папки после тестов (по умолчанию True)

    Returns:
        dict: Результаты тестирования
    """
    tester = TestPTVAnalyzer(test_data_path, cleanup=cleanup)
    return tester.run_all_tests()


if __name__ == "__main__":
    # Путь к тестовым данным - папка binary_filter_XXXX с подпапками cam_1 и cam_2
    # Измените путь на актуальный путь к вашим тестовым данным
    TEST_DATA_PATH = r"C:\Users\evils\PycharmProjects\ParticleAnalysisFE\tests\test_data_cam_sorted\binary_filter_10000"

    # cleanup=True - удалять выходные папки после тестов
    # cleanup=False - сохранять выходные папки для проверки результатов
    CLEANUP = False

    results = run_tests(TEST_DATA_PATH, cleanup=CLEANUP)

    if results['success']:
        print("\n✓ Все тесты пройдены успешно!")
    else:
        print(f"\n✗ Пройдено {results['passed']} из {results['total']} тестов")
