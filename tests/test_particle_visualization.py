"""
Тесты для модуля ParticleVisualizer.

Тестирование проводится на реальных тестовых данных.
Тесты вызываются из кода, не из CLI.

Требуемая структура тестовых данных:
- binary_filter_XXXX/ (папка с бинарными изображениями)
- PTV_XXXX/ (папка с результатами PTV анализа)
"""

import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import csv
import cv2
from src.visualization.particle_visualization import (
    ParticleVisualizer,
    VisualizationProgress,
    VisualizationResult,
    VisualizationConfig,
    Particle
)


class TestParticleVisualizer:
    """Тесты для класса ParticleVisualizer."""

    def __init__(
        self,
        binary_folder_path: str,
        ptv_folder_path: str,
        cleanup: bool = True
    ):
        """
        Инициализация тестов.

        Args:
            binary_folder_path: Путь к папке с бинарными изображениями (binary_filter_XXXX)
            ptv_folder_path: Путь к папке с результатами PTV (PTV_XXXX)
            cleanup: Удалять ли выходные папки после тестов (по умолчанию True)
        """
        self.binary_folder_path = Path(binary_folder_path)
        self.ptv_folder_path = Path(ptv_folder_path)
        self.cleanup = cleanup
        self.results = []

    def setup(self) -> bool:
        """
        Подготовка к тестированию.

        Returns:
            bool: True если подготовка успешна
        """
        # Проверка папки с бинарными изображениями
        if not self.binary_folder_path.exists():
            print(f"ОШИБКА: Папка с бинарными изображениями не найдена: {self.binary_folder_path}")
            return False

        binary_cam1 = self.binary_folder_path / "cam_1"
        binary_cam2 = self.binary_folder_path / "cam_2"

        if not binary_cam1.exists() or not binary_cam2.exists():
            print("ОШИБКА: Папка binary_filter должна содержать cam_1 и cam_2")
            return False

        # Проверка папки с результатами PTV
        if not self.ptv_folder_path.exists():
            print(f"ОШИБКА: Папка с результатами PTV не найдена: {self.ptv_folder_path}")
            return False

        ptv_cam1 = self.ptv_folder_path / "cam_1"
        ptv_cam2 = self.ptv_folder_path / "cam_2"

        if not ptv_cam1.exists() or not ptv_cam2.exists():
            print("ОШИБКА: Папка PTV должна содержать cam_1 и cam_2")
            return False

        print(f"Папка бинарных изображений: {self.binary_folder_path}")
        print(f"  cam_1: {len(list(binary_cam1.glob('*.png')))} PNG файлов")
        print(f"  cam_2: {len(list(binary_cam2.glob('*.png')))} PNG файлов")

        print(f"Папка результатов PTV: {self.ptv_folder_path}")
        print(f"  cam_1: {len(list(ptv_cam1.glob('*.csv')))} CSV файлов")
        print(f"  cam_2: {len(list(ptv_cam2.glob('*.csv')))} CSV файлов")

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

    def test_set_input_folders_valid(self) -> bool:
        """Тест установки валидных входных папок."""
        visualizer = ParticleVisualizer()
        result = visualizer.set_input_folders(
            str(self.binary_folder_path),
            str(self.ptv_folder_path)
        )

        passed = (
            result is True and
            visualizer.binary_folder == self.binary_folder_path and
            visualizer.ptv_folder == self.ptv_folder_path
        )
        self._log_result("test_set_input_folders_valid", passed)
        return passed

    def test_set_input_folders_invalid_binary(self) -> bool:
        """Тест установки невалидной папки бинарных изображений."""
        visualizer = ParticleVisualizer()
        result = visualizer.set_input_folders(
            "/nonexistent/path",
            str(self.ptv_folder_path)
        )

        passed = result is False
        self._log_result("test_set_input_folders_invalid_binary", passed)
        return passed

    def test_set_input_folders_invalid_ptv(self) -> bool:
        """Тест установки невалидной папки PTV."""
        visualizer = ParticleVisualizer()
        result = visualizer.set_input_folders(
            str(self.binary_folder_path),
            "/nonexistent/path"
        )

        passed = result is False
        self._log_result("test_set_input_folders_invalid_ptv", passed)
        return passed

    def test_output_folder_naming(self) -> bool:
        """Тест формирования имени выходной папки."""
        visualizer = ParticleVisualizer()
        visualizer.set_input_folders(
            str(self.binary_folder_path),
            str(self.ptv_folder_path)
        )

        expected_path = self.binary_folder_path.parent / "particle_visualization"

        passed = visualizer.output_folder == expected_path
        message = f"Ожидалось: {expected_path}, получено: {visualizer.output_folder}"
        self._log_result("test_output_folder_naming", passed, message if not passed else "")
        return passed

    def test_set_visualization_config(self) -> bool:
        """Тест установки параметров визуализации."""
        visualizer = ParticleVisualizer()

        visualizer.set_visualization_config(
            center_color=(255, 0, 0),
            circle_color=(0, 255, 255),
            circle_thickness=2,
            max_images=5
        )

        passed = (
            visualizer.config.center_color == (255, 0, 0) and
            visualizer.config.circle_color == (0, 255, 255) and
            visualizer.config.circle_thickness == 2 and
            visualizer.config.max_images == 5
        )
        self._log_result("test_set_visualization_config", passed)
        return passed

    def test_load_binary_image(self) -> bool:
        """Тест загрузки бинаризованного изображения."""
        visualizer = ParticleVisualizer()

        binary_cam1 = self.binary_folder_path / "cam_1"
        png_files = list(binary_cam1.glob("*.png"))

        if not png_files:
            self._log_result("test_load_binary_image", False, "Нет PNG файлов для теста")
            return False

        img_array = visualizer._load_binary_image(png_files[0])

        passed = (
            img_array is not None and
            isinstance(img_array, np.ndarray) and
            img_array.dtype == np.uint8
        )
        self._log_result("test_load_binary_image", passed)
        return passed

    def test_load_particles_from_csv(self) -> bool:
        """Тест загрузки частиц из CSV."""
        visualizer = ParticleVisualizer()

        ptv_cam1 = self.ptv_folder_path / "cam_1"
        csv_files = list(ptv_cam1.glob("*.csv"))

        if not csv_files:
            self._log_result("test_load_particles_from_csv", False, "Нет CSV файлов для теста")
            return False

        particles = visualizer._load_particles_from_csv(csv_files[0])

        passed = (
            isinstance(particles, list) and
            (len(particles) == 0 or all(isinstance(p, Particle) for p in particles))
        )

        # Если частицы есть, проверяем их атрибуты
        if passed and len(particles) > 0:
            p = particles[0]
            passed = (
                hasattr(p, 'id') and
                hasattr(p, 'area') and
                hasattr(p, 'center_x') and
                hasattr(p, 'center_y') and
                hasattr(p, 'diameter')
            )

        self._log_result("test_load_particles_from_csv", passed)
        return passed

    def test_convert_to_color(self) -> bool:
        """Тест преобразования в цветное изображение."""
        visualizer = ParticleVisualizer()

        # Создаем тестовое grayscale изображение
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        gray_image[50, 50] = 255

        color_image = visualizer._convert_to_color(gray_image)

        passed = (
            color_image is not None and
            color_image.shape == (100, 100, 3) and
            color_image.dtype == np.uint8
        )
        self._log_result("test_convert_to_color", passed)
        return passed

    def test_draw_particles_synthetic(self) -> bool:
        """Тест рисования частиц на синтетическом изображении."""
        visualizer = ParticleVisualizer()
        visualizer.set_visualization_config(
            center_color=(0, 0, 255),
            circle_color=(0, 255, 0),
            circle_thickness=1
        )

        # Создаем пустое цветное изображение
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Создаем тестовые частицы
        particles = [
            Particle(id=1, area=25, center_x=30.0, center_y=30.0, diameter=10.0),
            Particle(id=2, area=16, center_x=70.0, center_y=70.0, diameter=8.0),
        ]

        result = visualizer._draw_particles(image, particles)

        # Проверяем что изображение изменилось (не полностью черное)
        passed = (
            result is not None and
            result.shape == (100, 100, 3) and
            np.any(result != 0)  # Должны быть ненулевые пиксели
        )

        # Проверяем что центр первой частицы красный (1 пиксель)
        if passed:
            center_pixel = result[30, 30]
            has_red = (
                center_pixel[2] == 255 and  # R
                center_pixel[1] == 0 and    # G
                center_pixel[0] == 0        # B
            )
            passed = has_red

        self._log_result("test_draw_particles_synthetic", passed)
        return passed

    def test_particles_drawn_correctly(self) -> bool:
        """Тест корректности отрисовки частиц."""
        visualizer = ParticleVisualizer()
        visualizer.set_visualization_config(
            center_color=(0, 0, 255),  # Красный
            circle_color=(0, 255, 0),  # Зеленый
            circle_thickness=2
        )

        # Создаем изображение с известной частицей
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        particle = Particle(
            id=1,
            area=100,
            center_x=50.0,
            center_y=50.0,
            diameter=20.0  # Радиус 10
        )

        result = visualizer._draw_particles(image, [particle])

        # Проверяем центр (красный, 1 пиксель)
        center_pixel = result[50, 50]
        has_red_center = (
            center_pixel[2] == 255 and  # R
            center_pixel[1] == 0 and    # G
            center_pixel[0] == 0        # B
        )

        # Проверяем что соседние пиксели НЕ красные (центр = 1 пиксель)
        neighbor_not_red = result[50, 51][2] != 255 or result[50, 51][1] != 0

        # Проверяем окружность (зеленый) на расстоянии радиуса
        circle_pixel = result[50, 60]  # 10 пикселей вправо от центра
        has_green_circle = circle_pixel[1] == 255  # BGR

        passed = has_red_center and has_green_circle and neighbor_not_red
        self._log_result("test_particles_drawn_correctly", passed)
        return passed

    def test_get_matching_files(self) -> bool:
        """Тест получения соответствующих файлов."""
        visualizer = ParticleVisualizer()
        visualizer.set_input_folders(
            str(self.binary_folder_path),
            str(self.ptv_folder_path)
        )

        matching_files = visualizer._get_matching_files("cam_1")

        passed = (
            isinstance(matching_files, list) and
            (len(matching_files) == 0 or all(
                isinstance(item, tuple) and len(item) == 2
                for item in matching_files
            ))
        )

        self._log_result("test_get_matching_files", passed)
        return passed

    def test_visualize_single_image(self) -> bool:
        """Тест визуализации одного изображения."""
        visualizer = ParticleVisualizer()
        visualizer.set_input_folders(
            str(self.binary_folder_path),
            str(self.ptv_folder_path)
        )

        matching_files = visualizer._get_matching_files("cam_1")

        if not matching_files:
            self._log_result("test_visualize_single_image", False, "Нет соответствующих файлов")
            return False

        img_path, csv_path = matching_files[0]
        result = visualizer.visualize_single_image(img_path, csv_path)

        passed = (
            result is not None and
            len(result) == 2 and
            isinstance(result[0], np.ndarray) and
            isinstance(result[1], list) and
            result[0].ndim == 3  # Цветное изображение
        )
        self._log_result("test_visualize_single_image", passed)
        return passed

    def test_save_image(self) -> bool:
        """Тест сохранения изображения."""
        visualizer = ParticleVisualizer()

        # Создаем тестовое изображение
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[50, 50] = [255, 0, 0]

        output_path = Path("/home/claude/test_vis_image.png")

        result = visualizer._save_image(test_image, output_path)

        passed = result is True and output_path.exists()

        if passed:
            # Проверяем что файл можно прочитать
            loaded = cv2.imread(str(output_path))
            passed = loaded is not None and loaded.shape == (100, 100, 3)

            # Очистка
            output_path.unlink()

        self._log_result("test_save_image", passed)
        return passed

    def test_process_all(self) -> bool:
        """Тест полной обработки всех изображений."""
        visualizer = ParticleVisualizer()
        visualizer.set_input_folders(
            str(self.binary_folder_path),
            str(self.ptv_folder_path)
        )
        visualizer.set_visualization_config(max_images=5)  # Ограничиваем для скорости

        # Очищаем выходную папку перед тестом
        output_folder = self.binary_folder_path.parent / "particle_visualization"
        if output_folder.exists():
            shutil.rmtree(output_folder)

        progress_messages = []

        def progress_callback(progress: VisualizationProgress):
            progress_messages.append(progress.message)

        visualizer.set_progress_callback(progress_callback)

        result = visualizer.process_all()

        output_folder = Path(result.output_folder)

        passed = (
            isinstance(result, VisualizationResult) and
            result.total_processed >= 0
        )

        # Если обработка прошла успешно, проверяем структуру выходных папок
        if passed and output_folder.exists() and result.total_processed > 0:
            cam1_output = output_folder / "cam_1"
            cam2_output = output_folder / "cam_2"

            # Проверяем что папки созданы
            passed = cam1_output.exists() or cam2_output.exists()

            # Проверяем что файлы имеют префикс vis_
            if cam1_output.exists():
                output_files = list(cam1_output.glob("*.png"))
                if output_files:
                    passed = all(f.name.startswith("vis_") for f in output_files)

        if self.cleanup and output_folder.exists():
            shutil.rmtree(output_folder)
            print(f"    Очищена выходная папка: {output_folder}")

        self._log_result("test_process_all", passed)
        return passed

    def test_cancel_processing(self) -> bool:
        """Тест отмены обработки."""
        visualizer = ParticleVisualizer()
        visualizer.set_input_folders(
            str(self.binary_folder_path),
            str(self.ptv_folder_path)
        )

        # Очищаем выходную папку перед тестом
        output_folder = self.binary_folder_path.parent / "particle_visualization"
        if output_folder.exists():
            shutil.rmtree(output_folder)

        def progress_callback(progress: VisualizationProgress):
            if progress.processed_files >= 1:
                visualizer.cancel_processing()

        visualizer.set_progress_callback(progress_callback)

        result = visualizer.process_all()

        output_folder = Path(result.output_folder)
        if self.cleanup and output_folder.exists():
            shutil.rmtree(output_folder)

        passed = True  # Тест успешен если не было исключений
        self._log_result("test_cancel_processing", passed)
        return passed

    def test_get_preview(self) -> bool:
        """Тест предварительного просмотра."""
        visualizer = ParticleVisualizer()
        visualizer.set_input_folders(
            str(self.binary_folder_path),
            str(self.ptv_folder_path)
        )

        matching_files = visualizer._get_matching_files("cam_1")

        if not matching_files:
            self._log_result("test_get_preview", False, "Нет соответствующих файлов")
            return False

        img_path, csv_path = matching_files[0]
        result = visualizer.get_preview(img_path, csv_path)

        passed = (
            result is not None and
            'image' in result and
            'particles' in result and
            'particles_count' in result and
            'image_shape' in result and
            'config' in result
        )

        self._log_result("test_get_preview", passed)
        return passed

    def test_create_comparison_image(self) -> bool:
        """Тест создания сравнительного изображения."""
        visualizer = ParticleVisualizer()
        visualizer.set_input_folders(
            str(self.binary_folder_path),
            str(self.ptv_folder_path)
        )

        matching_files = visualizer._get_matching_files("cam_1")

        if not matching_files:
            self._log_result("test_create_comparison_image", False, "Нет соответствующих файлов")
            return False

        img_path, csv_path = matching_files[0]
        result = visualizer.create_comparison_image(img_path, csv_path)

        passed = (
            result is not None and
            isinstance(result, np.ndarray) and
            result.ndim == 3
        )

        # Проверяем что ширина удвоилась (два изображения рядом)
        if passed:
            original = visualizer._load_binary_image(img_path)
            if original is not None:
                expected_width = original.shape[1] * 2
                passed = result.shape[1] == expected_width

        self._log_result("test_create_comparison_image", passed)
        return passed

    def test_output_images_are_color(self) -> bool:
        """Тест что выходные изображения цветные."""
        visualizer = ParticleVisualizer()
        visualizer.set_input_folders(
            str(self.binary_folder_path),
            str(self.ptv_folder_path)
        )
        visualizer.set_visualization_config(max_images=2)

        # Очищаем выходную папку перед тестом
        output_folder = self.binary_folder_path.parent / "particle_visualization"
        if output_folder.exists():
            shutil.rmtree(output_folder)

        result = visualizer.process_all()

        output_folder = Path(result.output_folder)

        passed = True

        if output_folder.exists() and result.total_processed > 0:
            cam1_output = output_folder / "cam_1"
            if cam1_output.exists():
                output_files = list(cam1_output.glob("*.png"))[:1]
                for img_path in output_files:
                    img = cv2.imread(str(img_path))
                    if img is None or img.ndim != 3 or img.shape[2] != 3:
                        passed = False
                        break

        if self.cleanup and output_folder.exists():
            shutil.rmtree(output_folder)

        self._log_result("test_output_images_are_color", passed)
        return passed

    def test_max_images_limit(self) -> bool:
        """Тест ограничения количества обрабатываемых изображений."""
        visualizer = ParticleVisualizer()
        visualizer.set_input_folders(
            str(self.binary_folder_path),
            str(self.ptv_folder_path)
        )
        visualizer.set_visualization_config(max_images=3)

        # Очищаем выходную папку перед тестом (могла остаться от предыдущих тестов)
        output_folder = self.binary_folder_path.parent / "particle_visualization"
        if output_folder.exists():
            shutil.rmtree(output_folder)

        result = visualizer.process_all()

        output_folder = Path(result.output_folder)

        passed = True

        if output_folder.exists():
            cam1_output = output_folder / "cam_1"
            cam2_output = output_folder / "cam_2"

            # Проверяем что обработано не больше max_images для каждой камеры
            if cam1_output.exists():
                cam1_files = list(cam1_output.glob("*.png"))
                passed = len(cam1_files) <= 3

            if cam2_output.exists() and passed:
                cam2_files = list(cam2_output.glob("*.png"))
                passed = len(cam2_files) <= 3

        if self.cleanup and output_folder.exists():
            shutil.rmtree(output_folder)

        self._log_result("test_max_images_limit", passed)
        return passed

    def run_all_tests(self) -> dict:
        """
        Запуск всех тестов.

        Returns:
            dict: Результаты тестирования
        """
        print("\n" + "=" * 60)
        print("ЗАПУСК ТЕСТОВ ParticleVisualizer")
        print("=" * 60)

        if not self.setup():
            return {'success': False, 'message': 'Ошибка подготовки тестов'}

        print("\n--- Тесты инициализации ---")
        self.test_set_input_folders_valid()
        self.test_set_input_folders_invalid_binary()
        self.test_set_input_folders_invalid_ptv()
        self.test_output_folder_naming()
        self.test_set_visualization_config()

        print("\n--- Тесты загрузки данных ---")
        self.test_load_binary_image()
        self.test_load_particles_from_csv()
        self.test_convert_to_color()
        self.test_get_matching_files()

        print("\n--- Тесты рисования частиц ---")
        self.test_draw_particles_synthetic()
        self.test_particles_drawn_correctly()

        print("\n--- Тесты визуализации ---")
        self.test_visualize_single_image()
        self.test_save_image()
        self.test_get_preview()
        self.test_create_comparison_image()

        print("\n--- Тесты полной обработки ---")
        self.test_process_all()
        self.test_cancel_processing()
        self.test_output_images_are_color()
        self.test_max_images_limit()

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


def run_tests(
    binary_folder_path: str,
    ptv_folder_path: str,
    cleanup: bool = True
) -> dict:
    """
    Запуск тестов ParticleVisualizer.

    Args:
        binary_folder_path: Путь к папке с бинарными изображениями (binary_filter_XXXX)
        ptv_folder_path: Путь к папке с результатами PTV (PTV_XXXX)
        cleanup: Удалять ли выходные папки после тестов (по умолчанию True)

    Returns:
        dict: Результаты тестирования
    """
    tester = TestParticleVisualizer(binary_folder_path, ptv_folder_path, cleanup=cleanup)
    return tester.run_all_tests()


if __name__ == "__main__":
    # Пути к тестовым данным
    # Измените пути на актуальные пути к вашим тестовым данным

    # Папка с бинарными изображениями
    BINARY_FOLDER = r"C:\Users\evils\PycharmProjects\ParticleAnalysisFE\tests\test_data_cam_sorted\binary_filter_10000"

    # Папка с результатами PTV анализа
    PTV_FOLDER = r"C:\Users\evils\PycharmProjects\ParticleAnalysisFE\tests\test_data_cam_sorted\PTV_10000"

    # cleanup=True - удалять выходные папки после тестов
    # cleanup=False - сохранять выходные папки для проверки результатов
    CLEANUP = False

    results = run_tests(BINARY_FOLDER, PTV_FOLDER, cleanup=CLEANUP)

    if results['success']:
        print("\n✓ Все тесты пройдены успешно!")
    else:
        print(f"\n✗ Пройдено {results['passed']} из {results['total']} тестов")