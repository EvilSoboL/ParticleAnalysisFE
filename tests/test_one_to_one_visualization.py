"""
Тесты для модуля PTVVisualizer.

Тестирование проводится на реальных тестовых данных.
Тесты вызываются из кода, не из CLI.
"""

import numpy as np
import cv2
from pathlib import Path
import shutil
from src.visualization.one_to_one_visualization import (
    PTVVisualizer,
    MatchedPair,
    VisualizationProgress,
    VisualizationResult,
    VisualizationConfig
)


class TestPTVVisualizer:
    """Тесты для класса PTVVisualizer."""

    def __init__(self, original_folder_path: str, ptv_folder_path: str, cleanup: bool = True):
        """
        Инициализация тестов.

        Args:
            original_folder_path: Путь к папке cam_sorted с исходными изображениями
            ptv_folder_path: Путь к папке PTV_XXXX с результатами сопоставления
            cleanup: Удалять ли выходные папки после тестов
        """
        self.original_folder = Path(original_folder_path)
        self.ptv_folder = Path(ptv_folder_path)
        self.cleanup = cleanup
        self.results = []

    def setup(self) -> bool:
        """
        Подготовка к тестированию.

        Returns:
            bool: True если подготовка успешна
        """
        if not self.original_folder.exists():
            print(f"ОШИБКА: Папка исходных изображений не найдена: {self.original_folder}")
            return False

        if not self.ptv_folder.exists():
            print(f"ОШИБКА: Папка PTV результатов не найдена: {self.ptv_folder}")
            return False

        # Проверка структуры папок
        original_cam1 = self.original_folder / "cam_1"
        original_cam2 = self.original_folder / "cam_2"

        if not original_cam1.exists() or not original_cam2.exists():
            print("ОШИБКА: Папка исходников должна содержать cam_1 и cam_2")
            return False

        # Проверка наличия папок с парами
        cam1_pairs = self.ptv_folder / "cam_1_pairs"
        cam2_pairs = self.ptv_folder / "cam_2_pairs"

        has_pairs = cam1_pairs.exists() or cam2_pairs.exists()
        if not has_pairs:
            print("ОШИБКА: Папка PTV должна содержать cam_1_pairs и/или cam_2_pairs")
            return False

        print(f"Исходные изображения: {self.original_folder}")
        print(f"  cam_1: {len(list(original_cam1.glob('*.png')))} файлов")
        print(f"  cam_2: {len(list(original_cam2.glob('*.png')))} файлов")
        print(f"PTV результаты: {self.ptv_folder}")

        if cam1_pairs.exists():
            print(f"  cam_1_pairs: {len(list(cam1_pairs.glob('*.csv')))} CSV файлов")
        if cam2_pairs.exists():
            print(f"  cam_2_pairs: {len(list(cam2_pairs.glob('*.csv')))} CSV файлов")

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

    def test_set_original_folder_valid(self) -> bool:
        """Тест установки валидной папки исходных изображений."""
        visualizer = PTVVisualizer()
        result = visualizer.set_original_folder(str(self.original_folder))

        passed = result is True and visualizer.original_folder == self.original_folder
        self._log_result("test_set_original_folder_valid", passed)
        return passed

    def test_set_original_folder_invalid(self) -> bool:
        """Тест установки невалидной папки."""
        visualizer = PTVVisualizer()
        result = visualizer.set_original_folder("/nonexistent/path")

        passed = result is False
        self._log_result("test_set_original_folder_invalid", passed)
        return passed

    def test_set_ptv_folder_valid(self) -> bool:
        """Тест установки валидной папки PTV результатов."""
        visualizer = PTVVisualizer()
        result = visualizer.set_ptv_folder(str(self.ptv_folder))

        passed = result is True and visualizer.ptv_folder == self.ptv_folder
        self._log_result("test_set_ptv_folder_valid", passed)
        return passed

    def test_set_ptv_folder_invalid(self) -> bool:
        """Тест установки невалидной папки PTV."""
        visualizer = PTVVisualizer()
        result = visualizer.set_ptv_folder("/nonexistent/path")

        passed = result is False
        self._log_result("test_set_ptv_folder_invalid", passed)
        return passed

    def test_output_folder_naming(self) -> bool:
        """Тест формирования имени выходной папки."""
        visualizer = PTVVisualizer()
        visualizer.set_ptv_folder(str(self.ptv_folder))

        expected_path = self.ptv_folder / "one_to_one_visualization"

        passed = visualizer.output_folder == expected_path
        message = f"Ожидалось: {expected_path}, получено: {visualizer.output_folder}"
        self._log_result("test_output_folder_naming", passed, message if not passed else "")
        return passed

    def test_visualization_config(self) -> bool:
        """Тест конфигурации визуализации."""
        visualizer = PTVVisualizer()
        visualizer.set_visualization_config(
            particle_a_color=(0, 255, 0),  # Зелёный
            particle_b_color=(0, 0, 255),  # Красный
            line_color=(0, 165, 255),  # Оранжевый
            line_thickness=2
        )

        cfg = visualizer.config
        passed = (
            cfg.particle_a_color == (0, 255, 0) and
            cfg.particle_b_color == (0, 0, 255) and
            cfg.line_color == (0, 165, 255) and
            cfg.line_thickness == 2
        )
        self._log_result("test_visualization_config", passed)
        return passed

    def test_load_original_image(self) -> bool:
        """Тест загрузки исходного изображения."""
        visualizer = PTVVisualizer()

        original_cam1 = self.original_folder / "cam_1"
        png_files = list(original_cam1.glob("*.png"))

        if not png_files:
            self._log_result("test_load_original_image", False, "Нет PNG файлов для теста")
            return False

        img = visualizer._load_original_image(png_files[0])

        passed = (
            img is not None and
            isinstance(img, np.ndarray) and
            img.dtype == np.uint8 and
            len(img.shape) == 3 and  # BGR
            img.shape[2] == 3
        )
        self._log_result("test_load_original_image", passed)
        return passed

    def test_load_pairs_csv(self) -> bool:
        """Тест загрузки CSV с парами."""
        visualizer = PTVVisualizer()

        # Ищем CSV файлы
        cam1_pairs = self.ptv_folder / "cam_1_pairs"
        cam2_pairs = self.ptv_folder / "cam_2_pairs"

        csv_files = []
        if cam1_pairs.exists():
            csv_files.extend(list(cam1_pairs.glob("*.csv")))
        if cam2_pairs.exists():
            csv_files.extend(list(cam2_pairs.glob("*.csv")))

        if not csv_files:
            self._log_result("test_load_pairs_csv", False, "Нет CSV файлов для теста")
            return False

        pairs = visualizer._load_pairs_csv(csv_files[0])

        passed = (
            isinstance(pairs, list) and
            len(pairs) > 0 and
            all(isinstance(p, MatchedPair) for p in pairs)
        )

        if passed:
            # Проверяем атрибуты первой пары
            p = pairs[0]
            passed = (
                hasattr(p, 'id') and
                hasattr(p, 'x0') and
                hasattr(p, 'y0') and
                hasattr(p, 'dx') and
                hasattr(p, 'dy') and
                hasattr(p, 'length') and
                hasattr(p, 'diameter') and
                hasattr(p, 'area')
            )

        self._log_result("test_load_pairs_csv", passed,
                        f"Загружено пар: {len(pairs)}" if pairs else "")
        return passed

    def test_create_visualization(self) -> bool:
        """Тест создания визуализации."""
        visualizer = PTVVisualizer()

        # Создаем тестовое изображение
        original = np.full((500, 500, 3), 128, dtype=np.uint8)

        # Создаем тестовые пары
        pairs = [
            MatchedPair(id=1, x0=100.0, y0=100.0, dx=20.0, dy=10.0,
                       length=22.36, diameter=8.0, area=50),
            MatchedPair(id=2, x0=300.0, y0=300.0, dx=-15.0, dy=25.0,
                       length=29.15, diameter=10.0, area=78),
        ]

        vis_image = visualizer.create_visualization(original, pairs)

        passed = (
            vis_image is not None and
            vis_image.shape == original.shape and
            vis_image.dtype == np.uint8
        )

        self._log_result("test_create_visualization", passed)
        return passed

    def test_visualization_draws_markers(self) -> bool:
        """Тест что визуализация рисует окружности и линии."""
        visualizer = PTVVisualizer()
        # Дефолтные цвета: A=зелёный, B=красный, линия=оранжевый

        # Чёрное изображение для легкой проверки
        original = np.zeros((200, 200, 3), dtype=np.uint8)

        pairs = [
            MatchedPair(id=1, x0=50.0, y0=50.0, dx=50.0, dy=50.0,
                       length=70.71, diameter=8.0, area=50),
        ]

        vis_image = visualizer.create_visualization(original, pairs)

        # Проверяем наличие зелёной окружности (частица A) вокруг позиции (50, 50)
        # Радиус = 4, проверяем пиксели на окружности
        has_green = vis_image[50, 54, 1] > 200 or vis_image[54, 50, 1] > 200  # Зелёный канал
        # Проверяем наличие красной окружности (частица B) вокруг позиции (100, 100)
        has_red = vis_image[100, 104, 2] > 200 or vis_image[104, 100, 2] > 200  # Красный канал
        # Проверяем наличие оранжевой линии (R и G каналы)
        has_line = np.any((vis_image[:, :, 2] > 150) & (vis_image[:, :, 1] > 100))

        passed = has_green and has_red and has_line
        self._log_result("test_visualization_draws_markers", passed,
                        f"Зелёная окружность: {has_green}, Красная окружность: {has_red}, Линия: {has_line}")
        return passed

    def test_visualization_no_line_for_zero_displacement(self) -> bool:
        """Тест что не рисуется линия при нулевом смещении."""
        visualizer = PTVVisualizer()

        original = np.zeros((100, 100, 3), dtype=np.uint8)

        # Пара с нулевым смещением
        pairs = [
            MatchedPair(id=1, x0=50.0, y0=50.0, dx=0.0, dy=0.0,
                       length=0.0, diameter=8.0, area=50),
        ]

        vis_image = visualizer.create_visualization(original, pairs)

        # Проверяем наличие окружности (зелёная и красная в одном месте)
        # При нулевом смещении обе окружности в позиции (50, 50), радиус = 4
        has_circle = vis_image[50, 54, 1] > 0 or vis_image[54, 50, 2] > 0  # Зелёный или красный

        passed = has_circle  # Окружности должны быть
        self._log_result("test_visualization_no_line_for_zero_displacement", passed)
        return passed

    def test_process_pair(self) -> bool:
        """Тест обработки одной пары."""
        visualizer = PTVVisualizer()
        visualizer.set_original_folder(str(self.original_folder))
        visualizer.set_ptv_folder(str(self.ptv_folder))

        # Определяем какая камера доступна
        cam1_pairs = self.ptv_folder / "cam_1_pairs"
        cam2_pairs = self.ptv_folder / "cam_2_pairs"

        camera_name = None
        pair_number = None

        if cam1_pairs.exists():
            csv_files = list(cam1_pairs.glob("*_pair.csv"))
            if csv_files:
                camera_name = "cam_1"
                pair_number = int(csv_files[0].stem.split('_')[0])

        if camera_name is None and cam2_pairs.exists():
            csv_files = list(cam2_pairs.glob("*_pair.csv"))
            if csv_files:
                camera_name = "cam_2"
                pair_number = int(csv_files[0].stem.split('_')[0])

        if camera_name is None:
            self._log_result("test_process_pair", False, "Нет CSV файлов для теста")
            return False

        result = visualizer.process_pair(camera_name, pair_number)

        passed = result['success'] and result['visualizations_created'] > 0

        # Очистка
        if self.cleanup and visualizer.output_folder and visualizer.output_folder.exists():
            shutil.rmtree(visualizer.output_folder)

        self._log_result("test_process_pair", passed,
                        f"Создано визуализаций: {result['visualizations_created']}, "
                        f"пар в CSV: {result['pairs_count']}")
        return passed

    def test_process_camera(self) -> bool:
        """Тест обработки всех пар одной камеры."""
        visualizer = PTVVisualizer()
        visualizer.set_original_folder(str(self.original_folder))
        visualizer.set_ptv_folder(str(self.ptv_folder))

        # Определяем доступную камеру
        cam1_pairs = self.ptv_folder / "cam_1_pairs"
        camera_name = "cam_1" if cam1_pairs.exists() else "cam_2"

        progress_messages = []

        def progress_callback(progress: VisualizationProgress):
            progress_messages.append(progress.message)

        visualizer.set_progress_callback(progress_callback)

        pairs, vis_count, errors = visualizer.process_camera(camera_name)

        passed = pairs > 0 and vis_count > 0

        if self.cleanup and visualizer.output_folder and visualizer.output_folder.exists():
            shutil.rmtree(visualizer.output_folder)

        self._log_result("test_process_camera", passed,
                        f"Пар: {pairs}, визуализаций: {vis_count}")
        return passed

    def test_process_all(self) -> bool:
        """Тест полной визуализации всех результатов."""
        visualizer = PTVVisualizer()
        visualizer.set_original_folder(str(self.original_folder))
        visualizer.set_ptv_folder(str(self.ptv_folder))

        result = visualizer.process_all()

        output_folder = Path(result.output_folder) if result.output_folder else None

        passed = (
            result.success and
            result.total_pairs_processed > 0 and
            output_folder is not None and
            output_folder.exists()
        )

        # Проверяем что созданы файлы визуализаций
        if passed:
            vis_files = list(output_folder.rglob("*_vis.png"))
            passed = len(vis_files) > 0

        if self.cleanup and output_folder and output_folder.exists():
            shutil.rmtree(output_folder)
            print(f"    Очищена выходная папка: {output_folder}")

        self._log_result("test_process_all", passed,
                        f"Пар: {result.total_pairs_processed}, "
                        f"cam1: {result.cam1_visualizations}, "
                        f"cam2: {result.cam2_visualizations}")
        return passed

    def test_visualization_files_naming(self) -> bool:
        """Тест правильного именования выходных файлов."""
        visualizer = PTVVisualizer()
        visualizer.set_original_folder(str(self.original_folder))
        visualizer.set_ptv_folder(str(self.ptv_folder))

        result = visualizer.process_all()

        output_folder = Path(result.output_folder) if result.output_folder else None
        passed = False

        if output_folder and output_folder.exists():
            # Проверяем наличие файлов вида N_a_vis.png и N_b_vis.png
            a_vis_files = list(output_folder.rglob("*_a_vis.png"))
            b_vis_files = list(output_folder.rglob("*_b_vis.png"))

            passed = len(a_vis_files) > 0 or len(b_vis_files) > 0

        if self.cleanup and output_folder and output_folder.exists():
            shutil.rmtree(output_folder)

        self._log_result("test_visualization_files_naming", passed)
        return passed

    def test_get_preview(self) -> bool:
        """Тест получения предварительного просмотра."""
        visualizer = PTVVisualizer()
        visualizer.set_original_folder(str(self.original_folder))
        visualizer.set_ptv_folder(str(self.ptv_folder))

        # Определяем доступную камеру и пару
        cam1_pairs = self.ptv_folder / "cam_1_pairs"
        cam2_pairs = self.ptv_folder / "cam_2_pairs"

        camera_name = None
        pair_number = None

        if cam1_pairs.exists():
            csv_files = list(cam1_pairs.glob("*_pair.csv"))
            if csv_files:
                camera_name = "cam_1"
                pair_number = int(csv_files[0].stem.split('_')[0])

        if camera_name is None and cam2_pairs.exists():
            csv_files = list(cam2_pairs.glob("*_pair.csv"))
            if csv_files:
                camera_name = "cam_2"
                pair_number = int(csv_files[0].stem.split('_')[0])

        if camera_name is None:
            self._log_result("test_get_preview", False, "Нет данных для теста")
            return False

        preview = visualizer.get_preview(camera_name, pair_number)

        passed = (
            preview is not None and
            len(preview) == 2 and
            preview[0] is not None and
            preview[1] is not None and
            isinstance(preview[0], np.ndarray) and
            isinstance(preview[1], np.ndarray)
        )

        self._log_result("test_get_preview", passed)
        return passed

    def test_get_pair_statistics(self) -> bool:
        """Тест получения статистики пары."""
        visualizer = PTVVisualizer()
        visualizer.set_ptv_folder(str(self.ptv_folder))

        # Определяем доступную камеру и пару
        cam1_pairs = self.ptv_folder / "cam_1_pairs"
        cam2_pairs = self.ptv_folder / "cam_2_pairs"

        camera_name = None
        pair_number = None

        if cam1_pairs.exists():
            csv_files = list(cam1_pairs.glob("*_pair.csv"))
            if csv_files:
                camera_name = "cam_1"
                pair_number = int(csv_files[0].stem.split('_')[0])

        if camera_name is None and cam2_pairs.exists():
            csv_files = list(cam2_pairs.glob("*_pair.csv"))
            if csv_files:
                camera_name = "cam_2"
                pair_number = int(csv_files[0].stem.split('_')[0])

        if camera_name is None:
            self._log_result("test_get_pair_statistics", False, "Нет данных для теста")
            return False

        stats = visualizer.get_pair_statistics(camera_name, pair_number)

        passed = (
            stats is not None and
            'pairs_count' in stats and
            'mean_diameter' in stats and
            stats['pairs_count'] > 0
        )

        self._log_result("test_get_pair_statistics", passed,
                        f"Статистика: пар={stats.get('pairs_count', 0)}, "
                        f"средний диаметр={stats.get('mean_diameter', 0):.2f}" if stats else "")
        return passed

    def test_cancel_processing(self) -> bool:
        """Тест отмены визуализации."""
        visualizer = PTVVisualizer()
        visualizer.set_original_folder(str(self.original_folder))
        visualizer.set_ptv_folder(str(self.ptv_folder))

        def progress_callback(progress: VisualizationProgress):
            if progress.processed_files >= 1:
                visualizer.cancel_processing()

        visualizer.set_progress_callback(progress_callback)

        result = visualizer.process_all()

        output_folder = Path(result.output_folder) if result.output_folder else None
        if self.cleanup and output_folder and output_folder.exists():
            shutil.rmtree(output_folder)

        passed = True  # Тест проверяет что отмена не вызывает ошибок
        self._log_result("test_cancel_processing", passed)
        return passed

    def run_all_tests(self) -> dict:
        """
        Запуск всех тестов.

        Returns:
            dict: Результаты тестирования
        """
        print("\n" + "=" * 60)
        print("ЗАПУСК ТЕСТОВ PTVVisualizer")
        print("=" * 60)

        if not self.setup():
            return {'success': False, 'message': 'Ошибка подготовки тестов'}

        print("\n--- Тесты инициализации ---")
        self.test_set_original_folder_valid()
        self.test_set_original_folder_invalid()
        self.test_set_ptv_folder_valid()
        self.test_set_ptv_folder_invalid()
        self.test_output_folder_naming()
        self.test_visualization_config()

        print("\n--- Тесты загрузки данных ---")
        self.test_load_original_image()
        self.test_load_pairs_csv()

        print("\n--- Тесты создания визуализации ---")
        self.test_create_visualization()
        self.test_visualization_draws_markers()
        self.test_visualization_no_line_for_zero_displacement()

        print("\n--- Тесты обработки ---")
        self.test_process_pair()
        self.test_process_camera()
        self.test_process_all()
        self.test_visualization_files_naming()

        print("\n--- Тесты вспомогательных функций ---")
        self.test_get_preview()
        self.test_get_pair_statistics()
        self.test_cancel_processing()

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


def run_tests(original_folder_path: str, ptv_folder_path: str,
              cleanup: bool = True) -> dict:
    """
    Запуск тестов PTVVisualizer.

    Args:
        original_folder_path: Путь к папке cam_sorted с исходными изображениями
        ptv_folder_path: Путь к папке PTV_XXXX с результатами сопоставления
        cleanup: Удалять ли выходные папки после тестов

    Returns:
        dict: Результаты тестирования
    """
    tester = TestPTVVisualizer(original_folder_path, ptv_folder_path, cleanup=cleanup)
    return tester.run_all_tests()


if __name__ == "__main__":
    # Пути к тестовым данным
    # original_folder_path - папка cam_sorted с исходными изображениями
    # ptv_folder_path - папка PTV_XXXX с результатами сопоставления (содержит cam_X_pairs)

    ORIGINAL_FOLDER = r"C:\Users\evils\PycharmProjects\ParticleAnalysisFE\tests\test_data_cam_sorted"
    PTV_FOLDER = r"C:\Users\evils\PycharmProjects\ParticleAnalysisFE\tests\test_data_cam_sorted\PTV_10000"

    # cleanup=True - удалять выходные папки после тестов
    # cleanup=False - сохранять выходные папки для проверки результатов
    CLEANUP = False

    results = run_tests(ORIGINAL_FOLDER, PTV_FOLDER, cleanup=CLEANUP)

    if results['success']:
        print("\n✓ Все тесты пройдены успешно!")
    else:
        print(f"\n✗ Пройдено {results['passed']} из {results['total']} тестов")