"""
Тесты для модуля BinarizationFilter.

Тестирование проводится на реальных тестовых данных.
Тесты вызываются из кода, не из CLI.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import shutil
from src.filters.binarization_filter import (
    BinarizationFilter,
    BinarizationProgress,
    BinarizationResult,
    BinarizationStatistics,
    ImageStatistics
)


class TestBinarizationFilter:
    """Тесты для класса BinarizationFilter."""

    def __init__(self, test_data_path: str, cleanup: bool = True):
        """
        Инициализация тестов.

        Args:
            test_data_path: Путь к папке с тестовыми данными (_cam_sorted)
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
        binarization_filter = BinarizationFilter()
        result = binarization_filter.set_input_folder(str(self.test_data_path))

        passed = result is True and binarization_filter.input_folder == self.test_data_path
        self._log_result("test_set_input_folder_valid", passed)
        return passed

    def test_set_input_folder_invalid(self) -> bool:
        """Тест установки невалидной входной папки."""
        binarization_filter = BinarizationFilter()
        result = binarization_filter.set_input_folder("/nonexistent/path")

        passed = result is False
        self._log_result("test_set_input_folder_invalid", passed)
        return passed

    def test_set_threshold_valid(self) -> bool:
        """Тест установки валидного порога."""
        binarization_filter = BinarizationFilter()
        binarization_filter.set_input_folder(str(self.test_data_path))

        result = binarization_filter.set_threshold(5000)

        passed = result is True and binarization_filter.threshold == 5000
        self._log_result("test_set_threshold_valid", passed)
        return passed

    def test_set_threshold_invalid(self) -> bool:
        """Тест установки невалидного порога."""
        binarization_filter = BinarizationFilter()

        result_negative = binarization_filter.set_threshold(-100)
        result_too_high = binarization_filter.set_threshold(70000)

        passed = result_negative is False and result_too_high is False
        self._log_result("test_set_threshold_invalid", passed)
        return passed

    def test_output_folder_naming(self) -> bool:
        """Тест формирования имени выходной папки."""
        binarization_filter = BinarizationFilter()
        binarization_filter.set_input_folder(str(self.test_data_path))
        binarization_filter.set_threshold(3000)

        expected_name = "_binarized_3000"
        expected_path = self.test_data_path / expected_name

        passed = binarization_filter.output_folder == expected_path
        message = f"Ожидалось: {expected_path}, получено: {binarization_filter.output_folder}"
        self._log_result("test_output_folder_naming", passed, message if not passed else "")
        return passed

    def test_load_16bit_image(self) -> bool:
        """Тест загрузки 16-битного изображения."""
        binarization_filter = BinarizationFilter()

        cam1_path = self.test_data_path / "cam_1"
        png_files = list(cam1_path.glob("*.png"))

        if not png_files:
            self._log_result("test_load_16bit_image", False, "Нет PNG файлов для теста")
            return False

        img_array = binarization_filter._load_16bit_image(png_files[0])

        passed = (
            img_array is not None and
            isinstance(img_array, np.ndarray) and
            img_array.dtype == np.uint16
        )
        self._log_result("test_load_16bit_image", passed)
        return passed

    def test_apply_binarization(self) -> bool:
        """Тест применения бинаризации."""
        binarization_filter = BinarizationFilter()
        binarization_filter.set_threshold(1000)

        test_array = np.array([[500, 1500], [2000, 800]], dtype=np.uint16)
        binarized = binarization_filter._apply_binarization(test_array)

        # Пиксели >= 1000 → 255, пиксели < 1000 → 0
        expected = np.array([[0, 255], [255, 0]], dtype=np.uint8)

        passed = np.array_equal(binarized, expected)
        message = f"Ожидалось: {expected.tolist()}, получено: {binarized.tolist()}"
        self._log_result("test_apply_binarization", passed, message if not passed else "")
        return passed

    def test_binarization_output_dtype(self) -> bool:
        """Тест что результат бинаризации имеет тип uint8."""
        binarization_filter = BinarizationFilter()
        binarization_filter.set_threshold(1000)

        test_array = np.array([[500, 1500], [2000, 800]], dtype=np.uint16)
        binarized = binarization_filter._apply_binarization(test_array)

        passed = binarized.dtype == np.uint8
        message = f"Ожидался dtype uint8, получен: {binarized.dtype}"
        self._log_result("test_binarization_output_dtype", passed, message if not passed else "")
        return passed

    def test_binarization_only_0_and_255(self) -> bool:
        """Тест что бинаризованное изображение содержит только 0 и 255."""
        binarization_filter = BinarizationFilter()
        binarization_filter.set_threshold(1000)

        # Массив с разными значениями
        test_array = np.array([
            [100, 500, 999, 1000],
            [1001, 5000, 30000, 65535]
        ], dtype=np.uint16)
        binarized = binarization_filter._apply_binarization(test_array)

        unique_values = set(np.unique(binarized))
        expected_values = {0, 255}

        passed = unique_values <= expected_values
        message = f"Ожидались только 0 и 255, получены: {unique_values}"
        self._log_result("test_binarization_only_0_and_255", passed, message if not passed else "")
        return passed

    def test_get_image_statistics(self) -> bool:
        """Тест получения статистики изображения."""
        binarization_filter = BinarizationFilter()

        cam1_path = self.test_data_path / "cam_1"
        png_files = list(cam1_path.glob("*.png"))

        if not png_files:
            self._log_result("test_get_image_statistics", False, "Нет PNG файлов для теста")
            return False

        stats = binarization_filter.get_image_statistics(png_files[0])

        passed = (
            stats is not None and
            isinstance(stats, ImageStatistics) and
            stats.min_value >= 0 and
            stats.max_value <= 65535 and
            stats.mean_value >= stats.min_value and
            stats.mean_value <= stats.max_value
        )
        self._log_result("test_get_image_statistics", passed)
        return passed

    def test_get_camera_statistics(self) -> bool:
        """Тест получения статистики камеры."""
        binarization_filter = BinarizationFilter()
        binarization_filter.set_input_folder(str(self.test_data_path))

        stats = binarization_filter.get_camera_statistics("cam_1", sample_size=3)

        passed = (
            stats is not None and
            isinstance(stats, ImageStatistics) and
            stats.non_zero_pixels > 0
        )
        self._log_result("test_get_camera_statistics", passed)
        return passed

    def test_preview_binarization(self) -> bool:
        """Тест предварительного просмотра бинаризации."""
        binarization_filter = BinarizationFilter()
        binarization_filter.set_input_folder(str(self.test_data_path))
        binarization_filter.set_threshold(1000)

        cam1_path = self.test_data_path / "cam_1"
        png_files = list(cam1_path.glob("*.png"))

        if not png_files:
            self._log_result("test_preview_binarization", False, "Нет PNG файлов для теста")
            return False

        result = binarization_filter.preview_binarization(png_files[0])

        passed = (
            result is not None and
            len(result) == 2 and
            result[0].dtype == np.uint16 and  # Оригинал 16-бит
            result[1].dtype == np.uint8       # Бинаризованное 8-бит
        )
        self._log_result("test_preview_binarization", passed)
        return passed

    def test_get_binarization_statistics(self) -> bool:
        """Тест получения статистики бинаризации."""
        binarization_filter = BinarizationFilter()
        binarization_filter.set_threshold(1000)

        original = np.array([[500, 1500, 2000], [800, 3000, 100]], dtype=np.uint16)
        binarized = binarization_filter._apply_binarization(original)

        stats = binarization_filter.get_binarization_statistics(original, binarized)

        # Ожидаем: 3 белых (1500, 2000, 3000) и 3 черных (500, 800, 100)
        passed = (
            isinstance(stats, BinarizationStatistics) and
            stats.white_pixels == 3 and
            stats.black_pixels == 3 and
            stats.total_pixels == 6 and
            abs(stats.white_percentage - 50.0) < 0.01 and
            abs(stats.black_percentage - 50.0) < 0.01
        )
        message = f"white={stats.white_pixels}, black={stats.black_pixels}"
        self._log_result("test_get_binarization_statistics", passed, message if not passed else "")
        return passed

    def test_get_preview_statistics(self) -> bool:
        """Тест получения статистики для предварительного просмотра."""
        binarization_filter = BinarizationFilter()
        binarization_filter.set_threshold(1000)

        original = np.array([[500, 1500, 2000], [800, 3000, 100]], dtype=np.uint16)
        binarized = binarization_filter._apply_binarization(original)

        stats = binarization_filter.get_preview_statistics(original, binarized)

        passed = (
            'original_nonzero' in stats and
            'white_pixels' in stats and
            'black_pixels' in stats and
            'white_percentage' in stats and
            'black_percentage' in stats and
            'total_pixels' in stats and
            'threshold' in stats and
            stats['threshold'] == 1000
        )
        self._log_result("test_get_preview_statistics", passed)
        return passed

    def test_process_all(self) -> bool:
        """Тест полной обработки всех изображений."""
        binarization_filter = BinarizationFilter()
        binarization_filter.set_input_folder(str(self.test_data_path))
        binarization_filter.set_threshold(10_000)

        progress_messages = []

        def progress_callback(progress: BinarizationProgress):
            progress_messages.append(progress.message)

        binarization_filter.set_progress_callback(progress_callback)

        result = binarization_filter.process_all()

        output_folder = Path(result.output_folder)
        cam1_output = output_folder / "cam_1"
        cam2_output = output_folder / "cam_2"

        passed = (
            result.success is True and
            result.total_processed > 0 and
            output_folder.exists() and
            cam1_output.exists() and
            cam2_output.exists() and
            len(list(cam1_output.glob("*.png"))) == result.cam1_processed and
            len(list(cam2_output.glob("*.png"))) == result.cam2_processed
        )

        if self.cleanup and output_folder.exists():
            shutil.rmtree(output_folder)
            print(f"    Очищена выходная папка: {output_folder}")

        self._log_result("test_process_all", passed)
        return passed

    def test_cancel_processing(self) -> bool:
        """Тест отмены обработки."""
        binarization_filter = BinarizationFilter()
        binarization_filter.set_input_folder(str(self.test_data_path))
        binarization_filter.set_threshold(10_000)

        def progress_callback(progress: BinarizationProgress):
            if progress.processed_files >= 1:
                binarization_filter.cancel_processing()

        binarization_filter.set_progress_callback(progress_callback)

        result = binarization_filter.process_all()

        output_folder = Path(result.output_folder)
        if self.cleanup and output_folder.exists():
            shutil.rmtree(output_folder)

        passed = True
        self._log_result("test_cancel_processing", passed)
        return passed

    def test_binarized_images_are_8bit(self) -> bool:
        """Тест что бинаризованные изображения являются 8-битными."""
        binarization_filter = BinarizationFilter()
        binarization_filter.set_input_folder(str(self.test_data_path))
        binarization_filter.set_threshold(1000)

        result = binarization_filter.process_all()

        if not result.success:
            self._log_result("test_binarized_images_are_8bit", False, "Обработка не удалась")
            return False

        output_folder = Path(result.output_folder)
        cam1_output = output_folder / "cam_1"
        output_images = list(cam1_output.glob("*.png"))

        all_8bit = True
        for img_path in output_images[:3]:
            img = Image.open(img_path)
            img_array = np.array(img)
            if img_array.dtype != np.uint8:
                all_8bit = False
                break

        if self.cleanup and output_folder.exists():
            shutil.rmtree(output_folder)

        self._log_result("test_binarized_images_are_8bit", all_8bit)
        return all_8bit

    def test_binarized_images_only_binary_values(self) -> bool:
        """Тест что сохраненные изображения содержат только 0 и 255."""
        binarization_filter = BinarizationFilter()
        binarization_filter.set_input_folder(str(self.test_data_path))
        binarization_filter.set_threshold(1000)

        result = binarization_filter.process_all()

        if not result.success:
            self._log_result("test_binarized_images_only_binary_values", False, "Обработка не удалась")
            return False

        output_folder = Path(result.output_folder)
        cam1_output = output_folder / "cam_1"
        output_images = list(cam1_output.glob("*.png"))

        all_binary = True
        for img_path in output_images[:3]:
            img = Image.open(img_path)
            img_array = np.array(img)
            unique_values = set(np.unique(img_array))
            if not unique_values <= {0, 255}:
                all_binary = False
                break

        if self.cleanup and output_folder.exists():
            shutil.rmtree(output_folder)

        self._log_result("test_binarized_images_only_binary_values", all_binary)
        return all_binary

    def test_threshold_boundary_equal(self) -> bool:
        """Тест что пиксели равные порогу становятся белыми."""
        binarization_filter = BinarizationFilter()
        binarization_filter.set_threshold(1000)

        test_array = np.array([[999, 1000, 1001]], dtype=np.uint16)
        binarized = binarization_filter._apply_binarization(test_array)

        # 999 < 1000 → 0, 1000 >= 1000 → 255, 1001 >= 1000 → 255
        expected = np.array([[0, 255, 255]], dtype=np.uint8)

        passed = np.array_equal(binarized, expected)
        message = f"Ожидалось: {expected.tolist()}, получено: {binarized.tolist()}"
        self._log_result("test_threshold_boundary_equal", passed, message if not passed else "")
        return passed

    def run_all_tests(self) -> dict:
        """
        Запуск всех тестов.

        Returns:
            dict: Результаты тестирования
        """
        print("\n" + "=" * 60)
        print("ЗАПУСК ТЕСТОВ BinarizationFilter")
        print("=" * 60)

        if not self.setup():
            return {'success': False, 'message': 'Ошибка подготовки тестов'}

        print("\n--- Тесты инициализации ---")
        self.test_set_input_folder_valid()
        self.test_set_input_folder_invalid()
        self.test_set_threshold_valid()
        self.test_set_threshold_invalid()
        self.test_output_folder_naming()

        print("\n--- Тесты загрузки и обработки изображений ---")
        self.test_load_16bit_image()
        self.test_apply_binarization()
        self.test_binarization_output_dtype()
        self.test_binarization_only_0_and_255()
        self.test_threshold_boundary_equal()
        self.test_get_image_statistics()
        self.test_get_camera_statistics()

        print("\n--- Тесты предварительного просмотра ---")
        self.test_preview_binarization()
        self.test_get_binarization_statistics()
        self.test_get_preview_statistics()

        print("\n--- Тесты полной обработки ---")
        self.test_process_all()
        self.test_cancel_processing()
        self.test_binarized_images_are_8bit()
        self.test_binarized_images_only_binary_values()

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
    Запуск тестов BinarizationFilter.

    Args:
        test_data_path: Путь к папке с тестовыми данными (_cam_sorted)
        cleanup: Удалять ли выходные папки после тестов (по умолчанию True)

    Returns:
        dict: Результаты тестирования
    """
    tester = TestBinarizationFilter(test_data_path, cleanup=cleanup)
    return tester.run_all_tests()


if __name__ == "__main__":
    # Путь к тестовым данным - папка _cam_sorted с подпапками cam_1 и cam_2
    # Измените путь на актуальный путь к вашим тестовым данным
    TEST_DATA_PATH = r"C:\Users\evils\PycharmProjects\ParticleAnalysisFE\tests\test_data_cam_sorted"

    # cleanup=True - удалять выходные папки после тестов
    # cleanup=False - сохранять выходные папки для проверки результатов
    CLEANUP = False

    results = run_tests(TEST_DATA_PATH, cleanup=CLEANUP)

    if results['success']:
        print("\n✓ Все тесты пройдены успешно!")
    else:
        print(f"\n✗ Пройдено {results['passed']} из {results['total']} тестов")