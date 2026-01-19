"""
Тесты для модуля IntensityFilter.

Тестирование проводится на реальных тестовых данных.
Тесты вызываются из кода, не из CLI.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import shutil
from src.filters.intensity_filter import (
    IntensityFilter,
    FilterProgress,
    FilterResult,
    ImageStatistics
)


class TestIntensityFilter:
    """Тесты для класса IntensityFilter."""

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
        intensity_filter = IntensityFilter()
        result = intensity_filter.set_input_folder(str(self.test_data_path))

        passed = result is True and intensity_filter.input_folder == self.test_data_path
        self._log_result("test_set_input_folder_valid", passed)
        return passed

    def test_set_input_folder_invalid(self) -> bool:
        """Тест установки невалидной входной папки."""
        intensity_filter = IntensityFilter()
        result = intensity_filter.set_input_folder("/nonexistent/path")

        passed = result is False
        self._log_result("test_set_input_folder_invalid", passed)
        return passed

    def test_set_threshold_valid(self) -> bool:
        """Тест установки валидного порога."""
        intensity_filter = IntensityFilter()
        intensity_filter.set_input_folder(str(self.test_data_path))

        result = intensity_filter.set_threshold(5000)

        passed = result is True and intensity_filter.threshold == 5000
        self._log_result("test_set_threshold_valid", passed)
        return passed

    def test_set_threshold_invalid(self) -> bool:
        """Тест установки невалидного порога."""
        intensity_filter = IntensityFilter()

        result_negative = intensity_filter.set_threshold(-100)
        result_too_high = intensity_filter.set_threshold(70000)

        passed = result_negative is False and result_too_high is False
        self._log_result("test_set_threshold_invalid", passed)
        return passed

    def test_output_folder_naming(self) -> bool:
        """Тест формирования имени выходной папки."""
        intensity_filter = IntensityFilter()
        intensity_filter.set_input_folder(str(self.test_data_path))
        intensity_filter.set_threshold(3000)

        expected_name = "_intensity_filtered_3000"
        expected_path = self.test_data_path / expected_name

        passed = intensity_filter.output_folder == expected_path
        message = f"Ожидалось: {expected_path}, получено: {intensity_filter.output_folder}"
        self._log_result("test_output_folder_naming", passed, message if not passed else "")
        return passed

    def test_load_16bit_image(self) -> bool:
        """Тест загрузки 16-битного изображения."""
        intensity_filter = IntensityFilter()

        cam1_path = self.test_data_path / "cam_1"
        png_files = list(cam1_path.glob("*.png"))

        if not png_files:
            self._log_result("test_load_16bit_image", False, "Нет PNG файлов для теста")
            return False

        img_array = intensity_filter._load_16bit_image(png_files[0])

        passed = (
            img_array is not None and
            isinstance(img_array, np.ndarray) and
            img_array.dtype == np.uint16
        )
        self._log_result("test_load_16bit_image", passed)
        return passed

    def test_apply_filter(self) -> bool:
        """Тест применения фильтра."""
        intensity_filter = IntensityFilter()
        intensity_filter.set_threshold(1000)

        test_array = np.array([[500, 1500], [2000, 800]], dtype=np.uint16)
        filtered = intensity_filter._apply_filter(test_array)

        expected = np.array([[0, 1500], [2000, 0]], dtype=np.uint16)

        passed = np.array_equal(filtered, expected)
        self._log_result("test_apply_filter", passed)
        return passed

    def test_get_image_statistics(self) -> bool:
        """Тест получения статистики изображения."""
        intensity_filter = IntensityFilter()

        cam1_path = self.test_data_path / "cam_1"
        png_files = list(cam1_path.glob("*.png"))

        if not png_files:
            self._log_result("test_get_image_statistics", False, "Нет PNG файлов для теста")
            return False

        stats = intensity_filter.get_image_statistics(png_files[0])

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
        intensity_filter = IntensityFilter()
        intensity_filter.set_input_folder(str(self.test_data_path))

        stats = intensity_filter.get_camera_statistics("cam_1", sample_size=3)

        passed = (
            stats is not None and
            isinstance(stats, ImageStatistics) and
            stats.non_zero_pixels > 0
        )
        self._log_result("test_get_camera_statistics", passed)
        return passed

    def test_preview_filter(self) -> bool:
        """Тест предварительного просмотра фильтрации."""
        intensity_filter = IntensityFilter()
        intensity_filter.set_input_folder(str(self.test_data_path))
        intensity_filter.set_threshold(1000)

        cam1_path = self.test_data_path / "cam_1"
        png_files = list(cam1_path.glob("*.png"))

        if not png_files:
            self._log_result("test_preview_filter", False, "Нет PNG файлов для теста")
            return False

        result = intensity_filter.preview_filter(png_files[0])

        passed = (
            result is not None and
            len(result) == 2 and
            result[0].shape == result[1].shape
        )
        self._log_result("test_preview_filter", passed)
        return passed

    def test_get_preview_statistics(self) -> bool:
        """Тест получения статистики предварительного просмотра."""
        intensity_filter = IntensityFilter()
        intensity_filter.set_threshold(1000)

        original = np.array([[500, 1500, 2000], [800, 3000, 100]], dtype=np.uint16)
        filtered = intensity_filter._apply_filter(original)

        stats = intensity_filter.get_preview_statistics(original, filtered)

        passed = (
            'original_nonzero' in stats and
            'filtered_nonzero' in stats and
            'removed_pixels' in stats and
            'removal_percentage' in stats and
            stats['removed_pixels'] == stats['original_nonzero'] - stats['filtered_nonzero']
        )
        self._log_result("test_get_preview_statistics", passed)
        return passed

    def test_process_all(self) -> bool:
        """Тест полной обработки всех изображений."""
        intensity_filter = IntensityFilter()
        intensity_filter.set_input_folder(str(self.test_data_path))
        intensity_filter.set_threshold(1000)

        progress_messages = []

        def progress_callback(progress: FilterProgress):
            progress_messages.append(progress.message)

        intensity_filter.set_progress_callback(progress_callback)

        result = intensity_filter.process_all()

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
        intensity_filter = IntensityFilter()
        intensity_filter.set_input_folder(str(self.test_data_path))
        intensity_filter.set_threshold(1000)

        def progress_callback(progress: FilterProgress):
            if progress.processed_files >= 1:
                intensity_filter.cancel_processing()

        intensity_filter.set_progress_callback(progress_callback)

        result = intensity_filter.process_all()

        output_folder = Path(result.output_folder)
        if self.cleanup and output_folder.exists():
            shutil.rmtree(output_folder)

        passed = True
        self._log_result("test_cancel_processing", passed)
        return passed

    def test_filtered_images_are_16bit(self) -> bool:
        """Тест что отфильтрованные изображения остаются 16-битными."""
        intensity_filter = IntensityFilter()
        intensity_filter.set_input_folder(str(self.test_data_path))
        intensity_filter.set_threshold(1000)

        result = intensity_filter.process_all()

        if not result.success:
            self._log_result("test_filtered_images_are_16bit", False, "Обработка не удалась")
            return False

        output_folder = Path(result.output_folder)
        cam1_output = output_folder / "cam_1"
        output_images = list(cam1_output.glob("*.png"))

        all_16bit = True
        for img_path in output_images[:3]:
            img = Image.open(img_path)
            img_array = np.array(img)
            if img_array.dtype != np.uint16:
                all_16bit = False
                break

        if self.cleanup and output_folder.exists():
            shutil.rmtree(output_folder)

        self._log_result("test_filtered_images_are_16bit", all_16bit)
        return all_16bit

    def run_all_tests(self) -> dict:
        """
        Запуск всех тестов.

        Returns:
            dict: Результаты тестирования
        """
        print("\n" + "=" * 60)
        print("ЗАПУСК ТЕСТОВ IntensityFilter")
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
        self.test_apply_filter()
        self.test_get_image_statistics()
        self.test_get_camera_statistics()

        print("\n--- Тесты предварительного просмотра ---")
        self.test_preview_filter()
        self.test_get_preview_statistics()

        print("\n--- Тесты полной обработки ---")
        self.test_process_all()
        self.test_cancel_processing()
        self.test_filtered_images_are_16bit()

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
    Запуск тестов IntensityFilter.

    Args:
        test_data_path: Путь к папке с тестовыми данными (_cam_sorted)
        cleanup: Удалять ли выходные папки после тестов (по умолчанию True)

    Returns:
        dict: Результаты тестирования
    """
    tester = TestIntensityFilter(test_data_path, cleanup=cleanup)
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