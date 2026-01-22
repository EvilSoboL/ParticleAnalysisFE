"""
Тесты для модуля VectorFieldVisualizer.

Тестирование проводится на реальных тестовых данных.
Тесты вызываются из кода, не из CLI.
"""

import numpy as np
import cv2
from pathlib import Path
import shutil
from src.visualization.vector_field import (
    VectorFieldVisualizer,
    VectorData,
    VectorFieldConfig,
    VectorFieldResult
)


class TestVectorFieldVisualizer:
    """Тесты для класса VectorFieldVisualizer."""

    def __init__(self, ptv_folder_path: str, cam_sorted_path: str = None, cleanup: bool = False):
        """
        Инициализация тестов.

        Args:
            ptv_folder_path: Путь к папке PTV_XXXX с результатами
            cam_sorted_path: Путь к папке cam_sorted (опционально)
            cleanup: Удалять ли выходные папки после тестов
        """
        self.ptv_folder = Path(ptv_folder_path)
        self.cam_sorted_path = Path(cam_sorted_path) if cam_sorted_path else None
        self.cleanup = cleanup
        self.results = []

    def setup(self) -> bool:
        """
        Подготовка к тестированию.

        Returns:
            bool: True если подготовка успешна
        """
        if not self.ptv_folder.exists():
            print(f"ОШИБКА: Папка PTV результатов не найдена: {self.ptv_folder}")
            return False

        # Проверка наличия суммарных CSV файлов
        cam1_sum = self.ptv_folder / "cam_1_pairs_sum.csv"
        cam2_sum = self.ptv_folder / "cam_2_pairs_sum.csv"

        has_sum_files = cam1_sum.exists() or cam2_sum.exists()
        if not has_sum_files:
            print("ОШИБКА: Папка PTV должна содержать cam_1_pairs_sum.csv и/или cam_2_pairs_sum.csv")
            return False

        print(f"PTV результаты: {self.ptv_folder}")
        if cam1_sum.exists():
            print(f"  ✓ cam_1_pairs_sum.csv найден")
        if cam2_sum.exists():
            print(f"  ✓ cam_2_pairs_sum.csv найден")

        if self.cam_sorted_path:
            if self.cam_sorted_path.exists():
                print(f"Исходные изображения: {self.cam_sorted_path}")
            else:
                print(f"ВНИМАНИЕ: Папка cam_sorted не найдена: {self.cam_sorted_path}")

        return True

    def teardown(self) -> None:
        """Очистка после тестов."""
        if self.cleanup:
            output_folder = self.ptv_folder / "vector_field"
            if output_folder.exists():
                shutil.rmtree(output_folder)
                print(f"Удалена выходная папка: {output_folder}")

    def _log_result(self, test_name: str, passed: bool, message: str = "") -> None:
        """Логирование результата теста."""
        status = "✓ PASSED" if passed else "✗ FAILED"
        self.results.append((test_name, passed, message))
        print(f"  {status}: {test_name}")
        if message and not passed:
            print(f"    → {message}")

    def test_set_ptv_folder_valid(self) -> bool:
        """Тест установки валидной папки PTV результатов."""
        visualizer = VectorFieldVisualizer()
        result = visualizer.set_ptv_folder(str(self.ptv_folder))

        passed = result is True and visualizer.ptv_folder == self.ptv_folder
        self._log_result("test_set_ptv_folder_valid", passed)
        return passed

    def test_set_ptv_folder_invalid(self) -> bool:
        """Тест установки невалидной папки PTV."""
        visualizer = VectorFieldVisualizer()
        result = visualizer.set_ptv_folder("/nonexistent/path")

        passed = result is False
        self._log_result("test_set_ptv_folder_invalid", passed)
        return passed

    def test_output_folder_naming(self) -> bool:
        """Тест формирования имени выходной папки."""
        visualizer = VectorFieldVisualizer()
        visualizer.set_ptv_folder(str(self.ptv_folder))

        expected_path = self.ptv_folder / "vector_field"

        passed = visualizer.output_folder == expected_path
        message = f"Ожидалось: {expected_path}, получено: {visualizer.output_folder}"
        self._log_result("test_output_folder_naming", passed, message if not passed else "")
        return passed

    def test_set_original_folder(self) -> bool:
        """Тест установки папки с исходными изображениями."""
        if not self.cam_sorted_path or not self.cam_sorted_path.exists():
            print("  ⊘ SKIPPED: test_set_original_folder (cam_sorted не указан)")
            return True

        visualizer = VectorFieldVisualizer()
        result = visualizer.set_original_folder(str(self.cam_sorted_path))

        passed = result is True
        self._log_result("test_set_original_folder", passed)
        return passed

    def test_config_defaults(self) -> bool:
        """Тест значений конфигурации по умолчанию."""
        visualizer = VectorFieldVisualizer()
        cfg = visualizer.config

        passed = (
            cfg.image_width == 1024 and
            cfg.image_height == 1024 and
            cfg.arrow_thickness == 2 and
            cfg.scale_factor == 1.0 and
            cfg.draw_start_points is True
        )

        message = "Конфигурация по умолчанию не соответствует ожидаемой"
        self._log_result("test_config_defaults", passed, message if not passed else "")
        return passed

    def test_set_config(self) -> bool:
        """Тест установки пользовательской конфигурации."""
        visualizer = VectorFieldVisualizer()

        visualizer.set_config(
            image_width=2048,
            image_height=2048,
            arrow_color=(255, 0, 0),
            arrow_thickness=3,
            scale_factor=2.0,
            background_color=(0, 0, 0),
            draw_start_points=False,
            start_point_color=(255, 255, 0)
        )

        cfg = visualizer.config

        passed = (
            cfg.image_width == 2048 and
            cfg.image_height == 2048 and
            cfg.arrow_color == (255, 0, 0) and
            cfg.arrow_thickness == 3 and
            cfg.scale_factor == 2.0 and
            cfg.background_color == (0, 0, 0) and
            cfg.draw_start_points is False and
            cfg.start_point_color == (255, 255, 0)
        )

        message = "Конфигурация не обновилась корректно"
        self._log_result("test_set_config", passed, message if not passed else "")
        return passed

    def test_load_vectors_csv_cam1(self) -> bool:
        """Тест загрузки векторов из cam_1_pairs_sum.csv."""
        csv_path = self.ptv_folder / "cam_1_pairs_sum.csv"

        if not csv_path.exists():
            print("  ⊘ SKIPPED: test_load_vectors_csv_cam1 (файл не найден)")
            return True

        visualizer = VectorFieldVisualizer()
        visualizer.set_ptv_folder(str(self.ptv_folder))

        vectors = visualizer._load_vectors_csv(csv_path)

        passed = len(vectors) > 0
        message = f"Загружено векторов: {len(vectors)}"
        self._log_result("test_load_vectors_csv_cam1", passed, message if not passed else "")

        # Дополнительная проверка структуры данных
        if passed and len(vectors) > 0:
            v = vectors[0]
            has_attrs = hasattr(v, 'x0') and hasattr(v, 'y0') and hasattr(v, 'dx') and hasattr(v, 'dy')
            if not has_attrs:
                self._log_result("test_vector_data_structure", False, "VectorData не содержит нужных атрибутов")
                return False

        return passed

    def test_load_vectors_csv_cam2(self) -> bool:
        """Тест загрузки векторов из cam_2_pairs_sum.csv."""
        csv_path = self.ptv_folder / "cam_2_pairs_sum.csv"

        if not csv_path.exists():
            print("  ⊘ SKIPPED: test_load_vectors_csv_cam2 (файл не найден)")
            return True

        visualizer = VectorFieldVisualizer()
        visualizer.set_ptv_folder(str(self.ptv_folder))

        vectors = visualizer._load_vectors_csv(csv_path)

        passed = len(vectors) > 0
        message = f"Загружено векторов: {len(vectors)}"
        self._log_result("test_load_vectors_csv_cam2", passed, message if not passed else "")
        return passed

    def test_create_vector_field(self) -> bool:
        """Тест создания изображения векторного поля."""
        csv_path = self.ptv_folder / "cam_1_pairs_sum.csv"
        if not csv_path.exists():
            csv_path = self.ptv_folder / "cam_2_pairs_sum.csv"

        if not csv_path.exists():
            print("  ⊘ SKIPPED: test_create_vector_field (CSV файлы не найдены)")
            return True

        visualizer = VectorFieldVisualizer()
        visualizer.set_ptv_folder(str(self.ptv_folder))

        vectors = visualizer._load_vectors_csv(csv_path)
        if len(vectors) == 0:
            self._log_result("test_create_vector_field", False, "Нет векторов для визуализации")
            return False

        image = visualizer.create_vector_field(vectors)

        passed = (
            image is not None and
            isinstance(image, np.ndarray) and
            image.shape == (visualizer.config.image_height,
                          visualizer.config.image_width, 3)
        )

        message = f"Размер изображения: {image.shape if image is not None else 'None'}"
        self._log_result("test_create_vector_field", passed, message if not passed else "")
        return passed

    def test_process_camera_cam1(self) -> bool:
        """Тест обработки cam_1."""
        csv_path = self.ptv_folder / "cam_1_pairs_sum.csv"

        if not csv_path.exists():
            print("  ⊘ SKIPPED: test_process_camera_cam1 (файл не найден)")
            return True

        visualizer = VectorFieldVisualizer()
        visualizer.set_ptv_folder(str(self.ptv_folder))

        vectors_count, errors = visualizer.process_camera("cam_1")

        passed = vectors_count > 0 and len(errors) == 0
        message = f"Векторов: {vectors_count}, ошибок: {len(errors)}"
        self._log_result("test_process_camera_cam1", passed, message if not passed else "")

        # Проверка создания файла
        output_path = visualizer.output_folder / "cam_1_vector_field.png"
        file_exists = output_path.exists()
        self._log_result("test_output_file_cam1_exists", file_exists,
                        f"Файл не создан: {output_path}" if not file_exists else "")

        return passed and file_exists

    def test_process_camera_cam2(self) -> bool:
        """Тест обработки cam_2."""
        csv_path = self.ptv_folder / "cam_2_pairs_sum.csv"

        if not csv_path.exists():
            print("  ⊘ SKIPPED: test_process_camera_cam2 (файл не найден)")
            return True

        visualizer = VectorFieldVisualizer()
        visualizer.set_ptv_folder(str(self.ptv_folder))

        vectors_count, errors = visualizer.process_camera("cam_2")

        passed = vectors_count > 0 and len(errors) == 0
        message = f"Векторов: {vectors_count}, ошибок: {len(errors)}"
        self._log_result("test_process_camera_cam2", passed, message if not passed else "")

        # Проверка создания файла
        output_path = visualizer.output_folder / "cam_2_vector_field.png"
        file_exists = output_path.exists()
        self._log_result("test_output_file_cam2_exists", file_exists,
                        f"Файл не создан: {output_path}" if not file_exists else "")

        return passed and file_exists

    def test_process_all(self) -> bool:
        """Тест обработки всех камер."""
        visualizer = VectorFieldVisualizer()
        visualizer.set_ptv_folder(str(self.ptv_folder))

        if self.cam_sorted_path and self.cam_sorted_path.exists():
            visualizer.set_original_folder(str(self.cam_sorted_path))

        result = visualizer.process_all()

        total_vectors = result.cam1_vectors_count + result.cam2_vectors_count

        passed = result.success and total_vectors > 0 and len(result.errors) == 0
        message = (f"cam_1: {result.cam1_vectors_count}, "
                  f"cam_2: {result.cam2_vectors_count}, "
                  f"ошибок: {len(result.errors)}")

        self._log_result("test_process_all", passed, message if not passed else message)
        return passed

    def test_get_statistics_cam1(self) -> bool:
        """Тест получения статистики для cam_1."""
        csv_path = self.ptv_folder / "cam_1_pairs_sum.csv"

        if not csv_path.exists():
            print("  ⊘ SKIPPED: test_get_statistics_cam1 (файл не найден)")
            return True

        visualizer = VectorFieldVisualizer()
        visualizer.set_ptv_folder(str(self.ptv_folder))

        stats = visualizer.get_statistics("cam_1")

        passed = (
            stats is not None and
            'vectors_count' in stats and
            stats['vectors_count'] > 0
        )

        if passed and stats['vectors_with_displacement'] > 0:
            print(f"    Статистика cam_1:")
            print(f"      Векторов: {stats['vectors_count']}")
            print(f"      С ненулевым смещением: {stats['vectors_with_displacement']}")
            print(f"      Средняя длина: {stats.get('mean_length', 0):.2f}")
            print(f"      Среднее dx: {stats.get('mean_dx', 0):.2f}")
            print(f"      Среднее dy: {stats.get('mean_dy', 0):.2f}")

        self._log_result("test_get_statistics_cam1", passed)
        return passed

    def test_get_statistics_cam2(self) -> bool:
        """Тест получения статистики для cam_2."""
        csv_path = self.ptv_folder / "cam_2_pairs_sum.csv"

        if not csv_path.exists():
            print("  ⊘ SKIPPED: test_get_statistics_cam2 (файл не найден)")
            return True

        visualizer = VectorFieldVisualizer()
        visualizer.set_ptv_folder(str(self.ptv_folder))

        stats = visualizer.get_statistics("cam_2")

        passed = (
            stats is not None and
            'vectors_count' in stats and
            stats['vectors_count'] > 0
        )

        if passed and stats['vectors_with_displacement'] > 0:
            print(f"    Статистика cam_2:")
            print(f"      Векторов: {stats['vectors_count']}")
            print(f"      С ненулевым смещением: {stats['vectors_with_displacement']}")
            print(f"      Средняя длина: {stats.get('mean_length', 0):.2f}")
            print(f"      Среднее dx: {stats.get('mean_dx', 0):.2f}")
            print(f"      Среднее dy: {stats.get('mean_dy', 0):.2f}")

        self._log_result("test_get_statistics_cam2", passed)
        return passed

    def test_get_preview_cam1(self) -> bool:
        """Тест получения предварительного просмотра для cam_1."""
        csv_path = self.ptv_folder / "cam_1_pairs_sum.csv"

        if not csv_path.exists():
            print("  ⊘ SKIPPED: test_get_preview_cam1 (файл не найден)")
            return True

        visualizer = VectorFieldVisualizer()
        visualizer.set_ptv_folder(str(self.ptv_folder))

        preview = visualizer.get_preview("cam_1")

        passed = preview is not None and isinstance(preview, np.ndarray)
        message = f"Размер preview: {preview.shape if preview is not None else 'None'}"
        self._log_result("test_get_preview_cam1", passed, message if not passed else "")
        return passed

    def test_scale_factor(self) -> bool:
        """Тест применения масштабного коэффициента."""
        csv_path = self.ptv_folder / "cam_1_pairs_sum.csv"
        if not csv_path.exists():
            csv_path = self.ptv_folder / "cam_2_pairs_sum.csv"

        if not csv_path.exists():
            print("  ⊘ SKIPPED: test_scale_factor (CSV файлы не найдены)")
            return True

        visualizer = VectorFieldVisualizer()
        visualizer.set_ptv_folder(str(self.ptv_folder))

        # Тест с разными масштабами
        scales = [0.5, 1.0, 2.0]
        passed = True

        for scale in scales:
            visualizer.set_config(scale_factor=scale)
            vectors = visualizer._load_vectors_csv(csv_path)
            image = visualizer.create_vector_field(vectors)

            if image is None:
                passed = False
                break

        message = f"Протестированы масштабы: {scales}"
        self._log_result("test_scale_factor", passed, message if not passed else "")
        return passed

    def run_all_tests(self) -> bool:
        """
        Запуск всех тестов.

        Returns:
            bool: True если все тесты прошли успешно
        """
        print("\n" + "=" * 60)
        print("ЗАПУСК ТЕСТОВ: VectorFieldVisualizer")
        print("=" * 60)

        if not self.setup():
            print("Ошибка подготовки к тестированию")
            return False

        print("\nЗапуск тестов:")

        # Список всех тестов
        tests = [
            self.test_set_ptv_folder_valid,
            self.test_set_ptv_folder_invalid,
            self.test_output_folder_naming,
            self.test_set_original_folder,
            self.test_config_defaults,
            self.test_set_config,
            self.test_load_vectors_csv_cam1,
            self.test_load_vectors_csv_cam2,
            self.test_create_vector_field,
            self.test_process_camera_cam1,
            self.test_process_camera_cam2,
            self.test_process_all,
            self.test_get_statistics_cam1,
            self.test_get_statistics_cam2,
            self.test_get_preview_cam1,
            self.test_scale_factor,
        ]

        # Выполнение тестов
        for test in tests:
            try:
                test()
            except Exception as e:
                self._log_result(test.__name__, False, f"Exception: {e}")

        # Подведение итогов
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ ТЕСТОВ")
        print("=" * 60)

        passed_count = sum(1 for _, passed, _ in self.results if passed)
        total_count = len(self.results)

        print(f"Пройдено: {passed_count}/{total_count}")

        failed_tests = [(name, msg) for name, passed, msg in self.results if not passed]
        if failed_tests:
            print("\nПроваленные тесты:")
            for name, msg in failed_tests:
                print(f"  ✗ {name}")
                if msg:
                    print(f"    → {msg}")

        print("=" * 60)

        self.teardown()

        return len(failed_tests) == 0


# Пример использования
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python test_vector_field.py <путь_к_папке_PTV> [путь_к_cam_sorted]")
        sys.exit(1)

    ptv_path = sys.argv[1]
    cam_sorted = sys.argv[2] if len(sys.argv) > 2 else None

    tester = TestVectorFieldVisualizer(
        ptv_folder_path=ptv_path,
        cam_sorted_path=cam_sorted,
        cleanup=False  # Не удаляем результаты для проверки
    )

    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
