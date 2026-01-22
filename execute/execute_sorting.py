"""
Модуль выполнения сортировки изображений по камерам для GUI ParticleAnalysis.

Этот модуль предоставляет готовую к использованию структуру для интеграции
с графическим интерфейсом. Все параметры четко определены и могут быть
легко привязаны к элементам GUI.

Алгоритм сортировки:
- Первые 2 фото → cam_1 (с отражением по горизонтали)
- Следующие 2 фото → cam_2 (без изменений)
- Цикл повторяется

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import logging

# Добавление пути к модулям проекта
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.cam_sorted import CamSorter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SortingProgress:
    """
    Информация о прогрессе сортировки для GUI.

    Attributes:
        current_file: Имя текущего обрабатываемого файла
        total_files: Общее количество файлов
        processed_files: Количество обработанных файлов
        percentage: Процент выполнения (0-100)
        message: Текстовое сообщение о текущем состоянии
        current_camera: Текущая камера (cam_1 или cam_2)
    """
    current_file: str
    total_files: int
    processed_files: int
    percentage: float
    message: str
    current_camera: str


@dataclass
class SortingResult:
    """
    Результат сортировки изображений.

    Attributes:
        success: Успешность выполнения
        cam1_pairs: Количество пар изображений в cam_1
        cam2_pairs: Количество пар изображений в cam_2
        cam1_count: Количество файлов в cam_1
        cam2_count: Количество файлов в cam_2
        total_images: Общее количество обработанных изображений
        output_folder: Путь к выходной папке
        errors: Список ошибок
    """
    success: bool
    cam1_pairs: int
    cam2_pairs: int
    cam1_count: int
    cam2_count: int
    total_images: int
    output_folder: str
    errors: list[str]


@dataclass
class SortingParameters:
    """
    Параметры сортировки изображений для GUI.

    Все параметры должны устанавливаться через GUI элементы:
    - input_folder: путь к папке с исходными PNG изображениями (через file dialog)
    - validate_format: валидировать ли формат изображений (checkbox)
    - enable_progress_callback: включить обратную связь прогресса (checkbox)
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    input_folder: str  # Путь к папке с исходными PNG изображениями

    # ОПЦИОНАЛЬНЫЕ ПАРАМЕТРЫ
    validate_format: bool = True  # Валидировать формат изображений (16-bit PNG)
    enable_progress_callback: bool = True  # Включить callback для прогресса

    # GUI ИНФОРМАЦИЯ (не используется в обработке)
    algorithm_description: str = (
        "Алгоритм сортировки:\n"
        "• Первые 2 фото → cam_1 (с отражением)\n"
        "• Следующие 2 фото → cam_2 (без изменений)\n"
        "• Цикл повторяется"
    )

    def validate(self) -> tuple[bool, str]:
        """
        Валидация параметров.

        Returns:
            tuple[bool, str]: (success, error_message)
        """
        # Проверка входной папки
        input_path = Path(self.input_folder)
        if not input_path.exists():
            return False, f"Входная папка не существует: {self.input_folder}"

        if not input_path.is_dir():
            return False, f"Указанный путь не является папкой: {self.input_folder}"

        # Проверка наличия PNG файлов
        png_files = list(input_path.glob("*.png"))
        if not png_files:
            return False, f"В папке {self.input_folder} нет PNG файлов"

        # Проверка количества файлов (должно быть кратно 4)
        if len(png_files) % 4 != 0:
            return False, (
                f"Количество изображений ({len(png_files)}) не кратно 4. "
                f"Алгоритм работает с циклом из 4 изображений."
            )

        return True, ""


class SortingExecutor:
    """
    Класс для выполнения сортировки изображений по камерам с параметрами для GUI.

    Использование:
        1. Создать экземпляр SortingExecutor
        2. Задать параметры через SortingParameters
        3. (Опционально) Установить callback для прогресса
        4. Вызвать execute() для запуска сортировки
        5. Получить результат SortingResult
    """

    def __init__(self):
        """Инициализация исполнителя сортировки."""
        self.sorter: Optional[CamSorter] = None
        self.parameters: Optional[SortingParameters] = None
        self._progress_callback: Optional[Callable[[SortingProgress], None]] = None
        self._cancel_requested: bool = False

        logger.info("Инициализирован SortingExecutor")

    def set_parameters(self, parameters: SortingParameters) -> tuple[bool, str]:
        """
        Установка параметров сортировки.

        Args:
            parameters: Параметры сортировки

        Returns:
            tuple[bool, str]: (success, error_message)
        """
        # Валидация параметров
        success, error_msg = parameters.validate()
        if not success:
            logger.error(f"Ошибка валидации параметров: {error_msg}")
            return False, error_msg

        self.parameters = parameters

        # Создание сортировщика
        try:
            self.sorter = CamSorter(parameters.input_folder)
            logger.info(f"Параметры установлены: input_folder={parameters.input_folder}")
            return True, ""
        except Exception as e:
            error_msg = f"Ошибка создания сортировщика: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def set_progress_callback(self, callback: Callable[[SortingProgress], None]) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция обратного вызова для GUI
                     Принимает SortingProgress с полями:
                     - current_file: имя текущего файла
                     - total_files: общее количество файлов
                     - processed_files: обработано файлов
                     - percentage: процент выполнения (0-100)
                     - message: текстовое сообщение
                     - current_camera: текущая камера
        """
        self._progress_callback = callback
        logger.info("Установлен callback для прогресса")

    def cancel(self) -> None:
        """
        Отмена выполнения сортировки.

        Этот метод следует вызвать из GUI при нажатии кнопки "Отмена".
        """
        self._cancel_requested = True
        logger.info("Запрошена отмена сортировки")

    def _report_progress(self, current_file: str, total: int, processed: int, camera: str) -> None:
        """
        Отправка информации о прогрессе в GUI.

        Args:
            current_file: Имя текущего файла
            total: Общее количество файлов
            processed: Обработано файлов
            camera: Текущая камера
        """
        if self._progress_callback and self.parameters and self.parameters.enable_progress_callback:
            percentage = (processed / total * 100) if total > 0 else 0
            progress = SortingProgress(
                current_file=current_file,
                total_files=total,
                processed_files=processed,
                percentage=percentage,
                message=f"Обработка {camera}: {current_file}",
                current_camera=camera
            )
            self._progress_callback(progress)

    def execute(self) -> SortingResult:
        """
        Выполнение сортировки изображений по камерам.

        Returns:
            SortingResult с результатами сортировки:
            - success: успешность выполнения
            - cam1_pairs: количество пар в cam_1
            - cam2_pairs: количество пар в cam_2
            - cam1_count: количество файлов в cam_1
            - cam2_count: количество файлов в cam_2
            - total_images: общее количество изображений
            - output_folder: путь к выходной папке
            - errors: список ошибок
        """
        if self.parameters is None or self.sorter is None:
            logger.error("Параметры не установлены")
            return SortingResult(
                success=False,
                cam1_pairs=0,
                cam2_pairs=0,
                cam1_count=0,
                cam2_count=0,
                total_images=0,
                output_folder="",
                errors=["Параметры не установлены"]
            )

        logger.info("=" * 60)
        logger.info("ЗАПУСК СОРТИРОВКИ ИЗОБРАЖЕНИЙ")
        logger.info(f"Входная папка: {self.parameters.input_folder}")
        logger.info(f"Валидация формата: {'Да' if self.parameters.validate_format else 'Нет'}")
        logger.info(self.parameters.algorithm_description)
        logger.info("=" * 60)

        errors = []
        self._cancel_requested = False

        try:
            # Получаем список файлов для прогресса
            input_path = Path(self.parameters.input_folder)
            png_files = sorted(input_path.glob("*.png"))
            total_files = len(png_files)

            # Обработка с прогрессом
            if self._progress_callback:
                for i, img_path in enumerate(png_files):
                    if self._cancel_requested:
                        logger.info("Сортировка отменена пользователем")
                        break

                    camera = "cam_1" if (i % 4) < 2 else "cam_2"
                    self._report_progress(img_path.name, total_files, i, camera)

            # Выполнение сортировки
            cam1_pairs, cam2_pairs = self.sorter.sort_images(
                validate_format=self.parameters.validate_format
            )

            # Финальный прогресс
            if self._progress_callback and not self._cancel_requested:
                self._report_progress("", total_files, total_files, "completed")

            # Получение статистики
            stats = self.sorter.get_statistics()

            success = not self._cancel_requested

        except Exception as e:
            error_msg = f"Ошибка сортировки: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            return SortingResult(
                success=False,
                cam1_pairs=0,
                cam2_pairs=0,
                cam1_count=0,
                cam2_count=0,
                total_images=0,
                output_folder="",
                errors=errors
            )

        logger.info("=" * 60)
        logger.info("ЗАВЕРШЕНИЕ СОРТИРОВКИ")
        logger.info(f"Успешно: {success}")
        logger.info(f"cam_1: {stats['cam1_pairs']} пар ({stats['cam1_count']} файлов)")
        logger.info(f"cam_2: {stats['cam2_pairs']} пар ({stats['cam2_count']} файлов)")
        logger.info(f"Всего: {stats['total_images']} изображений")
        logger.info(f"Выходная папка: {stats['output_folder']}")
        logger.info("=" * 60)

        return SortingResult(
            success=success,
            cam1_pairs=stats['cam1_pairs'],
            cam2_pairs=stats['cam2_pairs'],
            cam1_count=stats['cam1_count'],
            cam2_count=stats['cam2_count'],
            total_images=stats['total_images'],
            output_folder=stats['output_folder'],
            errors=errors
        )

    def get_output_folder(self) -> Optional[str]:
        """
        Получение пути к выходной папке.

        Returns:
            Путь к выходной папке или None
        """
        if self.sorter and self.sorter.output_folder:
            return str(self.sorter.output_folder)
        return None

    def get_input_statistics(self) -> Optional[dict]:
        """
        Получение статистики входной папки до сортировки.

        Returns:
            Словарь со статистикой или None
        """
        if self.parameters is None:
            return None

        input_path = Path(self.parameters.input_folder)
        png_files = list(input_path.glob("*.png"))

        return {
            'total_files': len(png_files),
            'expected_cam1_pairs': len(png_files) // 4,
            'expected_cam2_pairs': len(png_files) // 4,
            'input_folder': str(input_path),
            'output_folder': str(input_path) + "_cam_sorted"
        }


def run_sorting(input_folder: str, validate_format: bool = True,
               progress_callback: Optional[Callable] = None) -> SortingResult:
    """
    Удобная функция для запуска сортировки без создания объектов.

    Args:
        input_folder: Путь к папке с исходными PNG изображениями
        validate_format: Валидировать ли формат изображений (по умолчанию True)
        progress_callback: Callback функция для прогресса (опционально)

    Returns:
        SortingResult с результатами

    Example:
        >>> result = run_sorting(
        ...     input_folder="path/to/images",
        ...     validate_format=True
        ... )
        >>> print(f"Создано пар: cam_1={result.cam1_pairs}, cam_2={result.cam2_pairs}")
    """
    # Создание параметров
    params = SortingParameters(
        input_folder=input_folder,
        validate_format=validate_format,
        enable_progress_callback=progress_callback is not None
    )

    # Создание исполнителя
    executor = SortingExecutor()

    # Установка параметров
    success, error_msg = executor.set_parameters(params)
    if not success:
        logger.error(f"Ошибка установки параметров: {error_msg}")
        return SortingResult(
            success=False,
            cam1_pairs=0,
            cam2_pairs=0,
            cam1_count=0,
            cam2_count=0,
            total_images=0,
            output_folder="",
            errors=[error_msg]
        )

    # Установка callback
    if progress_callback:
        executor.set_progress_callback(progress_callback)

    # Выполнение
    return executor.execute()


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ ДЛЯ GUI
# ============================================================================

def example_gui_usage():
    """
    Пример использования модуля с GUI.

    Этот пример показывает, как интегрировать модуль с графическим интерфейсом.
    """
    print("\n" + "=" * 60)
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ С GUI - СОРТИРОВКА ИЗОБРАЖЕНИЙ")
    print("=" * 60)

    # === ШАГ 1: Задание параметров (из GUI элементов) ===
    parameters = SortingParameters(
        input_folder=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit",
        validate_format=True,
        enable_progress_callback=True
    )

    print(f"\nПараметры:")
    print(f"  Входная папка: {parameters.input_folder}")
    print(f"  Валидация формата: {'Да' if parameters.validate_format else 'Нет'}")
    print(f"\nАлгоритм:")
    print(parameters.algorithm_description)

    # === ШАГ 2: Создание исполнителя ===
    executor = SortingExecutor()

    # === ШАГ 3: Валидация и установка параметров ===
    success, error_msg = executor.set_parameters(parameters)
    if not success:
        print(f"\nОШИБКА: {error_msg}")
        return

    print("\n✓ Параметры валидны")

    # === ШАГ 4: Получение статистики до сортировки ===
    input_stats = executor.get_input_statistics()
    if input_stats:
        print(f"\nСтатистика входной папки:")
        print(f"  Всего файлов: {input_stats['total_files']}")
        print(f"  Ожидаемо пар в cam_1: {input_stats['expected_cam1_pairs']}")
        print(f"  Ожидаемо пар в cam_2: {input_stats['expected_cam2_pairs']}")
        print(f"  Выходная папка: {input_stats['output_folder']}")

    # === ШАГ 5: Установка callback для прогресса (для GUI progress bar) ===
    def progress_callback(progress: SortingProgress):
        """Callback для обновления GUI."""
        print(f"  [{progress.current_camera}] {progress.percentage:.1f}% - {progress.message}")

    executor.set_progress_callback(progress_callback)

    # === ШАГ 6: Выполнение сортировки ===
    print("\nЗапуск сортировки...")
    result = executor.execute()

    # === ШАГ 7: Обработка результата ===
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Успешно: {result.success}")
    print(f"cam_1: {result.cam1_pairs} пар ({result.cam1_count} файлов)")
    print(f"cam_2: {result.cam2_pairs} пар ({result.cam2_count} файлов)")
    print(f"Всего изображений: {result.total_images}")
    print(f"Выходная папка: {result.output_folder}")

    if result.errors:
        print("\nОшибки:")
        for error in result.errors:
            print(f"  - {error}")

    print("=" * 60)


if __name__ == "__main__":
    # При запуске модуля напрямую - показать пример использования
    example_gui_usage()
