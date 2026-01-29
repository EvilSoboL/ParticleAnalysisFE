"""
Модуль фильтрации векторов по значениям U и V для GUI ParticleAnalysis.

Этот модуль позволяет фильтровать CSV файлы с векторами оптического потока
по допустимым диапазонам значений компонент скорости U и V.

Входной формат CSV (разделитель ;):
    X0;Y0;X1;Y1;U;V;Magnitude;Angle
    (столбцы Magnitude и Angle опциональны)

Выходной формат: исходный_файл_filtered.csv

Автор: ParticleAnalysis Team
Версия: 1.0
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict, Any
import logging
import csv

# Добавление пути к модулям проекта
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VectorFilterProgress:
    """Класс для передачи информации о прогрессе фильтрации."""
    current_file: str
    total_files: int
    processed_files: int
    percentage: float
    message: str


@dataclass
class VectorFilterResult:
    """Результат фильтрации векторов."""
    success: bool
    total_files_processed: int
    total_vectors_input: int
    total_vectors_output: int
    vectors_removed: int
    removal_percentage: float
    errors: List[str]
    warnings: List[str]
    output_files: List[str]


@dataclass
class VectorFilterStatistics:
    """Статистика фильтрации для одного файла."""
    input_vectors: int
    output_vectors: int
    removed_vectors: int
    removal_percentage: float
    u_min_filtered: int  # Отфильтровано по U < u_min
    u_max_filtered: int  # Отфильтровано по U > u_max
    v_min_filtered: int  # Отфильтровано по V < v_min
    v_max_filtered: int  # Отфильтровано по V > v_max


@dataclass
class VectorFilterParameters:
    """
    Параметры фильтрации векторов для GUI.

    Все параметры должны устанавливаться через GUI элементы:
    - input_path: путь к CSV файлу или папке с CSV файлами (через file dialog)
    - u_min, u_max: допустимый диапазон для компоненты U (через spinbox)
    - v_min, v_max: допустимый диапазон для компоненты V (через spinbox)
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    input_path: str  # Путь к CSV файлу или папке с CSV файлами

    # ПАРАМЕТРЫ ФИЛЬТРАЦИИ ПО U
    filter_u: bool = True  # Включить фильтрацию по U
    u_min: float = -100.0  # Минимально допустимое значение U
    u_max: float = 100.0  # Максимально допустимое значение U

    # ПАРАМЕТРЫ ФИЛЬТРАЦИИ ПО V
    filter_v: bool = True  # Включить фильтрацию по V
    v_min: float = -100.0  # Минимально допустимое значение V
    v_max: float = 100.0  # Максимально допустимое значение V

    # ПАРАМЕТРЫ ФИЛЬТРАЦИИ ПО МАГНИТУДЕ (опционально)
    filter_magnitude: bool = False  # Включить фильтрацию по магнитуде
    magnitude_min: float = 0.0  # Минимальная магнитуда
    magnitude_max: float = 1000.0  # Максимальная магнитуда

    # ПАРАМЕТРЫ ВЫХОДА
    output_folder: Optional[str] = None  # Папка для выходных файлов (None = та же папка)
    suffix: str = "_filtered"  # Суффикс для выходных файлов

    # ОПЦИОНАЛЬНЫЕ ПАРАМЕТРЫ
    recursive: bool = False  # Рекурсивный поиск файлов в подпапках
    file_pattern: str = "*.csv"  # Паттерн для поиска файлов
    enable_progress_callback: bool = True  # Включить callback для прогресса

    # GUI ПОДСКАЗКИ
    u_min_limit: float = -10000.0
    u_max_limit: float = 10000.0
    v_min_limit: float = -10000.0
    v_max_limit: float = 10000.0
    magnitude_min_limit: float = 0.0
    magnitude_max_limit: float = 10000.0

    def validate(self) -> tuple[bool, str]:
        """
        Валидация параметров.

        Returns:
            tuple[bool, str]: (success, error_message)
        """
        # Проверка входного пути
        input_path = Path(self.input_path)
        if not input_path.exists():
            return False, f"Путь не существует: {self.input_path}"

        # Проверка диапазонов U
        if self.filter_u:
            if self.u_min >= self.u_max:
                return False, f"u_min ({self.u_min}) должен быть меньше u_max ({self.u_max})"

        # Проверка диапазонов V
        if self.filter_v:
            if self.v_min >= self.v_max:
                return False, f"v_min ({self.v_min}) должен быть меньше v_max ({self.v_max})"

        # Проверка диапазонов магнитуды
        if self.filter_magnitude:
            if self.magnitude_min >= self.magnitude_max:
                return False, f"magnitude_min ({self.magnitude_min}) должен быть меньше magnitude_max ({self.magnitude_max})"
            if self.magnitude_min < 0:
                return False, f"magnitude_min должен быть >= 0: {self.magnitude_min}"

        # Проверка выходной папки
        if self.output_folder is not None:
            output_path = Path(self.output_folder)
            if not output_path.exists():
                try:
                    output_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    return False, f"Не удалось создать выходную папку: {e}"

        return True, ""


class VectorFilterExecutor:
    """
    Класс для выполнения фильтрации векторов с параметрами для GUI.

    Использование:
        1. Создать экземпляр VectorFilterExecutor
        2. Задать параметры через VectorFilterParameters
        3. (Опционально) Установить callback для прогресса
        4. Вызвать execute() для запуска фильтрации
        5. Получить результат VectorFilterResult
    """

    def __init__(self):
        """Инициализация исполнителя фильтрации векторов."""
        self.parameters: Optional[VectorFilterParameters] = None
        self._progress_callback: Optional[Callable[[VectorFilterProgress], None]] = None
        self._cancel_requested: bool = False

        logger.info("Инициализирован VectorFilterExecutor")

    def set_parameters(self, parameters: VectorFilterParameters) -> tuple[bool, str]:
        """
        Установка параметров фильтрации.

        Args:
            parameters: Параметры фильтрации

        Returns:
            tuple[bool, str]: (success, error_message)
        """
        # Валидация параметров
        success, error_msg = parameters.validate()
        if not success:
            logger.error(f"Ошибка валидации параметров: {error_msg}")
            return False, error_msg

        self.parameters = parameters

        logger.info(f"Параметры установлены:")
        if parameters.filter_u:
            logger.info(f"  U: [{parameters.u_min}, {parameters.u_max}]")
        if parameters.filter_v:
            logger.info(f"  V: [{parameters.v_min}, {parameters.v_max}]")
        if parameters.filter_magnitude:
            logger.info(f"  Magnitude: [{parameters.magnitude_min}, {parameters.magnitude_max}]")

        return True, ""

    def set_progress_callback(self, callback: Callable[[VectorFilterProgress], None]) -> None:
        """
        Установка callback функции для отслеживания прогресса.

        Args:
            callback: Функция, принимающая VectorFilterProgress
        """
        self._progress_callback = callback
        logger.debug("Установлен callback для прогресса")

    def cancel_processing(self) -> None:
        """Запрос на отмену обработки."""
        self._cancel_requested = True
        logger.info("Запрошена отмена обработки")

    def _find_csv_files(self) -> List[Path]:
        """
        Поиск CSV файлов для обработки.

        Returns:
            Список путей к CSV файлам
        """
        if self.parameters is None:
            return []

        input_path = Path(self.parameters.input_path)

        if input_path.is_file():
            # Одиночный файл
            if input_path.suffix.lower() == '.csv':
                return [input_path]
            return []

        # Папка - ищем CSV файлы
        if self.parameters.recursive:
            files = list(input_path.rglob(self.parameters.file_pattern))
        else:
            files = list(input_path.glob(self.parameters.file_pattern))

        # Исключаем уже отфильтрованные файлы
        suffix = self.parameters.suffix
        files = [f for f in files if not f.stem.endswith(suffix)]

        return sorted(files)

    def _get_output_path(self, input_path: Path) -> Path:
        """
        Получение пути к выходному файлу.

        Args:
            input_path: Путь к входному файлу

        Returns:
            Путь к выходному файлу
        """
        suffix = self.parameters.suffix if self.parameters else "_filtered"

        if self.parameters and self.parameters.output_folder:
            output_folder = Path(self.parameters.output_folder)
        else:
            output_folder = input_path.parent

        output_name = f"{input_path.stem}{suffix}{input_path.suffix}"
        return output_folder / output_name

    def _filter_vector(self, row: Dict[str, str]) -> Tuple[bool, Dict[str, int]]:
        """
        Проверка вектора на соответствие фильтрам.

        Args:
            row: Строка CSV как словарь

        Returns:
            Tuple[passed, filter_stats]: прошёл ли фильтр и статистика по фильтрам
        """
        if self.parameters is None:
            return True, {}

        stats = {
            'u_min': 0,
            'u_max': 0,
            'v_min': 0,
            'v_max': 0,
            'magnitude': 0
        }

        try:
            u = float(row.get('U', 0))
            v = float(row.get('V', 0))
            magnitude = float(row.get('Magnitude', 0)) if 'Magnitude' in row else None

            # Фильтрация по U
            if self.parameters.filter_u:
                if u < self.parameters.u_min:
                    stats['u_min'] = 1
                    return False, stats
                if u > self.parameters.u_max:
                    stats['u_max'] = 1
                    return False, stats

            # Фильтрация по V
            if self.parameters.filter_v:
                if v < self.parameters.v_min:
                    stats['v_min'] = 1
                    return False, stats
                if v > self.parameters.v_max:
                    stats['v_max'] = 1
                    return False, stats

            # Фильтрация по магнитуде
            if self.parameters.filter_magnitude and magnitude is not None:
                if magnitude < self.parameters.magnitude_min or magnitude > self.parameters.magnitude_max:
                    stats['magnitude'] = 1
                    return False, stats

            return True, stats

        except (ValueError, TypeError) as e:
            logger.warning(f"Ошибка парсинга значений: {e}")
            return False, stats

    def _process_file(self, input_path: Path) -> Tuple[VectorFilterStatistics, Optional[str], Optional[str]]:
        """
        Обработка одного CSV файла.

        Args:
            input_path: Путь к входному файлу

        Returns:
            Tuple[statistics, output_path, error]
        """
        stats = VectorFilterStatistics(
            input_vectors=0,
            output_vectors=0,
            removed_vectors=0,
            removal_percentage=0.0,
            u_min_filtered=0,
            u_max_filtered=0,
            v_min_filtered=0,
            v_max_filtered=0
        )

        output_path = self._get_output_path(input_path)

        try:
            # Чтение входного файла
            with open(input_path, 'r', encoding='utf-8') as f_in:
                reader = csv.DictReader(f_in, delimiter=';')
                fieldnames = reader.fieldnames

                if fieldnames is None:
                    return stats, None, f"Не удалось прочитать заголовки: {input_path.name}"

                # Проверка обязательных столбцов
                required_columns = ['U', 'V']
                for col in required_columns:
                    if col not in fieldnames:
                        return stats, None, f"Отсутствует столбец {col} в {input_path.name}"

                rows = list(reader)
                stats.input_vectors = len(rows)

            # Фильтрация
            filtered_rows = []
            for row in rows:
                passed, filter_stats = self._filter_vector(row)
                if passed:
                    filtered_rows.append(row)
                else:
                    stats.u_min_filtered += filter_stats.get('u_min', 0)
                    stats.u_max_filtered += filter_stats.get('u_max', 0)
                    stats.v_min_filtered += filter_stats.get('v_min', 0)
                    stats.v_max_filtered += filter_stats.get('v_max', 0)

            stats.output_vectors = len(filtered_rows)
            stats.removed_vectors = stats.input_vectors - stats.output_vectors
            stats.removal_percentage = (stats.removed_vectors / stats.input_vectors * 100) if stats.input_vectors > 0 else 0.0

            # Запись выходного файла
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                writer.writerows(filtered_rows)

            logger.info(f"Обработан {input_path.name}: {stats.input_vectors} -> {stats.output_vectors} "
                       f"(удалено {stats.removed_vectors}, {stats.removal_percentage:.1f}%)")

            return stats, str(output_path), None

        except Exception as e:
            logger.error(f"Ошибка обработки {input_path.name}: {e}")
            return stats, None, str(e)

    def execute(self) -> VectorFilterResult:
        """
        Выполнение фильтрации векторов.

        Returns:
            VectorFilterResult с результатами фильтрации
        """
        if self.parameters is None:
            logger.error("Параметры не установлены")
            return VectorFilterResult(
                success=False,
                total_files_processed=0,
                total_vectors_input=0,
                total_vectors_output=0,
                vectors_removed=0,
                removal_percentage=0.0,
                errors=["Параметры не установлены"],
                warnings=[],
                output_files=[]
            )

        self._cancel_requested = False

        # Поиск файлов
        csv_files = self._find_csv_files()

        if not csv_files:
            return VectorFilterResult(
                success=False,
                total_files_processed=0,
                total_vectors_input=0,
                total_vectors_output=0,
                vectors_removed=0,
                removal_percentage=0.0,
                errors=["Не найдены CSV файлы для обработки"],
                warnings=[],
                output_files=[]
            )

        logger.info("=" * 60)
        logger.info("ЗАПУСК ФИЛЬТРАЦИИ ВЕКТОРОВ")
        logger.info(f"Входной путь: {self.parameters.input_path}")
        logger.info(f"Найдено файлов: {len(csv_files)}")
        if self.parameters.filter_u:
            logger.info(f"Фильтр U: [{self.parameters.u_min}, {self.parameters.u_max}]")
        if self.parameters.filter_v:
            logger.info(f"Фильтр V: [{self.parameters.v_min}, {self.parameters.v_max}]")
        if self.parameters.filter_magnitude:
            logger.info(f"Фильтр Magnitude: [{self.parameters.magnitude_min}, {self.parameters.magnitude_max}]")
        logger.info("=" * 60)

        total_files = len(csv_files)
        processed = 0
        total_input = 0
        total_output = 0
        errors = []
        warnings = []
        output_files = []

        for idx, csv_file in enumerate(csv_files):
            if self._cancel_requested:
                logger.info("Обработка отменена")
                break

            # Прогресс
            if self._progress_callback and self.parameters.enable_progress_callback:
                progress = VectorFilterProgress(
                    current_file=csv_file.name,
                    total_files=total_files,
                    processed_files=idx,
                    percentage=(idx / total_files) * 100,
                    message=f"Обработка {csv_file.name}"
                )
                self._progress_callback(progress)

            # Обработка файла
            stats, output_path, error = self._process_file(csv_file)

            if error:
                errors.append(f"{csv_file.name}: {error}")
            else:
                processed += 1
                total_input += stats.input_vectors
                total_output += stats.output_vectors
                if output_path:
                    output_files.append(output_path)

        # Итоговый прогресс
        if self._progress_callback and self.parameters.enable_progress_callback and not self._cancel_requested:
            progress = VectorFilterProgress(
                current_file="",
                total_files=total_files,
                processed_files=total_files,
                percentage=100.0,
                message="Завершено"
            )
            self._progress_callback(progress)

        vectors_removed = total_input - total_output
        removal_percentage = (vectors_removed / total_input * 100) if total_input > 0 else 0.0
        success = len(errors) == 0 and processed > 0

        logger.info("=" * 60)
        logger.info("ЗАВЕРШЕНИЕ ФИЛЬТРАЦИИ ВЕКТОРОВ")
        logger.info(f"Успешно: {success}")
        logger.info(f"Обработано файлов: {processed}/{total_files}")
        logger.info(f"Всего векторов на входе: {total_input}")
        logger.info(f"Всего векторов на выходе: {total_output}")
        logger.info(f"Удалено векторов: {vectors_removed} ({removal_percentage:.1f}%)")
        logger.info(f"Ошибок: {len(errors)}")
        logger.info("=" * 60)

        return VectorFilterResult(
            success=success,
            total_files_processed=processed,
            total_vectors_input=total_input,
            total_vectors_output=total_output,
            vectors_removed=vectors_removed,
            removal_percentage=removal_percentage,
            errors=errors,
            warnings=warnings,
            output_files=output_files
        )

    def get_preview(self, file_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Получение предварительного просмотра фильтрации.

        Args:
            file_path: Путь к файлу для предпросмотра (None = первый найденный)

        Returns:
            Словарь со статистикой предпросмотра или None
        """
        if self.parameters is None:
            return None

        if file_path:
            csv_file = Path(file_path)
        else:
            csv_files = self._find_csv_files()
            if not csv_files:
                return None
            csv_file = csv_files[0]

        if not csv_file.exists():
            return None

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                rows = list(reader)

            total = len(rows)
            would_pass = 0
            would_fail = 0

            u_values = []
            v_values = []

            for row in rows:
                passed, _ = self._filter_vector(row)
                if passed:
                    would_pass += 1
                else:
                    would_fail += 1

                try:
                    u_values.append(float(row.get('U', 0)))
                    v_values.append(float(row.get('V', 0)))
                except (ValueError, TypeError):
                    pass

            import numpy as np
            u_arr = np.array(u_values) if u_values else np.array([0])
            v_arr = np.array(v_values) if v_values else np.array([0])

            return {
                'file': csv_file.name,
                'total_vectors': total,
                'would_pass': would_pass,
                'would_fail': would_fail,
                'removal_percentage': (would_fail / total * 100) if total > 0 else 0,
                'u_min': float(u_arr.min()),
                'u_max': float(u_arr.max()),
                'u_mean': float(u_arr.mean()),
                'u_std': float(u_arr.std()),
                'v_min': float(v_arr.min()),
                'v_max': float(v_arr.max()),
                'v_mean': float(v_arr.mean()),
                'v_std': float(v_arr.std())
            }

        except Exception as e:
            logger.error(f"Ошибка предпросмотра: {e}")
            return None


def run_vector_filter(
    input_path: str,
    u_min: float = -100.0,
    u_max: float = 100.0,
    v_min: float = -100.0,
    v_max: float = 100.0,
    filter_u: bool = True,
    filter_v: bool = True,
    output_folder: Optional[str] = None,
    suffix: str = "_filtered",
    progress_callback: Optional[Callable] = None
) -> VectorFilterResult:
    """
    Удобная функция для запуска фильтрации векторов без создания объектов.

    Args:
        input_path: Путь к CSV файлу или папке
        u_min: Минимально допустимое значение U
        u_max: Максимально допустимое значение U
        v_min: Минимально допустимое значение V
        v_max: Максимально допустимое значение V
        filter_u: Включить фильтрацию по U
        filter_v: Включить фильтрацию по V
        output_folder: Папка для выходных файлов (None = та же папка)
        suffix: Суффикс для выходных файлов
        progress_callback: Callback функция для прогресса

    Returns:
        VectorFilterResult с результатами

    Example:
        >>> result = run_vector_filter(
        ...     input_path="path/to/vectors.csv",
        ...     u_min=-50, u_max=50,
        ...     v_min=-50, v_max=50
        ... )
        >>> print(f"Удалено: {result.vectors_removed} ({result.removal_percentage:.1f}%)")
    """
    params = VectorFilterParameters(
        input_path=input_path,
        filter_u=filter_u,
        u_min=u_min,
        u_max=u_max,
        filter_v=filter_v,
        v_min=v_min,
        v_max=v_max,
        output_folder=output_folder,
        suffix=suffix,
        enable_progress_callback=progress_callback is not None
    )

    executor = VectorFilterExecutor()

    success, error_msg = executor.set_parameters(params)
    if not success:
        logger.error(f"Ошибка установки параметров: {error_msg}")
        return VectorFilterResult(
            success=False,
            total_files_processed=0,
            total_vectors_input=0,
            total_vectors_output=0,
            vectors_removed=0,
            removal_percentage=0.0,
            errors=[error_msg],
            warnings=[],
            output_files=[]
        )

    if progress_callback:
        executor.set_progress_callback(progress_callback)

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
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ С GUI - ФИЛЬТРАЦИЯ ВЕКТОРОВ")
    print("=" * 60)

    # === ШАГ 1: Задание параметров (из GUI элементов) ===
    parameters = VectorFilterParameters(
        input_path=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\LucasKanade_2000\cam_1",
        filter_u=True,
        u_min=-50.0,
        u_max=50.0,
        filter_v=True,
        v_min=-50.0,
        v_max=50.0,
        filter_magnitude=False,
        suffix="_filtered",
        enable_progress_callback=True
    )

    print(f"\nПараметры:")
    print(f"  Входной путь: {parameters.input_path}")
    print(f"  Фильтр U: [{parameters.u_min}, {parameters.u_max}]")
    print(f"  Фильтр V: [{parameters.v_min}, {parameters.v_max}]")
    print(f"  Суффикс: {parameters.suffix}")

    # === ШАГ 2: Создание исполнителя ===
    executor = VectorFilterExecutor()

    # === ШАГ 3: Валидация и установка параметров ===
    success, error_msg = executor.set_parameters(parameters)
    if not success:
        print(f"\nОШИБКА: {error_msg}")
        return

    print("\n[OK] Параметры валидны")

    # === ШАГ 4: Предварительный просмотр ===
    print("\nПредварительный просмотр...")
    preview = executor.get_preview()
    if preview:
        print(f"\nПредпросмотр для {preview['file']}:")
        print(f"  Всего векторов: {preview['total_vectors']}")
        print(f"  Пройдёт фильтр: {preview['would_pass']}")
        print(f"  Будет удалено: {preview['would_fail']} ({preview['removal_percentage']:.1f}%)")
        print(f"  U: [{preview['u_min']:.3f}, {preview['u_max']:.3f}], mean={preview['u_mean']:.3f}")
        print(f"  V: [{preview['v_min']:.3f}, {preview['v_max']:.3f}], mean={preview['v_mean']:.3f}")

    # === ШАГ 5: Установка callback для прогресса ===
    def progress_callback(progress: VectorFilterProgress):
        """Callback для обновления GUI."""
        print(f"  {progress.percentage:.1f}% - {progress.message}")

    executor.set_progress_callback(progress_callback)

    # === ШАГ 6: Выполнение фильтрации ===
    print("\nЗапуск фильтрации...")
    result = executor.execute()

    # === ШАГ 7: Обработка результата ===
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Успешно: {result.success}")
    print(f"Обработано файлов: {result.total_files_processed}")
    print(f"Векторов на входе: {result.total_vectors_input}")
    print(f"Векторов на выходе: {result.total_vectors_output}")
    print(f"Удалено: {result.vectors_removed} ({result.removal_percentage:.1f}%)")
    print(f"Ошибок: {len(result.errors)}")

    if result.output_files:
        print(f"\nВыходные файлы:")
        for f in result.output_files[:5]:
            print(f"  - {f}")
        if len(result.output_files) > 5:
            print(f"  ... и ещё {len(result.output_files) - 5} файлов")

    if result.errors:
        print("\nОшибки:")
        for error in result.errors[:5]:
            print(f"  - {error}")

    print("=" * 60)


if __name__ == "__main__":
    example_gui_usage()
