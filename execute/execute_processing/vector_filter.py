"""
Модуль фильтрации векторов по значениям U и V для GUI ParticleAnalysis.

Этот модуль позволяет фильтровать CSV файл с векторами оптического потока
по допустимым диапазонам значений компонент скорости U и V.

Входной формат CSV (разделитель ;):
    X0;Y0;dx;dy;L;Diameter;Area

Выходной формат: исходный_файл_filtered.csv

Автор: ParticleAnalysis Team
Версия: 1.1
"""

import sys
from pathlib import Path
from dataclasses import dataclass
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
class VectorFilterResult:
    """Результат фильтрации векторов."""
    success: bool
    input_vectors: int
    output_vectors: int
    vectors_removed: int
    removal_percentage: float
    u_min_filtered: int  # Отфильтровано по U < u_min
    u_max_filtered: int  # Отфильтровано по U > u_max
    v_min_filtered: int  # Отфильтровано по V < v_min
    v_max_filtered: int  # Отфильтровано по V > v_max
    errors: List[str]
    output_file: str


@dataclass
class VectorFilterParameters:
    """
    Параметры фильтрации векторов для GUI.

    Все параметры должны устанавливаться через GUI элементы:
    - input_file: путь к CSV файлу (через file dialog)
    - u_min, u_max: допустимый диапазон для компоненты U (через spinbox)
    - v_min, v_max: допустимый диапазон для компоненты V (через spinbox)
    """
    # ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ
    input_file: str  # Путь к CSV файлу

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
    output_folder: Optional[str] = None  # Папка для выходного файла (None = та же папка)
    suffix: str = "_filtered"  # Суффикс для выходного файла

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
        # Проверка входного файла
        input_path = Path(self.input_file)
        if not input_path.exists():
            return False, f"Файл не существует: {self.input_file}"

        if not input_path.is_file():
            return False, f"Путь не является файлом: {self.input_file}"

        if input_path.suffix.lower() != '.csv':
            return False, f"Файл должен иметь расширение .csv: {self.input_file}"

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
        3. Вызвать execute() для запуска фильтрации
        4. Получить результат VectorFilterResult
    """

    # Допустимые имена колонок для компонент вектора
    U_COLUMN_NAMES = ('U', 'dx')
    V_COLUMN_NAMES = ('V', 'dy')
    MAGNITUDE_COLUMN_NAMES = ('Magnitude', 'L')

    def __init__(self):
        """Инициализация исполнителя фильтрации векторов."""
        self.parameters: Optional[VectorFilterParameters] = None
        self._u_col: Optional[str] = None  # Фактическое имя колонки U/dx
        self._v_col: Optional[str] = None  # Фактическое имя колонки V/dy
        self._mag_col: Optional[str] = None  # Фактическое имя колонки Magnitude/L

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
        logger.info(f"  Входной файл: {parameters.input_file}")
        if parameters.filter_u:
            logger.info(f"  U: [{parameters.u_min}, {parameters.u_max}]")
        if parameters.filter_v:
            logger.info(f"  V: [{parameters.v_min}, {parameters.v_max}]")
        if parameters.filter_magnitude:
            logger.info(f"  Magnitude: [{parameters.magnitude_min}, {parameters.magnitude_max}]")

        return True, ""

    def _get_output_path(self) -> Path:
        """
        Получение пути к выходному файлу.

        Returns:
            Путь к выходному файлу
        """
        input_path = Path(self.parameters.input_file)
        suffix = self.parameters.suffix if self.parameters else "_filtered"

        if self.parameters and self.parameters.output_folder:
            output_folder = Path(self.parameters.output_folder)
        else:
            output_folder = input_path.parent

        output_name = f"{input_path.stem}{suffix}{input_path.suffix}"
        return output_folder / output_name

    def _detect_columns(self, fieldnames: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Определение имён колонок U, V и Magnitude в CSV файле.

        Поддерживаются имена: U/V, dx/dy, Magnitude/L.
        Если точного совпадения нет для U/V, используются 3-я и 4-я колонки (индексы 2, 3).

        Args:
            fieldnames: Список имён колонок

        Returns:
            Tuple[u_col, v_col, mag_col]: имена найденных колонок
        """
        u_col = None
        v_col = None
        mag_col = None

        for name in self.U_COLUMN_NAMES:
            if name in fieldnames:
                u_col = name
                break

        for name in self.V_COLUMN_NAMES:
            if name in fieldnames:
                v_col = name
                break

        for name in self.MAGNITUDE_COLUMN_NAMES:
            if name in fieldnames:
                mag_col = name
                break

        # Фоллбэк на 3-ю и 4-ю колонки
        if u_col is None and len(fieldnames) >= 3:
            u_col = fieldnames[2]
            logger.info(f"Колонка U не найдена, используется 3-я колонка: '{u_col}'")
        if v_col is None and len(fieldnames) >= 4:
            v_col = fieldnames[3]
            logger.info(f"Колонка V не найдена, используется 4-я колонка: '{v_col}'")

        return u_col, v_col, mag_col

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
            u = float(row.get(self._u_col, 0))
            v = float(row.get(self._v_col, 0))
            magnitude = float(row.get(self._mag_col, 0)) if self._mag_col and self._mag_col in row else None

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
                input_vectors=0,
                output_vectors=0,
                vectors_removed=0,
                removal_percentage=0.0,
                u_min_filtered=0,
                u_max_filtered=0,
                v_min_filtered=0,
                v_max_filtered=0,
                errors=["Параметры не установлены"],
                output_file=""
            )

        input_path = Path(self.parameters.input_file)
        output_path = self._get_output_path()

        logger.info("=" * 60)
        logger.info("ЗАПУСК ФИЛЬТРАЦИИ ВЕКТОРОВ")
        logger.info(f"Входной файл: {input_path}")
        logger.info(f"Выходной файл: {output_path}")
        if self.parameters.filter_u:
            logger.info(f"Фильтр U: [{self.parameters.u_min}, {self.parameters.u_max}]")
        if self.parameters.filter_v:
            logger.info(f"Фильтр V: [{self.parameters.v_min}, {self.parameters.v_max}]")
        if self.parameters.filter_magnitude:
            logger.info(f"Фильтр Magnitude: [{self.parameters.magnitude_min}, {self.parameters.magnitude_max}]")
        logger.info("=" * 60)

        # Статистика
        input_vectors = 0
        output_vectors = 0
        u_min_filtered = 0
        u_max_filtered = 0
        v_min_filtered = 0
        v_max_filtered = 0
        errors = []

        try:
            # Чтение входного файла
            with open(input_path, 'r', encoding='utf-8') as f_in:
                reader = csv.DictReader(f_in, delimiter=';')
                fieldnames = reader.fieldnames

                if fieldnames is None:
                    errors.append(f"Не удалось прочитать заголовки файла")
                    return VectorFilterResult(
                        success=False,
                        input_vectors=0,
                        output_vectors=0,
                        vectors_removed=0,
                        removal_percentage=0.0,
                        u_min_filtered=0,
                        u_max_filtered=0,
                        v_min_filtered=0,
                        v_max_filtered=0,
                        errors=errors,
                        output_file=""
                    )

                # Определение колонок U, V и Magnitude
                self._u_col, self._v_col, self._mag_col = self._detect_columns(fieldnames)

                if self._u_col is None or self._v_col is None:
                    missing = []
                    if self._u_col is None:
                        missing.append("U/dx")
                    if self._v_col is None:
                        missing.append("V/dy")
                    errors.append(
                        f"Не удалось определить колонки: {', '.join(missing)}. "
                        f"Ожидаются U/V или dx/dy (или минимум 4 колонки)"
                    )
                    return VectorFilterResult(
                        success=False,
                        input_vectors=0,
                        output_vectors=0,
                        vectors_removed=0,
                        removal_percentage=0.0,
                        u_min_filtered=0,
                        u_max_filtered=0,
                        v_min_filtered=0,
                        v_max_filtered=0,
                        errors=errors,
                        output_file=""
                    )

                logger.info(f"Колонки: U='{self._u_col}', V='{self._v_col}', Magnitude='{self._mag_col}'")

                rows = list(reader)
                input_vectors = len(rows)

            # Фильтрация
            filtered_rows = []
            for row in rows:
                passed, filter_stats = self._filter_vector(row)
                if passed:
                    filtered_rows.append(row)
                else:
                    u_min_filtered += filter_stats.get('u_min', 0)
                    u_max_filtered += filter_stats.get('u_max', 0)
                    v_min_filtered += filter_stats.get('v_min', 0)
                    v_max_filtered += filter_stats.get('v_max', 0)

            output_vectors = len(filtered_rows)
            vectors_removed = input_vectors - output_vectors
            removal_percentage = (vectors_removed / input_vectors * 100) if input_vectors > 0 else 0.0

            # Запись выходного файла
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                writer.writerows(filtered_rows)

            logger.info("=" * 60)
            logger.info("ЗАВЕРШЕНИЕ ФИЛЬТРАЦИИ ВЕКТОРОВ")
            logger.info(f"Векторов на входе: {input_vectors}")
            logger.info(f"Векторов на выходе: {output_vectors}")
            logger.info(f"Удалено: {vectors_removed} ({removal_percentage:.1f}%)")
            logger.info(f"  - по U < {self.parameters.u_min}: {u_min_filtered}")
            logger.info(f"  - по U > {self.parameters.u_max}: {u_max_filtered}")
            logger.info(f"  - по V < {self.parameters.v_min}: {v_min_filtered}")
            logger.info(f"  - по V > {self.parameters.v_max}: {v_max_filtered}")
            logger.info(f"Выходной файл: {output_path}")
            logger.info("=" * 60)

            return VectorFilterResult(
                success=True,
                input_vectors=input_vectors,
                output_vectors=output_vectors,
                vectors_removed=vectors_removed,
                removal_percentage=removal_percentage,
                u_min_filtered=u_min_filtered,
                u_max_filtered=u_max_filtered,
                v_min_filtered=v_min_filtered,
                v_max_filtered=v_max_filtered,
                errors=[],
                output_file=str(output_path)
            )

        except Exception as e:
            logger.error(f"Ошибка обработки: {e}")
            errors.append(str(e))
            return VectorFilterResult(
                success=False,
                input_vectors=input_vectors,
                output_vectors=0,
                vectors_removed=0,
                removal_percentage=0.0,
                u_min_filtered=0,
                u_max_filtered=0,
                v_min_filtered=0,
                v_max_filtered=0,
                errors=errors,
                output_file=""
            )

    def get_preview(self) -> Optional[Dict[str, Any]]:
        """
        Получение предварительного просмотра фильтрации.

        Returns:
            Словарь со статистикой предпросмотра или None
        """
        if self.parameters is None:
            return None

        input_path = Path(self.parameters.input_file)

        if not input_path.exists():
            return None

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                fieldnames = reader.fieldnames or []

                # Определение колонок, если ещё не определены
                if self._u_col is None or self._v_col is None:
                    self._u_col, self._v_col, self._mag_col = self._detect_columns(fieldnames)

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
                    u_values.append(float(row.get(self._u_col, 0)))
                    v_values.append(float(row.get(self._v_col, 0)))
                except (ValueError, TypeError):
                    pass

            import numpy as np
            u_arr = np.array(u_values) if u_values else np.array([0])
            v_arr = np.array(v_values) if v_values else np.array([0])

            return {
                'file': input_path.name,
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
    input_file: str,
    u_min: float = -100.0,
    u_max: float = 100.0,
    v_min: float = -100.0,
    v_max: float = 100.0,
    filter_u: bool = True,
    filter_v: bool = True,
    output_folder: Optional[str] = None,
    suffix: str = "_filtered"
) -> VectorFilterResult:
    """
    Удобная функция для запуска фильтрации векторов без создания объектов.

    Args:
        input_file: Путь к CSV файлу
        u_min: Минимально допустимое значение U
        u_max: Максимально допустимое значение U
        v_min: Минимально допустимое значение V
        v_max: Максимально допустимое значение V
        filter_u: Включить фильтрацию по U
        filter_v: Включить фильтрацию по V
        output_folder: Папка для выходного файла (None = та же папка)
        suffix: Суффикс для выходного файла

    Returns:
        VectorFilterResult с результатами

    Example:
        >>> result = run_vector_filter(
        ...     input_file="path/to/vectors.csv",
        ...     u_min=-50, u_max=50,
        ...     v_min=-50, v_max=50
        ... )
        >>> print(f"Удалено: {result.vectors_removed} ({result.removal_percentage:.1f}%)")
    """
    params = VectorFilterParameters(
        input_file=input_file,
        filter_u=filter_u,
        u_min=u_min,
        u_max=u_max,
        filter_v=filter_v,
        v_min=v_min,
        v_max=v_max,
        output_folder=output_folder,
        suffix=suffix
    )

    executor = VectorFilterExecutor()

    success, error_msg = executor.set_parameters(params)
    if not success:
        logger.error(f"Ошибка установки параметров: {error_msg}")
        return VectorFilterResult(
            success=False,
            input_vectors=0,
            output_vectors=0,
            vectors_removed=0,
            removal_percentage=0.0,
            u_min_filtered=0,
            u_max_filtered=0,
            v_min_filtered=0,
            v_max_filtered=0,
            errors=[error_msg],
            output_file=""
        )

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
        input_file=r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\PTV_2500\cam_1_pairs_sum.csv",
        filter_u=True,
        u_min=0,
        u_max=30,
        filter_v=True,
        v_min=-5.0,
        v_max=5.0,
        filter_magnitude=False,
        suffix="_filtered"
    )

    print(f"\nПараметры:")
    print(f"  Входной файл: {parameters.input_file}")
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

    # === ШАГ 5: Выполнение фильтрации ===
    print("\nЗапуск фильтрации...")
    result = executor.execute()

    # === ШАГ 6: Обработка результата ===
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Успешно: {result.success}")
    print(f"Векторов на входе: {result.input_vectors}")
    print(f"Векторов на выходе: {result.output_vectors}")
    print(f"Удалено: {result.vectors_removed} ({result.removal_percentage:.1f}%)")
    print(f"  - по U < u_min: {result.u_min_filtered}")
    print(f"  - по U > u_max: {result.u_max_filtered}")
    print(f"  - по V < v_min: {result.v_min_filtered}")
    print(f"  - по V > v_max: {result.v_max_filtered}")
    print(f"Выходной файл: {result.output_file}")

    if result.errors:
        print("\nОшибки:")
        for error in result.errors:
            print(f"  - {error}")

    print("=" * 60)


if __name__ == "__main__":
    example_gui_usage()
