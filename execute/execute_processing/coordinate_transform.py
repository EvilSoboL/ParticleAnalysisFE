"""
Модуль преобразования координат из пиксельных единиц в физические (метры, м/с).

Входной формат CSV (разделитель ;):
    Формат 1 (после усреднения): X_center;Y_center;dx_avg;dy_avg;L_avg;count
    Формат 2 (после PTV/фильтрации): X0;Y0;dx;dy;L;Diameter;Area

Преобразования:
    1. Сдвиг к началу координат: X_rel = X - X_origin, Y_rel = Y - Y_origin
    2. Поворот против часовой стрелки: X_rot = X_rel * cos(θ) - Y_rel * sin(θ),
                                       Y_rot = X_rel * sin(θ) + Y_rel * cos(θ)
    3. Масштабирование в мм: X_mm = X_rot * scale * 1000, Y_mm = Y_rot * scale * 1000
    4. Поворот векторов скорости: dx_rot = dx * cos(θ) - dy * sin(θ),
                                  dy_rot = dx * sin(θ) + dy * cos(θ)
    5. Масштабирование скоростей в м/с: dx_ms = dx_rot * scale / dt,
                                        dy_ms = dy_rot * scale / dt,
                                        L_ms = L * scale / dt

Выходной формат:
    Формат 1: X_mm;Y_mm;dx_ms;dy_ms;L_ms;count
    Формат 2: X_mm;Y_mm;dx_ms;dy_ms;L_ms
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import logging
import csv
import math

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CoordinateTransformResult:
    """Результат преобразования координат."""
    success: bool
    input_rows: int
    output_rows: int
    errors: List[str]
    output_file: str


@dataclass
class CoordinateTransformParameters:
    """Параметры преобразования координат."""
    input_file: str
    x_origin: float = 0.0        # Начало координат X (пиксели)
    y_origin: float = 0.0        # Начало координат Y (пиксели)
    rotation_angle: float = 0.0  # Угол поворота против часовой стрелки (градусы)
    scale: float = 0.001         # Масштаб: метры на пиксель
    dt: float = 0.001            # Временной интервал (секунды)
    output_folder: Optional[str] = None
    suffix: str = "_transformed"

    def validate(self) -> tuple[bool, str]:
        input_path = Path(self.input_file)
        if not input_path.exists():
            return False, f"Файл не существует: {self.input_file}"
        if not input_path.is_file():
            return False, f"Путь не является файлом: {self.input_file}"
        if input_path.suffix.lower() != '.csv':
            return False, f"Файл должен иметь расширение .csv: {self.input_file}"
        if self.scale <= 0:
            return False, f"scale должен быть > 0: {self.scale}"
        if self.dt <= 0:
            return False, f"dt должен быть > 0: {self.dt}"
        if self.output_folder is not None:
            output_path = Path(self.output_folder)
            if not output_path.exists():
                try:
                    output_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    return False, f"Не удалось создать выходную папку: {e}"
        return True, ""


class CoordinateTransformExecutor:
    """Исполнитель преобразования координат из пикселей в физические единицы."""

    # Формат 1: после усреднения
    FORMAT_AVG_COLS = ('X_center', 'Y_center', 'dx_avg', 'dy_avg', 'L_avg', 'count')
    # Формат 2: после PTV/фильтрации
    FORMAT_RAW_COLS = ('X0', 'Y0', 'dx', 'dy', 'L')

    def __init__(self):
        self.parameters: Optional[CoordinateTransformParameters] = None
        logger.info("Инициализирован CoordinateTransformExecutor")

    def set_parameters(self, parameters: CoordinateTransformParameters) -> tuple[bool, str]:
        success, error_msg = parameters.validate()
        if not success:
            logger.error(f"Ошибка валидации параметров: {error_msg}")
            return False, error_msg
        self.parameters = parameters
        logger.info(f"Параметры установлены:")
        logger.info(f"  Входной файл: {parameters.input_file}")
        logger.info(f"  X_origin: {parameters.x_origin}, Y_origin: {parameters.y_origin}")
        logger.info(f"  Rotation angle: {parameters.rotation_angle}°")
        logger.info(f"  Scale: {parameters.scale} м/пиксель")
        logger.info(f"  dt: {parameters.dt} с")
        return True, ""

    def _get_output_path(self) -> Path:
        input_path = Path(self.parameters.input_file)
        suffix = self.parameters.suffix
        if self.parameters.output_folder:
            output_folder = Path(self.parameters.output_folder)
        else:
            output_folder = input_path.parent
        output_name = f"{input_path.stem}{suffix}{input_path.suffix}"
        return output_folder / output_name

    def _detect_format(self, fieldnames: List[str]) -> Optional[str]:
        """Определение формата CSV: 'avg' или 'raw'."""
        if 'X_center' in fieldnames and 'Y_center' in fieldnames:
            return 'avg'
        if 'X0' in fieldnames and 'Y0' in fieldnames:
            return 'raw'
        return None

    def execute(self) -> CoordinateTransformResult:
        if self.parameters is None:
            return CoordinateTransformResult(
                success=False, input_rows=0, output_rows=0,
                errors=["Параметры не установлены"], output_file=""
            )

        input_path = Path(self.parameters.input_file)
        scale = self.parameters.scale
        dt = self.parameters.dt
        x_origin = self.parameters.x_origin
        y_origin = self.parameters.y_origin
        rotation_angle = self.parameters.rotation_angle

        # Преобразование угла в радианы и вычисление cos/sin
        theta_rad = math.radians(rotation_angle)
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)

        logger.info("=" * 60)
        logger.info("ЗАПУСК ПРЕОБРАЗОВАНИЯ КООРДИНАТ")
        logger.info(f"Входной файл: {input_path}")
        logger.info("=" * 60)

        errors = []
        input_rows = 0
        output_rows = 0

        try:
            with open(input_path, 'r', encoding='utf-8') as f_in:
                reader = csv.DictReader(f_in, delimiter=';')
                fieldnames = reader.fieldnames

                if fieldnames is None:
                    return CoordinateTransformResult(
                        success=False, input_rows=0, output_rows=0,
                        errors=["Не удалось прочитать заголовки файла"], output_file=""
                    )

                fmt = self._detect_format(fieldnames)
                if fmt is None:
                    return CoordinateTransformResult(
                        success=False, input_rows=0, output_rows=0,
                        errors=[
                            f"Не удалось определить формат CSV. "
                            f"Ожидаются колонки X_center/Y_center или X0/Y0. "
                            f"Найдены: {fieldnames}"
                        ],
                        output_file=""
                    )

                logger.info(f"Определён формат: {fmt}")
                rows = list(reader)
                input_rows = len(rows)

            # Определение колонок в зависимости от формата
            if fmt == 'avg':
                x_col, y_col = 'X_center', 'Y_center'
                dx_col, dy_col, l_col = 'dx_avg', 'dy_avg', 'L_avg'
                has_count = 'count' in fieldnames
                out_fieldnames = ['X_mm', 'Y_mm', 'dx_ms', 'dy_ms', 'L_ms']
                if has_count:
                    out_fieldnames.append('count')
            else:  # raw
                x_col, y_col = 'X0', 'Y0'
                dx_col, dy_col, l_col = 'dx', 'dy', 'L'
                has_count = False
                out_fieldnames = ['X_mm', 'Y_mm', 'dx_ms', 'dy_ms', 'L_ms']

            # Преобразование
            out_rows = []
            for i, row in enumerate(rows):
                try:
                    # 1. Сдвиг к началу координат
                    x_rel = float(row[x_col]) - x_origin
                    y_rel = float(row[y_col]) - y_origin

                    # 2. Поворот координат против часовой стрелки
                    x_rot = x_rel * cos_theta - y_rel * sin_theta
                    y_rot = x_rel * sin_theta + y_rel * cos_theta

                    # 3. Масштабирование в мм
                    x_mm = x_rot * scale * 1000.0
                    y_mm = y_rot * scale * 1000.0

                    # 4. Поворот векторов скорости
                    dx = float(row[dx_col])
                    dy = float(row[dy_col])
                    dx_rot = dx * cos_theta - dy * sin_theta
                    dy_rot = dx * sin_theta + dy * cos_theta

                    # 5. Масштабирование скоростей в м/с
                    dx_ms = dx_rot * scale / dt
                    dy_ms = dy_rot * scale / dt
                    l_ms = float(row[l_col]) * scale / dt

                    out_row = {
                        'X_mm': f"{x_mm:.4f}",
                        'Y_mm': f"{y_mm:.4f}",
                        'dx_ms': f"{dx_ms:.6f}",
                        'dy_ms': f"{dy_ms:.6f}",
                        'L_ms': f"{l_ms:.6f}",
                    }
                    if has_count:
                        out_row['count'] = row.get('count', '0')

                    out_rows.append(out_row)
                    output_rows += 1
                except (ValueError, KeyError) as e:
                    errors.append(f"Строка {i + 1}: {e}")

            output_path = self._get_output_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=out_fieldnames, delimiter=';')
                writer.writeheader()
                writer.writerows(out_rows)

            logger.info("=" * 60)
            logger.info("ЗАВЕРШЕНИЕ ПРЕОБРАЗОВАНИЯ КООРДИНАТ")
            logger.info(f"Строк на входе: {input_rows}")
            logger.info(f"Строк на выходе: {output_rows}")
            logger.info(f"Выходной файл: {output_path}")
            logger.info("=" * 60)

            return CoordinateTransformResult(
                success=True,
                input_rows=input_rows,
                output_rows=output_rows,
                errors=errors,
                output_file=str(output_path)
            )

        except Exception as e:
            logger.error(f"Ошибка обработки: {e}")
            errors.append(str(e))
            return CoordinateTransformResult(
                success=False,
                input_rows=input_rows,
                output_rows=output_rows,
                errors=errors,
                output_file=""
            )
