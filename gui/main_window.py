"""
GUI приложение ParticleAnalysis на PyQt5.

Четыре вкладки:
1. Sort + Binarize — сортировка и бинаризация
2. PTV Analysis — PTV анализ
3. PTV Processing — обработка результатов PTV (фильтрация, усреднение, визуализация)
4. Coordinate Transform — перевод координат и скоростей из пикселей в метры/м/с
"""

import sys
import traceback
from pathlib import Path
import cv2
import csv
import math
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Добавление корня проекта в sys.path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QProgressBar, QTextEdit, QGroupBox, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QDialog, QSplitter
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRectF, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont

from execute.execute_filter.execute_sort_binarize import (
    SortBinarizeExecutor, SortBinarizeParameters
)
from src.data_processing.experiment_preprocess import (
    build_output_base_folder,
    default_processed_root,
    scan_experiment_root,
)
from execute.execute_analysis.execute_ptv_analysis import (
    PTVExecutor, PTVParameters
)
from src.visualization.one_to_one_visualization import PTVVisualizer
from src.visualization.ptv_viewer import scan_ptv_pairs
from execute.execute_processing.vector_filter import (
    VectorFilterExecutor, VectorFilterParameters
)
from execute.execute_processing.vector_average import (
    VectorAverageExecutor, VectorAverageParameters
)
from execute.execute_processing.vector_plot import (
    VectorPlotExecutor, VectorPlotParameters
)
from execute.execute_processing.coordinate_transform import (
    CoordinateTransformExecutor, CoordinateTransformParameters
)


_EXPERIMENT_STATUS_LABELS = {
    "Skipped": "Пропущен",
    "Error": "Ошибка",
    "Warning": "Предупреждение",
    "OK": "OK",
}

_EXPERIMENT_MESSAGE_PREFIXES = [
    ("Root folder does not exist:", "Корневая папка не существует:"),
    ("Path is not a folder:", "Путь не является папкой:"),
    ("No experiment*.afxml files found in", "Не найдены файлы experiment*.afxml в"),
    ("Cannot parse XML:", "Не удалось разобрать XML:"),
    ("Cannot read file:", "Не удалось прочитать файл:"),
    ("node_record not found", "node_record не найден"),
    ("data link not found; using afxml file stem", "data link не найден; используется имя afxml файла"),
    ("Source folder does not exist:", "Исходная папка не существует:"),
    ("Source path is not a folder:", "Исходный путь не является папкой:"),
    ("No PNG files found", "PNG файлы не найдены"),
    ("PNG count is not divisible by 4:", "Количество PNG не делится на 4:"),
    ("Experiment name looks generic", "Название эксперимента выглядит слишком общим"),
]


def _experiment_status_label(status: str) -> str:
    return _EXPERIMENT_STATUS_LABELS.get(status, status)


def _translate_experiment_message(message: str) -> str:
    for english, russian in _EXPERIMENT_MESSAGE_PREFIXES:
        if message.startswith(english):
            return f"{russian}{message[len(english):]}"
    return message


def _experiment_issue_text(record) -> str:
    return "; ".join(_translate_experiment_message(item) for item in record.errors + record.warnings)


def _set_tooltip(text: str, *widgets) -> None:
    for widget in widgets:
        widget.setToolTip(text)


def _set_header_tooltips(table: QTableWidget, tooltips: dict[int, str]) -> None:
    for column, tooltip in tooltips.items():
        item = table.horizontalHeaderItem(column)
        if item is not None:
            item.setToolTip(tooltip)


# ---------------------------------------------------------------------------
# WorkerThread
# ---------------------------------------------------------------------------
class WorkerThread(QThread):
    """Универсальный worker для фоновых задач."""
    progress = pyqtSignal(float, str)   # percentage, message
    finished = pyqtSignal(object)       # result object
    error = pyqtSignal(str)             # error message

    def __init__(self, executor, parent=None):
        super().__init__(parent)
        self.executor = executor

    def run(self):
        try:
            def _on_progress(p):
                self.progress.emit(p.percentage, p.message)

            if hasattr(self.executor, 'set_progress_callback'):
                self.executor.set_progress_callback(_on_progress)

            result = self.executor.execute()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Helper: строка ввода папки с кнопкой Browse
# ---------------------------------------------------------------------------
def _folder_row(label_text: str, parent: QWidget):
    layout = QHBoxLayout()
    label = QLabel(label_text)
    label.setFixedWidth(110)
    line = QLineEdit()
    btn = QPushButton("Обзор...")
    btn.setFixedWidth(80)

    def _browse():
        folder = QFileDialog.getExistingDirectory(parent, label_text)
        if folder:
            line.setText(folder)

    btn.clicked.connect(_browse)
    layout.addWidget(label)
    layout.addWidget(line)
    layout.addWidget(btn)
    return layout, line


def _file_row(label_text: str, parent: QWidget, filter_str: str = "CSV files (*.csv)"):
    """Строка ввода файла с кнопкой Browse."""
    layout = QHBoxLayout()
    label = QLabel(label_text)
    label.setFixedWidth(110)
    line = QLineEdit()
    btn = QPushButton("Обзор...")
    btn.setFixedWidth(80)

    def _browse():
        file_path, _ = QFileDialog.getOpenFileName(parent, label_text, "", filter_str)
        if file_path:
            line.setText(file_path)

    btn.clicked.connect(_browse)
    layout.addWidget(label)
    layout.addWidget(line)
    layout.addWidget(btn)
    return layout, line


# ---------------------------------------------------------------------------
# Tab 0: Prepare Experiments
# ---------------------------------------------------------------------------
class PrepareExperimentsTab(QWidget):
    def __init__(self, sort_tab, tabs=None, parent=None):
        super().__init__(parent)
        self.sort_tab = sort_tab
        self.tabs = tabs
        self._records = []
        self._updating_table = False
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        root_layout, self.root_line = _folder_row("Корневая папка:", self)
        layout.addLayout(root_layout)

        output_layout, self.output_root_line = _folder_row("Папка вывода:", self)
        self.output_root_line.setPlaceholderText("Необязательно; по умолчанию <root>_processed")
        layout.addLayout(output_layout)

        btn_layout = QHBoxLayout()
        self.scan_btn = QPushButton("Сканировать")
        self.use_btn = QPushButton("Передать в сортировку + бинаризацию")
        self.use_btn.setEnabled(False)
        btn_layout.addWidget(self.scan_btn)
        btn_layout.addWidget(self.use_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.status_label = QLabel("Готово")
        layout.addWidget(self.status_label)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Пропуск", "ID", "Название эксперимента", "Исходная папка",
            "PNG", "Статус", "Папка вывода", "Проблемы"
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        header.setSectionResizeMode(7, QHeaderView.Stretch)
        layout.addWidget(self.table)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        layout.addWidget(self.log_text)

        self.scan_btn.clicked.connect(self._scan)
        self.use_btn.clicked.connect(self._use_selected)
        self.table.itemDoubleClicked.connect(self._on_table_item_double_clicked)
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.root_line.textEdited.connect(self._on_root_edited)
        self.output_root_line.textEdited.connect(self._refresh_output_base_column)

    def _on_root_edited(self, text: str):
        if text.strip() and not self.output_root_line.text().strip():
            self.output_root_line.setPlaceholderText(default_processed_root(text.strip()))

    def _scan(self):
        root_folder = self.root_line.text().strip()
        if not root_folder:
            QMessageBox.warning(self, "Внимание", "Выберите корневую папку экспериментов.")
            return

        if not self.output_root_line.text().strip():
            self.output_root_line.setText(default_processed_root(root_folder))

        self.log_text.clear()
        self._log(f"Сканирование: {root_folder}")
        scan_result = scan_experiment_root(root_folder)
        self._records = scan_result.records
        self._populate_table()

        if scan_result.errors:
            for error in scan_result.errors:
                self._log(f"ОШИБКА: {_translate_experiment_message(error)}")

        self._update_status_label()
        self.use_btn.setEnabled(bool(self._records))
        self._log(self.status_label.text())

    def _populate_table(self):
        self._updating_table = True
        self.table.setRowCount(len(self._records))
        for row, record in enumerate(self._records):
            output_base = self._output_base_for_record(record)

            skip_item = QTableWidgetItem()
            skip_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            skip_item.setCheckState(Qt.Checked if record.skipped else Qt.Unchecked)
            skip_item.setData(Qt.UserRole, row)
            skip_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, skip_item)

            values = [
                record.experiment_id,
                record.name,
                str(Path(record.source_folder).name),
                str(record.png_count),
                _experiment_status_label(record.status),
                output_base,
                _experiment_issue_text(record),
            ]

            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                table_col = col + 1
                if table_col == 1:
                    item.setData(Qt.UserRole, row)
                if table_col in {3, 6, 7}:
                    item.setToolTip(value)
                self.table.setItem(row, table_col, item)

        if self._records:
            self.table.selectRow(0)
        self._updating_table = False

    def _on_table_item_changed(self, item):
        if self._updating_table or item.column() != 0:
            return

        record_index = item.data(Qt.UserRole)
        if record_index is None:
            return

        record = self._records[int(record_index)]
        record.skipped = item.checkState() == Qt.Checked
        self._set_status_cell(item.row(), record.status)
        self._update_status_label()

    def _on_table_item_double_clicked(self, item):
        if item.column() != 0:
            self._use_selected()

    def _set_status_cell(self, row: int, status: str):
        item = self.table.item(row, 5)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, 5, item)
        item.setText(_experiment_status_label(status))

    def _refresh_output_base_column(self, *_args):
        for row, record in enumerate(self._records):
            output_base = self._output_base_for_record(record)
            item = self.table.item(row, 6)
            if item is None:
                item = QTableWidgetItem()
                self.table.setItem(row, 6, item)
            item.setText(output_base)
            item.setToolTip(output_base)

    def _update_status_label(self):
        ready_count = sum(1 for record in self._records if record.sort_ready and not record.skipped)
        skipped_count = sum(1 for record in self._records if record.skipped)
        self.status_label.setText(
            f"Найдено экспериментов: {len(self._records)}, готово: {ready_count}, пропущено: {skipped_count}"
        )

    def _selected_record(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return None

        item = self.table.item(selected_rows[0].row(), 1)
        if item is None:
            return None

        record_index = item.data(Qt.UserRole)
        if record_index is None:
            return None

        return self._records[int(record_index)]

    def _use_selected(self):
        record = self._selected_record()
        if record is None:
            QMessageBox.warning(self, "Внимание", "Сначала выберите эксперимент.")
            return

        if record.skipped:
            QMessageBox.warning(
                self,
                "Эксперимент пропущен",
                "Этот эксперимент отмечен как пропущенный и не будет использован для сортировки + бинаризации.",
            )
            return

        if not record.sort_ready:
            QMessageBox.critical(
                self,
                "Эксперимент не готов",
                _experiment_issue_text(record) or "Эксперимент нельзя использовать для сортировки + бинаризации.",
            )
            return

        output_base = self._output_base_for_record(record)
        self.sort_tab.set_prepared_experiment(
            input_folder=record.source_folder,
            experiment_name=record.name,
            output_base_folder=output_base,
        )
        self._log(f"Выбрано: {record.name}")
        self._log(f"Вход: {record.source_folder}")
        self._log(f"Папка вывода: {output_base}")

        if self.tabs is not None:
            self.tabs.setCurrentWidget(self.sort_tab)

    def _output_base_for_record(self, record):
        output_root = self.output_root_line.text().strip()
        if not output_root:
            root_folder = self.root_line.text().strip()
            output_root = default_processed_root(root_folder) if root_folder else ""
        return build_output_base_folder(output_root, record) if output_root else ""

    def _log(self, text: str):
        self.log_text.append(text)


# ---------------------------------------------------------------------------
# Tab 1: Sort + Binarize
# ---------------------------------------------------------------------------
class SortBinarizeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._executor = None
        self._experiment_name = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Input folder
        folder_layout, self.input_line = _folder_row("Входная папка:", self)
        layout.addLayout(folder_layout)

        # Optional output base folder
        output_layout, self.output_base_line = _folder_row("Папка вывода:", self)
        self.output_base_line.setPlaceholderText("Необязательно; по умолчанию <input>_cam_sorted")
        layout.addLayout(output_layout)

        self.experiment_label = QLabel("Эксперимент: -")
        layout.addWidget(self.experiment_label)

        # Threshold
        h = QHBoxLayout()
        threshold_label = QLabel("Порог:")
        h.addWidget(threshold_label)
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 65535)
        self.threshold_spin.setValue(2000)
        self.threshold_spin.setSingleStep(100)
        _set_tooltip(
            "Порог бинаризации для 16-bit PNG.\n"
            "Единицы: уровень яркости пикселя, диапазон 0..65535.\n"
            "Пиксели со значением >= порога станут белыми (255), остальные черными (0).",
            threshold_label,
            self.threshold_spin,
        )
        h.addWidget(self.threshold_spin)
        h.addStretch()
        layout.addLayout(h)

        median_layout = QHBoxLayout()
        self.median_filter_cb = QCheckBox("Медианная фильтрация исходного изображения")
        self.median_kernel_combo = QComboBox()
        self.median_kernel_combo.addItem("3x3", 3)
        self.median_kernel_combo.addItem("5x5", 5)
        self.median_kernel_combo.setCurrentIndex(1)
        _set_tooltip(
            "Опциональная предварительная обработка перед пороговой бинаризацией.\n"
            "Фильтр подавляет одиночный шум и мелкие выбросы яркости.\n"
            "Для 16-bit PNG OpenCV поддерживает окна 3x3 и 5x5; по умолчанию выбрано 5x5.",
            self.median_filter_cb,
            self.median_kernel_combo,
        )
        median_layout.addWidget(self.median_filter_cb)
        median_layout.addWidget(QLabel("Окно:"))
        median_layout.addWidget(self.median_kernel_combo)
        median_layout.addStretch()
        layout.addLayout(median_layout)
        self._update_median_controls()

        # Validate format
        self.validate_cb = QCheckBox("Проверять формат (16-bit PNG)")
        self.validate_cb.setChecked(True)
        self.validate_cb.setToolTip(
            "Проверяет, что входные изображения являются 16-bit PNG.\n"
            "Это важно: порог 0..65535 рассчитан именно на 16-битную яркость."
        )
        layout.addWidget(self.validate_cb)

        # Run / Cancel
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Запустить")
        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Готово")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Connections
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn.clicked.connect(self._cancel)
        self.input_line.textEdited.connect(self._clear_prepared_experiment)
        self.median_filter_cb.toggled.connect(self._update_median_controls)

    def _update_median_controls(self):
        self.median_kernel_combo.setEnabled(self.median_filter_cb.isChecked())

    def _run(self):
        folder = self.input_line.text().strip()
        if not folder:
            QMessageBox.warning(self, "Внимание", "Выберите входную папку.")
            return

        output_base = self.output_base_line.text().strip()
        params = SortBinarizeParameters(
            input_folder=folder,
            threshold=self.threshold_spin.value(),
            validate_format=self.validate_cb.isChecked(),
            output_base_folder=output_base or None,
            experiment_name=self._experiment_name,
            median_filter_enabled=self.median_filter_cb.isChecked(),
            median_kernel_size=self.median_kernel_combo.currentData(),
        )

        self._executor = SortBinarizeExecutor()
        ok, msg = self._executor.set_parameters(params)
        if not ok:
            QMessageBox.critical(self, "Ошибка", msg)
            return

        self._set_running(True)
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self._log("Запуск сортировки + бинаризации...")
        self._log(f"  Вход: {folder}")
        if self._experiment_name:
            self._log(f"  Эксперимент: {self._experiment_name}")
        if output_base:
            self._log(f"  Папка вывода: {output_base}")
        self._log(f"  Порог: {self.threshold_spin.value()}")
        if self.median_filter_cb.isChecked():
            kernel_size = self.median_kernel_combo.currentData()
            self._log(f"  Медианный фильтр: {kernel_size}x{kernel_size}")
        else:
            self._log("  Медианный фильтр: выключен")
        self._log(f"  Проверка: {self.validate_cb.isChecked()}")
        self._log("")

        self._worker = WorkerThread(self._executor, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _cancel(self):
        if self._executor:
            self._executor.cancel()
        self._log("Запрошена отмена...")

    def _log(self, text: str):
        self.log_text.append(text)

    def set_prepared_experiment(self, input_folder: str, experiment_name: str, output_base_folder: str):
        self._experiment_name = experiment_name
        self.input_line.setText(input_folder)
        self.output_base_line.setText(output_base_folder)
        self.experiment_label.setText(f"Эксперимент: {experiment_name}")
        self.log_text.clear()
        self._log("Выбран подготовленный эксперимент.")
        self._log(f"Вход: {input_folder}")
        self._log(f"Папка вывода: {output_base_folder}")

    def _clear_prepared_experiment(self, *_args):
        if self._experiment_name:
            self._experiment_name = None
            self.output_base_line.clear()
            self.experiment_label.setText("Эксперимент: -")

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(int(pct))
        self.status_label.setText(msg)
        self._log(f"[{pct:.1f}%] {msg}")

    def _on_finished(self, result):
        self._set_running(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Готово" if result.success else "Ошибка")
        self._log("--- Результат ---")
        self._log(f"Успешно: {result.success}")
        self._log(f"Кадров cam_1: {result.cam1_count}")
        self._log(f"Кадров cam_2: {result.cam2_count}")
        self._log(f"Всего обработано: {result.total_processed}")
        self._log(f"Вывод: {result.output_folder}")
        self._log(f"Порог: {result.threshold}")
        if result.errors:
            self._log(f"Ошибки: {result.errors}")

    def _on_error(self, msg):
        self._set_running(False)
        self.status_label.setText("Ошибка")
        self._log(f"ОШИБКА: {msg}")

    def _set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)


# ---------------------------------------------------------------------------
# Tab 2: PTV Analysis
# ---------------------------------------------------------------------------
class PTVAnalysisTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._executor = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Input folder
        folder_layout, self.input_line = _folder_row("Входная папка:", self)
        layout.addLayout(folder_layout)

        # Detection parameters
        det_group = QGroupBox("Параметры детекции")
        det_group.setToolTip(
            "Параметры поиска частиц на бинаризованных изображениях.\n"
            "Площадь измеряется в пикселях квадратных (px^2)."
        )
        det_layout = QVBoxLayout(det_group)

        h1 = QHBoxLayout()
        min_area_label = QLabel("min_area:")
        h1.addWidget(min_area_label)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 1000)
        self.min_area_spin.setValue(4)
        _set_tooltip(
            "Минимальная площадь связанной белой области, которая считается частицей.\n"
            "Единицы: px^2. Области меньше этого значения отбрасываются как шум.",
            min_area_label,
            self.min_area_spin,
        )
        h1.addWidget(self.min_area_spin)
        max_area_label = QLabel("max_area:")
        h1.addWidget(max_area_label)
        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(1, 1000)
        self.max_area_spin.setValue(150)
        _set_tooltip(
            "Максимальная площадь связанной белой области, которая считается частицей.\n"
            "Единицы: px^2. Более крупные области отбрасываются как слипшиеся частицы или артефакты.",
            max_area_label,
            self.max_area_spin,
        )
        h1.addWidget(self.max_area_spin)
        h1.addStretch()
        det_layout.addLayout(h1)
        layout.addWidget(det_group)

        # Matching parameters
        match_group = QGroupBox("Параметры сопоставления")
        match_group.setToolTip(
            "Параметры сопоставления частиц между двумя последовательными кадрами.\n"
            "Координаты и расстояния здесь измеряются в пикселях."
        )
        match_layout = QVBoxLayout(match_group)

        h2 = QHBoxLayout()
        max_dist_label = QLabel("max_distance:")
        h2.addWidget(max_dist_label)
        self.max_dist_spin = QDoubleSpinBox()
        self.max_dist_spin.setRange(1.0, 100.0)
        self.max_dist_spin.setValue(50.0)
        _set_tooltip(
            "Максимальный радиус поиска пары для частицы из кадра A в кадре B.\n"
            "Единицы: px. Большее значение допускает быстрые смещения, но повышает риск ложных пар.",
            max_dist_label,
            self.max_dist_spin,
        )
        h2.addWidget(self.max_dist_spin)
        max_diam_label = QLabel("max_diameter_diff:")
        h2.addWidget(max_diam_label)
        self.max_diam_spin = QDoubleSpinBox()
        self.max_diam_spin.setRange(0.0, 10.0)
        self.max_diam_spin.setValue(4.0)
        _set_tooltip(
            "Максимальная допустимая разница эквивалентных диаметров пары частиц.\n"
            "Единицы: px. Диаметр считается как 2 * sqrt(area / pi).",
            max_diam_label,
            self.max_diam_spin,
        )
        h2.addWidget(self.max_diam_spin)
        h2.addStretch()
        match_layout.addLayout(h2)
        layout.addWidget(match_group)

        # Run / Cancel
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Запустить")
        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Готово")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Connections
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn.clicked.connect(self._cancel)

    def _run(self):
        folder = self.input_line.text().strip()
        if not folder:
            QMessageBox.warning(self, "Внимание", "Выберите входную папку.")
            return

        params = PTVParameters(
            input_folder=folder,
            detection_min_area=self.min_area_spin.value(),
            detection_max_area=self.max_area_spin.value(),
            matching_max_distance=self.max_dist_spin.value(),
            matching_max_diameter_diff=self.max_diam_spin.value(),
        )

        self._executor = PTVExecutor()
        ok, msg = self._executor.set_parameters(params)
        if not ok:
            QMessageBox.critical(self, "Ошибка", msg)
            return

        self._set_running(True)
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self._log("Запуск PTV анализа...")
        self._log(f"  Вход: {folder}")
        self._log(f"  min_area: {self.min_area_spin.value()}, max_area: {self.max_area_spin.value()}")
        self._log(f"  max_distance: {self.max_dist_spin.value()}, max_diameter_diff: {self.max_diam_spin.value()}")
        self._log("")

        self._worker = WorkerThread(self._executor, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _cancel(self):
        if self._executor:
            self._executor.cancel()
        self._log("Запрошена отмена...")

    def _log(self, text: str):
        self.log_text.append(text)

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(int(pct))
        self.status_label.setText(msg)
        self._log(f"[{pct:.1f}%] {msg}")

    def _on_finished(self, result):
        self._set_running(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Готово" if result.success else "Ошибка")
        self._log("--- Результат ---")
        self._log(f"Успешно: {result.success}")
        self._log(f"Обработано изображений: {result.total_images_processed}")
        self._log(f"Найдено частиц: {result.total_particles_detected}")
        self._log(f"Сопоставлено пар: {result.total_pairs_matched}")
        self._log(f"Пар cam_1: {result.cam1_pairs_count}")
        self._log(f"Пар cam_2: {result.cam2_pairs_count}")
        self._log(f"Вывод: {result.output_folder}")
        if result.errors:
            self._log(f"Ошибки: {result.errors}")
        if result.warnings:
            self._log(f"Предупреждения: {result.warnings}")

    def _on_error(self, msg):
        self._set_running(False)
        self.status_label.setText("Ошибка")
        self._log(f"ОШИБКА: {msg}")

    def _set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)


# ---------------------------------------------------------------------------
# Tab 3: PTV Processing (фильтрация, усреднение, визуализация)
# ---------------------------------------------------------------------------
def _bgr_to_pixmap(image):
    if image is None:
        return QPixmap()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


class PTVImageView(QWidget):
    LEGEND_ITEMS = [
        ("Частицы кадра A", QColor(0, 255, 0)),
        ("Позиции кадра B", QColor(255, 0, 0)),
        ("Вектор смещения", QColor(255, 165, 0)),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._view_rect = QRectF()
        self._history = []
        self._drag_start = None
        self._drag_current = None
        self.setMinimumSize(360, 260)
        self.setMouseTracking(True)
        self.setToolTip("Выделите область мышью для увеличения. Правый клик или «Уменьшить» снижает масштаб.")

    def set_image(self, image):
        self._pixmap = _bgr_to_pixmap(image)
        self._history = []
        self._drag_start = None
        self._drag_current = None
        self.reset_zoom()

    def clear_image(self):
        self._pixmap = QPixmap()
        self._view_rect = QRectF()
        self._history = []
        self._drag_start = None
        self._drag_current = None
        self.update()

    def reset_zoom(self):
        if self._pixmap.isNull():
            self.update()
            return
        self._view_rect = QRectF(0, 0, self._pixmap.width(), self._pixmap.height())
        self._history = []
        self.update()

    def zoom_back(self):
        if not self._history:
            return
        self._view_rect = self._history.pop()
        self.update()

    def zoom_out(self, factor: float = 1.8):
        if self._pixmap.isNull() or self._view_rect.isNull():
            return
        full = self._full_rect()
        if self._rects_close(self._view_rect, full):
            self.reset_zoom()
            return

        center = self._view_rect.center()
        expanded = QRectF(
            center.x() - self._view_rect.width() * factor / 2,
            center.y() - self._view_rect.height() * factor / 2,
            self._view_rect.width() * factor,
            self._view_rect.height() * factor,
        )
        self._set_view_rect(expanded, push_history=True)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 10, 10))

        if self._pixmap.isNull() or self._view_rect.isNull():
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), Qt.AlignCenter, "Предпросмотр")
            return

        target = self._target_rect()
        painter.drawPixmap(target, self._pixmap, self._view_rect)
        self._draw_legend(painter)

        if self._drag_start is not None and self._drag_current is not None:
            selection = QRectF(self._drag_start, self._drag_current).normalized()
            selection = selection.intersected(target)
            if selection.width() > 0 and selection.height() > 0:
                painter.fillRect(selection, QColor(80, 160, 255, 45))
                painter.setPen(QPen(QColor(80, 180, 255), 2, Qt.DashLine))
                painter.drawRect(selection)

    def mousePressEvent(self, event):
        if self._pixmap.isNull():
            return
        if event.button() == Qt.RightButton:
            self.zoom_out()
            return
        if event.button() == Qt.LeftButton and self._target_rect().contains(event.pos()):
            self._drag_start = event.pos()
            self._drag_current = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self._drag_start is None:
            return
        self._drag_current = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self._drag_start is None:
            return

        target = self._target_rect()
        selection = QRectF(self._drag_start, event.pos()).normalized().intersected(target)
        self._drag_start = None
        self._drag_current = None

        if selection.width() >= 8 and selection.height() >= 8:
            self._set_view_rect(self._widget_rect_to_source(selection), push_history=True)
        else:
            self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.reset_zoom()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._pixmap.isNull() and not self._view_rect.isNull():
            self._view_rect = self._fit_source_aspect(self._view_rect)
            self.update()

    def _draw_legend(self, painter: QPainter):
        painter.save()
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        margin = 10
        row_h = 22
        box_w = 230
        box_h = margin * 2 + row_h * len(self.LEGEND_ITEMS)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 170)))
        painter.drawRect(0, 0, box_w, box_h)

        for idx, (label, color) in enumerate(self.LEGEND_ITEMS):
            y = margin + idx * row_h + 15
            painter.setPen(QPen(color, 2))
            if idx == 2:
                painter.drawLine(12, y - 5, 32, y + 5)
            else:
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(18, y - 7, 10, 10)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(42, y, label)

        painter.restore()

    def _target_rect(self) -> QRectF:
        if self._pixmap.isNull() or self._view_rect.isNull():
            return QRectF()
        viewport = QRectF(self.rect())
        if viewport.width() <= 0 or viewport.height() <= 0:
            return QRectF()

        source_aspect = self._view_rect.width() / self._view_rect.height()
        viewport_aspect = viewport.width() / viewport.height()
        if source_aspect > viewport_aspect:
            width = viewport.width()
            height = width / source_aspect
        else:
            height = viewport.height()
            width = height * source_aspect

        return QRectF(
            (viewport.width() - width) / 2,
            (viewport.height() - height) / 2,
            width,
            height,
        )

    def _widget_rect_to_source(self, widget_rect: QRectF) -> QRectF:
        target = self._target_rect()
        if target.isNull():
            return QRectF()
        left = self._view_rect.left() + (widget_rect.left() - target.left()) / target.width() * self._view_rect.width()
        top = self._view_rect.top() + (widget_rect.top() - target.top()) / target.height() * self._view_rect.height()
        right = self._view_rect.left() + (widget_rect.right() - target.left()) / target.width() * self._view_rect.width()
        bottom = self._view_rect.top() + (widget_rect.bottom() - target.top()) / target.height() * self._view_rect.height()
        return QRectF(left, top, right - left, bottom - top)

    def _set_view_rect(self, source_rect: QRectF, push_history: bool):
        if self._pixmap.isNull() or source_rect.isNull():
            return
        next_rect = self._fit_source_aspect(source_rect)
        if next_rect.width() < 2 or next_rect.height() < 2:
            return
        if self._rects_close(next_rect, self._view_rect):
            self.update()
            return
        if push_history and not self._view_rect.isNull():
            self._history.append(QRectF(self._view_rect))
            self._history = self._history[-25:]
        self._view_rect = next_rect
        self.update()

    def _fit_source_aspect(self, source_rect: QRectF) -> QRectF:
        source_rect = self._clamp_rect(source_rect)
        if source_rect.isNull() or self.height() <= 0:
            return source_rect

        viewport_aspect = max(1.0, self.width()) / max(1.0, self.height())
        rect_aspect = source_rect.width() / source_rect.height()
        width = source_rect.width()
        height = source_rect.height()
        if rect_aspect > viewport_aspect:
            height = width / viewport_aspect
        else:
            width = height * viewport_aspect

        center = source_rect.center()
        fitted = QRectF(center.x() - width / 2, center.y() - height / 2, width, height)
        return self._clamp_rect(fitted)

    def _clamp_rect(self, source_rect: QRectF) -> QRectF:
        full = self._full_rect()
        if full.isNull():
            return QRectF()

        width = max(2.0, min(source_rect.width(), full.width()))
        height = max(2.0, min(source_rect.height(), full.height()))
        left = source_rect.center().x() - width / 2
        top = source_rect.center().y() - height / 2
        left = max(full.left(), min(left, full.right() - width))
        top = max(full.top(), min(top, full.bottom() - height))
        return QRectF(left, top, width, height)

    def _full_rect(self) -> QRectF:
        if self._pixmap.isNull():
            return QRectF()
        return QRectF(0, 0, self._pixmap.width(), self._pixmap.height())

    @staticmethod
    def _rects_close(a: QRectF, b: QRectF) -> bool:
        return (
            abs(a.left() - b.left()) < 0.5 and
            abs(a.top() - b.top()) < 0.5 and
            abs(a.width() - b.width()) < 0.5 and
            abs(a.height() - b.height()) < 0.5
        )


class PTVPreviewDialog(QDialog):
    def __init__(self, image, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 800)

        layout = QVBoxLayout(self)
        btn_layout = QHBoxLayout()
        self.back_btn = QPushButton("Назад")
        self.zoom_out_btn = QPushButton("Уменьшить")
        self.reset_btn = QPushButton("Сброс")
        self.back_btn.setToolTip("Вернуться к предыдущей области просмотра")
        self.zoom_out_btn.setToolTip("Расширить текущую область вокруг её центра")
        self.reset_btn.setToolTip("Показать всё изображение")
        btn_layout.addWidget(self.back_btn)
        btn_layout.addWidget(self.zoom_out_btn)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.image_view = PTVImageView()
        self.image_view.set_image(image)
        layout.addWidget(self.image_view)

        self.back_btn.clicked.connect(self.image_view.zoom_back)
        self.zoom_out_btn.clicked.connect(self.image_view.zoom_out)
        self.reset_btn.clicked.connect(self.image_view.reset_zoom)


class PTVHistogramDialog(QDialog):
    def __init__(self, viewer_tab, record, parent=None):
        super().__init__(parent)
        self.viewer_tab = viewer_tab
        self.record = record
        self._current_image = None

        self.setWindowTitle(f"Гистограмма: {record.camera}, пара {record.pair_number}")
        self.resize(1200, 800)
        self._init_ui()
        self._redraw()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Параметр:"))
        self.metric_combo = QComboBox()
        for key, spec in self.viewer_tab.HISTOGRAM_METRICS.items():
            self.metric_combo.addItem(spec["label"], key)
        params_layout.addWidget(self.metric_combo)

        params_layout.addWidget(QLabel("Бары:"))
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(1, 500)
        self.bins_spin.setValue(20)
        params_layout.addWidget(self.bins_spin)

        self.range_cb = QCheckBox("Диапазон вручную")
        params_layout.addWidget(self.range_cb)
        params_layout.addWidget(QLabel("min:"))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(-1000000000.0, 1000000000.0)
        self.min_spin.setDecimals(3)
        self.min_spin.setKeyboardTracking(False)
        params_layout.addWidget(self.min_spin)
        params_layout.addWidget(QLabel("max:"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(-1000000000.0, 1000000000.0)
        self.max_spin.setDecimals(3)
        self.max_spin.setKeyboardTracking(False)
        params_layout.addWidget(self.max_spin)
        params_layout.addStretch()
        layout.addLayout(params_layout)

        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("Папка PNG:"))
        self.output_line = QLineEdit(self.viewer_tab._default_histogram_folder())
        self.browse_btn = QPushButton("Обзор...")
        self.save_btn = QPushButton("Сохранить PNG")
        save_layout.addWidget(self.output_line)
        save_layout.addWidget(self.browse_btn)
        save_layout.addWidget(self.save_btn)
        layout.addLayout(save_layout)

        self.status_label = QLabel("Готово")
        layout.addWidget(self.status_label)

        self.image_view = PTVImageView()
        layout.addWidget(self.image_view)

        self.metric_combo.currentIndexChanged.connect(self._metric_changed)
        self.bins_spin.valueChanged.connect(self._redraw)
        self.range_cb.toggled.connect(self._range_toggled)
        self.min_spin.valueChanged.connect(self._redraw)
        self.max_spin.valueChanged.connect(self._redraw)
        self.browse_btn.clicked.connect(self._browse_folder)
        self.save_btn.clicked.connect(self._save_png)
        self._update_range_controls()

    def _metric_changed(self):
        self._fill_range_from_data()
        self._redraw()

    def _range_toggled(self):
        self._update_range_controls()
        if self.range_cb.isChecked():
            self._fill_range_from_data()
        self._redraw()

    def _update_range_controls(self):
        enabled = self.range_cb.isChecked()
        self.min_spin.setEnabled(enabled)
        self.max_spin.setEnabled(enabled)

    def _fill_range_from_data(self):
        values = self.viewer_tab._histogram_values(
            self.record,
            self.metric_combo.currentData(),
        )
        if values.size == 0:
            return
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if math.isclose(min_value, max_value):
            padding = max(abs(min_value) * 0.05, 1.0)
            min_value -= padding
            max_value += padding
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        self.min_spin.setValue(min_value)
        self.max_spin.setValue(max_value)
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)

    def _histogram_range(self):
        if not self.range_cb.isChecked():
            return None
        min_value = min(self.min_spin.value(), self.max_spin.value())
        max_value = max(self.min_spin.value(), self.max_spin.value())
        if math.isclose(min_value, max_value):
            return None
        return min_value, max_value

    def _redraw(self):
        image, summary = self.viewer_tab._create_histogram_image(
            self.record,
            metric_key=self.metric_combo.currentData(),
            bins=self.bins_spin.value(),
            value_range=self._histogram_range(),
        )
        self._current_image = image
        self.image_view.set_image(image)
        self.status_label.setText(summary)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку для PNG",
            self.output_line.text().strip() or self.viewer_tab._default_histogram_folder(),
        )
        if folder:
            self.output_line.setText(folder)

    def _save_png(self):
        folder = Path(self.output_line.text().strip() or self.viewer_tab._default_histogram_folder())
        try:
            output_path = self.viewer_tab._save_histogram_png(
                self.record,
                folder,
                metric_key=self.metric_combo.currentData(),
                bins=self.bins_spin.value(),
                value_range=self._histogram_range(),
            )
        except OSError as exc:
            QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить PNG:\n{exc}")
            return
        self.status_label.setText(f"Сохранено: {output_path}")


class PTVViewerTab(QWidget):
    HISTOGRAM_METRICS = {
        "Diameter": {"label": "Диаметр частицы", "unit": "px"},
        "Area": {"label": "Площадь частицы", "unit": "px^2"},
        "L": {"label": "Смещение L", "unit": "px"},
        "dx": {"label": "Смещение dx", "unit": "px"},
        "dy": {"label": "Смещение dy", "unit": "px"},
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._records = []
        self._visualizer = PTVVisualizer()
        self._visualizer_ready = False
        self._preview_image = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        ptv_layout, self.ptv_line = _folder_row("Папка PTV:", self)
        layout.addLayout(ptv_layout)

        original_layout, self.original_line = _folder_row("Папка кадров:", self)
        self.original_line.setPlaceholderText("Необязательно; определяется из PTV_<threshold>")
        layout.addLayout(original_layout)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Режим отображения:"))
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItem("Сопоставление пар", "matches")
        self.display_mode_combo.addItem("Гистограмма распределения частиц", "histogram")
        mode_layout.addWidget(self.display_mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        btn_layout = QHBoxLayout()
        self.scan_btn = QPushButton("Сканировать")
        self.preview_btn = QPushButton("Показать выбранное")
        self.open_btn = QPushButton("Открыть крупно")
        self.save_btn = QPushButton("Сохранить выбранное")
        self.back_btn = QPushButton("Назад")
        self.zoom_out_btn = QPushButton("Уменьшить")
        self.reset_zoom_btn = QPushButton("Сброс")
        self.preview_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.back_btn.setEnabled(False)
        self.zoom_out_btn.setEnabled(False)
        self.reset_zoom_btn.setEnabled(False)
        self.back_btn.setToolTip("Вернуться к предыдущей области просмотра")
        self.zoom_out_btn.setToolTip("Расширить текущую область вокруг её центра")
        self.reset_zoom_btn.setToolTip("Показать всё изображение")
        btn_layout.addWidget(self.scan_btn)
        btn_layout.addWidget(self.preview_btn)
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.back_btn)
        btn_layout.addWidget(self.zoom_out_btn)
        btn_layout.addWidget(self.reset_zoom_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.status_label = QLabel("Готово")
        layout.addWidget(self.status_label)

        splitter = QSplitter(Qt.Horizontal)
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "Камера", "Пара", "Совпадений", "Средн. L", "Макс. L",
            "Средн. dx", "Средн. dy", "Кадры", "CSV"
        ])
        self.table.setToolTip(
            "Сводка по CSV файлам PTV анализа.\n"
            "dx, dy и L здесь измеряются в пикселях между кадром A и кадром B."
        )
        _set_header_tooltips(self.table, {
            2: "Количество сопоставленных частиц в выбранной паре кадров.",
            3: "Средняя длина вектора смещения L = sqrt(dx^2 + dy^2). Единицы: px.",
            4: "Максимальная длина вектора смещения L в этой паре. Единицы: px.",
            5: "Средняя горизонтальная компонента смещения: X_B - X_A. Единицы: px.",
            6: "Средняя вертикальная компонента смещения: Y_B - Y_A. Единицы: px.",
            7: "Наличие исходных кадров A/B для визуального предпросмотра пары.",
            8: "CSV файл с колонками X0, Y0, dx, dy, L, Diameter, Area.",
        })
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(8, QHeaderView.Stretch)
        splitter.addWidget(self.table)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        self.info_label = QLabel("Выберите пару")
        preview_layout.addWidget(self.info_label)

        self.preview_view = PTVImageView()
        preview_layout.addWidget(self.preview_view)
        splitter.addWidget(preview_panel)
        splitter.setSizes([520, 480])
        layout.addWidget(splitter)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        layout.addWidget(self.log_text)

        self.scan_btn.clicked.connect(self._scan)
        self.display_mode_combo.currentIndexChanged.connect(self._display_mode_changed)
        self.preview_btn.clicked.connect(self._preview_selected)
        self.open_btn.clicked.connect(self._open_large)
        self.save_btn.clicked.connect(self._save_selected)
        self.back_btn.clicked.connect(self.preview_view.zoom_back)
        self.zoom_out_btn.clicked.connect(self.preview_view.zoom_out)
        self.reset_zoom_btn.clicked.connect(self.preview_view.reset_zoom)
        self.table.itemSelectionChanged.connect(self._preview_selected)
        self.table.itemDoubleClicked.connect(lambda _item: self._open_large())

    def _display_mode(self):
        return self.display_mode_combo.currentData() or "matches"

    def _display_mode_changed(self):
        self._preview_image = None
        self._set_preview_labels(None)
        self.info_label.setText("Выберите пару")
        self._update_action_buttons()
        if self._selected_record() is not None:
            self._preview_selected()

    def _can_preview_current_mode(self):
        if not self._records:
            return False
        if self._display_mode() == "histogram":
            return True
        return self._visualizer_ready

    def _update_action_buttons(self):
        can_preview = self._can_preview_current_mode()
        has_preview = self._preview_image is not None
        histogram_mode = self._display_mode() == "histogram"

        self.preview_btn.setEnabled(can_preview)
        self.save_btn.setEnabled(can_preview)
        self.open_btn.setEnabled(has_preview or (histogram_mode and bool(self._records)))
        self.back_btn.setEnabled(has_preview)
        self.zoom_out_btn.setEnabled(has_preview)
        self.reset_zoom_btn.setEnabled(has_preview)
        self.save_btn.setText("Сохранить PNG" if histogram_mode else "Сохранить выбранное")

    def _scan(self):
        ptv_folder = self.ptv_line.text().strip()
        if not ptv_folder:
            QMessageBox.warning(self, "Внимание", "Выберите папку с результатами PTV.")
            return

        self.log_text.clear()
        self.status_label.setText("Сканирование...")
        self._preview_image = None
        self._set_preview_labels(None)
        self.info_label.setText("Выберите пару")
        self._visualizer_ready = False

        result = scan_ptv_pairs(ptv_folder, self.original_line.text().strip() or None)
        self._records = result.records
        if result.original_folder and not self.original_line.text().strip():
            self.original_line.setText(result.original_folder)

        for error in result.errors:
            self._log(f"Предупреждение: {error}")

        visualizer_ready = False
        if result.original_folder:
            original_ok = self._visualizer.set_original_folder(result.original_folder)
            ptv_ok = self._visualizer.set_ptv_folder(ptv_folder) if original_ok else False
            visualizer_ready = original_ok and ptv_ok
            if not visualizer_ready:
                self._log("Предупреждение: не удалось инициализировать предпросмотр для выбранных папок.")
        else:
            self._log("Предупреждение: папка кадров не найдена. Выберите её вручную, чтобы включить предпросмотр.")

        self._populate_table()

        self.status_label.setText(f"Найдено пар: {len(self._records)} в {ptv_folder}")
        self._visualizer_ready = visualizer_ready
        self._update_action_buttons()
        self._log(self.status_label.text())

    def _populate_table(self):
        self.table.blockSignals(True)
        self.table.clearSelection()
        self.table.setRowCount(len(self._records))
        for row, record in enumerate(self._records):
            values = [
                record.camera,
                str(record.pair_number),
                str(record.matches_count),
                f"{record.mean_l:.2f}",
                f"{record.max_l:.2f}",
                f"{record.mean_dx:.2f}",
                f"{record.mean_dy:.2f}",
                "OK" if record.source_ok else "Нет",
                Path(record.csv_path).name,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 0:
                    item.setData(Qt.UserRole, row)
                if col == 8:
                    item.setToolTip(record.csv_path)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)
        if self._records:
            self.table.selectRow(0)

    def _selected_record(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return None
        item = self.table.item(selected_rows[0].row(), 0)
        if item is None:
            return None
        record_index = item.data(Qt.UserRole)
        return self._records[int(record_index)] if record_index is not None else None

    def _preview_selected(self):
        record = self._selected_record()
        if record is None:
            return
        if self._display_mode() == "histogram":
            self._preview_histogram_selected(record)
        else:
            self._preview_matches_selected(record)

    def _preview_matches_selected(self, record):
        if not record.source_ok:
            self.info_label.setText("Исходные кадры не найдены")
            self._set_preview_labels(None)
            self._preview_image = None
            self._update_action_buttons()
            return

        preview = self._visualizer.get_preview_image(
            record.camera,
            record.pair_number,
            draw_legend=False,
        )
        self._preview_image = preview
        if preview is None:
            self.info_label.setText("Не удалось построить предпросмотр")
            self._set_preview_labels(None)
            self._preview_image = None
            self._update_action_buttons()
            return

        self.info_label.setText(
            f"{record.camera}, пара {record.pair_number}: "
            f"совпадений {record.matches_count}, средн. L={record.mean_l:.2f}, макс. L={record.max_l:.2f}"
        )
        self._set_preview_labels(preview)
        self._update_action_buttons()

    def _preview_histogram_selected(self, record):
        image, summary = self._create_histogram_image(record)
        self._preview_image = image
        self.info_label.setText(summary)
        self._set_preview_labels(image)
        self._update_action_buttons()

    def _set_preview_labels(self, preview):
        if preview is None:
            self.preview_view.clear_image()
            return
        self.preview_view.set_image(preview)

    @staticmethod
    def _to_float_value(value):
        return float(str(value).replace(",", "."))

    def _histogram_values(self, record, metric_key="Diameter"):
        values = []
        try:
            with open(record.csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    try:
                        values.append(self._to_float_value(row.get(metric_key, "")))
                    except (TypeError, ValueError):
                        continue
        except OSError:
            return np.array([], dtype=float)
        return np.array(values, dtype=float)

    def _create_histogram_image(self, record, metric_key="Diameter", bins=20, value_range=None):
        spec = self.HISTOGRAM_METRICS.get(metric_key, self.HISTOGRAM_METRICS["Diameter"])
        values = self._histogram_values(record, metric_key)

        figure = Figure(figsize=(10, 6.5), dpi=120)
        canvas = FigureCanvasAgg(figure)
        ax = figure.add_subplot(111)

        title = f"{record.camera}, пара {record.pair_number}: {spec['label']}"
        if values.size == 0:
            ax.text(0.5, 0.5, "Нет числовых данных для гистограммы", ha="center", va="center")
            ax.set_axis_off()
            summary = f"{title}: нет данных"
        else:
            ax.hist(
                values,
                bins=max(1, int(bins)),
                range=value_range,
                color="#2f80ed",
                edgecolor="white",
                linewidth=0.8,
                alpha=0.9,
            )
            mean_value = float(np.mean(values))
            median_value = float(np.median(values))
            ax.axvline(mean_value, color="#d62728", linestyle="--", linewidth=1.5, label=f"Среднее: {mean_value:.2f}")
            ax.axvline(median_value, color="#2ca02c", linestyle=":", linewidth=1.8, label=f"Медиана: {median_value:.2f}")
            ax.legend(loc="upper right")
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_xlabel(f"{spec['label']}, {spec['unit']}")
            ax.set_ylabel("Количество частиц")
            summary = (
                f"{title}: частиц {values.size}, "
                f"min={np.min(values):.2f}, max={np.max(values):.2f}, "
                f"mean={mean_value:.2f}"
            )

        ax.set_title(title)
        figure.tight_layout()
        canvas.draw()
        width, height = canvas.get_width_height()
        rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        return image, summary

    @staticmethod
    def _default_histogram_folder():
        desktop = Path.home() / "Desktop"
        return str(desktop if desktop.exists() else Path.home())

    @staticmethod
    def _histogram_filename(record, metric_key):
        safe_metric = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(metric_key))
        return f"histogram_{record.camera}_pair_{record.pair_number}_{safe_metric}.png"

    def _save_histogram_png(self, record, folder, metric_key="Diameter", bins=20, value_range=None):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        image, _summary = self._create_histogram_image(
            record,
            metric_key=metric_key,
            bins=bins,
            value_range=value_range,
        )
        output_path = folder / self._histogram_filename(record, metric_key)
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Не удалось записать файл: {output_path}")
        return output_path

    def _open_large(self):
        record = self._selected_record()
        if record is None:
            return
        if self._display_mode() == "histogram":
            dialog = PTVHistogramDialog(self, record, self)
            dialog.exec_()
            return
        if self._preview_image is None:
            self._preview_selected()
        if self._preview_image is None:
            QMessageBox.warning(self, "Внимание", "Для этой пары нет предпросмотра.")
            return
        dialog = PTVPreviewDialog(
            self._preview_image,
            f"{record.camera}, пара {record.pair_number}",
            self,
        )
        dialog.exec_()

    def _save_selected(self):
        record = self._selected_record()
        if record is None:
            QMessageBox.warning(self, "Внимание", "Сначала выберите пару.")
            return
        if self._display_mode() == "histogram":
            self._save_selected_histogram(record)
            return
        if not record.source_ok:
            QMessageBox.warning(self, "Внимание", "Исходные кадры не найдены.")
            return

        result = self._visualizer.process_pair_on_first_frame(record.camera, record.pair_number)
        if result.get("success"):
            self._log(
                f"Сохранено изображений: {result.get('visualizations_created', 0)} для "
                f"{record.camera}, пара {record.pair_number}"
            )
            self._log(f"Вывод: {result.get('output_file') or self._visualizer.output_folder}")
        else:
            QMessageBox.warning(self, "Внимание", "; ".join(result.get("errors", [])) or "Не удалось сохранить.")

    def _save_selected_histogram(self, record):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку для PNG",
            self._default_histogram_folder(),
        )
        if not folder:
            return
        try:
            output_path = self._save_histogram_png(record, folder)
        except OSError as exc:
            QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить PNG:\n{exc}")
            return
        self._log(f"Сохранена гистограмма: {output_path}")
        self.status_label.setText(f"Сохранено: {output_path}")

    def _log(self, text: str):
        self.log_text.append(text)


class PTVProcessingPipelineResult:
    def __init__(self, camera_results, output_folder: str):
        self.camera_results = camera_results
        self.output_folder = output_folder
        self.success = bool(camera_results) and all(item["success"] for item in camera_results)
        self.errors = [
            f"{item['camera']}: {error}"
            for item in camera_results
            for error in item.get("errors", [])
        ]

        self.input_vectors = sum(
            item["filter_result"].input_vectors
            for item in camera_results
            if item.get("filter_result") is not None
        )
        self.output_vectors = sum(
            item["filter_result"].output_vectors
            for item in camera_results
            if item.get("filter_result") is not None
        )
        self.vectors_removed = sum(
            item["filter_result"].vectors_removed
            for item in camera_results
            if item.get("filter_result") is not None
        )
        self.removal_percentage = (
            self.vectors_removed / self.input_vectors * 100 if self.input_vectors else 0.0
        )
        self.output_cells = sum(
            item["average_result"].output_cells
            for item in camera_results
            if item.get("average_result") is not None
        )
        self.empty_cells = sum(
            item["average_result"].empty_cells
            for item in camera_results
            if item.get("average_result") is not None
        )
        self.total_cells = sum(
            item["average_result"].total_cells
            for item in camera_results
            if item.get("average_result") is not None
        )


class PTVProcessingPipelineExecutor:
    def __init__(self, pair_sum_jobs, filter_settings, average_settings):
        self.pair_sum_jobs = pair_sum_jobs
        self.filter_settings = filter_settings
        self.average_settings = average_settings

    def execute(self):
        camera_results = []
        output_folders = set()
        for camera, input_file in self.pair_sum_jobs:
            source_path = Path(input_file)
            output_folders.add(str(source_path.parent))
            item = {
                "camera": camera,
                "source_file": str(source_path),
                "saved_source_file": str(source_path),
                "filter_result": None,
                "average_result": None,
                "success": False,
                "errors": [],
            }

            try:
                filter_params = VectorFilterParameters(
                    input_file=str(source_path),
                    **self.filter_settings,
                )
                filter_executor = VectorFilterExecutor()
                ok, msg = filter_executor.set_parameters(filter_params)
                if not ok:
                    item["errors"].append(msg)
                    camera_results.append(item)
                    continue

                filter_result = filter_executor.execute()
                item["filter_result"] = filter_result
                item["errors"].extend(filter_result.errors)
                if not filter_result.success:
                    camera_results.append(item)
                    continue

                average_params = VectorAverageParameters(
                    input_file=filter_result.output_file,
                    **self.average_settings,
                )
                average_executor = VectorAverageExecutor()
                ok, msg = average_executor.set_parameters(average_params)
                if not ok:
                    item["errors"].append(msg)
                    camera_results.append(item)
                    continue

                average_result = average_executor.execute()
                item["average_result"] = average_result
                item["errors"].extend(average_result.errors)
                item["success"] = filter_result.success and average_result.success
            except Exception as exc:
                item["errors"].append(str(exc))

            camera_results.append(item)

        output_folder = "; ".join(sorted(output_folders))
        return PTVProcessingPipelineResult(camera_results, output_folder)


class PTVProcessingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._executor = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- 3a. Vector Filter ---
        filt_group = QGroupBox("1. Фильтр векторов")
        filt_group.setToolTip(
            "Фильтрует векторы из pair_sum CSV по компонентам смещения.\n"
            "Для обычных PTV результатов U/dx и V/dy измеряются в пикселях между кадрами."
        )
        filt_layout = QVBoxLayout(filt_group)

        f_cam1_row, self.filt_cam1_file_line = _file_row("cam_1 CSV:", self)
        filt_layout.addLayout(f_cam1_row)
        f_cam2_row, self.filt_cam2_file_line = _file_row("cam_2 CSV:", self)
        filt_layout.addLayout(f_cam2_row)
        _set_tooltip(
            "CSV файл pair_sum для камеры.\n"
            "Ожидаются колонки X0, Y0, dx, dy, L, Diameter, Area.",
            self.filt_cam1_file_line,
            self.filt_cam2_file_line,
        )

        # U range
        h_u = QHBoxLayout()
        self.filter_u_cb = QCheckBox("Фильтр U")
        self.filter_u_cb.setChecked(True)
        self.filter_u_cb.setToolTip(
            "Включает фильтр по горизонтальной компоненте вектора.\n"
            "U обычно соответствует колонке dx. Единицы как в CSV; для pair_sum это px."
        )
        h_u.addWidget(self.filter_u_cb)
        h_u.addWidget(QLabel("min:"))
        self.u_min_spin = QDoubleSpinBox()
        self.u_min_spin.setRange(-10000.0, 10000.0)
        self.u_min_spin.setValue(0.0)
        _set_tooltip("Нижняя граница допустимого U/dx. Единицы как в CSV; для pair_sum это px.", self.u_min_spin)
        h_u.addWidget(self.u_min_spin)
        h_u.addWidget(QLabel("max:"))
        self.u_max_spin = QDoubleSpinBox()
        self.u_max_spin.setRange(-10000.0, 10000.0)
        self.u_max_spin.setValue(40.0)
        _set_tooltip("Верхняя граница допустимого U/dx. Единицы как в CSV; для pair_sum это px.", self.u_max_spin)
        h_u.addWidget(self.u_max_spin)
        h_u.addStretch()
        filt_layout.addLayout(h_u)

        # V range
        h_v = QHBoxLayout()
        self.filter_v_cb = QCheckBox("Фильтр V")
        self.filter_v_cb.setChecked(True)
        self.filter_v_cb.setToolTip(
            "Включает фильтр по вертикальной компоненте вектора.\n"
            "V обычно соответствует колонке dy. Единицы как в CSV; для pair_sum это px."
        )
        h_v.addWidget(self.filter_v_cb)
        h_v.addWidget(QLabel("min:"))
        self.v_min_spin = QDoubleSpinBox()
        self.v_min_spin.setRange(-10000.0, 10000.0)
        self.v_min_spin.setValue(-10.0)
        _set_tooltip("Нижняя граница допустимого V/dy. Единицы как в CSV; для pair_sum это px.", self.v_min_spin)
        h_v.addWidget(self.v_min_spin)
        h_v.addWidget(QLabel("max:"))
        self.v_max_spin = QDoubleSpinBox()
        self.v_max_spin.setRange(-10000.0, 10000.0)
        self.v_max_spin.setValue(10.0)
        _set_tooltip("Верхняя граница допустимого V/dy. Единицы как в CSV; для pair_sum это px.", self.v_max_spin)
        h_v.addWidget(self.v_max_spin)
        h_v.addStretch()
        filt_layout.addLayout(h_v)

        layout.addWidget(filt_group)

        # --- 3b. Vector Average ---
        avg_group = QGroupBox("2. Усреднение векторов")
        avg_group.setToolTip(
            "Разбивает поле на регулярную сетку и усредняет векторы внутри каждой ячейки.\n"
            "Размеры задаются в единицах координат входного CSV; для pair_sum это px."
        )
        avg_layout = QVBoxLayout(avg_group)

        h_plane = QHBoxLayout()
        h_plane.addWidget(QLabel("plane_width:"))
        self.plane_w_spin = QDoubleSpinBox()
        self.plane_w_spin.setRange(1.0, 100000.0)
        self.plane_w_spin.setValue(4904.0)
        self.plane_w_spin.setDecimals(1)
        _set_tooltip("Ширина расчетной области. Единицы как X/Y во входном CSV; для pair_sum это px.", self.plane_w_spin)
        h_plane.addWidget(self.plane_w_spin)
        h_plane.addWidget(QLabel("plane_height:"))
        self.plane_h_spin = QDoubleSpinBox()
        self.plane_h_spin.setRange(1.0, 100000.0)
        self.plane_h_spin.setValue(3280.0)
        self.plane_h_spin.setDecimals(1)
        _set_tooltip("Высота расчетной области. Единицы как X/Y во входном CSV; для pair_sum это px.", self.plane_h_spin)
        h_plane.addWidget(self.plane_h_spin)
        h_plane.addStretch()
        avg_layout.addLayout(h_plane)

        h_cell = QHBoxLayout()
        h_cell.addWidget(QLabel("cell_width:"))
        self.cell_w_spin = QDoubleSpinBox()
        self.cell_w_spin.setRange(1.0, 10000.0)
        self.cell_w_spin.setValue(66.0)
        self.cell_w_spin.setDecimals(1)
        _set_tooltip("Ширина одной ячейки усреднения. Единицы как X/Y во входном CSV; для pair_sum это px.", self.cell_w_spin)
        h_cell.addWidget(self.cell_w_spin)
        h_cell.addWidget(QLabel("cell_height:"))
        self.cell_h_spin = QDoubleSpinBox()
        self.cell_h_spin.setRange(1.0, 10000.0)
        self.cell_h_spin.setValue(66.0)
        self.cell_h_spin.setDecimals(1)
        _set_tooltip("Высота одной ячейки усреднения. Единицы как X/Y во входном CSV; для pair_sum это px.", self.cell_h_spin)
        h_cell.addWidget(self.cell_h_spin)
        h_cell.addWidget(QLabel("min_points:"))
        self.min_pts_spin = QSpinBox()
        self.min_pts_spin.setRange(1, 1000)
        self.min_pts_spin.setValue(1)
        self.min_pts_spin.setToolTip(
            "Минимальное число векторов в ячейке, чтобы записать ее в результат.\n"
            "Ячейки с меньшим числом точек считаются недостаточно надежными и пропускаются."
        )
        h_cell.addWidget(self.min_pts_spin)
        h_cell.addStretch()
        avg_layout.addLayout(h_cell)

        self.avg_run_btn = QPushButton("Запустить фильтр + усреднение")
        avg_layout.addWidget(self.avg_run_btn)
        layout.addWidget(avg_group)

        result_group = QGroupBox("Результаты обработки")
        result_group.setToolTip(
            "Показывает сохраненные исходники каждого этапа и краткую статистику обработки.\n"
            "Полные пути к CSV видны при наведении на имена файлов."
        )
        result_layout = QVBoxLayout(result_group)
        self.processing_results_table = QTableWidget()
        self.processing_results_table.setColumnCount(8)
        self.processing_results_table.setHorizontalHeaderLabels([
            "Камера", "pair_sum", "После фильтра", "После усреднения",
            "Вход", "После фильтра", "Удалено %", "Ячейки"
        ])
        _set_header_tooltips(self.processing_results_table, {
            1: "Исходный pair_sum CSV. Результаты фильтра и усреднения сохраняются рядом с ним.",
            2: "CSV после удаления векторов, не прошедших фильтры U/V.",
            3: "CSV после усреднения прошедших фильтр векторов по ячейкам сетки.",
            4: "Количество векторов во входном pair_sum CSV.",
            5: "Количество векторов, оставшихся после фильтра.",
            6: "Доля удаленных векторов относительно входного CSV, в процентах.",
            7: "Количество непустых ячеек, записанных после усреднения.",
        })
        self.processing_results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.processing_results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.processing_results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.processing_results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.processing_results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        result_layout.addWidget(self.processing_results_table)
        layout.addWidget(result_group)

        # Shared progress + log
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Готово")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Connections
        self.avg_run_btn.clicked.connect(self._run_filter_average)

    # ---- Run Filter + Average ----
    def _run_filter_average(self):
        input_paths = [
            ("cam_1", self.filt_cam1_file_line.text().strip()),
            ("cam_2", self.filt_cam2_file_line.text().strip()),
        ]
        jobs = []
        filter_settings = {
            "filter_u": self.filter_u_cb.isChecked(),
            "u_min": self.u_min_spin.value(),
            "u_max": self.u_max_spin.value(),
            "filter_v": self.filter_v_cb.isChecked(),
            "v_min": self.v_min_spin.value(),
            "v_max": self.v_max_spin.value(),
        }
        for camera, csv_path in input_paths:
            if not csv_path:
                continue
            params = VectorFilterParameters(input_file=csv_path, **filter_settings)
            ok, msg = params.validate()
            if not ok:
                QMessageBox.critical(self, "Ошибка", f"{camera}: {msg}")
                return
            jobs.append((camera, csv_path))

        if not jobs:
            QMessageBox.warning(self, "Внимание", "Выберите входной CSV для cam_1 и/или cam_2.")
            return

        average_settings = {
            "plane_width": self.plane_w_spin.value(),
            "plane_height": self.plane_h_spin.value(),
            "cell_width": self.cell_w_spin.value(),
            "cell_height": self.cell_h_spin.value(),
            "min_points_in_cell": self.min_pts_spin.value(),
        }
        avg_check = VectorAverageParameters(input_file=jobs[0][1], **average_settings)
        ok, msg = avg_check.validate()
        if not ok:
            QMessageBox.critical(self, "Ошибка", msg)
            return

        self._executor = PTVProcessingPipelineExecutor(jobs, filter_settings, average_settings)

        self.log_text.clear()
        self._clear_processing_results()
        grid_info = avg_check.get_grid_info()
        self._log("Запуск обработки PTV...")
        for camera, csv_path in jobs:
            self._log(f"  {camera} pair_sum: {csv_path}")
        if self.filter_u_cb.isChecked():
            self._log(f"  U: [{self.u_min_spin.value()}, {self.u_max_spin.value()}]")
        if self.filter_v_cb.isChecked():
            self._log(f"  V: [{self.v_min_spin.value()}, {self.v_max_spin.value()}]")
        self._log(f"  Плоскость усреднения: {self.plane_w_spin.value()} x {self.plane_h_spin.value()}")
        self._log(f"  Ячейка усреднения: {self.cell_w_spin.value()} x {self.cell_h_spin.value()}")
        self._log(f"  Сетка усреднения: {grid_info['nx']} x {grid_info['ny']} = {grid_info['total_cells']} ячеек")
        self._log("")
        self._start_worker()

    # ---- Common ----
    def _start_worker(self):
        self._set_running(True)
        self.progress_bar.setValue(0)

        self._worker = WorkerThread(self._executor, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _log(self, text: str):
        self.log_text.append(text)

    def _clear_processing_results(self):
        self.processing_results_table.setRowCount(0)

    def _populate_processing_results(self, result):
        self.processing_results_table.setRowCount(len(result.camera_results))
        for row, item in enumerate(result.camera_results):
            filter_result = item.get("filter_result")
            average_result = item.get("average_result")
            filtered_file = filter_result.output_file if filter_result is not None else ""
            averaged_file = average_result.output_file if average_result is not None else ""

            values = [
                item["camera"],
                item["saved_source_file"],
                filtered_file,
                averaged_file,
                str(filter_result.input_vectors) if filter_result is not None else "0",
                str(filter_result.output_vectors) if filter_result is not None else "0",
                f"{filter_result.removal_percentage:.1f}" if filter_result is not None else "0.0",
                str(average_result.output_cells) if average_result is not None else "0",
            ]
            for col, value in enumerate(values):
                display = Path(value).name if col in (1, 2, 3) and value else value
                table_item = QTableWidgetItem(display)
                if col in (1, 2, 3) and value:
                    table_item.setToolTip(value)
                self.processing_results_table.setItem(row, col, table_item)
        self.processing_results_table.resizeRowsToContents()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(int(pct))
        self.status_label.setText(msg)
        self._log(f"[{pct:.1f}%] {msg}")

    def _on_finished(self, result):
        self._set_running(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Готово" if result.success else "Ошибка")
        self._log("--- Результат ---")
        self._log(f"Успешно: {result.success}")

        # PTVProcessingPipelineResult
        if hasattr(result, 'camera_results'):
            self._populate_processing_results(result)
            self._log(f"Сохранено рядом с: {result.output_folder}")
            self._log(f"Векторов на входе: {result.input_vectors}")
            self._log(f"Векторов после фильтра: {result.output_vectors}")
            self._log(f"Удалено: {result.vectors_removed} ({result.removal_percentage:.1f}%)")
            self._log(f"Ячеек после усреднения: {result.output_cells}")
            for item in result.camera_results:
                self._log(f"--- {item['camera']} ---")
                self._log(f"pair_sum: {item['saved_source_file']}")
                filter_result = item.get("filter_result")
                average_result = item.get("average_result")
                if filter_result is not None:
                    self._log(f"После фильтра: {filter_result.output_file}")
                if average_result is not None:
                    self._log(f"После усреднения: {average_result.output_file}")

        # VectorFilterResult
        elif hasattr(result, 'input_vectors') and hasattr(result, 'vectors_removed'):
            self._log(f"Векторов на входе: {result.input_vectors}")
            self._log(f"Векторов на выходе: {result.output_vectors}")
            self._log(f"Удалено: {result.vectors_removed} ({result.removal_percentage:.1f}%)")
            if hasattr(result, 'output_files'):
                for camera, output_file in result.output_files:
                    self._log(f"{camera}, файл вывода: {output_file}")
            else:
                self._log(f"Файл вывода: {result.output_file}")

        # VectorAverageResult
        elif hasattr(result, 'output_cells'):
            self._log(f"Векторов на входе: {result.input_vectors}")
            self._log(f"Сетка: {result.grid_size[0]} x {result.grid_size[1]}")
            self._log(f"Непустых ячеек: {result.output_cells}")
            self._log(f"Пустых ячеек: {result.empty_cells}")
            self._log(f"Точек в ячейке: min={result.min_points_per_cell}, "
                       f"max={result.max_points_per_cell}, avg={result.avg_points_per_cell:.1f}")
            self._log(f"Файл вывода: {result.output_file}")

        # VectorPlotResult
        elif hasattr(result, 'vectors_count'):
            self._log(f"Построено векторов: {result.vectors_count}")
            self._log(f"dx: [{result.dx_min:.3f}, {result.dx_max:.3f}]")
            self._log(f"dy: [{result.dy_min:.3f}, {result.dy_max:.3f}]")
            self._log(f"L: [{result.l_min:.3f}, {result.l_max:.3f}]")
            self._log(f"Файл вывода: {result.output_file}")

        if result.errors:
            self._log(f"Ошибки: {result.errors}")

    def _on_error(self, msg):
        self._set_running(False)
        self.status_label.setText("Ошибка")
        self._log(f"ОШИБКА: {msg}")

    def _set_running(self, running: bool):
        self.avg_run_btn.setEnabled(not running)


# ---------------------------------------------------------------------------
# Tab 4: Vector Graphics
# ---------------------------------------------------------------------------
class LegacyVectorGraphicsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._executor = None
        self._last_output_file = None
        self._plot_unit_label = ""
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- 1. Vector Plot (pixel units) ---
        raw_plot_group = QGroupBox("1. Визуализация векторов (px)")
        raw_plot_group.setToolTip(
            "Строит PNG с векторным полем по CSV после усреднения.\n"
            "Координаты и смещения до преобразования измеряются в пикселях."
        )
        raw_plot_layout = QVBoxLayout(raw_plot_group)

        raw_row, self.raw_plot_file_line = _file_row("Входной CSV:", self)
        self.raw_plot_file_line.setToolTip("CSV для построения графика в пикселях: обычно файл после усреднения.")
        raw_plot_layout.addLayout(raw_row)

        raw_h_plot1 = QHBoxLayout()
        raw_h_plot1.addWidget(QLabel("arrow_scale:"))
        self.raw_arrow_scale_spin = QDoubleSpinBox()
        self.raw_arrow_scale_spin.setRange(0.01, 1000.0)
        self.raw_arrow_scale_spin.setValue(20.0)
        self.raw_arrow_scale_spin.setDecimals(2)
        self.raw_arrow_scale_spin.setToolTip(
            "Масштаб стрелок на графике Matplotlib.\n"
            "Чем меньше значение, тем длиннее визуальные стрелки; данные в CSV не меняются."
        )
        raw_h_plot1.addWidget(self.raw_arrow_scale_spin)
        raw_h_plot1.addWidget(QLabel("arrow_width:"))
        self.raw_arrow_width_spin = QDoubleSpinBox()
        self.raw_arrow_width_spin.setRange(0.001, 0.05)
        self.raw_arrow_width_spin.setValue(0.003)
        self.raw_arrow_width_spin.setSingleStep(0.001)
        self.raw_arrow_width_spin.setDecimals(3)
        self.raw_arrow_width_spin.setToolTip(
            "Толщина стрелок на итоговом графике.\n"
            "Это только визуальная настройка, без единиц измерения."
        )
        raw_h_plot1.addWidget(self.raw_arrow_width_spin)
        raw_h_plot1.addStretch()
        raw_plot_layout.addLayout(raw_h_plot1)

        raw_h_plot2 = QHBoxLayout()
        raw_h_plot2.addWidget(QLabel("colormap:"))
        self.raw_plot_cmap_combo = QComboBox()
        self.raw_plot_cmap_combo.addItems([
            "jet", "viridis", "plasma", "inferno", "magma", "cividis",
            "coolwarm", "RdYlBu", "Spectral"
        ])
        self.raw_plot_cmap_combo.setToolTip("Цветовая карта, которой окрашиваются стрелки по выбранной величине.")
        raw_h_plot2.addWidget(self.raw_plot_cmap_combo)
        raw_h_plot2.addWidget(QLabel("color_by:"))
        self.raw_color_by_combo = QComboBox()
        self.raw_color_by_combo.addItems(["L", "dx", "dy", "angle"])
        self.raw_color_by_combo.setToolTip(
            "Величина, по которой окрашиваются стрелки:\n"
            "L - длина вектора, dx/dy - компоненты смещения, angle - направление."
        )
        raw_h_plot2.addWidget(self.raw_color_by_combo)
        raw_h_plot2.addStretch()
        raw_plot_layout.addLayout(raw_h_plot2)

        self.raw_plot_run_btn = QPushButton("Построить график")
        raw_plot_layout.addWidget(self.raw_plot_run_btn)
        layout.addWidget(raw_plot_group)

        # --- 2. Coordinate Transform ---
        transform_group = QGroupBox("2. Преобразование координат")
        transform_group.setToolTip(
            "Переводит координаты и смещения из пикселей в физические величины.\n"
            "Координаты результата записываются в мм, скорости - в м/с."
        )
        transform_layout = QVBoxLayout(transform_group)

        # Input CSV
        f_row, self.input_file_line = _file_row("Входной CSV:", self)
        self.input_file_line.setToolTip(
            "CSV после PTV/фильтрации или после усреднения.\n"
            "Поддерживаются колонки X0/Y0/dx/dy/L или X_center/Y_center/dx_avg/dy_avg/L_avg."
        )
        transform_layout.addLayout(f_row)

        # Origin
        h_origin = QHBoxLayout()
        h_origin.addWidget(QLabel("X_origin (px):"))
        self.x_origin_spin = QDoubleSpinBox()
        self.x_origin_spin.setRange(-100000.0, 100000.0)
        self.x_origin_spin.setValue(0.0)
        self.x_origin_spin.setDecimals(1)
        self.x_origin_spin.setToolTip(
            "X координата нового начала системы координат в исходном изображении.\n"
            "Единицы: px. Перед масштабированием из всех X вычитается это значение."
        )
        h_origin.addWidget(self.x_origin_spin)
        h_origin.addWidget(QLabel("Y_origin (px):"))
        self.y_origin_spin = QDoubleSpinBox()
        self.y_origin_spin.setRange(-100000.0, 100000.0)
        self.y_origin_spin.setValue(0.0)
        self.y_origin_spin.setDecimals(1)
        self.y_origin_spin.setToolTip(
            "Y координата нового начала системы координат в исходном изображении.\n"
            "Единицы: px. Перед масштабированием из всех Y вычитается это значение."
        )
        h_origin.addWidget(self.y_origin_spin)
        h_origin.addStretch()
        transform_layout.addLayout(h_origin)

        # Rotation
        h_rotation = QHBoxLayout()
        h_rotation.addWidget(QLabel("Угол поворота (°):"))
        self.rotation_spin = QSpinBox()
        self.rotation_spin.setRange(0, 360)
        self.rotation_spin.setValue(0)
        self.rotation_spin.setToolTip(
            "Угол поворота координат и векторов против часовой стрелки.\n"
            "Единицы: градусы."
        )
        h_rotation.addWidget(self.rotation_spin)
        h_rotation.addStretch()
        transform_layout.addLayout(h_rotation)

        # Scale
        h_scale = QHBoxLayout()
        h_scale.addWidget(QLabel("Масштаб (м/px):"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(7)
        self.scale_spin.setRange(0.0000001, 1000.0)
        self.scale_spin.setSingleStep(0.0001)
        self.scale_spin.setValue(0.0000075)
        self.scale_spin.setToolTip(
            "Физический размер одного пикселя.\n"
            "Единицы: м/px. Координаты умножаются на scale и записываются в мм."
        )
        h_scale.addWidget(self.scale_spin)
        h_scale.addWidget(QLabel("dt (с):"))
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setDecimals(7)
        self.dt_spin.setRange(0.0000001, 1000.0)
        self.dt_spin.setSingleStep(0.0001)
        self.dt_spin.setValue(0.000002)
        self.dt_spin.setToolTip(
            "Временной интервал между кадрами A и B.\n"
            "Единицы: секунды. Скорости считаются как dx * scale / dt и dy * scale / dt."
        )
        h_scale.addWidget(self.dt_spin)
        h_scale.addStretch()
        transform_layout.addLayout(h_scale)

        self.transform_run_btn = QPushButton("Преобразовать")
        transform_layout.addWidget(self.transform_run_btn)
        layout.addWidget(transform_group)

        # --- 3. Vector Plot (physical units) ---
        plot_group = QGroupBox("3. Визуализация векторов (м, м/с)")
        plot_group.setToolTip(
            "Строит график по CSV после преобразования координат.\n"
            "Оси используют X_mm/Y_mm, компоненты скорости - dx_ms/dy_ms/L_ms."
        )
        plot_layout = QVBoxLayout(plot_group)

        p_row, self.plot_file_line = _file_row("Входной CSV:", self)
        self.plot_file_line.setToolTip("CSV после преобразования координат с колонками X_mm, Y_mm, dx_ms, dy_ms, L_ms.")
        plot_layout.addLayout(p_row)

        h_plot1 = QHBoxLayout()
        h_plot1.addWidget(QLabel("arrow_scale:"))
        self.arrow_scale_spin = QDoubleSpinBox()
        self.arrow_scale_spin.setRange(0.01, 1000.0)
        self.arrow_scale_spin.setValue(20.0)
        self.arrow_scale_spin.setDecimals(2)
        self.arrow_scale_spin.setToolTip(
            "Масштаб стрелок на графике Matplotlib.\n"
            "Чем меньше значение, тем длиннее визуальные стрелки; скорости в CSV не меняются."
        )
        h_plot1.addWidget(self.arrow_scale_spin)
        h_plot1.addWidget(QLabel("arrow_width:"))
        self.arrow_width_spin = QDoubleSpinBox()
        self.arrow_width_spin.setRange(0.001, 0.05)
        self.arrow_width_spin.setValue(0.003)
        self.arrow_width_spin.setSingleStep(0.001)
        self.arrow_width_spin.setDecimals(3)
        self.arrow_width_spin.setToolTip("Толщина стрелок на итоговом графике. Это визуальная настройка без единиц измерения.")
        h_plot1.addWidget(self.arrow_width_spin)
        h_plot1.addStretch()
        plot_layout.addLayout(h_plot1)

        h_plot2 = QHBoxLayout()
        h_plot2.addWidget(QLabel("colormap:"))
        self.plot_cmap_combo = QComboBox()
        self.plot_cmap_combo.addItems([
            "jet", "viridis", "plasma", "inferno", "magma", "cividis",
            "coolwarm", "RdYlBu", "Spectral"
        ])
        self.plot_cmap_combo.setToolTip("Цветовая карта, которой окрашиваются стрелки по выбранной физической величине.")
        h_plot2.addWidget(self.plot_cmap_combo)
        h_plot2.addWidget(QLabel("color_by:"))
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(["L", "dx", "dy", "angle"])
        self.color_by_combo.setToolTip(
            "Величина для окраски стрелок:\n"
            "L - модуль скорости, dx/dy - компоненты скорости, angle - направление."
        )
        h_plot2.addWidget(self.color_by_combo)
        h_plot2.addStretch()
        plot_layout.addLayout(h_plot2)

        self.plot_run_btn = QPushButton("Построить график")
        plot_layout.addWidget(self.plot_run_btn)
        layout.addWidget(plot_group)

        # Shared progress + log
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Готово")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Connections
        self.raw_plot_run_btn.clicked.connect(self._run_raw_plot)
        self.transform_run_btn.clicked.connect(self._run_transform)
        self.plot_run_btn.clicked.connect(self._run_plot)

    # ---- Run Raw Vector Plot ----
    def _run_raw_plot(self):
        csv_path = self.raw_plot_file_line.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "Внимание", "Выберите входной CSV файл.")
            return

        params = VectorPlotParameters(
            input_file=csv_path,
            arrow_scale=self.raw_arrow_scale_spin.value(),
            arrow_width=self.raw_arrow_width_spin.value(),
            colormap=self.raw_plot_cmap_combo.currentText(),
            color_by=self.raw_color_by_combo.currentText(),
            invert_y=False,
        )

        self._executor = VectorPlotExecutor()
        ok, msg = self._executor.set_parameters(params)
        if not ok:
            QMessageBox.critical(self, "Ошибка", msg)
            return

        self._plot_unit_label = "px"
        self.log_text.clear()
        self._log("Запуск визуализации векторов в пикселях...")
        self._log(f"  Вход: {csv_path}")
        self._log(f"  arrow_scale: {params.arrow_scale}, arrow_width: {params.arrow_width}")
        self._log(f"  colormap: {params.colormap}, color_by: {params.color_by}")
        self._log("  оси: X (px), Y (px)")
        self._log("")
        self._start_worker()

    # ---- Run Transform ----
    def _run_transform(self):
        csv_path = self.input_file_line.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "Внимание", "Выберите входной CSV файл.")
            return

        params = CoordinateTransformParameters(
            input_file=csv_path,
            x_origin=self.x_origin_spin.value(),
            y_origin=self.y_origin_spin.value(),
            rotation_angle=self.rotation_spin.value(),
            scale=self.scale_spin.value(),
            dt=self.dt_spin.value(),
        )

        self._executor = CoordinateTransformExecutor()
        ok, msg = self._executor.set_parameters(params)
        if not ok:
            QMessageBox.critical(self, "Ошибка", msg)
            return

        self._plot_unit_label = ""
        self.log_text.clear()
        self._log("Запуск преобразования координат...")
        self._log(f"  Вход: {csv_path}")
        self._log(f"  X_origin: {params.x_origin}, Y_origin: {params.y_origin}")
        self._log(f"  Угол поворота: {params.rotation_angle}°")
        self._log(f"  Масштаб: {params.scale} м/px")
        self._log(f"  dt: {params.dt} s")
        self._log("")
        self._start_worker()

    # ---- Run Plot ----
    def _run_plot(self):
        csv_path = self.plot_file_line.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "Внимание", "Выберите входной CSV файл.")
            return

        # Map color_by GUI value to transformed column name
        color_by_gui = self.color_by_combo.currentText()
        color_by_map = {"L": "L", "dx": "dx", "dy": "dy", "angle": "angle"}
        l_column_map = {"L": "L_ms", "dx": "dx_ms", "dy": "dy_ms", "angle": "L_ms"}
        colorbar_labels = {
            "L": "L (м/с)", "dx": "dx (м/с)", "dy": "dy (м/с)", "angle": "Угол (градусы)"
        }

        params = VectorPlotParameters(
            input_file=csv_path,
            arrow_scale=self.arrow_scale_spin.value(),
            arrow_width=self.arrow_width_spin.value(),
            colormap=self.plot_cmap_combo.currentText(),
            color_by=color_by_map[color_by_gui],
            colorbar_label=colorbar_labels[color_by_gui],
            x_column="X_mm",
            y_column="Y_mm",
            dx_column="dx_ms",
            dy_column="dy_ms",
            l_column="L_ms",
            title="Поле векторов (физические единицы)",
            xlabel="X (мм)",
            ylabel="Y (мм)",
            invert_y=False,
            suffix="_plot_phys",
        )

        self._executor = VectorPlotExecutor()
        ok, msg = self._executor.set_parameters(params)
        if not ok:
            QMessageBox.critical(self, "Ошибка", msg)
            return

        self._plot_unit_label = "м/с"
        self.log_text.clear()
        self._log("Запуск визуализации в физических единицах...")
        self._log(f"  Вход: {csv_path}")
        self._log(f"  arrow_scale: {params.arrow_scale}, arrow_width: {params.arrow_width}")
        self._log(f"  colormap: {params.colormap}, color_by: {params.color_by}")
        self._log("  оси: X (м), Y (м)")
        self._log("")
        self._start_worker()

    # ---- Common ----
    def _start_worker(self):
        self._set_running(True)
        self.progress_bar.setValue(0)

        self._worker = WorkerThread(self._executor, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _log(self, text: str):
        self.log_text.append(text)

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(int(pct))
        self.status_label.setText(msg)
        self._log(f"[{pct:.1f}%] {msg}")

    def _on_finished(self, result):
        self._set_running(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Готово" if result.success else "Ошибка")
        self._log("--- Результат ---")
        self._log(f"Успешно: {result.success}")

        # CoordinateTransformResult
        if hasattr(result, 'input_rows') and hasattr(result, 'output_rows'):
            self._log(f"Строк на входе: {result.input_rows}")
            self._log(f"Строк на выходе: {result.output_rows}")
            self._log(f"Файл вывода: {result.output_file}")
            if result.success and result.output_file:
                self._last_output_file = result.output_file
                self.plot_file_line.setText(result.output_file)

        # VectorPlotResult
        elif hasattr(result, 'vectors_count'):
            unit_suffix = f" {self._plot_unit_label}" if self._plot_unit_label else ""
            self._log(f"Построено векторов: {result.vectors_count}")
            self._log(f"dx: [{result.dx_min:.6f}, {result.dx_max:.6f}]{unit_suffix}")
            self._log(f"dy: [{result.dy_min:.6f}, {result.dy_max:.6f}]{unit_suffix}")
            self._log(f"L: [{result.l_min:.6f}, {result.l_max:.6f}]{unit_suffix}")
            self._log(f"Файл вывода: {result.output_file}")

        if result.errors:
            self._log(f"Ошибки: {result.errors}")

    def _on_error(self, msg):
        self._set_running(False)
        self.status_label.setText("Ошибка")
        self._log(f"ОШИБКА: {msg}")

    def _set_running(self, running: bool):
        self.raw_plot_run_btn.setEnabled(not running)
        self.transform_run_btn.setEnabled(not running)
        self.plot_run_btn.setEnabled(not running)


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------
class VectorPlotDialog(QDialog):
    def __init__(self, csv_path: str, parent=None):
        super().__init__(parent)
        self.csv_path = Path(csv_path)
        self._vectors = np.empty((0, 5), dtype=float)
        self._source_mode = "raw"
        self._load_error = ""
        self._origin_confirmed = False
        self._pick_preview = None
        self._plot_ax = None
        self._origin_vline = None
        self._origin_hline = None
        self._origin_marker = None
        self._origin_text_artist = None
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self._draw_plot)

        self.setWindowTitle(f"Векторный график: {self.csv_path.name}")
        self.resize(1150, 720)
        self._init_ui()
        self._load_csv()
        self._update_source_controls()
        self._connect_controls()
        self._schedule_redraw()

    @staticmethod
    def _make_axis_limit_spin():
        spin = QDoubleSpinBox()
        spin.setRange(-1000000000.0, 1000000000.0)
        spin.setDecimals(3)
        spin.setSingleStep(1.0)
        spin.setKeyboardTracking(False)
        return spin

    def _init_ui(self):
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        file_label = QLabel("CSV:")
        file_label.setFixedWidth(50)
        self.file_line = QLineEdit(str(self.csv_path))
        self.file_line.setReadOnly(True)
        self.save_btn = QPushButton("Сохранить PNG")
        header.addWidget(file_label)
        header.addWidget(self.file_line)
        header.addWidget(self.save_btn)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        controls = QWidget()
        controls.setMaximumWidth(340)
        controls_layout = QVBoxLayout(controls)

        transform_group = QGroupBox("Преобразование")
        transform_layout = QVBoxLayout(transform_group)

        origin_row = QHBoxLayout()
        origin_row.addWidget(QLabel("X_origin (px):"))
        self.x_origin_spin = QDoubleSpinBox()
        self.x_origin_spin.setRange(-100000.0, 100000.0)
        self.x_origin_spin.setDecimals(1)
        origin_row.addWidget(self.x_origin_spin)
        transform_layout.addLayout(origin_row)

        y_origin_row = QHBoxLayout()
        y_origin_row.addWidget(QLabel("Y_origin (px):"))
        self.y_origin_spin = QDoubleSpinBox()
        self.y_origin_spin.setRange(-100000.0, 100000.0)
        self.y_origin_spin.setDecimals(1)
        y_origin_row.addWidget(self.y_origin_spin)
        transform_layout.addLayout(y_origin_row)

        origin_actions_row = QHBoxLayout()
        self.pick_origin_btn = QPushButton("Выбрать origin")
        self.pick_origin_btn.setCheckable(True)
        self.confirm_origin_btn = QPushButton("Зафиксировать origin")
        origin_actions_row.addWidget(self.pick_origin_btn)
        origin_actions_row.addWidget(self.confirm_origin_btn)
        transform_layout.addLayout(origin_actions_row)

        self.origin_state_label = QLabel("Сначала задайте и зафиксируйте origin")
        self.origin_state_label.setWordWrap(True)
        transform_layout.addWidget(self.origin_state_label)

        units_row = QHBoxLayout()
        units_row.addWidget(QLabel("Единицы графика:"))
        self.units_combo = QComboBox()
        self.units_combo.addItems(["Пиксели", "мм и м/с"])
        units_row.addWidget(self.units_combo)
        transform_layout.addLayout(units_row)

        rotation_row = QHBoxLayout()
        rotation_row.addWidget(QLabel("Угол (°):"))
        self.rotation_spin = QSpinBox()
        self.rotation_spin.setRange(0, 360)
        rotation_row.addWidget(self.rotation_spin)
        transform_layout.addLayout(rotation_row)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Масштаб (м/px):"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(7)
        self.scale_spin.setRange(0.0000001, 1000.0)
        self.scale_spin.setSingleStep(0.0001)
        self.scale_spin.setValue(0.0000075)
        scale_row.addWidget(self.scale_spin)
        transform_layout.addLayout(scale_row)

        dt_row = QHBoxLayout()
        dt_row.addWidget(QLabel("dt (с):"))
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setDecimals(7)
        self.dt_spin.setRange(0.0000001, 1000.0)
        self.dt_spin.setSingleStep(0.0001)
        self.dt_spin.setValue(0.000002)
        dt_row.addWidget(self.dt_spin)
        transform_layout.addLayout(dt_row)
        controls_layout.addWidget(transform_group)

        plot_group = QGroupBox("График")
        plot_layout = QVBoxLayout(plot_group)

        arrow_scale_row = QHBoxLayout()
        arrow_scale_row.addWidget(QLabel("arrow_scale:"))
        self.arrow_scale_spin = QDoubleSpinBox()
        self.arrow_scale_spin.setRange(0.01, 1000.0)
        self.arrow_scale_spin.setDecimals(2)
        self.arrow_scale_spin.setValue(20.0)
        arrow_scale_row.addWidget(self.arrow_scale_spin)
        plot_layout.addLayout(arrow_scale_row)

        arrow_width_row = QHBoxLayout()
        arrow_width_row.addWidget(QLabel("arrow_width:"))
        self.arrow_width_spin = QDoubleSpinBox()
        self.arrow_width_spin.setRange(0.001, 0.05)
        self.arrow_width_spin.setSingleStep(0.001)
        self.arrow_width_spin.setDecimals(3)
        self.arrow_width_spin.setValue(0.003)
        arrow_width_row.addWidget(self.arrow_width_spin)
        plot_layout.addLayout(arrow_width_row)

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            "jet", "viridis", "plasma", "inferno", "magma", "cividis",
            "coolwarm", "RdYlBu", "Spectral"
        ])
        cmap_row.addWidget(self.cmap_combo)
        plot_layout.addLayout(cmap_row)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("color_by:"))
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(["L", "dx", "dy", "angle"])
        color_row.addWidget(self.color_by_combo)
        plot_layout.addLayout(color_row)

        crop_title = QLabel("Обрезка осей")
        plot_layout.addWidget(crop_title)

        crop_x_row = QHBoxLayout()
        self.crop_x_cb = QCheckBox("X")
        self.x_min_spin = self._make_axis_limit_spin()
        self.x_max_spin = self._make_axis_limit_spin()
        crop_x_row.addWidget(self.crop_x_cb)
        crop_x_row.addWidget(QLabel("min:"))
        crop_x_row.addWidget(self.x_min_spin)
        crop_x_row.addWidget(QLabel("max:"))
        crop_x_row.addWidget(self.x_max_spin)
        plot_layout.addLayout(crop_x_row)

        crop_y_row = QHBoxLayout()
        self.crop_y_cb = QCheckBox("Y")
        self.y_min_spin = self._make_axis_limit_spin()
        self.y_max_spin = self._make_axis_limit_spin()
        crop_y_row.addWidget(self.crop_y_cb)
        crop_y_row.addWidget(QLabel("min:"))
        crop_y_row.addWidget(self.y_min_spin)
        crop_y_row.addWidget(QLabel("max:"))
        crop_y_row.addWidget(self.y_max_spin)
        plot_layout.addLayout(crop_y_row)

        crop_actions_row = QHBoxLayout()
        self.fit_crop_btn = QPushButton("По данным")
        self.reset_crop_btn = QPushButton("Сбросить")
        crop_actions_row.addWidget(self.fit_crop_btn)
        crop_actions_row.addWidget(self.reset_crop_btn)
        plot_layout.addLayout(crop_actions_row)
        self._update_crop_controls()
        controls_layout.addWidget(plot_group)

        self.status_label = QLabel("Готово")
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch()

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        splitter.addWidget(controls)
        splitter.addWidget(self.canvas)
        splitter.setSizes([320, 830])
        layout.addWidget(splitter)

    def _connect_controls(self):
        self.x_origin_spin.valueChanged.connect(self._origin_changed)
        self.y_origin_spin.valueChanged.connect(self._origin_changed)
        for spin in (
            self.rotation_spin, self.scale_spin, self.dt_spin,
            self.arrow_scale_spin, self.arrow_width_spin,
            self.x_min_spin, self.x_max_spin, self.y_min_spin, self.y_max_spin
        ):
            spin.valueChanged.connect(self._schedule_redraw)
        self.crop_x_cb.stateChanged.connect(self._crop_toggled)
        self.crop_y_cb.stateChanged.connect(self._crop_toggled)
        self.fit_crop_btn.clicked.connect(self._fit_crop_to_data)
        self.reset_crop_btn.clicked.connect(self._reset_crop)
        self.units_combo.currentTextChanged.connect(self._units_changed)
        self.cmap_combo.currentTextChanged.connect(self._schedule_redraw)
        self.color_by_combo.currentTextChanged.connect(self._schedule_redraw)
        self.pick_origin_btn.toggled.connect(self._pick_origin_toggled)
        self.confirm_origin_btn.clicked.connect(self._confirm_origin)
        self.save_btn.clicked.connect(self._save_png)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)

    def _load_csv(self):
        if not self.csv_path.exists():
            self._load_error = f"Файл не существует: {self.csv_path}"
            return

        try:
            with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f, delimiter=";")
                fieldnames = reader.fieldnames or []
                mode, columns = self._detect_columns(fieldnames)
                if mode is None:
                    self._load_error = (
                        "Не удалось определить формат CSV. Ожидаются X0/Y0/dx/dy/L, "
                        "X_center/Y_center/dx_avg/dy_avg/L_avg или X_mm/Y_mm/dx_ms/dy_ms/L_ms."
                    )
                    return

                rows = []
                for row in reader:
                    try:
                        rows.append([self._to_float(row[column]) for column in columns])
                    except (KeyError, TypeError, ValueError):
                        continue

            if not rows:
                self._load_error = "В CSV нет числовых векторов для построения."
                return

            self._source_mode = mode
            self._vectors = np.array(rows, dtype=float)
        except OSError as exc:
            self._load_error = str(exc)

    def _detect_columns(self, fieldnames):
        names = set(fieldnames)
        if {"X_mm", "Y_mm", "dx_ms", "dy_ms", "L_ms"}.issubset(names):
            return "physical", ("X_mm", "Y_mm", "dx_ms", "dy_ms", "L_ms")
        if {"X_center", "Y_center", "dx_avg", "dy_avg", "L_avg"}.issubset(names):
            return "raw", ("X_center", "Y_center", "dx_avg", "dy_avg", "L_avg")
        if {"X0", "Y0", "dx", "dy", "L"}.issubset(names):
            return "raw", ("X0", "Y0", "dx", "dy", "L")
        return None, ()

    @staticmethod
    def _to_float(value):
        return float(str(value).replace(",", "."))

    def _is_pixel_origin_view(self):
        return self._source_mode != "physical" and self.units_combo.currentIndex() == 0

    def _origin_display_point(self):
        if self.pick_origin_btn.isChecked() and self._pick_preview is not None:
            return self._pick_preview
        return self.x_origin_spin.value(), self.y_origin_spin.value()

    @staticmethod
    def _format_origin_text(x, y):
        return f"X={x:.1f} px\nY={y:.1f} px"

    @staticmethod
    def _origin_text_position(ax, x, y):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        x_mid = (min(x0, x1) + max(x0, x1)) / 2.0
        y_mid = (min(y0, y1) + max(y0, y1)) / 2.0
        x_offset = 10 if x <= x_mid else -10
        y_offset = 10 if y <= y_mid else -10
        ha = "left" if x_offset > 0 else "right"
        va = "bottom" if y_offset > 0 else "top"
        return x_offset, y_offset, ha, va

    def _draw_origin_cursor(self, ax):
        x, y = self._origin_display_point()
        self._origin_vline = ax.axvline(
            x,
            color="red",
            linewidth=1.2,
            alpha=0.85,
            zorder=4,
        )
        self._origin_hline = ax.axhline(
            y,
            color="red",
            linewidth=1.2,
            alpha=0.85,
            zorder=4,
        )
        self._origin_marker = ax.scatter(
            [x],
            [y],
            marker="+",
            s=180,
            c="red",
            linewidths=2.0,
            zorder=5,
        )
        x_offset, y_offset, ha, va = self._origin_text_position(ax, x, y)
        self._origin_text_artist = ax.annotate(
            self._format_origin_text(x, y),
            xy=(x, y),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            color="red",
            fontsize=9,
            ha=ha,
            va=va,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "red", "alpha": 0.85},
            zorder=6,
        )

    def _update_origin_cursor(self, x, y):
        if not all((self._plot_ax, self._origin_vline, self._origin_hline, self._origin_marker, self._origin_text_artist)):
            self._schedule_redraw()
            return

        self._origin_vline.set_xdata([x, x])
        self._origin_hline.set_ydata([y, y])
        self._origin_marker.set_offsets(np.array([[x, y]], dtype=float))
        self._origin_text_artist.xy = (x, y)
        self._origin_text_artist.set_text(self._format_origin_text(x, y))
        x_offset, y_offset, ha, va = self._origin_text_position(self._plot_ax, x, y)
        self._origin_text_artist.set_position((x_offset, y_offset))
        self._origin_text_artist.set_ha(ha)
        self._origin_text_artist.set_va(va)
        self.canvas.draw_idle()

    def _is_valid_origin_event(self, event):
        return (
            self.pick_origin_btn.isChecked()
            and self._is_pixel_origin_view()
            and event.inaxes is self._plot_ax
            and event.xdata is not None
            and event.ydata is not None
        )

    def _origin_changed(self, *_args):
        if self._source_mode != "physical":
            self._origin_confirmed = False
            if self.pick_origin_btn.isChecked():
                self._pick_preview = (self.x_origin_spin.value(), self.y_origin_spin.value())
            else:
                self._pick_preview = None
            if self.rotation_spin.value() != 0:
                self.rotation_spin.blockSignals(True)
                self.rotation_spin.setValue(0)
                self.rotation_spin.blockSignals(False)
            self._update_source_controls()
        self._schedule_redraw()

    def _confirm_origin(self):
        if self._source_mode == "physical":
            return
        self._origin_confirmed = True
        self._update_source_controls()
        self._schedule_redraw()

    def _units_changed(self, *_args):
        self._reset_crop(schedule=False)
        self._update_source_controls()
        self._schedule_redraw()

    def _crop_toggled(self, *_args):
        self._update_crop_controls()
        self._schedule_redraw()

    def _update_crop_controls(self):
        x_enabled = self.crop_x_cb.isChecked()
        y_enabled = self.crop_y_cb.isChecked()
        self.x_min_spin.setEnabled(x_enabled)
        self.x_max_spin.setEnabled(x_enabled)
        self.y_min_spin.setEnabled(y_enabled)
        self.y_max_spin.setEnabled(y_enabled)

    @staticmethod
    def _set_checked_silent(checkbox, checked):
        checkbox.blockSignals(True)
        checkbox.setChecked(checked)
        checkbox.blockSignals(False)

    @staticmethod
    def _set_spin_silent(spin, value):
        spin.blockSignals(True)
        spin.setValue(float(value))
        spin.blockSignals(False)

    @staticmethod
    def _data_limits(values):
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return -1.0, 1.0
        min_value = float(finite_values.min())
        max_value = float(finite_values.max())
        if math.isclose(min_value, max_value):
            padding = max(abs(min_value) * 0.05, 1.0)
            return min_value - padding, max_value + padding
        return min_value, max_value

    def _fit_crop_to_data(self):
        if self._load_error or self._vectors.size == 0:
            return
        x, y, *_ = self._transformed_vectors()
        x_min, x_max = self._data_limits(x)
        y_min, y_max = self._data_limits(y)

        self._set_checked_silent(self.crop_x_cb, True)
        self._set_checked_silent(self.crop_y_cb, True)
        self._set_spin_silent(self.x_min_spin, x_min)
        self._set_spin_silent(self.x_max_spin, x_max)
        self._set_spin_silent(self.y_min_spin, y_min)
        self._set_spin_silent(self.y_max_spin, y_max)
        self._update_crop_controls()
        self._schedule_redraw()

    def _reset_crop(self, schedule=True):
        self._set_checked_silent(self.crop_x_cb, False)
        self._set_checked_silent(self.crop_y_cb, False)
        self._update_crop_controls()
        if schedule:
            self._schedule_redraw()

    def _axis_unit_label(self):
        return "px" if self._is_pixel_origin_view() else "мм"

    def _apply_axis_crop(self, ax):
        status_lines = []
        crop_applied = False
        unit = self._axis_unit_label()

        if self.crop_x_cb.isChecked():
            x_min = min(self.x_min_spin.value(), self.x_max_spin.value())
            x_max = max(self.x_min_spin.value(), self.x_max_spin.value())
            if math.isclose(x_min, x_max):
                status_lines.append("Обрезка X не применена: min = max")
            else:
                ax.set_xlim(x_min, x_max)
                crop_applied = True
                status_lines.append(f"Обрезка X: [{x_min:.3f}, {x_max:.3f}] {unit}")

        if self.crop_y_cb.isChecked():
            y_min = min(self.y_min_spin.value(), self.y_max_spin.value())
            y_max = max(self.y_min_spin.value(), self.y_max_spin.value())
            if math.isclose(y_min, y_max):
                status_lines.append("Обрезка Y не применена: min = max")
            else:
                ax.set_ylim(y_min, y_max)
                crop_applied = True
                status_lines.append(f"Обрезка Y: [{y_min:.3f}, {y_max:.3f}] {unit}")

        return status_lines, crop_applied

    def _axis_crop_mask(self, x, y):
        mask = np.ones(len(x), dtype=bool)

        if self.crop_x_cb.isChecked():
            x_min = min(self.x_min_spin.value(), self.x_max_spin.value())
            x_max = max(self.x_min_spin.value(), self.x_max_spin.value())
            if not math.isclose(x_min, x_max):
                mask &= (x >= x_min) & (x <= x_max)

        if self.crop_y_cb.isChecked():
            y_min = min(self.y_min_spin.value(), self.y_max_spin.value())
            y_max = max(self.y_min_spin.value(), self.y_max_spin.value())
            if not math.isclose(y_min, y_max):
                mask &= (y >= y_min) & (y <= y_max)

        return mask

    @staticmethod
    def _format_csv_value(value):
        return f"{float(value):.10g}"

    def _save_current_csv(self, csv_path):
        x, y, dx, dy, length, *_labels, unit = self._transformed_vectors()
        mask = self._axis_crop_mask(x, y)
        x = x[mask]
        y = y[mask]
        dx = dx[mask]
        dy = dy[mask]
        length = length[mask]
        angle = np.degrees(np.arctan2(dy, dx))

        if unit == "px":
            fieldnames = ["X0", "Y0", "dx", "dy", "L", "angle_deg"]
        else:
            fieldnames = ["X_mm", "Y_mm", "dx_ms", "dy_ms", "L_ms", "angle_deg"]

        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(fieldnames)
            for row in zip(x, y, dx, dy, length, angle):
                writer.writerow([self._format_csv_value(value) for value in row])

        return len(x)

    def _pick_origin_toggled(self, checked):
        if self._source_mode == "physical":
            self.pick_origin_btn.blockSignals(True)
            self.pick_origin_btn.setChecked(False)
            self.pick_origin_btn.blockSignals(False)
            self._pick_preview = None
            return

        if checked:
            if self.units_combo.currentIndex() != 0:
                self.units_combo.setCurrentIndex(0)
            self._pick_preview = (self.x_origin_spin.value(), self.y_origin_spin.value())
            self.status_label.setText("Кликните по точке origin на графике в пикселях.")
        else:
            self._pick_preview = None
        self._schedule_redraw()

    def _on_canvas_click(self, event):
        if not self._is_valid_origin_event(event):
            return

        self.x_origin_spin.blockSignals(True)
        self.y_origin_spin.blockSignals(True)
        self.x_origin_spin.setValue(float(event.xdata))
        self.y_origin_spin.setValue(float(event.ydata))
        self.x_origin_spin.blockSignals(False)
        self.y_origin_spin.blockSignals(False)
        if self.rotation_spin.value() != 0:
            self.rotation_spin.blockSignals(True)
            self.rotation_spin.setValue(0)
            self.rotation_spin.blockSignals(False)

        self.pick_origin_btn.setChecked(False)
        self._origin_confirmed = True
        self._update_source_controls()
        self._schedule_redraw()

    def _on_canvas_motion(self, event):
        if not self._is_valid_origin_event(event):
            return
        x = float(event.xdata)
        y = float(event.ydata)
        self._pick_preview = (x, y)
        self._update_origin_cursor(x, y)

    def _update_source_controls(self):
        is_physical = self._source_mode == "physical"
        if is_physical:
            self.units_combo.blockSignals(True)
            self.units_combo.setCurrentIndex(1)
            self.units_combo.blockSignals(False)

        is_pixel_view = not is_physical and self.units_combo.currentIndex() == 0
        if not is_pixel_view and self.pick_origin_btn.isChecked():
            self.pick_origin_btn.blockSignals(True)
            self.pick_origin_btn.setChecked(False)
            self.pick_origin_btn.blockSignals(False)
            self._pick_preview = None

        self.units_combo.setEnabled(not is_physical)
        self.x_origin_spin.setEnabled(not is_physical)
        self.y_origin_spin.setEnabled(not is_physical)
        self.confirm_origin_btn.setEnabled(not is_physical)
        self.pick_origin_btn.setEnabled(is_pixel_view)
        self.rotation_spin.setEnabled(not is_physical and self._origin_confirmed and not is_pixel_view)
        self.scale_spin.setEnabled(not is_physical and not is_pixel_view)
        self.dt_spin.setEnabled(not is_physical and not is_pixel_view)

        if is_physical:
            self.origin_state_label.setText("CSV уже содержит координаты в мм и скорости в м/с")
        elif self._origin_confirmed and is_pixel_view:
            self.origin_state_label.setText("Origin зафиксирован. Переключитесь в мм и м/с, чтобы менять угол")
        elif self._origin_confirmed:
            self.origin_state_label.setText("Origin зафиксирован, угол можно менять")
        else:
            self.origin_state_label.setText("Сначала задайте и зафиксируйте origin")

    def _schedule_redraw(self, *_args):
        self._redraw_timer.start(120)

    def _transformed_vectors(self):
        x = self._vectors[:, 0]
        y = self._vectors[:, 1]
        dx = self._vectors[:, 2]
        dy = self._vectors[:, 3]
        length = self._vectors[:, 4]

        if self._source_mode == "physical":
            return x, y, dx, dy, length, "X (мм)", "Y (мм)", "м/с"

        if self.units_combo.currentIndex() == 0:
            return x, y, dx, dy, length, "X (px)", "Y (px)", "px"

        theta = math.radians(self.rotation_spin.value())
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        scale = self.scale_spin.value()
        dt = self.dt_spin.value()

        x_rel = x - self.x_origin_spin.value()
        y_rel = y - self.y_origin_spin.value()
        x_rot = x_rel * cos_theta - y_rel * sin_theta
        y_rot = x_rel * sin_theta + y_rel * cos_theta
        dx_rot = dx * cos_theta - dy * sin_theta
        dy_rot = dx * sin_theta + dy * cos_theta

        x_mm = x_rot * scale * 1000.0
        y_mm = y_rot * scale * 1000.0
        dx_ms = dx_rot * scale / dt
        dy_ms = dy_rot * scale / dt
        l_ms = length * scale / dt
        return x_mm, y_mm, dx_ms, dy_ms, l_ms, "X (мм)", "Y (мм)", "м/с"

    def _draw_plot(self):
        self._plot_ax = None
        self._origin_vline = None
        self._origin_hline = None
        self._origin_marker = None
        self._origin_text_artist = None
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self._plot_ax = ax

        if self._load_error:
            ax.text(0.5, 0.5, self._load_error, ha="center", va="center", wrap=True)
            ax.set_axis_off()
            self.status_label.setText(self._load_error)
            self.canvas.draw_idle()
            return

        x, y, dx, dy, length, x_label, y_label, unit = self._transformed_vectors()
        angle = np.degrees(np.arctan2(dy, dx))
        color_by = self.color_by_combo.currentText()

        if color_by == "dx":
            colors = dx
            colorbar_label = f"dx ({unit})"
        elif color_by == "dy":
            colors = dy
            colorbar_label = f"dy ({unit})"
        elif color_by == "angle":
            colors = angle
            colorbar_label = "Угол (градусы)"
        else:
            colors = length
            colorbar_label = f"L ({unit})"

        quiver = ax.quiver(
            x, y, dx, dy, colors,
            cmap=self.cmap_combo.currentText(),
            scale=10.0 * self.arrow_scale_spin.value(),
            width=self.arrow_width_spin.value(),
            headwidth=4.0,
            headlength=5.0,
        )
        self.figure.colorbar(quiver, ax=ax, label=colorbar_label)
        ax.set_title(self.csv_path.name)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        crop_status_lines, crop_applied = self._apply_axis_crop(ax)
        ax.set_aspect("equal", adjustable="box" if crop_applied else "datalim")
        ax.grid(True, alpha=0.3)
        if self._source_mode != "physical" and self.units_combo.currentIndex() == 0:
            self._draw_origin_cursor(ax)
        self.figure.tight_layout()
        self.canvas.draw_idle()

        status_lines = []
        if self.pick_origin_btn.isChecked():
            status_lines.append("Кликните по точке origin на графике в пикселях")
            preview_x, preview_y = self._origin_display_point()
            status_lines.append(f"курсор: X={preview_x:.1f} px, Y={preview_y:.1f} px")
        if self._source_mode != "physical":
            status_lines.append(
                f"origin: X={self.x_origin_spin.value():.1f} px, "
                f"Y={self.y_origin_spin.value():.1f} px"
            )
            if not self._origin_confirmed:
                status_lines.append("Угол заблокирован до фиксации origin")
        status_lines.extend(crop_status_lines)
        status_lines.extend([
            f"Векторов: {len(x)}",
            f"dx: [{dx.min():.6f}, {dx.max():.6f}] {unit}",
            f"dy: [{dy.min():.6f}, {dy.max():.6f}] {unit}",
            f"L: [{length.min():.6f}, {length.max():.6f}] {unit}",
        ])
        self.status_label.setText("\n".join(status_lines))

    def _save_png(self):
        default_path = self.csv_path.with_name(f"{self.csv_path.stem}_vector_plot.png")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить график",
            str(default_path),
            "PNG files (*.png);;All files (*.*)",
        )
        if not file_path:
            return
        png_path = Path(file_path)
        csv_path = png_path.with_suffix(".csv")
        self.figure.savefig(str(png_path), dpi=150, bbox_inches="tight")
        try:
            vector_count = self._save_current_csv(csv_path)
        except OSError as exc:
            QMessageBox.warning(self, "Ошибка", f"PNG сохранён, но CSV сохранить не удалось:\n{exc}")
            self.status_label.setText(f"Сохранено: {png_path}\nCSV не сохранён: {exc}")
            return
        self.status_label.setText(
            f"Сохранено: {png_path}\n"
            f"CSV: {csv_path}\n"
            f"Векторов в CSV: {vector_count}"
        )


class VectorGraphicsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        group = QGroupBox("Векторный график")
        group_layout = QVBoxLayout(group)

        row = QHBoxLayout()
        label = QLabel("CSV:")
        label.setFixedWidth(110)
        self.csv_line = QLineEdit()
        self.csv_line.setToolTip("CSV с векторами: pair_sum, averaged или transformed.")
        self.browse_btn = QPushButton("Обзор...")
        self.open_btn = QPushButton("Открыть график")
        row.addWidget(label)
        row.addWidget(self.csv_line)
        row.addWidget(self.browse_btn)
        row.addWidget(self.open_btn)
        group_layout.addLayout(row)

        self.status_label = QLabel("Готово")
        group_layout.addWidget(self.status_label)
        layout.addWidget(group)
        layout.addStretch()

        self.browse_btn.clicked.connect(self._browse_csv)
        self.open_btn.clicked.connect(self._open_dialog)
        self.csv_line.returnPressed.connect(self._open_dialog)

    def _browse_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите CSV с векторами",
            "",
            "CSV files (*.csv);;All files (*.*)",
        )
        if not file_path:
            return
        self.csv_line.setText(file_path)
        self._open_dialog()

    def _open_dialog(self):
        csv_path = self.csv_line.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "Внимание", "Выберите CSV файл.")
            return
        if not Path(csv_path).exists():
            QMessageBox.warning(self, "Внимание", "CSV файл не найден.")
            return
        self.status_label.setText(Path(csv_path).name)
        dialog = VectorPlotDialog(csv_path, self)
        dialog.exec_()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ParticleAnalysis")
        self.resize(900, 650)

        tabs = QTabWidget()
        self.sort_binarize_tab = SortBinarizeTab()
        self.prepare_experiments_tab = PrepareExperimentsTab(self.sort_binarize_tab, tabs)
        tabs.addTab(self.prepare_experiments_tab, "Подготовка экспериментов")
        tabs.addTab(self.sort_binarize_tab, "Сортировка + бинаризация")
        tabs.addTab(PTVAnalysisTab(), "PTV анализ")
        tabs.addTab(PTVViewerTab(), "PTV результаты")
        tabs.addTab(PTVProcessingTab(), "PTV обработка")
        tabs.addTab(VectorGraphicsTab(), "Векторные графики")
        self.setCentralWidget(tabs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
