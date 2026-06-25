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
from datetime import datetime
import shutil
import cv2

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
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRectF
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
        h.addWidget(QLabel("Порог:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 65535)
        self.threshold_spin.setValue(2000)
        self.threshold_spin.setSingleStep(100)
        h.addWidget(self.threshold_spin)
        h.addStretch()
        layout.addLayout(h)

        # Validate format
        self.validate_cb = QCheckBox("Проверять формат (16-bit PNG)")
        self.validate_cb.setChecked(True)
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
        det_layout = QVBoxLayout(det_group)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("min_area:"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 1000)
        self.min_area_spin.setValue(4)
        h1.addWidget(self.min_area_spin)
        h1.addWidget(QLabel("max_area:"))
        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(1, 1000)
        self.max_area_spin.setValue(150)
        h1.addWidget(self.max_area_spin)
        h1.addStretch()
        det_layout.addLayout(h1)
        layout.addWidget(det_group)

        # Matching parameters
        match_group = QGroupBox("Параметры сопоставления")
        match_layout = QVBoxLayout(match_group)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("max_distance:"))
        self.max_dist_spin = QDoubleSpinBox()
        self.max_dist_spin.setRange(1.0, 100.0)
        self.max_dist_spin.setValue(50.0)
        h2.addWidget(self.max_dist_spin)
        h2.addWidget(QLabel("max_diameter_diff:"))
        self.max_diam_spin = QDoubleSpinBox()
        self.max_diam_spin.setRange(0.0, 10.0)
        self.max_diam_spin.setValue(4.0)
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


class PTVViewerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._records = []
        self._visualizer = PTVVisualizer()
        self._preview_image = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        ptv_layout, self.ptv_line = _folder_row("Папка PTV:", self)
        layout.addLayout(ptv_layout)

        original_layout, self.original_line = _folder_row("Папка кадров:", self)
        self.original_line.setPlaceholderText("Необязательно; определяется из PTV_<threshold>")
        layout.addLayout(original_layout)

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
        self.preview_btn.clicked.connect(self._preview_selected)
        self.open_btn.clicked.connect(self._open_large)
        self.save_btn.clicked.connect(self._save_selected)
        self.back_btn.clicked.connect(self.preview_view.zoom_back)
        self.zoom_out_btn.clicked.connect(self.preview_view.zoom_out)
        self.reset_zoom_btn.clicked.connect(self.preview_view.reset_zoom)
        self.table.itemSelectionChanged.connect(self._preview_selected)
        self.table.itemDoubleClicked.connect(lambda _item: self._open_large())

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
        self.preview_btn.setEnabled(bool(self._records) and visualizer_ready)
        self.save_btn.setEnabled(bool(self._records) and visualizer_ready)
        self.open_btn.setEnabled(self._preview_image is not None)
        self.back_btn.setEnabled(self._preview_image is not None)
        self.zoom_out_btn.setEnabled(self._preview_image is not None)
        self.reset_zoom_btn.setEnabled(self._preview_image is not None)
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
        if not record.source_ok:
            self.info_label.setText("Исходные кадры не найдены")
            self._set_preview_labels(None)
            self.open_btn.setEnabled(False)
            self.back_btn.setEnabled(False)
            self.zoom_out_btn.setEnabled(False)
            self.reset_zoom_btn.setEnabled(False)
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
            self.open_btn.setEnabled(False)
            self.back_btn.setEnabled(False)
            self.zoom_out_btn.setEnabled(False)
            self.reset_zoom_btn.setEnabled(False)
            return

        self.info_label.setText(
            f"{record.camera}, пара {record.pair_number}: "
            f"совпадений {record.matches_count}, средн. L={record.mean_l:.2f}, макс. L={record.max_l:.2f}"
        )
        self._set_preview_labels(preview)
        self.open_btn.setEnabled(True)
        self.back_btn.setEnabled(True)
        self.zoom_out_btn.setEnabled(True)
        self.reset_zoom_btn.setEnabled(True)

    def _set_preview_labels(self, preview):
        if preview is None:
            self.preview_view.clear_image()
            return
        self.preview_view.set_image(preview)

    def _open_large(self):
        record = self._selected_record()
        if record is None:
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
        run_folder = self._make_run_folder()
        pair_sum_folder = run_folder / "01_pair_sum"
        filtered_folder = run_folder / "02_filtered"
        averaged_folder = run_folder / "03_averaged"
        pair_sum_folder.mkdir(parents=True, exist_ok=True)
        filtered_folder.mkdir(parents=True, exist_ok=True)
        averaged_folder.mkdir(parents=True, exist_ok=True)

        camera_results = []
        for camera, input_file in self.pair_sum_jobs:
            source_path = Path(input_file)
            saved_source = pair_sum_folder / f"{camera}_pair_sum{source_path.suffix or '.csv'}"
            item = {
                "camera": camera,
                "source_file": str(source_path),
                "saved_source_file": str(saved_source),
                "filter_result": None,
                "average_result": None,
                "success": False,
                "errors": [],
            }

            try:
                shutil.copy2(source_path, saved_source)
                filter_params = VectorFilterParameters(
                    input_file=str(saved_source),
                    output_folder=str(filtered_folder),
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
                    output_folder=str(averaged_folder),
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

        return PTVProcessingPipelineResult(camera_results, str(run_folder))

    def _make_run_folder(self) -> Path:
        base_folder = Path(self.pair_sum_jobs[0][1]).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = base_folder / f"ptv_processing_{timestamp}"
        counter = 2
        while candidate.exists():
            candidate = base_folder / f"ptv_processing_{timestamp}_{counter}"
            counter += 1
        return candidate


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
        filt_layout = QVBoxLayout(filt_group)

        f_cam1_row, self.filt_cam1_file_line = _file_row("cam_1 CSV:", self)
        filt_layout.addLayout(f_cam1_row)
        f_cam2_row, self.filt_cam2_file_line = _file_row("cam_2 CSV:", self)
        filt_layout.addLayout(f_cam2_row)

        # U range
        h_u = QHBoxLayout()
        self.filter_u_cb = QCheckBox("Фильтр U")
        self.filter_u_cb.setChecked(True)
        h_u.addWidget(self.filter_u_cb)
        h_u.addWidget(QLabel("min:"))
        self.u_min_spin = QDoubleSpinBox()
        self.u_min_spin.setRange(-10000.0, 10000.0)
        self.u_min_spin.setValue(0.0)
        h_u.addWidget(self.u_min_spin)
        h_u.addWidget(QLabel("max:"))
        self.u_max_spin = QDoubleSpinBox()
        self.u_max_spin.setRange(-10000.0, 10000.0)
        self.u_max_spin.setValue(40.0)
        h_u.addWidget(self.u_max_spin)
        h_u.addStretch()
        filt_layout.addLayout(h_u)

        # V range
        h_v = QHBoxLayout()
        self.filter_v_cb = QCheckBox("Фильтр V")
        self.filter_v_cb.setChecked(True)
        h_v.addWidget(self.filter_v_cb)
        h_v.addWidget(QLabel("min:"))
        self.v_min_spin = QDoubleSpinBox()
        self.v_min_spin.setRange(-10000.0, 10000.0)
        self.v_min_spin.setValue(-10.0)
        h_v.addWidget(self.v_min_spin)
        h_v.addWidget(QLabel("max:"))
        self.v_max_spin = QDoubleSpinBox()
        self.v_max_spin.setRange(-10000.0, 10000.0)
        self.v_max_spin.setValue(10.0)
        h_v.addWidget(self.v_max_spin)
        h_v.addStretch()
        filt_layout.addLayout(h_v)

        layout.addWidget(filt_group)

        # --- 3b. Vector Average ---
        avg_group = QGroupBox("2. Усреднение векторов")
        avg_layout = QVBoxLayout(avg_group)

        h_plane = QHBoxLayout()
        h_plane.addWidget(QLabel("plane_width:"))
        self.plane_w_spin = QDoubleSpinBox()
        self.plane_w_spin.setRange(1.0, 100000.0)
        self.plane_w_spin.setValue(4904.0)
        self.plane_w_spin.setDecimals(1)
        h_plane.addWidget(self.plane_w_spin)
        h_plane.addWidget(QLabel("plane_height:"))
        self.plane_h_spin = QDoubleSpinBox()
        self.plane_h_spin.setRange(1.0, 100000.0)
        self.plane_h_spin.setValue(3280.0)
        self.plane_h_spin.setDecimals(1)
        h_plane.addWidget(self.plane_h_spin)
        h_plane.addStretch()
        avg_layout.addLayout(h_plane)

        h_cell = QHBoxLayout()
        h_cell.addWidget(QLabel("cell_width:"))
        self.cell_w_spin = QDoubleSpinBox()
        self.cell_w_spin.setRange(1.0, 10000.0)
        self.cell_w_spin.setValue(66.0)
        self.cell_w_spin.setDecimals(1)
        h_cell.addWidget(self.cell_w_spin)
        h_cell.addWidget(QLabel("cell_height:"))
        self.cell_h_spin = QDoubleSpinBox()
        self.cell_h_spin.setRange(1.0, 10000.0)
        self.cell_h_spin.setValue(66.0)
        self.cell_h_spin.setDecimals(1)
        h_cell.addWidget(self.cell_h_spin)
        h_cell.addWidget(QLabel("min_points:"))
        self.min_pts_spin = QSpinBox()
        self.min_pts_spin.setRange(1, 1000)
        self.min_pts_spin.setValue(1)
        h_cell.addWidget(self.min_pts_spin)
        h_cell.addStretch()
        avg_layout.addLayout(h_cell)

        self.avg_run_btn = QPushButton("Запустить фильтр + усреднение")
        avg_layout.addWidget(self.avg_run_btn)
        layout.addWidget(avg_group)

        result_group = QGroupBox("Результаты обработки")
        result_layout = QVBoxLayout(result_group)
        self.processing_results_table = QTableWidget()
        self.processing_results_table.setColumnCount(8)
        self.processing_results_table.setHorizontalHeaderLabels([
            "Камера", "pair_sum", "После фильтра", "После усреднения",
            "Вход", "После фильтра", "Удалено %", "Ячейки"
        ])
        self.processing_results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.processing_results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.processing_results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.processing_results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.processing_results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        result_layout.addWidget(self.processing_results_table)
        layout.addWidget(result_group)

        # --- 3c. Vector Plot ---
        plot_group = QGroupBox("3. Визуализация векторов")
        plot_layout = QVBoxLayout(plot_group)

        p_row, self.plot_file_line = _file_row("Входной CSV:", self)
        plot_layout.addLayout(p_row)

        h_plot1 = QHBoxLayout()
        h_plot1.addWidget(QLabel("arrow_scale:"))
        self.arrow_scale_spin = QDoubleSpinBox()
        self.arrow_scale_spin.setRange(0.01, 1000.0)
        self.arrow_scale_spin.setValue(20.0)
        self.arrow_scale_spin.setDecimals(2)
        h_plot1.addWidget(self.arrow_scale_spin)
        h_plot1.addWidget(QLabel("arrow_width:"))
        self.arrow_width_spin = QDoubleSpinBox()
        self.arrow_width_spin.setRange(0.001, 0.05)
        self.arrow_width_spin.setValue(0.003)
        self.arrow_width_spin.setSingleStep(0.001)
        self.arrow_width_spin.setDecimals(3)
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
        h_plot2.addWidget(self.plot_cmap_combo)
        h_plot2.addWidget(QLabel("color_by:"))
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(["L", "dx", "dy", "angle"])
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
        self.avg_run_btn.clicked.connect(self._run_filter_average)
        self.plot_run_btn.clicked.connect(self._run_plot)

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

    # ---- Run Plot ----
    def _run_plot(self):
        csv_path = self.plot_file_line.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "Внимание", "Выберите входной CSV файл.")
            return

        params = VectorPlotParameters(
            input_file=csv_path,
            arrow_scale=self.arrow_scale_spin.value(),
            arrow_width=self.arrow_width_spin.value(),
            colormap=self.plot_cmap_combo.currentText(),
            color_by=self.color_by_combo.currentText(),
            invert_y=False,
        )

        self._executor = VectorPlotExecutor()
        ok, msg = self._executor.set_parameters(params)
        if not ok:
            QMessageBox.critical(self, "Ошибка", msg)
            return

        self.log_text.clear()
        self._log("Запуск визуализации векторов...")
        self._log(f"  Вход: {csv_path}")
        self._log(f"  arrow_scale: {params.arrow_scale}, arrow_width: {params.arrow_width}")
        self._log(f"  colormap: {params.colormap}, color_by: {params.color_by}")
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
            self._log(f"Папка вывода: {result.output_folder}")
            self._log(f"Векторов на входе: {result.input_vectors}")
            self._log(f"Векторов после фильтра: {result.output_vectors}")
            self._log(f"Удалено: {result.vectors_removed} ({result.removal_percentage:.1f}%)")
            self._log(f"Ячеек после усреднения: {result.output_cells}")
            for item in result.camera_results:
                self._log(f"--- {item['camera']} ---")
                self._log(f"Копия pair_sum: {item['saved_source_file']}")
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
        self.plot_run_btn.setEnabled(not running)


# ---------------------------------------------------------------------------
# Tab 4: Coordinate Transform
# ---------------------------------------------------------------------------
class CoordinateTransformTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._executor = None
        self._last_output_file = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- 1. Coordinate Transform ---
        transform_group = QGroupBox("1. Преобразование координат")
        transform_layout = QVBoxLayout(transform_group)

        # Input CSV
        f_row, self.input_file_line = _file_row("Входной CSV:", self)
        transform_layout.addLayout(f_row)

        # Origin
        h_origin = QHBoxLayout()
        h_origin.addWidget(QLabel("X_origin (px):"))
        self.x_origin_spin = QDoubleSpinBox()
        self.x_origin_spin.setRange(-100000.0, 100000.0)
        self.x_origin_spin.setValue(0.0)
        self.x_origin_spin.setDecimals(1)
        h_origin.addWidget(self.x_origin_spin)
        h_origin.addWidget(QLabel("Y_origin (px):"))
        self.y_origin_spin = QDoubleSpinBox()
        self.y_origin_spin.setRange(-100000.0, 100000.0)
        self.y_origin_spin.setValue(0.0)
        self.y_origin_spin.setDecimals(1)
        h_origin.addWidget(self.y_origin_spin)
        h_origin.addStretch()
        transform_layout.addLayout(h_origin)

        # Rotation
        h_rotation = QHBoxLayout()
        h_rotation.addWidget(QLabel("Угол поворота (°):"))
        self.rotation_spin = QSpinBox()
        self.rotation_spin.setRange(0, 360)
        self.rotation_spin.setValue(0)
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
        h_scale.addWidget(self.scale_spin)
        h_scale.addWidget(QLabel("dt (с):"))
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setDecimals(7)
        self.dt_spin.setRange(0.0000001, 1000.0)
        self.dt_spin.setSingleStep(0.0001)
        self.dt_spin.setValue(0.000002)
        h_scale.addWidget(self.dt_spin)
        h_scale.addStretch()
        transform_layout.addLayout(h_scale)

        self.transform_run_btn = QPushButton("Преобразовать")
        transform_layout.addWidget(self.transform_run_btn)
        layout.addWidget(transform_group)

        # --- 2. Vector Plot (physical units) ---
        plot_group = QGroupBox("2. Визуализация векторов (м, м/с)")
        plot_layout = QVBoxLayout(plot_group)

        p_row, self.plot_file_line = _file_row("Входной CSV:", self)
        plot_layout.addLayout(p_row)

        h_plot1 = QHBoxLayout()
        h_plot1.addWidget(QLabel("arrow_scale:"))
        self.arrow_scale_spin = QDoubleSpinBox()
        self.arrow_scale_spin.setRange(0.01, 1000.0)
        self.arrow_scale_spin.setValue(20.0)
        self.arrow_scale_spin.setDecimals(2)
        h_plot1.addWidget(self.arrow_scale_spin)
        h_plot1.addWidget(QLabel("arrow_width:"))
        self.arrow_width_spin = QDoubleSpinBox()
        self.arrow_width_spin.setRange(0.001, 0.05)
        self.arrow_width_spin.setValue(0.003)
        self.arrow_width_spin.setSingleStep(0.001)
        self.arrow_width_spin.setDecimals(3)
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
        h_plot2.addWidget(self.plot_cmap_combo)
        h_plot2.addWidget(QLabel("color_by:"))
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(["L", "dx", "dy", "angle"])
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
        self.transform_run_btn.clicked.connect(self._run_transform)
        self.plot_run_btn.clicked.connect(self._run_plot)

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
            self._log(f"Построено векторов: {result.vectors_count}")
            self._log(f"dx: [{result.dx_min:.6f}, {result.dx_max:.6f}] m/s")
            self._log(f"dy: [{result.dy_min:.6f}, {result.dy_max:.6f}] m/s")
            self._log(f"L: [{result.l_min:.6f}, {result.l_max:.6f}] m/s")
            self._log(f"Файл вывода: {result.output_file}")

        if result.errors:
            self._log(f"Ошибки: {result.errors}")

    def _on_error(self, msg):
        self._set_running(False)
        self.status_label.setText("Ошибка")
        self._log(f"ОШИБКА: {msg}")

    def _set_running(self, running: bool):
        self.transform_run_btn.setEnabled(not running)
        self.plot_run_btn.setEnabled(not running)


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------
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
        tabs.addTab(CoordinateTransformTab(), "Преобразование координат")
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
