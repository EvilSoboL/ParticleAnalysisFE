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

# Добавление корня проекта в sys.path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QProgressBar, QTextEdit, QGroupBox, QFileDialog, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

from execute.execute_filter.execute_sort_binarize import (
    SortBinarizeExecutor, SortBinarizeParameters
)
from execute.execute_analysis.execute_ptv_analysis import (
    PTVExecutor, PTVParameters
)
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
    btn = QPushButton("Browse...")
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
    btn = QPushButton("Browse...")
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
# Tab 1: Sort + Binarize
# ---------------------------------------------------------------------------
class SortBinarizeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._executor = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Input folder
        folder_layout, self.input_line = _folder_row("Input folder:", self)
        layout.addLayout(folder_layout)

        # Threshold
        h = QHBoxLayout()
        h.addWidget(QLabel("Threshold:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 65535)
        self.threshold_spin.setValue(2000)
        self.threshold_spin.setSingleStep(100)
        h.addWidget(self.threshold_spin)
        h.addStretch()
        layout.addLayout(h)

        # Validate format
        self.validate_cb = QCheckBox("Validate format (16-bit PNG)")
        self.validate_cb.setChecked(True)
        layout.addWidget(self.validate_cb)

        # Run / Cancel
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
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
            QMessageBox.warning(self, "Warning", "Select input folder.")
            return

        params = SortBinarizeParameters(
            input_folder=folder,
            threshold=self.threshold_spin.value(),
            validate_format=self.validate_cb.isChecked(),
        )

        self._executor = SortBinarizeExecutor()
        ok, msg = self._executor.set_parameters(params)
        if not ok:
            QMessageBox.critical(self, "Error", msg)
            return

        self._set_running(True)
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self._log(f"Starting Sort + Binarize...")
        self._log(f"  Input: {folder}")
        self._log(f"  Threshold: {self.threshold_spin.value()}")
        self._log(f"  Validate: {self.validate_cb.isChecked()}")
        self._log("")

        self._worker = WorkerThread(self._executor, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _cancel(self):
        if self._executor:
            self._executor.cancel()
        self._log("Cancellation requested...")

    def _log(self, text: str):
        self.log_text.append(text)

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(int(pct))
        self.status_label.setText(msg)
        self._log(f"[{pct:.1f}%] {msg}")

    def _on_finished(self, result):
        self._set_running(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Done" if result.success else "Failed")
        self._log("--- Result ---")
        self._log(f"Success: {result.success}")
        self._log(f"Cam1 count: {result.cam1_count}")
        self._log(f"Cam2 count: {result.cam2_count}")
        self._log(f"Total processed: {result.total_processed}")
        self._log(f"Output: {result.output_folder}")
        self._log(f"Threshold: {result.threshold}")
        if result.errors:
            self._log(f"Errors: {result.errors}")

    def _on_error(self, msg):
        self._set_running(False)
        self.status_label.setText("Error")
        self._log(f"ERROR: {msg}")

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
        folder_layout, self.input_line = _folder_row("Input folder:", self)
        layout.addLayout(folder_layout)

        # Detection parameters
        det_group = QGroupBox("Detection parameters")
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
        match_group = QGroupBox("Matching parameters")
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
        self.run_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
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
            QMessageBox.warning(self, "Warning", "Select input folder.")
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
            QMessageBox.critical(self, "Error", msg)
            return

        self._set_running(True)
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self._log(f"Starting PTV Analysis...")
        self._log(f"  Input: {folder}")
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
        self._log("Cancellation requested...")

    def _log(self, text: str):
        self.log_text.append(text)

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(int(pct))
        self.status_label.setText(msg)
        self._log(f"[{pct:.1f}%] {msg}")

    def _on_finished(self, result):
        self._set_running(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Done" if result.success else "Failed")
        self._log("--- Result ---")
        self._log(f"Success: {result.success}")
        self._log(f"Total images processed: {result.total_images_processed}")
        self._log(f"Total particles detected: {result.total_particles_detected}")
        self._log(f"Total pairs matched: {result.total_pairs_matched}")
        self._log(f"Cam1 pairs: {result.cam1_pairs_count}")
        self._log(f"Cam2 pairs: {result.cam2_pairs_count}")
        self._log(f"Output: {result.output_folder}")
        if result.errors:
            self._log(f"Errors: {result.errors}")
        if result.warnings:
            self._log(f"Warnings: {result.warnings}")

    def _on_error(self, msg):
        self._set_running(False)
        self.status_label.setText("Error")
        self._log(f"ERROR: {msg}")

    def _set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)


# ---------------------------------------------------------------------------
# Tab 3: PTV Processing (фильтрация, усреднение, визуализация)
# ---------------------------------------------------------------------------
class PTVProcessingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._executor = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- 3a. Vector Filter ---
        filt_group = QGroupBox("1. Vector Filter")
        filt_layout = QVBoxLayout(filt_group)

        f_row, self.filt_file_line = _file_row("Input CSV:", self)
        filt_layout.addLayout(f_row)

        # U range
        h_u = QHBoxLayout()
        self.filter_u_cb = QCheckBox("Filter U")
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
        self.filter_v_cb = QCheckBox("Filter V")
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

        self.filt_run_btn = QPushButton("Run Filter")
        filt_layout.addWidget(self.filt_run_btn)
        layout.addWidget(filt_group)

        # --- 3b. Vector Average ---
        avg_group = QGroupBox("2. Vector Average")
        avg_layout = QVBoxLayout(avg_group)

        a_row, self.avg_file_line = _file_row("Input CSV:", self)
        avg_layout.addLayout(a_row)

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

        self.avg_run_btn = QPushButton("Run Average")
        avg_layout.addWidget(self.avg_run_btn)
        layout.addWidget(avg_group)

        # --- 3c. Vector Plot ---
        plot_group = QGroupBox("3. Vector Plot")
        plot_layout = QVBoxLayout(plot_group)

        p_row, self.plot_file_line = _file_row("Input CSV:", self)
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

        self.plot_run_btn = QPushButton("Run Plot")
        plot_layout.addWidget(self.plot_run_btn)
        layout.addWidget(plot_group)

        # Shared progress + log
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Connections
        self.filt_run_btn.clicked.connect(self._run_filter)
        self.avg_run_btn.clicked.connect(self._run_average)
        self.plot_run_btn.clicked.connect(self._run_plot)

    # ---- Run Filter ----
    def _run_filter(self):
        csv_path = self.filt_file_line.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "Warning", "Select input CSV file.")
            return

        params = VectorFilterParameters(
            input_file=csv_path,
            filter_u=self.filter_u_cb.isChecked(),
            u_min=self.u_min_spin.value(),
            u_max=self.u_max_spin.value(),
            filter_v=self.filter_v_cb.isChecked(),
            v_min=self.v_min_spin.value(),
            v_max=self.v_max_spin.value(),
        )

        self._executor = VectorFilterExecutor()
        ok, msg = self._executor.set_parameters(params)
        if not ok:
            QMessageBox.critical(self, "Error", msg)
            return

        self.log_text.clear()
        self._log("Starting Vector Filter...")
        self._log(f"  Input: {csv_path}")
        if params.filter_u:
            self._log(f"  U: [{params.u_min}, {params.u_max}]")
        if params.filter_v:
            self._log(f"  V: [{params.v_min}, {params.v_max}]")
        self._log("")
        self._start_worker()

    # ---- Run Average ----
    def _run_average(self):
        csv_path = self.avg_file_line.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "Warning", "Select input CSV file.")
            return

        params = VectorAverageParameters(
            input_file=csv_path,
            plane_width=self.plane_w_spin.value(),
            plane_height=self.plane_h_spin.value(),
            cell_width=self.cell_w_spin.value(),
            cell_height=self.cell_h_spin.value(),
            min_points_in_cell=self.min_pts_spin.value(),
        )

        self._executor = VectorAverageExecutor()
        ok, msg = self._executor.set_parameters(params)
        if not ok:
            QMessageBox.critical(self, "Error", msg)
            return

        grid_info = params.get_grid_info()
        self.log_text.clear()
        self._log("Starting Vector Average...")
        self._log(f"  Input: {csv_path}")
        self._log(f"  Plane: {params.plane_width} x {params.plane_height}")
        self._log(f"  Cell: {params.cell_width} x {params.cell_height}")
        self._log(f"  Grid: {grid_info['nx']} x {grid_info['ny']} = {grid_info['total_cells']} cells")
        self._log(f"  Min points: {params.min_points_in_cell}")
        self._log("")
        self._start_worker()

    # ---- Run Plot ----
    def _run_plot(self):
        csv_path = self.plot_file_line.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "Warning", "Select input CSV file.")
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
            QMessageBox.critical(self, "Error", msg)
            return

        self.log_text.clear()
        self._log("Starting Vector Plot...")
        self._log(f"  Input: {csv_path}")
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

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(int(pct))
        self.status_label.setText(msg)
        self._log(f"[{pct:.1f}%] {msg}")

    def _on_finished(self, result):
        self._set_running(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Done" if result.success else "Failed")
        self._log("--- Result ---")
        self._log(f"Success: {result.success}")

        # VectorFilterResult
        if hasattr(result, 'input_vectors') and hasattr(result, 'vectors_removed'):
            self._log(f"Input vectors: {result.input_vectors}")
            self._log(f"Output vectors: {result.output_vectors}")
            self._log(f"Removed: {result.vectors_removed} ({result.removal_percentage:.1f}%)")
            self._log(f"Output file: {result.output_file}")

        # VectorAverageResult
        elif hasattr(result, 'output_cells'):
            self._log(f"Input vectors: {result.input_vectors}")
            self._log(f"Grid: {result.grid_size[0]} x {result.grid_size[1]}")
            self._log(f"Non-empty cells: {result.output_cells}")
            self._log(f"Empty cells: {result.empty_cells}")
            self._log(f"Points per cell: min={result.min_points_per_cell}, "
                       f"max={result.max_points_per_cell}, avg={result.avg_points_per_cell:.1f}")
            self._log(f"Output file: {result.output_file}")

        # VectorPlotResult
        elif hasattr(result, 'vectors_count'):
            self._log(f"Vectors plotted: {result.vectors_count}")
            self._log(f"dx: [{result.dx_min:.3f}, {result.dx_max:.3f}]")
            self._log(f"dy: [{result.dy_min:.3f}, {result.dy_max:.3f}]")
            self._log(f"L: [{result.l_min:.3f}, {result.l_max:.3f}]")
            self._log(f"Output file: {result.output_file}")

        if result.errors:
            self._log(f"Errors: {result.errors}")

    def _on_error(self, msg):
        self._set_running(False)
        self.status_label.setText("Error")
        self._log(f"ERROR: {msg}")

    def _set_running(self, running: bool):
        self.filt_run_btn.setEnabled(not running)
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
        transform_group = QGroupBox("1. Coordinate Transform")
        transform_layout = QVBoxLayout(transform_group)

        # Input CSV
        f_row, self.input_file_line = _file_row("Input CSV:", self)
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

        # Scale
        h_scale = QHBoxLayout()
        h_scale.addWidget(QLabel("Scale (m/px):"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0000001, 1000.0)
        self.scale_spin.setValue(0.001)
        self.scale_spin.setDecimals(7)
        self.scale_spin.setSingleStep(0.0001)
        h_scale.addWidget(self.scale_spin)
        h_scale.addWidget(QLabel("dt (s):"))
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.0000001, 1000.0)
        self.dt_spin.setValue(0.001)
        self.dt_spin.setDecimals(7)
        self.dt_spin.setSingleStep(0.0001)
        h_scale.addWidget(self.dt_spin)
        h_scale.addStretch()
        transform_layout.addLayout(h_scale)

        self.transform_run_btn = QPushButton("Run Transform")
        transform_layout.addWidget(self.transform_run_btn)
        layout.addWidget(transform_group)

        # --- 2. Vector Plot (physical units) ---
        plot_group = QGroupBox("2. Vector Plot (m, m/s)")
        plot_layout = QVBoxLayout(plot_group)

        p_row, self.plot_file_line = _file_row("Input CSV:", self)
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

        self.plot_run_btn = QPushButton("Run Plot")
        plot_layout.addWidget(self.plot_run_btn)
        layout.addWidget(plot_group)

        # Shared progress + log
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
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
            QMessageBox.warning(self, "Warning", "Select input CSV file.")
            return

        params = CoordinateTransformParameters(
            input_file=csv_path,
            x_origin=self.x_origin_spin.value(),
            y_origin=self.y_origin_spin.value(),
            scale=self.scale_spin.value(),
            dt=self.dt_spin.value(),
        )

        self._executor = CoordinateTransformExecutor()
        ok, msg = self._executor.set_parameters(params)
        if not ok:
            QMessageBox.critical(self, "Error", msg)
            return

        self.log_text.clear()
        self._log("Starting Coordinate Transform...")
        self._log(f"  Input: {csv_path}")
        self._log(f"  X_origin: {params.x_origin}, Y_origin: {params.y_origin}")
        self._log(f"  Scale: {params.scale} m/px")
        self._log(f"  dt: {params.dt} s")
        self._log("")
        self._start_worker()

    # ---- Run Plot ----
    def _run_plot(self):
        csv_path = self.plot_file_line.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "Warning", "Select input CSV file.")
            return

        # Map color_by GUI value to transformed column name
        color_by_gui = self.color_by_combo.currentText()
        color_by_map = {"L": "L", "dx": "dx", "dy": "dy", "angle": "angle"}
        l_column_map = {"L": "L_ms", "dx": "dx_ms", "dy": "dy_ms", "angle": "L_ms"}
        colorbar_labels = {
            "L": "L (m/s)", "dx": "dx (m/s)", "dy": "dy (m/s)", "angle": "Angle (degrees)"
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
            title="Vector Field (physical units)",
            xlabel="X (mm)",
            ylabel="Y (mm)",
            invert_y=False,
            suffix="_plot_phys",
        )

        self._executor = VectorPlotExecutor()
        ok, msg = self._executor.set_parameters(params)
        if not ok:
            QMessageBox.critical(self, "Error", msg)
            return

        self.log_text.clear()
        self._log("Starting Vector Plot (physical units)...")
        self._log(f"  Input: {csv_path}")
        self._log(f"  arrow_scale: {params.arrow_scale}, arrow_width: {params.arrow_width}")
        self._log(f"  colormap: {params.colormap}, color_by: {params.color_by}")
        self._log(f"  axes: X (m), Y (m)")
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
        self.status_label.setText("Done" if result.success else "Failed")
        self._log("--- Result ---")
        self._log(f"Success: {result.success}")

        # CoordinateTransformResult
        if hasattr(result, 'input_rows') and hasattr(result, 'output_rows'):
            self._log(f"Input rows: {result.input_rows}")
            self._log(f"Output rows: {result.output_rows}")
            self._log(f"Output file: {result.output_file}")
            if result.success and result.output_file:
                self._last_output_file = result.output_file
                self.plot_file_line.setText(result.output_file)

        # VectorPlotResult
        elif hasattr(result, 'vectors_count'):
            self._log(f"Vectors plotted: {result.vectors_count}")
            self._log(f"dx: [{result.dx_min:.6f}, {result.dx_max:.6f}] m/s")
            self._log(f"dy: [{result.dy_min:.6f}, {result.dy_max:.6f}] m/s")
            self._log(f"L: [{result.l_min:.6f}, {result.l_max:.6f}] m/s")
            self._log(f"Output file: {result.output_file}")

        if result.errors:
            self._log(f"Errors: {result.errors}")

    def _on_error(self, msg):
        self._set_running(False)
        self.status_label.setText("Error")
        self._log(f"ERROR: {msg}")

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
        self.resize(700, 600)

        tabs = QTabWidget()
        tabs.addTab(SortBinarizeTab(), "Sort + Binarize")
        tabs.addTab(PTVAnalysisTab(), "PTV Analysis")
        tabs.addTab(PTVProcessingTab(), "PTV Processing")
        tabs.addTab(CoordinateTransformTab(), "Coordinate Transform")
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
