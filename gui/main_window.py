"""
GUI приложение ParticleAnalysis на PyQt5.

Три вкладки:
1. Sort + Binarize — сортировка и бинаризация
2. PTV Analysis — PTV анализ
3. Visualization — визуализация векторного поля и one-to-one
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
from execute.execute_analysis.execute_ptv_vector_field import (
    VectorFieldExecutor, VectorFieldParameters
)
from execute.execute_analysis.execute_ptv_one_to_one import (
    VisualizationExecutor, VisualizationParameters
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
# Tab 3: Visualization
# ---------------------------------------------------------------------------
class VisualizationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._executor = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- 3a. Vector Field ---
        vf_group = QGroupBox("Vector Field")
        vf_layout = QVBoxLayout(vf_group)

        folder_layout, self.vf_ptv_line = _folder_row("PTV folder:", self)
        vf_layout.addLayout(folder_layout)

        # Grid params
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("nx:"))
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(10, 200)
        self.nx_spin.setValue(73)
        h1.addWidget(self.nx_spin)
        h1.addWidget(QLabel("ny:"))
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(10, 200)
        self.ny_spin.setValue(50)
        h1.addWidget(self.ny_spin)
        h1.addWidget(QLabel("scale:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(1.0, 200.0)
        self.scale_spin.setValue(20.0)
        h1.addWidget(self.scale_spin)
        h1.addWidget(QLabel("width:"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.001, 0.02)
        self.width_spin.setValue(0.005)
        self.width_spin.setSingleStep(0.001)
        self.width_spin.setDecimals(3)
        h1.addWidget(self.width_spin)
        h1.addStretch()
        vf_layout.addLayout(h1)

        h_cmap = QHBoxLayout()
        h_cmap.addWidget(QLabel("cmap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            "jet", "viridis", "plasma", "inferno", "magma", "cividis",
            "coolwarm", "RdYlBu", "Spectral"
        ])
        h_cmap.addWidget(self.cmap_combo)
        h_cmap.addStretch()
        vf_layout.addLayout(h_cmap)

        self.vf_run_btn = QPushButton("Run Vector Field")
        vf_layout.addWidget(self.vf_run_btn)
        layout.addWidget(vf_group)

        # --- 3b. One-to-One ---
        oto_group = QGroupBox("One-to-One Visualization")
        oto_layout = QVBoxLayout(oto_group)

        f1, self.oto_orig_line = _folder_row("Original folder:", self)
        oto_layout.addLayout(f1)
        f2, self.oto_ptv_line = _folder_row("PTV folder:", self)
        oto_layout.addLayout(f2)

        h_lt = QHBoxLayout()
        h_lt.addWidget(QLabel("line_thickness:"))
        self.line_thick_spin = QSpinBox()
        self.line_thick_spin.setRange(1, 5)
        self.line_thick_spin.setValue(1)
        h_lt.addWidget(self.line_thick_spin)
        h_lt.addStretch()
        oto_layout.addLayout(h_lt)

        self.oto_run_btn = QPushButton("Run One-to-One")
        oto_layout.addWidget(self.oto_run_btn)
        layout.addWidget(oto_group)

        # Shared progress + log
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Connections
        self.vf_run_btn.clicked.connect(self._run_vector_field)
        self.oto_run_btn.clicked.connect(self._run_one_to_one)

    def _run_vector_field(self):
        folder = self.vf_ptv_line.text().strip()
        if not folder:
            QMessageBox.warning(self, "Warning", "Select PTV folder.")
            return

        params = VectorFieldParameters(
            ptv_folder=folder,
            nx=self.nx_spin.value(),
            ny=self.ny_spin.value(),
            scale=self.scale_spin.value(),
            width=self.width_spin.value(),
            cmap=self.cmap_combo.currentText(),
        )

        self._executor = VectorFieldExecutor()
        ok, msg = self._executor.set_parameters(params)
        if not ok:
            QMessageBox.critical(self, "Error", msg)
            return

        self.log_text.clear()
        self._log(f"Starting Vector Field...")
        self._log(f"  PTV folder: {folder}")
        self._log(f"  nx: {self.nx_spin.value()}, ny: {self.ny_spin.value()}")
        self._log(f"  scale: {self.scale_spin.value()}, width: {self.width_spin.value()}")
        self._log(f"  cmap: {self.cmap_combo.currentText()}")
        self._log("")
        self._start_worker()

    def _run_one_to_one(self):
        orig = self.oto_orig_line.text().strip()
        ptv = self.oto_ptv_line.text().strip()
        if not orig or not ptv:
            QMessageBox.warning(self, "Warning", "Select both folders.")
            return

        params = VisualizationParameters(
            original_folder=orig,
            ptv_folder=ptv,
            line_thickness=self.line_thick_spin.value(),
        )

        self._executor = VisualizationExecutor()
        ok, msg = self._executor.set_parameters(params)
        if not ok:
            QMessageBox.critical(self, "Error", msg)
            return

        self.log_text.clear()
        self._log(f"Starting One-to-One Visualization...")
        self._log(f"  Original: {orig}")
        self._log(f"  PTV folder: {ptv}")
        self._log(f"  line_thickness: {self.line_thick_spin.value()}")
        self._log("")
        self._start_worker()

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
        self._log(f"Output: {result.output_folder}")
        if result.errors:
            self._log(f"Errors: {result.errors}")

    def _on_error(self, msg):
        self._set_running(False)
        self.status_label.setText("Error")
        self._log(f"ERROR: {msg}")

    def _set_running(self, running: bool):
        self.vf_run_btn.setEnabled(not running)
        self.oto_run_btn.setEnabled(not running)


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
        tabs.addTab(VisualizationTab(), "Visualization")
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
