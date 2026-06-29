"""GUI tab for automated end-to-end experiment processing."""

from __future__ import annotations

import traceback
from pathlib import Path

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from execute.full_pipeline import AutomatedPipelineConfig, AutomatedPipelineExecutor
from src.data_processing.experiment_preprocess import default_processed_root, scan_experiment_root


class AutomatedPipelineWorker(QThread):
    progress = pyqtSignal(object)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, executor: AutomatedPipelineExecutor, parent=None):
        super().__init__(parent)
        self.executor = executor

    def run(self) -> None:
        try:
            self.executor.set_progress_callback(self.progress.emit)
            self.finished.emit(self.executor.execute())
        except Exception as exc:
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


def _folder_row(label_text: str, parent: QWidget):
    layout = QHBoxLayout()
    label = QLabel(label_text)
    label.setFixedWidth(125)
    line = QLineEdit()
    button = QPushButton("Обзор...")

    def browse() -> None:
        folder = QFileDialog.getExistingDirectory(parent, label_text, line.text().strip())
        if folder:
            line.setText(folder)

    button.clicked.connect(browse)
    layout.addWidget(label)
    layout.addWidget(line)
    layout.addWidget(button)
    return layout, line


def _double_spin(minimum: float, maximum: float, value: float, decimals: int = 2) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setValue(value)
    return spin


class AutomatedPipelineTab(QWidget):
    """Configure and run the complete workflow for all selected experiments."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._records = []
        self._row_by_id = {}
        self._executor = None
        self._worker = None
        self._last_log_key = None
        self._init_ui()

    def _init_ui(self) -> None:
        root_layout = QVBoxLayout(self)

        input_row, self.input_root_line = _folder_row("Папка замеров:", self)
        output_row, self.output_root_line = _folder_row("Папка результата:", self)
        self.output_root_line.setPlaceholderText("По умолчанию <папка замеров>_processed")
        root_layout.addLayout(input_row)
        root_layout.addLayout(output_row)

        command_row = QHBoxLayout()
        self.scan_btn = QPushButton("Сканировать эксперименты")
        self.run_btn = QPushButton("Запустить пайплайн")
        self.cancel_btn = QPushButton("Отмена")
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        command_row.addWidget(self.scan_btn)
        command_row.addWidget(self.run_btn)
        command_row.addWidget(self.cancel_btn)
        command_row.addStretch()
        root_layout.addLayout(command_row)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._create_settings_panel())
        splitter.addWidget(self._create_progress_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 720])
        root_layout.addWidget(splitter, 1)

        self.scan_btn.clicked.connect(self._scan)
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn.clicked.connect(self._cancel)
        self.input_root_line.textEdited.connect(self._input_root_changed)
        self.filter_group.toggled.connect(self._sync_dependencies)
        self.average_group.toggled.connect(self._sync_dependencies)
        self.transform_group.toggled.connect(self._sync_dependencies)
        self._sync_dependencies()

    def _create_settings_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(410)
        content = QWidget()
        layout = QVBoxLayout(content)

        sort_group = QGroupBox("1. Сортировка и бинаризация")
        sort_grid = QGridLayout(sort_group)
        sort_grid.addWidget(QLabel("Порог, 16-bit:"), 0, 0)
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 65535)
        self.threshold_spin.setSingleStep(100)
        self.threshold_spin.setValue(2000)
        sort_grid.addWidget(self.threshold_spin, 0, 1)
        self.validate_cb = QCheckBox("Проверять формат 16-bit PNG")
        self.validate_cb.setChecked(True)
        sort_grid.addWidget(self.validate_cb, 1, 0, 1, 2)
        self.median_cb = QCheckBox("Медианная фильтрация исходного изображения")
        sort_grid.addWidget(self.median_cb, 2, 0, 1, 2)
        sort_grid.addWidget(QLabel("Окно:"), 3, 0)
        self.median_combo = QComboBox()
        self.median_combo.addItem("3x3", 3)
        self.median_combo.addItem("5x5", 5)
        self.median_combo.setCurrentIndex(1)
        sort_grid.addWidget(self.median_combo, 3, 1)
        self.median_cb.toggled.connect(self.median_combo.setEnabled)
        self.median_combo.setEnabled(False)
        layout.addWidget(sort_group)

        ptv_group = QGroupBox("2. PTV анализ")
        ptv_grid = QGridLayout(ptv_group)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 100000)
        self.min_area_spin.setValue(4)
        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(1, 100000)
        self.max_area_spin.setValue(150)
        self.max_distance_spin = _double_spin(0.01, 10000.0, 50.0)
        self.max_diameter_spin = _double_spin(0.0, 10000.0, 4.0)
        ptv_grid.addWidget(QLabel("min_area, px^2:"), 0, 0)
        ptv_grid.addWidget(self.min_area_spin, 0, 1)
        ptv_grid.addWidget(QLabel("max_area, px^2:"), 1, 0)
        ptv_grid.addWidget(self.max_area_spin, 1, 1)
        ptv_grid.addWidget(QLabel("Макс. расстояние, px:"), 2, 0)
        ptv_grid.addWidget(self.max_distance_spin, 2, 1)
        ptv_grid.addWidget(QLabel("Разница диаметра, px:"), 3, 0)
        ptv_grid.addWidget(self.max_diameter_spin, 3, 1)
        layout.addWidget(ptv_group)

        self.filter_group = QGroupBox("3. Фильтрация векторов")
        self.filter_group.setCheckable(True)
        self.filter_group.setChecked(True)
        filter_grid = QGridLayout(self.filter_group)
        self.filter_u_cb = QCheckBox("Фильтр U/dx")
        self.filter_u_cb.setChecked(True)
        self.u_min_spin = _double_spin(-10000.0, 10000.0, 0.0)
        self.u_max_spin = _double_spin(-10000.0, 10000.0, 40.0)
        self.filter_v_cb = QCheckBox("Фильтр V/dy")
        self.filter_v_cb.setChecked(True)
        self.v_min_spin = _double_spin(-10000.0, 10000.0, -10.0)
        self.v_max_spin = _double_spin(-10000.0, 10000.0, 10.0)
        filter_grid.addWidget(self.filter_u_cb, 0, 0)
        filter_grid.addWidget(QLabel("min:"), 0, 1)
        filter_grid.addWidget(self.u_min_spin, 0, 2)
        filter_grid.addWidget(QLabel("max:"), 0, 3)
        filter_grid.addWidget(self.u_max_spin, 0, 4)
        filter_grid.addWidget(self.filter_v_cb, 1, 0)
        filter_grid.addWidget(QLabel("min:"), 1, 1)
        filter_grid.addWidget(self.v_min_spin, 1, 2)
        filter_grid.addWidget(QLabel("max:"), 1, 3)
        filter_grid.addWidget(self.v_max_spin, 1, 4)
        layout.addWidget(self.filter_group)

        self.average_group = QGroupBox("4. Усреднение векторов")
        self.average_group.setCheckable(True)
        self.average_group.setChecked(True)
        average_grid = QGridLayout(self.average_group)
        self.plane_width_spin = _double_spin(1.0, 100000.0, 4904.0, 1)
        self.plane_height_spin = _double_spin(1.0, 100000.0, 3280.0, 1)
        self.cell_width_spin = _double_spin(1.0, 10000.0, 66.0, 1)
        self.cell_height_spin = _double_spin(1.0, 10000.0, 66.0, 1)
        self.min_points_spin = QSpinBox()
        self.min_points_spin.setRange(1, 100000)
        self.min_points_spin.setValue(1)
        average_grid.addWidget(QLabel("Плоскость W x H, px:"), 0, 0)
        average_grid.addWidget(self.plane_width_spin, 0, 1)
        average_grid.addWidget(self.plane_height_spin, 0, 2)
        average_grid.addWidget(QLabel("Ячейка W x H, px:"), 1, 0)
        average_grid.addWidget(self.cell_width_spin, 1, 1)
        average_grid.addWidget(self.cell_height_spin, 1, 2)
        average_grid.addWidget(QLabel("Мин. точек в ячейке:"), 2, 0)
        average_grid.addWidget(self.min_points_spin, 2, 1)
        layout.addWidget(self.average_group)

        self.transform_group = QGroupBox("5. Преобразование координат")
        self.transform_group.setCheckable(True)
        self.transform_group.setChecked(False)
        transform_grid = QGridLayout(self.transform_group)
        self.cam1_x_spin = _double_spin(-100000.0, 100000.0, 0.0, 3)
        self.cam1_y_spin = _double_spin(-100000.0, 100000.0, 0.0, 3)
        self.cam1_angle_spin = _double_spin(-360.0, 360.0, 0.0, 3)
        self.cam2_x_spin = _double_spin(-100000.0, 100000.0, 0.0, 3)
        self.cam2_y_spin = _double_spin(-100000.0, 100000.0, 0.0, 3)
        self.cam2_angle_spin = _double_spin(-360.0, 360.0, 0.0, 3)
        self.scale_spin = _double_spin(0.000000001, 1000.0, 0.001, 9)
        self.dt_spin = _double_spin(0.000000001, 1000.0, 0.001, 9)
        transform_grid.addWidget(QLabel("Камера"), 0, 0)
        transform_grid.addWidget(QLabel("X origin, px"), 0, 1)
        transform_grid.addWidget(QLabel("Y origin, px"), 0, 2)
        transform_grid.addWidget(QLabel("Угол, deg"), 0, 3)
        transform_grid.addWidget(QLabel("cam_1"), 1, 0)
        transform_grid.addWidget(self.cam1_x_spin, 1, 1)
        transform_grid.addWidget(self.cam1_y_spin, 1, 2)
        transform_grid.addWidget(self.cam1_angle_spin, 1, 3)
        transform_grid.addWidget(QLabel("cam_2"), 2, 0)
        transform_grid.addWidget(self.cam2_x_spin, 2, 1)
        transform_grid.addWidget(self.cam2_y_spin, 2, 2)
        transform_grid.addWidget(self.cam2_angle_spin, 2, 3)
        transform_grid.addWidget(QLabel("Масштаб, m/px:"), 3, 0)
        transform_grid.addWidget(self.scale_spin, 3, 1)
        transform_grid.addWidget(QLabel("dt, s:"), 3, 2)
        transform_grid.addWidget(self.dt_spin, 3, 3)
        layout.addWidget(self.transform_group)

        graphs_group = QGroupBox("6. Графики и экспорт")
        graphs_grid = QGridLayout(graphs_group)
        self.hist_cam1_cb = QCheckBox("Гистограмма cam_1")
        self.hist_cam1_cb.setChecked(True)
        self.hist_cam2_cb = QCheckBox("Гистограмма cam_2")
        self.hist_cam2_cb.setChecked(True)
        self.hist_all_cb = QCheckBox("Общая гистограмма")
        self.hist_all_cb.setChecked(True)
        self.hist_width_spin = _double_spin(0.01, 10000.0, 1.0, 2)
        graphs_grid.addWidget(self.hist_cam1_cb, 0, 0, 1, 2)
        graphs_grid.addWidget(self.hist_cam2_cb, 1, 0, 1, 2)
        graphs_grid.addWidget(self.hist_all_cb, 2, 0, 1, 2)
        graphs_grid.addWidget(QLabel("Ширина бара, px:"), 3, 0)
        graphs_grid.addWidget(self.hist_width_spin, 3, 1)

        self.vector_raw_cb = QCheckBox("Векторы после PTV")
        self.vector_filtered_cb = QCheckBox("Векторы после фильтра")
        self.vector_average_cb = QCheckBox("Векторы после усреднения")
        self.vector_average_cb.setChecked(True)
        self.vector_transform_cb = QCheckBox("Векторы после преобразования")
        graphs_grid.addWidget(self.vector_raw_cb, 4, 0, 1, 2)
        graphs_grid.addWidget(self.vector_filtered_cb, 5, 0, 1, 2)
        graphs_grid.addWidget(self.vector_average_cb, 6, 0, 1, 2)
        graphs_grid.addWidget(self.vector_transform_cb, 7, 0, 1, 2)

        self.export_png_cb = QCheckBox("Сохранять PNG")
        self.export_png_cb.setChecked(True)
        self.export_csv_cb = QCheckBox("Сохранять данные графика CSV")
        self.export_csv_cb.setChecked(True)
        graphs_grid.addWidget(self.export_png_cb, 8, 0)
        graphs_grid.addWidget(self.export_csv_cb, 8, 1)
        self.arrow_scale_spin = _double_spin(0.001, 100000.0, 1.0, 3)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(50, 1200)
        self.dpi_spin.setValue(150)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["jet", "viridis", "plasma", "coolwarm", "RdYlBu"])
        graphs_grid.addWidget(QLabel("Масштаб стрелок:"), 9, 0)
        graphs_grid.addWidget(self.arrow_scale_spin, 9, 1)
        graphs_grid.addWidget(QLabel("DPI:"), 10, 0)
        graphs_grid.addWidget(self.dpi_spin, 10, 1)
        graphs_grid.addWidget(QLabel("Цветовая карта:"), 11, 0)
        graphs_grid.addWidget(self.colormap_combo, 11, 1)
        layout.addWidget(graphs_group)

        layout.addStretch()
        scroll.setWidget(content)
        return scroll

    def _create_progress_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.summary_label = QLabel("Выберите папку замеров и выполните сканирование")
        layout.addWidget(self.summary_label)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Обрабатывать", "ID", "Эксперимент", "PNG", "Проверка", "Текущий этап", "Результат"
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.stage_label = QLabel("Готово")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.stage_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        layout.addWidget(self.log_text)
        return panel

    def _input_root_changed(self, text: str) -> None:
        if text.strip() and not self.output_root_line.text().strip():
            self.output_root_line.setPlaceholderText(default_processed_root(text.strip()))
        self.run_btn.setEnabled(False)

    def _scan(self) -> None:
        root = self.input_root_line.text().strip()
        if not root:
            QMessageBox.warning(self, "Внимание", "Выберите папку с замерами.")
            return
        if not self.output_root_line.text().strip():
            self.output_root_line.setText(default_processed_root(root))

        scan_result = scan_experiment_root(root)
        self._records = scan_result.records
        self._populate_table()
        ready = sum(record.sort_ready for record in self._records)
        self.summary_label.setText(f"Найдено: {len(self._records)}, готово к обработке: {ready}")
        self.log_text.clear()
        self._log(f"Сканирование: {root}")
        for error in scan_result.errors:
            self._log(f"ОШИБКА: {error}")
        self.run_btn.setEnabled(ready > 0)

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self._records))
        self._row_by_id.clear()
        for row, record in enumerate(self._records):
            self._row_by_id[record.experiment_id] = row
            include_item = QTableWidgetItem()
            flags = Qt.ItemIsUserCheckable | Qt.ItemIsSelectable
            if record.sort_ready:
                flags |= Qt.ItemIsEnabled
                include_item.setCheckState(Qt.Checked)
            else:
                include_item.setCheckState(Qt.Unchecked)
            include_item.setFlags(flags)
            include_item.setData(Qt.UserRole, row)
            self.table.setItem(row, 0, include_item)

            values = [
                record.experiment_id,
                record.name,
                str(record.png_count),
                "Готов" if record.sort_ready else "Ошибка",
                "Ожидание",
                "",
            ]
            for column, value in enumerate(values, start=1):
                item = QTableWidgetItem(value)
                if column in (2, 6):
                    item.setToolTip(value)
                self.table.setItem(row, column, item)
            if record.errors or record.warnings:
                details = "; ".join(record.errors + record.warnings)
                self.table.item(row, 4).setToolTip(details)

    def _selected_records(self):
        selected = []
        for row, record in enumerate(self._records):
            item = self.table.item(row, 0)
            if record.sort_ready and item is not None and item.checkState() == Qt.Checked:
                selected.append(record)
        return selected

    def _build_config(self) -> AutomatedPipelineConfig:
        return AutomatedPipelineConfig(
            input_root=self.input_root_line.text().strip(),
            output_root=self.output_root_line.text().strip(),
            threshold=self.threshold_spin.value(),
            validate_format=self.validate_cb.isChecked(),
            median_filter_enabled=self.median_cb.isChecked(),
            median_kernel_size=self.median_combo.currentData(),
            detection_min_area=self.min_area_spin.value(),
            detection_max_area=self.max_area_spin.value(),
            matching_max_distance=self.max_distance_spin.value(),
            matching_max_diameter_diff=self.max_diameter_spin.value(),
            run_filter=self.filter_group.isChecked(),
            filter_u=self.filter_u_cb.isChecked(),
            u_min=self.u_min_spin.value(),
            u_max=self.u_max_spin.value(),
            filter_v=self.filter_v_cb.isChecked(),
            v_min=self.v_min_spin.value(),
            v_max=self.v_max_spin.value(),
            run_average=self.average_group.isChecked(),
            plane_width=self.plane_width_spin.value(),
            plane_height=self.plane_height_spin.value(),
            cell_width=self.cell_width_spin.value(),
            cell_height=self.cell_height_spin.value(),
            min_points_in_cell=self.min_points_spin.value(),
            run_transform=self.transform_group.isChecked(),
            cam1_x_origin=self.cam1_x_spin.value(),
            cam1_y_origin=self.cam1_y_spin.value(),
            cam1_rotation_angle=self.cam1_angle_spin.value(),
            cam2_x_origin=self.cam2_x_spin.value(),
            cam2_y_origin=self.cam2_y_spin.value(),
            cam2_rotation_angle=self.cam2_angle_spin.value(),
            scale_m_per_px=self.scale_spin.value(),
            dt_seconds=self.dt_spin.value(),
            histogram_cam1=self.hist_cam1_cb.isChecked(),
            histogram_cam2=self.hist_cam2_cb.isChecked(),
            histogram_combined=self.hist_all_cb.isChecked(),
            histogram_bin_width=self.hist_width_spin.value(),
            vector_plot_raw=self.vector_raw_cb.isChecked(),
            vector_plot_filtered=self.vector_filtered_cb.isChecked(),
            vector_plot_averaged=self.vector_average_cb.isChecked(),
            vector_plot_transformed=self.vector_transform_cb.isChecked(),
            export_png=self.export_png_cb.isChecked(),
            export_csv=self.export_csv_cb.isChecked(),
            plot_arrow_scale=self.arrow_scale_spin.value(),
            plot_dpi=self.dpi_spin.value(),
            plot_colormap=self.colormap_combo.currentText(),
        )

    def _run(self) -> None:
        selected = self._selected_records()
        if not selected:
            QMessageBox.warning(self, "Внимание", "Отметьте хотя бы один готовый эксперимент.")
            return
        config = self._build_config()
        valid, message = config.validate()
        if not valid:
            QMessageBox.critical(self, "Ошибка параметров", message)
            return

        self._executor = AutomatedPipelineExecutor(config, selected)
        self._worker = AutomatedPipelineWorker(self._executor, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._last_log_key = None
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self._log(f"Запуск: {len(selected)} экспериментов")
        self._log(f"Результаты: {config.output_root}")
        self._set_running(True)
        self._worker.start()

    def _cancel(self) -> None:
        if self._executor is not None:
            self._executor.cancel()
            self.stage_label.setText("Остановка после текущей операции...")
            self._log("Запрошена отмена")

    def _on_progress(self, progress) -> None:
        self.progress_bar.setValue(int(progress.percentage * 10))
        self.stage_label.setText(progress.message)
        if progress.experiment_id in self._row_by_id:
            row = self._row_by_id[progress.experiment_id]
            self.table.item(row, 5).setText(progress.stage)
            self.table.scrollToItem(self.table.item(row, 2))
        log_key = (progress.experiment_id, progress.stage)
        if progress.stage and log_key != self._last_log_key:
            self._last_log_key = log_key
            self._log(f"[{progress.percentage:.1f}%] {progress.experiment_name}: {progress.stage}")

    def _on_finished(self, result) -> None:
        self._set_running(False)
        if not result.cancelled:
            self.progress_bar.setValue(1000)
        for item in result.experiment_results:
            row = self._row_by_id.get(item.experiment_id)
            if row is None:
                continue
            self.table.item(row, 5).setText(item.status)
            result_item = self.table.item(row, 6)
            result_item.setText("Готово" if item.success else "; ".join(item.errors))
            result_item.setToolTip(item.ptv_folder or item.output_folder)
        if result.success:
            self.stage_label.setText("Пайплайн завершен")
        elif result.cancelled:
            self.stage_label.setText("Пайплайн отменен")
        else:
            self.stage_label.setText("Пайплайн завершен с ошибками")
        self._log(f"Завершено успешно: {result.completed_count} из {len(result.experiment_results)}")
        self._log(f"Отчет: {Path(result.output_root) / 'pipeline_report.csv'}")

    def _on_error(self, message: str) -> None:
        self._set_running(False)
        self.stage_label.setText("Ошибка выполнения")
        self._log(f"ОШИБКА: {message}")
        QMessageBox.critical(self, "Ошибка пайплайна", message)

    def _sync_dependencies(self) -> None:
        filter_enabled = self.filter_group.isChecked()
        average_enabled = self.average_group.isChecked()
        transform_enabled = self.transform_group.isChecked()
        self.vector_filtered_cb.setEnabled(filter_enabled)
        if not filter_enabled:
            self.vector_filtered_cb.setChecked(False)
        self.vector_average_cb.setEnabled(average_enabled)
        if not average_enabled:
            self.vector_average_cb.setChecked(False)
        self.vector_transform_cb.setEnabled(transform_enabled)
        if not transform_enabled:
            self.vector_transform_cb.setChecked(False)

    def _set_running(self, running: bool) -> None:
        self.scan_btn.setEnabled(not running)
        self.run_btn.setEnabled(not running and bool(self._records))
        self.cancel_btn.setEnabled(running)
        self.input_root_line.setEnabled(not running)
        self.output_root_line.setEnabled(not running)

    def _log(self, text: str) -> None:
        self.log_text.append(text)
