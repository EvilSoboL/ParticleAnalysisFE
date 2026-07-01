"""GUI tab for aggregate analysis over automated pipeline results."""

from __future__ import annotations

import traceback
from pathlib import Path

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from execute.summary_analysis import (
    CAMERAS,
    DISTANCES_MM,
    GROUP_ORDER,
    SummaryAnalysisConfig,
    SummaryAnalysisExecutor,
    TransformSettings,
)


class SummaryAnalysisWorker(QThread):
    progress = pyqtSignal(object)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, executor: SummaryAnalysisExecutor, parent=None):
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
    label.setFixedWidth(185)
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


def _double_spin(minimum: float, maximum: float, value: float, decimals: int = 3) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    return spin


class SummaryAnalysisTab(QWidget):
    """Aggregate particle-size distributions and vector plots across experiments."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._executor = None
        self._transform_spins = {}
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        input_row, self.pipeline_root_line = _folder_row("Папка результатов автопайплайна:", self)
        output_row, self.output_root_line = _folder_row("Папка суммарного анализа:", self)
        self.output_root_line.setPlaceholderText("По умолчанию <папка результатов>/summary_analysis")
        layout.addLayout(input_row)
        layout.addLayout(output_row)

        command_row = QHBoxLayout()
        self.scan_btn = QPushButton("Сканировать")
        self.run_btn = QPushButton("Выполнить суммарный анализ")
        self.run_btn.setEnabled(False)
        command_row.addWidget(self.scan_btn)
        command_row.addWidget(self.run_btn)
        command_row.addStretch()
        layout.addLayout(command_row)

        settings_row = QHBoxLayout()
        settings_row.addWidget(self._create_distribution_group())
        settings_row.addWidget(self._create_transform_group(), 1)
        layout.addLayout(settings_row)

        self.summary_label = QLabel("Выберите папку результатов автопайплайна и выполните сканирование")
        layout.addWidget(self.summary_label)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Группа", "Эксперименты", "Данные", "Статус", "CSV", "PNG"
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.stage_label = QLabel("Готово")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.stage_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(160)
        layout.addWidget(self.log_text)

        self.scan_btn.clicked.connect(self._scan)
        self.run_btn.clicked.connect(self._run)
        self.pipeline_root_line.textEdited.connect(self._input_changed)

    def _create_distribution_group(self) -> QGroupBox:
        group = QGroupBox("Дисперсный состав")
        grid = QGridLayout(group)

        self.bin_start_spin = _double_spin(0.0, 1000000.0, 1.0, 3)
        self.bin_width_spin = _double_spin(0.001, 1000000.0, 1.0, 3)
        self.scale_spin = _double_spin(0.000000001, 1000.0, 0.001, 9)
        self.dt_spin = _double_spin(0.000000001, 1000.0, 0.001, 9)
        self.arrow_scale_spin = _double_spin(0.001, 1000000.0, 1.0, 3)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(50, 1200)
        self.dpi_spin.setValue(150)
        self.max_vectors_spin = QSpinBox()
        self.max_vectors_spin.setRange(1, 1000000)
        self.max_vectors_spin.setValue(5000)
        self.export_png_cb = QCheckBox("PNG")
        self.export_png_cb.setChecked(True)
        self.export_csv_cb = QCheckBox("CSV")
        self.export_csv_cb.setChecked(True)

        rows = [
            ("Начало d, px:", self.bin_start_spin),
            ("Ширина d, px:", self.bin_width_spin),
            ("scale, m/px:", self.scale_spin),
            ("dt, s:", self.dt_spin),
            ("arrow_scale:", self.arrow_scale_spin),
            ("DPI:", self.dpi_spin),
            ("Лимит векторов PNG:", self.max_vectors_spin),
        ]
        for row, (label, widget) in enumerate(rows):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)
        grid.addWidget(QLabel("Форматы:"), len(rows), 0)
        formats = QHBoxLayout()
        formats.addWidget(self.export_png_cb)
        formats.addWidget(self.export_csv_cb)
        formats.addStretch()
        grid.addLayout(formats, len(rows), 1)

        group.setToolTip(
            "d берется из колонки Diameter в cam_X_pairs_sum.csv. "
            "При d=1 и ширине 1 будут интервалы 1-2 px, 2-3 px и так далее. "
            "Векторные графики строятся отдельно по каждому режиму из cam_X_pairs_sum*averaged*.csv."
        )
        return group

    def _create_transform_group(self) -> QGroupBox:
        group = QGroupBox("Поворот и смещение для векторных графиков")
        grid = QGridLayout(group)
        headers = ["Группа", "X origin, px", "Y origin, px", "Угол, deg"]
        for column, text in enumerate(headers):
            grid.addWidget(QLabel(text), 0, column)

        row = 1
        for camera, distance_mm in GROUP_ORDER:
            grid.addWidget(QLabel(f"{camera}, {distance_mm} mm"), row, 0)
            x_spin = _double_spin(-1000000.0, 1000000.0, 0.0, 3)
            y_spin = _double_spin(-1000000.0, 1000000.0, 0.0, 3)
            angle_spin = _double_spin(-360.0, 360.0, 0.0, 3)
            grid.addWidget(x_spin, row, 1)
            grid.addWidget(y_spin, row, 2)
            grid.addWidget(angle_spin, row, 3)
            self._transform_spins[(camera, distance_mm)] = (x_spin, y_spin, angle_spin)
            row += 1

        group.setToolTip(
            "Для каждой камеры и расстояния задается свой origin и угол. "
            "CSV по режимам содержат только X_mm, Y_mm, dx_ms и dy_ms. "
            "Параметры origin, угла, scale и arrow_scale сохраняются в analysis_parameters.csv."
        )
        return group

    def _input_changed(self, text: str) -> None:
        if text.strip() and not self.output_root_line.text().strip():
            self.output_root_line.setPlaceholderText(str(Path(text.strip()) / "summary_analysis"))
        self.run_btn.setEnabled(False)

    def _scan(self) -> None:
        config = self._build_config(scan_only=True)
        valid, message = config.validate()
        if not valid:
            QMessageBox.warning(self, "Внимание", message)
            return

        executor = SummaryAnalysisExecutor(config)
        sources, warnings = executor.scan_sources()
        counts = {(camera, distance): 0 for camera in CAMERAS for distance in DISTANCES_MM}
        for source in sources:
            for camera in source.files:
                counts[(camera, source.distance_mm)] += 1

        self.table.setRowCount(len(GROUP_ORDER))
        for row, (camera, distance_mm) in enumerate(GROUP_ORDER):
            self.table.setItem(row, 0, QTableWidgetItem(f"{camera}, {distance_mm} mm"))
            self.table.setItem(row, 1, QTableWidgetItem(str(counts[(camera, distance_mm)])))
            self.table.setItem(row, 2, QTableWidgetItem("-"))
            self.table.setItem(row, 3, QTableWidgetItem("Готово к обработке" if counts[(camera, distance_mm)] else "Нет данных"))
            self.table.setItem(row, 4, QTableWidgetItem(""))
            self.table.setItem(row, 5, QTableWidgetItem(""))

        ready_groups = sum(1 for value in counts.values() if value)
        self.summary_label.setText(f"Найдено экспериментов: {len(sources)}, групп с данными: {ready_groups} из 4")
        self.log_text.clear()
        self._log(f"Сканирование: {config.pipeline_root}")
        for warning in warnings:
            self._log(f"ПРЕДУПРЕЖДЕНИЕ: {warning}")
        self.run_btn.setEnabled(bool(sources))

    def _build_config(self, scan_only: bool = False) -> SummaryAnalysisConfig:
        pipeline_root = self.pipeline_root_line.text().strip()
        output_root = self.output_root_line.text().strip()
        if not output_root and pipeline_root:
            output_root = str(Path(pipeline_root) / "summary_analysis")

        transforms = {}
        for (camera, distance_mm), (x_spin, y_spin, angle_spin) in self._transform_spins.items():
            transforms[f"{camera}_{distance_mm}mm"] = TransformSettings(
                x_origin=x_spin.value(),
                y_origin=y_spin.value(),
                rotation_angle=angle_spin.value(),
            )

        return SummaryAnalysisConfig(
            pipeline_root=pipeline_root,
            output_root=output_root,
            bin_start=self.bin_start_spin.value(),
            bin_width=self.bin_width_spin.value(),
            scale_m_per_px=self.scale_spin.value(),
            dt_seconds=self.dt_spin.value(),
            export_png=self.export_png_cb.isChecked() or scan_only,
            export_csv=self.export_csv_cb.isChecked() or scan_only,
            plot_dpi=self.dpi_spin.value(),
            arrow_scale=self.arrow_scale_spin.value(),
            max_vectors_for_plot=self.max_vectors_spin.value(),
            transforms=transforms,
        )

    def _run(self) -> None:
        config = self._build_config()
        valid, message = config.validate()
        if not valid:
            QMessageBox.critical(self, "Ошибка параметров", message)
            return

        self.output_root_line.setText(config.output_root)
        self.progress_bar.setValue(0)
        self.stage_label.setText("Запуск суммарного анализа")
        self.log_text.clear()
        self._log(f"Вход: {config.pipeline_root}")
        self._log(f"Вывод: {config.output_root}")
        self._set_running(True)

        self._executor = SummaryAnalysisExecutor(config)
        self._worker = SummaryAnalysisWorker(self._executor, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, progress) -> None:
        self.progress_bar.setValue(int(progress.percentage * 10))
        self.stage_label.setText(progress.message)

    def _on_finished(self, result) -> None:
        self._set_running(False)
        if result.success:
            self.progress_bar.setValue(1000)
            self.stage_label.setText("Суммарный анализ завершен")
        else:
            self.stage_label.setText("Суммарный анализ завершен с ошибкой")

        group_rows = {(camera, distance): index for index, (camera, distance) in enumerate(GROUP_ORDER)}
        if self.table.rowCount() != len(GROUP_ORDER):
            self.table.setRowCount(len(GROUP_ORDER))
            for row, (camera, distance) in enumerate(GROUP_ORDER):
                self.table.setItem(row, 0, QTableWidgetItem(f"{camera}, {distance} mm"))
        for group in result.groups:
            row = group_rows.get((group.camera, group.distance_mm))
            if row is None:
                continue
            values = [
                f"{group.camera}, {group.distance_mm} mm",
                str(group.experiment_count),
                f"d: {group.particle_count}; avg: {group.vector_count}; режимы: {group.mode_count}",
                "Готово" if (group.particle_count or group.vector_count) else "Нет данных",
                group.vector_csv or group.diameter_csv,
                group.vector_png or group.diameter_png,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.table.setItem(row, column, item)

        for warning in result.warnings:
            self._log(f"ПРЕДУПРЕЖДЕНИЕ: {warning}")
        for error in result.errors:
            self._log(f"ОШИБКА: {error}")

        if result.success:
            self._log(f"Суммарный CSV: {result.summary_csv}")
            self._log(f"Суммарный PNG: {result.summary_png}")
            self._log(f"Параметры: {result.parameters_csv}")
            self.summary_label.setText(f"Готово: {result.output_root}")
        else:
            self.summary_label.setText("; ".join(result.errors) or "Суммарный анализ завершен с ошибкой")

    def _on_error(self, message: str) -> None:
        self._set_running(False)
        self.stage_label.setText("Ошибка выполнения")
        self._log(f"ОШИБКА: {message}")
        QMessageBox.critical(self, "Ошибка суммарного анализа", message)

    def _set_running(self, running: bool) -> None:
        self.scan_btn.setEnabled(not running)
        self.run_btn.setEnabled(not running)
        self.pipeline_root_line.setEnabled(not running)
        self.output_root_line.setEnabled(not running)

    def _log(self, text: str) -> None:
        self.log_text.append(text)
