"""Automated end-to-end processing for experiment folders."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from execute.execute_analysis.execute_ptv_analysis import PTVExecutor, PTVParameters
from execute.execute_filter.execute_sort_binarize import (
    SortBinarizeExecutor,
    SortBinarizeParameters,
)
from execute.execute_processing.coordinate_transform import (
    CoordinateTransformExecutor,
    CoordinateTransformParameters,
)
from execute.execute_processing.vector_average import (
    VectorAverageExecutor,
    VectorAverageParameters,
)
from execute.execute_processing.vector_filter import (
    VectorFilterExecutor,
    VectorFilterParameters,
)
from execute.execute_processing.vector_plot import VectorPlotExecutor, VectorPlotParameters
from src.data_processing.experiment_preprocess import (
    ExperimentRecord,
    build_output_base_folder,
)


@dataclass
class AutomatedPipelineConfig:
    input_root: str
    output_root: str

    threshold: int = 2000
    validate_format: bool = True
    median_filter_enabled: bool = False
    median_kernel_size: int = 5

    detection_min_area: int = 4
    detection_max_area: int = 150
    matching_max_distance: float = 50.0
    matching_max_diameter_diff: float = 4.0

    run_filter: bool = True
    filter_u: bool = True
    u_min: float = 0.0
    u_max: float = 40.0
    filter_v: bool = True
    v_min: float = -10.0
    v_max: float = 10.0

    run_average: bool = True
    plane_width: float = 4904.0
    plane_height: float = 3280.0
    cell_width: float = 66.0
    cell_height: float = 66.0
    min_points_in_cell: int = 1

    run_transform: bool = False
    cam1_x_origin: float = 0.0
    cam1_y_origin: float = 0.0
    cam1_rotation_angle: float = 0.0
    cam2_x_origin: float = 0.0
    cam2_y_origin: float = 0.0
    cam2_rotation_angle: float = 0.0
    scale_m_per_px: float = 0.001
    dt_seconds: float = 0.001

    histogram_cam1: bool = True
    histogram_cam2: bool = True
    histogram_combined: bool = True
    histogram_bin_width: float = 1.0
    vector_plot_raw: bool = False
    vector_plot_filtered: bool = False
    vector_plot_averaged: bool = True
    vector_plot_transformed: bool = False
    export_png: bool = True
    export_csv: bool = True
    plot_arrow_scale: float = 1.0
    plot_dpi: int = 150
    plot_colormap: str = "jet"

    def validate(self) -> tuple[bool, str]:
        if not Path(self.input_root).is_dir():
            return False, f"Корневая папка не существует: {self.input_root}"
        if not self.output_root.strip():
            return False, "Не указана папка вывода"
        if self.detection_min_area > self.detection_max_area:
            return False, "min_area не может быть больше max_area"
        if self.filter_u and self.u_min >= self.u_max:
            return False, "Минимум U должен быть меньше максимума U"
        if self.filter_v and self.v_min >= self.v_max:
            return False, "Минимум V должен быть меньше максимума V"
        if self.run_average:
            if min(self.plane_width, self.plane_height, self.cell_width, self.cell_height) <= 0:
                return False, "Размеры плоскости и ячейки должны быть больше нуля"
            if self.cell_width > self.plane_width or self.cell_height > self.plane_height:
                return False, "Размер ячейки не может быть больше размера плоскости"
        if self.run_transform and (self.scale_m_per_px <= 0 or self.dt_seconds <= 0):
            return False, "Масштаб и dt должны быть больше нуля"
        if self.histogram_bin_width <= 0:
            return False, "Ширина бара гистограммы должна быть больше нуля"
        if self.plot_arrow_scale <= 0 or self.plot_dpi <= 0:
            return False, "Масштаб стрелок и DPI должны быть больше нуля"
        if self.has_requested_graphs() and not (self.export_png or self.export_csv):
            return False, "Для выбранных графиков нужен хотя бы один формат: PNG или CSV"
        if self.vector_plot_filtered and not self.run_filter:
            return False, "График после фильтрации требует включенного этапа фильтрации"
        if self.vector_plot_averaged and not self.run_average:
            return False, "График после усреднения требует включенного этапа усреднения"
        if self.vector_plot_transformed and not self.run_transform:
            return False, "График после преобразования требует включенного преобразования координат"
        return True, ""

    def has_requested_graphs(self) -> bool:
        return any((
            self.histogram_cam1,
            self.histogram_cam2,
            self.histogram_combined,
            self.vector_plot_raw,
            self.vector_plot_filtered,
            self.vector_plot_averaged,
            self.vector_plot_transformed,
        ))


@dataclass
class PipelineProgress:
    percentage: float
    message: str
    experiment_id: str = ""
    experiment_name: str = ""
    stage: str = ""
    experiment_index: int = 0
    experiment_count: int = 0


@dataclass
class PipelineExperimentResult:
    experiment_id: str
    name: str
    success: bool = False
    status: str = "Ожидание"
    output_folder: str = ""
    ptv_folder: str = ""
    errors: List[str] = field(default_factory=list)
    files: Dict[str, str] = field(default_factory=dict)


@dataclass
class AutomatedPipelineResult:
    success: bool
    cancelled: bool
    output_root: str
    experiment_results: List[PipelineExperimentResult]
    errors: List[str] = field(default_factory=list)

    @property
    def completed_count(self) -> int:
        return sum(item.success for item in self.experiment_results)


class AutomatedPipelineExecutor:
    """Run all selected experiments through the configured processing stages."""

    def __init__(
        self,
        config: AutomatedPipelineConfig,
        experiments: Sequence[ExperimentRecord],
    ):
        self.config = config
        self.experiments = list(experiments)
        self._progress_callback: Optional[Callable[[PipelineProgress], None]] = None
        self._cancel_requested = False
        self._active_executor = None

    def set_progress_callback(self, callback: Callable[[PipelineProgress], None]) -> None:
        self._progress_callback = callback

    def cancel(self) -> None:
        self._cancel_requested = True
        if self._active_executor is not None and hasattr(self._active_executor, "cancel"):
            self._active_executor.cancel()

    def execute(self) -> AutomatedPipelineResult:
        valid, error = self.config.validate()
        if not valid:
            return AutomatedPipelineResult(False, False, self.config.output_root, [], [error])
        if not self.experiments:
            return AutomatedPipelineResult(
                False, False, self.config.output_root, [], ["Не выбраны эксперименты для обработки"]
            )

        output_root = Path(self.config.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        self._write_config(output_root)

        results: List[PipelineExperimentResult] = []
        total = len(self.experiments)
        for index, record in enumerate(self.experiments):
            if self._cancel_requested:
                break
            result = self._process_experiment(record, index, total)
            results.append(result)

        cancelled = self._cancel_requested
        self._write_report(output_root, results)
        success = not cancelled and bool(results) and all(item.success for item in results)
        self._emit(
            100.0 if not cancelled else self._overall_percent(len(results), total),
            "Пайплайн завершен" if success else ("Пайплайн отменен" if cancelled else "Пайплайн завершен с ошибками"),
        )
        return AutomatedPipelineResult(success, cancelled, str(output_root), results)

    def _process_experiment(
        self,
        record: ExperimentRecord,
        experiment_index: int,
        experiment_count: int,
    ) -> PipelineExperimentResult:
        result = PipelineExperimentResult(record.experiment_id, record.name)
        result.output_folder = build_output_base_folder(self.config.output_root, record)
        stages = self._stage_names()
        stage_count = len(stages)

        def emit_stage(stage_index: int, stage: str, local_percent: float, message: str) -> None:
            completed = experiment_index + (stage_index + local_percent / 100.0) / stage_count
            self._emit(
                completed / experiment_count * 100.0,
                message,
                record,
                stage,
                experiment_index,
                experiment_count,
            )

        try:
            stage_index = 0
            result.status = "Сортировка + бинаризация"
            emit_stage(stage_index, result.status, 0.0, f"{record.name}: запуск сортировки и бинаризации")
            sort_params = SortBinarizeParameters(
                input_folder=record.source_folder,
                threshold=self.config.threshold,
                validate_format=self.config.validate_format,
                output_base_folder=result.output_folder,
                experiment_name=record.name,
                median_filter_enabled=self.config.median_filter_enabled,
                median_kernel_size=self.config.median_kernel_size,
            )
            sort_executor = SortBinarizeExecutor()
            self._active_executor = sort_executor
            ok, message = sort_executor.set_parameters(sort_params)
            if not ok:
                raise RuntimeError(message)
            sort_executor.set_progress_callback(
                lambda progress: emit_stage(stage_index, result.status, progress.percentage, progress.message)
            )
            sort_result = sort_executor.execute()
            if not sort_result.success:
                self._ensure_not_cancelled()
                raise RuntimeError("; ".join(sort_result.errors) or "Ошибка сортировки и бинаризации")
            binary_folder = Path(sort_result.output_folder)
            result.files["binary_folder"] = str(binary_folder)
            if self._cancel_requested:
                raise InterruptedError("Обработка отменена")

            stage_index += 1
            result.status = "PTV анализ"
            emit_stage(stage_index, result.status, 0.0, f"{record.name}: запуск PTV анализа")
            ptv_params = PTVParameters(
                input_folder=str(binary_folder),
                detection_min_area=self.config.detection_min_area,
                detection_max_area=self.config.detection_max_area,
                matching_max_distance=self.config.matching_max_distance,
                matching_max_diameter_diff=self.config.matching_max_diameter_diff,
            )
            ptv_executor = PTVExecutor()
            self._active_executor = ptv_executor
            ok, message = ptv_executor.set_parameters(ptv_params)
            if not ok:
                raise RuntimeError(message)
            ptv_executor.set_progress_callback(
                lambda progress: emit_stage(stage_index, result.status, progress.percentage, progress.message)
            )
            ptv_result = ptv_executor.execute()
            if not ptv_result.success:
                self._ensure_not_cancelled()
                raise RuntimeError("; ".join(ptv_result.errors) or "Ошибка PTV анализа")
            ptv_folder = Path(ptv_result.output_folder)
            result.ptv_folder = str(ptv_folder)
            source_files = {
                camera: ptv_folder / f"{camera}_pairs_sum.csv"
                for camera in ("cam_1", "cam_2")
            }
            for camera, source in source_files.items():
                if not source.exists():
                    raise RuntimeError(f"Не найден результат PTV: {source}")
                result.files[f"{camera}_raw"] = str(source)
            stage_sources: Dict[str, Dict[str, Path]] = {"raw": source_files}
            if self._cancel_requested:
                raise InterruptedError("Обработка отменена")

            if self.config.run_filter:
                stage_index += 1
                result.status = "Фильтрация векторов"
                filtered = {}
                for camera_index, (camera, source) in enumerate(source_files.items()):
                    emit_stage(stage_index, result.status, camera_index * 50.0, f"{record.name}: фильтрация {camera}")
                    params = VectorFilterParameters(
                        input_file=str(source),
                        filter_u=self.config.filter_u,
                        u_min=self.config.u_min,
                        u_max=self.config.u_max,
                        filter_v=self.config.filter_v,
                        v_min=self.config.v_min,
                        v_max=self.config.v_max,
                    )
                    executor = VectorFilterExecutor()
                    self._active_executor = executor
                    ok, message = executor.set_parameters(params)
                    if not ok:
                        raise RuntimeError(f"{camera}: {message}")
                    stage_result = executor.execute()
                    if not stage_result.success:
                        raise RuntimeError(f"{camera}: {'; '.join(stage_result.errors)}")
                    filtered[camera] = Path(stage_result.output_file)
                    result.files[f"{camera}_filtered"] = stage_result.output_file
                    self._ensure_not_cancelled()
                stage_sources["filtered"] = filtered
                source_files = filtered

            if self.config.run_average:
                stage_index += 1
                result.status = "Усреднение векторов"
                averaged = {}
                for camera_index, (camera, source) in enumerate(source_files.items()):
                    emit_stage(stage_index, result.status, camera_index * 50.0, f"{record.name}: усреднение {camera}")
                    params = VectorAverageParameters(
                        input_file=str(source),
                        plane_width=self.config.plane_width,
                        plane_height=self.config.plane_height,
                        cell_width=self.config.cell_width,
                        cell_height=self.config.cell_height,
                        min_points_in_cell=self.config.min_points_in_cell,
                    )
                    executor = VectorAverageExecutor()
                    self._active_executor = executor
                    ok, message = executor.set_parameters(params)
                    if not ok:
                        raise RuntimeError(f"{camera}: {message}")
                    stage_result = executor.execute()
                    if not stage_result.success:
                        raise RuntimeError(f"{camera}: {'; '.join(stage_result.errors)}")
                    averaged[camera] = Path(stage_result.output_file)
                    result.files[f"{camera}_averaged"] = stage_result.output_file
                    self._ensure_not_cancelled()
                stage_sources["averaged"] = averaged
                source_files = averaged

            if self.config.run_transform:
                stage_index += 1
                result.status = "Преобразование координат"
                transformed = {}
                for camera_index, (camera, source) in enumerate(source_files.items()):
                    emit_stage(stage_index, result.status, camera_index * 50.0, f"{record.name}: координаты {camera}")
                    prefix = "cam1" if camera == "cam_1" else "cam2"
                    params = CoordinateTransformParameters(
                        input_file=str(source),
                        x_origin=getattr(self.config, f"{prefix}_x_origin"),
                        y_origin=getattr(self.config, f"{prefix}_y_origin"),
                        rotation_angle=getattr(self.config, f"{prefix}_rotation_angle"),
                        scale=self.config.scale_m_per_px,
                        dt=self.config.dt_seconds,
                    )
                    executor = CoordinateTransformExecutor()
                    self._active_executor = executor
                    ok, message = executor.set_parameters(params)
                    if not ok:
                        raise RuntimeError(f"{camera}: {message}")
                    stage_result = executor.execute()
                    if not stage_result.success:
                        raise RuntimeError(f"{camera}: {'; '.join(stage_result.errors)}")
                    transformed[camera] = Path(stage_result.output_file)
                    result.files[f"{camera}_transformed"] = stage_result.output_file
                    self._ensure_not_cancelled()
                stage_sources["transformed"] = transformed

            if self.config.has_requested_graphs():
                stage_index += 1
                result.status = "Построение графиков"
                emit_stage(stage_index, result.status, 0.0, f"{record.name}: построение графиков")
                graph_files = self._create_graphs(ptv_folder, stage_sources)
                result.files.update(graph_files)
                emit_stage(stage_index, result.status, 100.0, f"{record.name}: графики сохранены")

            result.success = True
            result.status = "Готово"
        except InterruptedError as exc:
            result.status = "Отменено"
            result.errors.append(str(exc))
        except Exception as exc:
            result.status = "Ошибка"
            result.errors.append(str(exc))
        finally:
            self._active_executor = None

        return result

    def _stage_names(self) -> List[str]:
        stages = ["Сортировка + бинаризация", "PTV анализ"]
        if self.config.run_filter:
            stages.append("Фильтрация векторов")
        if self.config.run_average:
            stages.append("Усреднение векторов")
        if self.config.run_transform:
            stages.append("Преобразование координат")
        if self.config.has_requested_graphs():
            stages.append("Построение графиков")
        return stages

    def _create_graphs(
        self,
        ptv_folder: Path,
        stage_sources: Dict[str, Dict[str, Path]],
    ) -> Dict[str, str]:
        graph_folder = ptv_folder / "graphs"
        graph_folder.mkdir(parents=True, exist_ok=True)
        outputs: Dict[str, str] = {}

        histogram_sources = []
        if self.config.histogram_cam1:
            histogram_sources.append(("cam_1", [stage_sources["raw"]["cam_1"]]))
        if self.config.histogram_cam2:
            histogram_sources.append(("cam_2", [stage_sources["raw"]["cam_2"]]))
        if self.config.histogram_combined:
            histogram_sources.append(("all_cameras", list(stage_sources["raw"].values())))
        for label, files in histogram_sources:
            self._ensure_not_cancelled()
            saved = self._save_histogram(files, graph_folder, label)
            outputs.update({f"histogram_{label}_{kind}": path for kind, path in saved.items()})

        selected_vector_stages = [
            ("raw", self.config.vector_plot_raw),
            ("filtered", self.config.vector_plot_filtered),
            ("averaged", self.config.vector_plot_averaged),
            ("transformed", self.config.vector_plot_transformed),
        ]
        for stage, enabled in selected_vector_stages:
            if not enabled or stage not in stage_sources:
                continue
            for camera, source in stage_sources[stage].items():
                self._ensure_not_cancelled()
                saved = self._save_vector_graph(source, graph_folder, camera, stage)
                outputs.update({f"vector_{stage}_{camera}_{kind}": path for kind, path in saved.items()})
        return outputs

    def _save_histogram(self, sources: Sequence[Path], output_folder: Path, label: str) -> Dict[str, str]:
        diameters: List[float] = []
        for source in sources:
            with source.open("r", encoding="utf-8") as stream:
                reader = csv.DictReader(stream, delimiter=";")
                for row in reader:
                    try:
                        diameters.append(float(row["Diameter"].replace(",", ".")))
                    except (KeyError, TypeError, ValueError):
                        continue
        if not diameters:
            raise RuntimeError(f"Нет данных Diameter для гистограммы {label}")

        values = np.asarray(diameters, dtype=float)
        width = self.config.histogram_bin_width
        lower = np.floor(values.min() / width) * width
        upper = np.ceil(values.max() / width) * width
        if upper <= lower:
            upper = lower + width
        bins = np.arange(lower, upper + width * 1.01, width)
        counts, edges = np.histogram(values, bins=bins)
        stem = f"particle_diameter_histogram_{label}"
        saved: Dict[str, str] = {}

        if self.config.export_csv:
            csv_path = output_folder / f"{stem}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, delimiter=";")
                writer.writerow(["bin_start_px", "bin_end_px", "count"])
                for left, right, count in zip(edges[:-1], edges[1:], counts):
                    writer.writerow([f"{left:.6g}", f"{right:.6g}", int(count)])
            saved["csv"] = str(csv_path)

        if self.config.export_png:
            figure = Figure(figsize=(10, 7), dpi=self.config.plot_dpi)
            FigureCanvasAgg(figure)
            axis = figure.add_subplot(111)
            axis.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="#2f7ed8", edgecolor="white")
            axis.set_title(f"Распределение диаметра частиц: {label}")
            axis.set_xlabel("Диаметр частицы, px")
            axis.set_ylabel("Количество частиц")
            axis.grid(axis="y", alpha=0.25)
            figure.tight_layout()
            png_path = output_folder / f"{stem}.png"
            figure.savefig(png_path, dpi=self.config.plot_dpi)
            figure.clear()
            saved["png"] = str(png_path)
        return saved

    def _save_vector_graph(
        self,
        source: Path,
        output_folder: Path,
        camera: str,
        stage: str,
    ) -> Dict[str, str]:
        columns = {
            "raw": ("X0", "Y0", "dx", "dy", "L", "X, px", "Y, px"),
            "filtered": ("X0", "Y0", "dx", "dy", "L", "X, px", "Y, px"),
            "averaged": ("X_center", "Y_center", "dx_avg", "dy_avg", "L_avg", "X, px", "Y, px"),
            "transformed": ("X_mm", "Y_mm", "dx_ms", "dy_ms", "L_ms", "X, mm", "Y, mm"),
        }[stage]
        x_col, y_col, dx_col, dy_col, l_col, xlabel, ylabel = columns
        suffix = f"_{stage}_vector"
        saved: Dict[str, str] = {}

        if self.config.export_csv:
            csv_path = output_folder / f"{camera}{suffix}.csv"
            shutil.copyfile(source, csv_path)
            saved["csv"] = str(csv_path)

        if self.config.export_png:
            params = VectorPlotParameters(
                input_file=str(source),
                output_folder=str(output_folder),
                suffix=suffix,
                output_format="png",
                dpi=self.config.plot_dpi,
                arrow_scale=self.config.plot_arrow_scale,
                colormap=self.config.plot_colormap,
                color_by="L",
                title=f"{camera}: {stage}",
                xlabel=xlabel,
                ylabel=ylabel,
                invert_y=stage != "transformed",
                x_column=x_col,
                y_column=y_col,
                dx_column=dx_col,
                dy_column=dy_col,
                l_column=l_col,
            )
            executor = VectorPlotExecutor()
            ok, message = executor.set_parameters(params)
            if not ok:
                raise RuntimeError(f"График {camera}/{stage}: {message}")
            plot_result = executor.execute()
            if not plot_result.success:
                raise RuntimeError(f"График {camera}/{stage}: {'; '.join(plot_result.errors)}")
            saved["png"] = plot_result.output_file
        return saved

    def _write_config(self, output_root: Path) -> None:
        config_path = output_root / "pipeline_config.json"
        payload = asdict(self.config)
        payload["experiments"] = [
            {"id": item.experiment_id, "name": item.name, "source_folder": item.source_folder}
            for item in self.experiments
        ]
        with config_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)

    @staticmethod
    def _write_report(output_root: Path, results: Sequence[PipelineExperimentResult]) -> None:
        report_path = output_root / "pipeline_report.csv"
        with report_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(["experiment_id", "name", "status", "success", "output_folder", "ptv_folder", "errors"])
            for item in results:
                writer.writerow([
                    item.experiment_id,
                    item.name,
                    item.status,
                    item.success,
                    item.output_folder,
                    item.ptv_folder,
                    " | ".join(item.errors),
                ])

    def _emit(
        self,
        percentage: float,
        message: str,
        record: Optional[ExperimentRecord] = None,
        stage: str = "",
        experiment_index: int = 0,
        experiment_count: int = 0,
    ) -> None:
        if self._progress_callback is None:
            return
        self._progress_callback(PipelineProgress(
            percentage=max(0.0, min(100.0, percentage)),
            message=message,
            experiment_id=record.experiment_id if record else "",
            experiment_name=record.name if record else "",
            stage=stage,
            experiment_index=experiment_index,
            experiment_count=experiment_count,
        ))

    def _ensure_not_cancelled(self) -> None:
        if self._cancel_requested:
            raise InterruptedError("Обработка отменена")

    @staticmethod
    def _overall_percent(processed: int, total: int) -> float:
        return processed / total * 100.0 if total else 0.0
