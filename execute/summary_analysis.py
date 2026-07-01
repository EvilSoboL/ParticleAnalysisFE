"""Summary analysis over automated PTV pipeline results."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


CAMERAS = ("cam_1", "cam_2")
DISTANCES_MM = (10, 30)
GROUP_ORDER = tuple((camera, distance) for camera in CAMERAS for distance in DISTANCES_MM)


@dataclass
class TransformSettings:
    x_origin: float = 0.0
    y_origin: float = 0.0
    rotation_angle: float = 0.0


@dataclass
class SummaryAnalysisConfig:
    pipeline_root: str
    output_root: str
    bin_start: float = 1.0
    bin_width: float = 1.0
    scale_m_per_px: float = 0.001
    dt_seconds: float = 0.001
    export_png: bool = True
    export_csv: bool = True
    plot_dpi: int = 150
    arrow_scale: float = 1.0
    max_vectors_for_plot: int = 5000
    transforms: Dict[str, TransformSettings] = field(default_factory=dict)

    def transform_for(self, camera: str, distance_mm: int) -> TransformSettings:
        return self.transforms.get(_group_key(camera, distance_mm), TransformSettings())

    def validate(self) -> tuple[bool, str]:
        if not Path(self.pipeline_root).is_dir():
            return False, f"Папка результатов автопайплайна не существует: {self.pipeline_root}"
        if not self.output_root.strip():
            return False, "Не указана папка вывода суммарного анализа"
        if self.bin_width <= 0:
            return False, "Ширина интервала d должна быть больше нуля"
        if self.scale_m_per_px <= 0:
            return False, "Масштаб m/px должен быть больше нуля"
        if self.dt_seconds <= 0:
            return False, "dt должен быть больше нуля"
        if not (self.export_png or self.export_csv):
            return False, "Нужно выбрать хотя бы один формат вывода: PNG или CSV"
        if self.plot_dpi <= 0:
            return False, "DPI должен быть больше нуля"
        if self.arrow_scale <= 0:
            return False, "Масштаб стрелок должен быть больше нуля"
        if self.max_vectors_for_plot < 1:
            return False, "Лимит векторов на график должен быть больше нуля"
        return True, ""


@dataclass
class SummaryProgress:
    percentage: float
    message: str


@dataclass
class SummarySource:
    experiment_id: str
    experiment_name: str
    distance_mm: int
    ptv_folder: Path
    files: Dict[str, Path]
    averaged_files: Dict[str, Path] = field(default_factory=dict)


@dataclass
class SummaryGroupResult:
    camera: str
    distance_mm: int
    experiment_count: int = 0
    vector_count: int = 0
    particle_count: int = 0
    mode_count: int = 0
    output_folder: str = ""
    diameter_csv: str = ""
    diameter_png: str = ""
    vector_csv: str = ""
    vector_png: str = ""
    vector_folder: str = ""


@dataclass
class SummaryAnalysisResult:
    success: bool
    output_root: str
    groups: List[SummaryGroupResult]
    summary_csv: str = ""
    summary_png: str = ""
    parameters_csv: str = ""
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class SummaryAnalysisExecutor:
    """Create aggregated particle-size and vector reports from PTV pipeline output."""

    def __init__(self, config: SummaryAnalysisConfig):
        self.config = config
        self._progress_callback: Optional[Callable[[SummaryProgress], None]] = None

    def set_progress_callback(self, callback: Callable[[SummaryProgress], None]) -> None:
        self._progress_callback = callback

    def scan_sources(self) -> tuple[List[SummarySource], List[str]]:
        pipeline_root = Path(self.config.pipeline_root)
        report_path = pipeline_root / "pipeline_report.csv"
        warnings: List[str] = []
        sources: List[SummarySource] = []

        if report_path.exists():
            sources.extend(self._scan_report(report_path, warnings))
        else:
            warnings.append("pipeline_report.csv не найден, выполнен поиск PTV_* папок рекурсивно")
            sources.extend(self._scan_ptv_folders(pipeline_root, warnings))

        sources.sort(key=lambda item: (item.distance_mm, item.experiment_name.lower()))
        return sources, warnings

    def execute(self) -> SummaryAnalysisResult:
        valid, message = self.config.validate()
        if not valid:
            return SummaryAnalysisResult(False, self.config.output_root, [], errors=[message])

        output_root = Path(self.config.output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        self._emit(2.0, "Сканирование результатов автопайплайна")
        sources, warnings = self.scan_sources()
        if not sources:
            return SummaryAnalysisResult(
                False,
                str(output_root),
                [],
                warnings=warnings,
                errors=["Не найдены PTV результаты для 10 mm или 30 mm"],
            )

        self._emit(12.0, f"Найдено экспериментов: {len(sources)}")
        grouped_rows, group_sources = self._load_grouped_rows(sources, warnings)
        if not any(grouped_rows.values()):
            return SummaryAnalysisResult(
                False,
                str(output_root),
                [],
                warnings=warnings,
                errors=["Не удалось прочитать данные Diameter из cam_X_pairs_sum.csv"],
            )

        self._emit(35.0, "Сохранение суммарного дисперсного состава")
        edges = self._global_edges(grouped_rows)
        summary_csv = ""
        summary_png = ""
        if self.config.export_csv:
            summary_csv = str(self._save_summary_distribution_csv(output_root, grouped_rows, edges))
        if self.config.export_png:
            summary_png = str(self._save_summary_distribution_png(output_root, grouped_rows, edges))

        self._emit(55.0, "Сохранение групповых CSV и PNG")
        group_results: List[SummaryGroupResult] = []
        for index, (camera, distance_mm) in enumerate(GROUP_ORDER):
            group_key = _group_key(camera, distance_mm)
            group_folder = output_root / f"{camera} {distance_mm} mm"
            group_folder.mkdir(parents=True, exist_ok=True)
            rows = grouped_rows.get(group_key, [])
            sources_for_group = group_sources.get(group_key, [])
            result = SummaryGroupResult(
                camera=camera,
                distance_mm=distance_mm,
                experiment_count=len(sources_for_group),
                particle_count=len(rows),
                output_folder=str(group_folder),
            )
            obsolete_sources_csv = group_folder / "source_files.csv"
            if obsolete_sources_csv.exists():
                try:
                    obsolete_sources_csv.unlink()
                except OSError as exc:
                    warnings.append(f"Не удалось удалить устаревший {obsolete_sources_csv}: {exc}")
            if self.config.export_csv:
                result.diameter_csv = str(self._save_group_distribution_csv(
                    group_folder,
                    rows,
                    edges,
                    sources_for_group,
                    camera,
                ))
            if self.config.export_png:
                result.diameter_png = str(self._save_group_distribution_png(
                    group_folder,
                    rows,
                    camera,
                    distance_mm,
                    edges,
                ))
            mode_result = self._save_mode_vector_outputs(
                group_folder,
                sources_for_group,
                camera,
                distance_mm,
                self.config.transform_for(camera, distance_mm),
                warnings,
            )
            result.mode_count = mode_result["mode_count"]
            result.vector_count = mode_result["vector_count"]
            result.vector_folder = mode_result["folder"]
            result.vector_csv = mode_result["csv"]
            result.vector_png = mode_result["png"]
            group_results.append(result)
            self._emit(55.0 + (index + 1) / len(GROUP_ORDER) * 35.0, f"Готово: {camera}, {distance_mm} mm")

        parameters_csv = str(self._save_parameters_csv(output_root, group_results))
        self._emit(100.0, "Суммарный анализ завершен")
        return SummaryAnalysisResult(
            True,
            str(output_root),
            group_results,
            summary_csv=summary_csv,
            summary_png=summary_png,
            parameters_csv=parameters_csv,
            warnings=warnings,
        )

    def _scan_report(self, report_path: Path, warnings: List[str]) -> List[SummarySource]:
        sources: List[SummarySource] = []
        with report_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=";")
            for row in reader:
                success_value = str(row.get("success", "")).strip().lower()
                if success_value not in {"true", "1", "yes", "да"}:
                    continue
                experiment_name = (row.get("name") or "").strip()
                distance_mm = _distance_from_name(experiment_name)
                if distance_mm is None:
                    warnings.append(f"Пропущен эксперимент без 10/30 mm в имени: {experiment_name}")
                    continue
                ptv_folder = Path(row.get("ptv_folder") or "")
                if not ptv_folder.is_dir():
                    warnings.append(f"PTV папка не найдена: {ptv_folder}")
                    continue
                files = _camera_files(ptv_folder)
                if not files:
                    warnings.append(f"В PTV папке нет cam_X_pairs_sum.csv: {ptv_folder}")
                    continue
                sources.append(SummarySource(
                    experiment_id=(row.get("experiment_id") or "").strip(),
                    experiment_name=experiment_name,
                    distance_mm=distance_mm,
                    ptv_folder=ptv_folder,
                    files=files,
                    averaged_files=_camera_averaged_files(ptv_folder),
                ))
        return sources

    def _scan_ptv_folders(self, pipeline_root: Path, warnings: List[str]) -> List[SummarySource]:
        sources: List[SummarySource] = []
        for ptv_folder in sorted(pipeline_root.rglob("PTV_*")):
            if not ptv_folder.is_dir():
                continue
            files = _camera_files(ptv_folder)
            if not files:
                continue
            experiment_folder = ptv_folder.parent
            experiment_name = experiment_folder.name
            distance_mm = _distance_from_name(experiment_name)
            if distance_mm is None:
                warnings.append(f"Пропущена PTV папка без 10/30 mm в имени эксперимента: {ptv_folder}")
                continue
            sources.append(SummarySource(
                experiment_id=experiment_name,
                experiment_name=experiment_name,
                distance_mm=distance_mm,
                ptv_folder=ptv_folder,
                files=files,
                averaged_files=_camera_averaged_files(ptv_folder),
            ))
        return sources

    def _load_grouped_rows(
        self,
        sources: Sequence[SummarySource],
        warnings: List[str],
    ) -> tuple[Dict[str, List[dict]], Dict[str, List[SummarySource]]]:
        grouped_rows: Dict[str, List[dict]] = {_group_key(camera, distance): [] for camera, distance in GROUP_ORDER}
        group_sources: Dict[str, List[SummarySource]] = {_group_key(camera, distance): [] for camera, distance in GROUP_ORDER}

        for source in sources:
            for camera, csv_path in source.files.items():
                group_key = _group_key(camera, source.distance_mm)
                loaded = list(_read_ptv_rows(csv_path, source, camera, warnings))
                if loaded:
                    grouped_rows[group_key].extend(loaded)
                    if source not in group_sources[group_key]:
                        group_sources[group_key].append(source)
        return grouped_rows, group_sources

    def _global_edges(self, grouped_rows: Dict[str, List[dict]]) -> np.ndarray:
        values = [
            float(row["Diameter"])
            for rows in grouped_rows.values()
            for row in rows
            if row.get("Diameter") is not None
        ]
        width = float(self.config.bin_width)
        start = float(self.config.bin_start)
        if values:
            min_value = min(values)
            max_value = max(values)
            if min_value < start:
                start = math.floor(min_value / width) * width
            stop = math.ceil(max(max_value, start + width) / width) * width
        else:
            stop = start + width
        if stop <= start:
            stop = start + width
        edge_count = max(2, int(math.ceil((stop - start) / width)) + 1)
        return start + np.arange(edge_count, dtype=float) * width

    def _save_summary_distribution_csv(
        self,
        output_root: Path,
        grouped_rows: Dict[str, List[dict]],
        edges: np.ndarray,
    ) -> Path:
        output_path = output_root / "diameter_distribution_summary.csv"
        counts_by_group = {
            _group_key(camera, distance): _counts_for_rows(grouped_rows[_group_key(camera, distance)], edges)
            for camera, distance in GROUP_ORDER
        }

        with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            header = ["d", "bin_start_px", "bin_end_px"]
            header.extend(_group_key(camera, distance) for camera, distance in GROUP_ORDER)
            header.append("total_count")
            writer.writerow(header)
            for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                group_counts = [int(counts_by_group[_group_key(camera, distance)][bin_index]) for camera, distance in GROUP_ORDER]
                writer.writerow([
                    _bin_label(left, right),
                    _format_float(left),
                    _format_float(right),
                    *group_counts,
                    sum(group_counts),
                ])
        return output_path

    def _save_summary_distribution_png(
        self,
        output_root: Path,
        grouped_rows: Dict[str, List[dict]],
        edges: np.ndarray,
    ) -> Path:
        output_path = output_root / "diameter_distribution_summary.png"
        figure = Figure(figsize=(12, 7), dpi=self.config.plot_dpi, facecolor="white")
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        width = np.diff(edges)
        offsets = np.linspace(-0.36, 0.36, len(GROUP_ORDER))
        bar_width = width[0] / max(1, len(GROUP_ORDER)) * 0.85
        colors = ["#2f80ed", "#56ccf2", "#f2994a", "#eb5757"]

        for index, (camera, distance) in enumerate(GROUP_ORDER):
            key = _group_key(camera, distance)
            counts = _counts_for_rows(grouped_rows[key], edges)
            positions = edges[:-1] + width / 2 + offsets[index] * width
            axis.bar(
                positions,
                counts,
                width=bar_width,
                label=f"{camera}, {distance} mm",
                color=colors[index],
                alpha=0.88,
            )

        axis.set_title("Суммарный дисперсный состав по всем экспериментам")
        axis.set_xlabel("Диаметр частицы d, px")
        axis.set_ylabel("Количество частиц")
        axis.set_xticks(edges[:-1] + width / 2)
        axis.set_xticklabels([_bin_label(left, right) for left, right in zip(edges[:-1], edges[1:])], rotation=45, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(loc="upper right")
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.config.plot_dpi)
        figure.clear()
        return output_path

    def _save_group_distribution_csv(
        self,
        group_folder: Path,
        rows: Sequence[dict],
        edges: np.ndarray,
        sources: Sequence[SummarySource],
        camera: str,
    ) -> Path:
        output_path = group_folder / "diameter_distribution.csv"
        used_labels: set[str] = set()
        mode_counts: List[tuple[str, np.ndarray]] = []
        for source in sources:
            source_file = source.files.get(camera)
            if source_file is None:
                continue
            mode_rows = [row for row in rows if row.get("source_file") == str(source_file)]
            label = _unique_label(source.experiment_name, used_labels)
            mode_counts.append((label, _counts_for_rows(mode_rows, edges)))

        with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(["d", "bin_start_px", "bin_end_px", *[label for label, _ in mode_counts]])
            for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                writer.writerow([
                    _bin_label(left, right),
                    _format_float(left),
                    _format_float(right),
                    *[int(counts[bin_index]) for _, counts in mode_counts],
                ])
        return output_path

    def _save_group_distribution_png(
        self,
        group_folder: Path,
        rows: Sequence[dict],
        camera: str,
        distance_mm: int,
        edges: np.ndarray,
    ) -> Path:
        output_path = group_folder / "diameter_distribution.png"
        counts = _counts_for_rows(rows, edges)
        figure = Figure(figsize=(10, 6.5), dpi=self.config.plot_dpi, facecolor="white")
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        axis.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="#2f80ed", edgecolor="white", alpha=0.9)
        axis.set_title(f"{camera}, {distance_mm} mm: суммарный дисперсный состав")
        axis.set_xlabel("Диаметр частицы d, px")
        axis.set_ylabel("Количество частиц")
        axis.set_xticks(edges[:-1] + np.diff(edges) / 2)
        axis.set_xticklabels([_bin_label(left, right) for left, right in zip(edges[:-1], edges[1:])], rotation=45, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.config.plot_dpi)
        figure.clear()
        return output_path

    def _save_group_vectors_csv(
        self,
        group_folder: Path,
        rows: Sequence[dict],
        camera: str,
        distance_mm: int,
        transform: TransformSettings,
    ) -> Path:
        output_path = group_folder / "vectors_transformed.csv"
        fieldnames = [
            "experiment_id", "experiment_name", "camera", "distance_mm", "source_file",
            "x_origin_px", "y_origin_px", "rotation_deg", "scale_m_per_px", "dt_s",
            "X0_px", "Y0_px", "dx_px", "dy_px", "L_px", "Diameter_px", "Area_px",
            "X_mm", "Y_mm", "dx_ms", "dy_ms", "L_ms",
        ]
        with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            for row in rows:
                transformed = _transform_row(row, transform, self.config.scale_m_per_px, self.config.dt_seconds)
                writer.writerow({
                    "experiment_id": row["experiment_id"],
                    "experiment_name": row["experiment_name"],
                    "camera": camera,
                    "distance_mm": distance_mm,
                    "source_file": row["source_file"],
                    "x_origin_px": _format_float(transform.x_origin),
                    "y_origin_px": _format_float(transform.y_origin),
                    "rotation_deg": _format_float(transform.rotation_angle),
                    "scale_m_per_px": _format_float(self.config.scale_m_per_px),
                    "dt_s": _format_float(self.config.dt_seconds),
                    "X0_px": _format_float(row["X0"]),
                    "Y0_px": _format_float(row["Y0"]),
                    "dx_px": _format_float(row["dx"]),
                    "dy_px": _format_float(row["dy"]),
                    "L_px": _format_float(row["L"]),
                    "Diameter_px": _format_float(row["Diameter"]),
                    "Area_px": _format_float(row["Area"]),
                    "X_mm": _format_float(transformed["X_mm"]),
                    "Y_mm": _format_float(transformed["Y_mm"]),
                    "dx_ms": _format_float(transformed["dx_ms"]),
                    "dy_ms": _format_float(transformed["dy_ms"]),
                    "L_ms": _format_float(transformed["L_ms"]),
                })
        return output_path

    def _save_group_vector_png(
        self,
        group_folder: Path,
        rows: Sequence[dict],
        camera: str,
        distance_mm: int,
        transform: TransformSettings,
    ) -> Path:
        output_path = group_folder / "vectors_transformed.png"
        transformed_rows = [
            _transform_row(row, transform, self.config.scale_m_per_px, self.config.dt_seconds)
            for row in rows
        ]
        if len(transformed_rows) > self.config.max_vectors_for_plot:
            indices = np.linspace(0, len(transformed_rows) - 1, self.config.max_vectors_for_plot, dtype=int)
            plot_rows = [transformed_rows[int(index)] for index in indices]
            sampled_text = f", показано {len(plot_rows)} из {len(transformed_rows)}"
        else:
            plot_rows = transformed_rows
            sampled_text = ""

        figure = Figure(figsize=(10, 7), dpi=self.config.plot_dpi, facecolor="white")
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)

        if plot_rows:
            x = np.asarray([row["X_mm"] for row in plot_rows], dtype=float)
            y = np.asarray([row["Y_mm"] for row in plot_rows], dtype=float)
            dx = np.asarray([row["dx_ms"] for row in plot_rows], dtype=float)
            dy = np.asarray([row["dy_ms"] for row in plot_rows], dtype=float)
            length = np.asarray([row["L_ms"] for row in plot_rows], dtype=float)
            quiver = axis.quiver(
                x,
                y,
                dx,
                dy,
                length,
                cmap="viridis",
                scale=max(1e-9, 10.0 * self.config.arrow_scale),
                width=0.003,
                headwidth=4.0,
                headlength=5.0,
            )
            figure.colorbar(quiver, ax=axis, label="L, m/s")
        else:
            axis.text(0.5, 0.5, "Нет векторов для графика", ha="center", va="center", transform=axis.transAxes)

        axis.set_title(
            f"{camera}, {distance_mm} mm: векторы\n"
            f"origin=({transform.x_origin:.2f}, {transform.y_origin:.2f}) px, "
            f"угол={transform.rotation_angle:.3g} deg{sampled_text}"
        )
        axis.set_xlabel("X, mm")
        axis.set_ylabel("Y, mm")
        axis.grid(True, alpha=0.25)
        axis.set_aspect("equal", adjustable="datalim")
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.config.plot_dpi)
        figure.clear()
        return output_path

    def _save_mode_vector_outputs(
        self,
        group_folder: Path,
        sources: Sequence[SummarySource],
        camera: str,
        distance_mm: int,
        transform: TransformSettings,
        warnings: List[str],
    ) -> Dict[str, object]:
        vector_folder = group_folder / "vectors_by_mode"
        vector_folder.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()
        csv_paths: List[str] = []
        png_paths: List[str] = []
        vector_count = 0
        mode_count = 0

        for source in sources:
            averaged_path = source.averaged_files.get(camera)
            if averaged_path is None:
                warnings.append(f"Нет усредненного CSV для {source.experiment_name}, {camera}: ожидается cam_X_pairs_sum*averaged*.csv")
                continue

            rows = list(_read_averaged_rows(averaged_path, source, camera, warnings))
            if not rows:
                warnings.append(f"Нет строк усредненных векторов: {averaged_path}")
                continue

            stem = _unique_stem(_safe_filename(source.experiment_name), used_names)
            mode_count += 1
            vector_count += len(rows)

            if self.config.export_csv:
                csv_paths.append(str(self._save_mode_vectors_csv(
                    vector_folder,
                    stem,
                    rows,
                    transform,
                )))
            if self.config.export_png:
                png_paths.append(str(self._save_mode_vector_png(
                    vector_folder,
                    stem,
                    source,
                    camera,
                    distance_mm,
                    rows,
                    transform,
                )))

        return {
            "folder": str(vector_folder),
            "csv": str(vector_folder) if csv_paths else "",
            "png": str(vector_folder) if png_paths else "",
            "csv_count": len(csv_paths),
            "png_count": len(png_paths),
            "mode_count": mode_count,
            "vector_count": vector_count,
        }

    def _save_mode_vectors_csv(
        self,
        vector_folder: Path,
        stem: str,
        rows: Sequence[dict],
        transform: TransformSettings,
    ) -> Path:
        output_path = vector_folder / f"vectors_{stem}_averaged_transformed.csv"
        fieldnames = ["X_mm", "Y_mm", "dx_ms", "dy_ms"]
        with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            for row in rows:
                transformed = _transform_averaged_row(row, transform, self.config.scale_m_per_px, self.config.dt_seconds)
                writer.writerow({
                    "X_mm": _format_float(transformed["X_mm"]),
                    "Y_mm": _format_float(transformed["Y_mm"]),
                    "dx_ms": _format_float(transformed["dx_ms"]),
                    "dy_ms": _format_float(transformed["dy_ms"]),
                })
        return output_path

    def _save_mode_vector_png(
        self,
        vector_folder: Path,
        stem: str,
        source: SummarySource,
        camera: str,
        distance_mm: int,
        rows: Sequence[dict],
        transform: TransformSettings,
    ) -> Path:
        output_path = vector_folder / f"vectors_{stem}_averaged_transformed.png"
        transformed_rows = [
            _transform_averaged_row(row, transform, self.config.scale_m_per_px, self.config.dt_seconds)
            for row in rows
        ]
        if len(transformed_rows) > self.config.max_vectors_for_plot:
            indices = np.linspace(0, len(transformed_rows) - 1, self.config.max_vectors_for_plot, dtype=int)
            plot_rows = [transformed_rows[int(index)] for index in indices]
            sampled_text = f", показано {len(plot_rows)} из {len(transformed_rows)}"
        else:
            plot_rows = transformed_rows
            sampled_text = ""

        figure = Figure(figsize=(10, 7), dpi=self.config.plot_dpi, facecolor="white")
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)

        if plot_rows:
            x = np.asarray([row["X_mm"] for row in plot_rows], dtype=float)
            y = np.asarray([row["Y_mm"] for row in plot_rows], dtype=float)
            dx = np.asarray([row["dx_ms"] for row in plot_rows], dtype=float)
            dy = np.asarray([row["dy_ms"] for row in plot_rows], dtype=float)
            length = np.asarray([row["L_ms"] for row in plot_rows], dtype=float)
            quiver = axis.quiver(
                x,
                y,
                dx,
                dy,
                length,
                cmap="viridis",
                scale=max(1e-9, 10.0 * self.config.arrow_scale),
                width=0.003,
                headwidth=4.0,
                headlength=5.0,
            )
            figure.colorbar(quiver, ax=axis, label="L, m/s")
        else:
            axis.text(0.5, 0.5, "Нет усредненных векторов для графика", ha="center", va="center", transform=axis.transAxes)

        axis.set_title(
            f"{source.experiment_name}: {camera}, {distance_mm} mm\n"
            f"origin=({transform.x_origin:.2f}, {transform.y_origin:.2f}) px, "
            f"угол={transform.rotation_angle:.3g} deg, "
            f"scale={self.config.scale_m_per_px:.3g} m/px, "
            f"arrow_scale={self.config.arrow_scale:.3g}{sampled_text}"
        )
        axis.set_xlabel("X, mm")
        axis.set_ylabel("Y, mm")
        axis.grid(True, alpha=0.25)
        axis.set_aspect("equal", adjustable="datalim")
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.config.plot_dpi)
        figure.clear()
        return output_path

    def _save_parameters_csv(self, output_root: Path, groups: Sequence[SummaryGroupResult]) -> Path:
        output_path = output_root / "analysis_parameters.csv"
        group_by_key = {_group_key(item.camera, item.distance_mm): item for item in groups}
        with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow([
                "camera", "distance_mm", "x_origin_px", "y_origin_px", "rotation_deg",
                "scale_m_per_px", "arrow_scale", "dt_s", "bin_start_px", "bin_width_px",
                "experiments", "particles", "modes", "averaged_vectors",
            ])
            for camera, distance_mm in GROUP_ORDER:
                transform = self.config.transform_for(camera, distance_mm)
                group = group_by_key.get(_group_key(camera, distance_mm), SummaryGroupResult(camera, distance_mm))
                writer.writerow([
                    camera,
                    distance_mm,
                    _format_float(transform.x_origin),
                    _format_float(transform.y_origin),
                    _format_float(transform.rotation_angle),
                    _format_float(self.config.scale_m_per_px),
                    _format_float(self.config.arrow_scale),
                    _format_float(self.config.dt_seconds),
                    _format_float(self.config.bin_start),
                    _format_float(self.config.bin_width),
                    group.experiment_count,
                    group.particle_count,
                    group.mode_count,
                    group.vector_count,
                ])
        return output_path

    def _emit(self, percentage: float, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(SummaryProgress(max(0.0, min(100.0, percentage)), message))


def _group_key(camera: str, distance_mm: int) -> str:
    return f"{camera}_{int(distance_mm)}mm"


def _distance_from_name(name: str) -> Optional[int]:
    match = re.search(r"(?<!\d)(10|30)\s*(?:mm|мм)(?!\d)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _camera_files(ptv_folder: Path) -> Dict[str, Path]:
    return {
        camera: ptv_folder / f"{camera}_pairs_sum.csv"
        for camera in CAMERAS
        if (ptv_folder / f"{camera}_pairs_sum.csv").is_file()
    }


def _camera_averaged_files(ptv_folder: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for camera in CAMERAS:
        candidates = [
            path for path in ptv_folder.glob(f"{camera}_pairs_sum*averaged*.csv")
            if "transformed" not in path.name.lower() and path.is_file()
        ]
        if candidates:
            files[camera] = sorted(candidates, key=lambda item: (item.stat().st_mtime, item.name.lower()))[-1]
    return files


def _read_ptv_rows(
    csv_path: Path,
    source: SummarySource,
    camera: str,
    warnings: List[str],
) -> Iterable[dict]:
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=";")
            for row_index, row in enumerate(reader, start=2):
                try:
                    yield {
                        "experiment_id": source.experiment_id,
                        "experiment_name": source.experiment_name,
                        "camera": camera,
                        "distance_mm": source.distance_mm,
                        "source_file": str(csv_path),
                        "X0": _parse_float(row.get("X0")),
                        "Y0": _parse_float(row.get("Y0")),
                        "dx": _parse_float(row.get("dx")),
                        "dy": _parse_float(row.get("dy")),
                        "L": _parse_float(row.get("L")),
                        "Diameter": _parse_float(row.get("Diameter")),
                        "Area": _parse_float(row.get("Area")),
                    }
                except (TypeError, ValueError) as exc:
                    warnings.append(f"{csv_path.name}, строка {row_index}: пропущена ({exc})")
    except OSError as exc:
        warnings.append(f"Не удалось прочитать {csv_path}: {exc}")


def _read_averaged_rows(
    csv_path: Path,
    source: SummarySource,
    camera: str,
    warnings: List[str],
) -> Iterable[dict]:
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=";")
            for row_index, row in enumerate(reader, start=2):
                try:
                    yield {
                        "experiment_id": source.experiment_id,
                        "experiment_name": source.experiment_name,
                        "camera": camera,
                        "distance_mm": source.distance_mm,
                        "source_file": str(csv_path),
                        "X_center": _parse_float(row.get("X_center")),
                        "Y_center": _parse_float(row.get("Y_center")),
                        "dx_avg": _parse_float(row.get("dx_avg")),
                        "dy_avg": _parse_float(row.get("dy_avg")),
                        "L_avg": _parse_float(row.get("L_avg")),
                        "count": int(round(_parse_float(row.get("count", 1)))),
                    }
                except (TypeError, ValueError) as exc:
                    warnings.append(f"{csv_path.name}, строка {row_index}: пропущена ({exc})")
    except OSError as exc:
        warnings.append(f"Не удалось прочитать {csv_path}: {exc}")


def _parse_float(value) -> float:
    if value is None:
        raise ValueError("пустое значение")
    return float(str(value).strip().replace(",", "."))


def _counts_for_rows(rows: Sequence[dict], edges: np.ndarray) -> np.ndarray:
    values = [float(row["Diameter"]) for row in rows if row.get("Diameter") is not None]
    if not values:
        return np.zeros(max(0, edges.size - 1), dtype=int)
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    return counts.astype(int)


def _bin_label(left: float, right: float) -> str:
    return f"{_format_float(left)}-{_format_float(right)} px"


def _format_float(value: float) -> str:
    return f"{float(value):.10g}"


def _transform_row(row: dict, transform: TransformSettings, scale: float, dt: float) -> dict:
    theta = math.radians(transform.rotation_angle)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    x_rel = float(row["X0"]) - transform.x_origin
    y_rel = float(row["Y0"]) - transform.y_origin
    x_rot = x_rel * cos_theta - y_rel * sin_theta
    y_rot = x_rel * sin_theta + y_rel * cos_theta

    dx = float(row["dx"])
    dy = float(row["dy"])
    dx_rot = dx * cos_theta - dy * sin_theta
    dy_rot = dx * sin_theta + dy * cos_theta

    return {
        "X_mm": x_rot * scale * 1000.0,
        "Y_mm": y_rot * scale * 1000.0,
        "dx_ms": dx_rot * scale / dt,
        "dy_ms": dy_rot * scale / dt,
        "L_ms": float(row["L"]) * scale / dt,
    }


def _transform_averaged_row(row: dict, transform: TransformSettings, scale: float, dt: float) -> dict:
    theta = math.radians(transform.rotation_angle)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    x_rel = float(row["X_center"]) - transform.x_origin
    y_rel = float(row["Y_center"]) - transform.y_origin
    x_rot = x_rel * cos_theta - y_rel * sin_theta
    y_rot = x_rel * sin_theta + y_rel * cos_theta

    dx = float(row["dx_avg"])
    dy = float(row["dy_avg"])
    dx_rot = dx * cos_theta - dy * sin_theta
    dy_rot = dx * sin_theta + dy * cos_theta

    return {
        "X_mm": x_rot * scale * 1000.0,
        "Y_mm": y_rot * scale * 1000.0,
        "dx_ms": dx_rot * scale / dt,
        "dy_ms": dy_rot * scale / dt,
        "L_ms": float(row["L_avg"]) * scale / dt,
    }


def _safe_filename(value: str) -> str:
    stem = re.sub(r"[^0-9A-Za-zА-Яа-я._-]+", "_", value.strip())
    stem = stem.strip("._-")
    return stem or "experiment"


def _unique_stem(stem: str, used_names: set[str]) -> str:
    candidate = stem
    index = 2
    while candidate.lower() in used_names:
        candidate = f"{stem}_{index}"
        index += 1
    used_names.add(candidate.lower())
    return candidate


def _unique_label(label: str, used_labels: set[str]) -> str:
    base = label.strip() or "experiment"
    candidate = base
    index = 2
    while candidate.lower() in used_labels:
        candidate = f"{base} ({index})"
        index += 1
    used_labels.add(candidate.lower())
    return candidate
