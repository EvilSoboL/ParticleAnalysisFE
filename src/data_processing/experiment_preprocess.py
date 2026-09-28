"""
Discovery helpers for raw experiment folders described by .afxml files.

The acquisition software stores image folders with technical names such as
``experiment178`` and keeps the readable experiment name in a sibling
``experiment178.afxml`` file.  This module builds a small manifest that the GUI
can use before running Sort + Binarize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import List, Optional
import re
import xml.etree.ElementTree as ET


WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


@dataclass
class SequenceMetadata:
    """Optional acquisition metadata supplied next to a frame sequence."""

    width_px: Optional[int] = None
    height_px: Optional[int] = None
    bit_depth: Optional[int] = None
    scale_m_per_px: Optional[float] = None
    dt_seconds: Optional[float] = None
    dark_particles: Optional[bool] = None


@dataclass
class ExperimentRecord:
    """One experiment discovered from an .afxml file."""

    experiment_id: str
    name: str
    safe_name: str
    source_folder: str
    afxml_path: str
    data_link: str = ""
    png_count: int = 0
    sort_ready: bool = False
    skipped: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    input_mode: str = "actual_flow"
    sequence_metadata: Optional[SequenceMetadata] = None

    @property
    def status(self) -> str:
        if self.skipped:
            return "Skipped"
        if self.errors:
            return "Error"
        if self.warnings:
            return "Warning"
        return "OK"

    @property
    def issue_text(self) -> str:
        return "; ".join(self.errors + self.warnings)


@dataclass
class ExperimentScanResult:
    """Result of scanning an experiment root folder."""

    root_folder: str
    records: List[ExperimentRecord]
    errors: List[str] = field(default_factory=list)


def sanitize_experiment_name(name: str, fallback: str = "experiment") -> str:
    """Return a filesystem-safe folder name for Windows paths."""
    safe = (name or "").strip()
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", safe)
    safe = re.sub(r"\s+", "_", safe)
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip(" ._")

    if not safe:
        safe = fallback

    if safe.upper() in WINDOWS_RESERVED_NAMES:
        safe = f"{safe}_experiment"

    return safe[:120]


def scan_experiment_root(root_folder: str) -> ExperimentScanResult:
    """Scan a folder with experiment*.afxml files and return a manifest."""
    root_path = Path(root_folder)
    if not root_path.exists():
        return ExperimentScanResult(
            root_folder=str(root_path),
            records=[],
            errors=[f"Root folder does not exist: {root_folder}"],
        )
    if not root_path.is_dir():
        return ExperimentScanResult(
            root_folder=str(root_path),
            records=[],
            errors=[f"Path is not a folder: {root_folder}"],
        )

    records: List[ExperimentRecord] = []
    used_safe_names: set[str] = set()

    for afxml_path in sorted(root_path.glob("experiment*.afxml")):
        record = parse_experiment_afxml(afxml_path)
        record.safe_name = _make_unique_safe_name(record.safe_name, record.experiment_id, used_safe_names)
        used_safe_names.add(record.safe_name)
        records.append(record)

    errors = []
    if not records:
        errors.append(f"No experiment*.afxml files found in {root_folder}")

    return ExperimentScanResult(root_folder=str(root_path), records=records, errors=errors)


def get_sequence_frames(folder: Path) -> List[Path]:
    """Return frame_<number>.png files in numeric order, validating the sequence."""
    try:
        png_files = list(folder.glob("*.png"))
    except OSError as exc:
        raise ValueError(f"Не удалось прочитать папку кадров: {exc}") from exc

    if not png_files:
        raise ValueError("В папке не найдены PNG-кадры")

    indexed: dict[int, Path] = {}
    for path in png_files:
        match = re.fullmatch(r"frame_(\d+)\.png", path.name)
        if not match:
            raise ValueError(f"Недопустимое имя PNG-файла: {path.name}. Ожидается frame_<число>.png")
        index = int(match.group(1))
        if index in indexed:
            raise ValueError(f"Повторяется номер кадра {index}: {indexed[index].name} и {path.name}")
        indexed[index] = path

    frames = [indexed[index] for index in sorted(indexed)]
    if len(frames) < 2:
        raise ValueError("Для серии требуется не менее двух кадров")
    if len(frames) % 2:
        raise ValueError(f"Количество кадров должно быть чётным: {len(frames)}")
    for previous, current in zip(sorted(indexed), sorted(indexed)[1:]):
        if current != previous + 1:
            raise ValueError(f"В нумерации кадров пропуск: после frame_{previous}.png отсутствует frame_{previous + 1}.png")
    return frames


def scan_frame_sequence(folder: str) -> ExperimentScanResult:
    """Scan a directory containing a single numbered PNG frame sequence."""
    folder_path = Path(folder)
    if not folder_path.exists():
        return ExperimentScanResult(str(folder_path), [], [f"Папка кадров не существует: {folder}"])
    if not folder_path.is_dir():
        return ExperimentScanResult(str(folder_path), [], [f"Путь к кадрам не является папкой: {folder}"])

    name = folder_path.parent.name if folder_path.name.lower() == "frames" else folder_path.name
    record = ExperimentRecord(
        experiment_id=name,
        name=name,
        safe_name=sanitize_experiment_name(name),
        source_folder=str(folder_path),
        afxml_path="",
        input_mode="frame_sequence",
    )
    record.sequence_metadata = _read_sequence_metadata(folder_path.parent / "config.json", record.warnings)
    try:
        frames = get_sequence_frames(folder_path)
        record.png_count = len(frames)
        record.sort_ready = True
    except ValueError as exc:
        record.png_count = len(list(folder_path.glob("*.png")))
        record.errors.append(str(exc))
    return ExperimentScanResult(str(folder_path), [record])


def _read_sequence_metadata(config_path: Path, warnings: List[str]) -> Optional[SequenceMetadata]:
    """Read recognized optional values from the sibling config.json."""
    metadata = SequenceMetadata()
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        warnings.append(f"Файл метаданных не найден: {config_path}")
        return metadata
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        warnings.append(f"Не удалось прочитать config.json: {exc}")
        return metadata

    if not isinstance(config, dict):
        warnings.append("Некорректный config.json: ожидается JSON-объект")
        return metadata

    camera = config.get("camera", {})
    if not isinstance(camera, dict):
        warnings.append("Некорректное поле camera в config.json")
        camera = {}
    background = config.get("background", {})
    if not isinstance(background, dict):
        warnings.append("Некорректное поле background в config.json")
        background = {}

    def positive_number(section: dict, key: str, label: str, *, integer: bool = False):
        if key not in section:
            return None
        value = section[key]
        valid_type = type(value) is int if integer else type(value) in (int, float)
        finite = isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
        if not valid_type or not finite or value <= 0:
            warnings.append(f"Некорректное значение {label} в config.json")
            return None
        if integer and value <= 0:
            warnings.append(f"Некорректное значение {label} в config.json")
            return None
        return value

    metadata.width_px = positive_number(camera, "width_px", "camera.width_px", integer=True)
    metadata.height_px = positive_number(camera, "height_px", "camera.height_px", integer=True)
    metadata.bit_depth = positive_number(camera, "bit_depth", "camera.bit_depth", integer=True)
    if metadata.bit_depth not in (None, 8, 16):
        warnings.append("Некорректное значение camera.bit_depth в config.json: допустимы 8 или 16")
        metadata.bit_depth = None

    scale = positive_number(camera, "scale_um_per_px", "camera.scale_um_per_px")
    if scale is not None:
        metadata.scale_m_per_px = scale * 1e-6
    interval = positive_number(camera, "frame_interval_us", "camera.frame_interval_us")
    if interval is not None:
        metadata.dt_seconds = interval * 1e-6

    if "tone" in background:
        tone = background["tone"]
        if tone == "light":
            metadata.dark_particles = True
        elif tone == "dark":
            metadata.dark_particles = False
        else:
            warnings.append("Некорректное значение background.tone в config.json: допустимы light или dark")
    return metadata


def parse_experiment_afxml(afxml_path: Path) -> ExperimentRecord:
    """Parse one .afxml metadata file."""
    fallback_id = _extract_id_from_stem(afxml_path.stem)
    fallback_folder = afxml_path.parent / afxml_path.stem

    record = ExperimentRecord(
        experiment_id=fallback_id,
        name=afxml_path.stem,
        safe_name=sanitize_experiment_name(afxml_path.stem),
        source_folder=str(fallback_folder),
        afxml_path=str(afxml_path),
    )

    try:
        tree = ET.parse(str(afxml_path))
        root = tree.getroot()
    except ET.ParseError as exc:
        record.errors.append(f"Cannot parse XML: {exc}")
        return record
    except OSError as exc:
        record.errors.append(f"Cannot read file: {exc}")
        return record

    node_record = root.find(".//node_record")
    if node_record is not None:
        name = node_record.attrib.get("name", "").strip()
        experiment_id = node_record.attrib.get("id", "").strip()
        if name:
            record.name = name
            record.safe_name = sanitize_experiment_name(name, fallback=afxml_path.stem)
        if experiment_id:
            record.experiment_id = experiment_id
    else:
        record.warnings.append("node_record not found")

    data_link = _find_data_link(root)
    if data_link:
        record.data_link = data_link
        record.source_folder = str(_resolve_data_link(afxml_path.parent, data_link))
    else:
        record.warnings.append("data link not found; using afxml file stem")

    _fill_folder_stats(record)
    return record


def default_processed_root(root_folder: str) -> str:
    """Default folder for named Sort + Binarize outputs."""
    root_path = Path(root_folder)
    return str(root_path.parent / f"{root_path.name}_processed")


def build_output_base_folder(output_root: str, record: ExperimentRecord) -> str:
    """Return the folder that Sort + Binarize should use as output base."""
    return str(Path(output_root) / record.safe_name)


def _find_data_link(root: ET.Element) -> Optional[str]:
    for link in root.findall(".//data_links/p"):
        value = link.attrib.get("v", "").strip()
        if value:
            return value
    return None


def _resolve_data_link(base_folder: Path, data_link: str) -> Path:
    link_path = Path(data_link)
    if link_path.is_absolute():
        return link_path
    return (base_folder / link_path).resolve()


def _fill_folder_stats(record: ExperimentRecord) -> None:
    source_folder = Path(record.source_folder)
    if not source_folder.exists():
        record.errors.append(f"Source folder does not exist: {source_folder}")
        return
    if not source_folder.is_dir():
        record.errors.append(f"Source path is not a folder: {source_folder}")
        return

    record.png_count = len(list(source_folder.glob("*.png")))
    if record.png_count == 0:
        record.errors.append("No PNG files found")
    elif record.png_count % 4 != 0:
        record.errors.append(f"PNG count is not divisible by 4: {record.png_count}")
    else:
        record.sort_ready = True

    if record.name.strip().lower() in {"new experiment", "experiment"}:
        record.warnings.append("Experiment name looks generic")


def _extract_id_from_stem(stem: str) -> str:
    match = re.search(r"(\d+)$", stem)
    return match.group(1) if match else stem


def _make_unique_safe_name(safe_name: str, experiment_id: str, used: set[str]) -> str:
    if safe_name not in used:
        return safe_name

    candidate = f"{safe_name}_{experiment_id}"
    if candidate not in used:
        return candidate

    index = 2
    while f"{candidate}_{index}" in used:
        index += 1
    return f"{candidate}_{index}"
