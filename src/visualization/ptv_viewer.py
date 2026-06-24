"""Utilities for browsing and previewing PTV analysis results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import csv
import re


@dataclass
class PTVPairRecord:
    """Summary for one PTV pair CSV."""

    camera: str
    pair_number: int
    csv_path: str
    matches_count: int
    mean_l: float
    max_l: float
    mean_dx: float
    mean_dy: float
    source_a_path: str
    source_b_path: str
    source_ok: bool


@dataclass
class PTVScanResult:
    """Result of scanning a PTV output folder."""

    ptv_folder: str
    original_folder: str
    records: List[PTVPairRecord]
    errors: List[str]


def infer_original_folder(ptv_folder: str) -> str:
    """Infer the binary_filter_<threshold> folder next to a PTV_<threshold> folder."""
    ptv_path = Path(ptv_folder)
    match = re.match(r"PTV_([^_]+)_", ptv_path.name)
    if not match:
        return ""

    threshold = match.group(1)
    candidate = ptv_path.parent / f"binary_filter_{threshold}"
    return str(candidate) if candidate.exists() else ""


def scan_ptv_pairs(ptv_folder: str, original_folder: Optional[str] = None) -> PTVScanResult:
    """Scan cam_X_pairs folders and calculate lightweight pair statistics."""
    ptv_path = Path(ptv_folder)
    errors: List[str] = []
    records: List[PTVPairRecord] = []

    if not ptv_path.exists():
        return PTVScanResult(ptv_folder=str(ptv_path), original_folder="", records=[], errors=[
            f"PTV folder does not exist: {ptv_folder}"
        ])
    if not ptv_path.is_dir():
        return PTVScanResult(ptv_folder=str(ptv_path), original_folder="", records=[], errors=[
            f"PTV path is not a folder: {ptv_folder}"
        ])

    original_path = Path(original_folder) if original_folder else None
    if original_path is None or not original_path.exists():
        inferred = infer_original_folder(ptv_folder)
        original_path = Path(inferred) if inferred else None

    if original_path is None or not original_path.exists():
        errors.append("Cannot find binary_filter_<threshold> folder with source images")

    for camera in ("cam_1", "cam_2"):
        pairs_folder = ptv_path / f"{camera}_pairs"
        if not pairs_folder.exists():
            errors.append(f"Pairs folder not found: {pairs_folder}")
            continue

        for csv_path in sorted(pairs_folder.glob("*_pair.csv"), key=_pair_sort_key):
            pair_number = _extract_pair_number(csv_path)
            if pair_number is None:
                continue

            stats = _read_pair_stats(csv_path)
            source_a = original_path / camera / f"{pair_number}_a.png" if original_path else Path()
            source_b = original_path / camera / f"{pair_number}_b.png" if original_path else Path()
            source_ok = original_path is not None and source_a.exists() and source_b.exists()

            records.append(PTVPairRecord(
                camera=camera,
                pair_number=pair_number,
                csv_path=str(csv_path),
                matches_count=stats["count"],
                mean_l=stats["mean_l"],
                max_l=stats["max_l"],
                mean_dx=stats["mean_dx"],
                mean_dy=stats["mean_dy"],
                source_a_path=str(source_a) if original_path else "",
                source_b_path=str(source_b) if original_path else "",
                source_ok=source_ok,
            ))

    return PTVScanResult(
        ptv_folder=str(ptv_path),
        original_folder=str(original_path) if original_path else "",
        records=records,
        errors=errors,
    )


def _read_pair_stats(csv_path: Path) -> dict:
    count = 0
    sum_l = 0.0
    sum_dx = 0.0
    sum_dy = 0.0
    max_l = 0.0

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                try:
                    length = float(row.get("L", 0) or 0)
                    dx = float(row.get("dx", 0) or 0)
                    dy = float(row.get("dy", 0) or 0)
                except ValueError:
                    continue
                count += 1
                sum_l += length
                sum_dx += dx
                sum_dy += dy
                max_l = max(max_l, length)
    except OSError:
        pass

    return {
        "count": count,
        "mean_l": sum_l / count if count else 0.0,
        "max_l": max_l,
        "mean_dx": sum_dx / count if count else 0.0,
        "mean_dy": sum_dy / count if count else 0.0,
    }


def _pair_sort_key(csv_path: Path):
    pair_number = _extract_pair_number(csv_path)
    return (0, pair_number) if pair_number is not None else (1, csv_path.name)


def _extract_pair_number(csv_path: Path) -> Optional[int]:
    try:
        return int(csv_path.stem.split("_")[0])
    except (ValueError, IndexError):
        return None
