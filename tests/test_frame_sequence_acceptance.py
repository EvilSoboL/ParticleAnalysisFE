import csv
from pathlib import Path

import numpy as np
from PIL import Image

from execute.full_pipeline import AutomatedPipelineConfig, AutomatedPipelineExecutor
from src.data_processing.experiment_preprocess import scan_experiment_root, scan_frame_sequence


def _make_sequence(folder: Path) -> None:
    folder.mkdir()
    # One 7x7 dark particle moves two pixels right per frame on a bright field.
    for index in range(1, 7):
        frame = np.full((32, 40), 240, dtype=np.uint8)
        x = 8 + (index - 1) * 2
        frame[12:19, x:x + 7] = 10
        Image.fromarray(frame).save(folder / f"frame_{index:06d}.png")


def test_frame_sequence_pipeline_preserves_direction_and_converts_speed(tmp_path):
    frames = tmp_path / "кадры"
    _make_sequence(frames)
    record = scan_frame_sequence(str(frames)).records[0]
    assert record.sort_ready

    config = AutomatedPipelineConfig(
        input_root=str(tmp_path),
        output_root=str(tmp_path / "результат"),
        input_mode="frame_sequence",
        dark_particles=True,
        threshold=128,
        detection_min_area=40,
        detection_max_area=60,
        matching_max_distance=5,
        matching_max_diameter_diff=1,
        run_filter=False,
        run_average=False,
        run_transform=True,
        scale_m_per_px=0.001,
        dt_seconds=0.25,
        histogram_cam1=False,
        histogram_cam2=False,
        histogram_combined=False,
        vector_plot_raw=False,
        vector_plot_filtered=False,
        vector_plot_averaged=False,
        vector_plot_transformed=False,
    )
    result = AutomatedPipelineExecutor(config, [record]).execute()

    assert result.success, result.errors + (result.experiment_results[0].errors if result.experiment_results else [])
    experiment = result.experiment_results[0]
    binary = Path(experiment.files["binary_folder"])
    cam_1 = binary / "cam_1"
    assert sorted(p.name for p in cam_1.glob("*.png")) == [
        f"{pair}_{side}.png" for pair in range(1, 4) for side in ("a", "b")
    ]
    assert not (binary / "cam_2").exists()
    oriented = np.asarray(Image.open(cam_1 / "1_a.png"))
    assert oriented[12:19, 8:15].min() == 255
    assert oriented[0, 0] == 0

    transformed = Path(experiment.files["cam_1_transformed"])
    with transformed.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    assert len(rows) == 3
    velocities = [float(row["dx_ms"].replace(",", ".")) for row in rows]
    assert velocities == [0.008, 0.008, 0.008]


def test_pipeline_rejects_mismatched_actual_flow_record(tmp_path):
    frames = tmp_path / "кадры"
    _make_sequence(frames)
    record = scan_frame_sequence(str(frames)).records[0]
    config = AutomatedPipelineConfig(
        input_root=str(tmp_path), output_root=str(tmp_path / "output"),
        input_mode="actual_flow",
    )

    result = AutomatedPipelineExecutor(config, [record]).execute()

    assert not result.success
    assert "не совпадает" in result.errors[0]
    assert not (tmp_path / "output").exists()


def test_frame_sequence_pipeline_can_be_cancelled(tmp_path):
    frames = tmp_path / "frames"
    _make_sequence(frames)
    record = scan_frame_sequence(str(frames)).records[0]
    config = AutomatedPipelineConfig(
        input_root=str(tmp_path), output_root=str(tmp_path / "cancelled"),
        input_mode="frame_sequence", dark_particles=True,
        threshold=128, run_filter=False, run_average=False,
        histogram_cam1=False, histogram_cam2=False, histogram_combined=False,
        vector_plot_averaged=False,
    )
    executor = AutomatedPipelineExecutor(config, [record])
    executor.set_progress_callback(lambda progress: executor.cancel()
                                    if progress.stage == "Сортировка + бинаризация"
                                    and progress.percentage > 0 else None)

    result = executor.execute()

    assert not result.success
    assert result.cancelled


def test_actual_flow_afxml_16bit_keeps_both_cameras(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (tmp_path / "experiment1.afxml").write_text(
        '<root><node_record name="Actual Flow regression" id="1" />'
        '<data_links><p v="raw" /></data_links></root>',
        encoding="utf-8",
    )
    names = ("image1_a.png", "image1_b.png", "image2_a.png", "image2_b.png")
    for index, name in enumerate(names):
        frame = np.full((32, 40), 1000, dtype=np.uint16)
        x = 8 + (index % 2) * 2
        frame[12:19, x:x + 7] = 50000
        Image.fromarray(frame).save(raw / name)
    record = scan_experiment_root(str(tmp_path)).records[0]
    assert record.sort_ready

    config = AutomatedPipelineConfig(
        input_root=str(tmp_path), output_root=str(tmp_path / "actual-output"),
        threshold=30000, detection_min_area=40, detection_max_area=60,
        matching_max_distance=5, matching_max_diameter_diff=1,
        run_filter=False, run_average=False, run_transform=False,
        histogram_cam1=False, histogram_cam2=False, histogram_combined=False,
        vector_plot_raw=False, vector_plot_averaged=False,
    )
    result = AutomatedPipelineExecutor(config, [record]).execute()

    assert result.success, result.errors + (result.experiment_results[0].errors if result.experiment_results else [])
    experiment = result.experiment_results[0]
    binary = Path(experiment.files["binary_folder"])
    assert sorted(p.name for p in (binary / "cam_1").glob("*.png")) == ["1_a.png", "1_b.png"]
    assert sorted(p.name for p in (binary / "cam_2").glob("*.png")) == ["1_a.png", "1_b.png"]
    assert Path(experiment.files["cam_1_raw"]).is_file()
    assert Path(experiment.files["cam_2_raw"]).is_file()
