import csv
from pathlib import Path

from execute.full_pipeline import AutomatedPipelineConfig, AutomatedPipelineExecutor


def _write_diameters(path, values):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["Diameter"])
        writer.writerows([[value] for value in values])


def _histogram_count(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(int(row["count"]) for row in csv.DictReader(stream, delimiter=";"))


def _executor(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    config = AutomatedPipelineConfig(
        input_root=str(input_root),
        output_root=str(tmp_path / "output"),
        run_filter=False,
        run_average=False,
        histogram_bin_width=1,
        plot_dpi=40,
    )
    return AutomatedPipelineExecutor(config, [])


def test_frame_sequence_graphs_only_include_available_camera(tmp_path):
    executor = _executor(tmp_path)
    executor.config.input_mode = "frame_sequence"
    source = tmp_path / "cam_1.csv"
    _write_diameters(source, [2, 3, 4])

    outputs = executor._create_graphs(tmp_path / "ptv", {"raw": {"cam_1": source}})

    assert set(outputs) == {
        "histogram_cam_1_csv", "histogram_cam_1_png",
        "histogram_all_cameras_csv", "histogram_all_cameras_png",
    }
    assert _histogram_count(tmp_path / "ptv/graphs/particle_diameter_histogram_cam_1.csv") == 3
    assert _histogram_count(tmp_path / "ptv/graphs/particle_diameter_histogram_all_cameras.csv") == 3
    assert all(Path(path).is_file() for path in outputs.values())


def test_actual_flow_graphs_keep_two_camera_and_combined_histograms(tmp_path):
    executor = _executor(tmp_path)
    cam1 = tmp_path / "cam_1.csv"
    cam2 = tmp_path / "cam_2.csv"
    _write_diameters(cam1, [2, 3])
    _write_diameters(cam2, [4, 5, 6])

    outputs = executor._create_graphs(
        tmp_path / "ptv", {"raw": {"cam_1": cam1, "cam_2": cam2}}
    )

    assert "histogram_cam_1_csv" in outputs
    assert "histogram_cam_2_csv" in outputs
    assert "histogram_all_cameras_csv" in outputs
    assert _histogram_count(tmp_path / "ptv/graphs/particle_diameter_histogram_all_cameras.csv") == 5
    assert _histogram_count(tmp_path / "ptv/graphs/particle_diameter_histogram_cam_1.csv") == 2
    assert _histogram_count(tmp_path / "ptv/graphs/particle_diameter_histogram_cam_2.csv") == 3
