import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    exp1_cam1_input = mo.ui.text(
        value=r"D:\2f_2c_DT_WA\A10_DT600_WA600_cam_sorted\cam_1",
        label="Эксперимент 1 — cam_1",
    )
    exp1_cam2_input = mo.ui.text(
        value=r"D:\2f_2c_DT_WA\A10_DT600_WA600_cam_sorted\cam_2",
        label="Эксперимент 1 — cam_2",
    )
    exp2_cam1_input = mo.ui.text(
        value=r"D:\2f_2c_DT_WA\S8_DT600_WA600_cam_sorted\cam_1",
        label="Эксперимент 2 — cam_1",
    )
    exp2_cam2_input = mo.ui.text(
        value=r"D:\2f_2c_DT_WA\S8_DT600_WA600_cam_sorted\cam_2",
        label="Эксперимент 2 — cam_2",
    )
    n_avg_input = mo.ui.number(
        start=1, stop=50, step=1, value=3,
        label="Количество фото для усреднения (N)",
    )
    load_btn = mo.ui.run_button(label="Загрузить файлы")

    mo.vstack([
        mo.md("## Настройка"),
        mo.hstack([
            mo.vstack([mo.md("### Эксперимент 1"), exp1_cam1_input, exp1_cam2_input]),
            mo.vstack([mo.md("### Эксперимент 2"), exp2_cam1_input, exp2_cam2_input]),
        ]),
        n_avg_input,
        load_btn,
    ])
    return (
        exp1_cam1_input,
        exp1_cam2_input,
        exp2_cam1_input,
        exp2_cam2_input,
        n_avg_input,
        load_btn,
    )


@app.cell
def _(
    exp1_cam1_input,
    exp1_cam2_input,
    exp2_cam1_input,
    exp2_cam2_input,
    load_btn,
    mo,
):
    import re as _re
    from pathlib import Path as _Path

    mo.stop(not load_btn.value, mo.md("*Нажмите «Загрузить файлы»*"))

    def _get_files(folder: str) -> dict:
        """Возвращает {номер_из_имени: путь}, отсортированный по номеру."""
        p = _Path(folder)
        if not p.exists():
            raise FileNotFoundError(f"Папка не найдена: {folder}")
        result = {}
        for f in p.iterdir():
            if f.is_file() and f.name.endswith("_b.png"):
                m = _re.search(r'(\d+)_b\.png$', f.name)
                if m:
                    result[int(m.group(1))] = str(f)
        return dict(sorted(result.items()))

    try:
        files_e1c1 = _get_files(exp1_cam1_input.value)
        files_e1c2 = _get_files(exp1_cam2_input.value)
        files_e2c1 = _get_files(exp2_cam1_input.value)
        files_e2c2 = _get_files(exp2_cam2_input.value)

        def _tbl(files: dict, label: str):
            return mo.vstack([
                mo.md(f"**{label}** — {len(files)} файлов"),
                mo.ui.table(
                    [{"#": k, "файл": _Path(v).name} for k, v in files.items()],
                    selection=None,
                    page_size=10,
                ),
            ])

        _display = mo.hstack([
            _tbl(files_e1c1, "Эксп. 1 — cam_1"),
            _tbl(files_e1c2, "Эксп. 1 — cam_2"),
            _tbl(files_e2c1, "Эксп. 2 — cam_1"),
            _tbl(files_e2c2, "Эксп. 2 — cam_2"),
        ])
    except FileNotFoundError as e:
        _display = mo.callout(mo.md(f"**Ошибка:** {e}"), kind="danger")
        files_e1c1 = files_e1c2 = files_e2c1 = files_e2c2 = {}

    _display
    return files_e1c1, files_e1c2, files_e2c1, files_e2c2


@app.cell
def _(files_e1c1, files_e1c2, files_e2c1, files_e2c2, mo, n_avg_input):
    import json as _json
    from pathlib import Path as _Path

    mo.stop(not files_e1c1, mo.md("*Сначала загрузите файлы*"))

    _N = n_avg_input.value
    _STATE_FILE = _Path(__file__).parent / "indices_state.json"

    _saved = {}
    if _STATE_FILE.exists():
        try:
            _saved = _json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            _saved = {}

    def _load_defaults(key, files: dict) -> list:
        keys = list(files.keys())
        vals = _saved.get(key, [])
        # дополняем до N ключами из файлов если не хватает
        extra = [k for k in keys if k not in vals]
        return (vals + extra)[:_N]

    def _idx_arr(defaults, files: dict, label: str):
        max_val = max(files.keys()) if files else 0
        return mo.ui.array(
            [mo.ui.number(start=0, stop=max_val, step=1, value=v) for v in defaults],
            label=label,
        )

    idx_e1c1 = _idx_arr(_load_defaults("e1c1", files_e1c1), files_e1c1, f"Эксп. 1 — cam_1 ({_N} номеров)")
    idx_e1c2 = _idx_arr(_load_defaults("e1c2", files_e1c2), files_e1c2, f"Эксп. 1 — cam_2 ({_N} номеров)")
    idx_e2c1 = _idx_arr(_load_defaults("e2c1", files_e2c1), files_e2c1, f"Эксп. 2 — cam_1 ({_N} номеров)")
    idx_e2c2 = _idx_arr(_load_defaults("e2c2", files_e2c2), files_e2c2, f"Эксп. 2 — cam_2 ({_N} номеров)")

    save_idx_btn = mo.ui.run_button(label="Сохранить номера")

    mo.vstack([
        mo.md("## Выбор фотографий для усреднения"),
        mo.md("Введите номера из колонки **#** таблиц выше"),
        mo.hstack([
            mo.vstack([mo.md("### Эксперимент 1"), idx_e1c1, idx_e1c2]),
            mo.vstack([mo.md("### Эксперимент 2"), idx_e2c1, idx_e2c2]),
        ]),
        save_idx_btn,
    ])
    return idx_e1c1, idx_e1c2, idx_e2c1, idx_e2c2, save_idx_btn


@app.cell
def _(idx_e1c1, idx_e1c2, idx_e2c1, idx_e2c2, mo, save_idx_btn):
    import json as _json2
    from pathlib import Path as _Path2

    mo.stop(not save_idx_btn.value or idx_e1c1 is None)

    _STATE_FILE2 = _Path2(__file__).parent / "indices_state.json"
    _data = {
        "e1c1": [int(v) for v in idx_e1c1.value],
        "e1c2": [int(v) for v in idx_e1c2.value],
        "e2c1": [int(v) for v in idx_e2c1.value],
        "e2c2": [int(v) for v in idx_e2c2.value],
    }
    _STATE_FILE2.write_text(_json2.dumps(_data, indent=2), encoding="utf-8")

    mo.callout(
        mo.md(f"Сохранено в `{_STATE_FILE2.name}`: {_data}"),
        kind="success",
    )


@app.cell
def _(
    files_e1c1,
    files_e1c2,
    files_e2c1,
    files_e2c2,
    idx_e1c1,
    idx_e1c2,
    idx_e2c1,
    idx_e2c2,
    mo,
):
    from pathlib import Path as _Path

    mo.stop(not files_e1c1 or idx_e1c1 is None, mo.md("*Сначала загрузите файлы и введите номера*"))

    def _col(files: dict, indices, label: str):
        rows = []
        for idx in indices:
            k = int(idx)
            if k in files:
                rows.append({"#": k, "файл": _Path(files[k]).name})
            else:
                rows.append({"#": k, "файл": f"не найден"})
        return mo.vstack([
            mo.md(f"#### {label}"),
            mo.ui.table(rows, selection=None),
        ])

    mo.vstack([
        mo.md("## Выбранные фотографии"),
        mo.hstack([
            _col(files_e1c1, idx_e1c1.value, "Эксп. 1 — cam_1"),
            _col(files_e1c2, idx_e1c2.value, "Эксп. 1 — cam_2"),
            _col(files_e2c1, idx_e2c1.value, "Эксп. 2 — cam_1"),
            _col(files_e2c2, idx_e2c2.value, "Эксп. 2 — cam_2"),
        ], justify="start"),
    ])


@app.cell
def _(mo):
    threshold_input = mo.ui.number(
        start=0, stop=65535, step=1, value=1000,
        label="Порог (0–65535)",
    )
    process_btn = mo.ui.run_button(label="Усреднить и показать")

    mo.vstack([
        mo.md("## Пороговый фильтр (16-bit → 8-bit)"),
        threshold_input,
        process_btn,
    ])
    return process_btn, threshold_input


@app.cell
def _(
    files_e1c1,
    files_e1c2,
    files_e2c1,
    files_e2c2,
    idx_e1c1,
    idx_e1c2,
    idx_e2c1,
    idx_e2c2,
    mo,
    process_btn,
    threshold_input,
):
    import numpy as _np
    from PIL import Image as _Image

    mo.stop(
        not process_btn.value or not files_e1c1 or idx_e1c1 is None,
        mo.md("*Нажмите «Усреднить и показать»*"),
    )

    _thr = threshold_input.value

    def _load_avg(files: dict, indices):
        arrays = []
        for idx in indices:
            arr = _np.array(_Image.open(files[int(idx)]), dtype=_np.float32)
            arr[arr < _thr] = 0
            arrays.append(arr)
        avg = _np.mean(arrays, axis=0)
        return _Image.fromarray((avg / 65535.0 * 255).astype(_np.uint8))

    try:
        _r_e1c1 = _load_avg(files_e1c1, idx_e1c1.value)
        _r_e1c2 = _load_avg(files_e1c2, idx_e1c2.value)
        _r_e2c1 = _load_avg(files_e2c1, idx_e2c1.value)
        _r_e2c2 = _load_avg(files_e2c2, idx_e2c2.value)

        mo.vstack([
            mo.md(f"## Результат усреднения (порог: {_thr} / 65535)"),
            mo.md("### Эксперимент 1"),
            mo.hstack([
                mo.vstack([mo.md("**cam_1**"), mo.image(_r_e1c1, width=700)]),
                mo.vstack([mo.md("**cam_2**"), mo.image(_r_e1c2, width=700)]),
            ]),
            mo.md("### Эксперимент 2"),
            mo.hstack([
                mo.vstack([mo.md("**cam_1**"), mo.image(_r_e2c1, width=700)]),
                mo.vstack([mo.md("**cam_2**"), mo.image(_r_e2c2, width=700)]),
            ]),
        ])
    except Exception as e:
        mo.callout(mo.md(f"**Ошибка:** {e}"), kind="danger")


if __name__ == "__main__":
    app.run()