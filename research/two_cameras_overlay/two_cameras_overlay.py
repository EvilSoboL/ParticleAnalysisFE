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
        load_btn,
        n_avg_input,
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
def _(idx_e1c1, idx_e1c2, idx_e2c1, idx_e2c2, lower_bound_input, mo, save_idx_btn, threshold_input):
    import json as _json2
    from pathlib import Path as _Path2

    mo.stop(not save_idx_btn.value or idx_e1c1 is None)

    _STATE_FILE2 = _Path2(__file__).parent / "indices_state.json"
    _data = {
        "threshold": int(threshold_input.value),
        "lower_bound": int(lower_bound_input.value),
        "e1c1": [int(v) for v in idx_e1c1.value],
        "e1c2": [int(v) for v in idx_e1c2.value],
        "e2c1": [int(v) for v in idx_e2c1.value],
        "e2c2": [int(v) for v in idx_e2c2.value],
    }
    _STATE_FILE2.write_text(_json2.dumps(_data, indent=2), encoding="utf-8")

    mo.callout(
        mo.md(f"Сохранено в `{_STATE_FILE2.name}`"),
        kind="success",
    )
    return


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
    return


@app.cell
def _(mo):
    import json as _json_thr
    from pathlib import Path as _Path_thr

    _STATE_FILE_THR = _Path_thr(__file__).parent / "indices_state.json"
    _thr_default = 1000
    _lb_default = 1000
    if _STATE_FILE_THR.exists():
        try:
            _saved_thr = _json_thr.loads(
                _STATE_FILE_THR.read_text(encoding="utf-8")
            )
            _thr_default = _saved_thr.get("threshold", 1000)
            _lb_default = _saved_thr.get("lower_bound", _thr_default)
        except Exception:
            pass

    threshold_input = mo.ui.number(
        start=0, stop=65535, step=1, value=_thr_default,
        label="Порог обнуления (0–65535)",
    )
    lower_bound_input = mo.ui.number(
        start=0, stop=65535, step=1, value=_lb_default,
        label="Нижняя граница сигнала (0–65535)",
    )
    mo.vstack([
        mo.md("## Пороговые фильтры (16-bit)"),
        mo.md("**Фильтр 1:** пиксели ниже порога обнуляются."),
        threshold_input,
        mo.md("**Фильтр 2:** ненулевые пиксели ниже нижней границы поднимаются до неё."),
        lower_bound_input,
    ])
    return (threshold_input, lower_bound_input)


@app.cell
def _(mo):
    preview_cam = mo.ui.dropdown(
        options={
            "Эксп. 1 — cam_1": "e1c1",
            "Эксп. 1 — cam_2": "e1c2",
            "Эксп. 2 — cam_1": "e2c1",
            "Эксп. 2 — cam_2": "e2c2",
        },
        value="Эксп. 1 — cam_1",
        label="Камера для превью",
    )
    preview_cam
    return (preview_cam,)


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
    lower_bound_input,
    mo,
    preview_cam,
    threshold_input,
):
    import numpy as _np
    from PIL import Image as _Image

    mo.stop(not files_e1c1 or idx_e1c1 is None, mo.md("*Сначала загрузите файлы и введите номера*"))

    _thr = threshold_input.value
    _lb = lower_bound_input.value

    _all = {
        "e1c1": (files_e1c1, idx_e1c1.value),
        "e1c2": (files_e1c2, idx_e1c2.value),
        "e2c1": (files_e2c1, idx_e2c1.value),
        "e2c2": (files_e2c2, idx_e2c2.value),
    }
    _files, _indices = _all[preview_cam.value]
    _k = int(_indices[0])

    if _k in _files:
        _arr = _np.array(_Image.open(_files[_k]), dtype=_np.float64)
        _arr_max = _arr.max() or 1.0
        _orig = _Image.fromarray((_np.clip(_arr / _arr_max * 255, 0, 255)).astype(_np.uint8))
        _filt = _arr.copy()
        _filt[_filt < _thr] = 0
        _filt = _np.where((_filt > 0) & (_filt < _lb), _lb, _filt)
        _filt_img = _Image.fromarray((_np.clip(_filt / _arr_max * 255, 0, 255)).astype(_np.uint8))
        _out = mo.vstack([
            mo.md(f"### Превью — `#{_k}` — порог {_thr}, нижняя граница {_lb}"),
            mo.hstack([
                mo.vstack([mo.md("**Оригинал**"), mo.image(_orig, width=800)]),
                mo.vstack([mo.md(f"**С фильтрами**"), mo.image(_filt_img, width=800)]),
            ]),
        ])
    else:
        _out = mo.callout(mo.md(f"Файл #{_k} не найден"), kind="warn")

    _out


@app.cell
def _(mo):
    process_btn = mo.ui.run_button(label="Суммировать и показать")
    mo.vstack([
        mo.md("### Применить ко всем выбранным фотографиям"),
        process_btn,
    ])
    return (process_btn,)


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
    lower_bound_input,
    mo,
    process_btn,
    threshold_input,
):
    import numpy as _np
    from PIL import Image as _Image

    mo.stop(
        not process_btn.value or not files_e1c1 or idx_e1c1 is None,
        mo.md("*Нажмите «Суммировать и показать»*"),
    )

    _thr = threshold_input.value
    _lb = lower_bound_input.value

    def _sum_cam_arr(files: dict, indices):
        """Возвращает numpy float64 массив 0-255, нормализованный по frame_max."""
        total = None
        frame_max = 0.0
        for idx in indices:
            arr = _np.array(_Image.open(files[int(idx)]), dtype=_np.float64)
            arr[arr < _thr] = 0
            arr = _np.where((arr > 0) & (arr < _lb), _lb, arr)
            frame_max = max(frame_max, float(arr.max()))
            total = arr if total is None else total + arr
        if frame_max > 0:
            total = _np.clip(total / frame_max * 255, 0, 255)
        return total

    def _to_pil(arr):
        return _Image.fromarray(_np.clip(arr, 0, 255).astype(_np.uint8))

    try:
        sum_e1c1 = _sum_cam_arr(files_e1c1, idx_e1c1.value)
        sum_e1c2 = _sum_cam_arr(files_e1c2, idx_e1c2.value)
        sum_e2c1 = _sum_cam_arr(files_e2c1, idx_e2c1.value)
        sum_e2c2 = _sum_cam_arr(files_e2c2, idx_e2c2.value)

        _out = mo.vstack([
            mo.md(f"## Суммарная интенсивность (порог: {_thr} / 65535)"),
            mo.md("### Эксперимент 1"),
            mo.hstack([
                mo.vstack([mo.md("**cam_1**"), mo.image(_to_pil(sum_e1c1), width=700)]),
                mo.vstack([mo.md("**cam_2**"), mo.image(_to_pil(sum_e1c2), width=700)]),
            ]),
            mo.md("### Эксперимент 2"),
            mo.hstack([
                mo.vstack([mo.md("**cam_1**"), mo.image(_to_pil(sum_e2c1), width=700)]),
                mo.vstack([mo.md("**cam_2**"), mo.image(_to_pil(sum_e2c2), width=700)]),
            ]),
        ])
    except Exception as e:
        _out = mo.callout(mo.md(f"**Ошибка:** {e}"), kind="danger")
        sum_e1c1 = sum_e1c2 = sum_e2c1 = sum_e2c2 = None

    _out
    return sum_e1c1, sum_e1c2, sum_e2c1, sum_e2c2


@app.cell
def _(mo):
    c1_r = mo.ui.number(0, 255, step=1, value=255, label="R")
    c1_g = mo.ui.number(0, 255, step=1, value=0,   label="G")
    c1_b = mo.ui.number(0, 255, step=1, value=0,   label="B")
    c2_r = mo.ui.number(0, 255, step=1, value=0,   label="R")
    c2_g = mo.ui.number(0, 255, step=1, value=0,   label="G")
    c2_b = mo.ui.number(0, 255, step=1, value=255, label="B")
    shift_input = mo.ui.number(-2000, 2000, step=1, value=0,
                               label="Сдвиг cam_1 вниз (пиксели, отриц. = вверх)")

    mo.vstack([
        mo.md("## Наложение cam_1 на cam_2"),
        mo.hstack([
            mo.vstack([mo.md("**Цвет cam_1**"), c1_r, c1_g, c1_b]),
            mo.vstack([mo.md("**Цвет cam_2**"), c2_r, c2_g, c2_b]),
        ]),
        shift_input,
    ])
    return c1_b, c1_g, c1_r, c2_b, c2_g, c2_r, shift_input


@app.cell
def _(
    c1_b, c1_g, c1_r,
    c2_b, c2_g, c2_r,
    mo,
    shift_input,
    sum_e1c1, sum_e1c2,
    sum_e2c1, sum_e2c2,
):
    import numpy as _np
    from PIL import Image as _Image

    mo.stop(sum_e1c1 is None, mo.md("*Нажмите «Суммировать и показать»*"))

    _col1 = (float(c1_r.value), float(c1_g.value), float(c1_b.value))
    _col2 = (float(c2_r.value), float(c2_g.value), float(c2_b.value))
    _shift = int(shift_input.value)

    def _shift_arr(arr):
        if _shift == 0:
            return arr
        H = arr.shape[0]
        out = _np.zeros_like(arr)
        if _shift > 0:
            out[_shift:] = arr[:max(0, H - _shift)]
        else:
            n = -_shift
            out[:max(0, H - n)] = arr[n:]
        return out

    def _make_overlay(arr_c1, arr_c2):
        I1 = _shift_arr(arr_c1) / 255.0   # 0=фон, 1=частица
        I2 = arr_c2 / 255.0
        # Белый фон + цветные частицы: вычитаем пигмент из белого
        R = _np.clip(255 - I1 * (255 - _col1[0]) - I2 * (255 - _col2[0]), 0, 255)
        G = _np.clip(255 - I1 * (255 - _col1[1]) - I2 * (255 - _col2[1]), 0, 255)
        B = _np.clip(255 - I1 * (255 - _col1[2]) - I2 * (255 - _col2[2]), 0, 255)
        rgb = _np.stack([R, G, B], axis=-1).astype(_np.uint8)
        return _Image.fromarray(rgb, mode="RGB")

    _ov_e1 = _make_overlay(sum_e1c1, sum_e1c2)
    _ov_e2 = _make_overlay(sum_e2c1, sum_e2c2)

    mo.vstack([
        mo.md(f"## Наложение (сдвиг cam_1: {_shift} пкс)"),
        mo.hstack([
            mo.vstack([mo.md("### Эксперимент 1"), mo.image(_ov_e1, width=900)]),
            mo.vstack([mo.md("### Эксперимент 2"), mo.image(_ov_e2, width=900)]),
        ]),
    ])


if __name__ == "__main__":
    app.run()
