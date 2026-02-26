import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Режимы для анализа: A10_DT600_WA600, S8_DT600_WA600
    """)
    return


@app.cell
def _():
    import os
    from pathlib import Path

    def get_first_images(cam_1_folder: str, cam_2_folder: str, n: int = 10):
        """
        Получает первые n изображений из двух папок.
    
        Args:
            cam_1_folder: путь к папке cam_1
            cam_2_folder: путь к папке cam_2
            n: количество изображений (по умолчанию 10)
    
        Returns:
            tuple: (список путей из cam_1, список путей из cam_2)
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

        def get_images_from_folder(folder: str, suffix: str = "_b.png", n: int = 10) -> list:
            folder_path = Path(folder)
            if not folder_path.exists():
                raise FileNotFoundError(f"Папка не найдена: {folder}")
        
            images = sorted([
                str(f) for f in folder_path.iterdir()
                if f.is_file() and f.name.endswith(suffix)
            ])
        
            return images[:n]

        cam_1_images = get_images_from_folder(cam_1_folder, n=n)
        cam_2_images = get_images_from_folder(cam_2_folder, n=n)
    
        return cam_1_images, cam_2_images

    return (get_first_images,)


@app.cell
def _(mo):
    # Ячейка 1 — UI
    cam_1_input = mo.ui.text(placeholder="путь к cam_1", label="Папка cam_1")
    cam_2_input = mo.ui.text(placeholder="путь к cam_2", label="Папка cam_2")
    run_button = mo.ui.run_button(label="Загрузить изображения")

    mo.vstack([cam_1_input, cam_2_input, run_button])
    return cam_1_input, cam_2_input, run_button


@app.cell
def _(cam_1_input, cam_2_input, get_first_images, mo, run_button):
    # Ячейка 2 — результат
    mo.stop(not run_button.value, mo.md("*Нажмите кнопку для загрузки*"))

    try:
        cam1_imgs, cam2_imgs = get_first_images(cam_1_input.value, cam_2_input.value)

        result = mo.vstack([
            mo.md(f"### cam_1 — найдено {len(cam1_imgs)} изображений"),
            mo.ui.table([{"path": p} for p in cam1_imgs]),
            mo.md(f"### cam_2 — найдено {len(cam2_imgs)} изображений"),
            mo.ui.table([{"path": p} for p in cam2_imgs]),
        ])
    except FileNotFoundError as e:
        result = mo.callout(mo.md(f"**Ошибка:** {e}"), kind="danger")

    result
    return


if __name__ == "__main__":
    app.run()
