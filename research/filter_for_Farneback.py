"""
Пороговая фильтрация и нормализация 16-битных изображений
1. Занулает значения интенсивности ниже порога
2. Нормализует _a относительно _b по средней интенсивности
3. Нормализует оба в диапазон 50-255
4. Сохраняет как 8-битные изображения
"""

import cv2
import numpy as np
from pathlib import Path


def apply_threshold_and_normalize(img_a_path: str, img_b_path: str,
                                   threshold: int = 3240,
                                   output_range: tuple = (50, 255),
                                   output_dir: str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Применяет пороговую фильтрацию и нормализацию к паре 16-битных изображений.

    Шаги обработки:
    1. Загрузка 16-битных изображений
    2. Зануление значений ниже порога
    3. Нормализация _a относительно _b по средней интенсивности (mean_b/mean_a)
    4. Нормализация обоих изображений в диапазон output_range
    5. Преобразование в 8-битный формат

    Parameters:
    -----------
    img_a_path : str
        Путь к первому изображению (..._a.png)
    img_b_path : str
        Путь ко второму изображению (..._b.png)
    threshold : int
        Порог интенсивности для зануления (по умолчанию 3240)
    output_range : tuple
        Диапазон нормализации (min, max), по умолчанию (50, 255)
    output_dir : str, optional
        Директория для сохранения результатов

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        Пара нормализованных 8-битных изображений (img_a_norm, img_b_norm)
    """

    # === Шаг 1: Загрузка 16-битных изображений ===
    img_a = cv2.imread(img_a_path, cv2.IMREAD_UNCHANGED)
    img_b = cv2.imread(img_b_path, cv2.IMREAD_UNCHANGED)

    if img_a is None:
        raise FileNotFoundError(f"Не удалось загрузить: {img_a_path}")
    if img_b is None:
        raise FileNotFoundError(f"Не удалось загрузить: {img_b_path}")

    print(f"Изображение A: {img_a.shape}, dtype: {img_a.dtype}")
    print(f"Изображение B: {img_b.shape}, dtype: {img_b.dtype}")
    print(f"Интенсивность A (исходная): min={img_a.min()}, max={img_a.max()}")
    print(f"Интенсивность B (исходная): min={img_b.min()}, max={img_b.max()}")

    # === Шаг 2: Зануление значений ниже порога ===
    img_a_thresh = np.where(img_a < threshold, 0, img_a).astype(np.float64)
    img_b_thresh = np.where(img_b < threshold, 0, img_b).astype(np.float64)

    pixels_zeroed_a = np.sum(img_a < threshold)
    pixels_zeroed_b = np.sum(img_b < threshold)
    total_pixels = img_a.size

    print(f"\n--- После пороговой фильтрации (порог={threshold}) ---")
    print(f"Обнулено пикселей A: {pixels_zeroed_a} ({100*pixels_zeroed_a/total_pixels:.2f}%)")
    print(f"Обнулено пикселей B: {pixels_zeroed_b} ({100*pixels_zeroed_b/total_pixels:.2f}%)")

    # === Шаг 3: Нормализация _a относительно _b по средней интенсивности ===
    # Вычисляем среднюю интенсивность только по ненулевым пикселям
    mask_a_nonzero = img_a_thresh > 0
    mask_b_nonzero = img_b_thresh > 0

    mean_a = img_a_thresh[mask_a_nonzero].mean() if mask_a_nonzero.any() else 0
    mean_b = img_b_thresh[mask_b_nonzero].mean() if mask_b_nonzero.any() else 0

    print(f"\nСредняя интенсивность A (ненулевые): {mean_a:.2f}")
    print(f"Средняя интенсивность B (ненулевые): {mean_b:.2f}")

    if mean_a > 0:
        # Вычисляем коэффициент и домножаем все пиксели A
        scale_factor = mean_b / mean_a
        img_a_scaled = img_a_thresh * scale_factor * 1.5
        print(f"Коэффициент масштабирования (mean_b/mean_a * 1.5): {scale_factor * 1.5:.4f}")
    else:
        img_a_scaled = img_a_thresh.copy() * 1.5
        print("Предупреждение: средняя A = 0, масштабирование пропущено (только *1.5)")

    img_b_scaled = img_b_thresh.copy()

    # === Шаг 4: Нормализация в диапазон output_range ===
    out_min, out_max = output_range

    # Находим общий максимум для обоих изображений (теперь он равен max_b)
    global_max = max(img_a_scaled.max(), img_b_scaled.max())

    print(f"\n--- Нормализация в диапазон [{out_min}, {out_max}] ---")
    print(f"Глобальный максимум: {global_max}")

    if global_max > 0:
        # Нормализуем: ненулевые значения переводим в [out_min, out_max]
        # Нулевые значения остаются нулевыми

        # Для A
        mask_a = img_a_scaled > 0
        img_a_norm = np.zeros_like(img_a_scaled)
        if mask_a.any():
            img_a_norm[mask_a] = out_min + (img_a_scaled[mask_a] / global_max) * (out_max - out_min)

        # Для B
        mask_b = img_b_scaled > 0
        img_b_norm = np.zeros_like(img_b_scaled)
        if mask_b.any():
            img_b_norm[mask_b] = out_min + (img_b_scaled[mask_b] / global_max) * (out_max - out_min)
    else:
        img_a_norm = img_a_scaled.copy()
        img_b_norm = img_b_scaled.copy()

    # === Шаг 5: Преобразование в 8-бит ===
    img_a_8bit = np.clip(img_a_norm, 0, 255).astype(np.uint8)
    img_b_8bit = np.clip(img_b_norm, 0, 255).astype(np.uint8)

    print(f"\nРезультат A: min={img_a_8bit[img_a_8bit > 0].min() if (img_a_8bit > 0).any() else 0}, max={img_a_8bit.max()}")
    print(f"Результат B: min={img_b_8bit[img_b_8bit > 0].min() if (img_b_8bit > 0).any() else 0}, max={img_b_8bit.max()}")

    # === Сохранение результатов ===
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Формируем имена выходных файлов
        name_a = Path(img_a_path).stem + "_norm.png"
        name_b = Path(img_b_path).stem + "_norm.png"

        cv2.imwrite(str(output_path / name_a), img_a_8bit)
        cv2.imwrite(str(output_path / name_b), img_b_8bit)

        print(f"\nСохранено: {output_path / name_a}")
        print(f"Сохранено: {output_path / name_b}")

    return img_a_8bit, img_b_8bit


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # === Пути к изображениям ===
    img_a_path = r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\cam_2\1_a.png"
    img_b_path = r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\cam_2\1_b.png"

    # === Параметры ===
    THRESHOLD = 1500
    OUTPUT_RANGE = (50, 255)

    # === Директория для сохранения ===
    output_dir = Path(__file__).parent / "normalized_8bit"

    # === Применяем обработку ===
    img_a_norm, img_b_norm = apply_threshold_and_normalize(
        img_a_path,
        img_b_path,
        threshold=THRESHOLD,
        output_range=OUTPUT_RANGE,
        output_dir=str(output_dir)
    )
