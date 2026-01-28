"""
Запуск пороговой фильтрации с несколькими порогами.
Результаты сохраняются в папку research/threshold_comparison
с именами файлов, содержащими значение порога.
"""

import cv2
import numpy as np
from pathlib import Path


def apply_threshold_and_normalize_with_custom_name(
    img_a_path: str,
    img_b_path: str,
    threshold: int,
    output_range: tuple = (50, 255),
    output_dir: str = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Применяет пороговую фильтрацию и нормализацию к паре 16-битных изображений.
    Сохраняет с именем, включающим порог.
    """

    # === Шаг 1: Загрузка 16-битных изображений ===
    img_a = cv2.imread(img_a_path, cv2.IMREAD_UNCHANGED)
    img_b = cv2.imread(img_b_path, cv2.IMREAD_UNCHANGED)

    if img_a is None:
        raise FileNotFoundError(f"Не удалось загрузить: {img_a_path}")
    if img_b is None:
        raise FileNotFoundError(f"Не удалось загрузить: {img_b_path}")

    print(f"\n{'='*60}")
    print(f"Порог: {threshold}")
    print(f"{'='*60}")
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
    mask_a_nonzero = img_a_thresh > 0
    mask_b_nonzero = img_b_thresh > 0

    mean_a = img_a_thresh[mask_a_nonzero].mean() if mask_a_nonzero.any() else 0
    mean_b = img_b_thresh[mask_b_nonzero].mean() if mask_b_nonzero.any() else 0

    print(f"\nСредняя интенсивность A (ненулевые): {mean_a:.2f}")
    print(f"Средняя интенсивность B (ненулевые): {mean_b:.2f}")

    if mean_a > 0:
        scale_factor = mean_b / mean_a
        img_a_scaled = img_a_thresh * scale_factor * 1.5
        print(f"Коэффициент масштабирования (mean_b/mean_a * 1.5): {scale_factor * 1.5:.4f}")
    else:
        img_a_scaled = img_a_thresh.copy() * 1.5
        print("Предупреждение: средняя A = 0, масштабирование пропущено (только *1.5)")

    img_b_scaled = img_b_thresh.copy()

    # === Шаг 4: Нормализация в диапазон output_range ===
    out_min, out_max = output_range
    global_max = max(img_a_scaled.max(), img_b_scaled.max())

    print(f"\n--- Нормализация в диапазон [{out_min}, {out_max}] ---")
    print(f"Глобальный максимум: {global_max}")

    if global_max > 0:
        mask_a = img_a_scaled > 0
        img_a_norm = np.zeros_like(img_a_scaled)
        if mask_a.any():
            img_a_norm[mask_a] = out_min + (img_a_scaled[mask_a] / global_max) * (out_max - out_min)

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

    # === Сохранение результатов с именем, включающим порог ===
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Формируем имена выходных файлов с порогом
        base_name_a = Path(img_a_path).stem  # например "1_a"
        base_name_b = Path(img_b_path).stem  # например "1_b"

        name_a = f"{base_name_a}_thresh{threshold}_norm.png"
        name_b = f"{base_name_b}_thresh{threshold}_norm.png"

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
    img_a_path = r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\cam_2\2_a.png"
    img_b_path = r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\cam_2\2_b.png"

    # === Пороги для обработки ===
    THRESHOLDS = [2300, 2400, 2500, 2750, 3000, 3240]

    # === Параметры ===
    OUTPUT_RANGE = (50, 255)

    # === Директория для сохранения ===
    output_dir = Path(__file__).parent / "threshold_comparison"

    # === Запуск для каждого порога ===
    print("Запуск обработки с несколькими порогами...")
    print(f"Пороги: {THRESHOLDS}")
    print(f"Выходная директория: {output_dir}")

    for threshold in THRESHOLDS:
        apply_threshold_and_normalize_with_custom_name(
            img_a_path,
            img_b_path,
            threshold=threshold,
            output_range=OUTPUT_RANGE,
            output_dir=str(output_dir)
        )

    print("\n" + "="*60)
    print("Обработка завершена!")
    print(f"Результаты сохранены в: {output_dir}")
    print("="*60)
