"""
Optical Flow Analysis для трекинга частиц
Методы: Lucas-Kanade (sparse) и Farneback (dense)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def lucas_kanade_flow(img1, img2, max_corners=500, quality_level=0.01, 
                       min_distance=10, win_size=21):
    """
    Lucas-Kanade Optical Flow (sparse - только на частицах)
    
    Parameters:
    -----------
    img1, img2 : np.ndarray
        Два последовательных кадра (grayscale)
    max_corners : int
        Максимальное количество точек для трекинга
    quality_level : float
        Минимальное качество угла (0-1)
    min_distance : int
        Минимальное расстояние между точками
    win_size : int
        Размер окна для вычисления потока
        
    Returns:
    --------
    dict с ключами:
        - 'points_old': координаты на кадре 1
        - 'points_new': координаты на кадре 2
        - 'displacement': смещения (dx, dy)
        - 'magnitude': магнитуда смещений
        - 'angle': углы направления (в градусах)
    """
    
    # Параметры детекции точек
    feature_params = dict(
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7
    )
    
    # Параметры Lucas-Kanade
    lk_params = dict(
        winSize=(win_size, win_size),
        maxLevel=2,  # уровни пирамиды
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
    # Находим точки на первом кадре
    p0 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)
    
    if p0 is None or len(p0) == 0:
        return None
    
    # Вычисляем optical flow
    p1, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)
    
    # Отбираем успешно отслеженные точки
    good_old = p0[status == 1]
    good_new = p1[status == 1]
    
    # Вычисляем смещения
    displacement = good_new - good_old
    magnitude = np.sqrt(displacement[:, 0]**2 + displacement[:, 1]**2)
    angle = np.arctan2(displacement[:, 1], displacement[:, 0]) * 180 / np.pi
    
    return {
        'points_old': good_old,
        'points_new': good_new,
        'displacement': displacement,
        'magnitude': magnitude,
        'angle': angle,
        'total_detected': len(p0),
        'total_tracked': len(good_old)
    }


def farneback_flow(img1, img2, pyr_scale=0.5, levels=3, winsize=15,
                   iterations=3, poly_n=5, poly_sigma=1.2):
    """
    Farneback Optical Flow (dense - для каждого пикселя)
    
    Parameters:
    -----------
    img1, img2 : np.ndarray
        Два последовательных кадра (grayscale)
    pyr_scale : float
        Масштаб пирамиды (< 1)
    levels : int
        Количество уровней пирамиды
    winsize : int
        Размер окна усреднения
    iterations : int
        Количество итераций на каждом уровне
    poly_n : int
        Размер окрестности для полиномиальной аппроксимации
    poly_sigma : float
        Стандартное отклонение гауссиана для сглаживания
        
    Returns:
    --------
    dict с ключами:
        - 'flow': поле скоростей (H, W, 2)
        - 'magnitude': магнитуда (H, W)
        - 'angle': угол направления (H, W)
    """
    
    flow = cv2.calcOpticalFlowFarneback(
        img1, img2, None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=0
    )
    
    magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    angle = np.arctan2(flow[:, :, 1], flow[:, :, 0]) * 180 / np.pi
    
    return {
        'flow': flow,
        'magnitude': magnitude,
        'angle': angle
    }


def visualize_results(img1, lk_result, fb_result, output_path=None, 
                      vector_scale=10, quiver_step=30):
    """
    Визуализация результатов обоих методов
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Lucas-Kanade: позиции частиц
    ax1 = axes[0, 0]
    ax1.imshow(img1, cmap='gray', vmin=0, vmax=100)
    ax1.scatter(lk_result['points_old'][:, 0], lk_result['points_old'][:, 1], 
                c='cyan', s=20, marker='o', label='Кадр 1')
    ax1.scatter(lk_result['points_new'][:, 0], lk_result['points_new'][:, 1], 
                c='red', s=20, marker='x', label='Кадр 2')
    ax1.set_title(f"Lucas-Kanade: {lk_result['total_tracked']} частиц")
    ax1.legend()
    ax1.axis('off')
    
    # 2. Lucas-Kanade: векторное поле
    ax2 = axes[0, 1]
    ax2.imshow(img1, cmap='gray', vmin=0, vmax=100)
    for i in range(len(lk_result['points_old'])):
        ax2.arrow(
            lk_result['points_old'][i, 0], 
            lk_result['points_old'][i, 1],
            lk_result['displacement'][i, 0] * vector_scale, 
            lk_result['displacement'][i, 1] * vector_scale,
            head_width=3, head_length=2, fc='yellow', ec='yellow', alpha=0.8
        )
    ax2.set_title(f'Lucas-Kanade: векторы (×{vector_scale})')
    ax2.axis('off')
    
    # 3. Farneback: магнитуда
    ax3 = axes[1, 0]
    im3 = ax3.imshow(fb_result['magnitude'], cmap='hot', vmin=0, vmax=5)
    plt.colorbar(im3, ax=ax3, label='Смещение (px)')
    ax3.set_title('Farneback: магнитуда смещения')
    ax3.axis('off')
    
    # 4. Farneback: векторное поле
    ax4 = axes[1, 1]
    ax4.imshow(img1, cmap='gray', vmin=0, vmax=100)
    h, w = img1.shape
    y, x = np.mgrid[quiver_step//2:h:quiver_step, quiver_step//2:w:quiver_step]
    u = fb_result['flow'][quiver_step//2::quiver_step, quiver_step//2::quiver_step, 0]
    v = fb_result['flow'][quiver_step//2::quiver_step, quiver_step//2::quiver_step, 1]
    ax4.quiver(x, y, u, v, color='yellow', scale=50, width=0.003)
    ax4.set_title('Farneback: векторное поле')
    ax4.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Сохранено: {output_path}")
    
    plt.show()
    plt.close()


def plot_histograms(lk_result, output_path=None):
    """
    Гистограммы распределения смещений и направлений
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Гистограмма смещений
    ax1 = axes[0]
    ax1.hist(lk_result['magnitude'], bins=30, color='steelblue', 
             edgecolor='black', alpha=0.7)
    mean_mag = lk_result['magnitude'].mean()
    ax1.axvline(mean_mag, color='red', linestyle='--', 
                label=f'Среднее: {mean_mag:.2f} px')
    ax1.set_xlabel('Смещение (px)')
    ax1.set_ylabel('Количество частиц')
    ax1.set_title('Распределение смещений')
    ax1.legend()
    
    # Гистограмма направлений
    ax2 = axes[1]
    ax2.hist(lk_result['angle'], bins=36, range=(-180, 180), 
             color='coral', edgecolor='black', alpha=0.7)
    mean_ang = lk_result['angle'].mean()
    ax2.axvline(mean_ang, color='red', linestyle='--', 
                label=f'Среднее: {mean_ang:.1f}°')
    ax2.set_xlabel('Угол (°)')
    ax2.set_ylabel('Количество частиц')
    ax2.set_title('Распределение направлений')
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Сохранено: {output_path}")
    
    plt.show()
    plt.close()


def export_to_csv(lk_result, output_path):
    """
    Экспорт результатов Lucas-Kanade в CSV
    """
    
    data = np.column_stack([
        lk_result['points_old'][:, 0],  # x1
        lk_result['points_old'][:, 1],  # y1
        lk_result['points_new'][:, 0],  # x2
        lk_result['points_new'][:, 1],  # y2
        lk_result['displacement'][:, 0],  # dx
        lk_result['displacement'][:, 1],  # dy
        lk_result['magnitude'],  # magnitude
        lk_result['angle']  # angle
    ])
    
    header = 'x1,y1,x2,y2,dx,dy,magnitude,angle_deg'
    np.savetxt(output_path, data, delimiter=',', header=header, comments='', fmt='%.4f')
    print(f"Экспортировано {len(data)} частиц в {output_path}")


def print_statistics(lk_result, fb_result, img1):
    """
    Вывод статистики
    """
    
    print("=" * 50)
    print("LUCAS-KANADE (sparse)")
    print("=" * 50)
    print(f"Найдено точек:     {lk_result['total_detected']}")
    print(f"Отслежено:         {lk_result['total_tracked']}")
    print(f"Смещение min:      {lk_result['magnitude'].min():.3f} px")
    print(f"Смещение max:      {lk_result['magnitude'].max():.3f} px")
    print(f"Смещение mean:     {lk_result['magnitude'].mean():.3f} px")
    print(f"Смещение std:      {lk_result['magnitude'].std():.3f} px")
    print(f"Угол mean:         {lk_result['angle'].mean():.1f}°")
    print(f"Угол std:          {lk_result['angle'].std():.1f}°")
    
    print("\n" + "=" * 50)
    print("FARNEBACK (dense)")
    print("=" * 50)
    print(f"Смещение mean (всё):     {fb_result['magnitude'].mean():.3f} px")
    print(f"Смещение max (всё):      {fb_result['magnitude'].max():.3f} px")
    
    # Статистика только на частицах
    mask = img1 > 30
    if mask.sum() > 0:
        mag_particles = fb_result['magnitude'][mask]
        print(f"Смещение mean (I>30):    {mag_particles.mean():.3f} px")
        print(f"Смещение max (I>30):     {mag_particles.max():.3f} px")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    
    # === Пути к изображениям ===
    img1_path = r"C:\Users\evils\PycharmProjects\ParticleAnalysisFE\research\threshold_comparison\2_a_thresh2300_norm.png"  # первый кадр
    img2_path = r"C:\Users\evils\PycharmProjects\ParticleAnalysisFE\research\threshold_comparison\2_b_thresh2300_norm.png"  # второй кадр
    
    # === Загрузка ===
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        raise FileNotFoundError("Не удалось загрузить изображения")
    
    print(f"Размер: {img1.shape}")
    print(f"Интенсивность img1: {img1.min()} - {img1.max()}")
    print(f"Интенсивность img2: {img2.min()} - {img2.max()}")
    
    # === Анализ Lucas-Kanade ===
    lk_result = lucas_kanade_flow(
        img1, img2,
        max_corners=5000,      # макс. количество частиц
        quality_level=0.005,   # порог качества (ниже = больше точек)
        min_distance=10,      # мин. расстояние между точками
        win_size=7           # размер окна (больше = устойчивее к шуму)
    )
    
    # === Анализ Farneback ===
    fb_result = farneback_flow(
        img1, img2,
        pyr_scale=0.5,
        levels=4,
        winsize=17,
        iterations=3,
        poly_n=5,
        poly_sigma=1.3
    )
    
    # === Статистика ===
    print_statistics(lk_result, fb_result, img1)
    
    # === Визуализация ===
    visualize_results(img1, lk_result, fb_result,
                      output_path="optical_flow_result.png",
                      vector_scale=10)
    
    plot_histograms(lk_result, output_path="optical_flow_histograms.png")
    
    # === Экспорт в CSV ===
    export_to_csv(lk_result, "optical_flow_data.csv")
