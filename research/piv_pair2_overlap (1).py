"""
PIV анализ методом центра масс интенсивности
Пара: 2_a.png, 2_b.png
Временной интервал: 2 мкс

Параметры:
- Размер ячейки: 256x256
- Перекрытие: 50%
- Отступ от краёв для расчёта: 10 пикселей
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_image(path):
    """Загрузка 16-bit изображения с сохранением динамического диапазона"""
    img = Image.open(path)
    arr = np.array(img, dtype=np.float64)

    # Если изображение цветное - конвертируем в grayscale
    if len(arr.shape) == 3:
        arr = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]

    return arr

def compute_centroid(region, margin=10):
    """
    Вычисление центра масс интенсивности в области
    с отступом margin пикселей от краёв
    
    Возвращает координаты относительно ВСЕЙ ячейки (не обрезанной)
    """
    h, w = region.shape
    
    # Обрезаем края - берём только центральную часть
    inner_region = region[margin:h-margin, margin:w-margin]
    
    # Вычитаем фон
    inner_region = inner_region - inner_region.min()
    
    total_intensity = inner_region.sum()
    if total_intensity == 0:
        # Возвращаем центр всей ячейки
        return w / 2, h / 2
    
    inner_h, inner_w = inner_region.shape
    
    # Сетка координат для внутренней области
    y_indices, x_indices = np.meshgrid(
        np.arange(inner_h),
        np.arange(inner_w),
        indexing='ij'
    )
    
    # Центр масс относительно внутренней области
    x_c_inner = np.sum(x_indices * inner_region) / total_intensity
    y_c_inner = np.sum(y_indices * inner_region) / total_intensity
    
    # Переводим в координаты всей ячейки (добавляем отступ)
    x_c = x_c_inner + margin
    y_c = y_c_inner + margin
    
    return x_c, y_c

# Параметры
cell_size = 264
overlap = 0.5  # 50% перекрытие
margin = 7    # отступ от краёв
dt = 1.0       # мкс

step = int(cell_size * (1 - overlap))  # шаг = 128 пикселей

# Загрузка изображений
img1 = load_image(r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\cam_2\1_a.png")
img2 = load_image(r"C:\Users\evils\OneDrive\Desktop\S6_DT600_WA600_16bit_cam_sorted\cam_2\1_b.png")

h, w = img1.shape

print(f"Размер изображений: {w} x {h} пикселей")
print(f"Размер ячейки: {cell_size} x {cell_size} пикселей")
print(f"Перекрытие: {int(overlap*100)}%")
print(f"Шаг сетки: {step} пикселей")
print(f"Отступ от краёв: {margin} пикселей")
print(f"Область расчёта: {cell_size - 2*margin} x {cell_size - 2*margin} пикселей")
print(f"Временной интервал: {dt} мкс")
print()

# Количество ячеек
n_cells_x = (w - cell_size) // step + 1
n_cells_y = (h - cell_size) // step + 1

print(f"Сетка ячеек: {n_cells_x} x {n_cells_y} = {n_cells_x * n_cells_y} ячеек")
print("=" * 90)

# Анализ
results = []

for j in range(n_cells_y):
    for i in range(n_cells_x):
        # Позиция ячейки с учётом перекрытия
        x_start = i * step
        y_start = j * step
        x_end = x_start + cell_size
        y_end = y_start + cell_size
        
        # Проверка границ
        if x_end > w or y_end > h:
            continue
        
        region1 = img1[y_start:y_end, x_start:x_end]
        region2 = img2[y_start:y_end, x_start:x_end]
        
        # Вычисляем центры масс с отступом
        x_c1, y_c1 = compute_centroid(region1, margin)
        x_c2, y_c2 = compute_centroid(region2, margin)
        
        dx = x_c2 - x_c1
        dy = y_c2 - y_c1
        displacement = np.sqrt(dx**2 + dy**2)
        
        # Скорость
        vx = dx / dt
        vy = dy / dt
        v_mag = displacement / dt
        
        # Глобальные координаты центра ячейки
        cell_center_x = x_start + cell_size / 2
        cell_center_y = y_start + cell_size / 2
        
        results.append({
            'i': i, 'j': j,
            'x_start': x_start, 'y_start': y_start,
            'x_center': cell_center_x,
            'y_center': cell_center_y,
            'cx1': x_c1, 'cy1': y_c1,
            'cx2': x_c2, 'cy2': y_c2,
            'dx': dx, 'dy': dy,
            'displacement': displacement,
            'vx': vx, 'vy': vy, 'v': v_mag
        })

# Вывод результатов
print(f"\n{'Ячейка':<10} {'Позиция':<14} {'dx (пикс)':<12} {'dy (пикс)':<12} {'|d| (пикс)':<12} {'|V| (пикс/мкс)':<15}")
print("-" * 80)

for r in results:
    cell = f"({r['i']},{r['j']})"
    pos = f"({r['x_start']},{r['y_start']})"
    print(f"{cell:<10} {pos:<14} {r['dx']:>+10.3f}   {r['dy']:>+10.3f}   {r['displacement']:>10.3f}   {r['v']:>12.3f}")

# Статистика
dx_arr = np.array([r['dx'] for r in results])
dy_arr = np.array([r['dy'] for r in results])
d_arr = np.array([r['displacement'] for r in results])
v_arr = np.array([r['v'] for r in results])

print("\n" + "=" * 90)
print("СТАТИСТИКА:")
print(f"  Всего ячеек:          {len(results)}")
print(f"  Среднее смещение:     dx = {np.mean(dx_arr):+.3f} пикс,  dy = {np.mean(dy_arr):+.3f} пикс")
print(f"  Среднее |d|:          {np.mean(d_arr):.3f} пикселей")
print(f"  Макс |d|:             {np.max(d_arr):.3f} пикселей")
print(f"  Мин |d|:              {np.min(d_arr):.3f} пикселей")
print(f"  Std dx:               {np.std(dx_arr):.3f}")
print(f"  Std dy:               {np.std(dy_arr):.3f}")
print()
print(f"  Средняя скорость |V|: {np.mean(v_arr):.3f} пикс/мкс")
print(f"  Макс |V|:             {np.max(v_arr):.3f} пикс/мкс")

# Визуализация (увеличенный размер для сохранения деталей)
fig, axes = plt.subplots(1, 3, figsize=(30, 10))

# Кадр A с сеткой
axes[0].imshow(img1, cmap='gray')
axes[0].set_title('Кадр A (2_a.png)', fontsize=12)
for r in results:
    # Внешняя граница ячейки
    rect = patches.Rectangle(
        (r['x_start'], r['y_start']),
        cell_size, cell_size,
        linewidth=0.5, edgecolor='yellow', facecolor='none', alpha=0.3
    )
    axes[0].add_patch(rect)
    # Внутренняя область (где считается центр масс)
    rect_inner = patches.Rectangle(
        (r['x_start'] + margin, r['y_start'] + margin),
        cell_size - 2*margin, cell_size - 2*margin,
        linewidth=0.5, edgecolor='cyan', facecolor='none', alpha=0.3
    )
    axes[0].add_patch(rect_inner)
    # Центр масс
    gx1 = r['x_start'] + r['cx1']
    gy1 = r['y_start'] + r['cy1']
    axes[0].plot(gx1, gy1, 'r+', markersize=6, mew=1)

# Кадр B с сеткой
axes[1].imshow(img2, cmap='gray')
axes[1].set_title('Кадр B (2_b.png)', fontsize=12)
for r in results:
    rect = patches.Rectangle(
        (r['x_start'], r['y_start']),
        cell_size, cell_size,
        linewidth=0.5, edgecolor='yellow', facecolor='none', alpha=0.3
    )
    axes[1].add_patch(rect)
    gx2 = r['x_start'] + r['cx2']
    gy2 = r['y_start'] + r['cy2']
    axes[1].plot(gx2, gy2, 'g+', markersize=6, mew=1)

# Векторное поле
axes[2].set_title(f'Поле смещений (Δt = {dt} мкс, overlap {int(overlap*100)}%)', fontsize=12)
axes[2].set_xlim(0, w)
axes[2].set_ylim(h, 0)
axes[2].set_aspect('equal')
axes[2].set_facecolor('black')
axes[2].set_xlabel('X (пикс)')
axes[2].set_ylabel('Y (пикс)')

# Нормализация цвета по величине смещения
colors = [r['displacement'] for r in results]
max_d = max(colors) if max(colors) > 0 else 1

for r in results:
    # Цвет зависит от величины смещения
    intensity = r['displacement'] / max_d
    color = plt.cm.plasma(intensity)
    
    axes[2].quiver(
        r['x_center'], r['y_center'],
        r['dx'], r['dy'],
        color=color, angles='xy', scale_units='xy', scale=0.1,
        width=0.004, headwidth=4
    )

# Colorbar
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, max_d))
sm.set_array([])
cbar = plt.colorbar(sm, ax=axes[2], shrink=0.8)
cbar.set_label('Смещение (пикс)')

plt.tight_layout()
plt.savefig('', dpi=200, bbox_inches='tight')
plt.close()

print("\nВизуализация сохранена: pair2_overlap.png")
