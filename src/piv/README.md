# PIV (Particle Image Velocimetry) Module

Модуль для анализа векторных полей скоростей с использованием библиотеки OpenPIV.

## Установка зависимостей

Для работы модуля требуется установить библиотеку OpenPIV:

```bash
pip install openpiv
```

## Описание

PIV анализ вычисляет векторное поле скоростей между последовательными кадрами, используя метод кросс-корреляции. Модуль включает:

- Анализ последовательных пар изображений
- Вычисление векторов смещения методом кросс-корреляции
- Валидацию векторов (signal-to-noise ratio)
- Фильтрацию и замену невалидных векторов
- Экспорт результатов в CSV формат

## Основные параметры

### Параметры окна корреляции

- **window_size** (16, 32, 64, 128) - размер окна для кросс-корреляции
  - Меньше окно = больше разрешение, но хуже точность
  - Больше окно = лучше точность, но меньше разрешение

- **overlap** - перекрытие окон (обычно window_size // 2)
  - Увеличение overlap увеличивает разрешение результата

- **search_area_size** - размер области поиска
  - Должен быть >= window_size
  - Больше область = больше максимальное смещение, но медленнее

### Физические параметры

- **dt** - временной интервал между кадрами (мс)
  - Используется для вычисления скорости из смещения

- **scaling_factor** - масштабный коэффициент (пиксели -> мм)
  - Для преобразования результатов в физические единицы

### Параметры валидации

- **sig2noise_threshold** - порог отношения сигнал/шум
  - Типичное значение: 1.2-1.5
  - Больше порог = строже валидация

## Использование

### Простой пример

```python
from execute.execute_piv_analysis import run_piv_analysis

result = run_piv_analysis(
    input_folder="path/to/intensity_filtered_3240",
    window_size=32,
    overlap=16,
    dt=1.0,
    scaling_factor=0.05  # 0.05 мм/пиксель
)

print(f"Обработано пар: {result.total_pairs_processed}")
print(f"Векторов cam_1: {result.cam1_vectors_count}")
print(f"Векторов cam_2: {result.cam2_vectors_count}")
```

### Пример с GUI интеграцией

```python
from execute.execute_piv_analysis import PIVExecutor, PIVParameters, PIVProgress

# Создание параметров
params = PIVParameters(
    input_folder="path/to/intensity_filtered_3240",
    window_size=32,
    overlap=16,
    search_area_size=64,
    dt=1.0,
    scaling_factor=0.05,
    sig2noise_threshold=1.3,
    enable_progress_callback=True
)

# Создание исполнителя
executor = PIVExecutor()

# Установка параметров
success, error = executor.set_parameters(params)
if not success:
    print(f"Ошибка: {error}")
    exit(1)

# Callback для прогресса
def on_progress(progress: PIVProgress):
    print(f"[{progress.current_camera}] {progress.percentage:.1f}% - {progress.message}")

executor.set_progress_callback(on_progress)

# Выполнение анализа
result = executor.execute()

print(f"Успешно: {result.success}")
print(f"Выходная папка: {result.output_folder}")
```

### Предварительный просмотр

```python
# Получение предпросмотра для первой пары изображений
preview = executor.get_preview("cam_1", pair_index=0)

if preview:
    print(f"Векторов: {preview['vectors_count']}")
    print(f"Средняя магнитуда: {preview['mean_magnitude']:.3f}")
    print(f"Макс. магнитуда: {preview['max_magnitude']:.3f}")

    # Доступные данные в preview:
    # - image_a, image_b: исходные изображения
    # - x, y: координаты векторов
    # - u, v: компоненты скорости
    # - magnitude: магнитуда векторов
    # - sig2noise: отношение сигнал/шум
```

## Формат выходных данных

Модуль создает папку `PIV_XXXX` рядом с входной папкой, содержащую:

```
PIV_3240/
├── cam_1/
│   ├── piv_frame_0000_to_frame_0001.csv
│   ├── piv_frame_0001_to_frame_0002.csv
│   ├── ...
│   └── cam_1_vectors_sum.csv  # Суммарный файл
└── cam_2/
    ├── piv_frame_0000_to_frame_0001.csv
    ├── piv_frame_0001_to_frame_0002.csv
    ├── ...
    └── cam_2_vectors_sum.csv  # Суммарный файл
```

### Формат CSV файлов

```csv
X;Y;U;V;Magnitude
12.500;8.500;0.125;-0.032;0.129
37.500;8.500;0.142;-0.028;0.145
...
```

Где:
- **X, Y** - координаты центра окна (с учетом scaling_factor)
- **U, V** - компоненты скорости (с учетом dt и scaling_factor)
- **Magnitude** - магнитуда вектора скорости

## Рекомендации по выбору параметров

### Для высокоскоростных потоков

```python
PIVParameters(
    window_size=64,      # Больше окно для лучшей точности
    overlap=32,          # 50% перекрытие
    search_area_size=128,# Большая область поиска
    sig2noise_threshold=1.5  # Строже валидация
)
```

### Для малых смещений

```python
PIVParameters(
    window_size=32,      # Средний размер окна
    overlap=16,          # 50% перекрытие
    search_area_size=64, # Меньшая область поиска
    sig2noise_threshold=1.2  # Более мягкая валидация
)
```

### Для высокого разрешения

```python
PIVParameters(
    window_size=32,      # Средний размер окна
    overlap=24,          # 75% перекрытие - больше векторов
    search_area_size=64,
    sig2noise_threshold=1.3
)
```

## Отличия от PTV

### PIV (Particle Image Velocimetry)
- Анализирует **регулярную сетку** окон
- Использует **кросс-корреляцию** для определения смещения
- Вычисляет **усредненное смещение** в каждом окне
- Подходит для **густых** полей частиц
- Результат: **регулярная сетка векторов**

### PTV (Particle Tracking Velocimetry)
- Отслеживает **отдельные частицы**
- Использует **сопоставление ближайших соседей**
- Вычисляет **индивидуальное смещение** каждой частицы
- Подходит для **разреженных** полей частиц
- Результат: **нерегулярное распределение векторов**

## Troubleshooting

### ImportError: No module named 'openpiv'

Установите OpenPIV:
```bash
pip install openpiv
```

### Слишком мало валидных векторов

- Увеличьте размер окна (window_size)
- Уменьшите sig2noise_threshold
- Проверьте качество входных изображений
- Убедитесь, что частицы видны на изображениях

### Векторы имеют шум

- Увеличьте sig2noise_threshold
- Увеличьте размер окна (window_size)
- Увеличьте max_filter_iteration
- Проверьте правильность dt

### Обработка слишком медленная

- Уменьшите overlap
- Уменьшите search_area_size
- Увеличьте window_size
- Обработайте меньше кадров

## Ссылки

- [OpenPIV документация](http://openpiv.readthedocs.io/)
- [OpenPIV GitHub](https://github.com/OpenPIV/openpiv-python)
- [PIV теория](https://en.wikipedia.org/wiki/Particle_image_velocimetry)