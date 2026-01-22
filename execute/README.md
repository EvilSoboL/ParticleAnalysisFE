# Execute Module

Модуль `execute` содержит готовые к использованию скрипты для выполнения операций ParticleAnalysis с параметрами, предназначенными для интеграции с GUI.

## Полный пайплайн обработки

ParticleAnalysis состоит из трех последовательных этапов:

```
1. Сортировка (execute_sorting.py)
   Входные данные: Папка с PNG изображениями
   Выходные данные: Папка {input}_cam_sorted с подпапками cam_1 и cam_2

   ↓

2. Бинаризация (execute_binarization.py)
   Входные данные: Папка {input}_cam_sorted
   Выходные данные: Папка binary_filter_{threshold}

   ↓

3. PTV Анализ (execute_ptv_analysis.py)
   Входные данные: Папка binary_filter_{threshold}
   Выходные данные: Папка PTV_{threshold} с CSV файлами

   ↓

4. Визуализация (execute_visualization.py) [опционально]
   Входные данные: {input}_cam_sorted + PTV_{threshold}
   Выходные данные: Папка PTV_{threshold}/one_to_one_visualization

   ↓

5. Векторное поле (execute_vector_field.py) [опционально]
   Входные данные: PTV_{threshold}
   Выходные данные: Папка PTV_{threshold}/vector_field
```

### Пример полного пайплайна

```python
from execute.execute_sorting import run_sorting
from execute.execute_binarization import run_binarization
from execute.execute_ptv_analysis import run_ptv_analysis
from execute.execute_visualization import run_visualization
from execute.execute_vector_field import run_vector_field

# Шаг 1: Сортировка
print("Шаг 1: Сортировка изображений...")
sort_result = run_sorting(
    input_folder="path/to/images",
    validate_format=True
)
print(f"✓ Создано пар: cam_1={sort_result.cam1_pairs}, cam_2={sort_result.cam2_pairs}")

# Шаг 2: Бинаризация
print("\nШаг 2: Бинаризация...")
bin_result = run_binarization(
    input_folder=sort_result.output_folder,
    threshold=10000
)
print(f"✓ Обработано: {bin_result.total_processed} изображений")

# Шаг 3: PTV Анализ
print("\nШаг 3: PTV Анализ...")
ptv_result = run_ptv_analysis(
    input_folder=bin_result.output_folder,
    detection_min_area=4,
    detection_max_area=150,
    matching_max_distance=30.0
)
print(f"✓ Обнаружено частиц: {ptv_result.total_particles_detected}")
print(f"✓ Сопоставлено пар: {ptv_result.total_pairs_matched}")

# Шаг 4: Визуализация (опционально)
print("\nШаг 4: Визуализация результатов...")
vis_result = run_visualization(
    original_folder=sort_result.output_folder,
    ptv_folder=ptv_result.output_folder,
    line_thickness=1
)
print(f"✓ Создано визуализаций: {vis_result.cam1_visualizations + vis_result.cam2_visualizations}")

# Шаг 5: Векторное поле (опционально)
print("\nШаг 5: Создание векторного поля...")
vf_result = run_vector_field(
    ptv_folder=ptv_result.output_folder,
    nx=73,
    ny=50,
    scale=20
)
print(f"✓ cam_1 векторов: {vf_result.cam1_vectors_count}")
print(f"✓ cam_2 векторов: {vf_result.cam2_vectors_count}")

print(f"\nРезультаты:")
print(f"  PTV анализ: {ptv_result.output_folder}")
print(f"  Визуализация: {vis_result.output_folder}")
print(f"  Векторное поле: {vf_result.output_folder}")
```

## Структура

```
execute/
├── __init__.py              # Инициализация модуля
├── execute_binarization.py  # Выполнение бинарной фильтрации
├── execute_sorting.py       # Сортировка изображений по камерам
├── execute_ptv_analysis.py  # PTV анализ (детектирование и сопоставление частиц)
├── execute_visualization.py # Визуализация PTV результатов (one-to-one matching)
├── execute_vector_field.py  # Визуализация векторного поля
└── README.md               # Эта документация
```

## execute_binarization.py

Модуль для выполнения бинарной фильтрации 16-битных изображений.

### Основные классы

#### BinarizationParameters

Dataclass с параметрами для GUI:

```python
@dataclass
class BinarizationParameters:
    input_folder: str           # Путь к папке cam_sorted
    threshold: int = 10000      # Пороговое значение (0-65535)
    enable_progress_callback: bool = True  # Callback для прогресса

    # GUI подсказки
    threshold_min: int = 0
    threshold_max: int = 65535
    threshold_default: int = 10000
    threshold_step: int = 100
```

#### BinarizationExecutor

Главный класс для выполнения бинаризации:

```python
executor = BinarizationExecutor()

# Установка параметров
params = BinarizationParameters(
    input_folder="path/to/cam_sorted",
    threshold=12000
)
success, error = executor.set_parameters(params)

# Установка callback для прогресса
def progress_callback(progress):
    print(f"{progress.percentage:.1f}% - {progress.message}")

executor.set_progress_callback(progress_callback)

# Выполнение
result = executor.execute()

# Результат
print(f"Обработано: {result.total_processed}")
print(f"Выходная папка: {result.output_folder}")
```

### Быстрый старт

Простой способ запуска без создания объектов:

```python
from execute.execute_binarization import run_binarization

result = run_binarization(
    input_folder="path/to/cam_sorted",
    threshold=12000
)

if result.success:
    print(f"Успешно обработано {result.total_processed} файлов")
else:
    print(f"Ошибки: {result.errors}")
```

### Получение статистики для выбора порога

```python
executor = BinarizationExecutor()
executor.set_parameters(params)

# Получить статистику для рекомендации порога
stats = executor.get_image_statistics("cam_1", sample_size=5)

if stats:
    print(f"Рекомендуемый порог (медиана): {stats['recommended_threshold_median']}")
    print(f"Диапазон: [{stats['recommended_threshold_low']}, {stats['recommended_threshold_high']}]")
```

### Интеграция с GUI

#### Пример для PyQt/PySide

```python
from PyQt5.QtWidgets import QProgressBar, QLabel
from execute.execute_binarization import BinarizationExecutor, BinarizationParameters

class BinarizationWidget:
    def __init__(self):
        self.executor = BinarizationExecutor()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel()

    def on_start_button_clicked(self):
        # Получить параметры из GUI
        params = BinarizationParameters(
            input_folder=self.folder_input.text(),
            threshold=self.threshold_slider.value()
        )

        # Установить параметры
        success, error = self.executor.set_parameters(params)
        if not success:
            self.show_error(error)
            return

        # Установить callback
        self.executor.set_progress_callback(self.update_progress)

        # Запустить в отдельном потоке
        self.worker = WorkerThread(self.executor)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def update_progress(self, progress):
        self.progress_bar.setValue(int(progress.percentage))
        self.status_label.setText(progress.message)

    def on_cancel_button_clicked(self):
        self.executor.cancel()
```

#### Пример для Tkinter

```python
import tkinter as tk
from tkinter import ttk
from execute.execute_binarization import BinarizationExecutor, BinarizationParameters

class BinarizationWindow:
    def __init__(self, root):
        self.root = root
        self.executor = BinarizationExecutor()

        # GUI элементы
        self.progress_bar = ttk.Progressbar(root, length=300)
        self.status_label = tk.Label(root, text="")

        # Кнопки
        self.start_button = tk.Button(root, text="Начать", command=self.start)
        self.cancel_button = tk.Button(root, text="Отмена", command=self.cancel)

    def start(self):
        params = BinarizationParameters(
            input_folder=self.folder_var.get(),
            threshold=self.threshold_var.get()
        )

        success, error = self.executor.set_parameters(params)
        if not success:
            tk.messagebox.showerror("Ошибка", error)
            return

        self.executor.set_progress_callback(self.update_progress)

        # Запуск в потоке
        import threading
        thread = threading.Thread(target=self.run_binarization)
        thread.start()

    def update_progress(self, progress):
        self.root.after(0, lambda: self.progress_bar['value'] = progress.percentage)
        self.root.after(0, lambda: self.status_label.config(text=progress.message))

    def run_binarization(self):
        result = self.executor.execute()
        self.root.after(0, lambda: self.on_finished(result))
```

### Запуск примера

```bash
cd execute
python execute_binarization.py
```

Это запустит пример использования с демонстрацией всех возможностей.

## execute_sorting.py

Модуль для сортировки исходных 16-битных изображений по камерам.

### Алгоритм

Изображения сортируются по следующему принципу:
- Первые 2 фото → cam_1 (с отражением по горизонтали)
- Следующие 2 фото → cam_2 (без изменений)
- Цикл повторяется

Все изображения переименовываются в формат `N_суффикс.png`, где N - номер пары.

### Основные классы

#### SortingParameters

Dataclass с параметрами для GUI:

```python
@dataclass
class SortingParameters:
    input_folder: str               # Путь к папке с PNG изображениями
    validate_format: bool = True    # Валидация формата (16-bit PNG)
    enable_progress_callback: bool = True  # Callback для прогресса
```

#### SortingExecutor

Главный класс для выполнения сортировки:

```python
executor = SortingExecutor()

# Установка параметров
params = SortingParameters(
    input_folder="path/to/images",
    validate_format=True
)
success, error = executor.set_parameters(params)

# Установка callback для прогресса
def progress_callback(progress):
    print(f"{progress.percentage:.1f}% - {progress.message}")

executor.set_progress_callback(progress_callback)

# Выполнение
result = executor.execute()

# Результат
print(f"cam_1: {result.cam1_pairs} пар")
print(f"cam_2: {result.cam2_pairs} пар")
print(f"Выходная папка: {result.output_folder}")
```

### Быстрый старт

Простой способ запуска без создания объектов:

```python
from execute.execute_sorting import run_sorting

result = run_sorting(
    input_folder="path/to/images",
    validate_format=True
)

if result.success:
    print(f"Создано пар: cam_1={result.cam1_pairs}, cam_2={result.cam2_pairs}")
else:
    print(f"Ошибки: {result.errors}")
```

### Получение статистики до сортировки

```python
executor = SortingExecutor()
executor.set_parameters(params)

# Получить информацию о входных данных
stats = executor.get_input_statistics()

if stats:
    print(f"Всего файлов: {stats['total_files']}")
    print(f"Ожидаемо пар в cam_1: {stats['expected_cam1_pairs']}")
    print(f"Ожидаемо пар в cam_2: {stats['expected_cam2_pairs']}")
```

### Интеграция с GUI

#### Пример для PyQt/PySide

```python
from PyQt5.QtWidgets import QProgressBar, QLabel
from execute.execute_sorting import SortingExecutor, SortingParameters

class SortingWidget:
    def __init__(self):
        self.executor = SortingExecutor()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel()

    def on_start_button_clicked(self):
        # Получить параметры из GUI
        params = SortingParameters(
            input_folder=self.folder_input.text(),
            validate_format=self.validate_checkbox.isChecked()
        )

        # Установить параметры
        success, error = self.executor.set_parameters(params)
        if not success:
            self.show_error(error)
            return

        # Показать ожидаемые результаты
        stats = self.executor.get_input_statistics()
        self.info_label.setText(
            f"Будет создано {stats['expected_cam1_pairs']} пар в cam_1 "
            f"и {stats['expected_cam2_pairs']} пар в cam_2"
        )

        # Установить callback
        self.executor.set_progress_callback(self.update_progress)

        # Запустить в отдельном потоке
        self.worker = WorkerThread(self.executor)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def update_progress(self, progress):
        self.progress_bar.setValue(int(progress.percentage))
        self.status_label.setText(f"[{progress.current_camera}] {progress.message}")

    def on_cancel_button_clicked(self):
        self.executor.cancel()
```

#### Пример для Tkinter

```python
import tkinter as tk
from tkinter import ttk
from execute.execute_sorting import SortingExecutor, SortingParameters

class SortingWindow:
    def __init__(self, root):
        self.root = root
        self.executor = SortingExecutor()

        # GUI элементы
        self.progress_bar = ttk.Progressbar(root, length=300)
        self.status_label = tk.Label(root, text="")
        self.validate_var = tk.BooleanVar(value=True)
        self.validate_checkbox = tk.Checkbutton(
            root, text="Валидировать формат", variable=self.validate_var
        )

        # Кнопки
        self.start_button = tk.Button(root, text="Начать", command=self.start)
        self.cancel_button = tk.Button(root, text="Отмена", command=self.cancel)

    def start(self):
        params = SortingParameters(
            input_folder=self.folder_var.get(),
            validate_format=self.validate_var.get()
        )

        success, error = self.executor.set_parameters(params)
        if not success:
            tk.messagebox.showerror("Ошибка", error)
            return

        # Показать статистику
        stats = self.executor.get_input_statistics()
        tk.messagebox.showinfo(
            "Информация",
            f"Будет обработано {stats['total_files']} файлов\n"
            f"cam_1: {stats['expected_cam1_pairs']} пар\n"
            f"cam_2: {stats['expected_cam2_pairs']} пар"
        )

        self.executor.set_progress_callback(self.update_progress)

        # Запуск в потоке
        import threading
        thread = threading.Thread(target=self.run_sorting)
        thread.start()

    def update_progress(self, progress):
        self.root.after(0, lambda: self.progress_bar['value'] = progress.percentage)
        self.root.after(0, lambda: self.status_label.config(text=progress.message))

    def run_sorting(self):
        result = self.executor.execute()
        self.root.after(0, lambda: self.on_finished(result))
```

### Запуск примера

```bash
cd execute
python execute_sorting.py
```

Это запустит пример использования с демонстрацией всех возможностей.

## execute_ptv_analysis.py

Модуль для выполнения PTV (Particle Tracking Velocimetry) анализа.

### Что делает модуль

PTV анализ включает две основные стадии:
1. **Детектирование** - поиск частиц на бинаризованных изображениях
2. **Сопоставление** - нахождение соответствий между частицами на последовательных кадрах

### Основные классы

#### PTVParameters

Dataclass с параметрами для GUI:

```python
@dataclass
class PTVParameters:
    input_folder: str               # Путь к папке binary_filter_XXXX

    # Параметры детектирования
    detection_min_area: int = 4     # Минимальная площадь частицы (пикс.)
    detection_max_area: int = 150   # Максимальная площадь частицы (пикс.)

    # Параметры сопоставления
    matching_max_distance: float = 30.0         # Максимальный радиус поиска (пикс.)
    matching_max_diameter_diff: float = 2.0     # Макс. разница диаметров (пикс.)

    enable_progress_callback: bool = True  # Callback для прогресса
```

#### PTVExecutor

Главный класс для выполнения PTV анализа:

```python
executor = PTVExecutor()

# Установка параметров
params = PTVParameters(
    input_folder="path/to/binary_filter_10000",
    detection_min_area=4,
    detection_max_area=150,
    matching_max_distance=30.0,
    matching_max_diameter_diff=2.0
)
success, error = executor.set_parameters(params)

# Установка callback для прогресса
def progress_callback(progress):
    print(f"[{progress.current_camera}] [{progress.current_stage}] {progress.percentage:.1f}%")

executor.set_progress_callback(progress_callback)

# Выполнение
result = executor.execute()

# Результат
print(f"Обнаружено частиц: {result.total_particles_detected}")
print(f"Сопоставлено пар: {result.total_pairs_matched}")
print(f"cam_1: {result.cam1_pairs_count} пар")
print(f"cam_2: {result.cam2_pairs_count} пар")
print(f"Выходная папка: {result.output_folder}")
```

### Быстрый старт

Простой способ запуска без создания объектов:

```python
from execute.execute_ptv_analysis import run_ptv_analysis

result = run_ptv_analysis(
    input_folder="path/to/binary_filter_10000",
    detection_min_area=4,
    detection_max_area=150,
    matching_max_distance=30.0,
    matching_max_diameter_diff=2.0
)

if result.success:
    print(f"Обнаружено {result.total_particles_detected} частиц")
    print(f"Сопоставлено {result.total_pairs_matched} пар")
else:
    print(f"Ошибки: {result.errors}")
```

### Предварительный просмотр

#### Просмотр детектирования для одного изображения

```python
executor = PTVExecutor()
executor.set_parameters(params)

# Получить результаты детектирования для просмотра
preview = executor.get_detection_preview("path/to/image.png")

if preview:
    print(f"Обнаружено частиц: {preview['particles_count']}")
    print(f"Параметры: min_area={preview['min_area']}, max_area={preview['max_area']}")
```

#### Просмотр сопоставления для пары изображений

```python
executor = PTVExecutor()
executor.set_parameters(params)

# Получить результаты сопоставления для пары
preview = executor.get_matching_preview("path/to/1_a.png", "path/to/1_b.png")

if preview:
    print(f"Частиц в кадре A: {preview['particles_a_count']}")
    print(f"Частиц в кадре B: {preview['particles_b_count']}")
    print(f"Сопоставлено пар: {preview['matched_count']}")
    print(f"Несопоставленных в A: {preview['unmatched_a_count']}")
    print(f"Несопоставленных в B: {preview['unmatched_b_count']}")
```

### Интеграция с GUI

#### Пример для PyQt/PySide

```python
from PyQt5.QtWidgets import QProgressBar, QLabel, QSpinBox, QDoubleSpinBox
from execute.execute_ptv_analysis import PTVExecutor, PTVParameters

class PTVWidget:
    def __init__(self):
        self.executor = PTVExecutor()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel()

        # GUI элементы для параметров
        self.min_area_spinbox = QSpinBox()
        self.min_area_spinbox.setRange(1, 1000)
        self.min_area_spinbox.setValue(4)

        self.max_area_spinbox = QSpinBox()
        self.max_area_spinbox.setRange(1, 1000)
        self.max_area_spinbox.setValue(150)

        self.max_distance_spinbox = QDoubleSpinBox()
        self.max_distance_spinbox.setRange(1.0, 100.0)
        self.max_distance_spinbox.setValue(30.0)

        self.max_diameter_diff_spinbox = QDoubleSpinBox()
        self.max_diameter_diff_spinbox.setRange(0.0, 10.0)
        self.max_diameter_diff_spinbox.setValue(2.0)

    def on_start_button_clicked(self):
        # Получить параметры из GUI
        params = PTVParameters(
            input_folder=self.folder_input.text(),
            detection_min_area=self.min_area_spinbox.value(),
            detection_max_area=self.max_area_spinbox.value(),
            matching_max_distance=self.max_distance_spinbox.value(),
            matching_max_diameter_diff=self.max_diameter_diff_spinbox.value()
        )

        # Установить параметры
        success, error = self.executor.set_parameters(params)
        if not success:
            self.show_error(error)
            return

        # Установить callback
        self.executor.set_progress_callback(self.update_progress)

        # Запустить в отдельном потоке
        self.worker = WorkerThread(self.executor)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def update_progress(self, progress):
        self.progress_bar.setValue(int(progress.percentage))
        self.status_label.setText(
            f"[{progress.current_camera}] [{progress.current_stage}] {progress.message}"
        )

    def on_preview_button_clicked(self):
        # Предварительный просмотр детектирования
        preview = self.executor.get_detection_preview(self.selected_image_path)
        if preview:
            self.show_detection_preview(preview)

    def on_cancel_button_clicked(self):
        self.executor.cancel()
```

#### Пример для Tkinter

```python
import tkinter as tk
from tkinter import ttk
from execute.execute_ptv_analysis import PTVExecutor, PTVParameters

class PTVWindow:
    def __init__(self, root):
        self.root = root
        self.executor = PTVExecutor()

        # GUI элементы
        self.progress_bar = ttk.Progressbar(root, length=300)
        self.status_label = tk.Label(root, text="")

        # Параметры детектирования
        self.min_area_var = tk.IntVar(value=4)
        self.max_area_var = tk.IntVar(value=150)

        # Параметры сопоставления
        self.max_distance_var = tk.DoubleVar(value=30.0)
        self.max_diameter_diff_var = tk.DoubleVar(value=2.0)

        # Spinboxes
        tk.Label(root, text="Детектирование:").pack()

        tk.Label(root, text="Мин. площадь (пикс.):").pack()
        tk.Spinbox(root, from_=1, to=1000, textvariable=self.min_area_var).pack()

        tk.Label(root, text="Макс. площадь (пикс.):").pack()
        tk.Spinbox(root, from_=1, to=1000, textvariable=self.max_area_var).pack()

        tk.Label(root, text="Сопоставление:").pack()

        tk.Label(root, text="Макс. расстояние (пикс.):").pack()
        tk.Spinbox(root, from_=1.0, to=100.0, increment=1.0,
                  textvariable=self.max_distance_var).pack()

        tk.Label(root, text="Макс. разница диаметров (пикс.):").pack()
        tk.Spinbox(root, from_=0.0, to=10.0, increment=0.1,
                  textvariable=self.max_diameter_diff_var).pack()

        # Кнопки
        self.start_button = tk.Button(root, text="Начать", command=self.start)
        self.start_button.pack()
        self.cancel_button = tk.Button(root, text="Отмена", command=self.cancel)
        self.cancel_button.pack()

    def start(self):
        params = PTVParameters(
            input_folder=self.folder_var.get(),
            detection_min_area=self.min_area_var.get(),
            detection_max_area=self.max_area_var.get(),
            matching_max_distance=self.max_distance_var.get(),
            matching_max_diameter_diff=self.max_diameter_diff_var.get()
        )

        success, error = self.executor.set_parameters(params)
        if not success:
            tk.messagebox.showerror("Ошибка", error)
            return

        self.executor.set_progress_callback(self.update_progress)

        # Запуск в потоке
        import threading
        thread = threading.Thread(target=self.run_ptv)
        thread.start()

    def update_progress(self, progress):
        self.root.after(0, lambda: self.progress_bar['value'] = progress.percentage)
        self.root.after(0, lambda: self.status_label.config(
            text=f"[{progress.current_camera}] [{progress.current_stage}] {progress.message}"
        ))

    def run_ptv(self):
        result = self.executor.execute()
        self.root.after(0, lambda: self.on_finished(result))
```

### Запуск примера

```bash
cd execute
python execute_ptv_analysis.py
```

Это запустит пример использования с демонстрацией всех возможностей.

## execute_visualization.py

Модуль для визуализации результатов PTV анализа (one-to-one matching).

### Что делает модуль

Визуализация накладывает результаты сопоставления на исходные изображения:
- **Зелёные окружности** - частицы в кадре A (начальная позиция)
- **Красные окружности** - частицы в кадре B (конечная позиция)
- **Оранжевые линии** - связи между сопоставленными частицами

### Основные классы

#### VisualizationParameters

Dataclass с параметрами для GUI:

```python
@dataclass
class VisualizationParameters:
    original_folder: str        # Путь к папке cam_sorted с исходными изображениями
    ptv_folder: str            # Путь к папке PTV_XXXX с результатами

    # Параметры визуализации (цвета в BGR)
    particle_a_color: Tuple[int, int, int] = (0, 255, 0)    # Зелёный (кадр A)
    particle_b_color: Tuple[int, int, int] = (0, 0, 255)    # Красный (кадр B)
    line_color: Tuple[int, int, int] = (0, 165, 255)        # Оранжевый (связи)
    line_thickness: int = 1                                  # Толщина линий

    enable_progress_callback: bool = True  # Callback для прогресса
```

#### VisualizationExecutor

Главный класс для выполнения визуализации:

```python
executor = VisualizationExecutor()

# Установка параметров
params = VisualizationParameters(
    original_folder="path/to/cam_sorted",
    ptv_folder="path/to/PTV_10000",
    particle_a_color=(0, 255, 0),  # Зелёный (BGR)
    particle_b_color=(0, 0, 255),  # Красный (BGR)
    line_color=(0, 165, 255),      # Оранжевый (BGR)
    line_thickness=1
)
success, error = executor.set_parameters(params)

# Установка callback для прогресса
def progress_callback(progress):
    print(f"[{progress.current_camera}] {progress.percentage:.1f}%")

executor.set_progress_callback(progress_callback)

# Выполнение
result = executor.execute()

# Результат
print(f"Обработано пар: {result.total_pairs_processed}")
print(f"Создано визуализаций:")
print(f"  cam_1: {result.cam1_visualizations}")
print(f"  cam_2: {result.cam2_visualizations}")
print(f"Выходная папка: {result.output_folder}")
```

### Быстрый старт

Простой способ запуска без создания объектов:

```python
from execute.execute_visualization import run_visualization

result = run_visualization(
    original_folder="path/to/cam_sorted",
    ptv_folder="path/to/PTV_10000",
    line_thickness=2
)

if result.success:
    total_vis = result.cam1_visualizations + result.cam2_visualizations
    print(f"Создано {total_vis} визуализаций")
else:
    print(f"Ошибки: {result.errors}")
```

### Предварительный просмотр

#### Просмотр визуализации для одной пары

```python
executor = VisualizationExecutor()
executor.set_parameters(params)

# Получить preview для одной пары
preview = executor.get_preview("cam_1", pair_number=1)

if preview:
    print(f"Камера: {preview['camera']}")
    print(f"Номер пары: {preview['pair_number']}")
    # preview['vis_a'] - визуализация кадра A (numpy array)
    # preview['vis_b'] - визуализация кадра B (numpy array)
```

#### Получение статистики для пары

```python
executor = VisualizationExecutor()
executor.set_parameters(params)

# Получить статистику для пары
stats = executor.get_pair_statistics("cam_1", pair_number=1)

if stats:
    print(f"Сопоставлено пар: {stats['pairs_count']}")
    print(f"Средний диаметр: {stats['mean_diameter']:.2f} пикс.")
    print(f"Среднее смещение: {stats['mean_displacement']:.2f} пикс.")
    print(f"Среднее dx: {stats['mean_dx']:.2f} пикс.")
    print(f"Среднее dy: {stats['mean_dy']:.2f} пикс.")
```

### Настройка цветов

Цвета задаются в формате BGR (Blue, Green, Red) со значениями 0-255:

```python
# Примеры цветов (BGR)
GREEN = (0, 255, 0)      # Зелёный
RED = (0, 0, 255)        # Красный
BLUE = (255, 0, 0)       # Синий
YELLOW = (0, 255, 255)   # Жёлтый
CYAN = (255, 255, 0)     # Циановый
MAGENTA = (255, 0, 255)  # Пурпурный
ORANGE = (0, 165, 255)   # Оранжевый
WHITE = (255, 255, 255)  # Белый
BLACK = (0, 0, 0)        # Чёрный

params = VisualizationParameters(
    original_folder="path/to/cam_sorted",
    ptv_folder="path/to/PTV_10000",
    particle_a_color=CYAN,     # Циановые частицы A
    particle_b_color=MAGENTA,  # Пурпурные частицы B
    line_color=YELLOW,         # Жёлтые линии связи
    line_thickness=2
)
```

### Интеграция с GUI

#### Пример для PyQt/PySide с выбором цветов

```python
from PyQt5.QtWidgets import (QProgressBar, QLabel, QSpinBox,
                             QColorDialog, QPushButton)
from PyQt5.QtGui import QColor
from execute.execute_visualization import VisualizationExecutor, VisualizationParameters

class VisualizationWidget:
    def __init__(self):
        self.executor = VisualizationExecutor()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel()

        # GUI элементы для выбора цветов
        self.particle_a_color_button = QPushButton("Цвет частиц A")
        self.particle_a_color_button.clicked.connect(self.choose_particle_a_color)
        self.particle_a_color = (0, 255, 0)  # Зелёный по умолчанию

        self.particle_b_color_button = QPushButton("Цвет частиц B")
        self.particle_b_color_button.clicked.connect(self.choose_particle_b_color)
        self.particle_b_color = (0, 0, 255)  # Красный по умолчанию

        self.line_color_button = QPushButton("Цвет линий")
        self.line_color_button.clicked.connect(self.choose_line_color)
        self.line_color = (0, 165, 255)  # Оранжевый по умолчанию

        # Толщина линий
        self.line_thickness_spinbox = QSpinBox()
        self.line_thickness_spinbox.setRange(1, 5)
        self.line_thickness_spinbox.setValue(1)

    def choose_particle_a_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            # Qt использует RGB, OpenCV использует BGR
            self.particle_a_color = (color.blue(), color.green(), color.red())
            self.particle_a_color_button.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()})"
            )

    def choose_particle_b_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.particle_b_color = (color.blue(), color.green(), color.red())
            self.particle_b_color_button.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()})"
            )

    def choose_line_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.line_color = (color.blue(), color.green(), color.red())
            self.line_color_button.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()})"
            )

    def on_start_button_clicked(self):
        # Получить параметры из GUI
        params = VisualizationParameters(
            original_folder=self.original_folder_input.text(),
            ptv_folder=self.ptv_folder_input.text(),
            particle_a_color=self.particle_a_color,
            particle_b_color=self.particle_b_color,
            line_color=self.line_color,
            line_thickness=self.line_thickness_spinbox.value()
        )

        # Установить параметры
        success, error = self.executor.set_parameters(params)
        if not success:
            self.show_error(error)
            return

        # Установить callback
        self.executor.set_progress_callback(self.update_progress)

        # Запустить в отдельном потоке
        self.worker = WorkerThread(self.executor)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def update_progress(self, progress):
        self.progress_bar.setValue(int(progress.percentage))
        self.status_label.setText(f"[{progress.current_camera}] {progress.message}")

    def on_preview_button_clicked(self):
        # Предварительный просмотр
        preview = self.executor.get_preview("cam_1", pair_number=1)
        if preview:
            self.show_preview(preview)

    def on_cancel_button_clicked(self):
        self.executor.cancel()
```

### Запуск примера

```bash
cd execute
python execute_visualization.py
```

Это запустит пример использования с демонстрацией всех возможностей.

## execute_vector_field.py

Модуль для визуализации векторного поля смещений частиц с использованием matplotlib quiver.

### Что делает модуль

Векторное поле создается на основе суммарных CSV файлов (cam_X_pairs_sum.csv):
- Векторы усредняются по ячейкам сетки (binning)
- Используется matplotlib quiver для отрисовки стрелок
- Цветовая карта показывает длину векторов
- Добавляется сетка и colorbar для удобства анализа

### Основные классы

#### VectorFieldParameters

Dataclass с параметрами для GUI:

```python
@dataclass
class VectorFieldParameters:
    ptv_folder: str             # Путь к папке PTV_XXXX

    # Параметры сетки
    nx: int = 73                # Количество ячеек по X
    ny: int = 50                # Количество ячеек по Y

    # Параметры quiver
    scale: float = 20           # Масштаб стрелок (меньше = длиннее)
    width: float = 0.005        # Толщина стрелок

    # Параметры цветовой карты
    cmap: str = "jet"           # Название цветовой карты matplotlib
    vmin: Optional[float] = None  # Минимум для colorbar (None = авто)
    vmax: Optional[float] = None  # Максимум для colorbar (None = авто)

    # Параметры сетки
    show_grid: bool = True      # Показывать сетку
    grid_color: str = "black"   # Цвет линий сетки
    grid_alpha: float = 0.25    # Прозрачность сетки (0-1)
    grid_linewidth: float = 0.4 # Толщина линий сетки

    # Параметры осей
    xlabel: str = "r, mm"       # Подпись оси X
    ylabel: str = "z, mm"       # Подпись оси Y
    figsize: Tuple[float, float] = (9, 6)  # Размер фигуры (дюймы)

    # Доступные цветовые карты
    available_cmaps: Tuple[str, ...] = (
        "jet", "viridis", "plasma", "inferno", "magma", "cividis",
        "twilight", "turbo", "hot", "cool", "spring", "summer",
        "autumn", "winter", "RdYlBu", "RdYlGn", "Spectral"
    )
```

#### VectorFieldExecutor

Главный класс для выполнения визуализации векторного поля:

```python
executor = VectorFieldExecutor()

# Установка параметров
params = VectorFieldParameters(
    ptv_folder="path/to/PTV_10000",
    nx=73,
    ny=50,
    scale=20,
    width=0.005,
    cmap="jet",
    show_grid=True
)
success, error = executor.set_parameters(params)

# Установка callback для прогресса (зарезервировано)
def progress_callback(progress):
    print(f"Прогресс: {progress}")

executor.set_progress_callback(progress_callback)

# Выполнение
result = executor.execute()

# Результат
print(f"cam_1 векторов: {result.cam1_vectors_count}")
print(f"cam_2 векторов: {result.cam2_vectors_count}")
print(f"Выходная папка: {result.output_folder}")
```

### Быстрый старт

Простой способ запуска без создания объектов:

```python
from execute.execute_vector_field import run_vector_field

result = run_vector_field(
    ptv_folder="path/to/PTV_10000",
    nx=73,
    ny=50,
    scale=20,
    width=0.005,
    cmap="jet"
)

if result.success:
    print(f"cam_1: {result.cam1_vectors_count} векторов")
    print(f"cam_2: {result.cam2_vectors_count} векторов")
else:
    print(f"Ошибки: {result.errors}")
```

### Предварительный просмотр

#### Просмотр векторного поля для одной камеры

```python
executor = VectorFieldExecutor()
executor.set_parameters(params)

# Получить preview для одной камеры
preview_image = executor.get_preview("cam_1")

if preview_image is not None:
    # preview_image - numpy array (BGR изображение)
    import cv2
    cv2.imshow("Preview cam_1", preview_image)
    cv2.waitKey(0)
```

#### Получение статистики векторов

```python
executor = VectorFieldExecutor()
executor.set_parameters(params)

# Получить статистику для камеры
stats = executor.get_statistics("cam_1")

if stats:
    print(f"Всего векторов: {stats['vectors_count']}")
    print(f"Векторов с смещением: {stats['vectors_with_displacement']}")
    print(f"Средняя длина: {stats['mean_length']:.2f}")
    print(f"Макс. длина: {stats['max_length']:.2f}")
    print(f"Мин. длина: {stats['min_length']:.2f}")
    print(f"Среднее dx: {stats['mean_dx']:.2f}")
    print(f"Среднее dy: {stats['mean_dy']:.2f}")
    print(f"Стд. откл. dx: {stats['std_dx']:.2f}")
    print(f"Стд. откл. dy: {stats['std_dy']:.2f}")
```

### Настройка цветовых карт

Доступные цветовые карты для визуализации:

```python
# Последовательные карты (хорошо для скалярных данных)
"jet"       # Классическая радужная (синий-зелёный-жёлтый-красный)
"viridis"   # Перцептивно-линейная (тёмно-синий-зелёный-жёлтый)
"plasma"    # Перцептивно-линейная (фиолетовый-розовый-жёлтый)
"inferno"   # Перцептивно-линейная (чёрный-красный-жёлтый)
"magma"     # Перцептивно-линейная (чёрный-фиолетовый-белый)
"cividis"   # Оптимизирована для дальтоников

# Циклические карты
"twilight"  # Симметричная циклическая (синий-белый-красный-чёрный)

# Температурные карты
"turbo"     # Улучшенная радужная карта
"hot"       # Чёрный-красный-жёлтый-белый
"cool"      # Циановый-пурпурный

# Сезонные карты
"spring"    # Пурпурный-жёлтый
"summer"    # Зелёный-жёлтый
"autumn"    # Красный-жёлтый
"winter"    # Синий-зелёный

# Расходящиеся карты (хорошо для данных с центральным значением)
"RdYlBu"    # Красный-жёлтый-синий
"RdYlGn"    # Красный-жёлтый-зелёный
"Spectral"  # Спектральная расходящаяся

# Пример использования
params = VectorFieldParameters(
    ptv_folder="path/to/PTV_10000",
    nx=73,
    ny=50,
    cmap="viridis",    # Используем viridis вместо jet
    vmin=0,            # Минимум colorbar
    vmax=10            # Максимум colorbar
)
```

### Интеграция с GUI

#### Пример для PyQt/PySide

```python
from PyQt5.QtWidgets import (QProgressBar, QLabel, QSpinBox,
                             QDoubleSpinBox, QComboBox)
from execute.execute_vector_field import VectorFieldExecutor, VectorFieldParameters

class VectorFieldWidget:
    def __init__(self):
        self.executor = VectorFieldExecutor()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel()

        # GUI элементы для параметров сетки
        self.nx_spinbox = QSpinBox()
        self.nx_spinbox.setRange(10, 200)
        self.nx_spinbox.setValue(73)

        self.ny_spinbox = QSpinBox()
        self.ny_spinbox.setRange(10, 200)
        self.ny_spinbox.setValue(50)

        # GUI элементы для параметров quiver
        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setRange(1.0, 200.0)
        self.scale_spinbox.setValue(20.0)

        self.width_spinbox = QDoubleSpinBox()
        self.width_spinbox.setRange(0.001, 0.02)
        self.width_spinbox.setSingleStep(0.001)
        self.width_spinbox.setValue(0.005)

        # Выбор цветовой карты
        self.cmap_combobox = QComboBox()
        cmaps = ["jet", "viridis", "plasma", "inferno", "magma", "cividis",
                 "twilight", "turbo", "hot", "cool", "spring", "summer",
                 "autumn", "winter", "RdYlBu", "RdYlGn", "Spectral"]
        self.cmap_combobox.addItems(cmaps)

    def on_start_button_clicked(self):
        # Получить параметры из GUI
        params = VectorFieldParameters(
            ptv_folder=self.ptv_folder_input.text(),
            nx=self.nx_spinbox.value(),
            ny=self.ny_spinbox.value(),
            scale=self.scale_spinbox.value(),
            width=self.width_spinbox.value(),
            cmap=self.cmap_combobox.currentText(),
            show_grid=self.show_grid_checkbox.isChecked()
        )

        # Установить параметры
        success, error = self.executor.set_parameters(params)
        if not success:
            self.show_error(error)
            return

        # Показать статистику перед выполнением
        self.show_statistics()

        # Установить callback
        self.executor.set_progress_callback(self.update_progress)

        # Запустить в отдельном потоке
        self.worker = WorkerThread(self.executor)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def show_statistics(self):
        """Показать статистику векторов перед выполнением."""
        cam1_stats = self.executor.get_statistics("cam_1")
        cam2_stats = self.executor.get_statistics("cam_2")

        info_text = ""
        if cam1_stats:
            info_text += f"cam_1: {cam1_stats['vectors_count']} векторов\n"
            if cam1_stats.get('mean_length'):
                info_text += f"  Средняя длина: {cam1_stats['mean_length']:.2f}\n"

        if cam2_stats:
            info_text += f"cam_2: {cam2_stats['vectors_count']} векторов\n"
            if cam2_stats.get('mean_length'):
                info_text += f"  Средняя длина: {cam2_stats['mean_length']:.2f}\n"

        self.info_label.setText(info_text)

    def on_preview_button_clicked(self):
        """Показать предпросмотр векторного поля."""
        camera = self.camera_combobox.currentText()  # cam_1 или cam_2
        preview = self.executor.get_preview(camera)
        if preview is not None:
            # Показать preview (numpy array BGR)
            self.show_preview_image(preview)

    def update_progress(self, progress):
        # В текущей версии не используется
        pass

    def on_cancel_button_clicked(self):
        self.executor.cancel()
```

#### Пример для Tkinter

```python
import tkinter as tk
from tkinter import ttk
from execute.execute_vector_field import VectorFieldExecutor, VectorFieldParameters

class VectorFieldWindow:
    def __init__(self, root):
        self.root = root
        self.executor = VectorFieldExecutor()

        # GUI элементы
        self.progress_bar = ttk.Progressbar(root, length=300)
        self.status_label = tk.Label(root, text="")

        # Параметры сетки
        tk.Label(root, text="Разрешение сетки:").pack()

        tk.Label(root, text="nx (ячеек по X):").pack()
        self.nx_var = tk.IntVar(value=73)
        tk.Spinbox(root, from_=10, to=200, textvariable=self.nx_var).pack()

        tk.Label(root, text="ny (ячеек по Y):").pack()
        self.ny_var = tk.IntVar(value=50)
        tk.Spinbox(root, from_=10, to=200, textvariable=self.ny_var).pack()

        # Параметры quiver
        tk.Label(root, text="Параметры quiver:").pack()

        tk.Label(root, text="Масштаб (scale):").pack()
        self.scale_var = tk.DoubleVar(value=20.0)
        tk.Spinbox(root, from_=1.0, to=200.0, increment=1.0,
                  textvariable=self.scale_var).pack()

        tk.Label(root, text="Толщина стрелок (width):").pack()
        self.width_var = tk.DoubleVar(value=0.005)
        tk.Spinbox(root, from_=0.001, to=0.02, increment=0.001,
                  textvariable=self.width_var).pack()

        # Цветовая карта
        tk.Label(root, text="Цветовая карта:").pack()
        self.cmap_var = tk.StringVar(value="jet")
        cmaps = ["jet", "viridis", "plasma", "inferno", "magma", "turbo"]
        cmap_menu = ttk.Combobox(root, textvariable=self.cmap_var, values=cmaps)
        cmap_menu.pack()

        # Кнопки
        self.start_button = tk.Button(root, text="Начать", command=self.start)
        self.start_button.pack()

        self.preview_button = tk.Button(root, text="Предпросмотр",
                                       command=self.preview)
        self.preview_button.pack()

        self.cancel_button = tk.Button(root, text="Отмена", command=self.cancel)
        self.cancel_button.pack()

    def start(self):
        params = VectorFieldParameters(
            ptv_folder=self.ptv_folder_var.get(),
            nx=self.nx_var.get(),
            ny=self.ny_var.get(),
            scale=self.scale_var.get(),
            width=self.width_var.get(),
            cmap=self.cmap_var.get()
        )

        success, error = self.executor.set_parameters(params)
        if not success:
            tk.messagebox.showerror("Ошибка", error)
            return

        # Показать статистику
        stats_text = self.get_statistics_text()
        if stats_text:
            tk.messagebox.showinfo("Статистика", stats_text)

        # Запуск в потоке
        import threading
        thread = threading.Thread(target=self.run_vector_field)
        thread.start()

    def preview(self):
        """Показать предпросмотр."""
        preview = self.executor.get_preview("cam_1")
        if preview is not None:
            # Сохранить и показать preview
            import cv2
            cv2.imwrite("preview.png", preview)
            tk.messagebox.showinfo("Предпросмотр",
                                  "Предпросмотр сохранён в preview.png")

    def get_statistics_text(self):
        """Получить текст статистики для отображения."""
        cam1_stats = self.executor.get_statistics("cam_1")
        cam2_stats = self.executor.get_statistics("cam_2")

        text = ""
        if cam1_stats:
            text += f"cam_1: {cam1_stats['vectors_count']} векторов\n"
            if cam1_stats.get('mean_length'):
                text += f"  Средняя длина: {cam1_stats['mean_length']:.2f}\n"

        if cam2_stats:
            text += f"cam_2: {cam2_stats['vectors_count']} векторов\n"
            if cam2_stats.get('mean_length'):
                text += f"  Средняя длина: {cam2_stats['mean_length']:.2f}\n"

        return text

    def run_vector_field(self):
        result = self.executor.execute()
        self.root.after(0, lambda: self.on_finished(result))

    def cancel(self):
        self.executor.cancel()
```

### Запуск примера

```bash
cd execute
python execute_vector_field.py
```

Это запустит пример использования с демонстрацией всех возможностей.

## Добавление новых модулей

При добавлении новых модулей выполнения (например, `execute_ptv.py`), следуйте этой структуре:

1. **Параметры** - создайте `@dataclass` с параметрами для GUI
2. **Executor** - создайте класс с методами:
   - `set_parameters()` - установка и валидация параметров
   - `set_progress_callback()` - установка callback для прогресса
   - `cancel()` - отмена выполнения
   - `execute()` - запуск обработки
3. **Функция быстрого старта** - создайте `run_xxx()` для простого использования
4. **Пример** - добавьте `example_gui_usage()` для демонстрации

## Требования

- Python 3.10+
- NumPy
- Pillow
- Модули из `src/` проекта ParticleAnalysis
