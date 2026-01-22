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
```

### Пример полного пайплайна

```python
from execute.execute_sorting import run_sorting
from execute.execute_binarization import run_binarization
from execute.execute_ptv_analysis import run_ptv_analysis

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

print(f"\nРезультаты в: {ptv_result.output_folder}")
```

## Структура

```
execute/
├── __init__.py              # Инициализация модуля
├── execute_binarization.py  # Выполнение бинарной фильтрации
├── execute_sorting.py       # Сортировка изображений по камерам
├── execute_ptv_analysis.py  # PTV анализ (детектирование и сопоставление частиц)
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
