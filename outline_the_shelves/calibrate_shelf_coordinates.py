"""
Интерактивный инструмент для визуализации и калибровки координат полок на изображении.

Этот модуль предоставляет графический интерфейс для определения координат полок
на изображении. Позволяет пользователю интерактивно отмечать полки кликами мыши
и сохранять их координаты в JSON файл для дальнейшего использования в системе мониторинга.

Основные возможности:
    - Интерактивное создание прямоугольников полок кликами мыши
    - Визуализация всех отмеченных полок с номерами
    - Сохранение координат в JSON файл
    - Загрузка ранее сохраненных координат
    - Удаление и редактирование полок
    - Предпросмотр создаваемой полки при движении мыши
    - Генерация Python кода для использования координат

Классы:
    ShelfCalibrator: Основной класс для калибровки координат полок

Методы ShelfCalibrator:
    - on_click(): Обработка кликов мыши для создания полок
    - save_coordinates(): Сохранение координат в JSON файл
    - load_coordinates(): Загрузка координат из JSON файла
    - remove_last_shelf(): Удаление последней добавленной полки
    - clear_all(): Очистка всех полок
    - show_python_code(): Вывод Python кода с координатами

Управление:
    - Левая кнопка мыши: Первый клик - начало полки, второй клик - конец полки
    - Правая кнопка мыши: Отмена создания текущей полки
    - Delete/Backspace: Удалить последнюю полку
    - Escape: Отменить создание текущей полки
    - C: Очистить все полки
    - S: Сохранить координаты
    - L: Загрузить координаты

Формат сохранения:
    JSON файл с именем: {имя_изображения}_shelf_coordinates.json
    Содержит:
        - image_path: путь к изображению
        - shelves: список координат [[x1, y1, x2, y2], ...]
        - image_size: размер изображения [width, height]

Использование:
    from MVP.outline_the_shelves.calibrate_shelf_coordinates import calibrate_shelves
    
    # Запуск калибровки
    shelves = calibrate_shelves('path/to/image.jpg')
    
    # Или с начальными координатами
    initial_shelves = [(100, 100, 200, 200), (300, 100, 400, 200)]
    shelves = calibrate_shelves('path/to/image.jpg', initial_shelves)
    
    print(f"Откалибровано полок: {len(shelves)}")

Примечание:
    После закрытия окна калибровки координаты будут доступны в возвращаемом списке.
    Рекомендуется сохранить координаты через кнопку "Сохранить координаты" для
    последующего использования в системе мониторинга.

Автор: [Ваше имя]
Дата: 2026-01-27
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional
import json
import os


class ShelfCalibrator:
    def __init__(self, image_path: str, initial_shelves: List[Tuple[float, float, float, float]] = None):
        """
        Инициализация калибратора.
        
        Args:
            image_path: Путь к изображению
            initial_shelves: Начальные координаты полок [(x1, y1, x2, y2), ...]
        """
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.shelves = initial_shelves.copy() if initial_shelves else []
        self.current_shelf = None  # Текущая полка в процессе создания (x1, y1, x2, y2)
        self.start_point = None  # Начальная точка для создания полки
        
        # Создаем фигуру и оси
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.ax.imshow(self.image)
        self.ax.set_title('Калибровка координат полок\n'
                         'Инструкция:\n'
                         '1. Кликните левой кнопкой мыши для начала полки\n'
                         '2. Кликните еще раз для завершения полки\n'
                         '3. Используйте кнопки для управления', 
                         fontsize=12, pad=20)
        self.ax.axis('off')
        
        # Подключаем обработчики событий
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Создаем кнопки
        self.create_buttons()
        
        # Обновляем визуализацию
        self.update_display()
    
    def create_buttons(self):
        """Создает кнопки управления."""
        # Кнопка "Удалить последнюю полку"
        ax_remove = plt.axes([0.02, 0.02, 0.15, 0.04])
        self.btn_remove = Button(ax_remove, 'Удалить последнюю')
        self.btn_remove.on_clicked(self.remove_last_shelf)
        
        # Кнопка "Очистить все"
        ax_clear = plt.axes([0.18, 0.02, 0.15, 0.04])
        self.btn_clear = Button(ax_clear, 'Очистить все')
        self.btn_clear.on_clicked(self.clear_all)
        
        # Кнопка "Сохранить координаты"
        ax_save = plt.axes([0.34, 0.02, 0.15, 0.04])
        self.btn_save = Button(ax_save, 'Сохранить координаты')
        self.btn_save.on_clicked(self.save_coordinates)
        
        # Кнопка "Загрузить координаты"
        ax_load = plt.axes([0.50, 0.02, 0.15, 0.04])
        self.btn_load = Button(ax_load, 'Загрузить координаты')
        self.btn_load.on_clicked(self.load_coordinates)
        
        # Кнопка "Показать код Python"
        ax_code = plt.axes([0.66, 0.02, 0.15, 0.04])
        self.btn_code = Button(ax_code, 'Показать код')
        self.btn_code.on_clicked(self.show_python_code)
    
    def on_click(self, event):
        """Обработчик клика мыши."""
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # Левая кнопка мыши
            x, y = event.xdata, event.ydata
            
            if self.start_point is None:
                # Начало создания новой полки
                self.start_point = (x, y)
                print(f"Начальная точка полки: ({x:.0f}, {y:.0f})")
            else:
                # Завершение создания полки
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Убеждаемся, что x1 < x2 и y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                shelf = (x1, y1, x2, y2)
                self.shelves.append(shelf)
                self.start_point = None
                print(f"Полка добавлена: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
                self.update_display()
        
        elif event.button == 3:  # Правая кнопка мыши - отмена текущей полки
            self.start_point = None
            self.update_display()
    
    def on_motion(self, event):
        """Обработчик движения мыши для предпросмотра полки."""
        if event.inaxes == self.ax and self.start_point is not None:
            self._last_mouse_pos = (event.xdata, event.ydata)
            self.update_display()
    
    def on_key(self, event):
        """Обработчик нажатия клавиш."""
        if event.key == 'delete' or event.key == 'backspace':
            self.remove_last_shelf(None)
        elif event.key == 'escape':
            self.start_point = None
            self.update_display()
        elif event.key == 'c':
            self.clear_all(None)
        elif event.key == 's':
            self.save_coordinates(None)
        elif event.key == 'l':
            self.load_coordinates(None)
    
    def remove_last_shelf(self, event):
        """Удаляет последнюю добавленную полку."""
        if self.shelves:
            removed = self.shelves.pop()
            print(f"Полка удалена: {removed}")
            self.update_display()
    
    def clear_all(self, event):
        """Очищает все полки."""
        self.shelves = []
        self.start_point = None
        print("Все полки очищены")
        self.update_display()
    
    def save_coordinates(self, event):
        """Сохраняет координаты в JSON файл."""
        if not self.shelves:
            print("Нет полок для сохранения!")
            return
        
        # Создаем имя файла на основе изображения
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        json_path = f"{base_name}_shelf_coordinates.json"
        
        data = {
            'image_path': self.image_path,
            'shelves': self.shelves,
            'image_size': self.image.size
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Координаты сохранены в: {json_path}")
        print(f"Всего полок: {len(self.shelves)}")
    
    def load_coordinates(self, event):
        """Загружает координаты из JSON файла."""
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        json_path = f"{base_name}_shelf_coordinates.json"
        
        if not os.path.exists(json_path):
            print(f"Файл не найден: {json_path}")
            return
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.shelves = [tuple(shelf) for shelf in data['shelves']]
        print(f"Координаты загружены из: {json_path}")
        print(f"Всего полок: {len(self.shelves)}")
        self.update_display()
    
    def show_python_code(self, event):
        """Показывает код Python для использования координат."""
        if not self.shelves:
            print("Нет полок для отображения!")
            return
        
        print("\n" + "="*60)
        print("КОД PYTHON ДЛЯ ИСПОЛЬЗОВАНИЯ В calculate_shelf_fill_percentage.py:")
        print("="*60)
        print("shelf_coordinates = [")
        for i, (x1, y1, x2, y2) in enumerate(self.shelves):
            comment = f"  # Полка {i+1}"
            print(f"{comment}")
            print(f"    ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}),")
        print("]")
        print("="*60 + "\n")
    
    def update_display(self):
        """Обновляет отображение полок на изображении."""
        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.set_title(f'Калибровка координат полок | Полок: {len(self.shelves)}\n'
                         'Инструкция: Кликните дважды для создания полки | '
                         'Delete - удалить последнюю | Esc - отмена | C - очистить все | S - сохранить | L - загрузить',
                         fontsize=11, pad=15)
        self.ax.axis('off')
        
        # Рисуем существующие полки
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.shelves), 1)))
        for i, (x1, y1, x2, y2) in enumerate(self.shelves):
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=colors[i % len(colors)], 
                facecolor='none', linestyle='--'
            )
            self.ax.add_patch(rect)
            
            # Добавляем номер полки
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            self.ax.text(center_x, center_y, f'Полка {i+1}',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor=colors[i % len(colors)], alpha=0.7),
                        fontsize=10, fontweight='bold', color='white')
        
        # Рисуем текущую полку в процессе создания
        if self.start_point is not None:
            x1, y1 = self.start_point
            # Получаем текущую позицию мыши (если доступна)
            if hasattr(self, '_last_mouse_pos'):
                x2, y2 = self._last_mouse_pos
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                rect = patches.Rectangle(
                    (x_min, y_min), width, height,
                    linewidth=2, edgecolor='red', 
                    facecolor='red', alpha=0.3, linestyle='-'
                )
                self.ax.add_patch(rect)
                self.ax.plot(x1, y1, 'ro', markersize=8, label='Начало')
        
        self.fig.canvas.draw()
    
    def show(self):
        """Показывает окно калибровки."""
        plt.tight_layout()
        plt.show()


def calibrate_shelves(image_path: str, initial_shelves: List[Tuple[float, float, float, float]] = None):
    """
    Запускает интерактивную калибровку координат полок.
    
    Args:
        image_path: Путь к изображению
        initial_shelves: Начальные координаты полок (опционально)
    
    Returns:
        Список координат полок [(x1, y1, x2, y2), ...]
    """
    calibrator = ShelfCalibrator(image_path, initial_shelves)
    calibrator.show()
    return calibrator.shelves


def load_shelf_coordinates_from_json(json_path: str) -> List[Tuple[float, float, float, float]]:
    """
    Загружает координаты полок из JSON файла.
    
    Args:
        json_path: Путь к JSON файлу
    
    Returns:
        Список координат полок
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [tuple(shelf) for shelf in data['shelves']]


def main():
    """Пример использования калибратора."""
    # Путь к изображению (относительно корня проекта)
    image_path = "dataset/images/train/shot_20260123_193334.jpg"
    
    # Начальные координаты (опционально) - можно загрузить из файла или задать вручную
    initial_shelves = [
        (700, 700, 1200, 850),
        (1200, 700, 1800, 900),
        (1200, 1200, 1800, 1400),
    ]
    
    # Запускаем калибровку
    shelves = calibrate_shelves(image_path, initial_shelves)
    
    # После закрытия окна координаты будут в переменной shelves
    print(f"\nФинальные координаты полок:")
    for i, shelf in enumerate(shelves):
        print(f"Полка {i+1}: {shelf}")


if __name__ == "__main__":
    main()

