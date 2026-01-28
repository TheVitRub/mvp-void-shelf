"""
Модуль вспомогательных функций для математических расчетов и визуализации.

Этот модуль содержит утилиты для работы с координатами полок, расчета площадей
перекрывающихся прямоугольников и визуализации результатов детекции.

Основные функции:
    - calculate_union_area_sweepline(): Вычисление объединенной площади прямоугольников
    - is_rectangle_inside_shelves(): Проверка принадлежности объекта к полкам
    - load_shelf_coordinates_from_json(): Загрузка координат полок из JSON
    - visualize_shelves_and_predictions(): Визуализация полок и детекций на изображении
    - merge_intervals(): Объединение перекрывающихся интервалов
    - remove_interval(): Удаление интервала из списка активных интервалов
    - calculate_union_height(): Вычисление общей высоты объединенных интервалов

Алгоритмы:
    Sweepline Algorithm: Эффективный алгоритм для вычисления объединенной площади
    перекрывающихся прямоугольников. Временная сложность: O(n² log n), где n - количество
    прямоугольников. Алгоритм работает путем сканирования плоскости слева направо и
    отслеживания активных интервалов по оси Y.

Формат координат:
    Все координаты представлены в формате (x1, y1, x2, y2), где:
        - (x1, y1): левый верхний угол прямоугольника
        - (x2, y2): правый нижний угол прямоугольника

Формат JSON для координат полок:
    {
        "image_path": "path/to/image.jpg",
        "shelves": [[x1, y1, x2, y2], ...],
        "image_size": [width, height]
    }

Использование:
    from MVP.area_calculation.calculations import (
        calculate_union_area_sweepline,
        load_shelf_coordinates_from_json,
        visualize_shelves_and_predictions
    )
    
    # Загрузка координат
    shelves = load_shelf_coordinates_from_json('shelf_coordinates.json')
    
    # Расчет площади
    rectangles = [(100, 100, 200, 200), (150, 150, 250, 250)]
    area = calculate_union_area_sweepline(rectangles)
    
    # Визуализация
    visualize_shelves_and_predictions(
        image_path='image.jpg',
        model_path='model.pt',
        shelf_coordinates=shelves,
        save_path='visualization.png'
    )

Примечание:
    Функции оптимизированы для работы с большим количеством прямоугольников
    и обеспечивают точные результаты даже при сложных перекрытиях.

Автор: [Ваше имя]
Дата: 2026-01-27
"""

import json
from typing import List, Tuple

import numpy as np
from PIL import Image
from matplotlib import patches, pyplot as plt
from ultralytics import YOLO


def calculate_union_area_sweepline(rectangles: List[Tuple[float, float, float, float]]) -> float:
    """
    Вычисляет объединенную площадь перекрывающихся прямоугольников используя Sweepline Algorithm.

    Args:
        rectangles: Список прямоугольников в формате [(x1, y1, x2, y2), ...]

    Returns:
        Объединенная площадь всех прямоугольников
    """
    if not rectangles:
        return 0.0

    # Создаем события: (x, тип, y1, y2)
    # тип: 1 = начало прямоугольника, -1 = конец прямоугольника
    events = []
    for x1, y1, x2, y2 in rectangles:
        events.append((x1, 1, y1, y2))  # Начало
        events.append((x2, -1, y1, y2))  # Конец

    # Сортируем события по x
    events.sort()

    # Отслеживаем активные интервалы по Y
    active_intervals = []  # Список (y1, y2) активных интервалов
    total_area = 0.0
    prev_x = None

    for x, event_type, y1, y2 in events:
        if prev_x is not None and x > prev_x and active_intervals:
            # Вычисляем площадь между prev_x и x
            height = calculate_union_height(active_intervals)
            width = x - prev_x
            total_area += height * width

        # Обновляем активные интервалы
        if event_type == 1:  # Начало прямоугольника
            active_intervals.append((y1, y2))
            active_intervals = merge_intervals(active_intervals)
        else:  # Конец прямоугольника
            active_intervals = remove_interval(active_intervals, (y1, y2))

        prev_x = x

    return total_area


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Объединяет перекрывающиеся интервалы.

    Args:
        intervals: Список интервалов [(y1, y2), ...]

    Returns:
        Список объединенных интервалов
    """
    if not intervals:
        return []

    # Сортируем по y1
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        last = merged[-1]
        # Если текущий интервал перекрывается или соприкасается с последним
        if current[0] <= last[1]:
            # Объединяем интервалы
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged


def remove_interval(intervals: List[Tuple[float, float]],
                    interval_to_remove: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Удаляет интервал из списка активных интервалов.

    Args:
        intervals: Список активных интервалов
        interval_to_remove: Интервал для удаления (y1, y2)

    Returns:
        Обновленный список интервалов
    """
    if not intervals:
        return []

    result = []
    y1_remove, y2_remove = interval_to_remove

    for y1, y2 in intervals:
        # Если интервал полностью покрыт удаляемым - пропускаем
        if y1 >= y1_remove and y2 <= y2_remove:
            continue
        # Если интервал полностью вне удаляемого - оставляем
        elif y2 <= y1_remove or y1 >= y2_remove:
            result.append((y1, y2))
        # Если интервал содержит удаляемый - разбиваем на части
        elif y1 < y1_remove and y2 > y2_remove:
            result.append((y1, y1_remove))
            result.append((y2_remove, y2))
        # Если перекрытие слева
        elif y1 < y1_remove and y2 > y1_remove:
            result.append((y1, y1_remove))
        # Если перекрытие справа
        elif y1 < y2_remove and y2 > y2_remove:
            result.append((y2_remove, y2))

    return merge_intervals(result)


def calculate_union_height(intervals: List[Tuple[float, float]]) -> float:
    """
    Вычисляет общую высоту объединенных интервалов.

    Args:
        intervals: Список интервалов [(y1, y2), ...]

    Returns:
        Суммарная высота
    """
    if not intervals:
        return 0.0

    merged = merge_intervals(intervals)
    return sum(y2 - y1 for y1, y2 in merged)


def is_rectangle_inside_shelves(rect: Tuple[float, float, float, float],
                                shelves: List[Tuple[float, float, float, float]]) -> bool:
    """
    Проверяет, находится ли прямоугольник внутри хотя бы одной полки.

    Args:
        rect: Прямоугольник (x1, y1, x2, y2)
        shelves: Список полок [(x1, y1, x2, y2), ...]

    Returns:
        True, если прямоугольник находится внутри хотя бы одной полки
    """
    x1, y1, x2, y2 = rect
    for sx1, sy1, sx2, sy2 in shelves:
        # Проверяем, находится ли центр прямоугольника внутри полки
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if sx1 <= center_x <= sx2 and sy1 <= center_y <= sy2:
            return True
    return False


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


def visualize_shelves_and_predictions(
        image_path: str,
        model_path: str = "my_best-shelf-void-model.pt",
        shelf_coordinates: List[Tuple[float, float, float, float]] = None,
        confidence_threshold: float = 0.25,
        save_path: str = None
):
    """
    Визуализирует полки и предсказания модели на изображении.

    Args:
        image_path: Путь к изображению
        model_path: Путь к модели YOLO
        shelf_coordinates: Список координат полок
        confidence_threshold: Порог уверенности
        save_path: Путь для сохранения изображения (опционально)
    """
    # Загружаем модель и делаем предсказание
    model = YOLO(model_path)
    results = model(image_path, conf=confidence_threshold)
    result = results[0]

    # Загружаем изображение
    img = Image.open(image_path)

    # Создаем фигуру
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Визуализация полок и предсказаний модели', fontsize=14, fontweight='bold', pad=15)

    # Рисуем полки
    if shelf_coordinates:
        shelf_colors = plt.cm.Set3(np.linspace(0, 1, len(shelf_coordinates)))
        for i, (x1, y1, x2, y2) in enumerate(shelf_coordinates):
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=3, edgecolor=shelf_colors[i],
                facecolor='none', linestyle='--', alpha=0.8
            )
            ax.add_patch(rect)

            # Добавляем номер полки
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            ax.text(center_x, y1 - 10, f'Полка {i + 1}',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round', facecolor=shelf_colors[i], alpha=0.7),
                    fontsize=12, fontweight='bold', color='white')

    # Рисуем предсказания модели
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls]

        color = colors[cls % len(colors)]
        width = x2 - x1
        height = y2 - y1

        # Рисуем прямоугольник
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color,
            facecolor='none', linestyle='-', alpha=0.7
        )
        ax.add_patch(rect)

        # Добавляем текст с классом и уверенностью
        label = f"{class_name} {conf:.2%}"
        ax.text(x1, y1 - 5, label,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                fontsize=9, color='white', weight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Визуализация сохранена: {save_path}")

    plt.show()