"""
Модуль для расчета площади объектов и процента наполнения полок.

Этот модуль является основным компонентом системы мониторинга полок.
Он использует обученную YOLO модель для детекции пустых мест на полках
и вычисляет процент их наполнения на основе площади обнаруженных объектов.

Основные возможности:
    - Детекция объектов на изображениях с помощью YOLO модели
    - Вычисление общей площади обнаруженных объектов
    - Расчет процента наполнения полок
    - Фильтрация объектов по принадлежности к полкам
    - Обработка как статических изображений, так и видеопотоков
    - Использование алгоритма Sweepline для точного расчета площади перекрывающихся объектов

Классы:
    AreaCalculator: Класс для расчета площади и процента наполнения

Методы AreaCalculator:
    - calculate_shelf_fill_percentage(): Вычисляет процент наполнения для изображения
    - process_camera_stream(): Обрабатывает видеопоток с камеры
    - frame_camera(): Обрабатывает один кадр с камеры

Возвращаемый формат результатов:
    {
        'total_objects_area': float,      # Общая площадь объектов (пиксели²)
        'shelf_total_area': float,        # Общая площадь полок (пиксели²)
        'fill_percentage': float,         # Процент наполнения (%)
        'num_objects': int,               # Количество обнаруженных объектов
        'objects_info': List[dict],       # Детальная информация об объектах
        'image_size': Tuple[int, int]    # Размер изображения (width, height)
    }

Использование:
    from MVP.area_calculation.area_calculation import AreaCalculator
    from ultralytics import YOLO
    
    model = YOLO('path/to/model.pt')
    calculator = AreaCalculator(model)
    
    # Обработка изображения
    results = calculator.calculate_shelf_fill_percentage(
        image='path/to/image.jpg',
        shelf_coordinates=[(x1, y1, x2, y2), ...],
        filter_objects_in_shelves=True
    )
    
    print(f"Процент наполнения: {results['fill_percentage']:.2f}%")

Алгоритм:
    Использует Sweepline Algorithm для точного вычисления объединенной площади
    перекрывающихся прямоугольников, что позволяет корректно обрабатывать случаи,
    когда объекты частично перекрывают друг друга.

Автор: [Ваше имя]
Дата: 2026-01-27
"""

import os
from typing import List, Tuple, Union, Optional
import numpy as np

from PIL import Image
from ultralytics import YOLO

from area_calculation.calculations import is_rectangle_inside_shelves, calculate_union_area_sweepline, \
    load_shelf_coordinates_from_json, visualize_shelves_and_predictions
from config import CONFIDENCE_THRESHOLD


class AreaCalculator:
    def __init__(self, model: YOLO):
        self.model = model
        self.confidence_threshold = CONFIDENCE_THRESHOLD

    def calculate_shelf_fill_percentage(
            self,
            image: Union[str, np.ndarray],
            shelf_coordinates: List[Tuple[float, float, float, float]] = None,
            filter_objects_in_shelves: bool = False
    ) -> dict:
        """
        Вычисляет площадь объектов на изображении и процент наполнения полок.

        Args:
            image: Путь к изображению (str) или numpy array (кадр из камеры)
            shelf_coordinates: Список координат полок в формате [(x1, y1, x2, y2), ...]
                              Если None, используется вся площадь изображения
            filter_objects_in_shelves: Если True, учитываются только объекты, находящиеся внутри полок

        Returns:
            Словарь с результатами:
            {
                'total_objects_area': общая площадь объектов,
                'shelf_total_area': общая площадь полок,
                'fill_percentage': процент наполнения,
                'num_objects': количество обнаруженных объектов,
                'objects_info': список информации об объектах
            }
        """
        # Делаем предсказание (YOLO работает как с путями, так и с numpy arrays)
        results = self.model(image, conf=self.confidence_threshold)
        result = results[0]

        # Получаем размеры изображения
        if isinstance(image, str):
            # Если это путь к файлу
            img = Image.open(image)
            img_width, img_height = img.size
        else:
            # Если это numpy array (кадр из камеры)
            # OpenCV использует формат (height, width), а PIL - (width, height)
            img_height, img_width = image.shape[:2]

        # Если координаты полок не заданы, используем всю площадь изображения
        if shelf_coordinates is None:
            shelf_coordinates = [(0, 0, img_width, img_height)]

        # Извлекаем bounding boxes объектов
        objects_rectangles = []
        objects_info = []

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls]

            rect = (float(x1), float(y1), float(x2), float(y2))

            # Фильтруем объекты, если требуется
            if filter_objects_in_shelves and shelf_coordinates:
                if not is_rectangle_inside_shelves(rect, shelf_coordinates):
                    continue

            objects_rectangles.append(rect)
            objects_info.append({
                'id': len(objects_rectangles),
                'class': class_name,
                'class_id': cls,
                'confidence': conf,
                'coordinates': rect,
                'area': (float(x2) - float(x1)) * (float(y2) - float(y1))
            })

        # Вычисляем объединенную площадь всех объектов используя Sweepline Algorithm
        total_objects_area = calculate_union_area_sweepline(objects_rectangles)

        # Вычисляем общую площадь полок
        shelf_total_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in shelf_coordinates)

        # Вычисляем процент наполнения
        if shelf_total_area > 0:
            fill_percentage = (total_objects_area / shelf_total_area) * 100
        else:
            fill_percentage = 0.0

        return {
            'total_objects_area': total_objects_area,
            'shelf_total_area': shelf_total_area,
            'fill_percentage': fill_percentage,
            'num_objects': len(objects_rectangles),
            'objects_info': objects_info,
            'image_size': (img_width, img_height)
        }

    def process_camera_stream(
            self,
            camera,
            shelf_coordinates: List[Tuple[float, float, float, float]] = None,
            filter_objects_in_shelves: bool = False,
            callback: Optional[callable] = None,
            skip_frames: int = 0
    ):
        """
        Обрабатывает поток кадров с камеры и вычисляет процент наполнения полок.

        Args:
            camera: Экземпляр класса Camera
            shelf_coordinates: Список координат полок в формате [(x1, y1, x2, y2), ...]
            filter_objects_in_shelves: Если True, учитываются только объекты, находящиеся внутри полок
            callback: Функция обратного вызова, которая будет вызвана с результатами для каждого кадра.
                     Принимает (frame, results_dict)
            skip_frames: Количество кадров для пропуска между обработками (для оптимизации производительности)

        Yields:
            Tuple (frame, results_dict) для каждого обработанного кадра
        """
        frame_count = 0

        while True:
            frame = camera.read_frame()

            if frame is None:
                continue

            # Пропускаем кадры для оптимизации
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue

            # Обрабатываем кадр
            results = self.calculate_shelf_fill_percentage(
                image=frame,
                shelf_coordinates=shelf_coordinates,
                filter_objects_in_shelves=filter_objects_in_shelves
            )

            # Вызываем callback, если он задан
            if callback:
                callback(frame, results)

            frame_count += 1
            yield frame, results
    def frame_camera(self,
            camera,
            shelf_coordinates: List[Tuple[float, float, float, float]] = None,
            filter_objects_in_shelves: bool = False
                     ):
        """
               Обрабатывает поток кадров с камеры и вычисляет процент наполнения полок.

               Args:
                   camera: Экземпляр класса Camera
                   shelf_coordinates: Список координат полок в формате [(x1, y1, x2, y2), ...]
                   filter_objects_in_shelves: Если True, учитываются только объекты, находящиеся внутри полок

               Returns:
                   Tuple (frame, results_dict) для обработанного кадра
               """


        # Используем read_fresh_frame для получения свежего кадра, очищая буфер
        # Это важно, когда кадры читаются редко (например, раз в минуту)
        frame = camera.read_fresh_frame()





        # Обрабатываем кадр
        results = self.calculate_shelf_fill_percentage(
                image=frame,
                shelf_coordinates=shelf_coordinates,
                filter_objects_in_shelves=filter_objects_in_shelves
            )

        # Вызываем callback, если он задан
        return frame, results


if __name__ == "__main__":
    model_path = r'C:\Users\ryabovva.VOLKOVKMR\PycharmProjects\learn_void_shelf\MVP\my_best-shelf-void-model.pt'
    model = YOLO(model_path)
    area = AreaCalculator(model)
    json_path = r'C:\Users\ryabovva.VOLKOVKMR\PycharmProjects\learn_void_shelf\shot_20260123_193334_shelf_coordinates.json'

    if os.path.exists(json_path):
        print(f"Загружаем координаты из {json_path}")
        shelf_coordinates = load_shelf_coordinates_from_json(json_path)
    else:
        # Пример для нескольких полок (если JSON файл не найден):
        shelf_coordinates = [
            # (x1, y1, x2, y2)
            # Полка 1
            (700, 700, 1200, 850),
            # Полка 2
            (1200, 700, 1800, 900),
            # Полка 3
            (1200, 1200, 1800, 1400),
            # Добавьте больше полок по необходимости
        ]
    # Пример 1: Обработка изображения из файла
    image_path = r'C:\Users\ryabovva.VOLKOVKMR\PycharmProjects\learn_void_shelf\learning\dataset\images\train\shot_20260123_173714.jpg'
    results = area.calculate_shelf_fill_percentage(
        image=image_path,
        shelf_coordinates=shelf_coordinates,
        filter_objects_in_shelves=True
    )

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("=" * 60)
    print(f"Размер изображения: {results['image_size'][0]}x{results['image_size'][1]}")
    print(f"\nОбнаружено объектов: {results['num_objects']}")
    print(f"Общая площадь объектов: {results['total_objects_area']:.2f} пикселей²")
    print(f"Общая площадь полок: {results['shelf_total_area']:.2f} пикселей²")
    print(f"Процент наполнения: {results['fill_percentage']:.2f}%")
    print("\n" + "-" * 60)
    print("Детали объектов:")
    print("-" * 60)

    for obj in results['objects_info']:
        print(f"Объект {obj['id']}:")
        print(f"  Класс: {obj['class']} (ID: {obj['class_id']})")
        print(f"  Уверенность: {obj['confidence']:.2%}")
        print(f"  Координаты: ({obj['coordinates'][0]:.0f}, {obj['coordinates'][1]:.0f}) -> "
              f"({obj['coordinates'][2]:.0f}, {obj['coordinates'][3]:.0f})")
        print(f"  Площадь: {obj['area']:.2f} пикселей²")
        print()

    visualize = True
    if visualize:
        visualize_shelves_and_predictions(
            image_path=image_path,
            model_path=model_path,
            shelf_coordinates=shelf_coordinates,
            confidence_threshold=0.25,
            save_path="shelf_visualization.png"
        )

    # Пример 2: Обработка потока с камеры
    # Раскомментируйте для использования:
    """
    from MVP.camera.camera import Camera
    from MVP.config import SKIP_FRAMES
    
    camera = Camera()
    
    # Функция обратного вызова для обработки результатов
    def on_frame_processed(frame, results):
        print(f"Процент наполнения: {results['fill_percentage']:.2f}%")
        print(f"Обнаружено объектов: {results['num_objects']}")
    
    # Обработка потока кадров
    try:
        for frame, results in area.process_camera_stream(
            camera=camera,
            shelf_coordinates=shelf_coordinates,
            filter_objects_in_shelves=True,
            callback=on_frame_processed,
            skip_frames=SKIP_FRAMES
        ):
            # Здесь можно добавить дополнительную обработку кадра
            # Например, отображение на экране или сохранение
            pass
    except KeyboardInterrupt:
        print("Остановка обработки потока...")
    finally:
        camera.release()
    """