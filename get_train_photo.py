"""
Скрипт для тестирования системы мониторинга наполнения полок.

Этот скрипт демонстрирует использование системы мониторинга полок с использованием
YOLO модели для детекции пустых мест. Поддерживает работу с одной или несколькими
камерами через RTSP потоки.

Основные возможности:
- Подключение к IP-камерам через RTSP
- Обработка видеопотока в реальном времени
- Детекция пустых мест на полках с помощью обученной YOLO модели
- Визуализация результатов с bounding boxes и информацией о наполнении
- Поддержка многопоточности для работы с несколькими камерами одновременно

Использование:
    1. Убедитесь, что модель YOLO находится по указанному пути
    2. Укажите путь к JSON файлу с координатами полок
    3. Настройте IP адреса камер в списке camera_config
    4. Запустите скрипт: python main.py

Зависимости:
    - ultralytics (YOLO)
    - camera.camera (класс Camera)
    - show_picture.show_picture (класс ShowPicture)

Автор: [Ваше имя]
Дата: 2026-01-27
"""

import os
from ultralytics.models import YOLO

from area_calculation.calculations import load_shelf_coordinates_from_json
from camera.camera import Camera
from show_picture.show_picture import ShowPicture
from config import ID_STORE, YOLO_MODEL

model_path=YOLO_MODEL
model = YOLO(model_path)
json_path=r'shot_20260123_193334_shelf_coordinates.json'
shelf_coordinates = load_shelf_coordinates_from_json(json_path)
camera1 = Camera(ip_camera=os.getenv('CAMERA_IP'))



threads = []
test = ShowPicture(
    model=model)


test.run_active_learning(shelf_coordinates=shelf_coordinates,
    camera=camera1)

