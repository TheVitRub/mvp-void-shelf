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
import threading
from ultralytics.models import YOLO

from area_calculation.calculations import load_shelf_coordinates_from_json
from camera.camera import Camera
from show_picture.show_picture import ShowPicture
from config import ID_STORE, YOLO_MODEL

# Конфигурация камер
# Можно указать несколько IP-адресов камер
# Вариант 1: Через переменные окружения (CAMERA_IP_1, CAMERA_IP_2, ...)
# Вариант 2: Напрямую в списке
camera_ips = []
shelf_coordinates_list = ['shot_20260123_193334_shelf_coordinates.json', 'shot_20260123_165640_shelf_coordinates.json', 'shot_20260123_090705_shelf_coordinates.json']
for i in range(1, 10):  # Проверяем до 9 камер
    ip = os.getenv(f'CAMERA_IP_{i}')
    if ip:
        camera_ips.append(ip)

# Если не найдено камер через переменные окружения, используем CAMERA_IP
if not camera_ips:
    default_ip = os.getenv('CAMERA_IP')
    if default_ip:
        camera_ips.append(default_ip)

# Если камеры не найдены, выводим предупреждение
if not camera_ips:
    print("ВНИМАНИЕ: Не найдено ни одной камеры!")
    print("Укажите IP-адреса камер через переменные окружения:")
    print("  CAMERA_IP - для одной камеры")
    print("  CAMERA_IP_1, CAMERA_IP_2, ... - для нескольких камер")
    exit(1)

print(f"Найдено камер: {len(camera_ips)}")
for i, ip in enumerate(camera_ips, 1):
    print(f"  Камера {i}: {ip}")

# Загрузка модели и координат полок
model_path = YOLO_MODEL
model = YOLO(model_path)

# Загружаем координаты полок для каждой камеры
print(f"\nЗагрузка координат полок из {len(shelf_coordinates_list)} файлов...")
shelf_coordinates = []
for i, json_file in enumerate(shelf_coordinates_list, 1):
    try:
        coords = load_shelf_coordinates_from_json(json_file)
        shelf_coordinates.append(coords)
        print(f"  Файл {i}: {json_file} - загружен успешно")
    except Exception as e:
        print(f"  Файл {i}: {json_file} - ОШИБКА загрузки: {e}")
        # Если файл не загрузился, используем пустой список или последний успешный
        if shelf_coordinates:
            shelf_coordinates.append(shelf_coordinates[-1])
            print(f"    Используются координаты из предыдущего файла")
        else:
            shelf_coordinates.append([])
            print(f"    ВНИМАНИЕ: Нет координат для этой камеры!")

# Проверяем соответствие количества камер и файлов координат
if len(camera_ips) > len(shelf_coordinates):
    print(f"\nВНИМАНИЕ: Камер больше ({len(camera_ips)}), чем файлов координат ({len(shelf_coordinates)})")
    print("Для дополнительных камер будут использованы координаты из последнего файла")
    # Дополняем список координат последним доступным
    while len(shelf_coordinates) < len(camera_ips):
        shelf_coordinates.append(shelf_coordinates[-1] if shelf_coordinates else [])

# Функция для запуска мониторинга в отдельном потоке
def run_camera_monitoring(camera_ip: str, camera_id: int, shelf_coordinates: list, id_store: int):
    """
    Запускает мониторинг для одной камеры в отдельном потоке с отправкой данных на API.
    
    Args:
        camera_ip: IP-адрес камеры
        camera_id: Уникальный идентификатор камеры (1-based, для логирования)
        shelf_coordinates: Список координат полок для всех камер
        id_store: ID магазина для отправки на API
    """
    try:
        # Преобразуем camera_id (1-based) в индекс (0-based)
        coord_index = camera_id - 1
        
        # Проверяем, что координаты существуют
        if coord_index >= len(shelf_coordinates) or not shelf_coordinates[coord_index]:
            print(f"[Камера {camera_id} ({camera_ip})] ОШИБКА: Нет координат полок для этой камеры!")
            return
        
        print(f"[Камера {camera_id} ({camera_ip})] Запуск потока мониторинга...")
        print(f"[Камера {camera_id} ({camera_ip})] Используются координаты из индекса {coord_index}")
        print(f"[Камера {camera_id} ({camera_ip})] ID магазина: {id_store}")
        
        # Создаем экземпляр камеры для этого потока
        camera = Camera(ip_camera=camera_ip)
        
        # Создаем отдельный экземпляр ShowPicture для этого потока
        # Каждый поток должен иметь свой экземпляр, чтобы избежать конфликтов
        show_picture = ShowPicture(model=model)
        
        # Запускаем мониторинг с отправкой данных на API
        show_picture.start_in_store(
            shelf_coordinates=shelf_coordinates[coord_index],
            camera=camera,
            id_store=id_store
        )
        
    except Exception as e:
        print(f"[Камера {camera_id} ({camera_ip})] Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[Камера {camera_id} ({camera_ip})] Поток завершен")

# Создаем и запускаем потоки для каждой камеры
threads = []
for idx, camera_ip in enumerate(camera_ips, 1):
    thread = threading.Thread(
        target=run_camera_monitoring,
        args=(camera_ip, idx, shelf_coordinates, ID_STORE),
        daemon=False,
        name=f"Camera-{idx}-{camera_ip}"
    )
    threads.append(thread)
    thread.start()
    print(f"[Камера {idx}] Поток запущен")

# Ждем завершения всех потоков
print("\nВсе потоки запущены. Для остановки нажмите Ctrl+C\n")
try:
    for thread in threads:
        thread.join()
except KeyboardInterrupt:
    print("\nПолучен сигнал остановки. Завершение работы...")
    # Потоки будут завершены автоматически при выходе из программы

