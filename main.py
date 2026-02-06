

import threading
from ultralytics.models import YOLO
from api.api_class import api_class
from api.camera_data_class import CameraData
from camera.camera import Camera
from config import ID_STORE, YOLO_MODEL
from show_picture.show_picture import ShowPicture


camera_data = api_class.get_camera_data()

if len(camera_data) == 0:
    print("Не получено никаких данных по камерам")
    exit(1)

model = YOLO(YOLO_MODEL)

def run_monitoring(camera_data_obj: CameraData):
    """
    Запускает мониторинг для одной камеры в отдельном потоке с отправкой данных на API."""
    try:
        print(f"[Камера {camera_data_obj.name_camera} ({camera_data_obj.ip_camera})] Запуск потока мониторинга...")
        print(f"[Камера {camera_data_obj.name_camera} ({camera_data_obj.ip_camera})] ID магазина: {ID_STORE}")
        camera = Camera(ip_camera=camera_data_obj.ip_camera)
        show_picture = ShowPicture(model=model)
        # Запускаем мониторинг с отправкой данных на API
        show_picture.start_in_store(
            shelf_coordinates=camera_data_obj.shelf_coordinates,
            camera=camera,
            id_store=ID_STORE
        )
        
    except Exception as e:
        print(f"[Камера {camera_data_obj.name_camera} ({camera_data_obj.ip_camera}) ID магазина: {ID_STORE}] Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[Камера {camera_data_obj.name_camera} ({camera_data_obj.ip_camera}) ID магазина: {ID_STORE}] Поток завершен")


# Создаем и запускаем потоки для каждой камеры
threads = []
for idx, camera in enumerate(camera_data):
    thread = threading.Thread(
        target=run_monitoring,
        args=(camera,),
        daemon=False,
        name=f"Camera-{idx}-{camera.ip_camera}"
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