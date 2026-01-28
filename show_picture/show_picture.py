"""
Модуль для визуализации и отображения результатов детекции пустых мест на полках.

Этот модуль предоставляет класс ShowPicture для отображения результатов работы
системы мониторинга полок. Поддерживает как непрерывный видеопоток, так и
периодический мониторинг с заданным интервалом.

Основные возможности:
    - Отображение видеопотока с камеры в реальном времени
    - Визуализация bounding boxes обнаруженных объектов
    - Отображение информации о проценте наполнения полок
    - Поддержка масштабирования кадров для удобного просмотра
    - Периодический вывод статистики (по умолчанию каждые 5 минут)
    - Обработка одного кадра для периодического мониторинга

Классы:
    ShowPicture: Основной класс для визуализации результатов детекции

Методы ShowPicture:
    - start(): Запускает непрерывную обработку видеопотока с отображением
    - frame(): Обрабатывает один кадр и возвращает результаты анализа
    - run_periodic(): Запускает периодический мониторинг с заданным интервалом
    - start_in_store(): Запускает периодический мониторинг с отправкой данных на API

Использование:
    from show_picture.show_picture import ShowPicture
    from camera.camera import Camera
    from area_calculation.calculations import load_shelf_coordinates_from_json
    from ultralytics import YOLO
    
    model = YOLO('path/to/model.pt')
    camera = Camera(ip_camera='192.168.1.100')
    show = ShowPicture(model=model)
    
    # Вариант 1: Отображение видеопотока
    show.start(camera=camera, json_path='shelf_coordinates.json')
    
    # Вариант 2: Отправка данных на API
    shelf_coordinates = load_shelf_coordinates_from_json('shelf_coordinates.json')
    show.start_in_store(camera=camera, shelf_coordinates=shelf_coordinates, 
                       id_store=1, time_interval=60)

Автор: [Ваше имя]
Дата: 2026-01-27
"""

import cv2
import time
import io
import requests
from ultralytics import YOLO

from area_calculation.area_calculation import AreaCalculator
from area_calculation.calculations import load_shelf_coordinates_from_json
from camera.camera import Camera
from config import SKIP_FRAMES, MAX_DISPLAY_WIDTH, API_BASE_URL


def resize_frame(frame, max_width=MAX_DISPLAY_WIDTH):
    """
    Масштабирует кадр, сохраняя пропорции, чтобы ширина не превышала max_width.

    Args:
        frame: Входной кадр (numpy array)
        max_width: Максимальная ширина для отображения

    Returns:
        Tuple: (масштабированный кадр, коэффициент масштабирования)
    """
    height, width = frame.shape[:2]

    if width <= max_width:
        return frame, 1.0

    # Вычисляем коэффициент масштабирования
    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Масштабируем кадр
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_frame, scale


def on_frame_processed(frame, results):
    print(f"Процент наполнения: {results['fill_percentage']:.2f}%")
    print(f"Обнаружено объектов: {results['num_objects']}")


class ShowPicture:
    def __init__(self, model:YOLO):

        self.area = AreaCalculator(model)
        self.last_output_time = 0
        self.output_interval = 300  # 5 минут в секундах
    def start(self, camera:Camera, json_path:str, video:bool = True):

        try:
            shelf_coordinates = load_shelf_coordinates_from_json(json_path)
            for frame, results in self.area.process_camera_stream(
                camera=camera,
                shelf_coordinates=shelf_coordinates,
                filter_objects_in_shelves=True,
                callback=on_frame_processed,
                skip_frames=SKIP_FRAMES
            ):
                if video:
                    # Сначала масштабируем кадр для отображения
                    display_frame, scale = resize_frame(frame.copy(), MAX_DISPLAY_WIDTH)

                    # Рисуем bounding boxes для всех обнаруженных объектов
                    # Координаты объектов масштабируются тем же коэффициентом, что и кадр
                    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                    for obj in results['objects_info']:
                        x1, y1, x2, y2 = obj['coordinates']
                        # Масштабируем координаты под масштабированный кадр
                        x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)

                        # Выбираем цвет в зависимости от класса
                        color = colors[obj['class_id'] % len(colors)]

                        # Рисуем прямоугольник (bounding box) с масштабированной толщиной
                        box_thickness = max(1, int(2 * scale)) if scale < 1.0 else 2
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, box_thickness)

                        # Формируем текст с классом и уверенностью
                        label = f"{obj['class']} {obj['confidence']:.2f}"

                        # Вычисляем размер текста для фона (масштабируем размер шрифта)
                        font_scale = 0.6 * scale if scale < 1.0 else 0.6
                        thickness = max(1, int(2 * scale)) if scale < 1.0 else 2
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                        )

                        # Рисуем фон для текста
                        cv2.rectangle(
                            display_frame,
                            (x1, y1 - text_height - baseline - 5),
                            (x1 + text_width, y1),
                            color,
                            -1
                        )

                        # Рисуем текст
                        cv2.putText(
                            display_frame,
                            label,
                            (x1, y1 - baseline - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (255, 255, 255),
                            thickness
                        )

                    # Добавляем текст с информацией о проценте наполнения
                    fill_text = f"Fill: {results['fill_percentage']:.1f}%"
                    objects_text = f"Objects: {results['num_objects']}"

                    # Масштабируем размеры информационной панели
                    info_font_scale = 1.0 * scale if scale < 1.0 else 1.0
                    info_thickness = max(1, int(2 * scale)) if scale < 1.0 else 2
                    info_panel_x2 = int(300 * scale) if scale < 1.0 else 300
                    info_panel_y2 = int(85 * scale) if scale < 1.0 else 85

                    # Рисуем фон для информационного текста
                    cv2.rectangle(display_frame, (5, 5), (info_panel_x2, info_panel_y2), (0, 0, 0), -1)
                    cv2.rectangle(display_frame, (5, 5), (info_panel_x2, info_panel_y2), (0, 255, 0), info_thickness)

                    # Рисуем текст на кадре
                    text_y1 = int(35 * scale) if scale < 1.0 else 35
                    text_y2 = int(75 * scale) if scale < 1.0 else 75
                    cv2.putText(display_frame, fill_text, (10, text_y1),
                                cv2.FONT_HERSHEY_SIMPLEX, info_font_scale, (0, 255, 0), info_thickness)
                    cv2.putText(display_frame, objects_text, (10, text_y2),
                                cv2.FONT_HERSHEY_SIMPLEX, info_font_scale, (0, 255, 0), info_thickness)

                    # Отображаем кадр (уже масштабированный)
                    cv2.imshow('Shelf Monitoring', display_frame)

                    # Выход по нажатию 'q' или ESC
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' или ESC
                        break

        except KeyboardInterrupt:
            print("Остановка обработки потока...")

        finally:
            camera.release()
            cv2.destroyAllWindows()


    def frame(self, camera:Camera, shelf_coordinates):
        """
        Получает один кадр с камеры, обрабатывает его и выводит результаты раз в 5 минут.
        
        Returns:
            Tuple (frame, results_dict) или None, если еще не прошло 5 минут с последнего вывода
        """

        # Получаем один кадр и обрабатываем его
        frame, results = self.area.frame_camera(
            camera=camera,
            shelf_coordinates=shelf_coordinates,
            filter_objects_in_shelves=True
        )


        return frame, results
    
    def run_periodic(self, camera:Camera, shelf_coordinates):
    
        """
        Запускает цикл, который получает данные с камеры и выводит их раз в 5 минут.
        Камера подключается один раз при инициализации класса.
        """
        print("Запуск периодического мониторинга...")
        print(f"Данные будут получаться и выводиться каждые {self.output_interval // 60} минут")
        print("Для остановки нажмите Ctrl+C\n")
        
        # Первый вывод сразу
        self.last_output_time = 0

        try:
            while True:
                # Получаем данные с камеры и выводим, если прошло 5 минут
                result = self.frame(camera=camera, shelf_coordinates=shelf_coordinates)
                
                if result is None:
                    # Еще не прошло 5 минут, ждем перед следующей проверкой
                    time.sleep(10)  # Проверяем каждые 10 секунд
                else:
                    # Данные были выведены, ждем 5 минут перед следующим получением
                    time.sleep(self.output_interval)
                
        except KeyboardInterrupt:
            print("\nОстановка мониторинга...")
        finally:
            camera.release()
            print("Камера отключена")
    def start_in_store(self, camera:Camera, shelf_coordinates:list, id_store:int, 
                       time_interval:int = 60, api_url:str = None):
        """
        Запускает периодический мониторинг полок с отправкой данных на API.
        
        Args:
            camera: Экземпляр класса Camera для получения кадров
            shelf_coordinates: Список координат полок [(x1, y1, x2, y2), ...]
            id_store: ID магазина для отправки на API
            time_interval: Интервал между отправками данных в секундах (по умолчанию 60)
            api_url: URL API эндпоинта (по умолчанию используется API_BASE_URL из config)
        
        Отправляет на API:
            - id_store: ID магазина
            - void: Процент наполнения (int, округленный)
            - ip_camera: IP адрес камеры
            - file: Изображение кадра в формате JPEG
        """
        if api_url is None:
            api_url = f"{API_BASE_URL}/entrance/photo"
        
        print(f"Запуск мониторинга для магазина ID: {id_store}")
        print(f"Интервал отправки данных: {time_interval} секунд")
        print(f"API URL: {api_url}")
        print("Для остановки нажмите Ctrl+C\n")
        
        try:
            while True:
                try:
                    # Получаем кадр и результаты анализа
                    frame, results = self.frame(camera=camera, shelf_coordinates=shelf_coordinates)
                    
                    if frame is None or results is None:
                        print("Не удалось получить кадр, пропускаем итерацию...")
                        time.sleep(time_interval)
                        continue
                    
                    ip_camera = camera.ip_camera
                    fill_percentage = results['fill_percentage']
                    void_percentage = int(round(fill_percentage))  # Округляем до целого числа
                    
                    # Конвертируем кадр в JPEG формат для отправки
                    # Используем cv2.imencode для кодирования изображения в память
                    success, buffer = cv2.imencode('.jpg', frame)
                    
                    if not success:
                        print("Ошибка кодирования изображения, пропускаем отправку...")
                        time.sleep(time_interval)
                        continue
                    
                    # Создаем BytesIO объект из буфера
                    image_bytes = io.BytesIO(buffer.tobytes())
                    
                    # Подготавливаем данные для отправки
                    files = {
                        'file': ('image.jpg', image_bytes, 'image/jpeg')
                    }
                    data = {
                        'id_store': id_store,
                        'void': void_percentage,
                        'ip_camera': ip_camera
                    }
                    
                    # Отправляем POST запрос на API
                    try:
                        response = requests.post(api_url, files=files, data=data, timeout=10)
                        
                        if response.status_code == 200:
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Данные успешно отправлены: "
                                  f"ID магазина={id_store}, Наполнение={void_percentage}%, IP={ip_camera}")
                        else:
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ошибка отправки данных: "
                                  f"HTTP {response.status_code} - {response.text}. response: {response}")
                    
                    except requests.exceptions.RequestException as e:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ошибка подключения к API: {e}")
                    
                    # Ждем перед следующей отправкой
                    time.sleep(time_interval)
                    
                except KeyboardInterrupt:
                    print("\nОстановка мониторинга...")
                    break
                except Exception as e:
                    print(f"Ошибка в цикле мониторинга: {e}")
                    time.sleep(time_interval)  # Ждем перед следующей попыткой
        finally:
            camera.release()
            print("Камера отключена")