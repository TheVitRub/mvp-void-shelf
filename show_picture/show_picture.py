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
    - process_active_learning(): Сохраняет кадры с низкой уверенностью для разметки
    - run_active_learning(): Запускает цикл сбора данных для Active Learning

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
    
    # Вариант 3: Сбор данных для Active Learning
    show = ShowPicture(model=model, active_learning_dir='training_data')
    show.run_active_learning(camera=camera, shelf_coordinates=shelf_coordinates,
                            save_interval=30, conf_range=(0.10, 0.60))

Автор: [Ваше имя]
Дата: 2026-01-27
"""

import cv2
import time
import io
import os
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
    def __init__(self, model:YOLO, active_learning_dir:str = "to_label"):
        """
        Инициализация класса ShowPicture.
        
        Args:
            model: Модель YOLO для детекции объектов
            active_learning_dir: Базовая директория для сохранения кадров Active Learning
        """
        self.area = AreaCalculator(model)
        self.last_output_time = 0
        self.output_interval = 300  # 5 минут в секундах
        
        # Настройки Active Learning
        self.active_learning_dir = active_learning_dir
        self.last_al_save_time = 0
        self.al_save_interval = 30  # Сохраняем "сомнительный" кадр не чаще раза в 30 секунд
        self.al_conf_low = 0.10    # Нижняя граница уверенности
        self.al_conf_mid = 0.40    # Граница между "maybe" и "almost"
        self.al_conf_high = 0.60   # Верхняя граница уверенности
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
                    
                    # Создаем копию кадра для рисования рамок
                    frame_with_boxes = frame.copy()
                    
                    # Рисуем красные рамки вокруг обнаруженных объектов
                    red_color = (0, 0, 255)  # Красный цвет в формате BGR
                    box_thickness = 3  # Толщина линии рамки
                    
                    for obj in results['objects_info']:
                        x1, y1, x2, y2 = obj['coordinates']
                        # Преобразуем координаты в целые числа
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        # Рисуем красный прямоугольник
                        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), red_color, box_thickness)
                    
                    # Конвертируем кадр с рамками в JPEG формат для отправки
                    # Используем cv2.imencode для кодирования изображения в память
                    success, buffer = cv2.imencode('.jpg', frame_with_boxes)
                    
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
    
    def process_active_learning(self, frame, results) -> bool:
        """
        Обрабатывает кадр для Active Learning: сохраняет кадры с низкой/средней 
        уверенностью детекции для последующей разметки.
        
        Стратегия "Двух папок":
            - "maybe" (10-40%): Галлюцинации или новые условия. Много мусора, 
              но могут быть ценные кадры с новых камер.
            - "almost" (40-60%): "Почти угадал". Золотой фонд для разметки.
        
        Для защиты от мусора используется таймер - кадры сохраняются не чаще 
        чем раз в al_save_interval секунд (по умолчанию 30).
        
        Args:
            frame: Кадр изображения (numpy array)
            results: Словарь с результатами детекции, содержащий 'objects_info'
                    с полями 'confidence' и 'coordinates' для каждого объекта
        
        Returns:
            bool: True если кадр был сохранён, False если нет
        
        Пример использования:
            frame, results = show.frame(camera, shelf_coordinates)
            if show.process_active_learning(frame, results):
                print("Кадр сохранён для разметки")
        
        Настройка параметров:
            show.al_save_interval = 60  # Сохранять не чаще раза в минуту
            show.al_conf_low = 0.15     # Нижняя граница уверенности
            show.al_conf_mid = 0.35     # Граница между "maybe" и "almost"
            show.al_conf_high = 0.55    # Верхняя граница уверенности
            show.active_learning_dir = "my_dataset"  # Директория для сохранения
        """
        if frame is None or results is None:
            return False
        
        current_time = time.time()
        
        # Проверяем, прошло ли достаточно времени с последнего сохранения
        if current_time - self.last_al_save_time < self.al_save_interval:
            return False
        
        objects_info = results.get('objects_info', [])
        if not objects_info:
            return False
        
        # Ищем объект с уверенностью в нужном диапазоне
        target_object = None
        for obj in objects_info:
            conf = obj.get('confidence', 0)
            if self.al_conf_low < conf < self.al_conf_high:
                target_object = obj
                break
        
        if target_object is None:
            return False
        
        conf = target_object['confidence']
        
        # Определяем подпапку: "maybe" для низкой уверенности, "almost" для средней
        subfolder = "maybe" if conf < self.al_conf_mid else "almost"
        
        # Создаём директории если не существуют
        save_dir = os.path.join(self.active_learning_dir, subfolder)
        os.makedirs(save_dir, exist_ok=True)
        
        # Формируем имя файла с временной меткой и уверенностью
        timestamp = int(current_time)
        conf_str = f"{conf:.2f}".replace(".", "_")  # 0.35 -> 0_35 для имени файла
        filename = f"frame_{timestamp}_conf{conf_str}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        # Создаем копию кадра и рисуем рамки вокруг всех обнаруженных объектов
        frame_with_boxes = frame.copy()
        
        for obj in objects_info:
            x1, y1, x2, y2 = obj['coordinates']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            obj_conf = obj.get('confidence', 0)
            
            # Цвет зависит от уверенности: красный для низкой, оранжевый для средней, зелёный для высокой
            if obj_conf < self.al_conf_mid:
                color = (0, 0, 255)      # Красный (BGR)
            elif obj_conf < self.al_conf_high:
                color = (0, 165, 255)    # Оранжевый (BGR)
            else:
                color = (0, 255, 0)      # Зелёный (BGR)
            
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            # Добавляем текст с уверенностью
            label = f"{obj_conf:.2f}"
            cv2.putText(frame_with_boxes, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Сохраняем кадр
        success = cv2.imwrite(filepath, frame_with_boxes)
        
        if success:
            self.last_al_save_time = current_time
            print(f"[Active Learning] Saved: {filepath} (conf: {conf:.2f}, folder: {subfolder})")
            return True
        else:
            print(f"[Active Learning] Ошибка сохранения: {filepath}")
            return False
    
    def run_active_learning(self, camera: Camera, shelf_coordinates: list, 
                           save_interval: int = 30, conf_range: tuple = (0.10, 0.60)):
        """
        Запускает непрерывный цикл сбора данных для Active Learning.
        
        Этот метод анализирует видеопоток и автоматически сохраняет кадры,
        на которых модель не уверена в детекции. Это позволяет собирать
        данные для дообучения модели на сложных случаях.
        
        Args:
            camera: Экземпляр класса Camera для получения кадров
            shelf_coordinates: Список координат полок [(x1, y1, x2, y2), ...]
            save_interval: Минимальный интервал между сохранениями в секундах (по умолчанию 30)
            conf_range: Кортеж (min_conf, max_conf) - диапазон уверенности для сохранения
        
        Структура сохранённых данных:
            to_label/
            ├── maybe/       # Уверенность 10-40% (возможно мусор, требует проверки)
            │   ├── frame_1706000000_conf0_15.jpg
            │   └── ...
            └── almost/      # Уверенность 40-60% (золотой фонд для разметки)
                ├── frame_1706000100_conf0_45.jpg
                └── ...
        
        Рекомендации:
            - Папку "almost" размечайте в первую очередь
            - Папку "maybe" просматривайте раз в неделю для поиска ценных кадров
            - При дообучении модели смешивайте новые данные со старыми!
        
        Пример использования:
            model = YOLO('model.pt')
            camera = Camera(ip_camera='192.168.1.100')
            show = ShowPicture(model=model, active_learning_dir='training_data')
            
            # Запуск с настройками по умолчанию
            show.run_active_learning(camera, shelf_coordinates)
            
            # Или с кастомными настройками
            show.run_active_learning(camera, shelf_coordinates, 
                                    save_interval=60, 
                                    conf_range=(0.15, 0.50))
        """
        # Применяем настройки
        self.al_save_interval = save_interval
        self.al_conf_low, self.al_conf_high = conf_range
        
        print("=" * 60)
        print("Запуск сбора данных для Active Learning")
        print("=" * 60)
        print(f"Директория сохранения: {self.active_learning_dir}/")
        print(f"  ├── maybe/   (уверенность {self.al_conf_low*100:.0f}%-{self.al_conf_mid*100:.0f}%)")
        print(f"  └── almost/  (уверенность {self.al_conf_mid*100:.0f}%-{self.al_conf_high*100:.0f}%)")
        print(f"Интервал сохранения: {save_interval} сек")
        print("Для остановки нажмите Ctrl+C")
        print("=" * 60 + "\n")
        
        saved_count = {"maybe": 0, "almost": 0}
        
        try:
            while True:
                try:
                    # Получаем кадр и результаты
                    frame, results = self.frame(camera=camera, shelf_coordinates=shelf_coordinates)
                    
                    if frame is None or results is None:
                        time.sleep(1)
                        continue
                    
                    # Пробуем сохранить для Active Learning
                    if self.process_active_learning(frame, results):
                        # Подсчитываем сохранённые кадры по категориям
                        for obj in results.get('objects_info', []):
                            conf = obj.get('confidence', 0)
                            if self.al_conf_low < conf < self.al_conf_high:
                                if conf < self.al_conf_mid:
                                    saved_count["maybe"] += 1
                                else:
                                    saved_count["almost"] += 1
                                break
                    
                    # Небольшая пауза для снижения нагрузки
                    time.sleep(save_interval)
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"[Active Learning] Ошибка: {e}")
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            print("\n" + "=" * 60)
            print("Остановка сбора данных Active Learning")
            print(f"Сохранено кадров:")
            print(f"  maybe:  {saved_count['maybe']}")
            print(f"  almost: {saved_count['almost']}")
            print(f"  всего:  {saved_count['maybe'] + saved_count['almost']}")
            print("=" * 60)
        finally:
            camera.release()
            print("Камера отключена")