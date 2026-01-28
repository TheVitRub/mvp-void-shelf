"""
Модуль для отслеживания объектов (трекинга) с использованием YOLO и ByteTrack.

Этот модуль предоставляет класс PersonDetector для отслеживания объектов
в видеопотоке с сохранением идентификаторов между кадрами. Использует
встроенный трекер ByteTrack от Ultralytics для стабильного трекинга
даже при перекрытиях объектов.

Основные возможности:
    - Загрузка YOLO модели для детекции объектов
    - Трекинг объектов с сохранением ID между кадрами
    - Использование алгоритма ByteTrack для надежного трекинга
    - Возврат координат, ID трека и уверенности для каждого объекта

Классы:
    PersonDetector: Класс для детекции и трекинга объектов

Методы PersonDetector:
    - __init__(): Инициализация с загрузкой модели YOLO
    - track(frame): Выполняет трекинг объектов на кадре

Возвращаемый формат:
    Список объектов: [[x1, y1, x2, y2, track_id, confidence], ...]
    где:
        - x1, y1, x2, y2: координаты bounding box
        - track_id: уникальный идентификатор трека
        - confidence: уверенность детекции (0-1)

Использование:
    from MVP.track.track import PersonDetector
    import cv2
    
    detector = PersonDetector()
    frame = cv2.imread('image.jpg')
    tracked_objects = detector.track(frame)
    
    for x1, y1, x2, y2, track_id, conf in tracked_objects:
        print(f"Объект {track_id}: ({x1}, {y1}) -> ({x2}, {y2}), уверенность: {conf:.2f}")

Примечание:
    Модуль настроен на использование ByteTrack трекера, который лучше работает
    с перекрытиями объектов по сравнению с BoTSORT.

Автор: [Ваше имя]
Дата: 2026-01-27
"""

import cv2
from ultralytics import YOLO

from config import YOLO_MODEL, DETECTION_IMG_SIZE, CONFIDENCE_THRESHOLD


class PersonDetector:
    def __init__(self):
        print(f"Загрузка модели: {YOLO_MODEL}...")
        self.model = YOLO(YOLO_MODEL)
        print("Модель готова.")

    def track(self, frame):
        """
        Использует встроенный трекер Ultralytics (ByteTrack).
        Возвращает список: [[x1, y1, x2, y2, track_id, conf], ...]
        """
        results = self.model.track(
            source=frame,
            persist=True,  # Важно для сохранения ID между кадрами
            imgsz=DETECTION_IMG_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            tracker="bytetrack.yaml",  # ByteTrack лучше работает с перекрытиями, чем BoTSORT
            verbose=False
        )[0]

        tracked_objects = []

        if results.boxes.id is not None:
            # Получаем координаты, конфиденс и ID
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                tracked_objects.append([x1, y1, x2, y2, track_id, conf])

        return tracked_objects