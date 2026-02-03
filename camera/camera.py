"""
Модуль для работы с IP-камерами через RTSP протокол.

Этот модуль предоставляет класс Camera для подключения к IP-камерам
и получения видеопотока через RTSP. Поддерживает автоматическое
переподключение при потере соединения и оптимизирован для работы
в условиях нестабильной сети.

Основные возможности:
    - Подключение к IP-камерам через RTSP протокол
    - Автоматическое переподключение при потере соединения
    - Настройка параметров подключения через переменные окружения
    - Оптимизация для работы в условиях плохого интернета (TCP режим)
    - Чтение отдельных кадров из видеопотока

Классы:
    Camera: Класс для работы с IP-камерами

Методы Camera:
    - __init__(ip_camera): Инициализация подключения к камере
    - read_frame(): Чтение одного кадра из видеопотока
    - read_fresh_frame(): Чтение свежего кадра с очисткой буфера (для редких чтений)
    - release(): Закрытие соединения с камерой

Переменные окружения (.env):
    CAMERA_IP: IP адрес камеры по умолчанию
    CAMERA_PORT: Порт RTSP (по умолчанию 554)
    CAMERA_LOGIN: Логин для доступа к камере
    CAMERA_PASSWORD: Пароль для доступа к камере
    CAMERA_STREAM_PATH: Путь к RTSP потоку (например, /h264)

Формат RTSP URL:
    rtsp://{login}:{password}@{ip}:{port}{path}

Использование:
    from MVP.camera.camera import Camera
    
    # Использование IP из переменных окружения
    camera = Camera()
    
    # Или указание IP напрямую
    camera = Camera(ip_camera='192.168.1.100')
    
    # Чтение кадров
    while True:
        frame = camera.read_frame()
        if frame is not None:
            # Обработка кадра
            pass
    
    # Закрытие соединения
    camera.release()

Примечание:
    Модуль автоматически переподключается при потере кадров или разрыве соединения.
    Использует TCP режим для стабильности при плохом интернете.

Автор: [Ваше имя]
Дата: 2026-01-27
"""

import os
import cv2
from dotenv import load_dotenv

load_dotenv()
# ОТКЛЮЧАЕМ логирование OpenCV и FFmpeg (чтобы не забить диск)
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

class Camera:
    def __init__(self,ip_camera:str = None):
        self.port = os.getenv('CAMERA_PORT', '554')
        self.password = os.getenv('CAMERA_PASSWORD')
        self.login = os.getenv('CAMERA_LOGIN')
        self.path = os.getenv('CAMERA_STREAM_PATH')
        # Явно проверяем, что ip_camera не None, чтобы переданный параметр всегда имел приоритет
        if ip_camera is not None:
            self.ip_camera = ip_camera
        else:
            self.ip_camera = os.getenv('CAMERA_IP')
        print(f"Переданный ip_camera: {ip_camera}")
        print(f"Используемый self.ip_camera: {self.ip_camera}")
        self.cap = None

        # Собираем URL
        self.rts_url = f'rtsp://{self.login}:{self.password}@{self.ip_camera}:{self.port}{self.path}'

        # Опции для стабильности при плохом интернете
        # Устанавливаем таймаут на открытие (5 секунд)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000000"

        self._connect()

    def _connect(self):
        """Внутренний метод для (пере)подключения"""
        if self.cap is not None:
            self.cap.release()
            import time
            time.sleep(0.1)  # Даем ОС время закрыть сокеты

        print(f"Попытка подключения к RTSP: {self.ip_camera}")
        # Явно указываем CAP_FFMPEG для стабильности на Linux
        self.cap = cv2.VideoCapture(self.rts_url, cv2.CAP_FFMPEG)

        # Для стабильности принудительно используем TCP
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self._connect()
            return None

        ret, frame = self.cap.read()

        if not ret:
            print("Потеря кадров. Попытка переподключения...")
            self._connect()
            return None

        return frame
    
    def read_fresh_frame(self, max_attempts=3):
        """
        Читает свежий кадр. Если соединение есть, очищает буфер.
        Если нет или кадр битый — переподключается.
        """
        import time
        import numpy as np
        
        for attempt in range(max_attempts):
            # 1. Если захват не открыт, пробуем подключиться
            if self.cap is None or not self.cap.isOpened():
                self._connect()
            
            if self.cap is None or not self.cap.isOpened():
                print(f"Попытка {attempt + 1}/{max_attempts}: Нет связи с камерой")
                time.sleep(1)
                continue
            
            # 2. Очищаем буфер RTSP (выбрасываем старые кадры)
            # grab() работает очень быстро, так как не декодирует изображение
            for _ in range(30):
                if not self.cap.grab():
                    break
            
            # 3. Читаем актуальный кадр
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                # Проверка на "битость" (черная полоса внизу)
                height = frame.shape[0]
                bottom_part = frame[int(height * 0.9):, :]
                if np.sum(bottom_part) > 1000:
                    return frame
                else:
                    print(f"Попытка {attempt + 1}: Битый кадр. Переподключаемся...")
                    self._connect()
            else:
                print(f"Попытка {attempt + 1}: Не удалось прочитать кадр. Переподключаемся...")
                self._connect()
            
            time.sleep(0.5)
        
        return None
    def release(self):
        if self.cap:
            self.cap.release()