

from io import BytesIO
import time
from typing import Any, List
import requests
from api.camera_data_class import CameraData
from config import API_BASE_URL, ID_STORE


class ApiClass:
    def __init__(self):
        self.url = API_BASE_URL
    

    def send_camera_data(self, files: dict[str, tuple[str, BytesIO, str]], data: dict[str, Any], void_percentage: int, ip_camera:int):
                    api_url = self.url + '/entrance/photo'

                    # Отправляем POST запрос на API
                    try:
                        response = requests.post(api_url, files=files, data=data, timeout=10)
                        
                        if response.status_code == 200:
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Данные успешно отправлены: "
                                  f"ID магазина={ID_STORE}, Наполнение={void_percentage}%, IP={ip_camera}")
                        else:
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ошибка отправки данных: "
                                  f"HTTP {response.status_code} - {response.text}. response: {response}")
                    
                    except requests.exceptions.RequestException as e:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ошибка подключения к API: {e}")
                    
    def get_camera_data(self) -> List[CameraData]:
        while True:
            try:
                url = self.url + '/entrance/camera-data/' + str(ID_STORE)
                response = requests.get(url=url)
                
                if response.status_code == 200:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Данные магазина {ID_STORE} успешно получены")
                    # Парсим JSON ответ
                    data = response.json()
                    # Преобразуем каждый элемент в объект CameraData
                    camera_data_list = []
                    for item in data:
                        camera_data = CameraData(
                            name_camera=item.get('name_camera', ''),
                            ip_camera=item.get('ip_camera', ''),
                            shelf_coordinates=item.get('shelf_coordinates', {})
                        )
                        camera_data_list.append(camera_data)
                    return camera_data_list
                else:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Данные магазина {ID_STORE} не получены. HTTP {response.status_code}")
                    time.sleep(10)
                    
            except requests.exceptions.RequestException as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ошибка подключения к API: {e}")
                time.sleep(10)



api_class = ApiClass()