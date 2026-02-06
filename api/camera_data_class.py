from typing import Union, Dict, List



class CameraData:
    def __init__(self, name_camera: str, ip_camera: str, shelf_coordinates: Union[Dict, List]):
        self.name_camera = name_camera
        self.ip_camera = ip_camera
        # Обрабатываем разные форматы shelf_coordinates
        if isinstance(shelf_coordinates, dict):
            # Если это словарь с ключом 'shelves'
            if 'shelves' in shelf_coordinates:
                self.shelf_coordinates = [tuple(shelf) for shelf in shelf_coordinates['shelves']]
            else:
                # Если это пустой словарь или другой формат
                self.shelf_coordinates = []
        elif isinstance(shelf_coordinates, list):
            # Если это уже список координат
            self.shelf_coordinates = [tuple(shelf) if not isinstance(shelf, tuple) else shelf for shelf in shelf_coordinates]
        else:
            self.shelf_coordinates = []