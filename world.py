import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from collections import Counter
import time # Добавим для замера времени

# ==========================
# 1. Загрузка карты
# ==========================
try:
    # Я убрал твой специфичный путь, положи world_map.jpeg рядом со скриптом
    img_path = "data/world_5.jpeg"
    img = Image.open(img_path).convert("RGB")
except FileNotFoundError:
    print(f"Ошибка: не найден файл {img_path}")
    print("Пожалуйста, убедись, что 'world_map.jpeg' находится в той же папке, что и скрипт.")
    exit()
    
img_data = np.array(img)
h, w, _ = img_data.shape

# ==========================
# 2. Параметры клеток
# ==========================
# УВЕЛИЧИВАЕМ РАЗМЕР КЛЕТКИ! 
# CELL_PIXELS = 1 был главной причиной медлительности.
CELL_PIXELS = 1 
nx, ny = w // CELL_PIXELS, h // CELL_PIXELS

print(f"Размер карты: {w}x{h} пикселей")
print(f"Размер сетки: {nx}x{ny} клеток ({CELL_PIXELS}x{CELL_PIXELS} пикселей на клетку)")

# ==========================
# 3. Новая функция сопоставления (векторизованная)
# ==========================

# Определяем "эталонные" (прототипные) цвета для каждого типа местности
# и связанные с ними данные. Это намного надежнее, чем жесткие диапазоны.
TERRAIN_DATA = {
    "Ocean": {
        "prototype_color": (0, 50, 150), # Темно-синий
        "elevation": -300,
        "soil": 0.0, "water": 1.0, "hab": 0.0,
        "vis_color": (0, 0, 200) # Синий для вывода
    },
    "Shelf": {
        "prototype_color": (110, 195, 215), # Светло-голубой
        "elevation": -100,
        "soil": 0.0, "water": 1.0, "hab": 0.1,
        "vis_color": (0, 200, 255) # Голубой
    },
    "Lowlands": {
        "prototype_color": (90, 170, 75), # Зеленый
        "elevation": 250,
        "soil": 0.8, "water": 0.5, "hab": 0.7,
        "vis_color": (0, 128, 0) # Зеленый
    },
    "Plateau": {
        "prototype_color": (205, 190, 115), # Желтый/Бежевый
        "elevation": 1200,
        "soil": 0.3, "water": 0.3, "hab": 0.3,
        "vis_color": (230, 230, 0) # Желтый
    },
    "Mountains": {
        "prototype_color": (145, 100, 70), # Коричневый
        "elevation": 3500,
        "soil": 0.2, "water": 0.2, "hab": 0.2,
        "vis_color": (139, 69, 19) # Коричневый
    },
    "Highlands/Ice": {
        "prototype_color": (230, 230, 230), # Белый/Светло-серый
        "elevation": 5000,
        "soil": 0.1, "water": 0.0, "hab": 0.0,
        "vis_color": (255, 255, 255) # Белый
    }
}
# "Unknown" нам больше не нужен, т.к. каждый пиксель найдет "ближайший" цвет

# --- Магия NumPy ---
def classify_pixels(img_data, terrain_info):
    """
    Классифицирует *все* пиксели на карте за один проход,
    используя векторизованные операции NumPy.
    """
    # Создаем массив эталонных цветов (n_terrains, 3)
    prototypes = np.array([info["prototype_color"] for info in terrain_info.values()])
    
    # Конвертируем для вычислений
    img_float = img_data.astype(np.float32)
    
    # Вычисляем "расстояние" от каждого пикселя до каждого эталона
    # (h, w, 1, 3) - (1, 1, n_terrains, 3) -> (h, w, n_terrains, 3)
    distances_sq = np.sum(
        (img_float[:, :, np.newaxis, :] - prototypes[np.newaxis, np.newaxis, :, :])**2,
        axis=3
    )
    
    # Находим индекс *минимального* расстояния для каждого пикселя
    # Это массив (h, w) с индексами (0=Ocean, 1=Shelf, и т.д.)
    indices_map = np.argmin(distances_sq, axis=2)
    
    # Создаем карты для имен и высот
    terrain_names = list(terrain_info.keys())
    name_lookup = np.array(terrain_names)
    elevation_lookup = np.array([info["elevation"] for info in terrain_info.values()])

    # Используем 'indices_map' для быстрой "подстановки" значений
    terrain_map = name_lookup[indices_map]
    elevation_map = elevation_lookup[indices_map]
    
    return terrain_map, elevation_map

# ==========================
# 4. Обработка карты (быстро)
# ==========================
print("Начинаю классификацию пикселей (это быстро)...")
start_time = time.time()

# Это самая "тяжелая" часть, но она выполняется за < 1 секунды
terrain_map, elevation_map = classify_pixels(img_data, TERRAIN_DATA)

print(f"Классификация завершена за {time.time() - start_time:.2f} сек.")

# ==========================
# 5. Создание карты клеток (теперь тоже быстро)
# ==========================
print("Начинаю обработку клеток...")
start_time = time.time()

cells_json = []
# Создаем массив для быстрой визуализации
vis_image = np.zeros((ny, nx, 3), dtype=np.uint8) 

for i in range(nx):
    for j in range(ny):
        x0, y0 = i * CELL_PIXELS, j * CELL_PIXELS
        x1, y1 = x0 + CELL_PIXELS, y0 + CELL_PIXELS
        
        # 1. Берем срез (10x10) из *уже готовых* карт
        cell_terrain_slice = terrain_map[y0:y1, x0:x1]
        cell_elev_slice = elevation_map[y0:y1, x0:x1]
        
        # 2. Находим доминирующий тип местности в клетке
        # (flatten() превращает 2D срез в 1D массив)
        most_common_terrain = Counter(cell_terrain_slice.flatten()).most_common(1)[0][0]
        
        # 3. Считаем среднюю высоту по клетке
        avg_elev = float(np.mean(cell_elev_slice))

        # 4. Получаем данные для этого типа местности
        data = TERRAIN_DATA[most_common_terrain]
        
        cells_json.append({
            "i": i, "j": j,
            "terrain_type": most_common_terrain,
            "elevation": avg_elev,
            "soil_quality": data["soil"],
            "water_access": data["water"],
            "habitability": data["hab"],
            "color": list(data["vis_color"])  # <--- ИЗМЕНЕНИЕ: Сохраняем цвет
        })
        
        # 5. Заполняем пиксель для итоговой картинки
        vis_image[j, i] = data["vis_color"]

print(f"Обработка клеток завершена за {time.time() - start_time:.2f} сек.")

# ==========================
# 6. Визуализация (МГНОВЕННАЯ)
# ==========================
plt.figure(figsize=(14, 7))
# plt.imshow() рисует все изображение за один раз
plt.imshow(vis_image, interpolation='none') # 'none' для четких пикселей
plt.title(f"World Map (Optimized - {nx}x{ny} cells)")
plt.axis('off') # Отключаем оси
plt.show()

# ==========================
# 7. Сохранение JSON
# ==========================
print("Сохраняю JSON...") # <--- ИЗМЕНЕНИЕ: Раскомментировано
with open("world_cells_from_image_optimized.json", "w") as f:
    json.dump(cells_json, f, indent=2)

print("Карта успешно обработана и сохранена!")