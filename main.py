import json
import numpy as np
import pyvista as pv

# --- НОВЫЕ ИМПОРТЫ ---
from biomes_data import RAW_BIOME_HEX_MAP, hex_to_rgb

# ===========================
# 1. Загружаем данные
# ===========================

# --- ИСПРАВЛЕНИЕ: Используем JSON, который мы сгенерировали ---
json_filename = "world_cells.json" 
try:
    with open(json_filename, "r") as f:
        cells = json.load(f)
except FileNotFoundError:
    print(f"Ошибка: Файл {json_filename} не найден!")
    print("Убедись, что он лежит в той же папке, что и скрипт.")
    exit()

nx = max(c["i"] for c in cells) + 1
ny = max(c["j"] for c in cells) + 1
print(f"Загружено {len(cells)} клеток (сетка {nx}x{ny})")

# ===========================
# 1.5. Создаем словарь цветов
# ===========================
print("Создаю палитру цветов из biomes_data...")
BIOME_RGB_MAP = {}
for name, hex_code in RAW_BIOME_HEX_MAP.items():
    BIOME_RGB_MAP[name] = hex_to_rgb(hex_code)

# Цвет "по умолчанию" на случай ошибки
UNKNOWN_COLOR = (255, 0, 255) # Розовый

# ===========================
# 2. Преобразуем в XYZ
# ===========================
print("Преобразую 2D сетку в 3D сферу...")
radius_earth = 6371.0  # Радиус Земли (в км)

# --- УЛУЧШЕНИЕ: Усиление рельефа ---
# (Реальный рельеф 8км на фоне 6371км не виден. Усилим его в 50 раз)
elevation_exaggeration = 50.0 

# Создаем массив, чтобы потом было удобно формировать треугольники
grid_points = np.zeros((nx, ny, 3))
grid_colors = np.zeros((nx, ny, 3), dtype=np.uint8)

for c in cells:
    i, j = c["i"], c["j"]
    
    # 1. Получаем координаты (Долгота и Широта)
    # theta (Longitude)
    theta = (i / (nx - 1)) * 2 * np.pi
    # phi (Latitude)
    phi = np.pi/2 - (j / (ny - 1)) * np.pi
    
    # 2. Получаем радиус (Высоту)
    r = (radius_earth + (c["elevation_m"] / 1000.0) * elevation_exaggeration)  if c["elevation_m"] > 0 else radius_earth

    # 3. Конвертируем в XYZ
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)

    grid_points[i, j] = [x, y, z]
    
    # 4. --- ИСПРАВЛЕНИЕ: Получаем цвет по имени биома ---
    biome_name = c["biome"]
    grid_colors[i, j] = BIOME_RGB_MAP.get(biome_name, UNKNOWN_COLOR)

# ===========================
# 3. Создаем треугольники для сетки
# ===========================
print("Создаю 'лица' (треугольники) для 3D модели...")
faces = []
for i in range(nx - 1):
    for j in range(ny - 1):
        # Индексы 4-х углов клетки
        p0 = i * ny + j
        p1 = (i + 1) * ny + j
        p2 = (i + 1) * ny + (j + 1)
        p3 = i * ny + (j + 1)

        # 2 треугольника
        faces.append([3, p0, p1, p2])
        faces.append([3, p0, p2, p3])

# Выравниваем массивы
points_flat = grid_points.reshape(-1, 3)
colors_flat = grid_colors.reshape(-1, 3)
faces_flat = np.hstack(faces)

# ===========================
# 4. Создаем PyVista mesh
# ===========================
print("Создаю 3D mesh в PyVista...")
mesh = pv.PolyData(points_flat, faces=faces_flat)

# Присваиваем цвета точкам (а не 'лицам')
mesh.point_data['colors'] = colors_flat

print("Запускаю 3D-просмотрщик...")
plotter = pv.Plotter(window_size=[1200, 900])

# 'scalars' - это данные для окрашивания, 'rgb=True' говорит, что это цвета
plotter.add_mesh(mesh, scalars='colors', rgb=True, smooth_shading=True)

plotter.add_axes()

# Улучшим камеру
plotter.camera_position = 'xy'
plotter.camera.zoom(1.2)

# Добавляем фон (темный космос)
plotter.set_background('black')

plotter.show()
print("Готово.")