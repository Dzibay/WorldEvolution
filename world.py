import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from collections import Counter
import time # Для замера времени
from biomes_data import RAW_BIOME_HEX_MAP, hex_to_rgb


# ==========================
# 1. Загрузка НОВЫХ КАРТ
# ==========================
try:
    # Карта 1: Карта БИОМОВ (ЦВЕТ)
    # Используем твой .jpg файл, новый алгоритм с ним справится
    img_path_color = "data/biomes.jpg" 
    img_color = Image.open(img_path_color).convert("RGB")
    
    # Карта 2: Карта ВЫСОТ (ДАННЫЕ)
    img_path_height = "data/topography_map_3.png"
    img_height = Image.open(img_path_height).convert("RGB")

except FileNotFoundError as e:
    print(f"Ошибка: не найден файл. {e}")
    print("Пожалуйста, убедись, что 'biomes.jpg' и 'topography_map.png' находятся в той же папке.")
    exit()

# Получаем базовые размеры из карты биомов
img_data_color = np.array(img_color)
h, w, _ = img_data_color.shape

print(f"Размер карты биомов: {w}x{h} пикселей")
print(f"Размер карты высот: {img_height.width}x{img_height.height} пикселей")

# ==========================
# 1.5. Синхронизация карт
# ==========================
print(f"Синхронизирую карту высот до {w}x{h}...")
img_height_resized = img_height.resize((w, h), Image.Resampling.BILINEAR)

# Используем КРАСНЫЙ канал карты высот (т.к. она Ч/Б, R=G=B)
elevation_data_raw = np.array(img_height_resized.getchannel('R')) 

print("Синхронизация завершена.")

# ==========================
# 2. Параметры клеток
# ==========================
CELL_PIXELS = 1
nx, ny = w // CELL_PIXELS, h // CELL_PIXELS

print(f"Размер сетки: {nx}x{ny} клеток ({CELL_PIXELS}x{CELL_PIXELS} пикселей на клетку)")

# ==========================
# 3. НОВЫЙ ГЕНЕРАТОР БИОМОВ
# ==========================


# 3.1. Создаем два словаря:
BIOME_COLOR_MAP = {} # (R,G,B) -> "Name"
BIOME_DATA = {}      # "Name" -> {vis_color: (R,G,B)}

print("Генерирую палитру биомов...")
for name, hex_code in RAW_BIOME_HEX_MAP.items():
    rgb_tuple = hex_to_rgb(hex_code)
    BIOME_COLOR_MAP[rgb_tuple] = name
    BIOME_DATA[name] = {
        "vis_color": rgb_tuple 
    }

# *** НОВАЯ ФУНКЦИЯ КЛАССИФИКАЦИИ ***
def classify_biomes_by_closest_color(img_data_color, color_map):
    """
    НОВЫЙ КЛАССИФИКАТОР: Находит БЛИЖАЙШИЙ цвет из палитры
    по квадрату евклидова расстояния.
    Это решает проблему с артефактами сжатия JPG/WEBP.
    """
    h, w, _ = img_data_color.shape
    
    # 1. Создаем массивы для сопоставления
    # (N,) массив имен, где N - кол-во биомов
    palette_names = np.array(list(color_map.values())) 
    # (N, 3) массив цветов. Важно: float для вычитания
    palette_colors = np.array(list(color_map.keys()), dtype=float) 
    
    # 2. "Выравниваем" изображение и конвертируем в float
    # M = h*w (кол-во пикселей)
    img_flat = img_data_color.reshape(-1, 3).astype(float) # (M, 3)
    
    # 3. Магия NumPy: вычисляем квадрат евклидова расстояния
    # Вычитаем (M, 1, 3) из (1, N, 3) -> получаем (M, N, 3)
    # Это массив (R_diff, G_diff, B_diff) для каждого пикселя до каждого цвета палитры
    diff = img_flat[:, np.newaxis, :] - palette_colors[np.newaxis, :, :]
    
    # 4. Возводим в квадрат
    # (M, N, 3) -> (M, N, 3)
    sq_diff = diff**2
    
    # 5. Суммируем квадраты по R,G,B осям
    # (M, N, 3) -> (M, N)
    # Это (R_diff^2 + G_diff^2 + B_diff^2) для каждого пикселя до каждого цвета палитры
    dist_sq = np.sum(sq_diff, axis=2)
    
    # 6. Находим индекс МИНИМАЛЬНОГО расстояния для каждого пикселя
    # (M, N) -> (M,)
    # np.argmin находит индекс (т.е. номер биома) с наименьшим dist_sq
    indices = np.argmin(dist_sq, axis=1)
    
    # 7. Создаем карту имен по индексам
    biome_map_flat = palette_names[indices]
    
    # 8. Возвращаем карте исходную форму
    # "Unknown" больше не нужен, т.к. *каждый* пиксель найдет "ближайшего"
    return biome_map_flat.reshape(h, w)


# ==========================
# 4. Обработка карты (классификация)
# ==========================
print("Начинаю классификацию биомов (это может занять несколько секунд)...")
start_time = time.time()

# --- ИЗМЕНЕНИЕ: Вызываем НОВЫЙ "умный" классификатор ---
biome_map = classify_biomes_by_closest_color(img_data_color, BIOME_COLOR_MAP)

print(f"Классификация завершена за {time.time() - start_time:.2f} сек.")

# Проверка на "Unknown" больше не нужна, т.к. все пиксели будут классифицированы

# ==========================
# 5. Создание карты клеток (объединение данных)
# ==========================
print("Начинаю обработку клеток (объединение карт)...")
start_time = time.time()

# Задаем реальные мин/макс высоты для масштабирования
MIN_ELEVATION_METERS = -11000 # Марианская впадина
MAX_ELEVATION_METERS = 8848  # Эверест

# <<< ИЗМЕНЕНИЕ 1: Добавляем эту строку
# Визуально определяем, что уровень моря (0м) на карте 
# соответствует пикселю с яркостью ~90.
SEA_LEVEL_RAW_VALUE = 90.0

cells_json = []
vis_image = np.zeros((ny, nx, 3), dtype=np.uint8) 

for i in range(nx):
    for j in range(ny):
        x0, y0 = i * CELL_PIXELS, j * CELL_PIXELS
        x1, y1 = x0 + CELL_PIXELS, y0 + CELL_PIXELS
        
        # ... (шаги 1-4 не меняются) ...
        cell_biome_slice = biome_map[y0:y1, x0:x1]
        most_common_biome = Counter(cell_biome_slice.flatten()).most_common(1)[0][0]
        cell_elev_slice_raw = elevation_data_raw[y0:y1, x0:x1]
        avg_elev_raw = float(np.mean(cell_elev_slice_raw))
        
        # 5. Масштабируем высоту в метры
        
        # <<< ИЗМЕНЕНИЕ 2: Заменяем старую строку np.interp на эту
        # Используем НЕЛИНЕЙНОЕ масштабирование с 3 точками
        scaled_elev = np.interp(
            avg_elev_raw, 
            [0, SEA_LEVEL_RAW_VALUE, 255], 
            [MIN_ELEVATION_METERS, 0, MAX_ELEVATION_METERS]
        )

        # 6. Получаем доп. данные по БИОМУ (только цвет)
        data = BIOME_DATA[most_common_biome]
        
        # --- ИЗМЕНЕНИЕ: Добавляем биом и ВЫСОТУ в JSON ---
        cells_json.append({
            "i": i, "j": j,
            "biome": most_common_biome,
            "elevation_m": round(scaled_elev, 2) # Добавляем высоту
        })
        
        # 7. Заполняем пиксель для итоговой картинки
        vis_image[j, i] = data["vis_color"]

print(f"Обработка клеток завершена за {time.time() - start_time:.2f} сек.")

# ==========================
# 6. Визуализация
# ==========================
plt.figure(figsize=(14, 7))
plt.imshow(vis_image, interpolation='none') 
plt.title(f"World Map (Optimized - {nx}x{ny} cells) - Biome Types (Closest-Match)")
plt.axis('off') 
plt.show()

# ==========================
# 7. Сохранение JSON
# ==========================
output_filename = "world_cells.json"
print(f"Сохраняю JSON в файл {output_filename}...") 
with open(output_filename, "w") as f:
    json.dump(cells_json, f, indent=2)

print("Карта успешно обработана и сохранена!")
print("Файл JSON теперь содержит биомы и среднюю высоту ('elevation_m').")
print("Проблема с 'неизвестными' пикселями решена.")