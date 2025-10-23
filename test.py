import json
import numpy as np
import pyvista as pv
import matplotlib.colors as mcolors


# ==========================
# 1. Константы и Настройки
# ==========================

JSON_FILE = "world_cells_from_image_optimized.json"

# Словарь цветов для PyVista (HEX)
TERRAIN_COLORS = {
    "Ocean": "#00008B",      # Темно-синий
    "Shelf": "#00BFFF",      # Голубой
    "Lowlands": "#006400",   # Темно-зеленый
    "Plateau": "#DAA520",    # Желтый (Золотой)
    "Mountains": "#8B4513",  # Коричневый
    "Highlands/Ice": "#FFFFFF", # Белый
    "Unknown": "#FF00FF"     # Розовый (на всякий случай)
}

# Базовый "радиус" нашей планеты для графика
BASE_RADIUS = 100

# Во сколько раз преувеличить высоту, чтобы она была видна
# (иначе 5км гор не будут видны на фоне 100 "радиуса")
ELEVATION_SCALE = 0.05

# ==========================
# 2. Загрузка и обработка данных
# ==========================
def load_and_process_data(json_path):
    """Загружает JSON и конвертирует 2D-сетку в 3D-точки и цвета."""
    
    print(f"Загрузка данных из {json_path}...")
    try:
        with open(json_path, 'r') as f:
            cells = json.load(f)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {json_path} не найден.")
        print("Пожалуйста, сначала запустите основной скрипт обработки карты.")
        return None, None

    if not cells:
        print("ОШИБКА: JSON-файл пуст.")
        return None, None

    # Находим максимальные i и j для нормализации
    nx = max(c['i'] for c in cells) + 1
    ny = max(c['j'] for c in cells) + 1
    
    print(f"Данные сетки {nx}x{ny} загружены. Идет преобразование в 3D...")

    # Списки для координат (N, 3) и цветов (N, 3)
    points_list = []
    colors_list_rgb = []

    for cell in cells:
        # --- 1. Преобразование 2D сетки (i, j) в сферические (lon, lat) ---
        
        # 'i' (горизонталь) -> Долгота (longitude) от 0 до 2*PI
        # Добавляем 0.5*PI, чтобы 0-я долгота смотрела на нас
        lon = (cell['i'] / nx) * 2 * np.pi + (np.pi / 2)
        
        # 'j' (вертикаль) -> Широта (latitude) от -PI/2 (юг) до +PI/2 (север)
        lat = ((cell['j'] / ny) - 0.5) * np.pi
        
        # --- 2. Радиус с преувеличенной высотой ---
        r = BASE_RADIUS + (cell['elevation'] * ELEVATION_SCALE)
        
        # --- 3. Преобразование сферических (lon, lat, r) в 3D декартовы (x, y, z) ---
        # В PyVista ось Z обычно "вверх"
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)
        
        points_list.append((x, y, z))
        
        # --- 4. Получение цвета ---
        color_hex = TERRAIN_COLORS.get(cell['terrain_type'], '#FF00FF')
        # Конвертируем HEX в RGB-массив [R, G, B]
        colors_list_rgb.append(mcolors.to_rgb(color_hex))



    # Конвертируем списки в NumPy-массивы для PyVista
    return np.array(points_list), np.array(colors_list_rgb)

# ==========================
# 3. Создание 3D-визуализации
# ==========================
def main():
    points, colors = load_and_process_data(JSON_FILE)
    if points is None:
        return

    cloud = pv.PolyData(points)
    cloud.point_data['colors'] = colors

    plotter = pv.Plotter(window_size=[1000, 800])
    plotter.set_background('black')
    plotter.add_mesh(cloud, scalars='colors', rgb=True, point_size=2.0, style='points')
    plotter.camera.elevation = 20

    # =========================
    # Вращение в цикле
    # =========================
    plotter.show(auto_close=False)  # открываем окно, но не закрываем его сразу

    try:
        while plotter.app_window.isVisible():  # пока окно открыто
            plotter.camera.azimuth += 0.5      # вращение по горизонтали
            plotter.update()                   # обновляем кадр
    except KeyboardInterrupt:
        pass




if __name__ == "__main__":
    # Убедитесь, что у вас установлен PyVista:
    # pip install pyvista numpy
    main()