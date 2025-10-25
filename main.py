import json
import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkCellPicker
from collections import Counter

# === 1. ЗАГРУЗКА ДАННЫХ ===

try:
    from biomes_properties import BIOME_DATA
except ImportError:
    print("Ошибка: не найден файл biome_properties.py!")
    exit()

json_filename = "world_cells.json"
try:
    with open(json_filename, "r") as f:
        cells = json.load(f)
except FileNotFoundError:
    print(f"Ошибка: Файл {json_filename} не найден!")
    exit()

nx = max(c["i"] for c in cells) + 1
ny = max(c["j"] for c in cells) + 1

UNKNOWN_COLOR = (255, 0, 255)
radius_earth = 6371.0
elevation_exaggeration = 50.0

# === 2. ПРОЕКЦИЯ В 3D ===
grid_points = np.zeros((nx, ny, 3))
grid_colors = np.zeros((nx, ny, 3), dtype=np.uint8)
cell_data_grid = np.full((nx, ny), None, dtype=object)

for c in cells:
    i, j = c["i"], c["j"]
    theta = (i / (nx - 1)) * 2 * np.pi
    phi = np.pi / 2 - (j / (ny - 1)) * np.pi
    r = radius_earth + (c["elevation_m"] / 1000.0) * elevation_exaggeration if c["elevation_m"] > 0 else radius_earth

    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    grid_points[i, j] = [x, y, z]

    biome_name = c["biome"]
    props = BIOME_DATA.get(biome_name)

    # <<< ИЗМЕНЕНО: Исправление проблемы с нулевыми средними
    # Мы должны объединить данные из BIOME_DATA (props) и world_cells.json (c)
    
    if props:
        grid_colors[i, j] = props["vis_color"]
        # Сначала берем все свойства биома из 'props'
        # Затем перезаписываем их 'c' (данные ячейки, как elevation_m)
        merged_data = {**props, **c}
        cell_data_grid[i, j] = merged_data
    else:
        # Если биом не найден, используем цвет по умолчанию и только данные 'c'
        grid_colors[i, j] = UNKNOWN_COLOR
        cell_data_grid[i, j] = c

# ... (Код создания faces, points_flat и т.д. не изменился) ...
faces = []
for i in range(nx - 1):
    for j in range(ny - 1):
        p0 = i * ny + j
        p1 = (i + 1) * ny + j
        p2 = (i + 1) * ny + (j + 1)
        p3 = i * ny + (j + 1)
        faces.append([3, p0, p1, p2])
        faces.append([3, p0, p2, p3])

points_flat = grid_points.reshape(-1, 3)
faces_flat = np.hstack(faces)
colors_flat = grid_colors.reshape(-1, 3)
mesh = pv.PolyData(points_flat, faces=faces_flat)
mesh.point_data['colors'] = colors_flat


# === 3. ВИЗУАЛИЗАЦИЯ ===
plotter = pv.Plotter(window_size=[1600, 1000])
plotter.set_background('black')
plotter.add_axes()
plotter.camera_position = 'xy'
plotter.add_mesh(mesh, scalars='colors', rgb=True, smooth_shading=True)
plotter.add_text("Зажми ПРОБЕЛ и тяни мышь — выделить область, отпустить — вычислить и показать", position="upper_left", font_size=12, color="gray")

picker = vtkCellPicker()
picker.SetTolerance(0.005)

# === 4. ПЕРЕМЕННЫЕ СОСТОЯНИЯ ===
drag_start = None
# <<< ИЗМЕНЕНО: Будем хранить список акторов (4 линии)
highlight_actors = [] 
current_text_actor = None

# === 5. ФУНКЦИЯ ДЛЯ СРЕДНИХ ===
def summarize_region(i_min, i_max, j_min, j_max):
    selected = []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            c = cell_data_grid[i % nx, j] # i % nx - позволяет "обернуться" вокруг глобуса
            if c:
                selected.append(c)
    if not selected:
        return "Нет данных"

    # Эта функция теперь будет работать, т.к. 'c' содержит все ключи
    def avg(key): return np.mean([c.get(key, 0) for c in selected if isinstance(c.get(key, 0), (int, float))])
    
    # <<< НАЧАЛО ИЗМЕНЕНИЯ
    biomes = [c["biome"] for c in selected]
    
    # Считаем биомы и сразу сортируем по убыванию
    biome_counts = Counter(biomes).most_common()
    
    # Берем топ-3
    top_3_biomes = biome_counts[:3]
    
    # Форматируем список строк
    biome_list = [f"{b} ({n})" for b, n in top_3_biomes]
    
    # Соединяем
    biome_stats = ", ".join(biome_list)
    
    # Если всего уникальных биомов было больше 3, добавляем троеточие
    if len(biome_counts) > 3:
        biome_stats += ", ..."
    # <<< КОНЕЦ ИЗМЕНЕНИЯ

    text = f"""--- Selected {len(selected)} cells ---
        Biomes: {biome_stats}

        Elevation: {avg('elevation_m'):.1f} m
        Food (Veg): {avg('food_vegetal'):.2f}
        Food (Animal): {avg('food_animal'):.2f}
        Water: {avg('fresh_water'):.2f}
        Wood: {avg('wood_yield'):.2f}
        Stone: {avg('stone_yield'):.2f}
        Ore: {avg('ore_yield'):.2f}
        Habitability: {avg('habitability'):.2f}
        Arable land: {avg('arable_land'):.2f}
        Movement cost: {avg('movement_cost'):.2f}
        """
    return text

# === 6. ОБРАБОТЧИКИ МЫШИ (выделение с пробелом) ===
space_pressed = False

def on_key_press(obj, event):
    global space_pressed, drag_start
    key = obj.GetKeySym()
    
    if key == "space" and not space_pressed:
        space_pressed = True
        
        click_pos = plotter.iren.get_event_position()
        picker.Pick(click_pos[0], click_pos[1], 0, plotter.renderer)
        idx = picker.GetPointId()
        
        if idx < 0: 
            space_pressed = False 
            drag_start = None
            return

        drag_start = (idx // ny, idx % ny)

def on_key_release(obj, event):
    # <<< ИЗМЕНЕНО: 'highlight_actor' -> 'highlight_actors'
    global space_pressed, drag_start, highlight_actors, current_text_actor
    key = obj.GetKeySym()

    if key == "space" and space_pressed:
        space_pressed = False 
        
        if drag_start is None:
            return

        i1, j1 = drag_start
        drag_start = None 

        click_pos = plotter.iren.get_event_position()
        picker.Pick(click_pos[0], click_pos[1], 0, plotter.renderer)
        idx = picker.GetPointId()

        # <<< ИЗМЕНЕНО: Очищаем старые линии при отмене
        if idx < 0: 
            if highlight_actors:
                for act in highlight_actors:
                    plotter.remove_actor(act)
                highlight_actors = []
            return
            
        i2, j2 = idx // ny, idx % ny

        i_min, i_max = sorted([i1, i2])
        j_min, j_max = sorted([j1, j2])
        summary = summarize_region(i_min, i_max, j_min, j_max)

        # обновляем текст
        if current_text_actor:
            plotter.remove_actor(current_text_actor)
        current_text_actor = plotter.add_text(summary, position="upper_right", font_size=10, color="white")

        # <<< ИЗМЕНЕНО: Логика подсветки полностью переписана
        
        # Сначала удаляем старые линии
        if highlight_actors:
            for act in highlight_actors:
                plotter.remove_actor(act)
            highlight_actors = []

        # Создаем 4 НОВЫХ линии, которые следуют по сетке
        
        # 1. Верхняя линия (вдоль j_min)
        top_points = np.array([grid_points[i % nx, j_min] for i in range(i_min, i_max + 1)])
        # 2. Нижняя линия (вдоль j_max)
        bottom_points = np.array([grid_points[i % nx, j_max] for i in range(i_min, i_max + 1)])
        # 3. Левая линия (вдоль i_min)
        left_points = np.array([grid_points[i_min % nx, j] for j in range(j_min, j_max + 1)])
        # 4. Правая линия (вдоль i_max)
        right_points = np.array([grid_points[i_max % nx, j] for j in range(j_min, j_max + 1)])

        # Собираем все 4 набора точек
        all_edge_points = [top_points, bottom_points, left_points, right_points]

        for points in all_edge_points:
            # Убедимся, что в линии есть хотя бы 2 точки
            if len(points) < 2:
                continue
            
            # 1. Создаем PolyData из точек
            poly = pv.PolyData(points)
            
            # 2. Создаем "линии", соединяющие точки по порядку (0-1, 1-2, 2-3...)
            # Это создает одну длинную ломаную линию (polyline)
            line_indices = np.arange(len(points))
            lines_array = np.hstack((len(points), line_indices))
            poly.lines = lines_array
            
            # 3. Превращаем 1D-линию в 3D-"трубу", чтобы ее было видно
            tube = poly.tube(radius=15.0) # Радиус можно настроить
            
            # 4. Добавляем "трубу" на сцену и сохраняем актора
            actor = plotter.add_mesh(tube, color="magenta", smooth_shading=True)
            highlight_actors.append(actor)

# (Функция on_mouse_move_space удалена)

# === 7. ПОДКЛЮЧАЕМ СОБЫТИЯ ===
# (Обработчик on_mouse_move_space удален)
plotter.iren.add_observer("KeyPressEvent", on_key_press)
plotter.iren.add_observer("KeyReleaseEvent", on_key_release)

# === 8. ЗАПУСК ===
plotter.show()
print("Готово.")