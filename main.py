import json
import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkCellPicker

# --- 1. ЗАГРУЗКА ДАННЫХ ---

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
print(f"Загружено {len(cells)} клеток (сетка {nx}x{ny})")
print(f"Загружено {len(BIOME_DATA)} типов биомов.")

UNKNOWN_COLOR = (255, 0, 255)

# --- 2. ПРОЕКЦИЯ НА СФЕРУ ---

radius_earth = 6371.0
elevation_exaggeration = 50.0

grid_points = np.zeros((nx, ny, 3))
grid_colors = np.zeros((nx, ny, 3), dtype=np.uint8)
cell_data_grid = np.full((nx, ny), None, dtype=object)

for c in cells:
    i, j = c["i"], c["j"]
    theta = (i / (nx - 1)) * 2 * np.pi
    phi = np.pi / 2 - (j / (ny - 1)) * np.pi
    r = radius_earth + (c["elevation_m"] / 1000.0) * elevation_exaggeration

    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    grid_points[i, j] = [x, y, z]

    biome_name = c["biome"]
    properties = BIOME_DATA.get(biome_name)
    grid_colors[i, j] = properties["vis_color"] if properties else UNKNOWN_COLOR
    cell_data_grid[i, j] = c

# --- 3. СОЗДАЕМ 3D-MESH ---

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
colors_flat = grid_colors.reshape(-1, 3)
faces_flat = np.hstack(faces)

mesh = pv.PolyData(points_flat, faces=faces_flat)
mesh.point_data['colors'] = colors_flat

# --- 4. ВИЗУАЛИЗАЦИЯ ---

plotter = pv.Plotter(window_size=[1600, 1000])
plotter.set_background('black')
plotter.add_axes()
plotter.camera_position = 'xy'
plotter.camera.zoom(1.2)

actor = plotter.add_mesh(mesh, scalars='colors', rgb=True, smooth_shading=True)

# Начальный текст
info_text = plotter.add_text(
    "Кликни по поверхности, чтобы увидеть информацию о клетке",
    position='upper_left', font_size=12, color='gray'
)

# --- 5. ПИКЕР ---

picker = vtkCellPicker()
picker.SetTolerance(0.001)

# --- 6. ПЕРЕМЕННЫЕ СОСТОЯНИЯ ---
current_text_actor = None  # ссылка на текущий текст
highlight_actor = None     # ссылка на подсветку

# --- 7. ОБРАБОТЧИК КЛИКА ---

def on_left_click(*args):
    global current_text_actor, highlight_actor

    click_pos = plotter.iren.get_event_position()
    picker.Pick(click_pos[0], click_pos[1], 0, plotter.renderer)
    picked_point = np.array(picker.GetPickPosition())

    if np.isnan(picked_point).any() or np.allclose(picked_point, 0):
        return

    distances = np.linalg.norm(points_flat - picked_point, axis=1)
    nearest_idx = np.argmin(distances)
    if distances[nearest_idx] > 1000:
        return

    i = nearest_idx // ny
    j = nearest_idx % ny
    cell_info = cell_data_grid[i, j]
    if cell_info is None:
        return

    biome_name = cell_info["biome"]
    properties = BIOME_DATA.get(biome_name)
    if not properties:
        return

    # --- Формирование текста ---
    text_content = f"""--- Cell [{i}, {j}] ---
Biome: {biome_name}
Elevation: {cell_info['elevation_m']:.2f} m

--- Resources ---
Food (Veg):    {properties['food_vegetal']}
Food (Animal): {properties['food_animal']}
Fresh Water:   {properties['fresh_water']}
Wood:          {properties['wood_yield']}
Stone:         {properties['stone_yield']}
Ore:           {properties['ore_yield']}

--- Civilization ---
Habitability:  {properties['habitability']}
Arable Land:   {properties['arable_land']}
Movement Cost: {properties['movement_cost']}
"""

    # --- Удаляем старый текст ---
    if current_text_actor is not None:
        try:
            plotter.remove_actor(current_text_actor)
        except Exception:
            pass

    # Добавляем новый текст и сохраняем ссылку
    current_text_actor = plotter.add_text(
        text_content, position='upper_right', font_size=10, color='white'
    )

    # --- Подсветка региона 3x3 ---
    if highlight_actor is not None:
        try:
            plotter.remove_actor(highlight_actor)
        except Exception:
            pass

    region_points = []
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ii = (i + di) % nx
            jj = min(max(j + dj, 0), ny - 1)
            region_points.append(grid_points[ii, jj])

    highlight = pv.PolyData(np.array(region_points))
    highlight_actor = plotter.add_mesh(
        highlight,
        color="cyan",
        point_size=15,
        render_points_as_spheres=True
    )

# --- 8. ПОДКЛЮЧАЕМ OBSERVER ---
plotter.iren.add_observer("RightButtonPressEvent", on_left_click)

# --- 9. ЗАПУСК ---
plotter.show()
print("Готово.")
