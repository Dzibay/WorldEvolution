import json
import sys
import time
from collections import Counter

import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkCellPicker

# === 0. ДАННЫЕ БИОМОВ ===
try:
    from biomes_properties import BIOME_DATA
except ImportError:
    print("Ошибка: не найден файл biomes_properties.py!")
    sys.exit(1)

# === 1. ЗАГРУЗКА КАРТЫ ЯЧЕЕК ===
JSON_FILE = "world_cells.json"
try:
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        cells = json.load(f)
except FileNotFoundError:
    print(f"Ошибка: файл {JSON_FILE} не найден!")
    sys.exit(1)

nx = max(c["i"] for c in cells) + 1
ny = max(c["j"] for c in cells) + 1

# Исходные единицы: км/м. Перейдём в единый масштаб сцены (радиус Земли = 1.0).
R_EARTH_KM = 6371.0
SCALE = 1.0 / R_EARTH_KM  # 1.0 сцена = 6371 км в реальности
ELEV_EXAG = 50.0  # вертикальное преувеличение (в разах, для POSITIVE высот)

UNKNOWN_COLOR = (255, 0, 255)

# === 2. ПРОЕКЦИЯ В 3D (радиус = 1.0) ===
grid_points = np.zeros((nx, ny, 3), dtype=float)
grid_colors = np.zeros((nx, ny, 3), dtype=np.uint8)
cell_data_grid = np.full((nx, ny), None, dtype=object)

for c in cells:
    i, j = c["i"], c["j"]

    # долгота/широта
    theta = (i / (nx - 1)) * 2.0 * np.pi   # 0..2π
    phi = np.pi / 2.0 - (j / (ny - 1)) * np.pi  # +π/2..-π/2

    # базовый радиус = 1.0; добавляем высоту (только положительную) с преувеличением
    elev_km = max(0.0, float(c.get("elevation_m", 0.0)) / 1000.0)
    r = 1.0 + elev_km * ELEV_EXAG * SCALE

    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    grid_points[i, j] = (x, y, z)

    biome_name = c.get("biome", "Unknown")
    props = BIOME_DATA.get(biome_name)

    # цвет и объединённые свойства
    if props:
        grid_colors[i, j] = props["vis_color"]
        merged = {**props, **c}
        cell_data_grid[i, j] = merged
    else:
        grid_colors[i, j] = UNKNOWN_COLOR
        cell_data_grid[i, j] = c

# === 3. МЕШ ПОВЕРХНОСТИ СФЕРЫ ===
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
mesh.point_data["colors"] = colors_flat

# === 4. ВИЗУАЛИЗАЦИЯ ===
plotter = pv.Plotter(window_size=(1600, 1000))
plotter.set_background("black")
plotter.add_axes(interactive=False)
plotter.add_mesh(mesh, scalars="colors", rgb=True, smooth_shading=True)

# HUD — создаём ОДИН раз
hud_actor = plotter.add_text("Год: —", position="lower_left", font_size=10, color="white")

# Информационная панель по клику — не пересоздаём каждый клик
info_actor = plotter.add_text("", position="upper_right", font_size=10, color="white")
try:
    info_actor.SetInput("")  # убедимся, что можно обновлять
except Exception:
    pass

# Стабилизация камеры и проектции
plotter.camera_position = "yz"  # начальный ракурс
plotter.enable_parallel_projection()
plotter.camera.zoom(1.2)

# Пикер для кликов
click_picker = vtkCellPicker()
click_picker.SetTolerance(0.002)

# === 5. УТИЛИТЫ ДЛЯ ИНФО ===
def summarize_region(i_min, i_max, j_min, j_max):
    selected = []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            c = cell_data_grid[i % nx, j % ny]
            if c:
                selected.append(c)
    if not selected:
        return "Нет данных"

    def avg(key):
        vals = [c.get(key, 0) for c in selected]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return float(np.mean(vals)) if vals else 0.0

    biomes = [c.get("biome", "Unknown") for c in selected]
    top_biomes = ", ".join([f"{b} ({n})" for b, n in Counter(biomes).most_common(3)])

    text = (
        f"--- Selected {len(selected)} cells ---\n"
        f"Biomes: {top_biomes}\n\n"
        f"Elevation: {avg('elevation_m'):.1f} m\n"
        f"Food (Veg): {avg('food_vegetal'):.2f}\n"
        f"Food (Animal): {avg('food_animal'):.2f}\n"
        f"Water: {avg('fresh_water'):.2f}\n"
        f"Wood: {avg('wood_yield'):.2f}\n"
        f"Stone: {avg('stone_yield'):.2f}\n"
        f"Ore: {avg('ore_yield'):.2f}\n"
        f"Habitability: {avg('habitability'):.2f}\n"
        f"Arable land: {avg('arable_land'):.2f}\n"
        f"Movement cost: {avg('movement_cost'):.2f}\n"
    )
    return text

def get_cell_info(i, j):
    c = cell_data_grid[i % nx, j % ny]
    if not c:
        return "Нет данных"
    biome = c.get("biome", "Unknown")
    h = c.get("elevation_m", 0)
    habit = c.get("habitability", 0)
    food = (c.get("food_vegetal", 0) + c.get("food_animal", 0)) / 2
    water = c.get("fresh_water", 0)
    return (
        f"--- Клетка ({i},{j}) ---\n"
        f"Биом: {biome}\n"
        f"Высота: {h:.0f} м\n"
        f"Пригодность: {habit:.2f}\n"
        f"Еда: {food:.2f}\n"
        f"Вода: {water:.2f}\n"
    )

def get_group_info(entity):
    info = f"--- {entity.stage.capitalize()} #{entity.id} ---\n"
    info += f"Позиция: ({entity.i},{entity.j})\n"
    info += f"Население: {int(entity.population)}\n"
    info += f"Еда: {getattr(entity, 'food', 0.0):.1f}\n"
    info += f"Технологии: {getattr(entity, 'tech', 0.0):.3f}\n"
    info += f"Возраст: {getattr(entity, 'age', 0)}\n"
    return info

# === 6. ОБРАБОТКА КЛИКА — ИНФО О КЛЕТКЕ/ГРУППЕ ===
def on_left_click(obj, event):
    click_pos = plotter.iren.get_event_position()
    click_picker.Pick(click_pos[0], click_pos[1], 0, plotter.renderer)
    idx = click_picker.GetPointId()
    if idx < 0:
        return
    i, j = idx // ny, idx % ny
    pos = grid_points[i, j]

    # Проверяем близость к актёрам групп
    target_text = get_cell_info(i, j)
    min_dist = 0.02  # порог близости для сцены с R=1.0
    if active_groups:
        for g in active_groups:
            if g.id in group_actors:
                gpos = np.array(group_actors[g.id].GetPosition())
                if np.linalg.norm(gpos - pos) < min_dist:
                    target_text = get_group_info(g)
                    break

    try:
        safe_update_text(info_actor, target_text, corner_slot=3)
    except Exception:
        pass

plotter.iren.add_observer("LeftButtonPressEvent", on_left_click)

# === 7. СИМУЛЯЦИЯ ЧЕЛОВЕЧЕСТВА ===
print("Запускаю симуляцию групп...")

from simulation import HumanGroup, load_world

from config import (
    STARTING_CELL_COORDS,
    STARTING_POPULATION,
    SIMULATION_STEP_YEARS,
)

# Загружаем ресурсную карту мира для симуляции
world_data = load_world()

# Начальная группа
active_groups = [HumanGroup(0, *STARTING_CELL_COORDS, STARTING_POPULATION)]

# Отрисованные актёры (сферы) и линии пути
group_actors = {}   # id -> actor
group_paths = {}    # id -> tube actor
actors_to_remove = []

def grid_to_xyz(i, j, lift=0.0003):
    """
    Координата точки на поверхности (радиус=1.0) с маленьким приподнятием.
    lift ~ 0.0003 ~= 2 км над местностью.
    """
    if 0 <= i < nx and 0 <= j < ny:
        base = grid_points[i, j]
        n = base / np.linalg.norm(base)
        return base + n * lift
    return np.zeros(3)

# Создаём визуализацию начальных групп
for g in active_groups:
    pos = grid_to_xyz(g.i, g.j)
    actor = plotter.add_mesh(
        pv.Sphere(radius=0.005, center=pos),  # радиус в сцене R=1.0
        color="red",
        smooth_shading=True,
    )
    group_actors[g.id] = actor

# === 8. ТРАЕКТОРИИ ===
def update_group_path(g):
    """Обновляет короткий «хвост» пути (последние ~10 точек)."""
    path_points = g.get_path_points() if hasattr(g, "get_path_points") else []
    if len(path_points) < 2:
        return

    recent = path_points[-10:]
    coords = np.array([grid_to_xyz(i, j, lift=0.0003) for (i, j) in recent])

    path_poly = pv.Spline(coords, n_points=len(recent) * 6)
    tube = path_poly.tube(radius=0.001)

    if g.id in group_paths:
        try:
            group_paths[g.id].mapper.SetInputData(tube)
            group_paths[g.id].mapper.Update()
            return
        except Exception:
            try:
                plotter.renderer.remove_actor(group_paths[g.id])
            except Exception:
                pass
            group_paths.pop(g.id, None)

    actor = plotter.add_mesh(tube, color="orange", smooth_shading=True)
    group_paths[g.id] = actor

def cleanup_actors():
    """Безопасное удаление актёров вне важной части рендера."""
    global actors_to_remove
    if not actors_to_remove:
        return
    to_remove = actors_to_remove[:]
    actors_to_remove = []
    for act in to_remove:
        try:
            plotter.renderer.remove_actor(act)
        except Exception:
            pass

def safe_update_text(actor, text: str, corner_slot: int = 0):
    """Обновляет текст для CornerAnnotation, TextActor и tuple-обёрток."""
    try:
        # CornerAnnotation (старый HUD PyVista)
        actor.SetText(corner_slot, text)
        return
    except AttributeError:
        pass

    try:
        # Классический vtkTextActor
        actor.SetInput(text)
        return
    except AttributeError:
        pass

    try:
        # Обёртка (actor, prop)
        if isinstance(actor, (tuple, list)) and hasattr(actor[0], "SetInput"):
            actor[0].SetInput(text)
    except Exception:
        pass

# === 9. ЦИКЛ СИМУЛЯЦИИ (ДИСКРЕТНЫЙ, В ГЛАВНОМ ПОТОКЕ) ===
current_year = -100000
last_step_time = time.time()
update_interval_s = 1.0  # по умолчанию 1 секунда между шагами
simulation_running = True

def update_simulation():
    """Один дискретный шаг симуляции с плавным движением."""
    global current_year, hud_actor

    current_year += SIMULATION_STEP_YEARS
    static_counter = getattr(update_simulation, "counter", 0)

    # перебор копии, чтобы можно было модифицировать active_groups
    for g in list(active_groups):
        # смерть сущности
        if not g.alive:
            if g.id in group_actors:
                actors_to_remove.append(group_actors[g.id])
                del group_actors[g.id]
            if g.id in group_paths:
                actors_to_remove.append(group_paths[g.id])
                del group_paths[g.id]
            active_groups.remove(g)
            continue

        # логический шаг
        result = g.step(world_data)

        # образование племени (HumanGroup.step может вернуть Tribe)
        try:
            from simulation import Tribe  # локальный импорт, чтобы избежать циклических deps
        except Exception:
            Tribe = None

        if Tribe is not None and isinstance(result, Tribe):
            tribe_pos = grid_to_xyz(result.i, result.j, lift=0.0003)
            tribe_actor = plotter.add_mesh(
                pv.Sphere(radius=0.006, center=tribe_pos),
                color="yellow",
                smooth_shading=True,
            )
            group_actors[result.id] = tribe_actor
            active_groups.append(result)
            # старая группа считается погибшей в step(); удалим в следующем тике
            continue

        # живые группы/племена — обновляем актёра
        actor = group_actors.get(g.id)
        if actor is None:
            # восстановление (на случай пропуска)
            pos = grid_to_xyz(g.i, g.j)
            actor = plotter.add_mesh(
                pv.Sphere(radius=0.005 if g.stage == "group" else 0.006, center=pos),
                color="red" if g.stage == "group" else "yellow",
                smooth_shading=True,
            )
            group_actors[g.id] = actor

        target_pos = grid_to_xyz(g.i, g.j, lift=0.0003)
        old_pos = np.array(actor.GetPosition())
        # плавное движение (25% к цели)
        smooth_pos = old_pos + (target_pos - old_pos) * 0.25
        actor.SetPosition(smooth_pos)
        actor.prop.color = "yellow" if g.stage == "tribe" else "red"

        # обновляем короткий хвост пути раз в несколько тиков
        if static_counter % 5 == 0:
            update_group_path(g)

    # HUD обновляем без пересоздания
    if hud_actor is not None:
        safe_update_text(hud_actor, f"Год: {current_year}", corner_slot=0)

    update_simulation.counter = static_counter + 1

def on_render_callback(p):
    """Таймер на основе рендер-цикла: строго дискретные шаги."""
    global last_step_time
    if not simulation_running:
        return
    now = time.time()
    if now - last_step_time >= update_interval_s:
        update_simulation()
        cleanup_actors()
        last_step_time = now

plotter.add_on_render_callback(on_render_callback)

# === 10. УПРАВЛЕНИЕ СКОРОСТЬЮ И ПАУЗОЙ ===
def on_speed_key(obj, event):
    global update_interval_s, simulation_running
    key = obj.GetKeySym().lower()
    if key in ("plus", "equal"):
        update_interval_s = max(0.1, update_interval_s / 1.5)
    elif key in ("minus", "underscore"):
        update_interval_s = min(5.0, update_interval_s * 1.5)
    elif key == "p":
        simulation_running = not simulation_running
        print("⏸ Пауза" if not simulation_running else "▶ Продолжение")
    print(f"⏱ Шаг каждые {update_interval_s:.2f} сек.")

plotter.iren.add_observer("KeyPressEvent", on_speed_key)

# === 11. ПУСК ===
plotter.reset_camera()
plotter.show()
print("Готово.")