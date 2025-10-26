import json
import sys
import numpy as np

import vispy.app
vispy.app.use_app('glfw')

from vispy import app, scene
from vispy.scene.visuals import Text

# === 0. Загрузка биомов ===
try:
    from biomes_properties import BIOME_DATA
except ImportError:
    print("Ошибка: не найден файл biomes_properties.py!")
    sys.exit(1)

# === 1. Загрузка карты ячеек ===
JSON_FILE = "world_cells.json"
try:
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        cells = json.load(f)
except FileNotFoundError:
    print(f"Ошибка: файл {JSON_FILE} не найден!")
    sys.exit(1)

nx = max(c["i"] for c in cells) + 1
ny = max(c["j"] for c in cells) + 1

R_EARTH_KM = 6371.0
SCALE = 1.0 / R_EARTH_KM
ELEV_EXAG = 50.0  # преувеличение рельефа
UNKNOWN_COLOR = (255, 0, 255)

# === 2. Сферическая сетка ===
points = np.zeros((nx, ny, 3), dtype=np.float32)
colors = np.zeros((nx, ny, 4), dtype=np.float32)

for c in cells:
    i, j = c["i"], c["j"]
    theta = (i / (nx - 1)) * 2.0 * np.pi
    phi = np.pi / 2.0 - (j / (ny - 1)) * np.pi
    elev_km = float(c.get("elevation_m", 0.0)) / 1000.0
    r = 1.0

    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    points[i, j] = (x, y, z)

    biome = c.get("biome", "Unknown")
    props = BIOME_DATA.get(biome)
    col = props["vis_color"] if props else UNKNOWN_COLOR
    colors[i, j] = [col[0]/255, col[1]/255, col[2]/255, 1.0]

# === 3. Генерация треугольников для меша ===
faces = []
for i in range(nx - 1):
    for j in range(ny - 1):
        p0 = i * ny + j
        p1 = (i + 1) * ny + j
        p2 = (i + 1) * ny + (j + 1)
        p3 = i * ny + (j + 1)
        faces.append((p0, p1, p2))
        faces.append((p0, p2, p3))
faces = np.array(faces, dtype=np.uint32)

verts = points.reshape(-1, 3)
cols = colors.reshape(-1, 4)

# === 4. Создание сцены ===
canvas = scene.SceneCanvas(keys="interactive", bgcolor="black", show=True, size=(1600, 1000))
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=0, elevation=30, distance=3)

# Меш Земли
earth_mesh = scene.visuals.Mesh(
    vertices=verts,
    faces=faces,
    vertex_colors=cols,
    shading='smooth',
    parent=view.scene
)

# === 5. Группы (красные точки) ===
from simulation import HumanGroup, load_world
from config import STARTING_CELL_COORDS, STARTING_POPULATION, SIMULATION_STEP_YEARS

world_data = load_world()
active_groups = [HumanGroup(0, *STARTING_CELL_COORDS, STARTING_POPULATION)]

group_markers = scene.visuals.Markers(parent=view.scene)
group_markers.set_data(np.array([[0, 0, 1]]), face_color='red', size=10)

def grid_to_xyz(i, j, lift=0.002):
    base = points[i % nx, j % ny]
    n = base / np.linalg.norm(base)
    return base + n * lift

# === 6. HUD (год) ===
hud = Text("Год: —", 
           parent=canvas.scene, 
           anchor_x='left',   # ← выравнивание по левому краю
           anchor_y='bottom',
           color='white')
hud.font_size = 20
hud.pos = (0, 0)

# === 7. Информационное окно (по клику) ===
def compute_ray_from_click(view, canvas, pos):
    """
    Устойчивый рэйкаст для TurntableCamera:
    - направлен точно к center
    - без завала/скоса при любом повороте/зуме
    """
    import numpy as np

    cam = view.camera
    W, H = canvas.size
    # экран -> NDC
    x_ndc = (2.0 * pos[0] / W) - 1.0
    y_ndc = 1.0 - (2.0 * pos[1] / H)

    fov = np.deg2rad(cam.fov)
    aspect = W / H

    # 1) позиция камеры из параметров turntable
    theta = np.deg2rad(cam.azimuth)
    phi   = np.deg2rad(cam.elevation)
    r     = cam.distance
    cam_pos = cam.center + r * np.array([
        np.cos(phi) * np.sin(theta),
        -np.cos(phi) * np.cos(theta),
        np.sin(phi)
    ], dtype=np.float32)

    # 2) базис камеры из center/pos
    forward = cam.center - cam_pos
    forward /= np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # если почти смотрим вдоль world_up — возьмём запасной "вверх"
    if abs(np.dot(forward, world_up)) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # ВНИМАНИЕ на порядок кроссов, чтобы "право" совпадало с экраном
    right = np.cross(forward, world_up); right /= np.linalg.norm(right)
    up    = np.cross(right,  forward);   up    /= np.linalg.norm(up)

    # 3) плоскость проекции
    half_h = np.tan(fov / 2.0)
    half_w = aspect * half_h

    # 4) направление луча
    dir_world = forward + right * (x_ndc * half_w) + up * (y_ndc * half_h)
    dir_world /= np.linalg.norm(dir_world)

    return cam_pos.astype(np.float32), dir_world.astype(np.float32)

info_box = Text(
    '', 
    parent=canvas.scene,
    anchor_x='left',
    anchor_y='bottom',
    color='white'
)
info_box.font_size = 16
info_box.pos = (0, 50)


def get_cell_info(i, j):
    """Возвращает подробную информацию о клетке (по координатам i,j)."""
    c = next((cell for cell in cells if cell["i"] == i and cell["j"] == j), None)
    if not c:
        return f"❌ Клетка ({i},{j}) не найдена."

    biome = c.get("biome", "Unknown")
    elev = c.get("elevation_m", 0.0)

    # 🔹 Достаём характеристики из BIOME_DATA
    props = BIOME_DATA.get(biome, {})
    fresh_water = props.get("fresh_water", 0)
    food_vegetal = props.get("food_vegetal", 0)
    food_animal = props.get("food_animal", 0)
    wood = props.get("wood_yield", 0)
    stone = props.get("stone_yield", 0)
    ore = props.get("ore_yield", 0)
    habit = props.get("habitability", 0)
    arable = props.get("arable_land", 0)
    move_cost = props.get("movement_cost", 0)
    is_ocean = props.get("is_ocean", False)
    is_fresh = props.get("is_fresh_water", False)

    avg_food = (food_vegetal + food_animal) / 2

    # --- Форматированный вывод ---
    lines = [
        f"--- Клетка ({i},{j}) ---",
        f"Биом: {biome}",
        f"Высота: {elev:.0f} м",
        "",
        f"Пригодность для жизни: {habit:.2f}",
        f"Пригодность для земледелия: {arable:.2f}",
        f"Стоимость передвижения: {move_cost:.2f}",
        "",
        f"Вода (пресная): {fresh_water:.2f}{' 💧' if is_fresh else ''}",
        f"Еда: {avg_food:.2f} (средняя)",
        f"     растительная {food_vegetal:.2f},",
        f"     животная {food_animal:.2f},",
        f"Ресурсы:",
        f"  Древесина: {wood:.2f}",
        f"  Камень: {stone:.2f}",
        f"  Руда: {ore:.2f}",
    ]
    return "\n".join(lines)


def get_group_info(entity):
    return (
        f"--- {entity.stage.capitalize()} #{entity.id} ---\n"
        f"Позиция: ({entity.i},{entity.j})\n"
        f"Население: {int(entity.population)}\n"
        f"Еда: {getattr(entity, 'food', 0):.1f}\n"
        f"Технологии: {getattr(entity, 'tech', 0):.3f}\n"
        f"Возраст: {getattr(entity, 'age', 0)}"
    )

# === Добавим глобальный маркер для визуализации точки клика ===
click_marker = scene.visuals.Markers(parent=view.scene)
click_marker.set_data(np.array([[0, 0, 0]]), face_color='yellow', size=10)

@canvas.events.mouse_press.connect
def on_mouse_click(event):
    """При клике мышью показываем информацию о ближайшей клетке или объекте."""
    if event.button != 1:
        return

    pos = event.pos
    ray_origin, ray_dir = compute_ray_from_click(view, canvas, pos)

    # === 1. Пересечение луча со сферой ===
    R = float(np.linalg.norm(points[0, 0]))  # радиус сферы из меша
    # R — радиус сферы (у тебя уже есть)
    a = np.dot(ray_dir, ray_dir)               # =1, но пусть будет общее
    b = 2.0 * np.dot(ray_origin, ray_dir)
    c = np.dot(ray_origin, ray_origin) - R*R
    delta = b*b - 4*a*c
    if delta < 0:
        print("❌ Промах"); return

    sqrt_delta = np.sqrt(delta)
    t_candidates = [t for t in ((-b - sqrt_delta)/(2*a), (-b + sqrt_delta)/(2*a)) if t > 0]
    if not t_candidates:
        print("❌ Пересечения позади камеры"); return

    t = min(t_candidates)  # ближайшая точка ВПЕРЕДИ камеры
    hit_point = ray_origin + t * ray_dir
    x, y, z = hit_point

    # === 3. Конвертация в сферические координаты ===
    theta = np.arctan2(y, x)
    if theta < 0:
        theta += 2*np.pi
    phi = np.arctan2(z, np.sqrt(x*x + y*y))

    i = int(np.rint(theta / (2*np.pi) * (nx - 1))) % nx
    j = int(np.rint((np.pi/2 - phi) / np.pi * (ny - 1)))
    j = int(np.clip(j, 0, ny - 1))

    # === 5. Обновляем маркер точки клика ===
    # Поднимаем точку на 0.5% радиуса над поверхностью
    lift = 0.005 * R  # 0.5% радиуса (можно изменить)
    n = hit_point / np.linalg.norm(hit_point)  # нормаль наружу
    hit_lifted = hit_point + n * lift

    click_marker.set_data(np.array([hit_lifted]), face_color='yellow', size=12)

    # === 6. Информация о клетке ===
    text = get_cell_info(i, j)

    # Проверяем наличие группы поблизости
    min_dist = 0.01
    for g in active_groups:
        pos_g = grid_to_xyz(g.i, g.j)
        if np.linalg.norm(pos_g - points[i, j]) < min_dist:
            text = get_group_info(g)
            break

    info_box.text = text


# === 8. Симуляция ===
current_year = -100000
simulation_running = True
update_interval_s = 0.5

def update_simulation(event):
    global current_year
    if not simulation_running:
        return

    current_year += SIMULATION_STEP_YEARS
    hud.text = f"Год: {current_year}"

    new_entities = []
    positions = []
    colors = []

    for g in list(active_groups):
        if not g.alive:
            active_groups.remove(g)
            continue

        result = g.step(world_data, debug=True)

        # Создание племени
        try:
            from simulation import Tribe
            if isinstance(result, Tribe):
                print(f"➡ Группа #{g.id} создала племя #{result.id} ({result.i},{result.j})")
                new_entities.append(result)
                if g in active_groups:
                    active_groups.remove(g)
                continue
        except Exception:
            pass

        pos = grid_to_xyz(g.i, g.j)
        positions.append(pos)
        if getattr(g, "stage", "group") == "tribe":
            colors.append((1.0, 1.0, 0.0, 1.0))
        else:
            colors.append((1.0, 0.0, 0.0, 1.0))

    if new_entities:
        active_groups.extend(new_entities)

    if positions:
        group_markers.set_data(np.array(positions), face_color=np.array(colors), size=10)

# === 9. Таймер ===
timer = app.Timer(interval=update_interval_s, connect=update_simulation, start=True)

# === 10. Управление ===
@canvas.events.key_press.connect
def on_key(event):
    global simulation_running, update_interval_s
    if event.key == 'P':
        simulation_running = not simulation_running
        print("⏸ Пауза" if not simulation_running else "▶ Продолжение")
    elif event.key in ['+', '=']:
        update_interval_s = max(0.05, update_interval_s / 1.5)
        timer.interval = update_interval_s
        print(f"⏱ Интервал: {update_interval_s:.2f} сек")
    elif event.key in ['-', '_']:
        update_interval_s = min(5.0, update_interval_s * 1.5)
        timer.interval = update_interval_s
        print(f"⏱ Интервал: {update_interval_s:.2f} сек")

# === 11. Запуск ===
if __name__ == "__main__":
    app.run()
