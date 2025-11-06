import json, os, numpy as np
from vispy import app, scene
from vispy.scene.visuals import Text
from PyQt6 import QtWidgets, QtCore
import gzip

app.use_app('pyqt6')  # важно

from biomes_properties import BIOME_DATA


# === Основной класс окна ===
class LogViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WorldEvolution Log Viewer")
        self.resize(1800, 1000)

        # Центральный виджет (разделитель)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # === Левая часть — VisPy Canvas ===
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', show=False, size=(1400, 1000))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=0, elevation=30, distance=3)
        layout.addWidget(self.canvas.native, stretch=4)
        self.canvas.events.mouse_press.connect(self.on_mouse_click)

        # === Правая панель (Qt) ===
        self.side = QtWidgets.QVBoxLayout()
        layout.addLayout(self.side, stretch=1)

        # === Элементы управления ===
        self.label_year = QtWidgets.QLabel("Год: —")
        self.label_year.setStyleSheet("color:white; font-size:16px;")
        self.side.addWidget(self.label_year)

        self.combo_log = QtWidgets.QComboBox()
        self.refresh_log_files()
        self.side.addWidget(self.combo_log)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.side.addWidget(self.slider)

        controls = QtWidgets.QHBoxLayout()
        self.side.addLayout(controls)
        self.btn_play = QtWidgets.QPushButton("▶ / ⏸")
        self.btn_fast = QtWidgets.QPushButton("⏩")
        self.btn_slow = QtWidgets.QPushButton("⏪")
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_fast)
        controls.addWidget(self.btn_slow)

        self.side.addStretch(1)

        # === Подключаем сигналы ===
        self.combo_log.activated.connect(self.on_log_selected)
        self.slider.valueChanged.connect(self.slider_changed)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_fast.clicked.connect(self.speed_up)
        self.btn_slow.clicked.connect(self.speed_down)

        # === Визуализация ===
        self.load_world()
        self.current_entities = []
        self.current_log = []
        self.frame_index = 0
        self.paused = False
        self.speed_factor = 1.0

        self.timer = app.Timer(interval=0.5, connect=self.update_frame, start=False)
        self.canvas.events.draw.connect(self.on_canvas_ready)

        # === Информационный блок (нижняя половина) ===
        self.info_box = QtWidgets.QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setStyleSheet("""
            background-color: #111;
            color: #FFD700;
            font-family: Consolas;
            font-size: 12px;
            border: 1px solid #333;
        """)
        self.side.addWidget(self.info_box, stretch=2)

        self.state_color_map = {}
        self.highlight_state = None
        self.highlight_color = (1.0, 1.0, 0.0, 1.0)  # жёлтый

        self.capital_markers = scene.visuals.Markers(parent=self.view.scene)
        self.capital_markers.set_gl_state('translucent', depth_test=True)


    def on_canvas_ready(self, event=None):
        """Запускаем таймер только когда сцена впервые отрисована (OpenGL готов)."""
        if getattr(self, "_canvas_ready", False):
            return  # уже запускали
        self._canvas_ready = True
        print("🟢 OpenGL контекст готов, стартуем таймер.")
        self.timer.start()

    def grid_to_xyz(self, i, j, lift=0.002):
        base = self.points[i % self.nx, j % self.ny]
        n = base / np.linalg.norm(base)
        return base + n * lift

    # ------------------------------------------------------------
    # === Логика загрузки мира и логов ===
    # ------------------------------------------------------------

    def refresh_log_files(self):
        files = [
            f for f in os.listdir("logs")
            if f.endswith(".json") or f.endswith(".json.gz")
        ]
        self.combo_log.addItems(files)

    def load_world(self):
        with open("world_cells.json", "r", encoding="utf-8") as f:
            cells = json.load(f)

        self.nx = max(c["i"] for c in cells) + 1
        self.ny = max(c["j"] for c in cells) + 1
        self.world_cells = [[{} for _ in range(self.nx)] for _ in range(self.ny)]
        for c in cells:
            self.world_cells[c["j"]][c["i"]] = c

        self.points = np.zeros((self.nx, self.ny, 3), dtype=np.float32)
        self.colors = np.zeros((self.nx, self.ny, 4), dtype=np.float32)
        for c in cells:
            i, j = c["i"], c["j"]
            theta = (i / (self.nx - 1)) * 2.0 * np.pi
            phi = np.pi / 2.0 - (j / (self.ny - 1)) * np.pi
            r = 1.0
            x = r * np.cos(phi) * np.cos(theta)
            y = r * np.cos(phi) * np.sin(theta)
            z = r * np.sin(phi)
            self.points[i, j] = (x, y, z)
            biome = c.get("biome", "Unknown")
            props = BIOME_DATA.get(biome)
            col = props["vis_color"] if props else (255, 0, 255)
            self.colors[i, j] = [col[0]/255, col[1]/255, col[2]/255, 1.0]
        self.verts = self.points.reshape(-1, 3)
        self.base_cols_flat = self.colors.reshape(-1, 4)

        faces = []
        for i in range(self.nx - 1):
            for j in range(self.ny - 1):
                p0 = i * self.ny + j
                p1 = (i + 1) * self.ny + j
                p2 = (i + 1) * self.ny + (j + 1)
                p3 = i * self.ny + (j + 1)
                faces.append((p0, p1, p2))
                faces.append((p0, p2, p3))
        faces = np.array(faces, dtype=np.uint32)
        self.faces = faces

        self.earth_mesh = scene.visuals.Mesh(
            vertices=self.verts, faces=self.faces, vertex_colors=self.base_cols_flat, shading='smooth', parent=self.view.scene
        )
        self.markers = scene.visuals.Markers(parent=self.view.scene)

    def on_log_selected(self, index):
        if index < 0:
            return
        filename = self.combo_log.itemText(index)
        self.load_log(filename)

    def load_log(self, filename):
        if not filename:
            return

        full_path = os.path.join("logs", filename)
        print(f"📜 Загружаем лог: {full_path}")

        try:
            if filename.endswith(".gz"):
                with gzip.open(full_path, "rt", encoding="utf-8") as f:
                    self.current_log = json.load(f)
                    print('yee')
            else:
                with open(full_path, "r", encoding="utf-8") as f:
                    self.current_log = json.load(f)

        except Exception as e:
            print("Ошибка при чтении лога:", e)
            return

        if not self.current_log:
            print("⚠ Лог пустой.")
            return

        self.slider.setRange(0, len(self.current_log)-1)
        self.slider.setValue(0)
        self.frame_index = 0

        # 🔹 моментальная отрисовка первого кадра
        self.draw_frame(0)
        self.view.camera.view_changed()
        self.canvas.update()

        # 🔹 активируем воспроизведение
        self.paused = False
        self.timer.start()


        print(f"✅ Лог успешно загружен ({len(self.current_log)} кадров)")

    # ------------------------------------------------------------
    # === Отрисовка и обновление ===
    # ------------------------------------------------------------

    def draw_frame(self, idx):
        print('drawing', idx)
        if not self.current_log or self.earth_mesh is None:
            print(self.earth_mesh)
            return

        if getattr(self.earth_mesh, "mesh_data", None) is None:
            QtCore.QTimer.singleShot(100, lambda: self.draw_frame(idx))
            return

        if getattr(self, "highlighted", False):
            self.highlighted = False
            self.earth_mesh.mesh_data.set_vertex_colors(self.base_cols_flat)


        snap = self.current_log[idx]
        year = snap["year"]
        self.current_entities = snap["entities"]
        self.label_year.setText(f"Год: {year}")

        mesh_cols = self.base_cols_flat.copy()
        positions, colors_list, sizes = [], [], []

        for e in self.current_entities:
            stage, i, j = e["stage"], e["i"], e["j"]

            if stage == "state":
                state_id = e["id"]

                # --- Если для государства нет цвета — генерируем новый ---
                if state_id not in self.state_color_map:
                    # Генерация уникального RGB на основе ID (детерминированно)
                    np.random.seed(state_id)  # чтобы каждый ID всегда имел одинаковый цвет
                    color_rgb = np.random.rand(3) * 0.8 + 0.2  # чуть ярче (0.2–1.0)
                    self.state_color_map[state_id] = color_rgb

                else:
                    color_rgb = self.state_color_map[state_id]

                color_rgba = (color_rgb[0], color_rgb[1], color_rgb[2], 1.0)

                # --- Раскрашиваем территорию ---
                for (ti, tj) in e.get("territory", []):
                    vertex_idx = ti * self.ny + tj
                    if 0 <= vertex_idx < len(mesh_cols):
                        mesh_cols[vertex_idx] = color_rgba

            else:
                pos = self.grid_to_xyz(i, j)
                positions.append(pos)
                if stage == "group":
                    colors_list.append((1, 0, 0, 1))
                    sizes.append(6)
                elif stage == "tribe":
                    colors_list.append((1, 1, 0, 1))
                    sizes.append(8)
                elif stage == "city":
                    colors_list.append((0, 1, 0, 1))
                    sizes.append(10)
                elif stage == "seafaring":
                    colors_list.append((1, 1, 1, 1))
                    sizes.append(8)
                else:
                    colors_list.append((0.5, 0.5, 0.5, 1))
                    sizes.append(5)
        
        # --- Если есть выделенное государство, подсвечиваем его ---
        if self.highlight_state:
            highlight_territory = self.highlight_state.get("territory", [])
            for (ti, tj) in highlight_territory:
                vertex_idx = ti * self.ny + tj
                if 0 <= vertex_idx < len(mesh_cols):
                    mesh_cols[vertex_idx] = self.highlight_color
        
        # === Отображение столиц государств ===
        capital_positions, capital_colors, capital_sizes = [], [], []

        for e in self.current_entities:
            if e["stage"] == "state":
                i, j = e["i"], e["j"]
                pos = self.grid_to_xyz(i, j, lift=0.004)
                capital_positions.append(pos)

                tech = e.get("tech", 0.0)
                base_size = 5 + tech * 6
                color = (1.0, 0.9, 0.3, 1.0)

                # --- если это выделенное государство ---
                if self.highlight_state and e["id"] == self.highlight_state["id"]:
                    base_size *= 1.8          # увеличиваем размер
                    color = (1.0, 1.0, 0.5, 1.0)  # делаем чуть ярче

                capital_colors.append(color)
                capital_sizes.append(base_size)

        if capital_positions:
            self.capital_markers.set_data(
                np.array(capital_positions),
                face_color=np.array(capital_colors),
                size=np.array(capital_sizes),
                symbol='star',
                edge_color='white'
            )
        else:
            self.capital_markers.set_data(np.empty((0, 3)))

        # --- безопасное обновление буфера ---
        if getattr(self.earth_mesh, "mesh_data", None):
            self.earth_mesh.mesh_data.set_vertex_colors(mesh_cols)
            self.earth_mesh.mesh_data_changed()
            self.earth_mesh.update()
            self.canvas.update()  # 🔹 гарантирует GPU-перерисовку

        # --- точки (агенты) ---
        if positions:
            self.markers.set_data(
                np.array(positions),
                face_color=np.array(colors_list),
                size=np.array(sizes)
            )
        else:
            self.markers.set_data(np.empty((0, 3)))

        # 🔹 принудительно сообщаем сцене, что кадр изменился
        self.view.camera.view_changed()
        self.canvas.update()

    def grid_to_xyz(self, i, j, lift=0.002):
        base = self.points[i % self.nx, j % self.ny]
        n = base / np.linalg.norm(base)
        return base + n * lift

    def update_frame(self, event):
        if self.paused or not self.current_log:
            return

        self.frame_index += 1
        if self.frame_index >= len(self.current_log):
            self.frame_index = len(self.current_log) - 1
            self.draw_frame(self.frame_index)
            self.timer.stop()
            self.paused = True
            print("⏹ Конец лога, воспроизведение остановлено.")
            return

        self.draw_frame(self.frame_index)

        # 🔹 обновляем ползунок без вызова slider_changed
        self.slider.blockSignals(True)
        self.slider.setValue(self.frame_index)
        self.slider.blockSignals(False)

        self.canvas.update()

    # ------------------------------------------------------------
    # === Управление ===
    # ------------------------------------------------------------

    def slider_changed(self, val):
        if getattr(self, "_updating_slider", False):
            return
        self._updating_slider = True
        self.frame_index = val
        self.draw_frame(val)
        self._updating_slider = False

    def toggle_play(self):
        self.paused = not self.paused
        if not self.paused:
            if not self.timer.running:
                self.timer.start()
            print("▶ Продолжение")
        else:
            print("⏸ Пауза")

    def speed_up(self):
        self.speed_factor = min(4.0, self.speed_factor * 1.5)
        self.timer.interval = max(0.05, 0.5 / self.speed_factor)
        print(f"⚡ Скорость x{self.speed_factor:.1f}")

    def speed_down(self):
        self.speed_factor = max(0.25, self.speed_factor / 1.5)
        self.timer.interval = 0.5 / self.speed_factor
        print(f"🐢 Скорость x{self.speed_factor:.1f}")

    def compute_ray_from_click(self, view, canvas, pos):
        cam = view.camera
        W, H = canvas.size
        x_ndc = (2.0 * pos[0] / W) - 1.0
        y_ndc = 1.0 - (2.0 * pos[1] / H)
        fov = np.deg2rad(cam.fov)
        aspect = W / H
        theta = np.deg2rad(cam.azimuth)
        phi = np.deg2rad(cam.elevation)
        r = cam.distance
        cam_pos = cam.center + r * np.array([
            np.cos(phi) * np.sin(theta),
            -np.cos(phi) * np.cos(theta),
            np.sin(phi)
        ], dtype=np.float32)
        forward = cam.center - cam_pos
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(np.dot(forward, world_up)) > 0.98:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        half_h = np.tan(fov / 2.0)
        half_w = aspect * half_h
        dir_world = forward + right * (x_ndc * half_w) + up * (y_ndc * half_h)
        dir_world /= np.linalg.norm(dir_world)
        return cam_pos.astype(np.float32), dir_world.astype(np.float32)

    def on_mouse_click(self, event):
        if event.button != 1:
            return

        pos = event.pos
        ray_origin, ray_dir = self.compute_ray_from_click(self.view, self.canvas, pos)

        # --- Пересечение луча со сферой ---
        R = 1.0
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(ray_origin, ray_dir)
        c = np.dot(ray_origin, ray_origin) - R*R
        delta = b*b - 4*a*c
        if delta < 0:
            return
        t = (-b - np.sqrt(delta)) / (2*a)
        hit_point = ray_origin + t * ray_dir
        x, y, z = hit_point

        # --- Конвертация в i,j ---
        theta = np.arctan2(y, x)
        if theta < 0:
            theta += 2*np.pi
        phi = np.arctan2(z, np.sqrt(x*x + y*y))
        i = int(np.rint(theta / (2*np.pi) * (self.nx - 1))) % self.nx
        j = int(np.rint((np.pi/2 - phi) / np.pi * (self.ny - 1)))
        j = int(np.clip(j, 0, self.ny - 1))

        # --- Базовая информация о клетке ---
        cell = self.world_cells[j][i]
        biome = cell.get("biome", "Unknown")
        elev = cell.get("elevation_m", 0.0)

        props = BIOME_DATA.get(biome, {})
        habit = props.get("habitability", 0)
        arable = props.get("arable_land", 0)
        move = props.get("movement_cost", 0)
        fresh_water = props.get("fresh_water", 0)
        food_veg = props.get("food_vegetal", 0)
        food_animal = props.get("food_animal", 0)
        wood = props.get("wood_yield", 0)
        stone = props.get("stone_yield", 0)
        ore = props.get("ore_yield", 0)

        avg_food = (food_veg + food_animal) / 2

        text = (
            f"--- Клетка ({i}, {j}) ---\n"
            f"Биом: {biome}\n"
            f"Высота: {elev:.0f} м\n\n"
            f"Пригодность: {habit:.2f}\n"
            f"Земледелие: {arable:.2f}\n"
            f"Стоимость движения: {move:.2f}\n\n"
            f"Пресная вода: {fresh_water:.2f}\n"
            f"Еда (средняя): {avg_food:.2f}\n"
            f"  растительная: {food_veg:.2f}\n"
            f"  животная: {food_animal:.2f}\n\n"
            f"Ресурсы:\n"
            f"  Древесина: {wood:.2f}\n"
            f"  Камень: {stone:.2f}\n"
            f"  Руда: {ore:.2f}\n"
        )

        # --- Поиск объекта в клетке ---
        selected_entity = None
        for e in self.current_entities:
            if e["i"] == i and e["j"] == j:
                selected_entity = e
                break

        # --- Поиск государства, которому принадлежит клетка ---
        clicked_state = None
        for e in self.current_entities:
            if e["stage"] == "state" and "territory" in e:
                if [i, j] in e["territory"]:
                    clicked_state = e
                    break

        # --- Формирование информации ---
        if selected_entity:
            e = selected_entity
            text += f"=== Объект: {e['stage'].upper()} #{e['id']} ===\n"
            text += f"Население: {e.get('population', 0)}\n"
            text += f"Технологии: {e.get('tech', 0):.3f}\n"
            if e["stage"] == "state":
                text += f"Территория: {len(e.get('territory', []))} клеток\n"

        elif clicked_state:
            e = clicked_state
            text += f"=== Государство #{e['id']} ===\n"
            text += f"Население: {e.get('population', 0)}\n"
            text += f"Технологии: {e.get('tech', 0):.3f}\n"
            text += f"Размер территории: {len(e.get('territory', []))} клеток\n"
            text += f"Столица: ({e['i']}, {e['j']})\n"

            # 🔹 Сохраняем выбранное государство для последующей подсветки
            self.highlight_state = e
            self.draw_frame(self.frame_index)
            self.canvas.update()

        else:
            text += "Объектов в этой клетке нет.\n"
            self.highlight_state = None
            self.draw_frame(self.frame_index)
            self.canvas.update()


        # --- Обновляем текст в интерфейсе ---
        self.info_box.setPlainText(text)

# === Запуск ===
if __name__ == "__main__":
    qapp = QtWidgets.QApplication([])
    w = LogViewer()
    w.show()
    app.run()
