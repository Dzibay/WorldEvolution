import json
import os
import gzip

import numpy as np
from vispy import app, scene
from vispy.scene.visuals import Text  # пока не используем, но пусть будет
from PyQt6 import QtWidgets, QtCore

from biomes_properties import BIOME_DATA

app.use_app("pyqt6")  # важно для корректной интеграции с PyQt6


class LogViewer(QtWidgets.QMainWindow):
    """
    Лог-вьюер с диагностическими выводами.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("WorldEvolution Log Viewer")
        self.resize(1900, 1050)

        # === Центральный layout ===
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # === Левая часть — VisPy Canvas ===
        self.canvas = scene.SceneCanvas(
            keys="interactive", bgcolor="black", show=False, size=(1500, 1000)
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45, azimuth=0, elevation=30, distance=3
        )
        layout.addWidget(self.canvas.native, stretch=4)
        self.canvas.events.mouse_press.connect(self.on_mouse_click)

        # === Правая панель управления ===
        side = QtWidgets.QVBoxLayout()
        layout.addLayout(side, stretch=2)
        self.side = side

        # Заголовки / выбор лога / кадра
        self.label_year = QtWidgets.QLabel("Год: —")
        self.label_year.setStyleSheet("color:white; font-size:16px;")
        side.addWidget(self.label_year)

        self.label_global = QtWidgets.QLabel("Мир: —")
        self.label_global.setStyleSheet("color:#cccccc; font-size:11px;")
        side.addWidget(self.label_global)

        # Список логов
        self.combo_log = QtWidgets.QComboBox()
        self.refresh_log_files()
        side.addWidget(self.combo_log)

        # Слайдер по кадрам
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        side.addWidget(self.slider)

        # Кнопки управления
        controls = QtWidgets.QHBoxLayout()
        side.addLayout(controls)
        self.btn_play = QtWidgets.QPushButton("▶ / ⏸")
        self.btn_fast = QtWidgets.QPushButton("⏩")
        self.btn_slow = QtWidgets.QPushButton("⏪")
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_fast)
        controls.addWidget(self.btn_slow)

        side.addSpacing(10)

        # Блок заголовка выбранного объекта
        self.label_selected = QtWidgets.QLabel("Выбрано: —")
        self.label_selected.setStyleSheet("color:#00ffff; font-size:13px;")
        side.addWidget(self.label_selected)

        # Информационный блок (нижняя большая панель)
        self.info_box = QtWidgets.QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setStyleSheet(
            """
            background-color: #111;
            color: #FFD700;
            font-family: Consolas;
            font-size: 12px;
            border: 1px solid #333;
        """
        )
        side.addWidget(self.info_box, stretch=2)

        side.addStretch(1)

        # === Сигналы ===
        self.combo_log.activated.connect(self.on_log_selected)
        self.slider.valueChanged.connect(self.slider_changed)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_fast.clicked.connect(self.speed_up)
        self.btn_slow.clicked.connect(self.speed_down)

        # === Данные мира и логов ===
        self.timer = app.Timer(
            interval=0.5, connect=self.update_frame, start=False
        )
        self.load_world()
        self.current_log = []
        self.current_entities = []
        self.current_summary = {}
        self.state_index = {}

        self.frame_index = 0
        self.paused = False
        self.speed_factor = 1.0

        # === Визуальные объекты ===
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.capital_markers = scene.visuals.Markers(parent=self.view.scene)

        # Линии дипломатии (между столицами)
        self.diplomacy_lines = scene.visuals.Line(
            parent=self.view.scene, connect="segments", width=2
        )

        # Цвета стран и выделение
        self.state_color_map = {}
        self.highlight_color = (1.0, 1.0, 0.0, 1.0)

        # Выбор пользователем
        self.selected_cell = None        # (i, j)
        self.selected_entity_id = None   # id
        self.selected_state_id = None    # id
        self.highlight_state_id = None   # id

        print("[INIT] Markers parent:", self.markers.parent)
        print("[INIT] Capital markers parent:", self.capital_markers.parent)

    # ------------------------------------------------------------------
    #  Инициализация / загрузка мира / логов
    # ------------------------------------------------------------------

    def refresh_log_files(self):
        if not os.path.isdir("logs"):
            os.makedirs("logs", exist_ok=True)
        files = [
            f
            for f in os.listdir("logs")
            if f.endswith(".json") or f.endswith(".json.gz")
        ]
        files.sort()
        self.combo_log.clear()
        self.combo_log.addItems(files)
        print("[LOG FILES] Найдены логи:", files)

    def load_world(self):
        """Загрузка world_cells.json и генерация сферической сетки."""
        print("[WORLD] Загружаем world_cells.json")
        with open("world_cells.json", "r", encoding="utf-8") as f:
            cells = json.load(f)

        self.nx = max(c["i"] for c in cells) + 1
        self.ny = max(c["j"] for c in cells) + 1
        print(f"[WORLD] nx={self.nx}, ny={self.ny}, cells={len(cells)}")

        # Для быстрого доступа к клетке
        self.world_cells = [[{} for _ in range(self.nx)] for _ in range(self.ny)]
        for c in cells:
            self.world_cells[c["j"]][c["i"]] = c

        # Вершины и цвета (биомы)
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
            self.colors[i, j] = [
                col[0] / 255,
                col[1] / 255,
                col[2] / 255,
                1.0,
            ]

        self.verts = self.points.reshape(-1, 3)
        self.base_cols_flat = self.colors.reshape(-1, 4)

        # Треугольники
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
            vertices=self.verts,
            faces=self.faces,
            vertex_colors=self.base_cols_flat,
            shading="smooth",
            parent=self.view.scene,
        )
        print("[WORLD] Меш Земли создан")

    # ------------------------------------------------------------------
    #  Геометрические утилиты
    # ------------------------------------------------------------------

    def grid_to_xyz(self, i, j, lift: float = 0.002):
        """Координаты на сфере с небольшим смещением наружу."""
        base = self.points[i % self.nx, j % self.ny]
        n = base / np.linalg.norm(base)
        return base + n * lift

    # ------------------------------------------------------------------
    #  Логика загрузки логов
    # ------------------------------------------------------------------

    def on_log_selected(self, index):
        if index < 0:
            return
        filename = self.combo_log.itemText(index)
        self.load_log(filename)

    def load_log(self, filename):
        if not filename:
            return

        full_path = os.path.join("logs", filename)
        print(f"\n📜 Загружаем лог: {full_path}")

        try:
            if filename.endswith(".gz"):
                with gzip.open(full_path, "rt", encoding="utf-8") as f:
                    self.current_log = json.load(f)
            else:
                with open(full_path, "r", encoding="utf-8") as f:
                    self.current_log = json.load(f)
        except Exception as e:
            print("Ошибка при чтении лога:", e)
            self.current_log = []
            return

        if not self.current_log:
            print("⚠ Лог пустой.")
            return

        print("[LOG] Кадров:", len(self.current_log))
        first = self.current_log[0]
        print("[LOG] Frame0 year:", first.get("year"))
        print("[LOG] Frame0 entities:", len(first.get("entities", [])))
        if first.get("entities"):
            print("[LOG] Пример сущности frame0:", first["entities"][0])

        self.slider.setRange(0, len(self.current_log) - 1)
        self.slider.setValue(0)
        self.frame_index = 0

        # Сбрасываем выбор
        self.selected_cell = None
        self.selected_entity_id = None
        self.selected_state_id = None
        self.highlight_state_id = None

        # Отрисовываем первый кадр
        self.draw_frame(0)

        # Запускаем проигрывание
        self.paused = False
        if not self.timer.running:
            self.timer.start()

        print(f"✅ Лог успешно загружен ({len(self.current_log)} кадров)")

    # ------------------------------------------------------------------
    #  Отрисовка кадра
    # ------------------------------------------------------------------

    def draw_frame(self, idx: int):
        try:
            print(f"[DRAW] start draw_frame({idx})")

            if not self.current_log or self.earth_mesh is None:
                print('no data or earth_mesh is None')
                print(True if self.current_log else False, self.earth_mesh)
                return

            if getattr(self.earth_mesh, "mesh_data", None) is None:
                # OpenGL ещё не готов — пробуем позже
                QtCore.QTimer.singleShot(100, lambda: self.draw_frame(idx))
                return

            snap = self.current_log[idx]
            year = snap.get("year", 0)
            self.current_entities = snap.get("entities", [])
            self.current_summary = snap.get("summary", {})
            self.frame_index = idx

            # Диагностика по кадру
            stage_counts = {}
            for e in self.current_entities:
                st = e.get("stage", "None")
                stage_counts[st] = stage_counts.get(st, 0) + 1

            print(
                f"[FRAME] idx={idx}, year={year}, "
                f"entities={len(self.current_entities)}, "
                f"stage_counts={stage_counts}"
            )

            # Сбрасываем цвета
            mesh_cols = self.base_cols_flat.copy()

            # Индекс государств по id
            self.state_index = {
                e["id"]: e for e in self.current_entities if e.get("stage") == "state"
            }

            self.label_year.setText(f"Год: {year}")

            # Немного глобальной статистики
            if self.current_summary:
                total_pop = self.current_summary.get("total_population", 0)
                total_entities = self.current_summary.get("total_entities", 0)
                stages = self.current_summary.get("stages", {})
                states_cnt = stages.get("state", 0)
                cities_cnt = stages.get("city", 0)
                tribes_cnt = stages.get("tribe", 0)
                groups_cnt = stages.get("group", 0)
                self.label_global.setText(
                    f"Объектов: {total_entities} | Население: {total_pop} | "
                    f"Гос-в: {states_cnt}, городов: {cities_cnt}, племён: {tribes_cnt}, групп: {groups_cnt}"
                )
            else:
                self.label_global.setText("Мир: —")

            # --- Раскрашиваем территории государств ---
            for e in self.current_entities:
                if e.get("stage") != "state":
                    continue

                state_id = e["id"]
                terr = e.get("territory", [])
                # Назначаем цвет
                if state_id not in self.state_color_map:
                    np.random.seed(state_id)
                    rgb = np.random.rand(3) * 0.8 + 0.2
                    self.state_color_map[state_id] = rgb
                else:
                    rgb = self.state_color_map[state_id]

                if self.highlight_state_id is not None and state_id == self.highlight_state_id:
                    rgb = np.clip(rgb + 0.3, 0, 1)

                color_rgba = (rgb[0], rgb[1], rgb[2], 1.0)

                for (ti, tj) in terr:
                    vertex_idx = ti * self.ny + tj
                    if 0 <= vertex_idx < len(mesh_cols):
                        mesh_cols[vertex_idx] = color_rgba

            # --- Обновляем цвета планеты ---
            if getattr(self.earth_mesh, "mesh_data", None):
                self.earth_mesh.mesh_data.set_vertex_colors(mesh_cols)
                self.earth_mesh.mesh_data_changed()
                self.earth_mesh.update()

            # === ОТРИСОВКА АГЕНТОВ ===
            positions = []
            colors_list = []
            sizes_list = []

            for e in self.current_entities:
                stage = e.get("stage", "")
                i, j = e.get("i"), e.get("j")

                if i is None or j is None:
                    continue

                pos = self.grid_to_xyz(i, j)

                if stage == "group":
                    color = (1, 0, 0, 1)
                    size = 6
                elif stage == "tribe":
                    color = (1, 1, 0, 1)
                    size = 8
                elif stage == "city":
                    color = (0, 1, 0, 1)
                    size = 10
                elif stage == "seafaring":
                    color = (1, 1, 1, 1)
                    size = 8
                else:
                    continue  # пропускаем state и неизвестные

                positions.append(pos)
                colors_list.append(color)
                sizes_list.append(size)

            print(
                f"[MARKERS] entities_for_markers={len(positions)} "
                f"(из {len(self.current_entities)})"
            )
            if positions:
                # выведем пример
                print("[MARKERS] Пример позиции:", positions[0])
                self.markers.set_data(
                    np.array(positions),
                    face_color=np.array(colors_list),
                    size=np.array(sizes_list),
                )
                print(
                    "[MARKERS] set_data called: pos.shape=",
                    np.array(positions).shape,
                    "sizes.shape=",
                    np.array(sizes_list).shape,
                )
            else:
                # для диагностики явно очищаем и пишем в лог
                self.markers.set_data(np.empty((0, 3)))
                print("[MARKERS] Пустой список позиций, маркеры очищены")

            # --- Столицы государств ---
            capital_positions = []
            capital_colors = []
            capital_sizes = []

            for e in self.current_entities:
                if e.get("stage") != "state":
                    continue
                i, j = e.get("i", 0), e.get("j", 0)
                pos = self.grid_to_xyz(i, j, lift=0.004)
                tech = float(e.get("tech", 0.0))

                base_size = 6 + tech * 8
                color = (1.0, 0.9, 0.3, 1.0)

                if self.highlight_state_id is not None and e["id"] == self.highlight_state_id:
                    base_size *= 1.7
                    color = (1.0, 1.0, 0.7, 1.0)

                capital_positions.append(pos)
                capital_colors.append(color)
                capital_sizes.append(base_size)

            if capital_positions:
                self.capital_markers.set_data(
                    np.array(capital_positions),
                    face_color=np.array(capital_colors),
                    size=np.array(capital_sizes),
                    symbol="star",
                    edge_color="white",
                )
            else:
                self.capital_markers.set_data(np.empty((0, 3)))

            # --- Обновляем линии дипломатии ---
            self.update_diplomacy_visuals()

            # --- Обновляем текстовую информацию о выбранной клетке/объекте ---
            self.update_info_panel()

            # Сообщаем камере / сцене, что всё поменялось
            self.view.camera.view_changed()
            self.canvas.update()

        except Exception as e:
            import traceback
            print(f"❌ EXCEPTION IN draw_frame({idx}):")
            traceback.print_exc()
            return


    # ------------------------------------------------------------------
    #  Дипломатия: линии между столицами
    # ------------------------------------------------------------------

    def update_diplomacy_visuals(self):
        # 1. Проверки на наличие выбранного государства
        if (
            self.highlight_state_id is None
            or self.highlight_state_id not in self.state_index
        ):
            self.diplomacy_lines.set_data(pos=np.empty((0, 3)))
            return

        st = self.state_index[self.highlight_state_id]
        neighbors = st.get("neighbors", [])

        if not neighbors:
            self.diplomacy_lines.set_data(pos=np.empty((0, 3)))
            return

        # 2. Находим сущность государства в current_entities, чтобы прочитать список "at_war"
        # (в state_index лежит урезанная копия, или та же самая - лучше найти наверняка)
        my_state_entity = next((e for e in self.current_entities if e["id"] == self.highlight_state_id), {})
        at_war_list = my_state_entity.get("at_war", [])

        pos_list = []
        color_list = []  # <--- Список для цветов каждой вершины

        my_cap = self.grid_to_xyz(st.get("i", 0), st.get("j", 0), lift=0.008)

        # 3. Проходим по соседям
        for nb in neighbors:
            nb_id = nb.get("id")
            if nb_id not in self.state_index:
                continue
            
            other = self.state_index[nb_id]
            other_cap = self.grid_to_xyz(
                other.get("i", 0), other.get("j", 0), lift=0.008
            )

            # Добавляем координаты (начало и конец отрезка)
            pos_list.append(my_cap)
            pos_list.append(other_cap)

            # --- ЛОГИКА ЦВЕТА ---
            # Если сосед в списке врагов - Красный, иначе - Голубой
            if nb_id in at_war_list:
                col = (1.0, 0.0, 0.0, 1.0) # 🔴 Красный (воюем)
            else:
                col = (0.4, 0.8, 1.0, 0.5) # 🔵 Голубой (мир/нейтралитет)

            # VisPy требует цвет для каждой вершины линии
            color_list.append(col) 
            color_list.append(col)

        if not pos_list:
            self.diplomacy_lines.set_data(pos=np.empty((0, 3)))
            return

        # 4. Передаем массивы в VisPy
        pos_arr = np.array(pos_list)
        col_arr = np.array(color_list)  # <--- Превращаем список цветов в массив

        # ВАЖНО: передаем color=col_arr, а не фиксированный кортеж!
        self.diplomacy_lines.set_data(pos=pos_arr, color=col_arr)

    # ------------------------------------------------------------------
    #  Обновление info_box по текущему кадру и выбранному объекту
    # ------------------------------------------------------------------

    def update_info_panel(self):
        text_lines = []

        if self.selected_cell is None:
            if self.current_summary:
                s = self.current_summary
                text_lines.append("=== Общая сводка по миру ===")
                text_lines.append(
                    f"Год: {self.current_log[self.frame_index].get('year', 0)}"
                )
                text_lines.append(
                    f"Всего объектов: {s.get('total_entities', 0)}"
                )
                text_lines.append(
                    f"Население: {s.get('total_population', 0)} "
                    f"(ср.: {s.get('avg_population', 0)}, макс: {s.get('max_population', 0)})"
                )
                text_lines.append(
                    f"Средний уровень технологий: {s.get('avg_tech', 0.0):.4f}"
                )
                stages = s.get("stages", {})
                text_lines.append(
                    f"Стадии: {stages.get('group', 0)} групп, "
                    f"{stages.get('tribe', 0)} племён, "
                    f"{stages.get('city', 0)} городов, "
                    f"{stages.get('state', 0)} государств, "
                    f"{stages.get('seafaring', 0)} мореплавателей"
                )
            else:
                text_lines.append("Нет выбранного объекта.")
            self.info_box.setPlainText("\n".join(text_lines))
            self.label_selected.setText("Выбрано: —")
            return

        i, j = self.selected_cell
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

        text_lines.append(f"--- Клетка ({i}, {j}) ---")
        text_lines.append(f"Биом: {biome}")
        text_lines.append(f"Высота: {elev:.0f} м")
        text_lines.append("")
        text_lines.append(f"Пригодность: {habit:.2f}")
        text_lines.append(f"Земледелие (arable): {arable:.2f}")
        text_lines.append(f"Стоимость движения: {move:.2f}")
        text_lines.append("")
        text_lines.append(f"Пресная вода: {fresh_water:.2f}")
        text_lines.append(f"Еда (средняя): {avg_food:.2f}")
        text_lines.append(f"  растительная: {food_veg:.2f}")
        text_lines.append(f"  животная:   {food_animal:.2f}")
        text_lines.append("")
        text_lines.append("Ресурсы:")
        text_lines.append(f"  Древесина: {wood:.2f}")
        text_lines.append(f"  Камень:    {stone:.2f}")
        text_lines.append(f"  Руда:      {ore:.2f}")
        text_lines.append("")

        # --- Поиск выбранного объекта / государства в текущем кадре ---
        selected_entity = None
        if self.selected_entity_id is not None:
            for e in self.current_entities:
                if e["id"] == self.selected_entity_id:
                    selected_entity = e
                    break

        if selected_entity is None:
            for e in self.current_entities:
                if e.get("i") == i and e.get("j") == j:
                    selected_entity = e
                    break

        owner_state = None
        if self.selected_state_id is not None and self.selected_state_id in self.state_index:
            owner_state = self.state_index[self.selected_state_id]
        else:
            for e in self.current_entities:
                if e.get("stage") == "state" and [i, j] in e.get("territory", []):
                    owner_state = e
                    break

        if selected_entity:
            st = selected_entity.get("stage", "?")
            st_id = selected_entity.get("id")
            stage_ru = {
                "group": "Группа",
                "tribe": "Племя",
                "city": "Город",
                "state": "Государство",
                "seafaring": "Мореплаватели",
            }.get(st, st)
            self.label_selected.setText(
                f"Выбрано: {stage_ru} #{st_id} @ ({selected_entity.get('i')},{selected_entity.get('j')})"
            )
        elif owner_state:
            self.label_selected.setText(
                f"Выбрано: Государство #{owner_state.get('id')} (по территории клетки)"
            )
        else:
            self.label_selected.setText("Выбрано: клетка без объектов")

        if selected_entity:
            e = selected_entity
            st = e.get("stage")
            text_lines.append("=== Объект ===")
            text_lines.append(
                f"{self.label_selected.text().replace('Выбрано: ', '')}"
            )
            text_lines.append(f"Стадия: {st}")
            text_lines.append(f"Население: {e.get('population', 0)}")
            text_lines.append(f"Технологии: {float(e.get('tech', 0.0)):.3f}")
            text_lines.append(f"Возраст: {e.get('age', 0)} лет")

            if "hunger" in e or "thirst" in e:
                text_lines.append(
                    f"Голод: {e.get('hunger', 0.0):.3f}, Жажда: {e.get('thirst', 0.0):.3f}"
                )
            if "food" in e or "water" in e:
                text_lines.append(
                    f"Запасы еды: {e.get('food', 0.0):.2f}, воды: {e.get('water', 0.0):.2f}"
                )

            if st == "group":
                text_lines.append("Тип: кочевая группа")
                if "is_migrating" in e:
                    text_lines.append(
                        f"Мигрирует: {bool(e.get('is_migrating'))}, шагов в пути: {e.get('steps_migrating', 0)}"
                    )
            elif st == "tribe":
                text_lines.append("Тип: оседлое племя")
            elif st == "city":
                text_lines.append("Тип: город")
                if "influence_radius" in e:
                    text_lines.append(
                        f"Радиус влияния: {e.get('influence_radius', 0)}"
                    )
            elif st == "seafaring":
                text_lines.append("Тип: мореплаватели")

            text_lines.append("")

        if owner_state:
            s = owner_state
            text_lines.append("=== Государство (владелец территории) ===")
            text_lines.append(f"ID: {s.get('id')}")
            text_lines.append(f"Население: {s.get('population', 0)}")
            text_lines.append(f"Технологии: {float(s.get('tech', 0.0)):.3f}")
            text_lines.append(f"Возраст: {s.get('age', 0)} лет")
            text_lines.append(
                f"Размер территории: {len(s.get('territory', []))} клеток"
            )
            text_lines.append(
                f"Число городов: {len(s.get('cities', []))}, Выход к морю: {bool(s.get('is_coastal', False))}"
            )
            text_lines.append(
                f"Бюджет экспансии: {float(s.get('expansion_budget', 0.0)):.3f}"
            )

            macro = s.get("macro", {})
            if macro:
                text_lines.append("")
                text_lines.append("— Макроэкономика —")
                text_lines.append(f"Клеток в территории: {macro.get('cells', 0)}")
                text_lines.append(
                    f"Средняя пригодность: {macro.get('avg_habitability', 0.0):.3f}"
                )
                text_lines.append(
                    f"Средняя плодородность: {macro.get('avg_arable', 0.0):.3f}"
                )
                text_lines.append(
                    f"Суммарный индекс еды: {macro.get('total_food_index', 0.0):.3f}"
                )
                text_lines.append(
                    f"Эффективная вместимость: {macro.get('effective_capacity', 0.0):.1f}"
                )
                text_lines.append(
                    f"Поп/вместимость: {macro.get('population_capacity_ratio', 0.0):.3f}"
                )
                text_lines.append(
                    f"Пр-во еды: {macro.get('food_production', 0.0):.1f}, "
                    f"потребность: {macro.get('food_needed', 0.0):.1f}"
                )
                text_lines.append(
                    f"Профицит еды (доля): {macro.get('food_surplus_ratio', 0.0):+.3f}"
                )
                text_lines.append(
                    f"Годовой темп роста (оценка): {macro.get('yearly_growth_rate', 0.0):+.4f}"
                )

            neighbors = s.get("neighbors", [])
            if neighbors:
                text_lines.append("")
                text_lines.append("— Дипломатия —")
                idx = self.state_index
                for nb in neighbors:
                    nb_id = nb.get("id")
                    border_len = nb.get("border", 0)
                    other = idx.get(nb_id)
                    if not other:
                        continue
                    pop_our = s.get("population", 0)
                    pop_their = other.get("population", 0)
                    tech_our = float(s.get("tech", 0.0))
                    tech_their = float(other.get("tech", 0.0))

                    rel_pop = (
                        "≈"
                        if pop_their and abs(pop_their - pop_our) / pop_their < 0.2
                        else ">"
                        if pop_our > pop_their
                        else "<"
                    )
                    rel_tech = (
                        "≈"
                        if abs(tech_our - tech_their) < 0.03
                        else ">"
                        if tech_our > tech_their
                        else "<"
                    )

                    threat_score = 0
                    if pop_their > pop_our * 1.3:
                        threat_score += 1
                    if tech_their > tech_our + 0.05:
                        threat_score += 1
                    if border_len > 25:
                        threat_score += 1

                    if threat_score == 0:
                        rel_str = "Нейтрально"
                    elif threat_score == 1:
                        rel_str = "Соперничество"
                    else:
                        rel_str = "Высокая напряжённость"

                    text_lines.append(
                        f"Сосед #{nb_id}: граница {border_len} клеток | "
                        f"Население: {pop_our} {rel_pop} {pop_their}, "
                        f"Технологии: {tech_our:.3f} {rel_tech} {tech_their:.3f} "
                        f"→ {rel_str}"
                    )

        self.info_box.setPlainText("\n".join(text_lines))

    # ------------------------------------------------------------------
    #  Управление воспроизведением
    # ------------------------------------------------------------------

    def update_frame(self, event):
        try:
            if self.paused or not self.current_log:
                return

            next_idx = self.frame_index + 1
            if next_idx >= len(self.current_log):
                next_idx = len(self.current_log) - 1
                self.draw_frame(next_idx)
                self.timer.stop()
                self.paused = True
                print("⏹ Конец лога, воспроизведение остановлено.")
                return

            print("[TIMER] switching to frame", next_idx)
            self.draw_frame(next_idx)

            self.slider.blockSignals(True)
            self.slider.setValue(next_idx)
            self.slider.blockSignals(False)

        except Exception as e:
            import traceback
            print("❌ EXCEPTION IN update_frame:")
            traceback.print_exc()
            self.timer.stop()
            self.paused = True


    def slider_changed(self, val):
        if not self.current_log:
            return
        self.draw_frame(val)

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

    # ------------------------------------------------------------------
    #  Пикинг по сфере (выбор клетки и объекта)
    # ------------------------------------------------------------------

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

        cam_pos = cam.center + r * np.array(
            [
                np.cos(phi) * np.sin(theta),
                -np.cos(phi) * np.cos(theta),
                np.sin(phi),
            ],
            dtype=np.float32,
        )
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
        print(f"[CLICK] pos={pos}")
        ray_origin, ray_dir = self.compute_ray_from_click(
            self.view, self.canvas, pos
        )

        R = 1.0
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(ray_origin, ray_dir)
        c = np.dot(ray_origin, ray_origin) - R * R
        delta = b * b - 4 * a * c
        if delta < 0:
            print("[CLICK] Луч не пересёк сферу")
            return
        t = (-b - np.sqrt(delta)) / (2 * a)
        hit_point = ray_origin + t * ray_dir
        x, y, z = hit_point

        theta = np.arctan2(y, x)
        if theta < 0:
            theta += 2 * np.pi
        phi = np.arctan2(z, np.sqrt(x * x + y * y))

        i = int(np.rint(theta / (2 * np.pi) * (self.nx - 1))) % self.nx
        j = int(np.rint((np.pi / 2 - phi) / np.pi * (self.ny - 1)))
        j = int(np.clip(j, 0, self.ny - 1))

        print(f"[CLICK] hit_point={hit_point}, grid=({i},{j})")

        self.selected_cell = (i, j)
        self.selected_entity_id = None
        self.selected_state_id = None
        self.highlight_state_id = None

        clicked_entity = None
        for e in self.current_entities:
            if e.get("i") == i and e.get("j") == j:
                clicked_entity = e
                break

        if clicked_entity:
            print("[CLICK] Найден объект в клетке:", clicked_entity)
            self.selected_entity_id = clicked_entity["id"]
            if clicked_entity.get("stage") == "state":
                self.selected_state_id = clicked_entity["id"]
                self.highlight_state_id = clicked_entity["id"]
        else:
            owner_state = None
            for e in self.current_entities:
                if e.get("stage") == "state" and [i, j] in e.get("territory", []):
                    owner_state = e
                    break
            if owner_state:
                print("[CLICK] Найдено гос-во по территории клетки:", owner_state["id"])
                self.selected_state_id = owner_state["id"]
                self.highlight_state_id = owner_state["id"]

        self.draw_frame(self.frame_index)


# === Запуск ===
if __name__ == "__main__":
    qapp = QtWidgets.QApplication([])
    w = LogViewer()
    w.show()

    # Дать Qt построить GL-контекст перед start()
    QtWidgets.QApplication.processEvents()

    # Теперь GL точно готов, можно запускать таймер
    w.timer.start()

    app.run()

