import json
import random
import math
from dataclasses import dataclass
from biomes_properties import BIOME_DATA
from config import *

# =======================================
# === 1. ЗАГРУЗКА КАРТЫ МИРА ============
# =======================================

@dataclass
class WorldCell:
    i: int
    j: int
    biome: str
    elevation: float
    properties: dict

    @property
    def is_land(self):
        return not self.properties.get("is_ocean", False)

    @property
    def food_availability(self):
        """Совокупная доступность еды (растительной и животной)."""
        return (self.properties.get("food_vegetal", 0) + self.properties.get("food_animal", 0)) / 2

    @property
    def water_availability(self):
        """Пресная вода из биома."""
        return self.properties.get("fresh_water", 0)

    @property
    def habitability(self):
        """Общая пригодность для жизни."""
        return self.properties.get("habitability", 0)

    @property
    def movement_cost(self):
        """Затраты на перемещение по биому."""
        return self.properties.get("movement_cost", 1.0)
    
    @property
    def arable(self):
        """Пригодность к земледелию."""
        return self.properties.get("arable_land", 1.0)


def load_world(filename="world_cells.json"):
    with open(filename, "r") as f:
        raw = json.load(f)

    world = {}
    for c in raw:
        biome_name = c["biome"]
        props = BIOME_DATA.get(biome_name, BIOME_DATA["Plains"]).copy()
        world[(c["i"], c["j"])] = WorldCell(
            i=c["i"],
            j=c["j"],
            biome=biome_name,
            elevation=c["elevation_m"],
            properties=props
        )
    return world


# =======================================
# === 2. БАЗОВЫЙ КЛАСС СОЦИАЛЬНОГО ОБЪЕКТА
# =======================================

class BaseEntity:
    """Базовый класс для всех социальных образований: группы, племени, города, государства."""

    def __init__(self, entity_id, i, j, population):
        self.id = entity_id
        self.i = i
        self.j = j
        self.population = population
        self.food = 0.0
        self.water = 0.0
        self.stage = "group"  # group / tribe / city / state
        self.alive = True

    def move_to(self, i, j):
        """Смещение объекта в новую клетку."""
        self.i = i
        self.j = j

    def __repr__(self):
        return f"<{self.stage.capitalize()} #{self.id} ({self.population} ppl) at ({self.i},{self.j}) have ({self.food}, {self.water})>"


# =======================================
# === 3. КЛАСС ГРУППЫ ЛЮДЕЙ ============
# =======================================

class HumanGroup(BaseEntity):
    """Кочевая группа людей, способная перемещаться, охотиться и искать место для поселения."""

    def __init__(self, entity_id, i, j, population):
        super().__init__(entity_id, i, j, population)
        self.food = 200.0
        self.hunger = 0.0
        self.age = 0
        self.tech = 0.0
        self.path = [(i, j)]

    def step(self, world):
        """Основной шаг жизнедеятельности группы."""
        if not self.alive:
            return

        cell = world.get((self.i, self.j))
        if not cell or not cell.is_land:
            self.alive = False
            return

        # === 1. Охота и собирательство ===
        self.hunt(cell)

        # === 2. Питьевая вода ===
        self.find_water(cell)

        # === 3. Питание и смертность ===
        self.consume_resources()

        # === 4. Естественный рост населения ===
        self.update_population(cell)

        # === 5. Рост технологий ===
        self.tech += 0.001 * cell.habitability

        # === 6. Решение осесть ===
        if self.should_settle(cell):
            from simulation import Tribe  # чтобы избежать циклических импортов
            print(f"Группа {self.id} основала племя в ({self.i},{self.j})")
            new_tribe = Tribe(self.id, self.i, self.j, int(self.population))
            self.alive = False  # группа превращается в племя
            return new_tribe


        # === 7. Перемещение ===
        self.maybe_move(world)
        self.record_path()
        self.age += 1
        if self.population <= 0 or self.hunger >= 1.0:
            self.alive = False
            print(f"Группа {self.id} погибла.")
            return

    # ----------------------------
    # === ВНУТРЕННЯЯ ЛОГИКА ===
    # ----------------------------

    def hunt(self, cell: WorldCell):
        """Охота и собирательство (меньше еды в горах и пустынях, больше в лесах и равнинах)."""
        biome_factor = (cell.food_availability + cell.habitability) / 2
        random_factor = random.uniform(0.6, 1.4)
        # население увеличивает добычу, но с убывающей отдачей
        population_factor = (self.population / 200) ** 0.5
        gain = biome_factor * random_factor * population_factor * 5
        self.food += gain
        self.hunger = max(0.0, self.hunger - 0.02)

    def find_water(self, cell: WorldCell):
        if cell.water_availability > 0.3:
            self.water = min(1.0, getattr(self, "water", 1.0) + 0.1)
        else:
            self.water = max(0.0, getattr(self, "water", 1.0) - 0.05)
            self.hunger += 0.01

    def consume_resources(self):
        need = self.population * 0.005
        if self.food >= need:
            self.food -= need
            self.hunger = max(0.0, self.hunger - 0.01)
        else:
            deficit = (need - self.food) / need
            self.food = 0
            self.hunger = min(1.0, self.hunger + 0.1 + deficit * 0.2)
            death_rate = 0.01 + deficit * 0.05
            self.population *= (1 - death_rate)
            self.population = int(self.population)

    def update_population(self, cell: WorldCell):
        """Естественный рост/смертность в зависимости от условий."""
        if self.hunger < 0.3 and cell.habitability > 0.5:
            growth = 0.005 * (1 + cell.habitability)
            self.population *= (1 + growth)
        else:
            decline = 0.001 + self.hunger * 0.01
            self.population *= (1 - decline)
        # ограничиваем минимальную и максимальную численность
        self.population = max(10, min(self.population, 5000))
        self.population = int(self.population)

    def should_settle(self, cell: WorldCell):
        """Решение осесть (образовать племя)."""
        print(f'Анализ местности {self.i},{self.j}: {cell.habitability}, {cell.arable}, {1 - self.hunger} = {cell.habitability * cell.arable * (1 - self.hunger)}')
        if cell.habitability * cell.arable * (1 - self.hunger) * random.random() > 0.4:
            return True
        return False

    def maybe_move(self, world):
        """Перемещение — выбор соседней клетки."""
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        random.shuffle(dirs)
        best = None
        best_score = -9999
        for dx, dy in dirs:
            nx, ny = self.i + dx, self.j + dy
            cell = world.get((nx, ny))
            if not cell or not cell.is_land:
                continue
            score = cell.habitability + cell.food_availability - cell.movement_cost * 0.2
            if score > best_score:
                best_score = score
                best = (nx, ny)
        if best and random.random() < 0.4:
            self.move_to(*best)
    
    def record_path(self):
        """Сохраняем позицию в историю пути (ограничена длиной 100 точек)."""
        if not self.path or self.path[-1] != (self.i, self.j):
            self.path.append((self.i, self.j))
        if len(self.path) > 100:
            self.path.pop(0)

    def get_path_points(self):
        """Возвращает путь в виде списка координат (i, j) для визуализации."""
        return list(self.path)


# =======================================
# === 4. КЛАСС ПЛЕМЕНИ ==================
# =======================================

class Tribe(BaseEntity):
    """Оседлое племя, которое занимается земледелием и растёт быстрее."""

    def __init__(self, entity_id, i, j, population):
        super().__init__(entity_id, i, j, population)
        self.stage = "tribe"
        self.food = 300.0
        self.tech = 0.05
        self.age = 0

    def step(self, world):
        cell = world.get((self.i, self.j))
        if not cell:
            self.alive = False
            return

        # Производство еды
        food_gain = (cell.arable + cell.habitability) * 5 * random.uniform(0.8, 1.2)
        self.food += food_gain

        # Потребление
        need = self.population * 0.004
        if self.food >= need:
            self.food -= need
        else:
            deficit = (need - self.food) / need
            self.food = 0
            self.population *= (1 - 0.01 - deficit * 0.05)

        # Рост населения
        growth_rate = 0.003 + cell.arable * 0.002
        self.population *= (1 + growth_rate)

        # Рост технологий
        self.tech += 0.001 * (cell.habitability + cell.arable)

        # Возможное превращение в город
        if self.population > CITY_FOUNDING_THRESHOLD:
            self.stage = "city"
            print(f"Племя #{self.id} выросло в город ({self.i},{self.j})!")
            return "city"

        self.population = int(min(self.population, 20000))
        self.age += 1
        return "tribe"




# =======================================
# === 4. ИНИЦИАЛИЗАЦИЯ И ТЕСТ ==========
# =======================================

if __name__ == "__main__":
    print("Загружаю карту...")
    world = load_world()
    print(f"Мир загружен ({len(world)} клеток)")

    # создаем первую группу
    g = HumanGroup(0, *STARTING_CELL_COORDS, STARTING_POPULATION)

    print("Начинаю симуляцию...")
    for year in range(START_YEAR, START_YEAR + 500, SIMULATION_STEP_YEARS):
        g.step(world)
        if not g.alive:
            print(f"Группа вымерла к {year} году")
            break
        print(f"{year}: {g}")
