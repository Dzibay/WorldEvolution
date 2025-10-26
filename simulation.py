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
    # РЕКОМЕНДУЕМЫЕ ГИПЕРПАРАМЕТРЫ ДЛЯ ХОДЬБЫ
    _RECENT_LOOKBACK = 12      # сколько последних позиций штрафуем (анти-петли)
    _BACKTRACK_STRONG_PEN = 3  # сильный штраф за возврат на предыдущую клетку
    _RECENT_PENALTY = 1.2      # базовый штраф за позицию из недавнего пути
    _NOISE = 0.03              # слегка рандомизируем, чтобы не застревать на плато

    def __init__(self, entity_id, i, j, population):
        super().__init__(entity_id, i, j, population)
        self.food = 200.0
        self.hunger = 0.0
        self.age = 0
        self.tech = 0.0
        self.path = [(i, j)]

    def step(self, world, debug=False):
        """Основной шаг жизнедеятельности группы."""
        if not self.alive:
            if debug:
                print(f"[DEBUG] Группа {self.id} мертва, шаг пропущен.")
            return

        cell = world.get((self.i, self.j))
        if not cell or not cell.is_land:
            if debug:
                print(f"[DEBUG] Группа {self.id} находится вне суши ({self.i},{self.j}), погибает.")
            self.alive = False
            return
        
        if debug:
            print("\n" + "="*60)
            print(f"[ГОД {self.age}] Группа #{self.id} | Позиция: ({self.i},{self.j}) | Биом: {cell.biome}")
            print(f"  Население: {self.population}")
            print(f"  Еда: {self.food:.2f}, Вода: {self.water:.2f}, Голод: {self.hunger:.2f}, Технологии: {self.tech:.3f}")
            print(f"  Характеристики клетки: habit={cell.habitability:.2f}, arable={cell.arable:.2f}, water={cell.water_availability:.2f}, food={cell.food_availability:.2f}")

        # === 1. Охота и собирательство ===
        prev_food = self.food
        self.hunt(cell)
        if debug:
            print(f"  [Охота] +{self.food - prev_food:.2f} еды (итого {self.food:.2f})")

        # === 2. Питьевая вода ===
        prev_water = self.water
        self.find_water(cell)
        if debug:
            print(f"  [Вода] {'+' if self.water > prev_water else ''}{self.water - prev_water:.2f} -> {self.water:.2f}")

        # === 3. Питание и смертность ===
        prev_pop, prev_food, prev_hunger = self.population, self.food, self.hunger
        self.consume_resources()
        if debug:
            print(f"  [Питание] Еда: {prev_food:.2f}→{self.food:.2f}, Голод: {prev_hunger:.2f}→{self.hunger:.2f}, Население: {prev_pop}→{self.population}")

        # === 4. Естественный рост населения ===
        prev_pop = self.population
        self.update_population(cell)
        if debug:
            print(f"  [Рост населения] {prev_pop} → {self.population}")

        # === 5. Рост технологий ===
        prev_tech = self.tech
        self.tech += 0.001 * cell.habitability
        if debug:
            print(f"  [Технологии] {prev_tech:.4f} → {self.tech:.4f}")

        # === 6. Решение осесть ===
        settle_decision = cell.habitability * cell.arable * (1 - self.hunger) * random.random()
        if debug:
            print(f"  [Решение осесть] формула={settle_decision:.3f} порог=0.4 → {'ДА' if settle_decision>0.4 else 'нет'}")
        if settle_decision > 0.4:
            from simulation import Tribe
            print(f"Группа {self.id} основала племя в ({self.i},{self.j})")
            new_tribe = Tribe(self.id, self.i, self.j, int(self.population))
            self.alive = False
            return new_tribe

        # === 7. Перемещение ===
        old_pos = (self.i, self.j)
        self.maybe_move(world)
        if debug:
            if (self.i, self.j) != old_pos:
                print(f"  [Перемещение] {old_pos} → ({self.i},{self.j})")
            else:
                print(f"  [Перемещение] осталась на месте")

        # === 8. Обновление возраста и пути ===
        self.record_path()
        self.age += 1

        if self.population <= 0 or self.hunger >= 1.0:
            self.alive = False
            print(f"Группа {self.id} погибла.")
            if debug:
                print(f"  [DEBUG] Причина: {'голод' if self.hunger>=1.0 else 'вымерла численность'}")
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

    def cell_score(self, cell: WorldCell) -> float:
        """
        Интегральная ценность клетки для кочующей группы.
        Весим всё, что помогает выживать на ранних стадиях.
        """
        # Больше еда/вода/обитаемость/пахотность -> лучше.
        # Высокая стоимость перемещения -> хуже.
        return (
            1.5 * cell.habitability +
            1.2 * cell.food_availability +
            0.8 * cell.water_availability +
            0.6 * cell.arable -
            0.8 * cell.movement_cost
        )

    def is_bad_cell(self, cell: WorldCell) -> bool:
        """
        Критерий «очень плохой» клетки -> вынужденная миграция.
        Плохая обитаемость + вода слабая/нет, или острый голод.
        """
        cond_env = (cell.habitability < 0.25 and cell.water_availability < 0.30)
        cond_food = (cell.food_availability < 0.20 and cell.habitability < 0.35)
        cond_hunger = (self.hunger > 0.50)
        return cond_env or cond_food or cond_hunger

    def recent_penalty(self, pos: tuple[int,int]) -> float:
        """
        Штраф за посещение недавних клеток.
        Сильнее всего штрафуем самый последний шаг (анти-назад).
        Ослабляем штраф по мере давности.
        """
        if len(self.path) >= 2 and pos == self.path[-2]:  # прямой возврат
            return self._BACKTRACK_STRONG_PEN

        # ищем pos среди последних _RECENT_LOOKBACK шагов
        recent = self.path[-self._RECENT_LOOKBACK:]
        if pos in recent:
            # чем ближе к концу (т.е. недавно были), тем штраф больше
            # индекс от конца: 1 (совсем недавно) .. N (давно)
            idx_from_end = len(recent) - 1 - recent.index(pos)  # 0..N-1
            age_factor = 1.0 / (1.0 + idx_from_end)            # 1, 1/2, 1/3, ...
            return self._RECENT_PENALTY * (1.0 + age_factor)   # 2.2, 1.8, 1.6, ...
        return 0.0

    def choose_best_neighbor(self, world, debug=False):
        """
        Выбираем лучшего соседа (4 или 8 направлений).
        Можно оставить 4-связность; при желании включить диагонали.
        """
        # dirs = [(1,0),(-1,0),(0,1),(0,-1)]                # 4-связная
        dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]  # 8-связная

        best = None
        best_score = -1e9
        current = (self.i, self.j)

        for dx, dy in dirs:
            nx, ny = self.i + dx, self.j + dy
            pos = (nx, ny)
            cell = world.get(pos)
            if not cell or not cell.is_land:
                continue

            score = self.cell_score(cell)
            pen = self.recent_penalty(pos)
            score_adj = score - pen + random.uniform(-self._NOISE, self._NOISE)

            if debug:
                print(f"    ├─ сосед {pos} | base={score:+.3f} pen={pen:+.2f} → adj={score_adj:+.3f} | biome={cell.biome}")

            if score_adj > best_score:
                best_score = score_adj
                best = pos

        return best, best_score

    def maybe_move(self, world, debug=False):
        """
        Новый алгоритм перемещения:
        1) считаем score текущей клетки;
        2) выбираем лучшего соседа с анти-циклом штрафами;
        3) вероятность миграции растёт с разницей (best - current);
           если клетка «очень плохая» — миграция гарантирована.
        """
        current_cell = world.get((self.i, self.j))
        if not current_cell:
            return

        curr_score = self.cell_score(current_cell)
        bad_now = self.is_bad_cell(current_cell)

        if debug:
            print(f"  [Перемещение] Текущая клетка score={curr_score:+.3f} | bad={bad_now} | biome={current_cell.biome}")

        best_pos, best_score = self.choose_best_neighbor(world, debug=debug)
        if not best_pos:
            if debug:
                print("    └─ подходящих соседей нет — остаёмся на месте")
            return

        # Разница качества между лучшим соседом и текущей клеткой
        delta = max(0.0, best_score - curr_score)

        # Базовая склонность мигрировать (как было раньше ~0.4), но адаптивно:
        # Чем больше delta, тем выше шанс. Сигмоида сглаживает.
        base = 0.20
        adapt = 0.60 * (1 / (1 + math.exp(-2.0 * (delta - 0.3))))  # 0..~0.6
        move_prob = min(0.95, base + adapt)

        # Если клетка очень плохая — идём гарантированно
        if bad_now:
            move_prob = 1.0

        if debug:
            print(f"    ├─ лучший сосед {best_pos} score={best_score:+.3f} | Δ={delta:+.3f}")
            print(f"    └─ вероятность миграции: {move_prob*100:.1f}% ({'FORCED' if bad_now else 'prob'})")

        if random.random() < move_prob:
            self.move_to(*best_pos)
    
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
