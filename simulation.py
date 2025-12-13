import json, random, math
from dataclasses import dataclass
from biomes_properties import BIOME_DATA
from config import *


# =======================================
# === ДОП. КОНСТАНТЫ ДЛЯ МАКРО-ДЕМО ====
# =======================================

# Базовые демографические параметры для государств (в год)
STATE_BASE_FERTILITY = 0.035      # базовая рождаемость (детей на взрослого в год)
STATE_CHILD_MORTALITY = 0.03      # смертность детей в год
STATE_ADULT_MORTALITY = 0.015     # смертность взрослых в год
STATE_ELDER_MORTALITY = 0.06      # смертность пожилых в год

# Возрастные "ширины" (лет) для перехода между группами
CHILD_YEARS = 15.0                # 0-14
ADULT_YEARS = 45.0                # 15-59 (условно)
ELDER_YEARS = 20.0                # 60-79+ (условно)

# Порог для "крупного" государства, где могут начаться эпидемии
EPIDEMIC_POP_THRESHOLD = 80_000
EPIDEMIC_DENSITY_FACTOR = 0.002   # влияние плотности
EPIDEMIC_BASE_CHANCE = 0.0008     # базовый шанс эпидемии на шаг
EPIDEMIC_DECAY_PER_STEP = 0.05    # как быстро эпидемия сходит на нет (0..1)
EPIDEMIC_MORTALITY_MULT = 0.03    # масштаб доп. смертности при эпидемии

# Голод
FAMINE_FOOD_DEFICIT_THRESHOLD = -0.1  # food_surplus_ratio ниже этого => голод
FAMINE_YEARS_SCALE = 50.0             # через сколько лет голода достигается max-эффект
FAMINE_EXTRA_MORTALITY = 0.01         # базовое добавление к смертности при сильном голоде
FAMINE_BIRTH_REDUCTION = 0.3          # насколько падает рождаемость при сильном голоде

# Война
WAR_EXTRA_MORTALITY_PER_ENEMY = 0.01  # доп. смертность от войны (на год) за каждого врага
WAR_MAX_EXTRA_MORTALITY = 0.03        # максимум доп. смертности от войн

# Общественный строй (влияет на рождаемость/смертность)
SOCIETY_TYPES = ("hunter_gatherer", "early_agrarian", "agrarian_empire", "proto_industrial")
SOCIETY_FERTILITY_MULT = {
    "hunter_gatherer": 0.8,
    "early_agrarian": 1.1,
    "agrarian_empire": 1.0,
    "proto_industrial": 0.8,
}
SOCIETY_MORTALITY_MULT = {
    "hunter_gatherer": 1.1,
    "early_agrarian": 0.95,
    "agrarian_empire": 1.0,
    "proto_industrial": 0.9,
}


# =======================================
# === 1. КЛАСС КЛЕТКИ ===================
# =======================================

@dataclass
class WorldCell:
    i: int
    j: int
    biome: str
    elevation: float
    properties: dict
    # Динамические (истощаемые) ресурсы
    current_food_base: float = 0.0
    current_water_base: float = 0.0

    def __post_init__(self):
        self.current_food_base = (
            self.properties.get("food_vegetal", 0) +
            self.properties.get("food_animal", 0)
        )
        self.current_water_base = self.properties.get("fresh_water", 0)

    @property
    def is_land(self):
        return not self.properties.get("is_ocean", False)

    @property
    def is_coastal(self):
        return self.properties.get("is_coastal", False)

    @property
    def food_availability(self):
        return self.current_food_base / 2

    @property
    def water_availability(self):
        return self.current_water_base

    @property
    def habitability(self):
        return self.properties.get("habitability", 0)

    @property
    def movement_cost(self):
        return self.properties.get("movement_cost", 1.0)

    @property
    def arable(self):
        return self.properties.get("arable_land", 1.0)

    def deplete(self, population):
        if self.is_land:
            depletion = population * RESOURCE_DEPLETION_RATE
            self.current_food_base = max(0.0, self.current_food_base - depletion)

    def regenerate(self):
        base_food = (
            self.properties.get("food_vegetal", 0) +
            self.properties.get("food_animal", 0)
        )
        if self.current_food_base < base_food:
            self.current_food_base = min(
                base_food,
                self.current_food_base * (1 + RESOURCE_REGENERATION_RATE)
            )

        base_water = self.properties.get("fresh_water", 0)
        if self.current_water_base < base_water:
            self.current_water_base = min(
                base_water,
                self.current_water_base * (1 + RESOURCE_REGENERATION_RATE)
            )


def load_world(filename, nx=None, ny=None):
    with open(filename) as f:
        raw = json.load(f)
    world = {}

    for c in raw:
        props = BIOME_DATA.get(c["biome"], BIOME_DATA["Plains"]).copy()
        world[(c["i"], c["j"])] = WorldCell(
            c["i"], c["j"], c["biome"], c["elevation_m"], props
        )

    if nx is None or ny is None:
        print("ПРЕДУПРЕЖДЕНИЕ: nx/ny не заданы, 'is_coastal' не вычислен.")
        return world

    print("Вычисление прибрежных зон...")
    for (i, j), cell in world.items():
        if cell.is_land:
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                check_pos = ((i + di) % nx, (j + dj) % ny)
                neighbor = world.get(check_pos)
                if neighbor and not neighbor.is_land:
                    cell.properties["is_coastal"] = True
                    break

    return world


# =======================================
# === 2. БАЗОВЫЙ КЛАСС АГЕНТА ===========
# =======================================

class BaseEntity:
    def __init__(self, entity_id, i, j, population, start_tech=0.01):
        self.id = entity_id
        self.i, self.j = i, j
        self.population = int(population)
        self.prev_population = population
        self.food = max(50.0, population * 0.5)
        self.water = 0.7
        self.tech = start_tech
        self.age = 0
        self.stage = "base"
        self.alive = True
        self.need_food_per_capita = 0.003
        self.hunger_level = 0.0
        self.thirst_level = 0.0
        self.sleep_timer = 0

    def gather_resources(self, cell):
        if cell.is_land:
            base_food = (cell.food_availability + cell.arable * 0.6) * self.population * 0.005
            tech_bonus = 1.0 + self.tech * 2.0
            self.food += base_food * tech_bonus
            cell.deplete(self.population)
        else:
            self.food += cell.properties.get("food_animal", 0) * self.population * 0.0008

        self.water = max(
            0.0,
            min(1.0, self.water * 0.6 + cell.water_availability + random.uniform(0.0, 0.1))
        )

    def consume_resources(self, cell):
        need_food = self.population * self.need_food_per_capita
        if self.food >= need_food:
            self.food -= need_food * (1.0 - FOOD_WASTAGE_RATE)
            self.hunger_level = 0.0
        else:
            deficit = need_food - self.food
            self.food = 0.0
            self.hunger_level = max(
                0.0, min(1.0, deficit / (need_food + 1e-9))
            )
        self.water = max(0.0, self.water - 0.15)
        self.thirst_level = (
            0.0 if self.water >= 0.6
            else max(0.0, min(1.0, (0.6 - self.water) / 0.6))
        )

    def tech_growth(self, cell):
        density_factor = min(1.0, self.population / (CARRYING_CAPACITY_FACTOR * 0.1))
        discovery_chance = TECH_DISCOVERY_CHANCE_BASE * (1 + density_factor * TECH_DENSITY_FACTOR)
        if random.random() < discovery_chance:
            # слегка усиленный прирост, чтобы к ~-15000 появлялись города
            gain = 0.002 * (cell.habitability + cell.arable * 0.5)
            self.tech = min(1.0, self.tech + gain)

    def update_population(self, cell):
        if not self.alive:
            return

        base_capacity = CARRYING_CAPACITY_FACTOR
        tech_capacity_multiplier = 1.0 + (self.tech * 5)

        stage_multiplier = 1.0
        if self.stage == "city":
            stage_multiplier = 2.0
        elif self.stage == "state":
            stage_multiplier = 5.0

        carrying_capacity = max(
            1.0,
            cell.habitability * base_capacity * tech_capacity_multiplier * stage_multiplier
        )

        base_birth = max(0.0, BIRTH_RATE_BASE * (cell.habitability + 0.2) * (1 + self.tech))
        base_death = max(0.0, DEATH_RATE_BASE * (1.0 - cell.habitability * 0.5))
        starvation_term = self.hunger_level * DEATH_RATE_STARVATION
        dehydration_term = self.thirst_level * (DEATH_RATE_STARVATION * 0.5)

        overpop = max(0.0, (self.population / (carrying_capacity + 1e-9)) - 1.0)
        overpop_death = overpop * 0.04

        age_penalty = max(0.8, 1.0 - self.age / 20000)

        yearly_birth = base_birth * age_penalty
        yearly_death = base_death + starvation_term + dehydration_term + overpop_death

        years = max(1, SIMULATION_STEP_YEARS)

        base_rate = 1.0 + yearly_birth - yearly_death
        clamped_base_rate = max(0.0, base_rate)
        growth_factor = clamped_base_rate ** years  # экспонента ок на уровне малых групп

        self.population = int(max(0, math.floor(self.population * growth_factor)))

        if self.population <= 0:
            self.alive = False
            return

        pop_growth = abs(self.population - self.prev_population) / (self.prev_population + 1e-9)
        if self.hunger_level < 0.1 and self.thirst_level < 0.1 and pop_growth < AGENT_STABLE_GROWTH_RATE:
            self.sleep_timer = AGENT_SLEEP_THRESHOLD_STEPS

        self.prev_population = self.population

    def step(self, cell, world, debug=False):
        if not self.alive:
            return
        self.age += SIMULATION_STEP_YEARS

        self.gather_resources(cell)
        self.consume_resources(cell)
        self.update_population(cell)
        self.tech_growth(cell)

        if debug:
            print(
                f"[{self.stage.capitalize()} #{self.id}] "
                f"Pop={self.population}, Food={self.food:.1f}, Water={self.water:.2f}, "
                f"Hunger={self.hunger_level:.2f}, Thirst={self.thirst_level:.2f}, "
                f"Tech={self.tech:.3f}, Cell={cell.biome}, Hab={cell.habitability:.2f}"
            )

    def move_to(self, i, j):
        self.i, self.j = i, j

    @property
    def is_coastal(self, world):
        cell = world.get((self.i, self.j))
        return cell and cell.is_coastal

    def __repr__(self):
        return f"<{self.stage.capitalize()} #{self.id} pop={self.population} tech={self.tech:.3f} food={self.food:.1f}>"


# =======================================
# === 3. ГРУППА ========================
# =======================================

class HumanGroup(BaseEntity):
    def __init__(self, entity_id, i, j, population, start_tech=0.01, home_coord=None):
        super().__init__(entity_id, i, j, population, start_tech)
        self.stage = "group"
        self.food = 500.0
        self.path = [(i, j)]
        self.is_migrating = True
        self.steps_migrating = 0
        self.next_pos = None
        self.home_coord = home_coord if home_coord else (i, j)

    def absorb(self, other_entity):
        self.population += other_entity.population
        self.food += other_entity.food
        self.tech = max(self.tech, other_entity.tech)
        other_entity.alive = False

    def _distance_from_home(self, i, j):
        if not self.home_coord:
            return 0
        return max(abs(i - self.home_coord[0]), abs(j - self.home_coord[1]))

    def choose_next_direction(self, world):
        if not hasattr(self, "direction"):
            self.direction = (
                random.choice([-1, 0, 1]),
                random.choice([-1, 0, 1])
            )

        dirs = [
            (dx, dy)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            if not (dx == 0 and dy == 0)
        ]
        best_pos, best_score = None, -999
        current_dist = self._distance_from_home(self.i, self.j)

        recent_tail = set(self.path[-5:])

        for dx, dy in dirs:
            nx, ny = self.i + dx, self.j + dy
            if (nx, ny) in recent_tail:
                continue
            cell = world.get((nx, ny))
            if not cell or not cell.is_land:
                continue

            dist_score = (self._distance_from_home(nx, ny) - current_dist) * 0.6
            terrain_score = (
                cell.habitability * 0.7 +
                cell.arable * 0.5 +
                cell.food_availability * 0.3
            )

            if (dx, dy) == self.direction:
                dir_alignment = 1.0
            elif dx * dy == 0:
                dir_alignment = 0.5
            else:
                dir_alignment = 0.3

            random_bonus = random.uniform(-0.2, 0.2)

            score = dist_score + terrain_score + dir_alignment + random_bonus

            if score > best_score:
                best_score, best_pos = score, (nx, ny)

        if best_pos:
            self.direction = (best_pos[0] - self.i, best_pos[1] - self.j)

        return best_pos

    def gather_resources_migrant(self, cell):
        if cell.is_land:
            base_food = (cell.food_availability * 0.7 + cell.arable * 0.5) * self.population * 0.004
            tech_bonus = 1.0 + self.tech
            self.food += base_food * tech_bonus
            cell.deplete(self.population * 0.1)

        self.water = max(
            0.0,
            min(1.0, self.water * 0.8 + cell.water_availability + random.uniform(0.0, 0.1))
        )

    def update_population_migrant(self):
        if not self.alive:
            return

        base_birth = BIRTH_RATE_BASE * 0.5
        base_death = DEATH_RATE_BASE * 0.8

        resource_factor = self.food / (self.population * self.need_food_per_capita * 100 + 1e-9)
        if resource_factor > 1.0:
            base_birth *= min(2.0, resource_factor)
        elif resource_factor < 0.5:
            base_birth *= resource_factor

        starvation_term = self.hunger_level * DEATH_RATE_STARVATION
        dehydration_term = self.thirst_level * (DEATH_RATE_STARVATION * 0.5)

        yearly_birth = base_birth
        yearly_death = base_death + starvation_term + dehydration_term

        years = max(1, SIMULATION_STEP_YEARS)
        base_rate = 1.0 + yearly_birth - yearly_death
        clamped_base_rate = max(0.0, base_rate)
        growth_factor = clamped_base_rate ** years

        self.population = int(max(0, math.floor(self.population * growth_factor)))

        if self.population <= 0:
            self.alive = False

    def step(self, cell, world, debug=False):
        if not self.alive:
            return None
        if not cell or not cell.is_land:
            self.alive = False
            return None

        self.next_pos = None
        self.age += SIMULATION_STEP_YEARS

        self.gather_resources_migrant(cell)
        self.consume_resources(cell)
        self.update_population_migrant()

        if not self.alive:
            return None

        self.steps_migrating += 1
        if self.steps_migrating > MIGRATION_IMMUNITY_STEPS:
            self.is_migrating = False

        evolve_cf = cell.arable * cell.habitability
        if self.population > TRIBE_FOUNDING_THRESHOLD and evolve_cf > 0.4 and not self.is_migrating:
            if debug:
                print(f"  [Эволюция] Группа #{self.id} основала племя в ({self.i},{self.j})")
            tribe = Tribe(self.id, self.i, self.j, self.population, start_tech=self.tech)
            self.alive = False
            return tribe

        new_pos = self.choose_next_direction(world)
        if new_pos:
            self.next_pos = new_pos
        else:
            self.is_migrating = False

        if cell.is_coastal and self.tech >= SEAFARING_TECH_THRESHOLD and random.random() < 0.5:
            if debug:
                print(f"  [Эволюция] Группа #{self.id} ({self.i},{self.j}) стала мореплавателями (tech={self.tech:.3f})")
            self.alive = False
            return SeafaringGroup(self.id, self.i, self.j, self.population, start_tech=self.tech)

        return None


# =======================================
# === 4. ПЛЕМЯ ==========================
# =======================================

class Tribe(BaseEntity):
    def __init__(self, entity_id, i, j, population, start_tech=0.05):
        super().__init__(entity_id, i, j, population, start_tech)
        self.stage = "tribe"
        self.food = 300.0

    def absorb(self, other_entity):
        self.population += other_entity.population
        self.food += other_entity.food
        self.tech = max(self.tech, other_entity.tech)
        other_entity.alive = False

    def find_spawn_location(self, world):
        dirs = [(1, 0), (-1, 0), (0, 1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        random.shuffle(dirs)

        for dx, dy in dirs:
            nx, ny = self.i + dx, self.j + dy
            cell = world.get((nx, ny))
            if cell and cell.is_land:
                return (nx, ny)
        return None

    def get_stress_level(self, cell):
        migration_capacity = max(1.0, cell.habitability * CELL_CAPACITY_SCALE)
        population_ratio = self.population / migration_capacity
        overpop_stress = max(0.0, population_ratio - OVERPOPULATION_THRESHOLD)
        stress = self.hunger_level + overpop_stress
        return stress

    def step(self, cell, world, debug=False):
        if not self.alive:
            return None
        if not cell or not cell.is_land:
            self.alive = False
            return None

        super().step(cell, world, debug)

        if self.population > CITY_FOUNDING_THRESHOLD and self.tech > 0.08:
            # немного снижен tech-порог для появления городов к ~-15000
            if debug:
                print(f"  [Эволюция] Племя #{self.id} стало городом ({self.i},{self.j})")
            self.alive = False
            return City(self.id, self.i, self.j, self.population, start_tech=self.tech)

        stress = self.get_stress_level(cell)
        if stress > MIGRATION_STRESS_THRESHOLD and self.population > 100 and random.random() < 0.1:
            new_pop = int(self.population * MIGRATION_PERCENTAGE)
            if new_pop > 50:
                spawn_pos = self.find_spawn_location(world)
                if not spawn_pos:
                    return None

                migrant_tech = self.tech * 0.8
                new_group = HumanGroup(
                    random.randint(10000, 99999),
                    *spawn_pos,
                    new_pop,
                    start_tech=migrant_tech,
                    home_coord=(self.i, self.j)
                )
                self.population -= new_pop
                if debug:
                    print(f"  [Миграция] Племя #{self.id} (стресс={stress:.2f}) породило группу #{new_group.id} (tech={migrant_tech:.3f})")
                return new_group

        return None


# =======================================
# === 5. ГОРОД ==========================
# =======================================

class City(BaseEntity):
    def __init__(self, entity_id, i, j, population, start_tech=0.2):
        super().__init__(entity_id, i, j, population, start_tech)
        self.stage = "city"
        self.food = 1000.0
        self.influence_radius = CITY_INFLUENCE_RADIUS

    def get_stress_level(self, cell):
        migration_capacity = max(1.0, cell.habitability * CELL_CAPACITY_SCALE)
        population_ratio = self.population / migration_capacity
        overpop_stress = max(0.0, population_ratio - OVERPOPULATION_THRESHOLD)
        stress = self.hunger_level + overpop_stress
        return stress

    def find_spawn_location(self, world):
        dirs = [(1, 0), (-1, 0), (0, 1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        random.shuffle(dirs)

        for dx, dy in dirs:
            nx, ny = self.i + dx, self.j + dy
            cell = world.get((nx, ny))
            if cell and cell.is_land:
                return (nx, ny)
        return None

    def absorb(self, other_entity):
        self.population += other_entity.population
        self.food += other_entity.food
        self.tech = max(self.tech, other_entity.tech)
        other_entity.alive = False

    def step(self, cell, world, debug=False):
        if not self.alive:
            return None
        if not cell:
            self.alive = False
            return None

        super().step(cell, world, debug)

        stress = self.get_stress_level(cell)
        if stress > MIGRATION_STRESS_THRESHOLD and self.population > 1000 and random.random() < 0.1:
            new_pop = int(self.population * MIGRATION_PERCENTAGE * 0.5)
            if new_pop > 100:
                spawn_pos = self.find_spawn_location(world)
                if not spawn_pos:
                    return None

                migrant_tech = self.tech * 0.8
                new_group = HumanGroup(
                    random.randint(10000, 99999),
                    *spawn_pos,
                    new_pop,
                    start_tech=migrant_tech,
                    home_coord=(self.i, self.j)
                )

                self.population -= new_pop
                if debug:
                    print(f"  [Миграция] Город #{self.id} (стресс={stress:.2f}) породил группу #{new_group.id} в {spawn_pos}")
                return new_group

        return None


# =======================================
# === 6. МОРЕПЛАВАТЕЛИ ==================
# =======================================

class SeafaringGroup(BaseEntity):
    def __init__(self, entity_id, i, j, population, start_tech=0.01):
        super().__init__(entity_id, i, j, population, start_tech)
        self.stage = "seafaring"
        self.food = SEAFARING_FOOD_START * (population / 50)
        self.water = 1.0  # Start fully hydrated
        self.need_food_per_capita = 0.001  # Lower food need at sea (fishing logic implied)
        
        self.origin_land = (i, j)
        self.ocean_age = 0
        
        # Initial push away from land
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        if dx == 0 and dy == 0: dx = 1
        self.direction = (dx, dy) 
        
        # Исправление 3: Более высокая инерция направления
        self.inertia_counter = 0 

    def dist_from_origin(self, x=None, y=None):
        if x is None: x = self.i
        if y is None: y = self.j
        ox, oy = self.origin_land
        return max(abs(x - ox), abs(y - oy))

    def update_population_seafaring(self):
        if not self.alive:
            return
        
        # Смертность в море ниже, чтобы дать шанс доплыть
        base_death = 0.005 
        starvation_term = self.hunger_level * 0.05
        dehydration_term = self.thirst_level * 0.05

        yearly_death = base_death + starvation_term + dehydration_term
        
        # Population declines slowly at sea unless starving
        growth_factor = (1.0 - yearly_death) ** SIMULATION_STEP_YEARS
        self.population = int(max(0, math.floor(self.population * growth_factor)))

        if self.population <= 0:
            self.alive = False

    def gather_resources(self, cell):
        # Рыбалка дает немного еды и чуть-чуть воды (дождь/рыба)
        fish_food = cell.properties.get("food_animal", 0) * self.population * 0.002
        self.food += fish_food
        
        # Исправление 1: Ресурсы тратятся намного медленнее или восполняются
        # Шанс дождя или опреснения примитивными методами/рыбой
        self.water = max(0.0, self.water - 0.008) # Хватит на ~120 ходов (1200 лет) без пополнения
        if random.random() < 0.2: # Rain
            self.water = min(1.0, self.water + 0.1)

    def scan_for_land(self, world):
        """Сканирует окрестности на наличие земли."""
        # Радиус растет с технологиями
        radius = 2
        if self.tech >= 0.2: radius = 5
        if self.tech >= 0.4: radius = 10
        if self.tech >= 0.6: radius = 15

        best_target = None
        best_score = -999

        for r in range(1, radius + 1):
            # Optimisation: check perimeter of square radius
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    # Check only the ring (perimeter)
                    if abs(dx) != r and abs(dy) != r:
                        continue
                        
                    tx, ty = self.i + dx, self.j + dy
                    cell = world.get((tx, ty))

                    if cell and cell.is_land:
                        dist_origin = self.dist_from_origin(tx, ty)
                        
                        # Исправление 2: ЛОГИКА ВЫСАДКИ
                        # Запрещаем высаживаться только если мы ВЕРНУЛИСЬ к месту старта (ближе 15 клеток)
                        if dist_origin < 15:
                            continue

                        # Оценка: чем ближе, тем лучше. Бонус за новые земли (далеко от дома)
                        score = -r + (dist_origin * 0.1) 
                        
                        if score > best_score:
                            best_score = score
                            best_target = (tx, ty)
        return best_target

    def step(self, cell, world, debug=False):
        if not self.alive: return None

        self.age += SIMULATION_STEP_YEARS
        self.gather_resources(cell)
        self.consume_resources(cell) # Note: consume reduces water too
        self.update_population_seafaring()
        
        if not self.alive:
            if debug: print(f"☠ Группа #{self.id} погибла в море.")
            return None

        # Исправление 1 (Главное): Скорость
        # Корабли делают несколько "микро-шагов" за один ход симуляции (10 лет).
        # Это позволяет переплыть океан за разумное время.
        MOVES_PER_STEP = 4 
        
        for _ in range(MOVES_PER_STEP):
            
            # 1. Сначала ищем землю
            land_target = self.scan_for_land(world)
            
            if land_target:
                # Если видим землю - идем прямо к ней
                lx, ly = land_target
                dx = 1 if lx > self.i else -1 if lx < self.i else 0
                dy = 1 if ly > self.j else -1 if ly < self.j else 0
                self.direction = (dx, dy)
                self.inertia_counter = 10 # Фиксируем курс на землю
            else:
                # 2. Если земли нет - плывем по инерции или дрейфуем
                if self.inertia_counter > 0:
                    self.inertia_counter -= 1
                else:
                    # Редкий шанс сменить курс (Дрейф)
                    if random.random() < 0.02: # Было 0.1 (слишком часто)
                        dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
                        # Предпочтение сохранению общего направления (не разворачиваться на 180)
                        current_dx, current_dy = self.direction
                        valid_dirs = [d for d in dirs if d[0]*current_dx + d[1]*current_dy >= 0]
                        if valid_dirs:
                            self.direction = random.choice(valid_dirs)
                        self.inertia_counter = random.randint(5, 20)

            # Выполняем движение
            nx = self.i + self.direction[0]
            ny = self.j + self.direction[1]
            
            # Проверка границ мира (цикличный мир по X, стена по Y)
            # В load_world нет явного nx/ny в объекте cell, но обычно мир зациклен.
            # Предположим, что словарь world обрабатывает координаты или возвращает None.
            target_cell = world.get((nx, ny))
            
            if target_cell:
                if target_cell.is_land:
                    # LAND HO!
                    self.alive = False
                    if debug:
                        print(f"⚓ [КОЛОНИЗАЦИЯ] Группа #{self.id} (Pop: {self.population}) открыла НОВЫЙ СВЕТ в ({nx},{ny})! Dist: {self.dist_from_origin(nx, ny)}")
                    return HumanGroup(self.id, nx, ny, self.population, self.tech)
                else:
                    # Просто вода, плывем дальше
                    self.move_to(nx, ny)
            else:
                # Край карты (если не зациклена) - "отскакиваем"
                self.direction = (-self.direction[0], -self.direction[1])
                self.inertia_counter = 5

        return None


# =======================================
# === 7. ГОСУДАРСТВО ====================
# =======================================

class State:
    """Макро-агент (гос-во), с демографией, войнами и дипломатией."""
    def __init__(self, entity_id, i, j, population, tech):
        self.id = entity_id
        self.i, self.j = i, j  # столица
        self.population = int(population)
        self.tech = tech
        self.age = 0
        self.alive = True
        self.stage = "state"
        self.territory = set()
        self.cities_coords = []
        self.is_coastal = False
        self.need_food_per_capita = 0.004
        self.expansion_budget = 0.0

        # Дипломатия
        self.relations = {}
        self.war_exhaustion = 0.0
        self.at_war = set()
        self.allies = set()
        self.relations_initialized = False

        # Баланс сил
        self.military_power = 0.0
        self.is_great_power = False

        # Вассалитет
        self.vassals = set()
        self.overlord_id = None
        self.vassal_loyalty = 1.0

        # Демография (пирамида)
        self.pop_children = 0
        self.pop_adults = 0
        self.pop_elderly = 0
        self.demography_initialized = False
        self.society_type = "hunter_gatherer"

        # Эпидемии и голод
        self.epidemic_severity = 0.0  # 0..1
        self.years_of_famine = 0.0    # накопленный голод

    # ===== Вспомогательные методы демографии =====

    def _reset_demography_from_total(self):
        """Начальная возрастная структура по текущему населению."""
        if self.population <= 0:
            self.pop_children = self.pop_adults = self.pop_elderly = 0
            return
        c = int(self.population * 0.35)
        e = int(self.population * 0.08)
        a = self.population - c - e
        if a < 0:
            a = 0
        self.pop_children = c
        self.pop_adults = a
        self.pop_elderly = e

    def _ensure_demography(self):
        if not self.demography_initialized:
            self._reset_demography_from_total()
            self.demography_initialized = True

    def _update_society_type(self, n_cells, avg_habitability):
        """Очень упрощённый общественный строй по тех. и масштабу."""
        if self.tech < 0.15:
            self.society_type = "hunter_gatherer"
        elif self.tech < 0.3 and n_cells < 50:
            self.society_type = "early_agrarian"
        elif self.tech < 0.6:
            self.society_type = "agrarian_empire"
        else:
            self.society_type = "proto_industrial"

    def _update_epidemic_and_famine(self, food_surplus_ratio, n_cells):
        """Обновляем эпидемии и голод на уровне государства."""
        # Голод
        if food_surplus_ratio < FAMINE_FOOD_DEFICIT_THRESHOLD:
            self.years_of_famine += SIMULATION_STEP_YEARS
        else:
            self.years_of_famine = max(0.0, self.years_of_famine - SIMULATION_STEP_YEARS)

        # Эпидемии: шанс растёт с населением и плотностью
        density = self.population / max(1.0, n_cells * CELL_CAPACITY_SCALE)
        if self.epidemic_severity > 0.0:
            self.epidemic_severity = max(
                0.0,
                self.epidemic_severity - EPIDEMIC_DECAY_PER_STEP
            )
        else:
            if (
                self.population > EPIDEMIC_POP_THRESHOLD and
                density > 0.5 and
                random.random() < EPIDEMIC_BASE_CHANCE * (1.0 + density * EPIDEMIC_DENSITY_FACTOR)
            ):
                self.epidemic_severity = random.uniform(0.2, 0.7)
                # print(f"☠ Эпидемия в государстве {self.id}, тяжесть={self.epidemic_severity:.2f}")

    def _demographic_step(self, years, K, food_surplus_ratio):
        """
        Линейно-логистическая демография:
        - рождаемость зависит от общества, еды и перенаселения
        - смертность зависит от возраста, голода, эпидемий, войн
        """
        self._ensure_demography()
        total_pop = self.pop_children + self.pop_adults + self.pop_elderly
        if total_pop <= 0:
            self.population = 0
            self.alive = False
            return

        # Логистический фактор (1 - P/K), ограничиваем [-1, 1]
        logistic_factor = 1.0 - total_pop / (K + 1e-9)
        logistic_factor = max(-1.0, min(1.0, logistic_factor))

        # Общественный строй
        fert_mult = SOCIETY_FERTILITY_MULT.get(self.society_type, 1.0)
        mort_mult = SOCIETY_MORTALITY_MULT.get(self.society_type, 1.0)

        # Голод
        famine_severity = min(1.5, self.years_of_famine / FAMINE_YEARS_SCALE)
        famine_birth_mult = max(0.0, 1.0 - FAMINE_BIRTH_REDUCTION * famine_severity)
        famine_mort_add = FAMINE_EXTRA_MORTALITY * famine_severity

        # Эпидемия
        epi = self.epidemic_severity
        epidemic_mort_add = EPIDEMIC_MORTALITY_MULT * epi

        # Война
        war_factor = min(
            WAR_MAX_EXTRA_MORTALITY,
            WAR_EXTRA_MORTALITY_PER_ENEMY * len(self.at_war)
        )

        # Логистика: при перенаселении снижаем рождаемость и повышаем смертность
        if logistic_factor < 0:
            fertility_logistic_mult = max(0.0, 1.0 + logistic_factor)  # до 0 при сильном перенаселении
            overpop_mort_add = -logistic_factor * 0.01
        else:
            fertility_logistic_mult = 1.0 + 0.2 * logistic_factor
            overpop_mort_add = 0.0

        # Итоговые коэффициенты
        annual_birth_rate = (
            STATE_BASE_FERTILITY *
            fert_mult *
            famine_birth_mult *
            fertility_logistic_mult
        )
        annual_birth_rate = max(0.0, min(0.12, annual_birth_rate))  # защита от странных значений

        child_mort = (
            STATE_CHILD_MORTALITY * mort_mult +
            famine_mort_add + epidemic_mort_add + overpop_mort_add + war_factor
        )
        adult_mort = (
            STATE_ADULT_MORTALITY * mort_mult +
            famine_mort_add + epidemic_mort_add + overpop_mort_add + war_factor
        )
        elder_mort = (
            STATE_ELDER_MORTALITY * mort_mult +
            famine_mort_add + epidemic_mort_add + overpop_mort_add + war_factor
        )

        # Ограничения (пер-год)
        child_mort = max(0.0, min(0.5, child_mort))
        adult_mort = max(0.0, min(0.3, adult_mort))
        elder_mort = max(0.0, min(0.8, elder_mort))

        # Рождения (от взрослых)
        births = int(self.pop_adults * annual_birth_rate * years)

        # Старение
        children_to_adults = int(self.pop_children * (years / CHILD_YEARS))
        adults_to_elderly = int(self.pop_adults * (years / ADULT_YEARS))

        children_to_adults = min(self.pop_children, children_to_adults)
        adults_to_elderly = min(self.pop_adults, adults_to_elderly)

        # Смертность
        deaths_children = int(self.pop_children * child_mort * years)
        deaths_adults = int(self.pop_adults * adult_mort * years)
        deaths_elderly = int(self.pop_elderly * elder_mort * years)

        deaths_children = min(self.pop_children, deaths_children)
        deaths_adults = min(self.pop_adults, deaths_adults)
        deaths_elderly = min(self.pop_elderly, deaths_elderly)

        # Обновляем группы
        new_children = self.pop_children + births - children_to_adults - deaths_children
        new_adults = self.pop_adults + children_to_adults - adults_to_elderly - deaths_adults
        new_elderly = self.pop_elderly + adults_to_elderly - deaths_elderly

        self.pop_children = max(0, new_children)
        self.pop_adults = max(0, new_adults)
        self.pop_elderly = max(0, new_elderly)

        self.population = self.pop_children + self.pop_adults + self.pop_elderly

        if self.population <= 0:
            self.alive = False

    # ===== Дипломатия и войны =====

    def init_relations(self, other_states):
        for s in other_states:
            if s.id == self.id:
                continue
            if s.id not in self.relations:
                val = random.uniform(-10, 10)
                self.relations[s.id] = val
                s.relations[self.id] = val

    def decay_relations(self):
        for sid in list(self.relations.keys()):
            self.relations[sid] *= (1 - RELATION_DECAY * 0.01)

    def check_war_state(self, other):
        rel = self.relations.get(other.id, 0.0)
        if other.id in self.allies:
            return
        if other.id in self.at_war:
            return
        if rel < WAR_THRESHOLD:
            self.start_war(other)

    def try_make_peace(self, other):
        rel = self.relations.get(other.id, 0.0)
        if rel > PEACE_THRESHOLD or self.war_exhaustion > 5:
            if other.id in self.at_war:
                self.at_war.discard(other.id)
                other.at_war.discard(self.id)
                self.war_exhaustion = 0.0
                other.war_exhaustion = 0.0

    def start_war(self, other):
        if other.id not in self.at_war:
            self.at_war.add(other.id)
            other.at_war.add(self.id)

    def add_war_exhaustion(self):
        if self.at_war:
            self.war_exhaustion += WAR_EXHAUSTION_RATE
            for enemy_id in list(self.at_war):
                self.relations[enemy_id] = self.relations.get(enemy_id, 0.0) + 0.5

    def get_border_cells(self):
        border = set()
        for (i, j) in self.territory:
            border.add((i + 1, j))
            border.add((i - 1, j))
            border.add((i, j + 1))
            border.add((i, j - 1))
        return border

    def maybe_vassalize(self, enemy):
        if enemy.id in self.vassals or enemy.overlord_id == self.id:
            return
        if not enemy.territory or enemy.population <= 0:
            return
        if self.military_power <= 0:
            return

        enemy_power = enemy.military_power if enemy.military_power > 0 else enemy.population * 0.0001
        power_ratio = self.military_power / max(1.0, enemy_power)
        rel = self.relations.get(enemy.id, 0.0)

        if power_ratio >= 2.0 and rel > VASSALIZATION_RELATION_FLOOR:
            self.vassals.add(enemy.id)
            enemy.overlord_id = self.id
            enemy.vassal_loyalty = 1.0

            self.at_war.discard(enemy.id)
            enemy.at_war.discard(self.id)

            self.relations[enemy.id] = max(self.relations.get(enemy.id, 0.0), 10.0)
            enemy.relations[self.id] = self.relations[enemy.id]

    def attack_enemy_cells(self, world, states_by_id):
        if not self.at_war:
            return

        border = self.get_border_cells()
        if not border:
            return

        border_list = list(border)

        for enemy_id in list(self.at_war):
            enemy_state = states_by_id.get(enemy_id)
            if not enemy_state or not enemy_state.territory:
                continue

            random.shuffle(border_list)
            for pos in border_list:
                if pos in enemy_state.territory:
                    if random.random() < TERRITORY_STEAL_CHANCE:
                        enemy_state.territory.remove(pos)
                        self.territory.add(pos)

                        # боевые потери (повлияют на демографию на следующем шаге)
                        self.population = int(self.population * (1 - BATTLE_DAMAGE_RATE))
                        enemy_state.population = int(enemy_state.population * (1 - BATTLE_DAMAGE_RATE))

                        self.maybe_vassalize(enemy_state)
                    break

    # ===== Поглощение и макро-рост =====

    def absorb_entity(self, entity, world):
        self.population += entity.population
        if entity.tech > self.tech:
            self.tech = min(1.0, self.tech + (entity.tech - self.tech) * 0.1)

        if not isinstance(entity, HumanGroup):
            self.territory.add((entity.i, entity.j))
            if isinstance(entity, City):
                self.cities_coords.append((entity.i, entity.j))

        if not self.is_coastal:
            cell = world.get((entity.i, entity.j))
            if cell and cell.is_coastal:
                self.is_coastal = True

        entity.alive = False

        # возрастную структуру пересобираем на следующем шаге
        self.demography_initialized = False

    def get_expansion_candidates(self, world, all_claimed_cells, nx, ny):
        if nx is None or ny is None:
            return []

        # Кандидаты: словарь {координата: {ortho_neighbors: int, diag_neighbors: int}}
        # ortho - соседи по сторонам (верх, низ, лево, право) - ЭТО ВАЖНО
        # diag - соседи по углам - ЭТО МЕНЕЕ ВАЖНО
        candidates_info = {}

        # 1. Сканируем ТОЛЬКО границы нашей территории
        for (i, j) in self.territory:
            # Проверяем 8 соседей
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0: continue
                    
                    ni, nj = (i + dx) % nx, (j + dy) % ny # Цикличный мир
                    
                    if (ni, nj) in self.territory or (ni, nj) in all_claimed_cells:
                        continue

                    # Если этой клетки еще нет в кандидатах - создаем
                    if (ni, nj) not in candidates_info:
                        candidates_info[(ni, nj)] = {"ortho": 0, "diag": 0}

                    # Считаем тип соседства
                    # Если dx*dy == 0 (один из них 0), значит это ортогональный сосед (крестом)
                    if dx * dy == 0:
                        candidates_info[(ni, nj)]["ortho"] += 1
                    else:
                        candidates_info[(ni, nj)]["diag"] += 1

        final_candidates = []

        # 2. Оцениваем кандидатов
        for pos, info in candidates_info.items():
            cell = world.get(pos)
            if not cell or not cell.is_land:
                continue

            # ❗ 1. Главное исправление — убираем диагональные захваты
            # Клетка должна иметь хотя бы ОДНО прямое (ортогональное) соприкосновение
            if info["ortho"] == 0:
                continue

            ortho = info["ortho"]
            diag = info["diag"]

            if info["diag"] >= 4:
                continue

            # 2. Базовая ценность клеток
            base_score = (
                cell.habitability * 5.0 +
                cell.arable * 2.0 +
                cell.food_availability * 1.0
            )

            # 3. Геометрический бонус — чтобы заливать дырки и делать границы ровными
            if ortho >= 4:
                shape_modifier = 100
            elif ortho == 3:
                shape_modifier = 25
            elif ortho == 2:
                shape_modifier = 6
            elif ortho == 1:
                shape_modifier = 1
            else:
                # бесполезно, даже не рассматривать — но мы уже отфильтровали ortho==0
                shape_modifier = 0.1


            # 4. Штраф за удалённость от столицы
            dx = abs(pos[0] - self.i)
            dy = abs(pos[1] - self.j)
            if dx > nx // 2: dx = nx - dx

            dist_sq = dx*dx + dy*dy
            distance_penalty = 100.0 / (100.0 + dist_sq)

            final_score = base_score * shape_modifier * distance_penalty

            final_candidates.append((final_score, pos))

        final_candidates.sort(key=lambda x: x[0], reverse=True)
        return final_candidates


    def step(self, world, debug=False):
        if not self.alive:
            return []
        self.age += SIMULATION_STEP_YEARS
        new_entities = []

        # 1. Агрегированные характеристики территории
        total_habitability = 0.0
        total_food_prod = 0.0
        total_arable = 0.0
        total_wood = 0.0
        total_stone = 0.0
        total_ore = 0.0

        for (i, j) in self.territory:
            cell = world.get((i, j))
            if not cell:
                continue
            total_habitability += cell.habitability
            total_food_prod += (cell.arable + cell.food_availability)
            total_arable += cell.arable
            props = cell.properties
            total_wood += props.get("wood_yield", 0.0)
            total_stone += props.get("stone_yield", 0.0)
            total_ore += props.get("ore_yield", 0.0)

        n_cells = max(1, len(self.territory))
        avg_habitability = total_habitability / n_cells
        avg_arable = total_arable / n_cells
        avg_wood = total_wood / n_cells
        avg_stone = total_stone / n_cells
        avg_ore = total_ore / n_cells

        # ёмкость среды (K)
        base_capacity = n_cells * avg_habitability * CARRYING_CAPACITY_FACTOR
        tech_capacity_multiplier = 1.0 + (self.tech * 4.0)
        effective_capacity = base_capacity * tech_capacity_multiplier

        # производство еды
        resource_bonus = 1.0 + (
            avg_arable * 0.4 +
            avg_wood * 0.2 +
            avg_stone * 0.2 +
            avg_ore * 0.3
        )

        food_production = (
            total_food_prod *
            (1 + self.tech) *
            MACRO_FOOD_PRODUCTION_FACTOR *
            n_cells *
            resource_bonus
        )
        food_needed = self.population * self.need_food_per_capita
        food_surplus_ratio = (food_production - food_needed) / (food_needed + 1e-9)

        # Общественный строй, эпидемии и голод
        self._update_society_type(n_cells, avg_habitability)
        self._update_epidemic_and_famine(food_surplus_ratio, n_cells)

        # 2. Демографический шаг (убираем экспоненциальный рост)
        self._demographic_step(SIMULATION_STEP_YEARS, effective_capacity, food_surplus_ratio)
        if not self.alive:
            return []

        # 3. Рост технологий (макро)
        tech_gain = (
            (len(self.cities_coords) / 5.0) *
            (self.population / (effective_capacity + 1e-9)) *
            MACRO_TECH_FACTOR
        )
        self.tech = min(1.0, self.tech + tech_gain)

        # 4. Морские колонии
        if (
            self.is_coastal and
            self.tech > SEAFARING_TECH_THRESHOLD and
            random.random() < SEAFARING_SPAWN_CHANCE
        ):
            start_coord = random.choice(self.cities_coords) if self.cities_coords else (self.i, self.j)
            new_pop = random.randint(100, 300)
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                check_pos = (start_coord[0] + di, start_coord[1] + dj)
                cell = world.get(check_pos)
                if cell and not cell.is_land:
                    self.population = max(1, self.population - new_pop)
                    new_colonists = SeafaringGroup(
                        random.randint(10000, 99999),
                        *check_pos,
                        new_pop,
                        start_tech=self.tech * 0.7
                    )
                    new_entities.append(new_colonists)
                    if debug:
                        print(f"  [Колонизация] Гос-во #{self.id} отправило флот из {start_coord}")
                    break

        if debug:
            print(
                f"[STATE #{self.id}] Pop={self.population}, K={int(effective_capacity)}, "
                f"Tech={self.tech:.3f}, Terr={len(self.territory)}, FoodΔ={food_surplus_ratio:+.2f}, "
                f"Soc={self.society_type}, FamineYears={self.years_of_famine:.0f}, Epi={self.epidemic_severity:.2f}"
            )

        return new_entities


# =======================================
# === 7.5 ДИПЛОМАТИЧЕСКИЙ МЕНЕДЖЕР =====
# =======================================

class DiplomacyManager:
    def __init__(self):
        self.total_power = 0.0

    def initialize_relations(self, states):
        for s in states:
            if not s.relations_initialized:
                s.init_relations(states)
                s.relations_initialized = True

    def update_balance_of_power(self, states):
        total_power = 0.0
        for s in states:
            territory_factor = max(1.0, len(s.territory))
            s.military_power = (
                s.population * 0.0001 * (1.0 + s.tech * 2.0) +
                territory_factor * 0.5
            )
            total_power += s.military_power

        self.total_power = total_power
        if total_power <= 0:
            for s in states:
                s.is_great_power = False
            return

        sorted_states = sorted(states, key=lambda st: st.military_power, reverse=True)
        top_k = max(1, len(sorted_states) // 4)
        top_ids = {st.id for st in sorted_states[:top_k]}
        for s in states:
            s.is_great_power = s.id in top_ids

    def handle_coalitions_and_vassals(self, states):
        if not states or self.total_power <= 0:
            return

        id_to_state = {s.id: s for s in states}

        hegemon = max(states, key=lambda s: s.military_power)
        if hegemon.military_power / self.total_power < 0.35:
            hegemon = None

        if hegemon:
            anti = [
                s for s in states
                if s.id != hegemon.id and s.relations.get(hegemon.id, 0.0) < 0.0
            ]
            for i in range(len(anti)):
                for j in range(i + 1, len(anti)):
                    a, b = anti[i], anti[j]
                    a.allies.add(b.id)
                    b.allies.add(a.id)

        # Лояльность вассалов
        for s in states:
            if s.overlord_id is not None:
                s.vassal_loyalty += 0.01 * (SIMULATION_STEP_YEARS / 10.0)
                s.vassal_loyalty -= len(s.at_war) * VASSAL_WAR_LOYALTY_PENALTY * (SIMULATION_STEP_YEARS / 10.0)
                s.vassal_loyalty = max(0.0, min(1.5, s.vassal_loyalty))

                if s.vassal_loyalty < VASSAL_REVOLT_THRESHOLD:
                    overlord = id_to_state.get(s.overlord_id)
                    if overlord:
                        overlord.vassals.discard(s.id)
                        overlord.relations[s.id] = min(
                            overlord.relations.get(s.id, 0.0),
                            -15.0
                        )
                        s.relations[overlord.id] = -15.0
                    s.overlord_id = None

        # "Дань": небольшой перенос тех. от вассалов
        for s in states:
            if s.vassals:
                for vid in list(s.vassals):
                    v = id_to_state.get(vid)
                    if not v:
                        continue
                    tribute = v.tech * VASSAL_TRIBUTE_RATE
                    v.tech = max(0.0, v.tech - tribute * 0.3)
                    s.tech = min(1.0, s.tech + tribute)

    def update_diplomacy(self, states):
        if len(states) < 2:
            return

        self.initialize_relations(states)
        self.update_balance_of_power(states)

        for s in states:
            s.decay_relations()
            s.add_war_exhaustion()
            
            # Добавляем "Естественную напряженность" и случайные обиды
            for other_id in list(s.relations.keys()):
                # 1. Случайное падение отношений (интриги, оскорбления)
                if random.random() < 0.1:
                    s.relations[other_id] -= random.uniform(1.0, 5.0)
                
                # 2. Если государство-сосед слишком сильное — мы его боимся/ненавидим
                # (упрощенная проверка, т.к. полная карта соседей сложна для расчета каждый ход)
                other_state = next((x for x in states if x.id == other_id), None)
                if other_state:
                    dist = distance(s.i, s.j, other_state.i, other_state.j)
                    # Если столицы близко (менее 30 клеток), считаем их соседями -> трение
                    if dist < 30: 
                        s.relations[other_id] -= 1.5  # Ежегодное ухудшение отношений с соседями

        n = len(states)
        for i in range(n):
            a = states[i]
            for j in range(i + 1, n):
                b = states[j]
                a.check_war_state(b)
                b.check_war_state(a)
                a.try_make_peace(b)
                b.try_make_peace(a)

        self.handle_coalitions_and_vassals(states)

    def apply_war_actions(self, states, world):
        states_by_id = {s.id: s for s in states}
        for s in states:
            if s.at_war:
                s.attack_enemy_cells(world, states_by_id)


# =======================================
# === 8. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ========
# =======================================

def distance(i1, j1, i2, j2):
    return max(abs(i1 - i2), abs(j1 - j2))


# =======================================
# === 9. СИМУЛЯЦИЯ ======================
# =======================================

class Simulation:
    def __init__(self, world_file="world_cells.json", nx=None, ny=None):
        self.world = load_world(world_file, nx, ny)
        self.entities = []
        self.year = START_YEAR
        self.running = True
        self.occupied_cells = set()
        self.nx = nx
        self.ny = ny
        self.diplomacy = DiplomacyManager()

    def initialize(self):
        start = HumanGroup(0, *STARTING_CELL_COORDS, STARTING_POPULATION)
        self.entities = [start]

    def regenerate_world(self):
        all_cells_coords = list(self.world.keys())
        random.shuffle(all_cells_coords)

        sample_size = int(len(all_cells_coords) * CELL_REGEN_TICK_RATE)
        for idx in range(sample_size):
            coord = all_cells_coords[idx]
            if coord not in self.occupied_cells:
                self.world[coord].regenerate()

    def resolve_interactions(self, debug=False):
        occupied_cells = {}

        for e in self.entities:
            if not e.alive:
                continue
            occupied_cells.setdefault((e.i, e.j), []).append(e)

        for coord, occupants in occupied_cells.items():
            if len(occupants) <= 1:
                continue

            def get_entity_priority(e):
                if isinstance(e, State):
                    return 4
                if isinstance(e, City):
                    return 3
                if isinstance(e, Tribe):
                    return 2
                if isinstance(e, HumanGroup):
                    return 1
                return 0

            occupants.sort(
                key=lambda e: (get_entity_priority(e), e.population),
                reverse=True
            )

            owner = occupants[0]
            losers = occupants[1:]

            if debug and losers:
                print(
                    f"  [Интеракция] В клетке {coord}: {owner.stage} #{owner.id} "
                    f"(Pop: {owner.population}) 'победил' {len(losers)} других агентов."
                )

            for loser in losers:
                if isinstance(owner, State):
                    owner.absorb_entity(loser, self.world)
                elif isinstance(owner, City):
                    owner.absorb(loser)
                elif hasattr(owner, "absorb"):
                    owner.absorb(loser)
                else:
                    loser.alive = False

                if isinstance(owner, HumanGroup):
                    owner.is_migrating = False

    def step_aggregation(self, debug=False):
        cities = [e for e in self.entities if isinstance(e, City) and e.alive]
        tribes = [e for e in self.entities if isinstance(e, Tribe) and e.alive]
        states = [e for e in self.entities if isinstance(e, State) and e.alive]

        new_states = []
        entities_to_remove = set()

        # Города поглощают племена
        for city in cities:
            if city in entities_to_remove:
                continue
            for tribe in tribes:
                if tribe in entities_to_remove:
                    continue
                if distance(city.i, city.j, tribe.i, tribe.j) <= city.influence_radius:
                    city.absorb(tribe)
                    entities_to_remove.add(tribe)

        # Города и племена поглощаются существующими гос-вами
        for state in states:
            for entity in (cities + tribes):
                if entity in entities_to_remove:
                    continue
                if (entity.i, entity.j) in state.territory:
                    state.absorb_entity(entity, self.world)
                    entities_to_remove.add(entity)
                    continue
                if distance(state.i, state.j, entity.i, entity.j) <= STATE_INFLUENCE_RADIUS:
                    state.absorb_entity(entity, self.world)
                    entities_to_remove.add(entity)

        # Города формируют новые гос-ва
        eligible_cities = [
            c for c in cities
            if (
                c.population > STATE_FOUNDING_POP and
                c.tech > STATE_FOUNDING_TECH and
                c not in entities_to_remove
            )
        ]

        for city in eligible_cities:
            if city in entities_to_remove:
                continue

            if debug:
                print(f"  [Эволюция] Город #{city.id} ({city.i},{city.j}) основывает ГОСУДАРСТВО!")

            new_state = State(city.id, city.i, city.j, 0, city.tech)
            entities_to_remove.add(city)

            entities_to_absorb = [
                e for e in self.entities
                if isinstance(e, (Tribe, City)) and e.alive and e not in entities_to_remove
            ]

            for entity in entities_to_absorb:
                if distance(city.i, city.j, entity.i, entity.j) <= STATE_INFLUENCE_RADIUS:
                    new_state.absorb_entity(entity, self.world)
                    entities_to_remove.add(entity)

            if (city.i, city.j) not in new_state.territory:
                new_state.absorb_entity(city, self.world)

            new_states.append(new_state)

        if new_states:
            self.entities.extend(new_states)

        if entities_to_remove:
            self.entities = [e for e in self.entities if e not in entities_to_remove]

    def smooth_territory(self, territory, nx, ny):
        territory_set = set(territory)
        new_territory = set()

        for i, j in territory_set:
            ortho = 0
            diag = 0

            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                if ((i+dx) % nx, (j+dy) % ny) in territory_set:
                    ortho += 1

            for dx, dy in ((1,1),(1,-1),(-1,1),(-1,-1)):
                if ((i+dx) % nx, (j+dy) % ny) in territory_set:
                    diag += 1

            # Условие сглаживания
            if ortho >= 2 or (ortho == 1 and diag >= 2):
                new_territory.add((i, j))

        return new_territory

    def step(self, debug=False):
        if not self.running or not self.entities:
            self.running = False
            return self.entities, self.year

        self.year += SIMULATION_STEP_YEARS
        new_entities = []
        entities_to_remove = set()
        move_requests = []

        self.occupied_cells = {
            (e.i, e.j)
            for e in self.entities
            if e.alive and isinstance(e, BaseEntity)
        }

        # 1. Регенерация мира
        self.regenerate_world()

        # 2. Логика агентов
        for e in list(self.entities):
            if not e.alive:
                entities_to_remove.add(e)
                continue

            if isinstance(e, BaseEntity) and e.sleep_timer > 0:
                e.sleep_timer -= 1
                continue

            if isinstance(e, State):
                results = e.step(self.world, debug=debug)
                if results:
                    new_entities.extend(results)
            elif isinstance(e, BaseEntity):
                cell = self.world.get((e.i, e.j))
                if not cell:
                    e.alive = False
                    entities_to_remove.add(e)
                    continue

                result = e.step(cell, self.world, debug=debug)
                if result:
                    new_entities.append(result)

                if isinstance(e, HumanGroup) and e.next_pos:
                    move_requests.append(e)

        if new_entities:
            self.entities.extend(new_entities)

        if entities_to_remove:
            self.entities = [e for e in self.entities if e not in entities_to_remove]

        # 3. Движение групп
        for group in move_requests:
            if group.alive:
                group.move_to(*group.next_pos)
                group.path.append(group.next_pos)
                if len(group.path) > 100:
                    group.path.pop(0)
                group.next_pos = None

        # 4. Столкновения
        if move_requests:
            self.resolve_interactions(debug=debug)

        # 5. Агрегация (города, гос-ва)
        self.step_aggregation(debug=debug)

        # 6. Дипломатия и войны
        states = [e for e in self.entities if isinstance(e, State)]
        if states:
            self.diplomacy.update_diplomacy(states)
            self.diplomacy.apply_war_actions(states, self.world)

        # 7. Расширение территорий
        all_claimed_cells = set()
        for s in states:
            all_claimed_cells.update(s.territory)

        for s in states:
            total_habitability = 0.0
            total_food = 0.0
            total_arable = 0.0
            total_cells = max(1, len(s.territory))

            for (i, j) in s.territory:
                cell = self.world.get((i, j))
                if not cell:
                    continue
                total_habitability += cell.habitability
                total_food += (cell.food_availability + cell.arable)
                total_arable += cell.arable

            avg_habit = total_habitability / total_cells
            avg_food = total_food / total_cells
            avg_arable = total_arable / total_cells

            avg_resource = (avg_food + avg_arable + avg_habit)
            resource_factor = max(0.1, avg_resource)

            # 1. Базовое значение (государство всегда немного расширяется)
            base_expansion = 0.7 
            # 2. Делитель уменьшен с 120к до 10к, чтобы население влияло сильнее
            pop_factor = s.population / 80_000.0 
            
            s.expansion_budget += (base_expansion + pop_factor + (s.tech * 2.0)) * resource_factor

            candidates = s.get_expansion_candidates(self.world, all_claimed_cells, self.nx, self.ny)

            while s.expansion_budget >= 1.0:
                candidates = s.get_expansion_candidates(self.world, all_claimed_cells, self.nx, self.ny)

                if not candidates:
                    break

                _, pos = candidates[0]

                s.territory.add(pos)
                all_claimed_cells.add(pos)
                s.expansion_budget -= 1.0

                if not s.is_coastal:
                    cell = self.world.get(pos)
                    if cell and cell.is_coastal:
                        s.is_coastal = True

            # Сглаживание территории
            s.territory = self.smooth_territory(s.territory, self.nx, self.ny)
            all_claimed_cells.update(s.territory)



        # 8. Очистка мертвых
        self.entities = [e for e in self.entities if e.alive]

        if not self.entities:
            self.running = False

        return self.entities, self.year


# =======================================
# === 10. ТЕСТ ЗАПУСКА ==================
# =======================================

if __name__ == "__main__":
    sim = Simulation()
    sim.initialize()
    for i in range(1000):
        entities, year = sim.step(debug=False)

        if i % 10 == 0:
            counts = {"Group": 0, "Tribe": 0, "City": 0, "Seafaring": 0, "State": 0}
            total_pop = 0
            techs = []
            for e in entities:
                counts[e.stage.capitalize()] += 1
                if hasattr(e, "population"):
                    total_pop += e.population
                if hasattr(e, "tech"):
                    techs.append(e.tech)

            print(f"--- Год: {year} | Агентов: {len(entities)} | Всего населения: {total_pop} ---")
            if counts["City"]:
                print(f"  Городов: {counts['City']}")
            if counts["State"]:
                print(f"  Государств: {counts['State']}")
            if techs:
                print(f"  Тех: макс={max(techs):.3f}, ср={sum(techs)/len(techs):.3f}")
            print(f"    {counts}")

            if not entities:
                print("Симуляция завершена: все вымерли.")
                break

    print("Финальный отчет:")
    print(f"{year}: {entities}")
