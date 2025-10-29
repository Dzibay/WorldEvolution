import json, random, math
from dataclasses import dataclass
from biomes_properties import BIOME_DATA
from config import *


# =======================================
# === 1. КЛАСС КЛЕТКИ (Изменен) ========
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
        # Инициализируем динамические ресурсы базовыми
        self.current_food_base = (self.properties.get("food_vegetal", 0) + self.properties.get("food_animal", 0))
        self.current_water_base = self.properties.get("fresh_water", 0)

    @property
    def is_land(self): return not self.properties.get("is_ocean", False)
    @property
    def is_coastal(self): return self.properties.get("is_coastal", False) # Предполагаем, что это есть в biomes
    
    # Свойства теперь ссылаются на динамические ресурсы
    @property
    def food_availability(self): return self.current_food_base / 2
    @property
    def water_availability(self): return self.current_water_base
    
    @property
    def habitability(self): return self.properties.get("habitability", 0)
    @property
    def movement_cost(self): return self.properties.get("movement_cost", 1.0)
    @property
    def arable(self): return self.properties.get("arable_land", 1.0)

    def deplete(self, population):
        """Истощает ресурсы от деятельности человека"""
        if self.is_land:
            depletion = population * RESOURCE_DEPLETION_RATE
            self.current_food_base = max(0.0, self.current_food_base - depletion)

    def regenerate(self):
        """Медленно восстанавливает ресурсы до базовых"""
        base_food = (self.properties.get("food_vegetal", 0) + self.properties.get("food_animal", 0))
        if self.current_food_base < base_food:
            self.current_food_base = min(base_food, self.current_food_base * (1 + RESOURCE_REGENERATION_RATE))
        
        base_water = self.properties.get("fresh_water", 0)
        if self.current_water_base < base_water:
            self.current_water_base = min(base_water, self.current_water_base * (1 + RESOURCE_REGENERATION_RATE))


def load_world(filename, nx=None, ny=None):
    with open(filename) as f:
        raw = json.load(f)
    world = {}
    
    # --- 1. Первый проход: Создаем все ячейки ---
    for c in raw:
        props = BIOME_DATA.get(c["biome"], BIOME_DATA["Plains"]).copy()
        world[(c["i"], c["j"])] = WorldCell(c["i"], c["j"], c["biome"], c["elevation_m"], props)

    # --- 2. Второй проход: Динамически вычисляем 'is_coastal' ---
    if nx is None or ny is None:
        print("ПРЕДУПРЕЖДЕНИЕ: Размеры карты (nx, ny) не переданы, 'is_coastal' не будет вычислен.")
        return world # Возвращаем как есть

    print("Вычисление прибрежных зон...")
    for (i, j), cell in world.items():
        # Нас интересует только 'суша'
        if cell.is_land:
            # Проверяем 4 соседей
            for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
                check_pos = ((i + di) % nx, (j + dj) % ny)
                neighbor = world.get(check_pos)
                
                # Если сосед - океан, эта клетка - прибрежная
                if neighbor and not neighbor.is_land:
                    cell.properties["is_coastal"] = True
                    break # Достаточно одного соседа-океана
    
    return world


# =======================================
# === 2. БАЗОВЫЙ КЛАСС (Изменен) ========
# =======================================

class BaseEntity:
    def __init__(self, entity_id, i, j, population, start_tech=0.01):
        self.id = entity_id
        self.i, self.j = i, j
        self.population = int(population)
        self.prev_population = population # Для проверки "сна"
        self.food = max(50.0, population * 0.5)
        self.water = 0.7
        self.tech = start_tech
        self.age = 0
        self.stage = "base"
        self.alive = True
        self.need_food_per_capita = 0.004
        self.hunger_level = 0.0
        self.thirst_level = 0.0
        self.sleep_timer = 0 # Таймер "сна" для оптимизации
        self.sleep_timer = 0

    # --- Сбор ресурсов (с истощением) ---
    def gather_resources(self, cell):
        if cell.is_land:
            # Используем истощаемые ресурсы
            base_food = (cell.food_availability + cell.arable * 0.6) * self.population * 0.0025
            tech_bonus = 1.0 + self.tech * 2.0
            self.food += base_food * tech_bonus
            # Истощаем клетку
            cell.deplete(self.population)
        else:
            self.food += cell.properties.get("food_animal", 0) * self.population * 0.0008

        self.water = max(0.0, min(1.0, self.water * 0.6 + cell.water_availability + random.uniform(0.0, 0.1)))

    # --- Потребление (без изменений) ---
    def consume_resources(self, cell):
        # (Код без изменений)
        need_food = self.population * self.need_food_per_capita
        if self.food >= need_food:
            self.food -= need_food * (1.0 - FOOD_WASTAGE_RATE)
            self.hunger_level = 0.0
        else:
            deficit = need_food - self.food
            self.food = 0.0
            self.hunger_level = max(0.0, min(1.0, deficit / (need_food + 1e-9)))
        self.water = max(0.0, self.water - 0.15)
        self.thirst_level = 0.0 if self.water >= 0.6 else max(0.0, min(1.0, (0.6 - self.water) / 0.6))


    # --- Рост технологий (без изменений) ---
    def tech_growth(self, cell):
        # (Код без изменений)
        density_factor = min(1.0, self.population / (CARRYING_CAPACITY_FACTOR * 0.1))
        discovery_chance = TECH_DISCOVERY_CHANCE_BASE * (1 + density_factor * TECH_DENSITY_FACTOR)
        if random.random() < discovery_chance:
            gain = 0.001 * (cell.habitability + cell.arable * 0.5)
            self.tech = min(1.0, self.tech + gain)

    # --- Обновление популяции (единственное место с рождаемостью/смертностью) ---
    def update_population(self, cell):
        if not self.alive:
            return

        # === ИЗМЕНЕНИЕ ЗДЕСЬ ===
        
        # 1. Базовая вместимость (от config)
        base_capacity = CARRYING_CAPACITY_FACTOR 
        
        # 2. Бонус от технологий (сельское хозяйство, ирригация, инфраструктура)
        # Технологии *экспоненциально* влияют на вместимость.
        # (1 + 0.1*5) = 1.5x (раннее фермерство)
        # (1 + 0.3*5) = 2.5x (развитое фермерство)
        # (1 + 1.0*5) = 6.0x (промышленное с/х)
        tech_capacity_multiplier = 1.0 + (self.tech * 5) 
        
        # 3. Бонус для Городов и Государств (торговля, логистика)
        stage_multiplier = 1.0
        if self.stage == 'city':
             stage_multiplier = 2.0  # Города в 2 раза эффективнее племен
        elif self.stage == 'state':
             stage_multiplier = 5.0  # Гос-ва (макро-агент) еще эффективнее
        
        # Итоговая вместимость = База * Пригодность * Технологии * Бонус Стадии
        carrying_capacity = max(1.0, (cell.habitability * base_capacity * tech_capacity_multiplier * stage_multiplier))
        
        # === КОНЕЦ ИЗМЕНЕНИЯ ===


        # (Весь расчет growth_factor без изменений)
        base_birth = max(0.0, BIRTH_RATE_BASE * (cell.habitability + 0.2) * (1 + self.tech))
        base_death = max(0.0, DEATH_RATE_BASE * (1.0 - cell.habitability * 0.5))
        starvation_term = self.hunger_level * DEATH_RATE_STARVATION
        dehydration_term = self.thirst_level * (DEATH_RATE_STARVATION * 0.5)
        
        # 'overpop' теперь будет рассчитываться от *новой*, динамической 'carrying_capacity'
        overpop = max(0.0, (self.population / (carrying_capacity + 1e-9)) - 1.0)
        overpop_death = overpop * 0.04
        
        age_penalty = max(0.8, 1.0 - self.age / 20000) # (Используем мягкий штраф из прошлой итерации)
        
        yearly_birth = base_birth * age_penalty
        yearly_death = base_death + starvation_term + dehydration_term + overpop_death
        
        years = max(1, SIMULATION_STEP_YEARS)
        
        base_rate = 1.0 + yearly_birth - yearly_death
        clamped_base_rate = max(0.0, base_rate) 
        growth_factor = clamped_base_rate ** years
        
        pop_before = self.population
        self.population = int(max(0, math.floor(self.population * growth_factor)))

        if self.population <= 0:
            self.alive = False
            return

        # === ЛОГИКА СНА ===
        pop_growth = abs(self.population - self.prev_population) / (self.prev_population + 1e-9)
        if self.hunger_level < 0.1 and self.thirst_level < 0.1 and pop_growth < AGENT_STABLE_GROWTH_RATE:
            self.sleep_timer = AGENT_SLEEP_THRESHOLD_STEPS
        
        self.prev_population = self.population


    # --- Общий шаг (сигнатура изменена) ---
    def step(self, cell, world, debug=False):
        """Сигнатура изменена: (self, cell, world, debug=False)"""
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
        """Проверяет, стоит ли агент на побережье"""
        cell = world.get((self.i, self.j))
        return cell and cell.is_coastal

    def __repr__(self):
        return f"<{self.stage.capitalize()} #{self.id} pop={self.population} tech={self.tech:.3f} food={self.food:.1f}>"



# =======================================
# === 3. ГРУППА (Изменена) ==============
# =======================================

class HumanGroup(BaseEntity):
    def __init__(self, entity_id, i, j, population, start_tech=0.01, home_coord=None):
        super().__init__(entity_id, i, j, population, start_tech)
        self.stage = "group"
        self.food = 300.0 
        self.path = [(i, j)]
        self.is_migrating = True
        self.steps_migrating = 0
        self.next_pos = None
        # Запоминаем "дом", чтобы уйти от него
        self.home_coord = home_coord if home_coord else (i, j)

    def absorb(self, other_entity):
        """Поглощает другую группу"""
        # (Используется, если эта группа "выиграла" клетку)
        self.population += other_entity.population
        self.food += other_entity.food
        self.tech = max(self.tech, other_entity.tech)
        other_entity.alive = False

    def _distance_from_home(self, i, j):
        """Helper: Считает 'шаговое' расстояние от дома (дистанция Чебышева)"""
        if not self.home_coord:
            return 0
        return max(abs(i - self.home_coord[0]), abs(j - self.home_coord[1]))
    
    def choose_next_direction(self, world):
        """
        Более реалистичная логика миграции:
        - стремимся уйти от дома;
        - предпочитаем более пригодные для жизни клетки;
        - сохраняем общее направление (инерция);
        - избегаем топтания на месте.
        """
        if not hasattr(self, "direction"):
            self.direction = (random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))

        dirs = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
        best_pos, best_score = None, -999
        current_dist = self._distance_from_home(self.i, self.j)

        for dx, dy in dirs:
            nx, ny = self.i + dx, self.j + dy
            if (nx, ny) in self.path[-5:]:
                continue
            cell = world.get((nx, ny))
            if not cell or not cell.is_land:
                continue

            # 1. Увеличение расстояния от дома
            dist_score = (self._distance_from_home(nx, ny) - current_dist) * 0.6

            # 2. Пригодность местности
            terrain_score = (cell.habitability * 0.7 + cell.arable * 0.5 + cell.food_availability * 0.3)

            # 3. Инерция — предпочтение прежнего направления
            dir_alignment = 1.0 if (dx, dy) == self.direction else 0.5 if (dx*dy == 0) else 0.3

            # 4. Небольшая случайность
            random_bonus = random.uniform(-0.2, 0.2)

            score = dist_score + terrain_score + dir_alignment + random_bonus

            if score > best_score:
                best_score, best_pos = score, (nx, ny)

        # обновляем направление
        if best_pos:
            self.direction = (best_pos[0] - self.i, best_pos[1] - self.j)

        return best_pos

    
    def gather_resources_migrant(self, cell):
        """Упрощенный сбор (собирательство/охота на ходу)"""
        # Собирают в 10 раз меньше, чем оседлое племя
        if cell.is_land:
            base_food = (cell.food_availability + cell.arable * 0.2) * self.population * 0.0003
            tech_bonus = 1.0 + self.tech
            self.food += base_food * tech_bonus
            cell.deplete(self.population * 0.1) # Истощают, но меньше

        self.water = max(0.0, min(1.0, self.water * 0.8 + cell.water_availability + random.uniform(0.0, 0.1)))
    
    def update_population_migrant(self):
        """
        Демография кочевой группы:
        - может немного расти, если обеспечена едой и водой;
        - голод/жажда дают высокую смертность;
        - возраст не учитывается, так как это "молодые" мигранты.
        """
        if not self.alive:
            return

        # --- Базовые коэффициенты ---
        base_birth = BIRTH_RATE_BASE * 0.5  # кочевники размножаются медленнее
        base_death = DEATH_RATE_BASE * 0.8  # немного ниже (молодые, мобильные)
        
        # --- Влияние состояния ---
        # Если еды и воды хватает — повышаем рождаемость
        resource_factor = (self.food / (self.population * self.need_food_per_capita * 100 + 1e-9))
        if resource_factor > 1.0:
            base_birth *= min(2.0, resource_factor)  # максимум ×2
        elif resource_factor < 0.5:
            base_birth *= resource_factor            # меньше еды — меньше рождаемость

        # --- Стресс от голода и жажды ---
        starvation_term = self.hunger_level * DEATH_RATE_STARVATION
        dehydration_term = self.thirst_level * (DEATH_RATE_STARVATION * 0.5)

        # --- Итоговые ставки ---
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
        """
        Шаг с "реакцией" на плохую клетку
        """
        if not self.alive: return None
        if not cell or not cell.is_land:
            self.alive = False
            return None
            
        self.next_pos = None # Сбрасываем заявку на ход

        # === НОВАЯ ЛОГИКА: РЕАКЦИЯ НА КЛЕТКУ ===
        # (Мы уже *вошли* в эту клетку в прошлом шаге)
        
        # Оцениваем, насколько плохая клетка
        # cell_quality = cell.habitability + cell.food_availability
        
        # # 0.25 - это "очень плохая" (пустыня, ледник)
        # if cell_quality < 0.25 and len(self.path) > 1:
        #     # Шанс 50% "испугаться" и отступить
        #     if random.random() < 0.5:
        #         last_pos = self.path[-2] # [-1] - это *текущая* клетка
        #         self.next_pos = last_pos
        #         if debug:
        #             print(f"  [Отступление] Группа #{self.id} отступает из {cell.biome} в {last_pos}")
        #         return None # Пропускаем остаток шага (еду, смерть и т.д.)
        # === КОНЕЦ НОВОЙ ЛОГИКИ ===

        self.age += SIMULATION_STEP_YEARS

        # 1. Упрощенная жизнь мигранта
        self.gather_resources_migrant(cell)
        self.consume_resources(cell)
        self.update_population_migrant() # (Используем твой "исправленный" migrant_pop)

        if not self.alive: # Умерли в пути
            return None

        # 2. Обновляем таймер "иммунитета"
        self.steps_migrating += 1
        if self.steps_migrating > MIGRATION_IMMUNITY_STEPS:
            self.is_migrating = False

        # 3. Эволюция в племя (если нашли хорошее место и выжили)
        evolve_cf = cell.arable * cell.habitability
        if self.population > TRIBE_FOUNDING_THRESHOLD and evolve_cf > 0.4 and not self.is_migrating:
            if debug:
                print(f"  [Эволюция] Группа #{self.id} основала племя в ({self.i},{self.j})")
            tribe = Tribe(self.id, self.i, self.j, self.population, start_tech=self.tech) 
            self.alive = False 
            return tribe 

        # 4. Движение
        new_pos = self.choose_next_direction(world)
        if new_pos:
            self.next_pos = new_pos 
        else:
            # Некуда идти, вынужденно оседаем
            self.is_migrating = False
        
        # 5. Превращение в мореплавателей, если достигли побережья и технологий
        if cell.is_coastal and self.tech >= SEAFARING_TECH_THRESHOLD and random.random() < 0.5:
            if debug:
                print(f"  [Эволюция] Группа #{self.id} ({self.i},{self.j}) стала мореплавателями (tech={self.tech:.3f})")
            self.alive = False
            return SeafaringGroup(self.id, self.i, self.j, self.population, start_tech=self.tech)
            
        return None


# =======================================
# === 4. ПЛЕМЯ (Изменено) ===============
# =======================================

class Tribe(BaseEntity):
    def __init__(self, entity_id, i, j, population, start_tech=0.05): 
        super().__init__(entity_id, i, j, population, start_tech) 
        self.stage = "tribe"
        self.food = 300.0
    
    def absorb(self, other_entity):
        """Поглощает другую сущность (группу или племя)"""
        self.population += other_entity.population
        self.food += other_entity.food
        self.tech = max(self.tech, other_entity.tech)
        other_entity.alive = False
        # print(f"  [Агрегация] Племя #{self.id} поглотило сущность #{other_entity.id}")

    def find_spawn_location(self, world):
        """Ищет безопасную соседнюю клетку для спавна мигрантов"""
        dirs = [(1,0),(-1,0),(0,1),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]
        random.shuffle(dirs)
        
        for dx, dy in dirs:
            nx, ny = self.i + dx, self.j + dy
            cell = world.get((nx, ny))
            # Ищем любую соседнюю *сушу*
            if cell and cell.is_land:
                return (nx, ny) 
        
        # Если мы на острове 1x1, миграция невозможна
        return None
    
    def get_stress_level(self, cell):
        """
        Расчет "уровня стресса" для миграции (0..1+)
        ИСПОЛЬЗУЕТ "МЯГКИЙ" ПРЕДЕЛ ИЗ КОНФИГА
        """
        # "Мягкая" вместимость (для миграции), ИЗ CONFIG
        migration_capacity = max(1.0, cell.habitability * CELL_CAPACITY_SCALE) 
        population_ratio = self.population / migration_capacity
        
        # Стресс от перенаселения НАЧИНАЕТСЯ после порога (из config)
        overpop_stress = max(0.0, population_ratio - OVERPOPULATION_THRESHOLD)
        
        # Итоговый стресс = Голод + Перенаселение.
        stress = self.hunger_level + overpop_stress
        return stress

    def step(self, cell, world, debug=False):
        if not self.alive: return None
        if not cell or not cell.is_land:
            self.alive = False
            return None

        super().step(cell, world, debug)
        
        # 1. Эволюция в Город
        if self.population > CITY_FOUNDING_THRESHOLD and self.tech > 0.1:
            if debug:
                print(f"  [Эволюция] Племя #{self.id} стало городом ({self.i},{self.j})")
            self.alive = False 
            return City(self.id, self.i, self.j, self.population, start_tech=self.tech)

        # 2. Миграция (сухопутная)
        stress = self.get_stress_level(cell)
        if stress > MIGRATION_STRESS_THRESHOLD and self.population > 100 and random.random() < 0.1: # <--- Твой код с random
            new_pop = int(self.population * MIGRATION_PERCENTAGE)
            if new_pop > 50: 
                spawn_pos = self.find_spawn_location(world)
                if not spawn_pos: # Некуда "вытолкнуть" группу (например, остров 1x1)
                    return None # Миграция не удалась
                
                migrant_tech = self.tech * 0.8 
                # Создаем группу в *соседней* клетке
                new_group = HumanGroup(random.randint(10000, 99999), *spawn_pos, new_pop, 
                                   start_tech=migrant_tech, home_coord=(self.i, self.j))
                self.population -= new_pop
                if debug:
                    print(f"  [Миграция] Племя #{self.id} (стресс={stress:.2f}) породило группу #{new_group.id} (tech={migrant_tech:.3f})")
                return new_group 
        
        return None


# =======================================
# === 5. ГОРОД (Изменен) ================
# =======================================

class City(BaseEntity):
    def __init__(self, entity_id, i, j, population, start_tech=0.2): 
        super().__init__(entity_id, i, j, population, start_tech) 
        self.stage = "city"
        self.food = 1000.0
        self.influence_radius = CITY_INFLUENCE_RADIUS

    def get_stress_level(self, cell):
        """
        Расчет "уровня стресса" для миграции (0..1+)
        ИСПОЛЬЗUЕТ "МЯГКИЙ" ПРЕДЕЛ ИЗ КОНФИГА
        (Логика идентична Tribe.get_stress_level)
        """
        # "Мягкая" вместимость (для миграции), ИЗ CONFIG
        migration_capacity = max(1.0, cell.habitability * CELL_CAPACITY_SCALE) 
        population_ratio = self.population / migration_capacity
        
        # Стресс от перенаселения НАЧИНАЕТСЯ после порога (из config)
        overpop_stress = max(0.0, population_ratio - OVERPOPULATION_THRESHOLD)
        
        # Итоговый стресс = Голод + Перенаселение.
        stress = self.hunger_level + overpop_stress
        return stress

    def find_spawn_location(self, world):
        """Ищет безопасную соседнюю клетку для спавна мигрантов"""
        dirs = [(1,0),(-1,0),(0,1),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]
        random.shuffle(dirs)
        
        for dx, dy in dirs:
            nx, ny = self.i + dx, self.j + dy
            cell = world.get((nx, ny))
            # Ищем любую соседнюю *сушу*
            if cell and cell.is_land:
                return (nx, ny) 
        
        # Если мы на острове 1x1, миграция невозможна
        return None
    
    def absorb(self, other_entity):
        """Поглощает другую сущность (группу или племя)"""
        # (Код из твоего absorb(tribe) просто обобщен)
        self.population += other_entity.population
        self.food += other_entity.food
        self.tech = max(self.tech, other_entity.tech)
        other_entity.alive = False
        # print(f"  [Агрегация] Город #{self.id} поглотил сущность #{other_entity.id}")

    def step(self, cell, world, debug=False):
        if not self.alive: return None
        if not cell: 
            self.alive = False
            return None
            
        super().step(cell, world, debug)
        
        # Эволюция в Государство обрабатывается в Simulation.step()

        # Миграция (сухопутная)
        stress = self.get_stress_level(cell)
        if stress > MIGRATION_STRESS_THRESHOLD and self.population > 1000 and random.random() < 0.1: # <--- Твой код с random
            new_pop = int(self.population * MIGRATION_PERCENTAGE * 0.5) 
            if new_pop > 100:
                spawn_pos = self.find_spawn_location(world)
                if not spawn_pos: 
                    return None 
                
                migrant_tech = self.tech * 0.8
                # Создаем группу в *соседней* клетке
                new_group = HumanGroup(random.randint(10000, 99999), *spawn_pos, new_pop, 
                                   start_tech=migrant_tech, home_coord=(self.i, self.j))
                
                self.population -= new_pop
                if debug:
                     print(f"  [Миграция] Город #{self.id} (стресс={stress:.2f}) породил группу #{new_group.id} в {spawn_pos}")
                return new_group
        return None


# =======================================
# === 6. МОРЕПЛАВАТЕЛИ (Новый) =========
# =======================================

class SeafaringGroup(BaseEntity):
    def __init__(self, entity_id, i, j, population, start_tech=0.01):
        super().__init__(entity_id, i, j, population, start_tech)
        self.stage = "seafaring"
        self.food = SEAFARING_FOOD_START * (population / 50)
        self.water = 0.9
        self.need_food_per_capita = 0.003
        self.steps_at_sea = 0
        self.ignore_land_steps = 8  # 🔹 первые 8 шагов не возвращаемся к берегу
        self.direction = random.choice([(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)])

    def update_population_seafaring(self):
        """
        Упрощенная демография для моряков. 
        НЕТ "overpopulation_death". Смерть только от голода и тягот пути.
        (Скопировано из HumanGroup)
        """
        if not self.alive:
            return

        # В пути нет чистого прироста, только базовая смертность и стресс
        yearly_birth = DEATH_RATE_BASE 
        yearly_death = DEATH_RATE_BASE
        
        # Стресс от голода и жажды
        starvation_term = self.hunger_level * DEATH_RATE_STARVATION
        dehydration_term = self.thirst_level * (DEATH_RATE_STARVATION * 0.5)

        years = max(1, SIMULATION_STEP_YEARS)
        
        # Расчет базы БЕЗ 'overpop_death' и БЕЗ 'age_penalty'
        base_rate = 1.0 + yearly_birth - (yearly_death + starvation_term + dehydration_term)
        clamped_base_rate = max(0.0, base_rate) 
        growth_factor = clamped_base_rate ** years

        self.population = int(max(0, math.floor(self.population * growth_factor)))

        if self.population <= 0:
            self.alive = False

    def gather_resources(self, cell):
        """Рыбалка в океане"""
        self.food += cell.properties.get("food_animal", 0) * self.population * 0.0005
        # Вода только тратится
        self.water = max(0.0, self.water - 0.05) 

    def choose_next_direction(self, world):
        """Плавание с инерцией и временным игнорированием суши"""
        self.steps_at_sea += 1

        # Пока не истёк "игнор суши" — плывём по инерции
        if self.steps_at_sea < self.ignore_land_steps:
            nx, ny = self.i + self.direction[0], self.j + self.direction[1]
            cell = world.get((nx, ny))
            if cell and not cell.is_land:
                return (nx, ny)

        # Ищем землю только после того, как уплыли достаточно далеко
        best_land_pos = None
        min_dist = SEAFARING_LAND_SENSE_RADIUS + 1
        for r in range(1, SEAFARING_LAND_SENSE_RADIUS + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = self.i + dx, self.j + dy
                    cell = world.get((nx, ny))
                    if cell and cell.is_land:
                        dist = max(abs(dx), abs(dy))
                        if dist < min_dist:
                            min_dist = dist
                            best_land_pos = (self.i + (1 if dx > 0 else -1 if dx < 0 else 0),
                                             self.j + (1 if dy > 0 else -1 if dy < 0 else 0))
            if best_land_pos:
                # не сразу к суше — шанс 50% "игнорировать" её
                if random.random() < 0.5:
                    continue
                self.direction = (best_land_pos[0] - self.i, best_land_pos[1] - self.j)
                return best_land_pos

        # иначе — случайное дрейфование с инерцией
        dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = self.i + dx, self.j + dy
            cell = world.get((nx, ny))
            if cell and not cell.is_land:
                self.direction = (dx, dy)
                return (nx, ny)

        return None

    def step(self, cell, world, debug=False):
        if not self.alive: return None
        
        # 1. Мы приплыли!
        if cell.is_land:
            self.alive = False
            if debug:
                print(f"  [Колонизация] Группа #{self.id} высадилась в ({self.i},{self.j})!")
            # Превращаемся в обычную группу мигрантов
            return HumanGroup(self.id, self.i, self.j, self.population, self.tech)

        # 2. Мы еще в море
        # super().step() тут не подходит, т.к. другая логика
        self.age += SIMULATION_STEP_YEARS
        self.gather_resources(cell)
        self.consume_resources(cell) # Тратим еду/воду
        self.update_population_seafaring() # Люди могут умирать в пути

        if not self.alive: # Погибли в море
            if debug: print(f"  [Потеря] Группа #{self.id} погибла в океане.")
            return None

        new_pos = self.choose_next_direction(world)
        if new_pos:
            self.move_to(*new_pos)
        else:
            self.alive = False # Застряли (например, в озере)
            
        return None


# =======================================
# === 7. ГОСУДАРСТВО (Новый) ===========
# =======================================

class State:
    """Макро-агент, не наследуется от BaseEntity!"""
    def __init__(self, entity_id, i, j, population, tech):
        self.id = entity_id
        self.i, self.j = i, j # Столица
        self.population = int(population)
        self.tech = tech
        self.age = 0
        self.alive = True
        self.stage = "state"
        self.territory = set() # Набор (i, j) всех клеток
        self.cities_coords = [] # Координаты (i, j)
        self.is_coastal = False # Есть ли выход к морю
        self.need_food_per_capita = 0.004
        self.expansion_budget = 0.0 # <--- ДОБАВИТЬ "ОЧКИ ВЛИЯНИЯ"

    def absorb_entity(self, entity, world):
        """Поглощает сущность при формировании ИЛИ мигрантов"""
        self.population += entity.population
        # Технологии мигрантов ассимилируются (с бонусом)
        if entity.tech > self.tech:
            self.tech = min(1.0, self.tech + (entity.tech - self.tech) * 0.1) 
        
        # Добавляем территорию, если это племя или город
        if not isinstance(entity, HumanGroup):
             self.territory.add((entity.i, entity.j))
             if isinstance(entity, City):
                 self.cities_coords.append((entity.i, entity.j))
        
        if not self.is_coastal:
            cell = world.get((entity.i, entity.j))
            if cell and cell.is_coastal:
                self.is_coastal = True

        entity.alive = False

    def get_expansion_candidates(self, world, all_claimed_cells, nx, ny):
        """
        Новая "умная" логика: находит и ОЦЕНИВАЕТ все
        пограничные клетки по их "желанности".
        """
        candidates = {} # dict[coord, score]
        
        if nx is None or ny is None:
            return [] # Не можем работать без размеров карты

        for (i, j) in self.territory:
            # Проверяем 8 соседей (для "заполнения" дыр)
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0: continue
                    
                    check_pos = ((i + di) % nx, (j + dj) % ny)
                    
                    # 1. Кандидат?
                    # - Не наша территория И не чужая
                    if check_pos not in self.territory and check_pos not in all_claimed_cells:
                        cell = world.get(check_pos)
                        
                        # 2. Это суша?
                        if cell and cell.is_land:
                            
                            # 3. Оценка "желанности"
                            score = candidates.get(check_pos, 0.0)
                            
                            # +1.0 за пригодность (базовая ценность)
                            score += cell.habitability
                            
                            # +2.0 за плодородность (сельское хозяйство)
                            score += cell.arable * 2.0 
                            
                            # +1.5 за побережье (торговля/флот)
                            if cell.is_coastal:
                                score += 1.5
                                
                            # +0.5 за *каждого* соседа, который УЖЕ наш
                            # (!!!) ЭТО РЕАЛИЗУЕТ ТВОЙ ЗАПРОС "СОЕДИНИТЬ ТЕРРИТОРИИ"
                            # Клетка- "дырка", окруженная 8-ю нашими,
                            # получит +4.0 и будет захвачена в первую очередь.
                            if (i, j) in self.territory:
                                score += 0.5
                                
                            candidates[check_pos] = score

        # Возвращаем отсортированный список: (оценка, (i, j))
        sorted_candidates = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        return [(score, pos) for pos, score in sorted_candidates]

    def step(self, world, debug=False):
        """Шаг макро-агента (упрощенная экономика и рост)"""
        if not self.alive: return []
        self.age += SIMULATION_STEP_YEARS
        
        new_entities = []
        
        # 1. Макро-Экономика (очень упрощенно)
        total_habitability = 0.0
        total_food_prod = 0.0
        for (i, j) in self.territory:
            cell = world.get((i, j))
            if cell:
                total_habitability += cell.habitability
                total_food_prod += (cell.arable + cell.food_availability)
        
        avg_habitability = total_habitability / (len(self.territory) + 1e-9)
        food_production = total_food_prod * (1 + self.tech) * MACRO_FOOD_PRODUCTION_FACTOR * len(self.territory)
        food_needed = self.population * self.need_food_per_capita
        
        food_surplus_ratio = (food_production - food_needed) / (food_needed + 1e-9)
        
        # 2. Макро-Демография
        # (Используем годовые ставки и возводим в степень)
        years = max(1, SIMULATION_STEP_YEARS)
        yearly_birth = MACRO_BIRTH_RATE * (1 + avg_habitability * 0.5) * (1 + min(0.5, food_surplus_ratio))
        yearly_death = MACRO_DEATH_RATE * (1 - avg_habitability * 0.5) * (1 - max(-0.5, food_surplus_ratio * 0.5))
        
        growth_factor = (1.0 + yearly_birth - yearly_death) ** years
        self.population = int(max(1, self.population * growth_factor))

        # 3. Макро-Технологии
        tech_gain = (len(self.cities_coords) * 0.1) * (self.population / 100000.0) * MACRO_TECH_FACTOR
        self.tech = min(1.0, self.tech + tech_gain)
        
        # 4. Колонизация (только морская, как по заданию)
        if self.is_coastal and self.tech > SEAFARING_TECH_THRESHOLD and random.random() < SEAFARING_SPAWN_CHANCE:
            # Ищем случайный прибрежный город для старта
            start_coord = random.choice(self.cities_coords) if self.cities_coords else (self.i, self.j)
            new_pop = random.randint(100, 300)
            
            # Находим ближайшую к городу *водную* клетку
            water_cell_pos = None
            for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
                check_pos = (start_coord[0] + di, start_coord[1] + dj)
                cell = world.get(check_pos)
                if cell and not cell.is_land:
                    water_cell_pos = check_pos
                    break
            
            if water_cell_pos:
                self.population -= new_pop
                new_colonists = SeafaringGroup(random.randint(10000, 99999), *water_cell_pos, new_pop, start_tech=self.tech * 0.7)
                new_entities.append(new_colonists)
                if debug:
                    print(f"  [Колонизация] Гос-во #{self.id} отправило группу #{new_colonists.id} в море!")

        if debug:
            print(f"[STATE #{self.id}] Pop={self.population}, Tech={self.tech:.4f}, Territory={len(self.territory)} cells, Cities={len(self.cities_coords)}")

        return new_entities


# =======================================
# === 8. СИМУЛЯЦИЯ (Изменена) ==========
# =======================================

def distance(i1, j1, i2, j2):
    """Простая дистанция (Чебышев) для расчета радиусов"""
    return max(abs(i1 - i2), abs(j1 - j2))

class Simulation:
    def __init__(self, world_file="world_cells.json", nx=None, ny=None):
        self.world = load_world(world_file, nx, ny) # <--- ИЗМЕНЕНО
        self.entities = []
        self.year = START_YEAR
        self.running = True
        self.occupied_cells = set() # Для оптимизации регенерации
        self.nx = nx 
        self.ny = ny

    def initialize(self):
        start = HumanGroup(0, *STARTING_CELL_COORDS, STARTING_POPULATION)
        self.entities = [start]

    def regenerate_world(self):
        """Регенерирует случайную часть пустых клеток (для FPS)"""
        all_cells_coords = list(self.world.keys())
        random.shuffle(all_cells_coords)
        
        sample_size = int(len(all_cells_coords) * CELL_REGEN_TICK_RATE)
        
        for i in range(sample_size):
            coord = all_cells_coords[i]
            if coord not in self.occupied_cells:
                self.world[coord].regenerate()
    
    def resolve_interactions(self, debug=False):
        """
        НОВАЯ ФАЗА 3:
        Находит клетки с >1 агентом и разрешает конфликты/слияния.
        """
        occupied_cells = {} # dict[coord, list[Entity]]
        
        # 1. Собрать всех, кто где стоит
        for e in self.entities:
            if not e.alive: continue
            occupied_cells.setdefault((e.i, e.j), []).append(e)
            
        # 2. Обработать "конфликтные" клетки
        for coord, occupants in occupied_cells.items():
            if len(occupants) <= 1:
                continue

            # --- Логика, кто "владелец" клетки ---
            # 1. Гос-во > Город > Племя > Группа
            # 2. Если равны - побеждает тот, у кого больше населения
            
            def get_entity_priority(e):
                if isinstance(e, State): return 4
                if isinstance(e, City): return 3
                if isinstance(e, Tribe): return 2
                if isinstance(e, HumanGroup): return 1
                return 0
            
            # Сортируем: самый "сильный" будет первым
            occupants.sort(key=lambda e: (get_entity_priority(e), e.population), reverse=True)
            
            owner = occupants[0]
            losers = occupants[1:]
            
            if debug and losers:
                print(f"  [Интеракция] В клетке {coord}: {owner.stage} #{owner.id} (Pop: {owner.population}) 'победил' {len(losers)} других агентов.")

            # 3. "Владелец" поглощает всех остальных
            for loser in losers:
                # В State и City своя логика поглощения, в Tribe/Group - общая
                if isinstance(owner, State) or isinstance(owner, City):
                    owner.absorb_entity(loser, self.world) if isinstance(owner, State) else owner.absorb(loser)
                elif hasattr(owner, 'absorb'):
                    owner.absorb(loser)
                else:
                    # На всякий случай, если у "владельца" нет .absorb 
                    # (например, SeafaringGroup - хотя он не должен быть на суше)
                    loser.alive = False 
                
                # Если "владелец" - это Группа, она прекращает миграцию
                if isinstance(owner, HumanGroup):
                    owner.is_migrating = False

    def step_aggregation(self, debug=False):
        """Фаза агрегации: Города поглощают Племена, Города становятся Государствами"""
        
        # Разделяем для удобства (но работаем с self.entities)
        cities = [e for e in self.entities if isinstance(e, City) and e.alive]
        tribes = [e for e in self.entities if isinstance(e, Tribe) and e.alive]
        states = [e for e in self.entities if isinstance(e, State) and e.alive]
        
        new_states = []
        entities_to_remove = set()

        # 1. Города поглощают Племена
        for city in cities:
            if city in entities_to_remove: continue
            for tribe in tribes:
                if tribe in entities_to_remove: continue
                if distance(city.i, city.j, tribe.i, tribe.j) <= city.influence_radius:
                    city.absorb(tribe)
                    entities_to_remove.add(tribe)

        # 2. Города (и племена) поглощаются существующими Государствами
        for state in states:
            for entity in (cities + tribes):
                if entity in entities_to_remove: continue
                # Проверяем, находится ли агент на территории гос-ва (грубо)
                if (entity.i, entity.j) in state.territory:
                     state.absorb_entity(entity, self.world)
                     entities_to_remove.add(entity)
                     continue
                # Проверяем, не вошел ли он в радиус столицы (грубо)
                if distance(state.i, state.j, entity.i, entity.j) <= STATE_INFLUENCE_RADIUS:
                    state.absorb_entity(entity, self.world)
                    entities_to_remove.add(entity)

        # 3. Города формируют новые Государства
        eligible_cities = [c for c in cities if c.population > STATE_FOUNDING_POP and c.tech > STATE_FOUNDING_TECH and c not in entities_to_remove]
        
        for city in eligible_cities:
            if city in entities_to_remove: continue
            
            if debug:
                print(f"  [Эволюция] Город #{city.id} ({city.i},{city.j}) основывает ГОСУДАРСТВО!")
            
            # 1. Создаем новое Государство
            new_state = State(city.id, city.i, city.j, 0, city.tech)
            entities_to_remove.add(city)
            
            # 2. Поглощаем всех в радиусе
            entities_to_absorb = [e for e in self.entities if isinstance(e, (Tribe, City)) and e.alive and e not in entities_to_remove]
            
            for entity in entities_to_absorb:
                if distance(city.i, city.j, entity.i, entity.j) <= STATE_INFLUENCE_RADIUS:
                    new_state.absorb_entity(entity, self.world)
                    entities_to_remove.add(entity)
            
            # Если столица не поглотилась (редко, но бывает), добавляем ее
            if (city.i, city.j) not in new_state.territory:
                 new_state.absorb_entity(city, self.world) # city уже в entities_to_remove

            new_states.append(new_state)

        # Применяем изменения
        if new_states:
            self.entities.extend(new_states)
        
        if entities_to_remove:
            self.entities = [e for e in self.entities if e not in entities_to_remove]

    def step(self, debug=False):
        if not self.running or not self.entities:
            self.running = False
            return self.entities, self.year
            
        self.year += SIMULATION_STEP_YEARS
        new_entities = []
        entities_to_remove = set()
        move_requests = [] # <--- НОВОЕ: список для заявок на ход
        
        self.occupied_cells = {e.i: e.j for e in self.entities if e.alive and isinstance(e, BaseEntity)}
        
        # 1. Шаг регенерации мира
        self.regenerate_world()

        # ===============================================
        # === ФАЗА 1: ЛОГИКА И ЗАЯВКИ НА ХОД ==========
        # ===============================================
        for e in list(self.entities):
            if not e.alive:
                entities_to_remove.add(e)
                continue
            
            # Логика "Сна"
            if isinstance(e, BaseEntity) and e.sleep_timer > 0:
                e.sleep_timer -= 1
                continue
            
            # Шаг для Макро-Агентов (Государств)
            if isinstance(e, State):
                results = e.step(self.world, debug=debug)
                if results:
                    new_entities.extend(results)
            
            # Шаг для Базовых Агентов
            elif isinstance(e, BaseEntity):
                cell = self.world.get((e.i, e.j))
                if not cell:
                    e.alive = False
                    entities_to_remove.add(e)
                    continue
                
                # VVV Убираем entity_map из сигнатуры VVV
                result = e.step(cell, self.world, debug=debug) 
                
                if result:
                    new_entities.append(result)
                
                # VVV НОВЫЙ БЛОК: Собираем "заявки на ход" VVV
                if isinstance(e, HumanGroup) and e.next_pos:
                    move_requests.append(e)

        # 3. Применяем добавление/удаление
        if new_entities:
            self.entities.extend(new_entities)
        
        if entities_to_remove:
            self.entities = [e for e in self.entities if e not in entities_to_remove]
        
        # ===============================================
        # === ФАЗА 2: ДВИЖЕНИЕ ==========================
        # ===============================================
        for group in move_requests:
            if group.alive: # Мог умереть в Фазе 1
                group.move_to(*group.next_pos)
                
                # VVV ДОБАВИТЬ ЭТИ 3 СТРОКИ VVV
                group.path.append(group.next_pos)
                if len(group.path) > 100:
                    group.path.pop(0)
                
                group.next_pos = None

        # ===============================================
        # === ФАЗА 3: РАЗРЕШЕНИЕ ИНТЕРАКЦИЙ ===========
        # ===============================================
        # (Проверяем, только если были движения)
        if move_requests:
             self.resolve_interactions(debug=debug)

        # 4. Фаза Агрегации (Города -> Гос-ва)
        self.step_aggregation(debug=debug)
        
        # === НОВАЯ ФАЗА 4.5: РЕАЛИСТИЧНОЕ РАСШИРЕНИЕ ===
        
        # 1. Собираем ВСЕ занятые гос-вами клетки
        all_claimed_cells = set()
        states = [e for e in self.entities if isinstance(e, State)]
        for s in states:
            all_claimed_cells.update(s.territory)
            
        # 2. Каждый штат накапливает "бюджет" и тратит его
        for s in states:
            # Накапливаем "очки влияния" (бюджет)
            # (Зависит от населения и технологий)
            s.expansion_budget += (s.population / 150000.0) + (s.tech * 1.0)
            
            # Получаем список лучших клеток для захвата
            candidates = s.get_expansion_candidates(self.world, all_claimed_cells, self.nx, self.ny)
            
            # Тратим бюджет, пока он не кончится
            # (Стоимость захвата = 1.0 очка)
            while s.expansion_budget >= 1.0 and candidates:
                # Берем лучшего кандидата
                best_cell = candidates.pop(0)
                best_pos = best_cell[1]
                
                # Захватываем
                s.territory.add(best_pos)
                all_claimed_cells.add(best_pos) # "Резервируем" клетку
                s.expansion_budget -= 1.0

                # Проверяем, стала ли страна прибрежной
                if not s.is_coastal:
                    cell = self.world.get(best_pos)
                    if cell and cell.is_coastal:
                        s.is_coastal = True
                
                # (Нужно также обновить список кандидатов, 
                # т.к. мы изменили границу, но для простоты пропустим)
             
        # 5. Очистка мертвых
        self.entities = [e for e in self.entities if e.alive]
        
        if not self.entities:
            self.running = False

        return self.entities, self.year


# =======================================
# === 9. ТЕСТ ===========================
# =======================================

if __name__ == "__main__":
    sim = Simulation()
    sim.initialize()
    for i in range(1000): # Увеличим время симуляции
        entities, year = sim.step(debug=False) # Выключаем debug для скорости
        
        # Печатаем сводку каждые 100 лет
        if i % 10 == 0:
            counts = {"Group": 0, "Tribe": 0, "City": 0, "Seafaring": 0, "State": 0}
            total_pop = 0
            for e in entities:
                counts[e.stage.capitalize()] += 1
                total_pop += e.population
            
            print(f"--- Год: {year} | Агентов: {len(entities)} | Всего населения: {total_pop} ---")
            print(f'Население: {max([e.population for e in entities])}, средний: {sum([e.population for e in entities]) / len(entities)}')
            print(f'Технологии: {max([e.tech for e in entities])}, средний: {sum([e.tech for e in entities]) / len(entities)}')
            print(f"    {counts}")
            
            if not entities:
                print("Симуляция завершена: все вымерли.")
                break
    
    print("Финальный отчет:")
    print(f"{year}: {entities}")