# Параметры симуляции
# --- Временные ---
START_YEAR = -100000  # 10,000 лет до н.э.
SIMULATION_STEP_YEARS = 10 # 1 шаг = 10 лет для ускорения
END_YEAR = 2000
CHECKPOINT_INTERVAL = 10 # Сохранять состояние каждые 100 лет (в headless)

# --- Стартовые ---
# Координаты (i, j) для старта в Африке (нужно найти на вашей карте)
STARTING_CELL_COORDS = (420, 170) # Пример, замените на реальные
STARTING_POPULATION = 1000

# --- Биология и Демография ---
CARRYING_CAPACITY_FACTOR = 5000
FOOD_WASTAGE_RATE = 0.1 # 10% еды портится
BIRTH_RATE_BASE = 0.03 # 3% в год при идеальных условиях
DEATH_RATE_BASE = 0.02 # 2% в год (старость, болезни)
DEATH_RATE_STARVATION = 0.1 # 10% доп. смертность при нехватке еды

FOOD_YIELD_SCALE = 5000      # скалирует производство пищи (ранее не было отдельно)
FOOD_CONSUMPTION_PER_CAPITA = 0.9   # Количество еды на человека

# --- Миграция ---
CELL_CAPACITY_SCALE = 1000   # Вместимость клетки (сколько людей может комфортно жить)
OVERPOPULATION_THRESHOLD = 0.7 # Порог населения (как % от "вместимости"), > которого начинается миграция
MIGRATION_PERCENTAGE = 0.1 # % населения, который мигрирует
MAX_GROUPS = 5000
# Логика миграции
MIGRATION_IMMUNITY_STEPS = 15     # 150 лет "иммунитета" у новых групп
MIGRATION_STRESS_THRESHOLD = 0.6  # Порог для миграции племен/городов (голод/перенаселение)

# --- Политика ---
# Порог населения для основания "племени"
TRIBE_FOUNDING_THRESHOLD = 100
# Порог населения для "города"
CITY_FOUNDING_THRESHOLD = 10000
# Порог населения, Порог технологий для "Государства"
STATE_FOUNDING_POP = 200000
STATE_FOUNDING_TECH = 0.4

# --- Технологии ---
# Шанс на "прорыв" в технологии за шаг (на группу)
TECH_DISCOVERY_CHANCE_BASE = 0.1
# Как сильно плотность населения влияет на открытия
TECH_DENSITY_FACTOR = 0.1
# Технологии, необходимые для морских путешествий
SHIPBUILDING_TECH_FOR_COASTAL = 0.1 # Побережье
SHIPBUILDING_TECH_FOR_OCEAN = 0.4 # Открытый океан

# Истощение и регенерация
RESOURCE_REGENERATION_RATE = 0.005  # 0.5% в шаг
RESOURCE_DEPLETION_RATE = 0.00001 # коэфф. истощения от населения
CELL_REGEN_TICK_RATE = 0.1 # 10% клеток регенерируют каждый шаг (для FPS)

# Логика сна
AGENT_SLEEP_THRESHOLD_STEPS = 5 # "Уснуть" на 5 шагов (50 лет)
AGENT_STABLE_GROWTH_RATE = 0.01 # Стабильный, если рост < 1%

# Логика мореплавания
SEAFARING_TECH_THRESHOLD = 0.3    # Технология для постройки лодок
SEAFARING_FOOD_START = 100.0      # Запас еды на старте
SEAFARING_LAND_SENSE_RADIUS = 5   # "Видимость" земли из океана (в клетках)
SEAFARING_SPAWN_CHANCE = 0.1     # Шанс для Гос-ва отправить колонистов

# Логика агрегации
CITY_INFLUENCE_RADIUS = 5        # "Зона влияния" города (поглощает племена)

STATE_INFLUENCE_RADIUS = 7       # "Зона влияния" при формировании Гос-ва

# Логика макро-агента "Государство"
MACRO_FOOD_PRODUCTION_FACTOR = 0.05 # Базовый коэфф. еды
MACRO_TECH_FACTOR = 0.0001        # Скорость роста технологий
MACRO_BIRTH_RATE = 0.02           # Базовая годовая рождаемость
MACRO_DEATH_RATE = 0.015          # Базовая годовая смертность

# --- Конец игры ---
# Биомы, которые не нужно колонизировать для победы
POLAR_BIOMES = {"Snowy Tundra", "Snowy Mountains", "Snowy Taiga Mountains", "Deep Frozen Ocean"}