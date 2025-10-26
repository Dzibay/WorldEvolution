# config.py
# Параметры симуляции

# --- Временные ---
START_YEAR = -100000  # 100,000 лет до н.э.
SIMULATION_STEP_YEARS = 10 # 1 шаг = 10 лет для ускорения
END_YEAR = 2024
CHECKPOINT_INTERVAL = 100 # Сохранять состояние каждые 100 лет (в headless)

# --- Стартовые ---
# Координаты (i, j) для старта в Африке (нужно найти на вашей карте)
STARTING_CELL_COORDS = (420, 160) # Пример, замените на реальные
STARTING_POPULATION = 1000

# --- Биология и Демография ---
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


# --- Политика ---
# Порог населения для основания "племени"
TRIBE_FOUNDING_THRESHOLD = 100
# Порог населения для "города"
CITY_FOUNDING_THRESHOLD = 2000
# Порог технологии "управления" для слияния племен в государство
STATE_MERGE_TECH_THRESHOLD = 0.3

# --- Технологии ---
# Шанс на "прорыв" в технологии за шаг (на группу)
TECH_DISCOVERY_CHANCE_BASE = 0.001
# Как сильно плотность населения влияет на открытия
TECH_DENSITY_FACTOR = 0.05
# Технологии, необходимые для морских путешествий
SHIPBUILDING_TECH_FOR_COASTAL = 0.1 # Побережье
SHIPBUILDING_TECH_FOR_OCEAN = 0.4 # Открытый океан

# --- Конец игры ---
# Биомы, которые не нужно колонизировать для победы
POLAR_BIOMES = {"Snowy Tundra", "Snowy Mountains", "Snowy Taiga Mountains", "Deep Frozen Ocean"}