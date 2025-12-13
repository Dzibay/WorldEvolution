# ============================================
# REALISTIC WORLD EVOLUTION — CONFIGURATION
# tuned for:
#   • first cities   ≈ −15000
#   • first states   ≈ −10000
#   • stable demography, no exponential boom
#   • realistic tech spread
#   • calibrated migration
# ============================================


# ======================
# === TIME PARAMETERS ==
# ======================

START_YEAR = -100000              # initial human diffusion period
SIMULATION_STEP_YEARS = 10        # each simulation step = 10 years
END_YEAR = 2000
CHECKPOINT_INTERVAL = 100


# =========================
# === INITIAL POPULATION ==
# =========================

STARTING_CELL_COORDS = (420, 170)
STARTING_POPULATION = 1000       # typical size of large hunter-gatherer population


# =======================
# === HUMAN BIOLOGY =====
# =======================

# K factor for small agents (groups, tribes, cities)
CARRYING_CAPACITY_FACTOR = 8000   # adjusted for earlier city formation

FOOD_WASTAGE_RATE = 0.1
BIRTH_RATE_BASE = 0.025           # base for small groups
DEATH_RATE_BASE = 0.02
DEATH_RATE_STARVATION = 0.1

FOOD_YIELD_SCALE = 5000
FOOD_CONSUMPTION_PER_CAPITA = 0.9


# ======================
# === MIGRATION LOGIC ==
# ======================

CELL_CAPACITY_SCALE = 4000        # effective land capacity per cell
OVERPOPULATION_THRESHOLD = 1.0    # overpop starts only when > 100%
MIGRATION_PERCENTAGE = 0.1
MAX_GROUPS = 5000

MIGRATION_IMMUNITY_STEPS = 15
MIGRATION_STRESS_THRESHOLD = 1.2  # prevents excessive splitting


# ========================
# === SOCIETAL EVOLUTION =
# ========================

TRIBE_FOUNDING_THRESHOLD = 500
CITY_FOUNDING_THRESHOLD = 5000    # earlier city formation ≈ −15000
STATE_FOUNDING_POP = 25000
STATE_FOUNDING_TECH = 0.25       # calibrated for ≈ −10000 appearance


# ============================
# === TECHNOLOGY EVOLUTION ===
# ============================

# per-step discovery chance (adjusted to produce tech ≈ 0.1 by −15000)
TECH_DISCOVERY_CHANCE_BASE = 0.01

# influence of population density
TECH_DENSITY_FACTOR = 0.7

# seafaring
SHIPBUILDING_TECH_FOR_COASTAL = 0.1
SHIPBUILDING_TECH_FOR_OCEAN = 0.4

# macro technological growth inside states
MACRO_TECH_FACTOR = 0.0005


# =================
# === RESOURCES ===
# =================

RESOURCE_REGENERATION_RATE = 0.01      # faster environmental recovery
RESOURCE_DEPLETION_RATE = 0.000003     # much weaker depletion
CELL_REGEN_TICK_RATE = 0.1


# ======================
# === AGENT SLEEP ======
# ======================

AGENT_SLEEP_THRESHOLD_STEPS = 5
AGENT_STABLE_GROWTH_RATE = 0.01


# =======================
# === SEAFARING LOGIC ===
# =======================

SEAFARING_TECH_THRESHOLD = 0.1
SEAFARING_FOOD_START = 100.0
SEAFARING_LAND_SENSE_RADIUS = 5
SEAFARING_SPAWN_CHANCE = 0.05   # lower to avoid too many early colonists


# =======================
# === AGGREGATION LOGIC =
# =======================

CITY_INFLUENCE_RADIUS = 5
STATE_INFLUENCE_RADIUS = 7


# ====================
# === STATE MACRO ====
# ====================

MACRO_FOOD_PRODUCTION_FACTOR = 0.05
MACRO_BIRTH_RATE = 0.01             # macro birth reduced → slower growth
MACRO_DEATH_RATE = 0.009


# ===========================
# === DIPLOMACY & WARFARE ===
# ===========================

DIPLOMACY_VERBOSITY = True

RELATION_DECAY = 0.4                # slower relation decay
WAR_THRESHOLD = -25.0               # hostile threshold
PEACE_THRESHOLD = 0.0
ALLIANCE_THRESHOLD = 60.0
BASE_AGGRESSION = 0.05

# battles
BATTLE_DAMAGE_RATE = 0.1
TERRITORY_STEAL_CHANCE = 0.3
WAR_EXHAUSTION_RATE = 0.02


# =========================
# === VASSALAGE LOGIC =====
# =========================

# loyalty behavior
VASSAL_LOYALTY_GAIN_PEACE = 0.01
VASSAL_WAR_LOYALTY_PENALTY = 0.02

# thresholds
VASSAL_LOYALTY_RECOVERY_SPEED = 0.03
VASSAL_FORCED_ANNEX_THRESHOLD = -30
VASSAL_REVOLT_THRESHOLD = 0.3

# tech flow
VASSAL_TECH_FLOW = 0.05
VASSAL_TRIBUTE_RATE = 0.02

# power ratio needed to create vassal
VASSALIZATION_POWER_RATIO = 2.0
VASSALIZATION_RELATION_FLOOR = -20


# ===================
# === END OF GAME ===
# ===================

POLAR_BIOMES = {
    "Snowy Tundra",
    "Snowy Mountains",
    "Snowy Taiga Mountains",
    "Deep Frozen Ocean"
}
