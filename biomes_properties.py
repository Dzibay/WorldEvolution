from biomes_data import RAW_BIOME_HEX_MAP, hex_to_rgb


# ======================================================
# DEFAULTS — базовая "средняя" суша (не используется напрямую)
# ======================================================

_DEFAULT = dict(
    is_ocean=False,
    is_fresh_water=False,

    fresh_water=0.3,
    food_vegetal=0.3,
    food_animal=0.3,

    wood_yield=0.2,
    stone_yield=0.2,
    ore_yield=0.1,

    habitability=0.4,
    arable_land=0.3,
    movement_cost=2.0
)


# ======================================================
# РЕАЛИСТИЧНЫЕ ПАРАМЕТРЫ ВСЕХ БИОМОВ ЗЕМЛИ
# ======================================================

BIOME_PROPERTIES = {

    # --------------------------------------------------
    # 🌊 ОКЕАНЫ
    # --------------------------------------------------
    "Ocean": dict(
        is_ocean=True,
        fresh_water=0.0,
        food_vegetal=0.0,
        food_animal=0.4,
        wood_yield=0.0,
        stone_yield=0.0,
        ore_yield=0.0,
        habitability=0.0,
        arable_land=0.0,
        movement_cost=12.0
    ),
    "Warm Ocean": dict(
        is_ocean=True,
        food_animal=0.6,
        movement_cost=12.0
    ),
    "Lukewarm Ocean": dict(is_ocean=True, food_animal=0.5, movement_cost=12.0),
    "Cold Ocean": dict(is_ocean=True, food_animal=0.3, movement_cost=12.0),
    "Deep Warm Ocean": dict(is_ocean=True, food_animal=0.3),
    "Deep Lukewarm Ocean": dict(is_ocean=True, food_animal=0.2),
    "Deep Cold Ocean": dict(is_ocean=True, food_animal=0.1),
    "Deep Frozen Ocean": dict(is_ocean=True, food_animal=0.05, habitability=0.0),

    # --------------------------------------------------
    # 🌊 РЕКИ И ОАЗИСЫ
    # --------------------------------------------------
    "River": dict(
        is_fresh_water=True,
        fresh_water=1.0,
        food_vegetal=0.5,
        food_animal=0.6,
        wood_yield=0.1,
        stone_yield=0.1,
        habitability=0.8,
        arable_land=0.9,
        movement_cost=8.0
    ),
    "Desert Lakes": dict(
        is_fresh_water=True,
        fresh_water=1.0,
        food_vegetal=0.2,
        food_animal=0.4,
        habitability=0.6,
        arable_land=0.3,
        movement_cost=2.0
    ),

    # --------------------------------------------------
    # 🌾 РАВНИНЫ — выгоднейший биом Земли
    # --------------------------------------------------
    "Plains": dict(
        fresh_water=0.5,
        food_vegetal=0.7,
        food_animal=0.8,
        wood_yield=0.1,
        stone_yield=0.1,
        habitability=0.95,
        arable_land=1.0,
        movement_cost=1.0
    ),
    "Sunflower Plains": dict(
        fresh_water=0.6,
        food_vegetal=0.8,
        food_animal=0.8,
        habitability=1.0,
        arable_land=1.0,
        movement_cost=1.0
    ),

    # --------------------------------------------------
    # 🏝 ПРИБРЕЖНЫЕ ЗОНЫ
    # --------------------------------------------------
    "Beach": dict(
        fresh_water=0.1,
        food_vegetal=0.1,
        food_animal=0.6,
        habitability=0.5,
        arable_land=0.1,
        movement_cost=1.2
    ),
    "Snowy Beach": dict(
        fresh_water=0.05,
        food_vegetal=0.0,
        food_animal=0.3,
        habitability=0.2,
        arable_land=0.0,
        movement_cost=1.5
    ),

    # --------------------------------------------------
    # 🌳 ЛЕСА
    # --------------------------------------------------
    "Forest": dict(
        fresh_water=0.6,
        food_vegetal=0.7,
        food_animal=0.6,
        wood_yield=1.0,
        stone_yield=0.2,
        habitability=0.7,
        arable_land=0.4,
        movement_cost=3.5
    ),
    "Flower Forest": dict(
        fresh_water=0.7,
        food_vegetal=0.9,
        food_animal=0.6,
        wood_yield=1.0,
        habitability=0.8,
        arable_land=0.5,
        movement_cost=3.0
    ),
    "Dark Forest Hills": dict(
        fresh_water=0.5,
        food_vegetal=0.5,
        food_animal=0.4,
        wood_yield=1.0,
        stone_yield=0.6,
        ore_yield=0.4,
        habitability=0.45,
        arable_land=0.2,
        movement_cost=4.5
    ),

    # --------------------------------------------------
    # 🌴 ДЖУНГЛИ (низкая пригодность из-за паразитов, болезней)
    # --------------------------------------------------
    "Jungle": dict(
        fresh_water=0.8,
        food_vegetal=1.0,
        food_animal=0.5,
        wood_yield=1.0,
        habitability=0.35,
        arable_land=0.2,
        movement_cost=5.0
    ),
    "Jungle Hills": dict(
        fresh_water=0.7,
        food_vegetal=0.9,
        stone_yield=0.5,
        ore_yield=0.3,
        habitability=0.25,
        arable_land=0.1,
        movement_cost=6.0
    ),
    "Jungle Edge": dict(
        fresh_water=0.6,
        food_vegetal=0.8,
        habitability=0.6,
        arable_land=0.5,
        movement_cost=3.0
    ),
    "Modified Jungle Edge": dict(
        fresh_water=0.6,
        food_vegetal=0.8,
        stone_yield=0.3,
        habitability=0.55,
        arable_land=0.4,
        movement_cost=3.5
    ),

    # --------------------------------------------------
    # 🌾 САВАННА (историческая колыбель человечества)
    # --------------------------------------------------
    "Savanna": dict(
        fresh_water=0.3,
        food_vegetal=0.4,
        food_animal=1.0,
        wood_yield=0.2,
        habitability=0.8,
        arable_land=0.6,
        movement_cost=1.0
    ),
    "Savanna Plateau": dict(
        fresh_water=0.3,
        food_vegetal=0.4,
        stone_yield=0.5,
        ore_yield=0.3,
        habitability=0.6,
        arable_land=0.4,
        movement_cost=2.0
    ),

    # --------------------------------------------------
    # 🏜 ПУСТЫНИ
    # --------------------------------------------------
    "Desert": dict(
        fresh_water=0.0,
        food_vegetal=0.03,
        food_animal=0.05,
        wood_yield=0.0,
        stone_yield=0.3,
        ore_yield=0.3,
        habitability=0.03,
        arable_land=0.0,
        movement_cost=2.0
    ),
    "Desert Hills": dict(
        fresh_water=0.02,
        stone_yield=0.7,
        ore_yield=0.5,
        habitability=0.02,
        arable_land=0.0,
        movement_cost=3.0
    ),

    # --------------------------------------------------
    # 🌲 ТАЙГА (средняя продуктивность, плохие почвы)
    # --------------------------------------------------
    "Taiga": dict(
        fresh_water=0.6,
        food_vegetal=0.3,
        food_animal=0.4,
        wood_yield=0.9,
        habitability=0.35,
        arable_land=0.1,
        movement_cost=2.5
    ),
    "Taiga Hills": dict(
        fresh_water=0.5,
        stone_yield=0.7,
        habitability=0.25,
        arable_land=0.05,
        movement_cost=3.5
    ),
    "Giant Tree Taiga": dict(
        fresh_water=0.6,
        wood_yield=1.0,
        habitability=0.35,
        arable_land=0.1,
        movement_cost=3.0
    ),
    "Giant Tree Taiga Hills": dict(
        fresh_water=0.5,
        wood_yield=1.0,
        stone_yield=0.7,
        ore_yield=0.5,
        habitability=0.25,
        arable_land=0.05,
        movement_cost=4.0
    ),
    "Taiga Mountains": dict(
        fresh_water=0.4,
        food_vegetal=0.2,
        stone_yield=1.0,
        ore_yield=1.0,
        habitability=0.15,
        arable_land=0.02,
        movement_cost=5.5
    ),
    "Snowy Taiga Mountains": dict(
        fresh_water=0.3,
        food_vegetal=0.1,
        food_animal=0.2,
        stone_yield=1.0,
        ore_yield=1.0,
        habitability=0.1,
        arable_land=0.0,
        movement_cost=6.0
    ),

    # --------------------------------------------------
    # ❄ ТУНДРА
    # --------------------------------------------------
    "Snowy Tundra": dict(
        fresh_water=0.2,
        food_vegetal=0.05,
        food_animal=0.2,
        wood_yield=0.05,
        habitability=0.05,
        arable_land=0.0,
        movement_cost=3.0
    ),

    # --------------------------------------------------
    # 🏔 ГОРЫ
    # --------------------------------------------------
    "Mountains": dict(
        fresh_water=0.3,
        food_vegetal=0.05,
        food_animal=0.1,
        wood_yield=0.1,
        stone_yield=1.0,
        ore_yield=1.0,
        habitability=0.1,
        arable_land=0.0,
        movement_cost=6.0
    ),
    "Wooded Mountains": dict(
        fresh_water=0.4,
        food_vegetal=0.2,
        wood_yield=0.7,
        stone_yield=1.0,
        ore_yield=1.0,
        habitability=0.2,
        arable_land=0.05,
        movement_cost=5.0
    ),
    "Snowy Mountains": dict(
        fresh_water=0.3,
        food_vegetal=0.0,
        food_animal=0.1,
        wood_yield=0.1,
        stone_yield=1.0,
        ore_yield=0.8,
        habitability=0.05,
        arable_land=0.0,
        movement_cost=6.5
    ),

    # --------------------------------------------------
    # 🟫 БЭДЛЕНДЫ (эрозионные пустыни)
    # --------------------------------------------------
    "Badlands": dict(
        fresh_water=0.1,
        food_vegetal=0.1,
        food_animal=0.1,
        stone_yield=0.9,
        ore_yield=0.8,
        habitability=0.08,
        arable_land=0.0,
        movement_cost=2.5
    ),
    "Badlands Plateau": dict(
        fresh_water=0.1,
        stone_yield=1.0,
        ore_yield=0.9,
        habitability=0.07,
        movement_cost=3.0
    ),

    # --------------------------------------------------
    # 🌫 БОЛОТА
    # --------------------------------------------------
    "Swamp": dict(
        fresh_water=0.9,
        food_vegetal=0.5,
        food_animal=0.4,
        wood_yield=0.4,
        habitability=0.15,
        arable_land=0.05,
        movement_cost=6.0
    ),
}


# ======================================================
# АВТОГЕНЕРАЦИЯ ОКОНЧАТЕЛЬНОГО НАБОРА ДАННЫХ
# ======================================================

BIOME_DATA = {}
for name, hex_code in RAW_BIOME_HEX_MAP.items():

    p = _DEFAULT.copy()
    if name in BIOME_PROPERTIES:
        p.update(BIOME_PROPERTIES[name])
    else:
        print(f"[⚠] Биом {name} не описан, используется default.")

    p["vis_color"] = hex_to_rgb(hex_code)

    BIOME_DATA[name] = p

