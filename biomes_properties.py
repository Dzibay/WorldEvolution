# Файл: biome_properties.py
# (Содержит характеристики биомов для симуляции)

# 1. Импортируем оригинальные цвета из твоего файла
try:
    from biomes_data import RAW_BIOME_HEX_MAP, hex_to_rgb
except ImportError:
    print("Ошибка: не найден файл biomes_data.py!")
    print("Убедись, что 'biomes_data.py' находится в той же папке.")
    exit()

# 2. Определяем СВОЙСТВА ПО УМОЛЧАНИЮ для "стандартного" сухопутного биома
# Это помогает не повторять одно и то же для каждого биома
_DEFAULT_PROPERTIES = {
    # --- Основные флаги ---
    "is_ocean": False,        # Это океан? (блокирует наземное)
    "is_fresh_water": False,  # Это река/озеро? (источник воды)
    
    # --- Ресурсы (0.0 - 1.0) ---
    "fresh_water": 0.4,       # Доступ к ПРЕСНОЙ воде (дождь, реки, озера)
    "food_vegetal": 0.4,      # Еда (собирательство: фрукты, коренья)
    "food_animal": 0.4,       # Еда (охота: животные, дичь)
    "wood_yield": 0.2,        # Древесина (топливо, строительство)
    "stone_yield": 0.2,       # Камень (инструменты, строительство)
    "ore_yield": 0.1,         # Металлы (железо, медь и т.д.)
    
    # --- Параметры для цивилизации ---
    "habitability": 0.5,      # Общая пригодность для жизни (климат, безопасность)
    "arable_land": 0.5,       # Пригодность земли для сельского хозяйства
    "movement_cost": 2.0      # "Стоимость" передвижения. 1.0 = легко (равнины)
}

# 3. Определяем УНИКАЛЬНЫЕ СВОЙСТВА для каждого биома
# Здесь мы "перезаписываем" значения по умолчанию
_BIOME_PROPERTIES = {
    # --- Вода ---
    "Ocean": {"is_ocean": True, "fresh_water": 0.0, "food_vegetal": 0.0, "food_animal": 0.5, "wood_yield": 0.0, "stone_yield": 0.0, "ore_yield": 0.0, "habitability": 0.0, "arable_land": 0.0, "movement_cost": 10.0},
    "Warm Ocean": {"is_ocean": True, "fresh_water": 0.0, "food_vegetal": 0.0, "food_animal": 0.6, "wood_yield": 0.0, "stone_yield": 0.0, "ore_yield": 0.0, "habitability": 0.0, "arable_land": 0.0, "movement_cost": 10.0},
    "Lukewarm Ocean": {"is_ocean": True, "fresh_water": 0.0, "food_vegetal": 0.0, "food_animal": 0.5, "wood_yield": 0.0, "stone_yield": 0.0, "ore_yield": 0.0, "habitability": 0.0, "arable_land": 0.0, "movement_cost": 10.0},
    "Cold Ocean": {"is_ocean": True, "fresh_water": 0.0, "food_vegetal": 0.0, "food_animal": 0.4, "wood_yield": 0.0, "stone_yield": 0.0, "ore_yield": 0.0, "habitability": 0.0, "arable_land": 0.0, "movement_cost": 10.0},
    "Deep Warm Ocean": {"is_ocean": True, "fresh_water": 0.0, "food_vegetal": 0.0, "food_animal": 0.3, "wood_yield": 0.0, "stone_yield": 0.0, "ore_yield": 0.0, "habitability": 0.0, "arable_land": 0.0, "movement_cost": 10.0},
    "Deep Lukewarm Ocean": {"is_ocean": True, "fresh_water": 0.0, "food_vegetal": 0.0, "food_animal": 0.2, "wood_yield": 0.0, "stone_yield": 0.0, "ore_yield": 0.0, "habitability": 0.0, "arable_land": 0.0, "movement_cost": 10.0},
    "Deep Cold Ocean": {"is_ocean": True, "fresh_water": 0.0, "food_vegetal": 0.0, "food_animal": 0.1, "wood_yield": 0.0, "stone_yield": 0.0, "ore_yield": 0.0, "habitability": 0.0, "arable_land": 0.0, "movement_cost": 10.0},
    "Deep Frozen Ocean": {"is_ocean": True, "fresh_water": 0.0, "food_vegetal": 0.0, "food_animal": 0.1, "wood_yield": 0.0, "stone_yield": 0.0, "ore_yield": 0.0, "habitability": 0.0, "arable_land": 0.0, "movement_cost": 10.0},

    "River": {"is_fresh_water": True, "fresh_water": 1.0, "food_vegetal": 0.1, "food_animal": 0.6, "wood_yield": 0.0, "stone_yield": 0.1, "habitability": 0.1, "arable_land": 0.0, "movement_cost": 8.0},
    "Desert Lakes": {"is_fresh_water": True, "fresh_water": 1.0, "food_vegetal": 0.2, "food_animal": 0.3, "wood_yield": 0.0, "habitability": 0.3, "arable_land": 0.1, "movement_cost": 1.5},

    # --- Суша: Стандартные ---
    "Plains": {"fresh_water": 0.5, "food_vegetal": 0.6, "food_animal": 0.8, "wood_yield": 0.1, "stone_yield": 0.1, "habitability": 0.9, "arable_land": 1.0, "movement_cost": 1.0},
    "Sunflower Plains": {"fresh_water": 0.6, "food_vegetal": 0.7, "food_animal": 0.8, "wood_yield": 0.1, "stone_yield": 0.1, "habitability": 1.0, "arable_land": 1.0, "movement_cost": 1.0},
    "Beach": {"fresh_water": 0.1, "food_vegetal": 0.1, "food_animal": 0.6, "wood_yield": 0.05, "stone_yield": 0.1, "habitability": 0.5, "arable_land": 0.1, "movement_cost": 1.2},
    "Snowy Beach": {"fresh_water": 0.1, "food_vegetal": 0.0, "food_animal": 0.4, "wood_yield": 0.0, "stone_yield": 0.1, "habitability": 0.2, "arable_land": 0.0, "movement_cost": 1.5},

    # --- Суша: Леса ---
    "Forest": {"fresh_water": 0.6, "food_vegetal": 0.7, "food_animal": 0.6, "wood_yield": 1.0, "stone_yield": 0.2, "habitability": 0.7, "arable_land": 0.3, "movement_cost": 3.0},
    "Flower Forest": {"fresh_water": 0.7, "food_vegetal": 0.9, "food_animal": 0.6, "wood_yield": 0.9, "stone_yield": 0.2, "habitability": 0.8, "arable_land": 0.4, "movement_cost": 2.5},
    "Dark Forest Hills": {"fresh_water": 0.6, "food_vegetal": 0.5, "food_animal": 0.4, "wood_yield": 1.0, "stone_yield": 0.6, "ore_yield": 0.4, "habitability": 0.4, "arable_land": 0.1, "movement_cost": 4.0},

    # --- Суша: Джунгли ---
    "Jungle": {"fresh_water": 0.8, "food_vegetal": 1.0, "food_animal": 0.5, "wood_yield": 1.0, "stone_yield": 0.1, "habitability": 0.4, "arable_land": 0.2, "movement_cost": 4.5},
    "Jungle Hills": {"fresh_water": 0.8, "food_vegetal": 1.0, "food_animal": 0.5, "wood_yield": 1.0, "stone_yield": 0.5, "ore_yield": 0.4, "habitability": 0.3, "arable_land": 0.1, "movement_cost": 5.0},
    "Jungle Edge": {"fresh_water": 0.7, "food_vegetal": 0.8, "food_animal": 0.6, "wood_yield": 0.7, "stone_yield": 0.2, "habitability": 0.6, "arable_land": 0.5, "movement_cost": 2.5},
    "Modified Jungle Edge": {"fresh_water": 0.7, "food_vegetal": 0.8, "food_animal": 0.6, "wood_yield": 0.7, "stone_yield": 0.3, "ore_yield": 0.2, "habitability": 0.6, "arable_land": 0.4, "movement_cost": 3.0},

    # --- Суша: Саванна ---
    "Savanna": {"fresh_water": 0.3, "food_vegetal": 0.4, "food_animal": 1.0, "wood_yield": 0.2, "stone_yield": 0.1, "habitability": 0.7, "arable_land": 0.6, "movement_cost": 1.0},
    "Savanna Plateau": {"fresh_water": 0.3, "food_vegetal": 0.4, "food_animal": 1.0, "wood_yield": 0.2, "stone_yield": 0.5, "ore_yield": 0.3, "habitability": 0.6, "arable_land": 0.4, "movement_cost": 2.0},

    # --- Суша: Пустыня ---
    "Desert": {"fresh_water": 0.0, "food_vegetal": 0.05, "food_animal": 0.05, "wood_yield": 0.0, "stone_yield": 0.3, "ore_yield": 0.3, "habitability": 0.05, "arable_land": 0.0, "movement_cost": 1.5},
    "Desert Hills": {"fresh_water": 0.05, "food_vegetal": 0.05, "food_animal": 0.05, "wood_yield": 0.0, "stone_yield": 0.8, "ore_yield": 0.6, "habitability": 0.05, "arable_land": 0.0, "movement_cost": 2.5},

    # --- Суша: Холодные ---
    "Taiga": {"fresh_water": 0.5, "food_vegetal": 0.3, "food_animal": 0.5, "wood_yield": 0.9, "stone_yield": 0.2, "ore_yield": 0.2, "habitability": 0.3, "arable_land": 0.1, "movement_cost": 2.5},
    "Taiga Hills": {"fresh_water": 0.5, "food_vegetal": 0.3, "food_animal": 0.5, "wood_yield": 0.9, "stone_yield": 0.7, "ore_yield": 0.5, "habitability": 0.2, "arable_land": 0.05, "movement_cost": 3.5},
    "Giant Tree Taiga": {"fresh_water": 0.5, "food_vegetal": 0.4, "food_animal": 0.5, "wood_yield": 1.0, "stone_yield": 0.2, "habitability": 0.3, "arable_land": 0.1, "movement_cost": 3.0},
    "Giant Tree Taiga Hills": {"fresh_water": 0.5, "food_vegetal": 0.4, "food_animal": 0.5, "wood_yield": 1.0, "stone_yield": 0.7, "ore_yield": 0.5, "habitability": 0.2, "arable_land": 0.05, "movement_cost": 4.0},
    "Snowy Tundra": {"fresh_water": 0.2, "food_vegetal": 0.1, "food_animal": 0.3, "wood_yield": 0.05, "stone_yield": 0.1, "habitability": 0.1, "arable_land": 0.0, "movement_cost": 1.5},

    # --- Суша: Горы ---
    "Mountains": {"fresh_water": 0.4, "food_vegetal": 0.1, "food_animal": 0.2, "wood_yield": 0.2, "stone_yield": 1.0, "ore_yield": 1.0, "habitability": 0.1, "arable_land": 0.01, "movement_cost": 5.0},
    "Wooded Mountains": {"fresh_water": 0.5, "food_vegetal": 0.3, "food_animal": 0.4, "wood_yield": 0.8, "stone_yield": 1.0, "ore_yield": 1.0, "habitability": 0.2, "arable_land": 0.01, "movement_cost": 5.0},
    "Snowy Mountains": {"fresh_water": 0.3, "food_vegetal": 0.0, "food_animal": 0.1, "wood_yield": 0.1, "stone_yield": 1.0, "ore_yield": 0.9, "habitability": 0.05, "arable_land": 0.0, "movement_cost": 6.0},
    "Taiga Mountains": {"fresh_water": 0.4, "food_vegetal": 0.2, "food_animal": 0.3, "wood_yield": 0.7, "stone_yield": 1.0, "ore_yield": 1.0, "habitability": 0.1, "arable_land": 0.01, "movement_cost": 5.5},
    "Snowy Taiga Mountains": {"fresh_water": 0.3, "food_vegetal": 0.1, "food_animal": 0.2, "wood_yield": 0.5, "stone_yield": 1.0, "ore_yield": 1.0, "habitability": 0.05, "arable_land": 0.0, "movement_cost": 6.0},

    # --- Суша: Особые ---
    "Swamp": {"fresh_water": 0.9, "food_vegetal": 0.5, "food_animal": 0.4, "wood_yield": 0.4, "stone_yield": 0.0, "ore_yield": 0.0, "habitability": 0.1, "arable_land": 0.0, "movement_cost": 5.0},
    "Badlands": {"fresh_water": 0.1, "food_vegetal": 0.1, "food_animal": 0.1, "wood_yield": 0.1, "stone_yield": 0.9, "ore_yield": 0.8, "habitability": 0.1, "arable_land": 0.0, "movement_cost": 2.0},
    "Badlands Plateau": {"fresh_water": 0.1, "food_vegetal": 0.1, "food_animal": 0.1, "wood_yield": 0.1, "stone_yield": 1.0, "ore_yield": 0.9, "habitability": 0.1, "arable_land": 0.0, "movement_cost": 2.5},
}

# 4. Генерируем ИТОГОВЫЙ словарь BIOME_DATA
print("Загрузка данных о биомах...")
BIOME_DATA = {}
for name, hex_code in RAW_BIOME_HEX_MAP.items():
    
    # 1. Начинаем с копии "умолчаний"
    properties = _DEFAULT_PROPERTIES.copy()
    
    # 2. Получаем уникальные свойства для этого биома
    overrides = _BIOME_PROPERTIES.get(name)
    
    if overrides:
        # 3. "Перезаписываем" значения по умолчанию
        properties.update(overrides)
    else:
        # Эта строчка на случай, если ты добавишь биом в biomes_data,
        # но забудешь добавить его в _BIOME_PROPERTIES
        print(f"ВНИМАНИЕ: Биом '{name}' не найден в _BIOME_PROPERTIES. Использую значения по умолчанию.")

    # 4. Добавляем цвет (для визуализации)
    properties["vis_color"] = hex_to_rgb(hex_code)
    
    # 5. Сохраняем
    BIOME_DATA[name] = properties

print(f"Готово. {len(BIOME_DATA)} биомов загружено с полными характеристиками.")

# Это позволяет запускать файл напрямую, чтобы проверить данные
if __name__ == "__main__":
    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    
    print("\n--- Пример данных для 'Plains' ---")
    pp.pprint(BIOME_DATA["Plains"])
    
    print("\n--- Пример данных для 'Desert' ---")
    pp.pprint(BIOME_DATA["Desert"])
    
    print("\n--- Пример данных для 'Snowy Mountains' ---")
    pp.pprint(BIOME_DATA["Snowy Mountains"])