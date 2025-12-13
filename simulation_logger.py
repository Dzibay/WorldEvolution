import json, gzip, os, sys
from statistics import mean

from simulation import Simulation, State, City, Tribe, HumanGroup, SeafaringGroup
from config import (
    CHECKPOINT_INTERVAL,
    END_YEAR,
    CARRYING_CAPACITY_FACTOR,
    MACRO_FOOD_PRODUCTION_FACTOR,
    MACRO_BIRTH_RATE,
    MACRO_DEATH_RATE,
    MACRO_TECH_FACTOR,
    SIMULATION_STEP_YEARS,
)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

SEGMENT_YEARS = 10_000  # сколько лет в одном gzip-файле


# === Вспомогательные функции ===


def compute_state_macro(state: State, sim: Simulation):
    """
    Максимально подробная макро-аналитика для государства:
    - средняя пригодность
    - средняя плодородность
    - производство еды
    - эффективная вместимость
    - коэффициент перенаселения
    """
    total_habitability = 0.0
    total_food_prod = 0.0
    total_arable = 0.0

    for (i, j) in state.territory:
        cell = sim.world.get((i, j))
        if not cell:
            continue
        total_habitability += cell.habitability
        total_food_prod += (cell.arable + cell.food_availability)
        total_arable += cell.arable

    n_cells = max(1, len(state.territory))
    avg_habitability = total_habitability / n_cells
    avg_arable = total_arable / n_cells

    # Вместимость (как в State.step)
    base_capacity = n_cells * avg_habitability * CARRYING_CAPACITY_FACTOR
    tech_capacity_multiplier = 1.0 + (state.tech * 4.0)
    effective_capacity = base_capacity * tech_capacity_multiplier

    # Производство еды и профицит
    food_production = (
        total_food_prod * (1 + state.tech) * MACRO_FOOD_PRODUCTION_FACTOR * n_cells
    )
    food_needed = state.population * state.need_food_per_capita
    food_surplus_ratio = (food_production - food_needed) / (food_needed + 1e-9)

    # Простой логистический рост (как оценка, state.step уже изменил population)
    growth_base = MACRO_BIRTH_RATE * (1 + avg_habitability * 0.5) * (
        1 + min(0.5, food_surplus_ratio)
    )
    death_base = MACRO_DEATH_RATE * (1 - avg_habitability * 0.5)
    overpop_penalty = max(
        0.0, (state.population / (effective_capacity + 1e-9)) - 1.0
    ) * 0.05
    yearly_growth = growth_base - death_base - overpop_penalty

    return {
        "cells": n_cells,
        "avg_habitability": round(avg_habitability, 4),
        "avg_arable": round(avg_arable, 4),
        "total_food_index": round(total_food_prod, 4),
        "effective_capacity": round(effective_capacity, 2),
        "food_production": round(food_production, 2),
        "food_needed": round(food_needed, 2),
        "food_surplus_ratio": round(food_surplus_ratio, 4),
        "yearly_growth_rate": round(yearly_growth, 5),
        "population_capacity_ratio": round(
            state.population / (effective_capacity + 1e-9), 4
        ),
    }


def compute_state_neighbors(states, sim: Simulation):
    """
    Строит карту дипломатии: для каждого государства рассчитывает соседей и длину общей границы.
    Возвращает dict[state_id] = {other_state_id: border_cells_count}
    """
    neighbors_map = {}
    cell_owner = {}

    # 1. Карта: клетка -> id государства
    for s in states:
        for (i, j) in s.territory:
            cell_owner[(i, j)] = s.id

    # 2. Для каждой клетки государства проверяем 4-х соседей
    for s in states:
        border_counts = {}
        for (i, j) in s.territory:
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb_coord = (i + di, j + dj)
                owner = cell_owner.get(nb_coord)
                if owner is None or owner == s.id:
                    continue
                border_counts[owner] = border_counts.get(owner, 0) + 1

        neighbors_map[s.id] = border_counts

    return neighbors_map


def serialize_entity(e, sim: Simulation = None, neighbors_map=None):
    """
    Максимально подробная сериализация сущности.
    Логируем всё, что может пригодиться viewer'у или анализу.
    """
    base = {
        "id": e.id,
        "stage": e.stage,
        "i": e.i,
        "j": e.j,
        "population": int(getattr(e, "population", 0)),
        "tech": round(float(getattr(e, "tech", 0.0)), 4),
        "alive": bool(getattr(e, "alive", True)),
        "age": int(getattr(e, "age", 0)),
    }

    # Общие параметры базовых агентов
    for attr, key, precision in [
        ("hunger_level", "hunger", 4),
        ("thirst_level", "thirst", 4),
        ("food", "food", 3),
        ("water", "water", 3),
        ("need_food_per_capita", "need_food_per_capita", 5),
        ("sleep_timer", "sleep_timer", None),
    ]:
        if hasattr(e, attr):
            val = getattr(e, attr)
            if precision is not None:
                val = round(float(val), precision)
            base[key] = val

    # Стадия-специфичные параметры
    if isinstance(e, HumanGroup):
        base["type"] = "human_group"
        base["is_migrating"] = bool(getattr(e, "is_migrating", False))
        base["steps_migrating"] = int(getattr(e, "steps_migrating", 0))
        if hasattr(e, "home_coord") and e.home_coord is not None:
            base["home_coord"] = list(e.home_coord)

    elif isinstance(e, Tribe):
        base["type"] = "tribe"

    elif isinstance(e, City):
        base["type"] = "city"
        base["influence_radius"] = int(getattr(e, "influence_radius", 0))

    elif isinstance(e, SeafaringGroup):
        base["type"] = "seafaring_group"
        base["steps_at_sea"] = int(getattr(e, "steps_at_sea", 0))

    elif isinstance(e, State):
        base["type"] = "state"
        base["at_war"] = list(getattr(e, "at_war", []))
        # Территория и города
        base["territory"] = [list(t) for t in getattr(e, "territory", set())]
        base["cities"] = [list(c) for c in getattr(e, "cities_coords", [])]
        base["is_coastal"] = bool(getattr(e, "is_coastal", False))
        base["expansion_budget"] = round(
            float(getattr(e, "expansion_budget", 0.0)), 3
        )

        # Макроэкономика
        if sim is not None:
            macro = compute_state_macro(e, sim)
            base["macro"] = macro

        # Дипломатия (соседи и общая граница)
        if neighbors_map is not None:
            nb = neighbors_map.get(e.id, {})
            # Превращаем в удобный список {id, border}
            neighbors_list = [
                {"id": int(sid), "border": int(border)}
                for sid, border in sorted(
                    nb.items(), key=lambda x: x[1], reverse=True
                )
            ]
            base["neighbors"] = neighbors_list

    return base


def summarize_world(entities):
    stages = {}
    pops = []
    techs = []

    for e in entities:
        stages[e.stage] = stages.get(e.stage, 0) + 1
        pops.append(getattr(e, "population", 0))
        techs.append(getattr(e, "tech", 0.0))

    total_pop = int(sum(pops)) if pops else 0
    avg_pop = round(total_pop / len(pops), 2) if pops else 0
    max_pop = max(pops) if pops else 0
    avg_tech = round(mean(techs), 4) if techs else 0.0

    return {
        "total_entities": len(entities),
        "total_population": total_pop,
        "avg_population": avg_pop,
        "max_population": max_pop,
        "avg_tech": avg_tech,
        "stages": stages,
    }


def run_and_log_simulation(debug=False):
    JSON_FILE = "world_cells.json"
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            cells = json.load(f)
    except FileNotFoundError:
        print(f"Ошибка: файл {JSON_FILE} не найден!")
        sys.exit(1)

    nx = max(c["i"] for c in cells) + 1
    ny = max(c["j"] for c in cells) + 1

    sim = Simulation(nx=nx, ny=ny)
    sim.initialize()

    segment_log = []
    segment_start_year = sim.year
    last_logged_year = sim.year
    total_snapshots = 0

    print(f"🌍 Старт симуляции: {segment_start_year} → {END_YEAR}")

    while sim.year < END_YEAR:
        entities, year = sim.step(debug=debug)
        if not sim.running:
            print("❌ Симуляция остановлена (все сущности вымерли).")
            break

        # Каждые CHECKPOINT_INTERVAL лет — лог
        if (year - last_logged_year) >= CHECKPOINT_INTERVAL:
            summary = summarize_world(entities)

            # Для расширенной дипломатии и макро-аналитики
            states = [e for e in entities if isinstance(e, State) and e.alive]
            neighbors_map = compute_state_neighbors(states, sim)

            snapshot = {
                "year": year,
                "summary": summary,
                "entities": [
                    serialize_entity(e, sim, neighbors_map)
                    for e in entities
                    if getattr(e, "alive", False)
                ],
            }

            segment_log.append(snapshot)
            total_snapshots += 1
            print(
                f"🧭 {year}: {summary['total_entities']} объектов, "
                f"население {summary['total_population']}, "
                f"гос-в {summary['stages'].get('state', 0)}"
            )
            last_logged_year = year

        # --- сохраняем сегмент каждые SEGMENT_YEARS ---
        if (year - segment_start_year) >= SEGMENT_YEARS or year >= END_YEAR:
            seg_filename = os.path.join(
                LOG_DIR, f"simulation_{segment_start_year}_{year}.json.gz"
            )
            with gzip.open(seg_filename, "wt", encoding="utf-8") as f:
                json.dump(
                    segment_log, f, ensure_ascii=False, separators=(",", ":")
                )
            print(
                f"💾 Сегмент {segment_start_year} → {year} сохранён "
                f"({len(segment_log)} кадров, gzip {round(os.path.getsize(seg_filename)/1024/1024,2)} MB)"
            )
            segment_log = []
            segment_start_year = year

    print(f"✅ Симуляция завершена ({total_snapshots} кадров)")
    return True


if __name__ == "__main__":
    run_and_log_simulation(debug=False)
