import json
import os
from statistics import mean
from simulation import Simulation
from config import CHECKPOINT_INTERVAL, END_YEAR
import sys

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def serialize_entity(e):
    """Конвертирует объект сущности в компактный формат для JSON"""
    base = {
        "id": e.id,
        "stage": e.stage,
        "i": e.i,
        "j": e.j,
        "population": int(e.population),
        "tech": round(e.tech, 4),
        "alive": e.alive
    }

    # 🔹 Если это Государство — добавляем территорию
    if hasattr(e, "territory"):
        base["territory"] = list(map(list, e.territory))  # [[i,j], [i,j], ...]

    return base

def summarize_world(entities):
    """Создаёт сводку по состоянию мира"""
    stages = {}
    pops = []
    techs = []

    for e in entities:
        stages[e.stage] = stages.get(e.stage, 0) + 1
        pops.append(e.population)
        techs.append(e.tech)

    summary = {
        "total_entities": len(entities),
        "total_population": int(sum(pops)),
        "avg_population": round(sum(pops) / len(pops), 2) if pops else 0,
        "max_population": max(pops) if pops else 0,
        "avg_tech": round(mean(techs), 4) if techs else 0.0,
        "stages": stages
    }
    return summary

def run_and_log_simulation(steps=5000, debug=False):
    """Запускает симуляцию и сохраняет промежуточные состояния"""
    # === 1. Загрузка карты ячеек ===
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

    log = []  # Список состояний
    last_logged_year = sim.year

    for step in range(steps):
        entities, year = sim.step(debug=debug)

        if not sim.running:
            print("Симуляция завершена (все сущности вымерли).")
            break

        # Каждые CHECKPOINT_INTERVAL лет сохраняем снимок
        if (year - last_logged_year) >= CHECKPOINT_INTERVAL or year >= END_YEAR:
            summary = summarize_world(entities)
            snapshot = {
                "year": year,
                "summary": summary,
                "entities": [serialize_entity(e) for e in entities if e.alive]
            }
            log.append(snapshot)
            print(f"🧭 {year}: {summary['total_entities']} объектов, "
                  f"население {summary['total_population']}, "
                  f"гос-в {summary['stages'].get('state', 0)}")
            last_logged_year = year

        if year >= END_YEAR:
            print("Достигнут конец симуляции.")
            break

    # --- Сохраняем результат ---
    log_filename = os.path.join(LOG_DIR, f"simulation_log_{log[0]['year']}_{year}.json")
    with open(log_filename, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"✅ Лог сохранён: {log_filename} ({len(log)} записей)")

    last = log[-1]
    states = [e for e in last["entities"] if e["stage"] == "state"]
    print(f"Последний снимок: {len(states)} государств")
    for s in states[:3]:
        print("  ▶", s["id"], len(s.get("territory", [])), "клеток")

    return log_filename

if __name__ == "__main__":
    run_and_log_simulation(steps=1500, debug=False)
