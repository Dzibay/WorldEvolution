import json, gzip, os, sys
from statistics import mean
from simulation import Simulation
from config import CHECKPOINT_INTERVAL, END_YEAR

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

SEGMENT_YEARS = 10_000  # сколько лет в одном gzip-файле

# === Вспомогательные функции ===

def serialize_entity(e):
    base = {
        "id": e.id,
        "stage": e.stage,
        "i": e.i,
        "j": e.j,
        "population": int(e.population),
        "tech": round(e.tech, 4),
        "alive": e.alive
    }
    if hasattr(e, "territory"):
        base["territory"] = list(map(list, e.territory))
    return base


def summarize_world(entities):
    stages = {}
    pops = []
    techs = []

    for e in entities:
        stages[e.stage] = stages.get(e.stage, 0) + 1
        pops.append(e.population)
        techs.append(e.tech)

    return {
        "total_entities": len(entities),
        "total_population": int(sum(pops)),
        "avg_population": round(sum(pops) / len(pops), 2) if pops else 0,
        "max_population": max(pops) if pops else 0,
        "avg_tech": round(mean(techs), 4) if techs else 0.0,
        "stages": stages
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
            snapshot = {
                "year": year,
                "summary": summary,
                "entities": [serialize_entity(e) for e in entities if e.alive]
            }
            segment_log.append(snapshot)
            total_snapshots += 1
            print(f"🧭 {year}: {summary['total_entities']} объектов, "
                  f"население {summary['total_population']}, "
                  f"гос-в {summary['stages'].get('state', 0)}")
            last_logged_year = year

        # --- сохраняем сегмент каждые SEGMENT_YEARS ---
        if (year - segment_start_year) >= SEGMENT_YEARS or year >= END_YEAR:
            seg_filename = os.path.join(LOG_DIR, f"simulation_{segment_start_year}_{year}.json.gz")
            with gzip.open(seg_filename, "wt", encoding="utf-8") as f:
                json.dump(segment_log, f, ensure_ascii=False, separators=(",", ":"))
            print(f"💾 Сегмент {segment_start_year} → {year} сохранён "
                  f"({len(segment_log)} кадров, gzip {round(os.path.getsize(seg_filename)/1024/1024,2)} MB)")
            segment_log = []
            segment_start_year = year

    print(f"✅ Симуляция завершена ({total_snapshots} кадров)")
    return True


if __name__ == "__main__":
    run_and_log_simulation(debug=False)
