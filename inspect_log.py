import gzip, json

with gzip.open("logs/simulation_-100000_-90000.json.gz", "rt", encoding="utf-8") as f:
    data = json.load(f)

e = data[0]["entities"][0]
print(e)
print("Keys:", list(e.keys()))
