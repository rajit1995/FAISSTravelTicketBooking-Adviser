
import json
import random
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
NUM_FLIGHTS = 1_000_000          # Change to 1_000_000 if you really need it
RANDOM_SEED = 42
OUTPUT_FILE = "flights.json"
BASE_DATE   = datetime(2026, 6, 20)
# ─────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)

AIRLINES = ["IndiGo", "Air India", "SpiceJet", "Vistara", "GoAir"]
CITIES   = [
    "Kolkata", "Delhi", "Mumbai", "Chennai", "Bangalore",
    "Hyderabad", "Pune", "Jaipur", "Ahmedabad", "Lucknow",
]

flights = []
for i in range(NUM_FLIGHTS):
    airline = random.choice(AIRLINES)
    price   = random.randint(4_500, 7_000)          # stored as int, not string
    depart  = BASE_DATE + timedelta(hours=i % 24, days=i // 24)
    origin, destination = random.sample(CITIES, 2)

    flights.append({
        "flight_id":   i + 1,
        "airline":     airline,
        "origin":      origin,
        "destination": destination,
        "price":       price,                        # integer ₹ value
        "departure":   depart.strftime("%Y-%m-%d %H:%M"),
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(flights, f, indent=2, ensure_ascii=False)

print(f"✅ {OUTPUT_FILE} generated with {NUM_FLIGHTS:,} records.")
