import json
import random
from datetime import datetime, timedelta

airlines = ["IndiGo", "Air India", "SpiceJet", "Vistara", "GoAir"]
cities = ["Kolkata", "Delhi", "Mumbai", "Chennai", "Bangalore", "Hyderabad", "Pune", "Jaipur", "Ahmedabad", "Lucknow"]

base_date = datetime(2026, 6, 20)

flights = []
for i in range(10000):  # ⚠️ reduced for testing; 10M is huge
    airline = random.choice(airlines)
    price_value = random.randint(4500, 7000)
    price = f"₹{price_value}"  # ✅ store as string with ₹
    departure_time = base_date + timedelta(hours=i % 24, days=i // 24)

    # Ensure origin and destination are different
    origin, destination = random.sample(cities, 2)

    flights.append({
        "airline": airline,
        "origin": origin,
        "destination": destination,
        "price": price,  # ✅ now includes ₹
        "departure": departure_time.strftime("%Y-%m-%d %H:%M")
    })

with open("flights.json", "w") as f:
    json.dump(flights, f, indent=2)

with open("flights.json", "r") as f:
    data = json.load(f)

print(f"Number of entries: {len(data)}")
# Since price is now a string, we can’t compute min/max directly
# but we can strip the ₹ and convert back to int if needed:
prices = [int(f["price"].replace("₹", "")) for f in data]
print(f"Min price: ₹{min(prices)}")
print(f"Max price: ₹{max(prices)}")
