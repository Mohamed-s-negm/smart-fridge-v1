import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 1) Food specs & numeric encoding
food_specs = {
    "Bread":    {"code":1, "temp_mu":20, "temp_sd":3, "hum_mu":65, "hum_sd":8,  "gas_mu":50,  "gas_sd":20},
    "Meat":     {"code":2, "temp_mu":4,  "temp_sd":1, "hum_mu":75, "hum_sd":5,  "gas_mu":200, "gas_sd":80},
    "Chicken":  {"code":3, "temp_mu":5,  "temp_sd":1, "hum_mu":80, "hum_sd":5,  "gas_mu":180, "gas_sd":70},
    "Fish":     {"code":4, "temp_mu":2,  "temp_sd":1, "hum_mu":85, "hum_sd":4,  "gas_mu":220, "gas_sd":90},
    "Fruit":    {"code":5, "temp_mu":15, "temp_sd":4, "hum_mu":75, "hum_sd":6,  "gas_mu":30,  "gas_sd":15},
    "Vegetable":{"code":6, "temp_mu":10, "temp_sd":3, "hum_mu":85, "hum_sd":5,  "gas_mu":40,  "gas_sd":20},
    "Cheese":   {"code":7, "temp_mu":8,  "temp_sd":2, "hum_mu":80, "hum_sd":5,  "gas_mu":60,  "gas_sd":20},
    "Eggs":     {"code":8, "temp_mu":5,  "temp_sd":1.5,"hum_mu":90, "hum_sd":3,  "gas_mu":20,  "gas_sd":10},
}

# 2) Targets for 150k samples
TOTAL = 150_000
targets = {
    "Good":         int(TOTAL * 0.33),
    "Near Spoiled": int(TOTAL * 0.33),
    "Spoiled":      TOTAL - int(TOTAL*0.33)*2
}

# 3) Helper to simulate one row for a given label
def simulate_row(label):
    food = random.choice(list(food_specs))
    spec = food_specs[food]
    # timestamp in last 30 days
    ts = datetime.now() - timedelta(days=random.randint(0,30),
                                    seconds=random.randint(0,86400))
    # base draws
    temp   = np.random.normal(spec["temp_mu"], spec["temp_sd"])
    hum    = np.random.normal(spec["hum_mu"], spec["hum_sd"])
    gas    = np.random.normal(spec["gas_mu"], spec["gas_sd"])
    days   = random.randint(0,10)
    weight = np.random.normal(400, 200)
    # push features toward the desired label
    if label == "Good":
        temp  -= abs(np.random.uniform(0, spec["temp_sd"]))
        hum   -= abs(np.random.uniform(0, spec["hum_sd"]))
        gas   -= abs(np.random.uniform(0, spec["gas_sd"]))
        days  = random.randint(0,4)
    elif label == "Near Spoiled":
        temp  += np.random.uniform(0, spec["temp_sd"])
        hum   += np.random.uniform(0, spec["hum_sd"])
        gas   += np.random.uniform(0, spec["gas_sd"])
        days   = random.randint(3,7)
    else:  # Spoiled
        temp  += abs(np.random.uniform(spec["temp_sd"], 2*spec["temp_sd"]))
        hum   += abs(np.random.uniform(spec["hum_sd"], 2*spec["hum_sd"]))
        gas   += abs(np.random.uniform(spec["gas_sd"], 2*spec["gas_sd"]))
        days   = random.randint(6,15)

    amount = "Enough" if weight >= 300 else "Not Enough"
    return {
        "timestamp":           ts.strftime("%Y-%m-%d %H:%M:%S"),
        "food_name":           food,
        "food_numeric":        spec["code"],
        "temperature_celsius": round(temp, 2),
        "humidity_percent":    round(hum, 2),
        "gas_ppm":             round(gas, 2),
        "days_since_added":    days,
        "weight_grams":        round(weight, 2),
        "is_spoiled":          label,
        "amount_status":       amount
    }

# 4) Generate balanced dataset
data = []
counts = {"Good":0, "Near Spoiled":0, "Spoiled":0}
while sum(counts.values()) < TOTAL:
    for label in ["Good","Near Spoiled","Spoiled"]:
        if counts[label] < targets[label]:
            data.append(simulate_row(label))
            counts[label] += 1

# 5) Shuffle, DataFrame & CSV
random.shuffle(data)
df = pd.DataFrame(data)
df.to_csv("smart_fridge_realistic_dataset.csv", index=False)

print("Done! Distribution:", df["is_spoiled"].value_counts())
