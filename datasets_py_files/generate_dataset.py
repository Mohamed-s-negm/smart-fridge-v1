import numpy as np
import pandas as pd

# 1) Define your 50 food classes
food_types = [
    'apple-pie', 'baklava', 'beef_carpaccio', 'beef_salad', 'bread_pudding',
    'breakfast_burrito', 'caesar_salad', 'carrot_cake', 'ceviche', 'cheese_plate',
    'cheesecake', 'chicken_curry', 'chicken_wings', 'chocolate_cake',
    'creme_brulee', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings',
    'edamame', 'falafel', 'filet_mignon', 'fish_and_chips', 'french_fries',
    'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
    'garlic_bread', 'greek_salad', 'grilled_salmon', 'hot_and_sour_soup',
    'hot_dog', 'hummus', 'ice_cream', 'lobster_roll_sandwich', 'macaroni_and_cheese',
    'nachos', 'omelette', 'onion_rings', 'pancakes', 'pizza', 'ramen', 'risotto',
    'sashimi', 'seaweed_salad', 'spaghetti_carbonara', 'strawberry_shortcake',
    'sushi', 'waffles'
]

# Build mapping from food_type string → integer class index
class_to_idx = {food: idx for idx, food in enumerate(food_types)}

n_per_class = 1000
rows = []
idx = 1

for food in food_types:
    for _ in range(n_per_class):
        # weight in grams
        weight = np.random.normal(loc=350, scale=100)
        weight = float(np.clip(weight, 50, 800))

        # temperature (°C)
        if food in ('ice_cream','creme_brulee'):
            temp = np.random.normal(2, 1)
        elif any(x in food for x in ('sushi','ceviche','sashimi','fish')):
            temp = np.random.normal(4, 1)
        else:
            temp = np.random.normal(5, 2)
        temp = round(float(np.clip(temp, 0, 10)), 1)

        # humidity (%)
        if any(x in food for x in ('bread','pizza','baklava')):
            humidity = np.random.normal(40, 10)
        else:
            humidity = np.random.normal(60, 15)
        humidity = round(float(np.clip(humidity, 20, 100)), 1)

        # gas_ppm
        if any(x in food for x in ('apple','carrot','salad','edamame','dumplings')):
            gas = np.random.normal(20, 5)
        else:
            gas = np.random.normal(5, 2)
        gas = round(max(0.0, gas), 1)

        # days since added
        if any(x in food for x in ('salad','sushi','ceviche')):
            days = np.random.randint(0, 4)
        else:
            days = np.random.randint(0, 8)

        # determine spoilage condition
        if temp > 8 or days > 6 or gas > 25:
            condition = 'spoiled'
        elif temp > 6 or days > 4 or gas > 15:
            condition = 'near_spoiled'
        else:
            condition = 'fine'

        # sufficient?
        sufficient = 'sufficient' if weight > 300 else 'not_sufficient'

        rows.append({
            'index': idx,
            'food_type': food,
            'class_index': class_to_idx[food],
            'weight_g': round(weight, 1),
            'temperature_C': temp,
            'humidity_%': humidity,
            'gas_ppm': gas,
            'days_since_added': days,
            'condition': condition,
            'sufficient': sufficient
        })
        idx += 1

# Create DataFrame, shuffle, and save
df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('smart_fridge_dataset_v1.csv', index=False)
print(f"Generated {len(df)} rows with class_index 0–{len(food_types)-1} and saved to smart_fridge_dataset.csv")
