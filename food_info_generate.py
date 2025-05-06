import requests
import pandas as pd
import time
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import json
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# List of meals
meals = [
    'hummus', 'cheese_plate', 'apple-pie', 'chicken_wings', 'onion_rings', 'ramen',
    'cheesecake', 'waffles', 'bread_pudding', 'macaroni_and_cheese', 'risotto',
    'omelette', 'caesar_salad', 'baklava', 'dumplings', 'greek_salad', 'cup_cakes',
    'ice_cream', 'chocolate_cake', 'fried_calamari', 'grilled_salmon', 'nachos',
    'french_onion_soup', 'breakfast_burrito', 'beef_salad', 'filet_mignon',
    'strawberry_shortcake', 'chicken_curry', 'pizza', 'fish_and_chips',
    'creme_brulee', 'garlic_bread', 'donuts', 'pancakes', 'seaweed_salad',
    'falafel', 'spaghetti_carbonara', 'sushi', 'sashimi', 'hot_dog',
    'beef_carpaccio', 'ceviche', 'fried_rice', 'deviled_eggs', 'french_toast',
    'lobster_roll_sandwich', 'carrot_cake', 'hot_and_sour_soup', 'french_fries',
    'edamame'
]

# CalorieNinjas API configuration
CALORIE_NINJAS_API_KEY = "YxQqL9F5Fz7I+IK9EF5OTw==v6B2a9QGgwqkNc8w"  # Replace with your CalorieNinjas API key
CALORIE_NINJAS_API_URL = "https://api.calorieninjas.com/v1/nutrition"

def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def fetch_meal_data(meal_name, session):
    params = {
        'query': meal_name.replace('_', ' ')
    }
    
    headers = {
        'X-Api-Key': CALORIE_NINJAS_API_KEY,
        'Accept': 'application/json'
    }
    
    try:
        response = session.get(
            CALORIE_NINJAS_API_URL,
            params=params,
            headers=headers,
            timeout=(30, 30),
            verify=False
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get('items'):
            print(f"No data found for {meal_name}")
            return None
            
        item = data['items'][0]
        
        return {
            'meal': meal_name,
            'calories': item.get('calories'),
            'protein_g': item.get('protein_g'),
            'fat_g': item.get('fat_total_g'),
            'carbs_g': item.get('carbohydrates_total_g'),
            'sugar_g': item.get('sugar_g'),
            'fiber_g': item.get('fiber_g'),
            'sodium_mg': item.get('sodium_mg')
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {meal_name}: {str(e)}")
        time.sleep(5)
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON for {meal_name}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error for {meal_name}: {str(e)}")
        return None

def main():
    if CALORIE_NINJAS_API_KEY == "YOUR_API_KEY":
        print("Please set your CalorieNinjas API key in the script!")
        print("You can get a free API key by:")
        print("1. Going to https://calorieninjas.com/")
        print("2. Creating a free account")
        print("3. Copying your API key")
        return

    session = create_session()
    meal_data_list = []
    
    print("Starting to fetch meal data...")
    for i, meal in enumerate(meals, 1):
        print(f"Processing {i}/{len(meals)}: {meal}")
        data = fetch_meal_data(meal, session)
        if data:
            meal_data_list.append(data)
        time.sleep(3)  # Respect API rate limits
    
    if meal_data_list:
        df = pd.DataFrame(meal_data_list)
        df.to_csv('meal_nutrition_dataset.csv', index=False)
        print(f"Successfully saved data for {len(meal_data_list)} meals to 'meal_nutrition_dataset.csv'")
    else:
        print("No data was collected. Please check the error messages above.")

if __name__ == "__main__":
    main()
