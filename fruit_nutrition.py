import requests
import pandas as pd

# Open Food Facts API endpoint
OFF_ENDPOINT = "https://world.openfoodfacts.org/cgi/search.pl"

def get_fruit_nutrition(fruit_name):
    """
    Get nutrition information for a specific fruit from Open Food Facts API
    """
    try:
        # Prepare the query parameters
        params = {
            "search_terms": fruit_name,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": 1
        }
        
        # Make API request
        response = requests.get(OFF_ENDPOINT, params=params)
        response.raise_for_status()
        
        # Extract nutrition data
        data = response.json()
        if 'products' in data and len(data['products']) > 0:
            product = data['products'][0]
            nutriments = product.get('nutriments', {})
            
            # Extract relevant nutrition information
            nutrition_info = {
                'meal': fruit_name.lower().replace(' ', '_'),
                'calories': nutriments.get('energy-kcal_100g', 0),
                'protein_g': nutriments.get('proteins_100g', 0),
                'fat_g': nutriments.get('fat_100g', 0),
                'carbs_g': nutriments.get('carbohydrates_100g', 0),
                'sugar_g': nutriments.get('sugars_100g', 0),
                'fiber_g': nutriments.get('fiber_100g', 0),
                'sodium_mg': nutriments.get('sodium_100g', 0) * 1000  # Convert to mg
            }
            
            return nutrition_info
        else:
            print(f"No nutrition data found for {fruit_name}")
            return None
            
    except Exception as e:
        print(f"Error fetching nutrition data for {fruit_name}: {str(e)}")
        return None

def generate_fruit_nutrition_dataset():
    """
    Generate a dataset with nutrition information for apples, oranges, and bananas
    """
    fruits = ['apple', 'orange', 'banana']
    nutrition_data = []
    
    for fruit in fruits:
        nutrition_info = get_fruit_nutrition(fruit)
        if nutrition_info:
            nutrition_data.append(nutrition_info)
    
    # Create DataFrame
    df = pd.DataFrame(nutrition_data)
    
    # Save to CSV
    output_file = "datasets/fruit_nutrition_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"Nutrition dataset saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Generate the dataset
    df = generate_fruit_nutrition_dataset()
    print("\nGenerated nutrition data:")
    print(df) 