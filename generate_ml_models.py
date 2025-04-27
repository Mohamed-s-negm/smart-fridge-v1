from ml_model import ML_Model
import os

path = os.path.join("datasets", "smart_fridge_realistic_dataset.csv")

x_1 = ['food_numeric', 'temperature_celsius', 'humidity_percent', 'gas_ppm', 'days_since_added']
y_1 = ['is_spoiled']

x_2 = ['weight_grams']
y_2 = ['amount_status']

spoil_model = ML_Model(data_path=path, x_cols=x_1, y_cols=y_1)
amount_model = ML_Model(data_path=path, x_cols=x_2, y_cols=y_2)

amount_model.random_forest()

test_1 = [[3, 5, 85, 258, 4]]
test_2 = [[291]]

print(spoil_model.predict(test_1))
print(amount_model.predict(test_2))