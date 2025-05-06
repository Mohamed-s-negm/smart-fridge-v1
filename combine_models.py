import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from ml_model import ML_Model
from VGG16_model import DL_ImgClass
import pandas as pd

path_1 = "./trained_models/rf_spoil_model.pkl"
path_2 = "./trained_models/rf_amount_model.pkl"
df = pd.read_csv("./datasets/smart_fridge_dataset_v1.csv")
image_path = "./apple-pie.jpeg"
dl_path = "./dl_trained_models/VGG16_3rd.pth"
data_path = Path("C:/Users/msen6/Documents/Github Projects/datasets/modified-food-101")

# Load models once
print("Loading DL model...")
dl_model = DL_ImgClass(model_path=dl_path, num_classes=50)
dl_model.prepare_data(data_path)
dl_model.load_model(dl_path)

print("Loading ML models...")
spoil_model = ML_Model(model_file=path_1)
amount_model = ML_Model(model_file=path_2)

def predict_food_name(image_path):
    return dl_model.predict(image_path)

def predict_condition_and_sufficiency(class_ind, temp, humidity, gas, days, weight):
    test_1 = [[class_ind, temp, humidity, gas, days]]
    test_2 = [[class_ind, weight]]
    condition = spoil_model.predict(test_1)[0]
    sufficiency = amount_model.predict(test_2)[0]
    return condition, sufficiency

food_name = predict_food_name(image_path)
food_to_index = df[['food_type', 'class_index']].drop_duplicates().set_index('food_type')['class_index'].to_dict()
class_ind = food_to_index.get(food_name)
temp = input("temp: ")
humidity = input("humidity: ")
gas = input("gas: ")
days = input("days: ")
weight = input("weight: ")

result = predict_condition_and_sufficiency(class_ind, temp, humidity, gas, days, weight)
print(f"{food_name}'s results are {result}")