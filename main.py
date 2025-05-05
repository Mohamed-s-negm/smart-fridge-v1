from ml_model import ML_Model
from VGG16_model import DL_ImgClass
from resNet50_model import ResNet50
from pathlib import Path
import numpy as np
import pandas as pd

try:
    path_1 = "./trained_models/rf_spoil_model.pkl"
    path_2 = "./trained_models/rf_amount_model.pkl"
    df = pd.read_csv("./datasets/smart_fridge_dataset_v1.csv")
    dl_path = "./dl_trained_models/VGG16_3rd.pth"
    image_path = "./apple-pie.jpeg"
    path = Path("C:/Users/msen6/Documents/Github Projects/datasets/modified-food-101")
    

    spoil_model = ML_Model(model_file=path_1)
    amount_model = ML_Model(model_file=path_2)
    
    # Load the models explicitly
    spoil_model.load_model(path_1)
    amount_model.load_model(path_2)

    print("\nLoading VGG16 model...")
    dl_model = DL_ImgClass(model_path=dl_path, num_classes=50)
    dl_model.prepare_data(path)
    dl_model.load_model(dl_path)


    print("\nPredicting with DL model...")
    prediction_vgg = dl_model.predict(image_path)
    print(prediction_vgg)


    temp = input("temp: ")
    humi = input("humidity: ")
    days_since = input("Days since added: ")
    gas = input("gas level: ")
    weight = input("Weight: ")

    food_to_index = df[['food_type', 'class_index']].drop_duplicates().set_index('food_type')['class_index'].to_dict()

    class_ind = food_to_index.get(prediction_vgg)

    test_1 = [[class_ind, temp, humi, gas, days_since]]
    test_2 = [[class_ind, weight]]

    print("Predicting with spoil model...")
    print(spoil_model.predict(test_1))
    print("\nPredicting with amount model...")
    print(amount_model.predict(test_2))

    

except Exception as e:
    print(f"\nError occurred: {str(e)}")
