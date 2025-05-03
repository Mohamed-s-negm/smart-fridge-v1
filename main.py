from ml_model import ML_Model
from dl_model import DL_ImgClass
from pathlib import Path

try:
    path_1 = "./trained_models/trained_spoil_rf_model.pkl"
    path_2 = "./trained_models/trained_amount_rf_model.pkl"
    dl_path = "./dl_trained_models/VGG16_3rd.pth"
    image_path = "./cw.webp"
    path = Path("C:/Users/msen6/Documents/Github Projects/datasets/modified-food-101")

    spoil_model = ML_Model(model_file=path_1)
    amount_model = ML_Model(model_file=path_2)

    test_1 = [[3, 5, 85, 258, 4]]
    test_2 = [[291]]

    print("Predicting with spoil model...")
    print(spoil_model.predict(test_1))
    print("\nPredicting with amount model...")
    print(amount_model.predict(test_2))

    print("\nLoading DL model...")
    dl_model = DL_ImgClass(model_path=dl_path, num_classes=50)
    dl_model.prepare_data(path)
    dl_model.load_model(dl_path)

    print("\nPredicting with DL model...")
    prediction = dl_model.predict(image_path)
    print(prediction)

except Exception as e:
    print(f"\nError occurred: {str(e)}")
