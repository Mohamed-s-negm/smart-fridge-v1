from ml_model import ML_Model
from VGG16_model import DL_ImgClass
from resNet50_model import ResNet50
from pathlib import Path

try:
    path_1 = "./trained_models/trained_spoil_rf_model.pkl"
    path_2 = "./trained_models/trained_amount_rf_model.pkl"
    dl_path = "./dl_trained_models/VGG16_3rd.pth"
    image_path = "./apple-pie.jpeg"
    path = Path("C:/Users/msen6/Documents/Github Projects/datasets/modified-food-101")

    spoil_model = ML_Model(model_file=path_1)
    amount_model = ML_Model(model_file=path_2)

    test_1 = [[3, 5, 85, 258, 4]]
    test_2 = [[291]]

    print("Predicting with spoil model...")
    print(spoil_model.predict(test_1))
    print("\nPredicting with amount model...")
    print(amount_model.predict(test_2))

    print("\nLoading VGG16 model...")
    dl_model = DL_ImgClass(model_path=dl_path, num_classes=50)
    dl_model.prepare_data(path)
    dl_model.load_model(dl_path)

    #print("\nLoading ResNet50 model...")
    #res_model = ResNet50(model_path=dl_path, num_classes=50)
    #res_model.prepare_data(path)
    #res_model.load_model(dl_path)

    print("\nPredicting with DL model...")
    prediction_vgg = dl_model.predict(image_path)
    print(prediction_vgg)
    #prediction_res = res_model.predict(image_path)
    #print(prediction_res)

except Exception as e:
    print(f"\nError occurred: {str(e)}")
