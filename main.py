from ml_model import ML_Model
from dl_model import DL_ImgClass
from pathlib import Path

path_1 = "./ml_models/trained_spoil_rf_model.pkl"
path_2 = "./ml_models/trained_amount_rf_model.pkl"
dl_path = Path("C:/Users/msen6/Documents/Github Projects/datasets/food-101-dataset/food-101")
image_path = "./apple.jpg"

spoil_model = ML_Model(model_file=path_1)
amount_model = ML_Model(model_file=path_2)

test_1 = [[3, 5, 85, 258, 4]]
test_2 = [[291]]

print(spoil_model.predict(test_1))
print(amount_model.predict(test_2))

dl_model = DL_ImgClass(num_classes=101)
dl_model.prepare_data(str(dl_path))
dl_model.train()
dl_model.predict(image_path)