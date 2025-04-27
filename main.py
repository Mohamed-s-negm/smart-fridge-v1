from ml_model import ML_Model
import os

path_1 = "./ml_models/trained_spoil_rf_model.pkl"
path_2 = "./ml_models/trained_amount_rf_model.pkl"

spoil_model = ML_Model(model_file=path_1)
amount_model = ML_Model(model_file=path_2)

test_1 = [[3, 5, 85, 258, 4]]
test_2 = [[291]]

print(spoil_model.predict(test_1))
print(amount_model.predict(test_2))