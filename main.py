from ml_model import ML_Model
from VGG16_model import DL_ImgClass
import pandas as pd
from pathlib import Path
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QDoubleSpinBox, QSpinBox,
    QPushButton, QFileDialog, QVBoxLayout, QFormLayout, QMessageBox,
    QHBoxLayout, QScrollArea
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

# Paths and setup
path_1 = "./trained_models/rf_spoil_model.pkl"
path_2 = "./trained_models/rf_amount_model.pkl"
df = pd.read_csv("./datasets/smart_fridge_dataset_v1.csv")
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

# Prediction functions using preloaded models
def predict_food_name(image_path):
    return dl_model.predict(image_path)

def predict_condition_and_sufficiency(class_ind, temp, humidity, gas, days, weight):
    test_1 = [[class_ind, temp, humidity, gas, days]]
    test_2 = [[class_ind, weight]]
    condition = spoil_model.predict(test_1)[0]
    sufficiency = amount_model.predict(test_2)[0]
    return condition, sufficiency

# App setup
app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("Smart Fridge Analyzer")
window.resize(400, 600)

# Scroll Area
scroll = QScrollArea()
scroll.setWidgetResizable(True)
content_widget = QWidget()
scroll_layout = QVBoxLayout(content_widget)

# Form for common inputs
form_layout = QFormLayout()

temp_input = QDoubleSpinBox()
temp_input.setRange(-20, 100)
temp_input.setSuffix(" °C")
form_layout.addRow("Temperature:", temp_input)

humidity_input = QDoubleSpinBox()
humidity_input.setRange(0, 100)
humidity_input.setSuffix(" %")
form_layout.addRow("Humidity:", humidity_input)

gas_input = QDoubleSpinBox()
gas_input.setRange(0, 50)
gas_input.setSuffix(" ppm")
form_layout.addRow("Gas:", gas_input)

# Image display and storage
image_label = QLabel("No image uploaded")
image_label.setAlignment(Qt.AlignCenter)
image_label.setStyleSheet("border: 1px solid gray;")
image_label.setFixedHeight(200)
image_label.setScaledContents(True)

image_data = []  # Stores: {path, weight_input, days_input}

# Upload images
def upload_images():
    files, _ = QFileDialog.getOpenFileNames(window, "Select Images", "", "Images (*.png *.jpg *.jpeg)")
    if files:
        image_data.clear()

        # Clear previous dynamic widgets
        while scroll_layout.count() > 4:
            child = scroll_layout.takeAt(4)
            if child.widget():
                child.widget().deleteLater()

        for file in files:
            weight_input = QDoubleSpinBox()
            weight_input.setRange(0, 5000)
            weight_input.setSuffix(" g")

            days_input = QSpinBox()
            days_input.setRange(0, 365)

            label = QLabel(f"Image: {file.split('/')[-1]}")
            label.setStyleSheet("font-weight: bold; margin-top: 10px;")

            row = QHBoxLayout()
            row.addWidget(QLabel("Weight:"))
            row.addWidget(weight_input)
            row.addWidget(QLabel("Days:"))
            row.addWidget(days_input)

            scroll_layout.addWidget(label)
            scroll_layout.addLayout(row)

            image_data.append({
                "path": file,
                "weight_input": weight_input,
                "days_input": days_input
            })

        pixmap = QPixmap(files[0])
        image_label.setPixmap(pixmap)
    else:
        image_label.setText("No image uploaded")

# Analyze button logic
def analyze_photos():
    if not image_data:
        QMessageBox.warning(window, "No Images", "Please upload at least one image.")
        return

    temp = temp_input.value()
    humidity = humidity_input.value()
    gas = gas_input.value()
    food_to_index = df[['food_type', 'class_index']].drop_duplicates().set_index('food_type')['class_index'].to_dict()

    for item in image_data:
        img_path = item['path']
        weight = item['weight_input'].value()
        days = item['days_input'].value()

        food_name_predicted = predict_food_name(img_path)
        class_ind = food_to_index.get(food_name_predicted)

        condition, sufficiency = predict_condition_and_sufficiency(class_ind, temp, humidity, gas, days, weight)

        QMessageBox.information(
            window,
            f"Prediction for {food_name_predicted}",
            f"Food: {food_name_predicted}\n"
            f"Condition: {condition}\n"
            f"Sufficiency: {sufficiency} amount"
        )


# Buttons
upload_button = QPushButton("Upload Images")
upload_button.clicked.connect(upload_images)

analyze_button = QPushButton("Analyze")
analyze_button.clicked.connect(analyze_photos)

# Add widgets to layout
scroll_layout.addLayout(form_layout)
scroll_layout.addWidget(upload_button)
scroll_layout.addWidget(image_label)
scroll_layout.addWidget(analyze_button)

scroll.setWidget(content_widget)

main_layout = QVBoxLayout(window)
main_layout.addWidget(scroll)

window.show()
sys.exit(app.exec())
