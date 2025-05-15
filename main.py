from ml_model import ML_Model
from VGG16_model import DL_ImgClass
import pandas as pd
from pathlib import Path
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QDoubleSpinBox, QSpinBox,
    QPushButton, QFileDialog, QVBoxLayout, QFormLayout, QMessageBox,
    QHBoxLayout, QScrollArea, QComboBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

# Paths and setup
path_1 = "./trained_models/rf_spoil_model.pkl"
df = pd.read_csv("./datasets/smart_fridge_dataset_v1.csv")
dl_path = "./dl_trained_models/VGG16_3rd.pth"
data_path = Path("C:/Users/msen6/Documents/Github Projects/datasets/modified-food-101")
meal_nutrition = "./datasets/meal_nutrition_dataset.csv"

# Load models once
print("Loading DL model...")
dl_model = DL_ImgClass(model_path=dl_path, num_classes=50)
dl_model.prepare_data(data_path)
dl_model.load_model(dl_path)

print("Loading ML models...")
spoil_model = ML_Model(model_file=path_1)

# Prediction functions using preloaded models
def predict_food_name(image_path):
    return dl_model.predict(image_path)

def predict_condition(class_ind, temp, humidity, gas, days):
    test_1 = [[class_ind, temp, humidity, gas, days]]
    condition = spoil_model.predict(test_1)[0]
    return condition

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

# Add User Information section
user_info_label = QLabel("User Information")
user_info_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
scroll_layout.addWidget(user_info_label)

# Add gender selection
gender_combo = QComboBox()
gender_combo.addItems(['male', 'female'])
form_layout.addRow("Gender:", gender_combo)

age_input = QDoubleSpinBox()
age_input.setRange(0, 120)
age_input.setSuffix(" years")
form_layout.addRow("Age:", age_input)

user_weight_input = QDoubleSpinBox()
user_weight_input.setRange(0, 500)
user_weight_input.setSuffix(" kg")
form_layout.addRow("Weight:", user_weight_input)

user_height_input = QDoubleSpinBox()
user_height_input.setRange(0, 300)
user_height_input.setSuffix(" cm")
form_layout.addRow("Height:", user_height_input)

# Add Fridge Information section
fridge_info_label = QLabel("Fridge Information")
fridge_info_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
form_layout.addRow(fridge_info_label)

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

# Add a scroll area for uploaded images
uploaded_images_label = QLabel("Uploaded Images")
uploaded_images_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
uploaded_images_scroll = QScrollArea()
uploaded_images_scroll.setWidgetResizable(True)
uploaded_images_widget = QWidget()
uploaded_images_layout = QVBoxLayout(uploaded_images_widget)
uploaded_images_scroll.setWidget(uploaded_images_widget)
uploaded_images_scroll.setFixedHeight(150)

# Store widget references to prevent deletion
image_data = []  # Stores: {path, weight_input, days_input, row_widget, image_label}

# Upload images
def upload_images():
    files, _ = QFileDialog.getOpenFileNames(window, "Select Images", "", "Images (*.png *.jpg *.jpeg)")
    if files:
        # Clear previous dynamic widgets
        while uploaded_images_layout.count():
            child = uploaded_images_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        image_data.clear()  # Clear the data list

        for file in files:
            # Create a horizontal layout for each image
            image_row = QHBoxLayout()
            
            # Create thumbnail
            thumb_label = QLabel()
            pixmap = QPixmap(file)
            thumb_label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio))
            thumb_label.setFixedSize(100, 100)
            thumb_label.setStyleSheet("border: 1px solid gray;")
            
            # Create inputs
            weight_input = QDoubleSpinBox()
            weight_input.setRange(0, 5000)
            weight_input.setSuffix(" g")

            days_input = QSpinBox()
            days_input.setRange(0, 365)

            # Add widgets to row
            image_row.addWidget(thumb_label)
            image_row.addWidget(QLabel("Weight:"))
            image_row.addWidget(weight_input)
            image_row.addWidget(QLabel("Days:"))
            image_row.addWidget(days_input)

            # Create container widget for the row
            row_widget = QWidget()
            row_widget.setLayout(image_row)
            uploaded_images_layout.addWidget(row_widget)

            # Store references to prevent deletion
            image_data.append({
                "path": file,
                "weight_input": weight_input,
                "days_input": days_input,
                "row_widget": row_widget,
                "image_label": thumb_label
            })

        # Show first image in main display
        pixmap = QPixmap(files[0])
        image_label.setPixmap(pixmap.scaled(image_label.size(), Qt.KeepAspectRatio))
    else:
        image_label.setText("No image uploaded")

# Delete All button logic
def delete_all_images():
    # Clear all the dynamic widgets from the uploaded images layout
    while uploaded_images_layout.count():
        child = uploaded_images_layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()
    
    # Clear the internal data list
    image_data.clear()

    # Reset image label text
    image_label.setText("No image uploaded")

#Food nutrition
def food_info(food_name):
    age = age_input.value()
    user_height = user_height_input.value()
    user_weight = user_weight_input.value()
    gender = gender_combo.currentText()

    if gender == 'male':
        bmr = 10 * user_weight + 6.25 * user_height - 5 * age + 5
    else:
        bmr = 10 * user_weight + 6.25 * user_height - 5 * age - 161
    
    cals_per_meal = round(bmr / 3)
    
    # Read the nutrition dataset using the correct path
    try:
        print(f"Reading nutrition data from: {meal_nutrition}")  # Debug print
        nutrition_df = pd.read_csv(meal_nutrition)
        print(f"Available meals: {nutrition_df['meal'].unique()}")  # Debug print
        print(f"Looking for: {food_name}")  # Debug print
        print(f"Available columns: {nutrition_df.columns.tolist()}")  # Debug print
        
        # Convert food_name to match the format in the dataset
        formatted_food_name = food_name.lower().replace(' ', '_')
        print(f"Formatted food name: {formatted_food_name}")  # Debug print
        
        # Find the food in the dataset
        food_data = nutrition_df[nutrition_df['meal'] == formatted_food_name]
        
        if not food_data.empty:
            nutrition_info = food_data.iloc[0].to_dict()
            print(f"Found nutrition info: {nutrition_info}")  # Debug print
            
            # Store all values in variables
            global food_calories_value, protein_value, fat_value, carbs_value, sugar_value, fiber_value, sodium_value
            
            food_calories_value = float(nutrition_info['calories'])
            protein_value = float(nutrition_info['protein_g'])
            fat_value = float(nutrition_info['fat_g'])  # Fixed column name
            carbs_value = float(nutrition_info['carbs_g'])
            sugar_value = float(nutrition_info['sugar_g'])
            fiber_value = float(nutrition_info['fiber_g'])
            sodium_value = float(nutrition_info['sodium_mg'])
            
            portion = cals_per_meal / food_calories_value * 100
            
            return portion, food_calories_value, protein_value, fat_value, carbs_value, sugar_value, fiber_value, sodium_value
        else:
            QMessageBox.warning(window, "Error", f"Nutrition information not found for {food_name}. Available meals: {', '.join(nutrition_df['meal'].unique())}")
            return None, None, None, None, None, None, None, None
    except FileNotFoundError:
        QMessageBox.warning(window, "Error", f"Nutrition dataset file not found at {meal_nutrition}")
        return None, None, None, None, None, None, None, None
    except Exception as e:
        QMessageBox.warning(window, "Error", f"Error reading nutrition data: {str(e)}\nFile path: {meal_nutrition}\nAvailable columns: {nutrition_df.columns.tolist() if 'nutrition_df' in locals() else 'Not loaded'}")
        return None, None, None, None, None, None, None, None

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
        try:
            img_path = item['path']
            weight = item['weight_input'].value()
            days = item['days_input'].value()

            food_name_predicted = predict_food_name(img_path)
            print(f"Predicted food: {food_name_predicted}")  # Debug print
            class_ind = food_to_index.get(food_name_predicted)

            # Get nutrition information and portion
            result = food_info(food_name_predicted)
            if result[0] is None:  # Check if there was an error
                continue
                
            portion, calories, protein, fat, carbs, sugar, fiber, sodium = result
            
            # Determine sufficiency based on portion
            sufficiency = "sufficient" if weight >= portion else "not enough"
            
            # Get condition from spoil model
            condition = predict_condition(class_ind, temp, humidity, gas, days)

            QMessageBox.information(
                window,
                f"Prediction for {food_name_predicted}",
                f"Food: {food_name_predicted}\n"
                f"Condition: {condition}\n"
                f"Sufficiency: {sufficiency}\n\n"
                f"Nutrition Information:\n"
                f"Calories: {calories:.1f} kcal\n"
                f"Protein: {protein:.1f}g\n"
                f"Fat: {fat:.1f}g\n"
                f"Carbs: {carbs:.1f}g\n"
                f"Sugar: {sugar:.1f}g\n"
                f"Fiber: {fiber:.1f}g\n"
                f"Sodium: {sodium:.1f}mg\n\n"
                f"Recommended portion: {portion:.1f}g"
            )
        except Exception as e:
            QMessageBox.warning(window, "Error", f"Error processing image: {str(e)}")
            continue

# Buttons
upload_button = QPushButton("Upload Images")
upload_button.clicked.connect(upload_images)
upload_button.setStyleSheet("padding: 8px; margin: 5px;")

delete_all_button = QPushButton("Delete All Images")
delete_all_button.clicked.connect(delete_all_images)
delete_all_button.setStyleSheet("padding: 8px; margin: 5px;")

analyze_button = QPushButton("Analyze")
analyze_button.clicked.connect(analyze_photos)
analyze_button.setStyleSheet("padding: 8px; margin: 5px; background-color: #4CAF50; color: white;")

# Add widgets to layout
scroll_layout.addLayout(form_layout)
scroll_layout.addWidget(upload_button)
scroll_layout.addWidget(delete_all_button)
scroll_layout.addWidget(image_label)
scroll_layout.addWidget(uploaded_images_label)
scroll_layout.addWidget(uploaded_images_scroll)
scroll_layout.addWidget(analyze_button)

scroll.setWidget(content_widget)

main_layout = QVBoxLayout(window)
main_layout.addWidget(scroll)

window.show()
sys.exit(app.exec())
