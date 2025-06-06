import paho.mqtt.client as mqtt
import json
import base64
import pandas as pd
from pathlib import Path
from dl_pi import DL_ImgClass
import time
import os

class SmartFridgeLaptop:
    def __init__(self, broker="localhost", port=1883):
        # Initialize MQTT client
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(broker, port)
        
        # Load model and nutrition data
        self.load_resources()
        
        # Create temp directory for images if it doesn't exist
        os.makedirs("temp_images", exist_ok=True)

    def load_resources(self):
        """Load the model and nutrition data"""
        print("Loading resources...")
        
        # Load model
        dl_path = "./dl_trained_models/best_model_fruit.pth"
        data_path = Path("C:/Users/msen6/Documents/Github Projects/datasets/fresh_fruits/dataset")
        self.dl_model = DL_ImgClass(model_path=dl_path, num_classes=6)
        self.dl_model.prepare_data(data_path)
        self.dl_model.load_model(dl_path)
        
        # Load nutrition data
        self.nutrition_df = pd.read_csv("./datasets/fruit_nutrition_dataset.csv")
        
        print("Resources loaded successfully")

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        # Subscribe to the image topic
        client.subscribe("smart_fridge/image")

    def on_message(self, client, userdata, msg):
        try:
            # Parse the message
            message = json.loads(msg.payload.decode())
            image_data = base64.b64decode(message['image'])
            
            # Save the image temporarily
            temp_image_path = f"temp_images/temp_{int(time.time())}.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(image_data)
            
            # Get prediction
            prediction = self.dl_model.predict(temp_image_path)
            
            # Get nutrition information
            food_data = self.nutrition_df[self.nutrition_df['meal'] == prediction.lower().replace(' ', '_')]
            
            # Prepare response
            response = {
                "prediction": prediction,
                "timestamp": time.time()
            }
            
            if not food_data.empty:
                response["nutrition"] = food_data.iloc[0].to_dict()
            else:
                response["error"] = "No nutrition data found"
            
            # Send response
            self.client.publish("smart_fridge/response", json.dumps(response))
            
            # Clean up temporary image
            os.remove(temp_image_path)
            
        except Exception as e:
            # Send error response
            error_response = {
                "error": str(e),
                "timestamp": time.time()
            }
            self.client.publish("smart_fridge/response", json.dumps(error_response))

    def run(self):
        """Start the MQTT client loop"""
        try:
            print("Starting MQTT client...")
            self.client.loop_forever()
        except KeyboardInterrupt:
            print("\nExiting...")
            self.client.disconnect()

if __name__ == "__main__":
    # Create and run the SmartFridgeLaptop instance
    laptop = SmartFridgeLaptop(broker="localhost", port=1883)
    laptop.run() 