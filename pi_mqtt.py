import paho.mqtt.client as mqtt
import json
import base64
import time
from picamera2 import Picamera2
import board
import busio
import adafruit_ssd1306
from PIL import Image, ImageDraw, ImageFont
import os

class SmartFridgePi:
    def __init__(self, broker="192.168.1.3", port=1883):
        # Initialize OLED display
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.oled = adafruit_ssd1306.SSD1306_I2C(128, 64, self.i2c, addr=0x3C)
        self.oled.fill(0)
        self.oled.show()

        # Initialize camera
        self.picam2 = Picamera2()
        camera_config = self.picam2.create_still_configuration()
        self.picam2.configure(camera_config)
        self.picam2.start()

        # Initialize MQTT client
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(broker, port)

        # Image path
        self.image_path = "./captured_image.jpg"

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        # Subscribe to the response topic
        client.subscribe("smart_fridge/response")

    def on_message(self, client, userdata, msg):
        try:
            # Parse the response
            response = json.loads(msg.payload.decode())
            
            if "error" in response:
                self.display_text(f"Error: {response['error']}", 0)
                return
            
            # Display prediction
            self.clear_display()
            self.display_text(f"Food: {response['prediction']}", 0)
            time.sleep(2)
            
            # Display nutrition information if available
            if "nutrition" in response:
                self.display_nutrition_info(response["nutrition"])
            
        except Exception as e:
            self.display_text(f"Error: {str(e)}", 0)

    def capture_image(self):
        """Capture image using Pi Camera and save it"""
        print("Capturing image...")
        self.picam2.capture_file(self.image_path)
        print(f"Image saved to {self.image_path}")

    def send_image(self):
        """Send captured image to MQTT broker"""
        try:
            # Read image file
            with open(self.image_path, "rb") as f:
                image_data = f.read()
            
            # Encode image to base64
            image_b64 = base64.b64encode(image_data).decode()
            
            # Prepare message
            message = {
                "image": image_b64,
                "timestamp": time.time()
            }
            
            # Send message
            self.client.publish("smart_fridge/image", json.dumps(message))
            
        except Exception as e:
            self.display_text(f"Error: {str(e)}", 0)

    def display_text(self, text, y_position):
        """Display text on OLED screen"""
        # Create a new image with a black background
        image = Image.new('1', (self.oled.width, self.oled.height))
        draw = ImageDraw.Draw(image)
        
        # Load default font
        font = ImageFont.load_default()
        
        # Draw text
        draw.text((0, y_position), text, font=font, fill=255)
        
        # Display image
        self.oled.image(image)
        self.oled.show()

    def clear_display(self):
        """Clear the OLED display"""
        self.oled.fill(0)
        self.oled.show()

    def display_nutrition_info(self, nutrition_info):
        """Display nutrition information on OLED screen"""
        self.clear_display()
        
        # Create a new image with a black background
        image = Image.new('1', (self.oled.width, self.oled.height))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        
        # Display nutrition information
        y_pos = 0
        draw.text((0, y_pos), f"Cal: {nutrition_info['calories']:.0f}", font=font, fill=255)
        y_pos += 10
        draw.text((0, y_pos), f"P: {nutrition_info['protein_g']:.1f}g", font=font, fill=255)
        y_pos += 10
        draw.text((0, y_pos), f"F: {nutrition_info['fat_g']:.1f}g", font=font, fill=255)
        y_pos += 10
        draw.text((0, y_pos), f"C: {nutrition_info['carbs_g']:.1f}g", font=font, fill=255)
        y_pos += 10
        draw.text((0, y_pos), f"S: {nutrition_info['sugar_g']:.1f}g", font=font, fill=255)
        y_pos += 10
        draw.text((0, y_pos), f"Fi: {nutrition_info['fiber_g']:.1f}g", font=font, fill=255)
        
        # Display image
        self.oled.image(image)
        self.oled.show()

    def run(self):
        """Main loop"""
        try:
            # Start MQTT client loop in background
            self.client.loop_start()
            
            while True:
                # Capture and send image
                self.capture_image()
                self.send_image()
                
                # Wait before next capture
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\nExiting...")
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.clear_display()
        self.picam2.stop()
        self.client.loop_stop()
        self.client.disconnect()

if __name__ == "__main__":
    # Create and run the SmartFridgePi instance
    # Replace "localhost" with your laptop's IP address
    fridge = SmartFridgePi(broker="localhost", port=1883)
    try:
        fridge.run()
    finally:
        fridge.cleanup() 