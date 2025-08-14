import paho.mqtt.client as mqtt
import time
import logging
import json
import base64
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MQTT Broker details
BROKER_ADDRESS = "localhost"  # or your specific broker IP if different
BROKER_PORT = 1883
IMAGE_TOPIC = "smart_fridge/image"

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info("Successfully connected to MQTT Broker!")
        # Subscribe to the image topic
        client.subscribe(IMAGE_TOPIC)
        logger.info(f"Subscribed to {IMAGE_TOPIC}")
    else:
        logger.error(f"Failed to connect to MQTT Broker, return code: {rc}")

def on_message(client, userdata, msg):
    try:
        # Parse the received message
        data = json.loads(msg.payload.decode())
        
        # Decode the base64 image
        image_data = base64.b64decode(data['image'])
        
        # Generate filename with timestamp
        timestamp = datetime.fromtimestamp(data['timestamp']).strftime('%Y%m%d_%H%M%S')
        filename = f"received_image_{timestamp}.jpg"
        
        # Save the image
        with open(filename, 'wb') as f:
            f.write(image_data)
        
        logger.info(f"Image received and saved as {filename}")
        
    except Exception as e:
        logger.error(f"Error processing received image: {str(e)}")

def on_disconnect(client, userdata, rc):
    if rc != 0:
        logger.warning(f"Unexpected disconnection (rc={rc})")
    else:
        logger.info("Disconnected successfully")

def main():
    # Create MQTT client instance
    client = mqtt.Client(protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    try:
        # Connect to broker
        logger.info(f"Connecting to MQTT broker at {BROKER_ADDRESS}:{BROKER_PORT}")
        client.connect(BROKER_ADDRESS, BROKER_PORT, 60)
        client.loop_start()

        # Keep the script running
        logger.info("Waiting for images... Press Ctrl+C to exit")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Program stopped by user")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        # Clean up
        client.loop_stop()
        client.disconnect()
        logger.info("Disconnected from MQTT Broker")

if __name__ == "__main__":
    main()