import paho.mqtt.client as mqtt
import time

# MQTT Broker details
BROKER_ADDRESS = "localhost"
BROKER_PORT = 1883
TEST_TOPIC = "test/topic"
TEST_MESSAGE = "Hello MQTT!"

# Callback functions
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}\n")

def on_message(client, userdata, msg):
    print(f"Received message: {msg.payload.decode()} on topic {msg.topic}")

def on_publish(client, userdata, mid):
    print(f"Message {mid} published successfully")

# Create and configure publisher
publisher = mqtt.Client("Publisher", protocol=mqtt.MQTTv311)
publisher.on_connect = on_connect
publisher.on_publish = on_publish

# Create and configure subscriber
subscriber = mqtt.Client("Subscriber", protocol=mqtt.MQTTv311)
subscriber.on_connect = on_connect
subscriber.on_message = on_message

try:
    # Connect both clients
    print("Connecting to broker...")
    publisher.connect(BROKER_ADDRESS, BROKER_PORT, 60)
    subscriber.connect(BROKER_ADDRESS, BROKER_PORT, 60)
    
    # Start the loops
    publisher.loop_start()
    subscriber.loop_start()
    
    # Subscribe to the test topic
    subscriber.subscribe(TEST_TOPIC)
    print(f"Subscribed to {TEST_TOPIC}")
    
    # Wait a moment for the subscription to take effect
    time.sleep(1)
    
    # Publish a test message
    print(f"Publishing message: {TEST_MESSAGE}")
    publisher.publish(TEST_TOPIC, TEST_MESSAGE)
    
    # Wait to receive the message
    time.sleep(2)
    
except Exception as e:
    print(f"An error occurred: {e}")
    
finally:
    # Clean up
    publisher.loop_stop()
    subscriber.loop_stop()
    publisher.disconnect()
    subscriber.disconnect()
    print("Test completed and connections closed.") 