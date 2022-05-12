from tkinter import image_types
from keras.models import load_model
import sys
import numpy as np
import time
import glob
import os

if ('C:/Users/Oguz/Desktop/AirSim/PythonClient/' not in sys.path):
    sys.path.insert(0, 'C:/Users/Oguz/Desktop/AirSim/PythonClient/')
from airsim import *

# << Set this to the path of the model >>
# If None, then the model with the lowest validation loss from training will be used
MODEL_PATH = "C:/Users/Oguz/Desktop/py/EpisodeDatas/Cooked/models/model_model.20-0.0001977.h5"


    
print('Using model {0} for testing.'.format(MODEL_PATH))
model = load_model(MODEL_PATH)

client = CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = CarControls()
print('Connection established!')
car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0


image_buf = np.zeros((1, 66, 200, 3))
state_buf = np.zeros((1,4))

def get_image():
    image_responses = client.simGetImages([ImageRequest(0, ImageType.Scene, False, False)])
    image_response = image_responses[0]
    image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 3)
    return image_rgba[74:140,26:226,0:3].astype(float)

  
while (True):

    car_state = client.getCarState()
    
    image_buf[0] = get_image()
    state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])
    model_output = model.predict([image_buf, state_buf])
    car_controls.steering = round(0.5 * float(model_output[0][0]), 2)
        
    if (abs(car_controls.steering) < 0.10):
        car_controls.throttle = 0.4
        car_controls.brake = 0
    elif((abs(car_controls.steering) > 0.35) or (car_state.speed > 45)):
        car_controls.throttle = 0
        car_controls.brake = 0.4
    elif(abs(car_controls.steering) > 0.55):
        car_controls.throttle = 0
        car_controls.brake = 0.8
    else:
        car_controls.throttle = 0.5

    print('Sending steering = {0}, throttle = {1}'.format(car_controls.steering, car_controls.throttle))
    
    client.setCarControls(car_controls)
