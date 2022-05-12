import json
from turtle import forward
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os
import glob
import Cooking

RAW_DATA_DIR = 'C:/Users/Oguz/Desktop/Carla_Dataset/'
COOKED_DIR = 'C:/Users/Oguz/Desktop/py/EpisodeDatas/'
DIRS = ['episode_00000/', 'episode_00001/', 'episode_00002/', 'episode_00003/', 'episode_00004/', 'episode_00005/', 'episode_00006/',
'episode_00007/', 'episode_00008/', 'episode_00009/', 'episode_00010/', 'episode_00011/', 'episode_00012/', 'episode_00013/', 'episode_00014/',
 'episode_00015/', 'episode_00016/', 'episode_00017/', 'episode_00018/', 'episode_00019/', 'episode_00020/',  'episode_00021/',  'episode_00022/',
 'episode_00023/', 'episode_00024/', 'episode_00025/', 'episode_00026/', 'episode_00027/', 'episode_00028/',  'episode_00029/',  'episode_00030/',
 'episode_00031/', 'episode_00032/', 'episode_00033/', 'episode_00034/', 'episode_00035/', 'episode_00036/',  'episode_00037/',  'episode_00038/',
 'episode_00039/', 'episode_00040/', 'episode_00041/', 'episode_00042/', 'episode_00043/', 'episode_00044/',  'episode_00045/',  'episode_00046/']
 
COLUMNS = ['Brake', 'Reverse', 'Steering', 'Throttle', 'Speed (kmph)', 'ImagePath']

def Configure_data():
    for dir in DIRS:
        print('Starting -> ' + dir)
        cooked_dir = COOKED_DIR + dir.replace('/', '')
        new_dir = os.path.join(RAW_DATA_DIR, dir)
        json_path = os.path.join(new_dir, '*.json')
        
        image_list = []
        data = []
        json_list = []
        last_forward_speed = 0
        for file in glob.glob(json_path):
            if file.split('\\').pop() == 'metadata.json':
                continue
            else:
                with open(file, encoding='latin-1') as f:
                    doc = json.loads(''.join(f.readlines()))
                    try: #sometimes the forward speed is missing, in that case we should insert it the same as last one before the current
                        forward_speed = (doc['playerMeasurements']['''forwardSpeed'''] * 3.6) #CARLA simulator gives speed in m/s, multiply by 3.6 to converting to km/h
                        last_forward_speed = forward_speed
                    except Exception:
                        forward_speed = last_forward_speed
                    """if (round(forward_speed) < 3): #Don't append if the speed is lower than 3kmh
                        continue"""
                    image_path = file.replace('measurements', 'CentralRGB')
                    image_path = image_path.replace('json', 'png')
                    lst = [doc['brake'], doc['reverse'], doc['steer'], doc['throttle'], forward_speed, image_path]
                    data.append(lst)
                    json_list.append(file)

        data_frame = pd.DataFrame(columns= COLUMNS, data = data)
        data_frame.to_csv(path_or_buf= (cooked_dir + '.txt'), sep='\t', index = False)
        print('Writing -> ' + dir)

def AddDataFrames():
    data_list = []
    for dir in DIRS:
        data_path = COOKED_DIR + dir.replace('/', '') + '.txt'
        data = pd.read_csv(data_path, sep='\t')
        data_list.append(data)
    return (pd.concat(data_list, ignore_index= True))

def Plot(data): #data should be a pd DataFrame
    labels = data[['Steering']]
    print(data)
    bins = np.arange(-1, 1, 0.05)
    plt.figure(figsize=(10,10))
    n, b, p = plt.hist(labels.to_numpy(), bins, density=True, facecolor="green")
    plt.xlabel('Steering Angle')
    plt.show()


final_data = AddDataFrames()

final_data.to_csv(path_or_buf= (COOKED_DIR + 'final.txt'), sep='\t', index = False)
folders = [COOKED_DIR]
train_eval_test_split = [0.70, 0.15, 0.15]
Cooking.cook(folders, (COOKED_DIR + 'Cooked/'), train_eval_test_split)

