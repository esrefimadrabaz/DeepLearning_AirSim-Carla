from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate, BatchNormalization
from keras.layers.advanced_activations import ELU
from tensorflow.keras.optimizers import Adam, SGD, Adamax, Nadam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
import keras.backend as K
from keras.preprocessing import image

from keras_tqdm import TQDMNotebookCallback

import json
import os
import numpy as np
import pandas as pd
from Generator import DriveDataGenerator
from Cooking import checkAndCreateDir
import h5py
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt

# << The directory containing the cooked data from the previous step >>
COOKED_DATA_DIR = 'C:/Users/Oguz/Desktop/py/EpisodeDatas/Cooked/'

# << The directory in which the model output will be placed >>
MODEL_OUTPUT_DIR = 'C:/Users/Oguz/Desktop/py/EpisodeDatas/Cooked/'
train_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'train.h5'), 'r')
eval_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'eval.h5'), 'r')
test_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'test.h5'), 'r')

num_train_examples = train_dataset['image'].shape[0]
num_eval_examples = eval_dataset['image'].shape[0]
num_test_examples = test_dataset['image'].shape[0]

batch_size=32
data_generator = DriveDataGenerator(rescale=1./255., horizontal_flip=True, brightness_range=(0.4, 1.4))
train_generator = data_generator.flow\
    (train_dataset['image'], train_dataset['previous_state'], train_dataset['label'], batch_size=batch_size, zero_drop_percentage=0.8, roi=[18,84,0,200])
eval_generator = data_generator.flow\
    (eval_dataset['image'], eval_dataset['previous_state'], eval_dataset['label'], batch_size=batch_size, zero_drop_percentage=0.8, roi=[18,84,0,200])    

[sample_batch_train_data, sample_batch_test_data] = next(train_generator)

image_input_shape = sample_batch_train_data[0].shape[1:]
state_input_shape = sample_batch_train_data[1].shape[1:]

activation = 'relu'

#Create the convolutional stacks
pic_input = Input(shape=image_input_shape)

img_stack = Conv2D(24, (5, 5), strides = (2,2), padding='same', activation=activation)(pic_input)
img_stack = Conv2D(36, (5, 5), strides = (2,2), activation=activation, padding='same')(img_stack)
img_stack = Conv2D(48, (5, 5), strides = (2,2), activation=activation, padding='same')(img_stack)
img_stack = Conv2D(64, (3, 3), strides = (2,2), activation=activation)(img_stack)
img_stack = Dropout(0.2)(img_stack)
img_stack = Conv2D(64, (3, 3), strides = (1,1), activation=activation)(img_stack)
img_stack = Dropout(0.2)(img_stack)
img_stack = Flatten()(img_stack)


#Inject the state input
state_input = Input(shape=state_input_shape)
merged = concatenate([img_stack, state_input])

# Add a few dense layers to finish the model
merged = Dense(100, activation=activation)(merged)
merged = Dropout(0.2)(merged)
merged = Dense(50, activation=activation)(merged)
merged = Dropout(0.2)(merged)
merged = Dense(10, activation=activation)(merged)
merged = Dense(1, name='output')(merged)

adam = Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model = Model(inputs=[pic_input, state_input], outputs=merged)
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
model.summary()



plateau_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00005, verbose=1)
checkpoint_filepath = os.path.join(MODEL_OUTPUT_DIR, 'models', '{0}_model.{1}-{2}.h5'.format('model', '{epoch:02d}', '{val_loss:.7f}'))
checkAndCreateDir(checkpoint_filepath)
checkpoint_callback = ModelCheckpoint(checkpoint_filepath, save_best_only=True, verbose=1)
csv_callback = CSVLogger(os.path.join(MODEL_OUTPUT_DIR, 'training_log.csv'))
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
callbacks=[plateau_callback, csv_callback, checkpoint_callback, early_stopping_callback]

history = model.fit_generator(train_generator, steps_per_epoch=num_train_examples//batch_size, epochs=100, callbacks=callbacks,\
                   validation_data=eval_generator, validation_steps=num_eval_examples//batch_size)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()