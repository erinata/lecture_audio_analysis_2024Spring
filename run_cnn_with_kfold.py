import pandas

from sklearn.preprocessing import StandardScaler
import numpy

from sklearn.model_selection import KFold
from sklearn import metrics

import keras
from keras.models import Sequential
from keras import layers

from keras.preprocessing.image import ImageDataGenerator


dataset = ImageDataGenerator(rescale = 1/255, validation_split=0.25)
training_dataset = dataset.flow_from_directory(directory='spectrogram_dataset/', 
                                              target_size=(100,50), 
                                              shuffle=True, 
                                              color_mode="grayscale", 
                                              subset="training")
validation_dataset = dataset.flow_from_directory(directory='spectrogram_dataset/', 
                                                target_size=(100,50), 
                                                shuffle=True, 
                                                color_mode="grayscale", 
                                                subset="validation")

machine = Sequential()
machine.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(100,50,1)))
machine.add(layers.Conv2D(filters=64, activation='relu', kernel_size=(3,3)))
machine.add(layers.Flatten())
machine.add(layers.Dense(units=64, activation='relu'))
machine.add(layers.Dense(10, activation='softmax'))

machine.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
machine.fit(x=training_dataset, 
            validation_data=validation_dataset, 
            batch_size=16, 
            epochs=30) 





