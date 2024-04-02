import pandas

from sklearn.preprocessing import StandardScaler
import numpy

from sklearn.model_selection import KFold
from sklearn import metrics

import keras
from keras import layers
from keras.models import Sequential 

dataset = pandas.read_csv("dataset.csv")
dataset = dataset.drop(["audio_file"] , axis=1)
dataset = dataset.sample(frac=1).reset_index(drop=True)

target = dataset.iloc[:,-1]
target, key = pandas.factorize(target)

data = dataset.iloc[:,:-1]
scaler = StandardScaler()
data = scaler.fit_transform(numpy.array(data, dtype=float))



split_number = 4

kfold_object = KFold(n_splits=split_number)
kfold_object.get_n_splits(data)

result = []


for training_index, test_index in kfold_object.split(data):
  data_training = data[training_index]
  target_training = target[training_index]
  data_test = data[test_index]
  target_test = target[test_index]
  
  machine = Sequential()
  machine.add(layers.Dense(256, activation="relu", input_shape=(data_training.shape[1],) ))
  machine.add(layers.Dense(128, activation="relu"))
  machine.add(layers.Dense(64, activation="relu"))
  machine.add(layers.Dense(32, activation="relu"))
  machine.add(layers.Dense(10, activation="softmax"))
  
  machine.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
  
  machine.fit(data_training, target_training, batch_size=32, epochs=100)
  
  prediction = numpy.argmax(machine.predict(data_test), axis=-1)
  result.append(metrics.accuracy_score(target_test, prediction))
  
print(result)
  
  






