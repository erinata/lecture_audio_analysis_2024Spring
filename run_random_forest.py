import pandas

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy

from sklearn.model_selection import KFold
from sklearn import metrics


dataset = pandas.read_csv("dataset.csv")
dataset = dataset.drop(["audio_file"] , axis=1)

print(dataset)
dataset = dataset.sample(frac=1).reset_index(drop=True)

target = dataset.iloc[:,-1]
target, key = pandas.factorize(target)
print(target)
print(key)

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

  machine = RandomForestClassifier(criterion="gini", max_depth=30, n_estimators = 50, bootstrap=True)
  machine.fit(data_training, target_training)
  prediction = machine.predict(data_test)
  result.append(metrics.accuracy_score(target_test, prediction))

print(result)




