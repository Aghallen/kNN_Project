# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
from data_reader import DataReader
import math
import numpy as np

# Format output of numpy arrays.
np.set_printoptions(formatter={'float': lambda x: "{0:.1f}".format(x).rjust(4)})

X, y = DataReader().read()
# print(X.shape); print(y.shape); exit()


from my_splitter import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

# print(y_test);exit()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# print(X[:5]); print(); print(X_train[:5]); print()
from dataclasses import dataclass


@dataclass
class TrainingData:
    features: list      # Features of a training sample
    label: str          # Label of a training sample
    distance: float     # Distance between a test sample and this trainings sample


@dataclass
class PredictionData:
    predicted_label: str
    features: list      # The features of a test sample. Used for debug.
    neighbors: list     # The features of the k nearest neighbors. Used for debug.


# exit()
from statistics import mode
class MyClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.distance_data_list = []
        self.prediction_data_list = []

    def fit(self, X_train, y_train):
        for features, label in zip(X_train, y_train):
            td = TrainingData(features, label, -1)
            self.distance_data_list.append(td)

    def predict_test_sample(self, test_sample_features):
        # Calculate distance to every sample in X_train.
        for dd in self.distance_data_list:
            squared = [(x1 - x2) * (x1 - x2) for x1, x2 in zip(dd.features, test_sample_features)]
            dd.distance = math.sqrt(sum(squared))

        # Select the k closest neighbors.
        sorted_by_distance = sorted(self.distance_data_list, key=lambda x: x.distance)
        k_neighbors = sorted_by_distance[:self.n_neighbors]

        # Select most frequent neighbor of the k closest.
        predicted_label = mode([x.label for x in k_neighbors])
        pd = PredictionData(predicted_label, test_sample_features, k_neighbors)
        self.prediction_data_list.append(pd)

    def predict(self, X_test):
        for test_sample in X_test:
            self.predict_test_sample(test_sample)

        # for elem in self.prediction_data_list:
        #     print(elem.features, elem.predicted)

        y_pred = [x.predicted_label for x in self.prediction_data_list]
        return y_pred


c = MyClassifier(5)
c.fit(X_train, y_train)
my_y_pred = c.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, my_y_pred))
print()
print(classification_report(y_test, my_y_pred))

exit()
print('my_y_pred')
for single_y_pred in my_y_pred:
    print(single_y_pred)
print()
# exit()

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print('y_pred')
for single_y_pred in y_pred:
    print(single_y_pred)
print()
exit()

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))



