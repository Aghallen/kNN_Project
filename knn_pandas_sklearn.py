from data_reader import DataReader
import math
import numpy as np

# Format output from numpy arrays.
np.set_printoptions(formatter={'float': lambda x: "{0:.1f}".format(x).rjust(4)})

X, y = DataReader().read()
print(X.shape); print(y.shape); exit()
exit()

# print(dataset.head(3));print();print(X[:3]);print();print(y[:3])

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from my_splitter import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# print(y_test);exit()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# print(X[:5]); print(); print(X_train[:5]); print()
from dataclasses import dataclass


@dataclass
class DistanceData:
    features: list
    target: str
    distance: float

@dataclass
class PredictionData:
    predicted: str
    features: list  # The features of a test sample. Used for debug.
    neighbors: list


# exit()
from statistics import mode
class MyClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.distance_data_list = []
        self.predicted_list = []
        for x, y in zip(X_train, y_train):
            dd = DistanceData(x, y, -1)
            self.distance_data_list.append(dd)

    def predict_row(self, y_test_row):
        # Calculate distance to every sample in X_train.
        for dd in self.distance_data_list:
            squared = [(x1 - x2) * (x1 - x2) for x1, x2 in zip(dd.features, y_test_row)]
            dd.distance = math.sqrt(sum(squared))

        # Select the k closest neighbors.
        sorted_by_distance = sorted(self.distance_data_list, key=lambda x: x.distance)
        k_neighbors = sorted_by_distance[:self.n_neighbors]

        # Select most frequent neighbor of the k closest.
        predicted = mode([x.target for x in k_neighbors])
        tg = PredictionData(predicted, y_test_row, k_neighbors)
        self.predicted_list.append(tg)

    def predict(self, X_test):
        for test_row in X_test:
            self.predict_row(test_row)

        # for elem in self.predicted_list:
        #     print(elem.features, elem.predicted)

        y_pred = [x.predicted for x in self.predicted_list]
        return y_pred

c = MyClassifier(5)
c.fit(X_train, y_train)
my_y_pred = c.predict(X_test)
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



