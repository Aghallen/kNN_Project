# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
from data_reader import DataReader
from my_classifier import MyClassifier

import numpy as np
from sklearn.preprocessing import StandardScaler
from my_splitter import train_test_split


# Format output of numpy arrays.
np.set_printoptions(formatter={'float': lambda x: "{0:.1f}".format(x).rjust(4)})

X, y = DataReader().read()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

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



