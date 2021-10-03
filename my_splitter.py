import numpy as np
import my_iris

def train_test_split(X, y, test_size=0.33, random_state=None):
    indexes = np.array(range(len(X)))
    np.random.seed(random_state)
    np.random.shuffle(indexes)
    test_count = round(len(indexes) * test_size)

    test_indexes = indexes[:test_count]
    train_indexes = indexes[test_count:]

    X_train = X[train_indexes]
    X_test = X[test_indexes]
    y_train = y[train_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    iris = my_iris.Iris.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print(f'X_train.shape: {X_train.shape}')
    print(f'X_test.shape: {X_test.shape}')
    print(f'y_train.shape: {y_train.shape}')
    print(f'y_test.shape: {y_test.shape}')