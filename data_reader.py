import pandas as pd


class DataReader:
    def read(self, read_from_file=True):
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
        if read_from_file:
            import pathlib
            filename = 'iris.data'  # Located in the same directory as the current file.
            full_filename = str(pathlib.Path().absolute().joinpath(filename))
            df = pd.read_csv(full_filename, names=names)  # pandas.core.frame.DataFrame
        else:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
            df = pd.read_csv(url, names=names)  # pandas.core.frame.DataFrame

        X = df.iloc[:, :-1].values
        y = df.iloc[:, 4].values

        return X, y