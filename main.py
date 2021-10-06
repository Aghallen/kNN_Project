import my_iris
import my_splitter as ms

iris= my_iris.Iris.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.1, random_state=42)
print(f'X_train.shape: {X_train.shape}')
print(f'X_test.shape: {X_test.shape}')

# print(X)
exit()

# rows = []
# for row in X:
#     elements = [str(element) for element in row]
#     single_row = '[' + ', '.join(elements) +']'
#     rows.append(single_row)
#
# all_rows = '[' + ',\n'.join(rows) + ']'
# print(all_rows)
# print(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# Print shape of data to confirm data is loaded
# print(iris.data)
# print(y)


# One thread per python interpreter