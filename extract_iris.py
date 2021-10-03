from sklearn import datasets

# This application will extract data from sklearn and save it as a Python file.
FILENAME = r'C:\PycharmProjects\kNN_Project\my_iris.py'

iris = datasets.load_iris()
X = iris.data
y = iris.target

code = ''
code += 'import numpy as np\n'
code += 'from dataclasses import dataclass\n\n'
code += '@dataclass\n'
code += 'class IrisData:\n'
code += '\tdata: np.ndarray\n'
code += '\ttarget: np.ndarray\n'
code += '\t\n'

code += 'class Iris:\n'
code += '\t@classmethod\n'
code += '\tdef create_X(cls):\n'
code += '\t\tX = []\n'

for row in X:
    elements = [str(element) for element in row]
    single_row = 'X.append([' + ', '.join(elements) + '])\n'
    code += '\t\t' + single_row

code += '\n\t\treturn np.array(X)\n\n'

code += '\t@classmethod\n'
code += '\tdef create_y(cls):\n'
code += '\t\ty = []\n'

for row in y:
    code += '\t\ty.append(' + str(row) + ')\n'

code += '\n\t\treturn np.array(y)\n\n'

code += '\t@classmethod\n'
code += '\tdef load_iris(cls):\n'
code += '\t\tiris = IrisData(Iris.create_X(), Iris.create_y())\n'
code += '\t\treturn iris\n'

code += '\n\nif __name__ == \'__main__\':\n'
code += '\tiris = Iris.load_iris()\n'
code += '\tprint(iris.data)\n'
code += '\tprint()\n'
code += '\tprint(iris.target)\n'

with open(FILENAME, 'w') as f:
    f.write(code)
print(code)



