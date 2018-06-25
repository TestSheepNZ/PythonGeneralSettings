import numpy as np
from sklearn import linear_model

# Initialise arrays
xList = []
yList = []

# Entry 1
xList.append([1.0, 1])
yList.append(1.9)

# Entry 2
xList.append([2.0, 1])
yList.append(4.1)

# Entry 3
xList.append([4.0, 1])
yList.append(7.9)

# Entry 4
xList.append([5.0, 1])
yList.append(10.1)

# Turn lists into NumPy arrays
X = np.array(xList)
Y = np.array(yList)

# Turn data into a model
reg = linear_model.LinearRegression()
reg.fit (X, Y)
reg.coef_

print ("Predicted time to travel 3m is ", reg.predict([[3.0, 1]]))