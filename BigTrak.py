import numpy as np
from sklearn import linear_model

# Initialise arrays
inputArray = []
outputArray = []

# Entry 1
inputArray.append([1.0, 1])
outputArray.append(1.9)

# Entry 2
inputArray.append([2.0, 1])
outputArray.append(4.1)

# Entry 3
inputArray.append([4.0, 1])
outputArray.append(7.9)

# Entry 4
inputArray.append([5.0, 1])
outputArray.append(10.1)

# Turn lists into NumPy arrays
inputData = np.array(inputArray)
outputData = np.array(outputArray)

# Turn data into a model
reg = linear_model.LinearRegression()
reg.fit (inputData, outputData)
reg.coef_

print ("Predicted time to travel 3m is ", reg.predict([[3.0, 1]]))