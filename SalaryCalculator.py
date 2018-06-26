import numpy as np
# For input - first dimension is years experience
#           - second dimension is 0 for female, 1 for male

from sklearn import linear_model


# Initialise arrays
inputArray = []
outputArray = []

# Data entries - feel free to play around and alter the data here
# ===============================================================
inputArray.append([1, 0])
outputArray.append(65)

inputArray.append([5, 0])
outputArray.append(80)

inputArray.append([10, 0])
outputArray.append(95)

inputArray.append([2, 0])
outputArray.append(67)

inputArray.append([3, 0])
outputArray.append(76)

inputArray.append([4, 0])
outputArray.append(80)

inputArray.append([5, 0])
outputArray.append(84)

inputArray.append([6, 0])
outputArray.append(87)

inputArray.append([7, 0])
outputArray.append(90)

inputArray.append([8, 0])
outputArray.append(93)

inputArray.append([9, 0])
outputArray.append(94)

inputArray.append([10, 0])
outputArray.append(101)

inputArray.append([1, 0])
outputArray.append(60)

inputArray.append([1, 0])
outputArray.append(70)

inputArray.append([10, 0])
outputArray.append(110)

inputArray.append([9, 0])
outputArray.append(98)

inputArray.append([7, 0])
outputArray.append(88)

inputArray.append([8, 0])
outputArray.append(86)

inputArray.append([6, 0])
outputArray.append(100)

inputArray.append([5, 0])
outputArray.append(75)

inputArray.append([4, 0])
outputArray.append(74)

inputArray.append([3, 0])
outputArray.append(72)

inputArray.append([2, 0])
outputArray.append(66)

inputArray.append([2, 0])
outputArray.append(70)

inputArray.append([6, 0])
outputArray.append(80)

inputArray.append([6, 0])
outputArray.append(78)

inputArray.append([7, 0])
outputArray.append(81)

inputArray.append([10, 0])
outputArray.append(120)

inputArray.append([9, 0])
outputArray.append(114)

inputArray.append([1, 0])
outputArray.append(70)

inputArray.append([2, 0])
outputArray.append(77)

inputArray.append([3, 0])
outputArray.append(88)

inputArray.append([4, 0])
outputArray.append(94)

inputArray.append([5, 0])
outputArray.append(104)

inputArray.append([5, 0])
outputArray.append(100)

inputArray.append([5, 0])
outputArray.append(95)

inputArray.append([2, 0])
outputArray.append(82)

inputArray.append([3, 0])
outputArray.append(83)

inputArray.append([4, 0])
outputArray.append(96)

inputArray.append([4, 0])
outputArray.append(93)

inputArray.append([3, 0])
outputArray.append(91)

inputArray.append([6, 0])
outputArray.append(102)

inputArray.append([6, 0])
outputArray.append(105)

inputArray.append([6, 0])
outputArray.append(103)

inputArray.append([7, 0])
outputArray.append(109)

inputArray.append([7, 0])
outputArray.append(111)

inputArray.append([7, 0])
outputArray.append(103)

inputArray.append([8, 0])
outputArray.append(112)

inputArray.append([8, 0])
outputArray.append(117)

inputArray.append([8, 0])
outputArray.append(120)

inputArray.append([9, 0])
outputArray.append(125)

inputArray.append([10, 0])
outputArray.append(135)

inputArray.append([10, 0])
outputArray.append(120)

inputArray.append([9, 0])
outputArray.append(130)

inputArray.append([9, 0])
outputArray.append(118)

inputArray.append([9, 0])
outputArray.append(110)

inputArray.append([10, 0])
outputArray.append(123)




# Turn lists into NumPy arrays
inputData = np.array(inputArray)
outputData = np.array(outputArray)

# Turn data into a model and print predictions
# ============================================


# Linear regression model
reg = linear_model.LinearRegression()
reg.fit (inputData, outputData)
reg.coef_
print ("USING LINEAR REGRESSION")
print ("-----------------------")
print ("Man with 5 years experience salary ", reg.predict([[5, 1]]))
print ("Woman with 5 years experience salary ", reg.predict([[5, 0]]))
print ("Man with 4.5 years experience salary ", reg.predict([[4.5, 1]]))
print ("Woman with 4.5 years experience salary ", reg.predict([[4.5, 0]]))
print

# Baysean Ridge model
reg = linear_model.BayesianRidge()
reg.fit (inputData, outputData)
reg.coef_
print ("USING BAYESIAN RIDGE")
print ("-----------------------")
print ("Man with 5 years experience salary ", reg.predict([[5, 1]]))
print ("Woman with 5 years experience salary ", reg.predict([[5, 0]]))
print

# Lasso model
reg = linear_model.Lasso(alpha = 0.1)
reg.fit (inputData, outputData)
reg.coef_
print ("USING LASSO MODEL")
print ("-----------------------")
print ("Man with 5 years experience salary ", reg.predict([[5, 1]]))
print ("Woman with 5 years experience salary ", reg.predict([[5, 0]]))
print