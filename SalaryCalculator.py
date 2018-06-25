import numpy as np
# For X - first dimension is years experience
#       - second dimension is 0 for female, 1 for male
X = np.array([[1, 0], [5, 0], [10, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [1, 0], [1, 0], [10, 0], [9, 0], [7, 0], [8, 0], [6, 0], [5, 0], [4, 0], [3, 0], [2, 0], [2, 0], [6, 0], [6, 0], [7, 0], [10, 0], [9, 0], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [5, 1], [5, 1], [2, 1], [3, 1], [4, 1], [4, 1], [3, 1], [6, 1], [6, 1], [6, 1], [7, 1], [7, 1], [7, 1], [8, 1], [8, 1], [8, 1], [9, 1], [10, 1], [10, 1], [9, 1], [9, 1], [9, 1], [10, 1]])
Y = np.array([65, 80, 95, 67, 76, 80, 84, 87, 90, 93, 94, 101, 60, 70, 110, 98, 88, 86, 100, 75, 74, 72, 66, 70, 80, 78, 81, 120, 114, 70, 77, 88, 94, 104, 100, 95, 82, 83, 96, 93, 91, 102, 105, 103, 109, 111, 103, 112, 117, 120, 125, 135, 120, 130, 118, 110, 123])

from sklearn import linear_model


# Initialise arrays
xList = []
yList = []

# Data entries - feel free to play around and alter the data here
# ===============================================================
xList.append([1, 0])
yList.append(65)

xList.append([5, 0])
yList.append(80)

xList.append([10, 0])
yList.append(95)

xList.append([2, 0])
yList.append(67)

xList.append([3, 0])
yList.append(76)

xList.append([4, 0])
yList.append(80)

xList.append([5, 0])
yList.append(84)

xList.append([6, 0])
yList.append(87)

xList.append([7, 0])
yList.append(90)

xList.append([8, 0])
yList.append(93)

xList.append([9, 0])
yList.append(94)

xList.append([10, 0])
yList.append(101)

xList.append([1, 0])
yList.append(60)

xList.append([1, 0])
yList.append(70)

xList.append([10, 0])
yList.append(110)

xList.append([9, 0])
yList.append(98)

xList.append([7, 0])
yList.append(88)

xList.append([8, 0])
yList.append(86)

xList.append([6, 0])
yList.append(100)

xList.append([5, 0])
yList.append(75)

xList.append([4, 0])
yList.append(74)

xList.append([3, 0])
yList.append(72)

xList.append([2, 0])
yList.append(66)

xList.append([2, 0])
yList.append(70)

xList.append([6, 0])
yList.append(80)

xList.append([6, 0])
yList.append(78)

xList.append([7, 0])
yList.append(81)

xList.append([10, 0])
yList.append(120)

xList.append([9, 0])
yList.append(114)

xList.append([1, 0])
yList.append(70)

xList.append([2, 0])
yList.append(77)

xList.append([3, 0])
yList.append(88)

xList.append([4, 0])
yList.append(94)

xList.append([5, 0])
yList.append(104)

xList.append([5, 0])
yList.append(100)

xList.append([5, 0])
yList.append(95)

xList.append([2, 0])
yList.append(82)

xList.append([3, 0])
yList.append(83)

xList.append([4, 0])
yList.append(96)

xList.append([4, 0])
yList.append(93)

xList.append([3, 0])
yList.append(91)

xList.append([6, 0])
yList.append(102)

xList.append([6, 0])
yList.append(105)

xList.append([6, 0])
yList.append(103)

xList.append([7, 0])
yList.append(109)

xList.append([7, 0])
yList.append(111)

xList.append([7, 0])
yList.append(103)

xList.append([8, 0])
yList.append(112)

xList.append([8, 0])
yList.append(117)

xList.append([8, 0])
yList.append(120)

xList.append([9, 0])
yList.append(125)

xList.append([10, 0])
yList.append(135)

xList.append([10, 0])
yList.append(120)

xList.append([9, 0])
yList.append(130)

xList.append([9, 0])
yList.append(118)

xList.append([9, 0])
yList.append(110)

xList.append([10, 0])
yList.append(123)




# Turn lists into NumPy arrays
X = np.array(xList)
Y = np.array(yList)

# Turn data into a model and print predictions
# ============================================


# Linear regression model
reg = linear_model.LinearRegression()
reg.fit (X , Y)
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
reg.fit (X , Y)
reg.coef_
print ("USING BAYESIAN RIDGE")
print ("-----------------------")
print ("Man with 5 years experience salary ", reg.predict([[5, 1]]))
print ("Woman with 5 years experience salary ", reg.predict([[5, 0]]))
print

# Lasso model
reg = linear_model.Lasso(alpha = 0.1)
reg.fit (X , Y)
reg.coef_
print ("USING LASSO MODEL")
print ("-----------------------")
print ("Man with 5 years experience salary ", reg.predict([[5, 1]]))
print ("Woman with 5 years experience salary ", reg.predict([[5, 0]]))
print