import numpy
import matplotlib.pyplot as plt

import sys
import pprint


from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import read_csv
import numpy as np
 
data = read_csv("humiditydiff.csv")
data = data['Open'].tolist()
l = 2
data = seasonal_decompose(data, period=l, two_sided=False).trend[2:len(data)-2]

x = []
y = []
look_back = 4

for i in range((len(data)-look_back)):

    x.append(data[i:i+look_back])
    y.append(data[i+look_back])
#print(x)
X = np.array(x)
y = np.array(y)
print(X[0], y[0])

# Assume you have independent variables X and a dependent variable y
#X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
#X = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])

#y = np.array([1, 2, 3, 4])


# Create an instance of the LinearRegression class
reg = LinearRegression()
 
# Fit the model to the data
reg.fit(X, y)
 
# Print the coefficients of the model
print(reg.coef_)
mass = 1
masshistory = []

for i in range(len(data)-look_back):
    xvals = [data[i:i+look_back]]

    if reg.predict(xvals) > 0:
        mass *= (1 + data[i+look_back])
    elif reg.predict(xvals) < 0:
        mass *= 1 / (1 + data[i+look_back])
    masshistory.append(mass)

plt.plot(masshistory)
plt.show()




""" 
from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import lag_plot
series = read_csv('humiditydiff.csv', header=0, index_col=0)
lag_plot(series)
pyplot.show()

 """
""" 
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
series = read_csv('humiditydiff.csv', header=0, index_col=0)
values = DataFrame(series.values)

dataframe = concat([values.shift(1), values], axis=1)
dataframe = concat([values.shift(2), dataframe], axis=1)
dataframe = concat([values.shift(3), dataframe], axis=1)
dataframe = concat([values.shift(4), dataframe], axis=1)
dataframe = concat([values.shift(5), dataframe], axis=1)
dataframe = concat([values.shift(6), dataframe], axis=1)
dataframe = concat([values.shift(7), dataframe], axis=1)
dataframe = concat([values.shift(8), dataframe], axis=1)
dataframe = concat([values.shift(9), dataframe], axis=1)
dataframe = concat([values.shift(10), dataframe], axis=1)

#print(dataframe[0:20])
dataframe.columns = ['t', 't+1', 't+2', 't+3', 't+4', 't+5', 't+6', 't+7', 't+8', 't+9', 't+10']
result = dataframe.corr()
print(result)

print(result.corr())
 """
""" 
import scipy.optimize as optimize


def f(params):
    # print(params)  # <-- you'll see that params is a NumPy array
    a, b, c = params # <-- for readability you may wish to assign names to the component variables
    return c**2+1

initial_guess = [1, 1, 1]
result = optimize.minimize(f, initial_guess)

if result.success:
    fitted_params = result.x
    print(fitted_params)
else:
    raise ValueError(result.message)
 """
