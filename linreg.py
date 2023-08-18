import numpy
import matplotlib.pyplot as plt
from statistics import mean
import sys
import pprint
import math

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.seasonal import seasonal_decompose, convolution_filter
from pandas import read_csv
import numpy as np
import math
from statistics import mean
data = read_csv("datashop/humidityhourdiff.csv")
data = data['Open'].tolist()
ccdata = data
l = 2

#data = seasonal_decompose(ccdata, period=l, two_sided=False).trend
#data2 = seasonal_decompose(ccdata, period=l, two_sided=False).resid

#data = seasonal_decompose(ccdata[0:8], period=l, two_sided=False).trend




data = data[l:len(data)]
#data2 = data2[l:len(data2)]
ccdata = ccdata[l:len(ccdata)]

#plt.plot(data, )
#plt.plot(ccdata)
#plt.show()

x = []
y = []
look_back = 5


for i in range(int((len(data)-look_back) * .4)):
    #xvals = np.append(data[i:i+look_back], mean(data[i:i+look_back]))
    #xvals = np.atleast_2d(xvals)
    #print(xvals)
    xvals = data[i:i+look_back]
    #print(xvals)
    xvals.append(np.std(data[i:i+look_back]))

    #print(xvals)
    #print()
    x.append(xvals)
    y.append(ccdata[i+look_back])

i = 0

X = np.array(x)
y = np.array(y)



# Create an instance of the LinearRegression class
reg = LinearRegression()
#poly = PolynomialFeatures(degree=2)
#X_ = poly.fit_transform(X)
#y_ = poly.fit_transform(y)

#close checkmark
 
# Fit the model to the data
reg.fit(X, y)
#reg.fit(X_, y_)

mass = 1
masshistory = []

for i in range(len(data)-look_back):
    #xvals = np.append(data[i:i+look_back], mean(data[i:i+look_back]))

    xvals = [data[i:i+look_back]]
    #print(xvals, type(xvals))
    xvals[0].append(np.std(data[i:i+look_back]))

    #print(xvals, type(xvals))
    if reg.predict(xvals) > 0:
        mass *= (1 + ccdata[i+look_back])
    elif reg.predict(xvals) < 0:
        mass *= 1 / (1 + ccdata[i+look_back])
    masshistory.append(mass)

print(mass)

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
