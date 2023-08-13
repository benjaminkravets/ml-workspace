# create and evaluate an updated autoregressive model
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose, convolution_filter
#from statsmodels.tsa.tsatools import detrend
from scipy.signal import detrend
from math import sqrt
import sys
from statistics import mean
import numpy as np

series = read_csv('spydailydiff.csv', header=0, squeeze=True, usecols=[1])
series2 = seasonal_decompose(series, period=20, two_sided=False)

series2.plot()
pyplot.show()

sys.exit()
mass = 1
masshistory = []
for i in range(len(series)):
	if i < 20:
		continue
	if(mean(series2[i-20:i-1]) > 0):
		mass *= (1+series[i])
	elif(mean(series2[i-20:i-1]) < 0):
		mass *= 1 / (1+series[i])
	masshistory.append(mass)

pyplot.plot(masshistory)
pyplot.show()