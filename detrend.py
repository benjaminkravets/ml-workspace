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
import numpy as np

series = read_csv('spydailydiff.csv', header=0, squeeze=True, usecols=[1], index_col=[0])
series2 = seasonal_decompose(series)
series2.plot()
#pyplot.plot(series2)
pyplot.show()

#pyplot.plot(series2, color="green")
#pyplot.plot(series, color="brown")
#pyplot.show()