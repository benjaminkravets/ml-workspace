# create and evaluate an updated autoregressive model
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose, convolution_filter
from math import sqrt
import sys
import numpy as np
# load dataset
series = read_csv('spydailydiff.csv', header=0, squeeze=True, usecols=[1])
# split dataset
x = series.values
y = series.values

x2 = seasonal_decompose(x, period=2, two_sided=False).trend
x2[np.isnan(x2)] = 0
print(x.shape, x2.shape)

train, test = x[1:int(len(x) * .7)], x[1:len(x)]

# train autoregression
window = 3
model = AutoReg(train, lags=window)
model_fit = model.fit()
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
mass = 1
masshistory = []
for t in range(len(test)):
	length = len(history)
	prox = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * prox[window-d-1]
	obs = test[t]
	tol = .00000

	if yhat > 0 + tol:
		mass *= (1 + obs) * (1-.000)
	if yhat < 0 - tol:
		mass *= (1 / (1 + obs) * (1-.000))

	#if yhat > prox[window-1] + tol:
	#	mass *= (obs / prox[window-1]) * (1-.000)
	#if yhat < prox[window-1] - tol:
	#	mass *= (prox[window-1] / obs) * (1-.000)

	masshistory.append(mass)
	#print(prox, yhat, obs)
	predictions.append(yhat)
	history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot
if(0):
	pyplot.plot(test)
	pyplot.plot(predictions, color='red')
	pyplot.show()
if(1):
	pyplot.plot(masshistory)
	pyplot.show()