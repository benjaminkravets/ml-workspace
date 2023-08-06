# create and evaluate an updated autoregressive model
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg

#from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.exponential_smoothing import ets

from sklearn.metrics import mean_squared_error
from math import sqrt
import sys

# load dataset
series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

series = series.values.astype('float64')

#sys.exit()
#pyplot.plot(series)
l = 2
ccseries = series
#for z in seasonal_decompose(series, period=l, two_sided=False).weights:
#	print(z)
series = seasonal_decompose(series, period=l, two_sided=False).trend

print(series[0:5])
print(ccseries[0:5])

series = series[l:len(series)-l]
ccseries = ccseries[l:len(ccseries)-l]
#print(len(series), len(ccseries))


#pyplot.plot(ccseries)
#pyplot.plot(series)
#pyplot.show()



#sys.exit()
# split dataset
X = series

train, test = X[1:len(X)-7], X[len(X)-7:]

train, test = X[0:40], X[0:(len(X))]

print(train.shape, test.shape)
#sys.exit()
# train autoregression
window = 2
model = AutoReg(train, lags=window)
#model = ARIMA(train, order=(window,3,0))
model_fit = model.fit()
coef = model_fit.params

for z in coef:
	print(z)

#sys.exit()

# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()

mass = 1
masshistory = []

for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat, obs))
	#print(test[t-1])
	if(0):
		if(yhat > 0):
			mass *= (obs + 1)
		elif(yhat < 0):
			mass *= (1 / (obs + 1))
	if(0):
		if(yhat > 0):
			mass *= (ccseries + 1)
		elif(yhat < 0):
			mass *= (1 / (ccseries + 1))
	if(0):

		if(yhat > test[t-1]):
			mass *= (obs / test[t-1])
			
		elif(yhat < test[t-1]):
			mass *= (test[t-1] / obs)

	#masshistory.append(mass[0])
	
		
		

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot
if(1):
	pyplot.plot(test)
	pyplot.plot(predictions, color='red')
	pyplot.show()

if(0):
	pyplot.plot(masshistory)
	#print(len(masshistory))
	pyplot.show()
 