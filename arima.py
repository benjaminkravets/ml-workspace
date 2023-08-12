# evaluate an ARIMA model using a walk-forward validation
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import DataFrame
import sys

series = read_csv('humidityhourdiff.csv', header=0, usecols=[1]).values[0:20000]

look_back = 3
# fit model
model = SARIMAX(series[0:int(len(series) * 1)], order=(3,1,1), trend=[0,1])
model_fit = model.fit()


# split into train and test sets
""" 
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

# walk-forward validation
print(type(series), type(history))
sys.exit()
 """
mass = 1
masshistory = []
for t in range(len(series)-10):
	#model = ARIMA(history, order=(5,1,0))
	#model_fit = model.fit()
	#output = model_fit.forecast()
	#print(output)
	output = model_fit.predict(start=t)
	yhat = output[0]
	obs = series[t]

	#print(series[t-3:t])
	#print(obs, output)
	#print()

	tol = .00000

	if yhat > 0 + tol:
		mass *= (1 + obs) * (1-.000)
	if yhat < 0 - tol:
		mass *= (1 / (1 + obs) * (1-.000))

	masshistory.append(float(mass))


	#print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts

pyplot.plot(masshistory)
pyplot.show()