# evaluate an ARIMA model using a walk-forward validation
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import DataFrame


series = read_csv('spydailydiff.csv', header=0, usecols=[1])

# fit model
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit()

if(False):
	# summary of fit model
	print(model_fit.summary())
	# line plot of residuals
	residuals = DataFrame(model_fit.resid)
	residuals.plot()
	pyplot.show()
	# density plot of residuals
	residuals.plot(kind='kde')
	pyplot.show()
	# summary stats of residuals
	print(residuals.describe())


# split into train and test sets
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
mass = 1
masshistory = []
# walk-forward validation
for t in range(len(test)):
	#model = ARIMA(history, order=(5,1,0))
	#model_fit = model.fit()
	#output = model_fit.forecast()
	yhat = float(model_fit.predict(start=t+5,end=t+5))

	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))


	tol = .00000

	if (yhat > 0 + tol):
		mass *= (1 + obs) * (1-.000)
	if (yhat < 0 - tol):
		mass *= (1 / (1 + obs) * (1-.000))
	
	masshistory.append(float(mass))

# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
""" 
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
 """

pyplot.plot(masshistory)
pyplot.show()