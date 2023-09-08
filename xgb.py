# forecast monthly births with xgboost
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from matplotlib import pyplot

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]
model = XGBRegressor(objective='reg:squarederror', max_depth=1, n_estimators=1000)
# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX, trains):
	global model
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	#model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
	if trains:
		model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict(asarray([testX]))
	return yhat[0]

# walk-forward validation for univariate data
masshistory = []
def walk_forward_validation(data, n_test):
	mass = 1
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = xgboost_forecast(history, testX, i == 0)


		#print(testX, testy, yhat)
		if(1):
			if yhat > 0:
				mass *= (1 + testy)
			elif yhat < 0:
				mass *= 1 / (1 + testy)
			print(mass)
		#if(1):


		masshistory.append(mass)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		#print('>expected=%.5f, predicted=%.5f' % (testy, yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions

# load the dataset
series = read_csv('datashop/births.csv', header=0, index_col=0)
series = read_csv('datashop/humiditydailydiff.csv', header=0, usecols=[1])
values = series.values
# transform the time series data into supervised learning
data = series_to_supervised(values, n_in=5)
# evaluate
mae, y, yhat = walk_forward_validation(data, 1000)
print('MAE: %.3f' % mae)
# plot expected vs preducted
pyplot.plot(masshistory)
pyplot.show()