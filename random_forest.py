# forecast monthly births with random forest
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
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

model = RandomForestRegressor(n_estimators=100)

# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX, trains):
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	
	if(trains):
		model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]
masshistory = []
# walk-forward validation for univariate data
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
		if i < 2:
			yhat = random_forest_forecast(history, testX, 1)
		else:
			yhat = random_forest_forecast(history, testX, 0)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		#print('>expected=%.5f, predicted=%.5f' % (testy, yhat))
		#print(test[i], yhat, testy)
		if(0):
			if yhat > 0:
				mass *= (1 + testy)
			elif yhat < 0:
				mass *= 1 / (1 + testy)
		if(1):
			#print(yhat, testy, test[i], test[i][-2])
			if yhat > test[i][-2]:
				mass *= testy / test[i][-2]
			if yhat < test[i][-2]:
				mass *= test[i][-2] / testy

		masshistory.append(mass)


	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions

# load the dataset
#series = read_csv('datashop/births.csv', header=0, index_col=0)
series = read_csv('datashop/humidityhour.csv', header=0, usecols=[1])[0:20000]
values = series.values
# transform the time series data into supervised learning
data = series_to_supervised(values, n_in=5)
# evaluate
mae, y, yhat = walk_forward_validation(data, 10000)
print('MAE: %.3f' % mae)
# plot expected vs predicted
pyplot.plot(masshistory)
pyplot.show()