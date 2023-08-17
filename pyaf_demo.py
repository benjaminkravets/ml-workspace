import numpy as np, pandas as pd
import pyaf.ForecastEngine as autof
from pandas import read_csv
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import logging
import sys, os

logger = logging.getLogger('pyaf.std')
logger.propagate = False

if __name__ == '__main__':
	# generate a daily signal covering one year 2016 in a pandas dataframe
	N = 360
	df_train = pd.DataFrame({"Date": pd.date_range(start="2016-01-25", periods=N, freq='D'),
								  "Signal": (np.arange(N)//40 + np.arange(N) % 21 + np.random.randn(N))})
	
	#print(df_train)
	
	
	df_train = read_csv('datashop/spydaily.csv', header=0)
	df_train['Date'] = pd.to_datetime(df_train['Date'])
	df_train = df_train.rename(columns={'Open':'Signal'})




	#print(df_train)
				  
	# create a forecast engine, the main object handling all the operations
	lEngine = autof.cForecastEngine()

	# get the best time series model for predicting one week
	lEngine.train(iInputDS=df_train[0:3000], iTime='Date', iSignal='Signal', iHorizon=1);

	#lEngine.getModelInfo() # => relative error 7% (MAPE)

	# predict one week
	#df_forecast = lEngine.forecast(iInputDS=df_train[0:7664], iHorizon=1)
	# list the columns of the forecast dataset
	#print(df_forecast.columns)

	# print the real forecasts
	# Future dates : ['2017-01-19T00:00:00.000000000' '2017-01-20T00:00:00.000000000' '2017-01-21T00:00:00.000000000' '2017-01-22T00:00:00.000000000' '2017-01-23T00:00:00.000000000' '2017-01-24T00:00:00.000000000' '2017-01-25T00:00:00.000000000']
	#print(df_forecast['Date'].tail(1).values)
	# signal forecast : [ 9.74934646  10.04419761  12.15136455  12.20369717  14.09607727 15.68086323  16.22296559]
	
	#print(df_forecast['Signal_Forecast'])
	#print(df_forecast['Signal_Forecast'])
	mass = 1
	masshistory = []
	for i in range(len(df_train)):
		if i < 200:
			continue
		df_forecast = lEngine.forecast(iInputDS=df_train[i-100:i], iHorizon=1)
		yhat = df_forecast['Signal_Forecast'].tail(1).values
		#if(i < 250):
		#	print(df_train[0:i])
		#	print(df_train['Signal'][i])


		#if yhat > 0:
		#	mass *= (1 + df_train['Signal'][i])
		#elif yhat < 0:
		#	mass *= (1 / (1 + df_train['Signal'][i]))

		if yhat > (df_train['Signal'][i-1]):
			mass *= (df_train['Signal'][i] / df_train['Signal'][i - 1])
		elif yhat < (df_train['Signal'][i-1]):
			mass *= (df_train['Signal'][i - 1] / df_train['Signal'][i])

		if i % 260 == 0:
			print(mass)
		masshistory.append(mass)

	import matplotlib.pyplot as plt
	plt.plot(masshistory)
	plt.show()