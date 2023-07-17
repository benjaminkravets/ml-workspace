# LSTM for international airline passengers problem with regression framing
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys
import joblib
import statistics, csv
from dateutil.parser import parse
from statistics import mean
from sklearn.preprocessing import MinMaxScaler

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def glower():
	targetcsv = "humidityfull.csv"
	dataframe = read_csv(targetcsv, usecols=[1])
	dataset = dataframe.values

	scaler = MinMaxScaler()
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)

	model = keras.models.load_model("models/model.keras")
	look_back = 3
	mass = 1
	masshistory = []
	dataset = scaler.transform(dataset)
	#print(dataset.shape)
	testX, testY = create_dataset(dataset, look_back)
	#print(testX[1], testY[1])
	#print(testX.shape)
	#testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	#print(testX.shape)

	testPredict = model.predict(testX)

	#print(testPredict)

	testPredict = scaler.inverse_transform(testPredict)

	testX = np.reshape(testX, (testX.shape[0], look_back))
	testX = scaler.inverse_transform(testX)
	dataset = scaler.inverse_transform(dataset)
	#print(testX[0], testPredict[0], dataset[look_back])

	#print(testX[0][look_back-1])

	tolerance = .00
	
	for i, x in enumerate(testX):

		#print("Current:",testX[i][look_back-1],"Prox:",testX[i], "Prediction:",testPredict[i],"Actual:",dataset[i+look_back])
		if(1):
			if testPredict[i] > testX[i][look_back-1] + tolerance:
				mass *= dataset[i+look_back] / testX[i][look_back-1]
			elif testPredict[i] < testX[i][look_back-1] - tolerance:
				mass *= testX[i][look_back-1] / dataset[i+look_back]
		if(0):
			if testPredict[i] > 1:
				mass *= dataset[i+look_back]
			if testPredict[i] < 1:
				mass *= 1 / dataset[i+look_back]

		masshistory.append(mass[0])

		if i % 260 == 0:
			print(i, mass)
	#print(masshistory)
	plt.plot(masshistory)
	plt.show()


	sys.exit()
	
	for i, x in enumerate(dataset):
		if(i > len(dataset)-5):
			continue
		prox = dataset[i:(i+5)]

		proxscaled = scaler.transform(prox)

		proxscaled = np.reshape(proxscaled, (1,1,5))
		
		guess = scaler.inverse_transform(model.predict(proxscaled, verbose = 0))

		nextopen = dataset[i+5]

		if(guess > prox[4]):
			mass *= nextopen / prox[4]
		elif(guess < prox[4]):
			mass *= prox[4] / nextopen
		
		
		if i % 260 == 0:

			print( i, mass)
		#np.append(masshistory, mass, axis=0)
		
		
#rise()
glower()

