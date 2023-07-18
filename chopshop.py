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
	# load the dataset
	dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
	dataset = dataframe.values
	dataset = dataset.astype('float32')

	#test only
	test = dataset[0:len(dataset),:]

	# reshape dataset
	look_back = 3

	testX, testY = create_dataset(test, look_back)
	

	model = keras.models.load_model("models/model.keras")

	mass = 1
	masshistory = []
	valhistory = []

	#print(testX.shape, testY.shape)

	testY = model.predict(testX)


	#print(testX.shape, testY.shape)
	#sys.exit()

	
	
	for i, x in enumerate(testX):

		#print("Current:",testX[i][look_back-1],"Prox:",testX[i], "Prediction:",testY[i],"Actual:",dataset[i+look_back])
		predicted = testY[i]
		current = testX[i][look_back-1]
		actual = dataset[i+look_back]
		z = 1
		if(z == 1):
			if predicted > current:
				mass *= float(actual / current)
			elif predicted < current:
				mass *= float(current / actual)
			pass

		if(z == 0):
			if predicted > 0:
				mass *= float((1 + actual))
			elif predicted < 0:
				mass *= float((1 / (1 + actual)))
     
		
		#print(mass)

		masshistory.append(mass)
		valhistory.append(current)
	
	plt.plot(masshistory, color="gold")
	#plt.plot(valhistory, color="green")
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

