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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scalin = 0

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def glower():
	# load the dataset
	dataframe = read_csv('humidity.csv', usecols=[1], engine='python')
	dataset = dataframe.values
	dataset = dataset.astype('float32')
	if(scalin):
		scaler = StandardScaler()
		dataset = scaler.fit_transform(dataset)

	#test only
	test = dataset[0:len(dataset),:]

	# reshape dataset
	look_back = 3

	testX, testY = create_dataset(test, look_back)

	model = keras.models.load_model("models/model.keras")

	mass = 1
	masshistory = []
	valhistory = []


	testY = model.predict(testX)

	if(scalin):

		dataset = scaler.inverse_transform(dataset)
		test = dataset[0:len(dataset),:]
		# reshape dataset
		look_back = 3

		testX, testY = create_dataset(test, look_back)


	
	
	for i, x in enumerate(testX):

		print("Current:",testX[i][look_back-1],"Prox:",testX[i], "Prediction:",testY[i],"Actual:",dataset[i+look_back])
		predicted = testY[i]
		current = testX[i][look_back-1]
		actual = dataset[i+look_back]


			
		z = 1
		if(z == 1):
			if predicted > current:
				mass *= float(actual / current)
			elif predicted < current:
				mass *= float(current / actual)
			

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
			



glower()

