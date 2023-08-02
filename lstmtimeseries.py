# Multilayer Perceptron to Predict International Airline Passengers (t+1, given t, t-1, t-2)
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, MultiHeadAttention, Flatten, LayerNormalization
import sys
import keras as k
from keras import backend as K
from keras import optimizers, initializers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils.vis_utils import plot_model
import os
import tensorflow as tf
import ntakouris
import keras

scaling = 0
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def custom_loss(y_true, y_pred):
	def loss_fn(y_true_t, y_pred_t):
		if ((y_true_t[0] > 0 and y_pred_t[0] < 0) or (y_true_t[0] < 0 and y_pred_t[0] > 0)):
			print(y_pred_t, y_true_t)
			return K.square(y_true_t - y_pred_t)
		else:
			return float(0)
	loss = tf.map_fn(lambda x: loss_fn(x[0], x[1]), (y_true, y_pred), dtype=tf.float32)
	
	return K.mean(loss)

	

# load the dataset
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
if (scaling):
	scaler = StandardScaler()
	dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.5)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


# reshape dataset
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

#print(trainX.shape,trainY.shape)


#myoptimizer = optimizers.Adam(loss='mean_squared_error', lr=.0001)
try:
	os.remove("models/model.keras")
except:
	print()
# create and fit Multilayer Perceptron model
es = EarlyStopping(monitor='loss', patience=20, mode="auto", restore_best_weights=True)
if(0):
	model = Sequential()
	model.add(Dense(12, input_dim=look_back, activation='relu'))
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(.2))
	model.add(Dense(1))

if(0):
	inputs = keras.Input(shape=(trainX.shape[1],1))
	x = inputs
	x = MultiHeadAttention(num_heads=2, key_dim=2, dropout=.2)(inputs, inputs)
	x = Flatten()(x)
	x = Dense(1)(x)

	model = keras.Model(inputs, x)

	#sys.exit()
if(1):
	inputs = keras.Input(shape=(trainX.shape[1],1))
	x = inputs
	x = ntakouris.ModelTrunk()(x)
	#x = Flatten()(x)
	x = LayerNormalization()(x)
	x = Dense(1)(x)

	model = keras.Model(inputs, x)
	#sys.exit()

#for layer in model.layers:
#    print(layer.output_shape)
#sys.exit()
	

model.compile(loss='mse', optimizer=optimizers.Adam())
model.fit(trainX, trainY, epochs=500, batch_size=2, verbose=1, callbacks=es)
#print(testX.shape, testY.shape)
#sys.exit()
#model.save("models/model.keras")
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)



# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

