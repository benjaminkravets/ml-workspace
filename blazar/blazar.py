# univariate convlstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
from tensorflow import keras
from keras import layers
import csv
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

def read():
	raw_seq = []
	with open("test.csv") as CSV_FILE:
		CSV_READER = csv.reader(x.replace('\0','') for x in CSV_FILE)
		headers = [x.strip() for x in next (CSV_READER)]
		for row in CSV_READER:
			raw_seq.append(float(row[6]))
	return raw_seq


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]
#raw_seq = read()

print(raw_seq[0])
print(raw_seq[-1])

# choose a number of time steps
n_steps = 4
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
n_features = 1
n_seq = 2
n_steps = 2
print(str(X.shape[0]) + " " + str(n_seq) + " " + str(n_steps) + " " + str(n_features))
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
# define model
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics="mean_squared_error")
model.summary()
# fit model
model.fit(X, y, epochs=100, verbose=2)
model.save("recent")




def test():
	model = keras.model.load_model("recent")
	# demonstrate prediction
	x_input = array([80, 90, 100, 110])
	x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
	yhat = model.predict(x_input, verbose=0)
	print(yhat)

test()